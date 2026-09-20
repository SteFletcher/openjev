"""Structured reads on Apple silicon: DiffusionGemma in-process through MLX.

Selected with OPENJEV_BACKEND=mlx. The read (prefill, one decoder pass, no
self-conditioning, temperature-1 log-softmax per slot) matches what vLLM's
read-only step reports. mlx, mlx_vlm and PIL are imported lazily so the vLLM
path never needs them.

Images go through the processor's chat template and prefill as pixel values;
the decoder pass is the same either way. Not supported yet: think, more than
one denoise step, text generation. Requesting one of these gets a 400 (or 501),
not a wrong answer.
"""
import asyncio
import base64
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from .engine import TOPK, Engine, SchemaError, slot_distribution

# Re-reads and samples repeat a prompt exactly, so its prefill is cached, evicted by a
# token budget (not entry count) so a few long prompts can't pin memory.
PROMPT_CACHE_TOKENS = 16384


class ImagePrompt:
    """An image read, as everything the runtime thread needs to build it: the
    system and state text, and each image's data URL. The expansion into image
    tokens is the processor's, so it cannot be done here, off that thread."""

    __slots__ = ("sys_text", "state_text", "images", "key")

    def __init__(self, sys_text, state_text, images):
        self.sys_text = sys_text
        self.state_text = state_text
        self.images = images
        # the digests keep a cached prefill off a different image, and off the same text with none
        digests = tuple(hashlib.sha256(u.encode()).digest() for u in images)
        self.key = (sys_text, state_text, digests)

    def pil(self):
        from io import BytesIO

        from PIL import Image

        return [Image.open(BytesIO(base64.b64decode(u.partition(",")[2]))).convert("RGB")
                for u in self.images]


class MlxRuntime:
    """The model and the one thread that touches it. MLX work is kept off the
    event loop and on a single thread, from loading on."""

    def __init__(self, model_path):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openjev-mlx")
        self.prefills = OrderedDict()
        self.pool.submit(self._load, model_path).result()

    def _load(self, model_path):
        import mlx.core as mx
        from mlx_vlm import load

        self.mx = mx
        self.model, self.processor = load(model_path, trust_remote_code=False)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _inputs(self, prompt):
        """(cache key, prefill kwargs, prompt length in tokens) for a read."""
        if not isinstance(prompt, ImagePrompt):
            return tuple(prompt), {"input_ids": self.mx.array([prompt])}, len(prompt)
        from mlx_vlm.utils import prepare_inputs

        images = prompt.pil()
        # this model's message format is LIST_WITH_IMAGE_TYPE_TEXT, so the placeholders
        # the processor later expands must come from structured content, images first
        text = self.processor.apply_chat_template(
            [{"role": "system", "content": prompt.sys_text},
             {"role": "user", "content": [{"type": "image"}] * len(images)
                                         + [{"type": "text", "text": prompt.state_text}]}],
            add_generation_prompt=True, tokenize=False)
        inputs = prepare_inputs(self.processor, images=images, prompts=text)
        ids = inputs["input_ids"]
        kwargs = {"input_ids": ids, "pixel_values": inputs.get("pixel_values"),
                  "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                  "attention_mask": inputs.get("attention_mask")}
        return prompt.key, kwargs, int(ids.shape[-1])

    def _prefill(self, prompt, max_tokens):
        key, kwargs, n = self._inputs(prompt)
        # an image prompt's length is only known here, after the processor expanded it
        if n > max_tokens:
            raise SchemaError(f"the request is {n} tokens; the limit is {max_tokens}")
        hit = self.prefills.get(key)
        if hit is None:
            cache = self.model.diffusion_prefill_cache(**kwargs)
            self.prefills[key] = (cache, n)
            while len(self.prefills) > 1 and sum(t for _, t in self.prefills.values()) > PROMPT_CACHE_TOKENS:
                self.prefills.popitem(last=False)
        else:
            cache = hit[0]
            self.prefills.move_to_end(key)
        return cache, n

    def read(self, prompt, canvas, slots, max_tokens):
        """(logprobs at each slot, prompt tokens). The logprobs are {token id:
        logprob} for the top-k tokens and every label. The token count is the
        expanded prompt, image tokens included. Runs on the runtime's thread."""
        mx = self.mx
        cache, n = self._prefill(prompt, max_tokens)
        ids = mx.array([canvas])
        masks = self.model.diffusion_decoder_masks(ids, cache, None)
        logits = self.model.diffusion_decoder_logits(ids, cache=cache, self_conditioning=None,
                                                     decoder_attention_mask=masks)
        out = []
        for s in slots:
            row = logits[0, s["pos"]].astype(mx.float32)
            lp = row - mx.logsumexp(row)
            keep = sorted(set(mx.argpartition(-lp, TOPK)[:TOPK].tolist()) | set(s["label_ids"]))
            out.append(dict(zip(keep, lp[mx.array(keep)].tolist())))
        return out, n


class MlxEngine(Engine):
    def __init__(self, settings, tokenizer):
        super().__init__(settings, tokenizer)
        self.runtime = MlxRuntime(settings.mlx_model)

    async def close(self):
        await super().close()
        self.runtime.close()

    async def decide(self, questions, state, seed, images=None, options=None):
        opts = options or {}
        for field, asked in (("think", opts.get("think")), ("steps", (opts.get("steps") or 1) > 1)):
            if asked:
                raise SchemaError(f"{field} is not supported on the MLX backend yet", ("body", field))
        return await super().decide(questions, state, seed, images, options)

    async def one_read(self, template, slots, sys_text, content, seed, steps=1, prefix=None):
        if isinstance(content, list):
            # images and think/sequential are mutually exclusive, so prefix is None here
            *parts, state = content
            prompt = ImagePrompt(sys_text, state["text"], [p["image_url"]["url"] for p in parts])
        else:
            prompt = prefix if prefix is not None else self.chat_prompt_ids(sys_text, content)
            if len(prompt) > self.s.mlx_max_prompt:
                raise SchemaError(f"the request is {len(prompt)} tokens; the limit is {self.s.mlx_max_prompt}")
        canvas = self.build_canvas(template, slots, seed)
        async with self.slots:
            tops, billed = await asyncio.get_running_loop().run_in_executor(
                self.runtime.pool, self.runtime.read, prompt, canvas, slots, self.s.mlx_max_prompt)
        return [slot_distribution(top, s["label_ids"]) for top, s in zip(tops, slots)], billed
