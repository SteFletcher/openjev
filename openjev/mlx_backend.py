"""Structured reads on Apple silicon: DiffusionGemma in-process through MLX.

Selected with OPENJEV_BACKEND=mlx. Everything up to the read is the Engine's:
schema, template, slots and the seeded canvas. A read is then the prompt's
prefill, one decoder pass over the canvas with no self-conditioning, and a
temperature-1 log-softmax at each slot, which is what vLLM's read-only step
reports. mlx and mlx_vlm are imported only when a runtime is built, so the
vLLM path never needs them.

Not on this backend yet: images, think, more than one denoise step, and text
generation. A request for one gets a 400 (or a 501) instead of a wrong answer.
"""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from .engine import TOPK, Engine, SchemaError, slot_distribution

# Re-reads and samples repeat a prompt exactly, so its prefill is kept. The budget is in
# tokens, not entries, so a few long prompts cannot pin memory.
PROMPT_CACHE_TOKENS = 16384


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
        self.model, _ = load(model_path, trust_remote_code=False)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _prefill(self, prompt):
        key = tuple(prompt)
        cache = self.prefills.get(key)
        if cache is None:
            cache = self.model.diffusion_prefill_cache(self.mx.array([prompt]))
            self.prefills[key] = cache
            while len(self.prefills) > 1 and sum(map(len, self.prefills)) > PROMPT_CACHE_TOKENS:
                self.prefills.popitem(last=False)
        else:
            self.prefills.move_to_end(key)
        return cache

    def read(self, prompt, canvas, slots):
        """Logprobs at each slot, {token id: logprob}, for the top-k tokens and
        every label. Runs on the runtime's thread."""
        mx = self.mx
        cache = self._prefill(prompt)
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
        return out


class MlxEngine(Engine):
    def __init__(self, settings, tokenizer):
        super().__init__(settings, tokenizer)
        self.runtime = MlxRuntime(settings.mlx_model)

    async def close(self):
        await super().close()
        self.runtime.close()

    async def decide(self, questions, state, seed, images=None, options=None):
        opts = options or {}
        for field, asked in (("images", images), ("think", opts.get("think")), ("steps", (opts.get("steps") or 1) > 1)):
            if asked:
                raise SchemaError(f"{field} is not supported on the MLX backend yet", ("body", field))
        return await super().decide(questions, state, seed, images, options)

    async def one_read(self, template, slots, sys_text, content, seed, steps=1, prefix=None):
        prompt = prefix if prefix is not None else self.chat_prompt_ids(sys_text, content)
        if len(prompt) > self.s.mlx_max_prompt:
            raise SchemaError(f"the request is {len(prompt)} tokens; the limit is {self.s.mlx_max_prompt}")
        canvas = self.build_canvas(template, slots, seed)
        async with self.slots:
            tops = await asyncio.get_running_loop().run_in_executor(
                self.runtime.pool, self.runtime.read, prompt, canvas, slots)
        return [slot_distribution(top, s["label_ids"]) for top, s in zip(tops, slots)], len(prompt)
