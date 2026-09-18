"""Structured reads on a DiffusionGemma vLLM server.

Adapted from vLLM's examples/features/diffusion_reads/structured_server.py
(vllm-project/vllm PR #57250, Apache-2.0). A discrete diffusion model denoises
the whole answer canvas per forward pass. Seed the canvas with the answer
template, leave only the label slots as noise, run one read-only denoise step,
and each slot's logprobs are a distribution over that question's labels.
"""
import asyncio
import json
import math
import random

import httpx

VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = 20
MAX_LABEL_IDS = 128  # vLLM's logprob_token_ids cap per request
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"  # the empty thought block the chat template leaves to the model

# Answer template shapes: (join between questions, what precedes the label,
# reply instruction). "indexed" costs fewer rows a question; past ten
# questions the saved rows keep a schema in one read.
FORMATS = {
    "lines": ("\n", "{id}: ", 'Reply with one line per question, in this order, formatted as "id: label".'),
    "indexed": (" ", "{id}", "Reply on one line with each question's id immediately followed by its label, separated by single spaces."),
}


class SchemaError(ValueError):
    """A request the model cannot answer as asked; surfaced as a 422."""

    def __init__(self, msg, loc=("body",)):
        super().__init__(msg)
        self.loc = list(loc)


class Overloaded(RuntimeError):
    pass


def text_of(value):
    """Jev descriptions and instructions may be strings, objects or arrays."""
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class Engine:
    def __init__(self, settings, tokenizer):
        self.s = settings
        self.tok = tokenizer
        self.scaffold = self.enc(SCAFFOLD_TEXT)
        self.client = httpx.AsyncClient(base_url=settings.upstream.rstrip("/"),
                                        timeout=httpx.Timeout(120.0, connect=5.0),
                                        limits=httpx.Limits(max_connections=settings.max_inflight * 2))
        self.slots = asyncio.Semaphore(settings.max_inflight)
        self.waiting = 0
        self.choice_labels = self._single_token_labels()
        self._templates = {}

    async def close(self):
        await self.client.aclose()

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Labels and templates
    # ------------------------------------------------------------------

    def _single_token_labels(self):
        """Choice labels that stay one token after "q1: ", in a stable order."""
        base = self.enc("q1: A")
        cands = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        cands += [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        out, seen = [], set()
        for c in cands:
            e = self.enc("q1: " + c)
            if len(e) == len(base) and e[:-1] == base[:-1] and e[-1] not in seen:
                seen.add(e[-1])
                out.append(c)
            if len(out) == MAX_LABEL_IDS:
                break
        return out

    def build_schema(self, questions):
        """Jev questions -> internal question list. Ids are never shown to the
        model; it sees q1, q2, ... and the answers map back by position."""
        qs = []
        for i, (qid, q) in enumerate(questions.items()):
            loc = ("body", "questions", qid, "criteria")
            kind = q["type"]
            if kind == "noul":
                crit = q.get("criteria") or {}
                choices = [("yes", text_of(crit.get("true"))), ("no", text_of(crit.get("false")))]
                labels = ["yes", "no"]
            elif kind == "choice":
                crit = q["criteria"]
                if len(crit) < 2:
                    raise SchemaError("a choice needs at least two options", loc)
                if len(crit) > len(self.choice_labels):
                    raise SchemaError(f"OpenJev supports at most {len(self.choice_labels)} options per choice", loc)
                choices = [(name, text_of(desc)) for name, desc in crit.items()]
                labels = self.choice_labels[: len(choices)]
            elif kind == "score":
                crit = q["criteria"]
                if not 2 <= len(crit) <= 10:
                    raise SchemaError("a score takes 2 to 10 levels", loc)
                choices = [(str(i), text_of(c)) for i, c in enumerate(crit)]
                labels = [str(i) for i in range(len(crit))]
            else:  # the request model rejects this first
                raise SchemaError(f"unknown question type {kind!r}", ("body", "questions", qid, "type"))
            qs.append({"key": qid, "id": f"q{i + 1}", "type": kind, "instructions": text_of(q.get("instructions")),
                       "choices": choices, "labels": labels,
                       # score legends echo the criteria exactly as sent
                       "legend": list(q["criteria"]) if kind == "score" else None})
        return {"questions": qs, "format": "lines" if len(qs) <= 10 else "indexed"}

    def system_text(self, qs, fmt, chunked=False):
        s = ("Answer a fixed set of questions about the state the user provides. "
             "Each question lists its allowed answers; reply with exactly one label per question.\n")
        for q in qs:
            s += f"\nQuestion {q['id']}: {q['instructions'] or 'Answer about the state.'}\n"
            for (name, desc), label in zip(q["choices"], q["labels"]):
                if q["type"] == "noul":
                    s += f"  {label}: {desc}\n" if desc else f"  {label}\n"
                elif q["type"] == "score":
                    s += f"  {label}: {desc}\n"
                else:
                    s += f"  {label}: {name} ({desc})\n" if desc else f"  {label}: {name}\n"
        s += "\n" + FORMATS[fmt][2]
        if chunked:
            s += " A reply may cover only some of the questions; answer every line that is present."
        return s

    def answer_text(self, qs, labels, fmt):
        join, lead, _ = FORMATS[fmt]
        return join.join(lead.format(id=q["id"]) + q["labels"][l] for q, l in zip(qs, labels))

    def resolve_template(self, qs, fmt):
        """Tokenize the answer template and find each question's slot. Every
        label must change exactly one token, at the same position for all of a
        question's labels."""
        key = json.dumps([fmt] + [(q["id"], q["labels"]) for q in qs])
        hit = self._templates.get(key)
        if hit:
            return hit
        head = self.scaffold
        base_labels = [0] * len(qs)
        base = head + self.enc(self.answer_text(qs, base_labels, fmt))
        if len(base) + 1 > self.s.canvas:
            raise SchemaError(f"answer template is {len(base)} tokens; the canvas holds {self.s.canvas - 1}")
        slots = []
        for qi, q in enumerate(qs):
            pos = None
            ids = [0] * len(q["labels"])
            for li in range(1, len(q["labels"])):
                labels = list(base_labels)
                labels[qi] = li
                e = head + self.enc(self.answer_text(qs, labels, fmt))
                diffs = [i for i in range(min(len(e), len(base))) if e[i] != base[i]]
                if len(e) != len(base) or len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                    raise SchemaError(f"question {q['key']!r}: labels do not share one template slot")
                pos = diffs[0]
                ids[li] = e[pos]
            ids[0] = base[pos]
            slots.append({"pos": pos, "label_ids": ids})
        if len(self._templates) > 4096:
            self._templates.clear()
        self._templates[key] = (base, slots)
        return base, slots

    def groups(self, qs, fmt):
        """Split questions, in order, into the fewest groups whose answer
        templates fit the canvas."""
        out, group = [], []
        for q in qs:
            trial = group + [q]
            rows = len(self.scaffold) + len(self.enc(self.answer_text(trial, [0] * len(trial), fmt))) + 1
            if rows > self.s.canvas and group:
                out.append(group)
                group = [q]
            else:
                group = trial
        out.append(group)
        return out

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def canvas_width(self, template):
        need = len(template) + 1
        step = self.s.canvas_step
        return min(self.s.canvas, -(-need // step) * step)

    def build_canvas(self, template, slots, seed):
        rng = random.Random(seed)
        canvas = list(template) + [TURN_CLOSE]
        canvas += [PAD] * (self.canvas_width(template) - len(canvas))
        for s in slots:
            canvas[s["pos"]] = rng.randrange(VOCAB)
        return canvas

    async def one_read(self, template, slots, sys_text, state_text, seed):
        body = {
            "model": self.s.upstream_model,
            "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": state_text}],
            "max_tokens": len(template) + 1,
            "logprobs": True,
            "top_logprobs": TOPK,
            # exact logprobs for every label at every position; long option
            # lists rarely rank inside the top-k
            "logprob_token_ids": sorted({i for s in slots for i in s["label_ids"]})[:MAX_LABEL_IDS],
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {"diffusion_seed_canvas": self.build_canvas(template, slots, seed),
                           "diffusion_canvas_length": self.canvas_width(template),
                           "diffusion_max_steps": 1, "diffusion_read_only": True},
        }
        async with self.slots:
            r = await self.client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()
        content = d["choices"][0]["logprobs"]["content"]
        out = []
        for s in slots:
            top = {int(t["token"].split(":")[1]): t["logprob"] for t in content[s["pos"]]["top_logprobs"]}
            out.append(slot_distribution(top, s["label_ids"]))
        return out, d.get("usage", {}).get("prompt_tokens", 0)

    async def read_group(self, qs, fmt, sys_text, state_text, seed):
        template, slots = self.resolve_template(qs, fmt)
        first, prompt_tokens = await self.one_read(template, slots, sys_text, state_text, seed)
        reads = [first]
        if self.s.auto_max > 1 and max(r["entropy"] for r in first) > self.s.auto_threshold:
            more = await asyncio.gather(*[self.one_read(template, slots, sys_text, state_text, seed + k * 7919)
                                          for k in range(1, self.s.auto_max)])
            reads += [m[0] for m in more]
        return reads, prompt_tokens

    async def decide(self, questions, state, seed):
        """Answer a Jev request. Returns (answers keyed by question id, billed input tokens)."""
        if self.waiting >= self.s.max_queue:
            raise Overloaded("OpenJev is at capacity. Retry shortly.")
        self.waiting += 1
        try:
            schema = self.build_schema(questions)
            fmt = schema["format"]
            state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            groups = self.groups(schema["questions"], fmt)
            chunked = len(groups) > 1
            results = await asyncio.gather(*[
                self.read_group(g, fmt, self.system_text(g, fmt, chunked), state_text, seed + 104729 * k)
                for k, g in enumerate(groups)])
        finally:
            self.waiting -= 1
        answers, billed = {}, 0
        for g, (reads, prompt_tokens) in zip(groups, results):
            billed += prompt_tokens
            for qi, q in enumerate(g):
                mean = [sum(r[qi]["probs"][l] for r in reads) / len(reads) for l in range(len(q["labels"]))]
                answers[q["key"]] = to_answer(q, mean)
        return answers, billed


def slot_distribution(top, label_ids):
    """Label probabilities at one slot. Read-only logprobs are at temperature
    1, so the label softmax uses them directly. The entropy is over the
    returned top-k set and drives the re-read policy."""
    floor = min(top.values()) - 5.0
    lp = [top.get(i, floor) for i in label_ids]
    mx = max(lp)
    ex = [math.exp(x - mx) for x in lp]
    z = sum(ex)
    top_p = [math.exp(v) for v in top.values()]
    return {"probs": [e / z for e in ex], "entropy": -sum(p * math.log(p) for p in top_p if p > 0)}


def confidence(p):
    """How peaked a distribution is: 1 - H(p)/ln(K). 1 is certain, 0 uniform."""
    k = len(p)
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def to_answer(q, p):
    """Jev's answer shapes."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[0]}
    if q["type"] == "choice":
        top = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": q["choices"][top][0],
                "probabilities": {c[0]: v for c, v in zip(q["choices"], p)}, "confidence": confidence(p)}
    return {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
            "legend": {str(i): q["legend"][i] for i in range(len(p))},
            "probabilities": {str(i): v for i, v in enumerate(p)}, "confidence": confidence(p)}
