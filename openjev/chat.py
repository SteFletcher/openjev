"""POST /v1/chat/completions: ordinary text generation from the same DiffusionGemma.

OpenAI-compatible, so tools that talk to a chat model (jev-ultrafast's text
model, the openai SDKs) can point their base URL here. vLLM refuses a few
things for diffusion models (structured outputs, temperature other than 1,
seeds, logit bias), so requests are normalized first: those fields are
dropped, and JSON mode becomes an instruction plus extraction of the first
JSON object from the reply.
"""
import asyncio
import json
import re

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .config import GEN_MODEL

# everything else -- temperature, seed, min_p, logit_bias, penalties, reasoning
# switches, response_format, n, ... -- is dropped, per the module docstring above.
PASSTHROUGH = {"messages", "max_tokens", "stop", "top_p", "top_k", "stream", "stream_options",
               "tools", "tool_choice", "logprobs", "top_logprobs", "chat_template_kwargs"}
DEFAULT_MAX_TOKENS = 1024
JSON_INSTRUCTION = "Reply with exactly one JSON object and nothing else: no prose, no code fences."
MODEL_NAMES = {GEN_MODEL, "diffusiongemma"}


def oai_error(status, message, type_="invalid_request_error", code=None, headers=None):
    return JSONResponse({"error": {"message": message, "type": type_, "code": code}}, status_code=status, headers=headers)


class Generator:
    def __init__(self, settings):
        self.s = settings
        self.client = httpx.AsyncClient(base_url=settings.upstream.rstrip("/"),
                                        timeout=httpx.Timeout(300.0, connect=5.0))
        self.slots = asyncio.Semaphore(settings.gen_max_inflight)
        self.running = 0

    async def close(self):
        await self.client.aclose()

    def normalize(self, body):
        """The vLLM request for an OpenAI-style one, and whether the caller
        asked for JSON."""
        out = {k: v for k, v in body.items() if k in PASSTHROUGH}
        out["model"] = self.s.upstream_model
        if "max_tokens" not in out and isinstance(body.get("max_completion_tokens"), int):
            out["max_tokens"] = body["max_completion_tokens"]
        out["max_tokens"] = max(1, min(int(out.get("max_tokens") or DEFAULT_MAX_TOKENS), self.s.gen_max_tokens))
        out["chat_template_kwargs"] = {"enable_thinking": False, **(out.get("chat_template_kwargs") or {})}
        if out.get("stream"):
            out["stream_options"] = {**(out.get("stream_options") or {}), "include_usage": True}
        fmt = body.get("response_format") or {}
        json_mode = fmt.get("type") in ("json_object", "json_schema")
        if json_mode:
            note = JSON_INSTRUCTION
            schema = (fmt.get("json_schema") or {}).get("schema")
            if schema:
                note += " It must match this JSON schema: " + json.dumps(schema, ensure_ascii=False)
            msgs = [dict(m) for m in out["messages"]]
            if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
                msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + note
            else:
                msgs.insert(0, {"role": "system", "content": note})
            out["messages"] = msgs
        return out, json_mode


def extract_json(text):
    """The first JSON object or array in a reply, as text; the reply unchanged
    when there is none."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    dec = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(stripped[i:])
                return json.dumps(obj, ensure_ascii=False)
            except ValueError:
                continue
    return text


def upstream_message(r):
    try:
        d = r.json()
        return (d.get("error") or {}).get("message") or d.get("message") or r.text
    except ValueError:
        return r.text


def add_chat_routes(app):
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        gen = request.app.state.generator
        try:
            body = await request.json()
        except ValueError:
            return oai_error(400, "The request body is not valid JSON.")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return oai_error(400, "messages must be a non-empty array.")
        if body.get("model") not in MODEL_NAMES:
            return oai_error(404, f"Model {body.get('model')!r} not found. Available: {GEN_MODEL}.", code="model_not_found")
        if gen.running >= gen.s.gen_max_inflight + gen.s.gen_max_queue:
            return oai_error(529, "Text generation is at capacity. Retry shortly.", "overloaded_error", headers={"retry-after": "2"})
        upstream, json_mode = gen.normalize(body)

        if upstream.get("stream"):
            gen.running += 1
            await gen.slots.acquire()

            def release():
                gen.slots.release()
                gen.running -= 1

            try:
                r = await gen.client.send(gen.client.build_request("POST", "/v1/chat/completions", json=upstream), stream=True)
            except httpx.HTTPError as e:
                release()
                return oai_error(503, f"inference backend unavailable: {type(e).__name__}", "api_error", headers={"retry-after": "2"})
            if r.status_code >= 400:
                await r.aread()
                await r.aclose()
                release()
                return oai_error(400 if r.status_code < 500 else 503, upstream_message(r), "invalid_request_error" if r.status_code < 500 else "api_error")

            async def lines():
                async for line in r.aiter_lines():
                    yield line.replace(f'"model":"{gen.s.upstream_model}"', f'"model":"{GEN_MODEL}"') + "\n"

            async def done():
                await r.aclose()
                release()

            return StreamingResponse(lines(), media_type="text/event-stream", background=BackgroundTask(done))

        gen.running += 1
        try:
            async with gen.slots:
                r = await gen.client.post("/v1/chat/completions", json=upstream)
        except httpx.HTTPError as e:
            return oai_error(503, f"inference backend unavailable: {type(e).__name__}", "api_error", headers={"retry-after": "2"})
        finally:
            gen.running -= 1
        if r.status_code >= 400:
            return oai_error(400 if r.status_code < 500 else 503, upstream_message(r), "invalid_request_error" if r.status_code < 500 else "api_error")
        d = r.json()
        d["model"] = GEN_MODEL
        if json_mode:
            for c in d.get("choices", []):
                msg = c.get("message") or {}
                if isinstance(msg.get("content"), str):
                    msg["content"] = extract_json(msg["content"])
        return d
