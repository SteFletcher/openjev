"""Jev-compatible HTTP API: POST /v1/systemone and GET /v1/models.

Request, response and error shapes follow TypeSafe's published OpenAPI 0.2.0,
so their SDKs work against this server by pointing TYPESAFE_BASE_URL at it.
Optional request fields beyond that contract (images, steps, samples, think,
sequential) are ignored by the SDKs and change nothing when left out.
POST /v1/chat/completions (openjev.chat) serves ordinary text generation.
"""
import base64
import binascii
import hashlib
import hmac
import json
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .chat import Generator, add_chat_routes
from .config import MODEL_ALIASES, MODEL_VERSION, MODELS, Settings
from .engine import Engine, Overloaded, SchemaError, Upstream

JSONContent = Union[str, dict[str, Any], list[Any]]
Described = Union[str, dict[str, Any], list[Any], None]


class NoulCriteria(BaseModel):
    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Described = None
    criteria: list[JSONContent]


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class ImageObject(BaseModel):
    content_type: str
    base64: str


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    # OpenJev extensions; each is optional and off by default.
    images: list[Union[str, ImageObject]] | None = None
    steps: int | None = Field(default=None, ge=1, le=8)
    samples: int | None = Field(default=None, ge=1, le=32)
    think: int | None = Field(default=None, ge=0, le=4096)
    sequential: bool | None = None


IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def image_parts(images, settings):
    """Validate the request's images and turn them into OpenAI-style parts."""
    if len(images) > settings.max_images:
        raise SchemaError(f"at most {settings.max_images} images per request", ("body", "images"))
    parts = []
    for i, im in enumerate(images):
        loc = ("body", "images", i)
        if isinstance(im, str):
            head, sep, data = im.partition(",")
            ctype = head.removeprefix("data:").removesuffix(";base64")
            if not (sep and head.startswith("data:") and head.endswith(";base64")):
                raise SchemaError("an image is a data:image/...;base64, URL or {content_type, base64}", loc)
        else:
            ctype, data = im.content_type, im.base64
        if ctype not in IMAGE_TYPES:
            raise SchemaError(f"image type {ctype!r} is not supported; use JPEG, PNG, WebP or GIF", loc)
        try:
            size = len(base64.b64decode(data, validate=True))
        except (binascii.Error, ValueError):
            raise SchemaError("image data is not valid base64", loc) from None
        if size > settings.max_image_bytes:
            raise SchemaError(f"image is {size} bytes; the limit is {settings.max_image_bytes}", loc)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{ctype};base64,{data}"}})
    return parts


def error(status, error_type, message, headers=None):
    return JSONResponse({"detail": {"error_type": error_type, "message": message}}, status_code=status, headers=headers)


def validation_error(loc, msg, value=None):
    return JSONResponse({"detail": [{"type": "value_error", "loc": loc, "msg": msg, "input": value}]}, status_code=422)


def create_app(settings=None, tokenizer=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        tok = tokenizer
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(settings.tokenizer)
        app.state.engine = Engine(settings, tok)
        app.state.generator = Generator(settings)
        yield
        await app.state.engine.close()
        await app.state.generator.close()

    app = FastAPI(title="OpenJev", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next):
        rid = "req_" + secrets.token_hex(16)
        request.state.request_id = rid
        if request.url.path.startswith("/v1/"):
            denied = check_auth(settings, request)
            if denied is not None:
                denied.headers["x-typesafe-request-id"] = rid
                denied.headers["x-request-id"] = rid
                return denied
        response = await call_next(request)
        response.headers["x-typesafe-request-id"] = rid
        response.headers["x-request-id"] = rid
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"models": MODELS}

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest, request: Request):
        if req.model not in MODEL_ALIASES:
            return error(404, "not_found_error", f"Model {req.model!r} not found. Available: openjev-latest.")
        questions = {k: q.model_dump() for k, q in req.questions.items()}
        options = {"steps": req.steps, "samples": req.samples, "think": req.think, "sequential": req.sequential}
        engine = request.app.state.engine
        try:
            images = image_parts(req.images, settings) if req.images else None
            # Same request, same noise draws: answers are reproducible.
            key = [req.state, questions] + ([[p["image_url"]["url"] for p in images]] if images else [])
            seed = int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()[:4], "big")
            answers, input_tokens, thought_tokens = await engine.decide(questions, req.state, seed, images, options)
        except SchemaError as e:
            return validation_error(e.loc, str(e))
        except Upstream as e:
            return validation_error(["body"], f"the model rejected this request: {e}")
        except Overloaded as e:
            return error(529, "overloaded_error", str(e), {"retry-after": "1"})
        except httpx.HTTPError as e:
            return error(503, "api_error", f"inference backend unavailable: {type(e).__name__}", {"retry-after": "2"})
        # output_tokens stays 0 as in Jev's contract unless a thought was generated
        return {"model": MODEL_VERSION, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": thought_tokens}}

    add_chat_routes(app)

    return app


def check_auth(settings, request):
    if settings.origin_secret:
        got = request.headers.get("x-origin-secret", "")
        if not hmac.compare_digest(got, settings.origin_secret):
            return error(403, "permission_error", "Direct access to this origin is not allowed.")
    if settings.api_key:
        auth = request.headers.get("authorization", "")
        if not auth:
            return error(403, "authentication_error", "Must supply an API key! Check your request and try again.")
        token = auth.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, settings.api_key):
            return error(401, "authentication_error", "Cannot authenticate with the server. Please check your API key and try again.")
    return None
