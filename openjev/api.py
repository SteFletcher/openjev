"""Jev-compatible HTTP API: POST /v1/systemone and GET /v1/models.

Request, response and error shapes follow TypeSafe's published OpenAPI 0.2.0,
so their SDKs work against this server by pointing TYPESAFE_BASE_URL at it.
"""
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

from .config import MODEL_ALIASES, MODEL_VERSION, MODELS, Settings
from .engine import Engine, Overloaded, SchemaError

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


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)


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
        yield
        await app.state.engine.close()

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
        # Same request, same noise draws: answers are reproducible.
        seed = int.from_bytes(hashlib.sha256(json.dumps([req.state, questions], sort_keys=True).encode()).digest()[:4], "big")
        engine = request.app.state.engine
        try:
            answers, input_tokens = await engine.decide(questions, req.state, seed)
        except SchemaError as e:
            return validation_error(e.loc, str(e))
        except Overloaded as e:
            return error(529, "overloaded_error", str(e), {"retry-after": "1"})
        except httpx.HTTPError as e:
            return error(503, "api_error", f"inference backend unavailable: {type(e).__name__}", {"retry-after": "2"})
        return {"model": MODEL_VERSION, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}

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
