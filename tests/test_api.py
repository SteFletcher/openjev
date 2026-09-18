"""Offline tests: the real tokenizer, a stubbed vLLM read."""
import math

import msgspec
import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from openjev.api import create_app
from openjev.config import Settings
from openjev.engine import Engine, confidence, to_answer

TOKENIZER = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
EXAMPLE = {  # Jev's quickstart request, verbatim
    "state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
    "model": "jev-latest",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this",
                       "criteria": {"billing": "Payment or subscription issues",
                                    "technical": "Bugs or integration problems",
                                    "sales": "Pricing or account questions"}},
        "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
    },
}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def client(tok, monkeypatch):
    reads = []

    async def fake_read(self, template, slots, sys_text, state_text, seed):
        reads.append(sys_text)
        # first label 70%, the rest share 30%
        out = []
        for s in slots:
            n = len(s["label_ids"])
            probs = [0.7] + [0.3 / (n - 1)] * (n - 1)
            out.append({"probs": probs, "entropy": 0.05})
        return out, 123

    monkeypatch.setattr(Engine, "one_read", fake_read)
    with TestClient(create_app(Settings(), tokenizer=tok)) as c:
        c.reads = reads
        yield c


def test_labels_are_single_tokens(tok):
    eng = Engine(Settings(), tok)
    assert len(eng.choice_labels) == 128
    assert eng.choice_labels[:3] == ["A", "B", "C"]


def test_many_questions_chunk(tok):
    eng = Engine(Settings(canvas=32), tok)
    schema = eng.build_schema({f"k{i}": {"type": "noul", "instructions": "x"} for i in range(30)})
    groups = eng.groups(schema["questions"], schema["format"])
    assert len(groups) > 1
    for g in groups:
        eng.resolve_template(g, schema["format"])  # every chunk fits and resolves


def test_quickstart_decodes_with_typesafe_sdk(client):
    from typesafe_sdk._core.response_types import _RESPONSE_DECODER

    r = client.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 200, r.text
    assert r.headers["x-typesafe-request-id"].startswith("req_")
    body = _RESPONSE_DECODER.decode(r.content)  # the SDK's strict decoder
    assert body.answers["department"].choice == "billing"
    assert set(body.answers["department"].probabilities) == {"billing", "technical", "sales"}
    assert body.answers["frustration"].legend[0] == "Calm, just stating facts"
    assert math.isclose(body.answers["frustration"].score, 0.15 * 1 + 0.15 * 2)
    assert math.isclose(body.answers["is_urgent"].noul, 0.7)
    assert set(r.json()["answers"]["is_urgent"]) == {"type", "noul"}
    assert body.usage.input_tokens == 123 and body.usage.output_tokens == 0
    assert "department" not in client.reads[0]  # question ids stay out of the prompt


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["name"] == "openjev-latest"


def test_validation_shapes(client):
    r = client.post("/v1/systemone", json={"model": "jev-latest", "questions": {"a": {"type": "noul"}}})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "state"]
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": {}})
    assert r.status_code == 422
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest",
                                           "questions": {"a": {"type": "score", "criteria": ["only one"]}}})
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)
    r = client.post("/v1/systemone", json={"state": "x", "model": "gpt-4", "questions": {"a": {"type": "noul"}}})
    assert r.status_code == 404 and r.json()["detail"]["error_type"] == "not_found_error"


def test_auth(tok, monkeypatch):
    with TestClient(create_app(Settings(api_key="sk-test"), tokenizer=tok)) as c:
        r = c.get("/v1/models")
        assert r.status_code == 403 and r.json()["detail"]["error_type"] == "authentication_error"
        assert c.get("/v1/models", headers={"authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers={"authorization": "Bearer sk-test"}).status_code == 200
    with TestClient(create_app(Settings(origin_secret="s3"), tokenizer=tok)) as c:
        assert c.get("/v1/models").status_code == 403
        assert c.get("/v1/models", headers={"x-origin-secret": "s3"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_confidence():
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)
    # matches Jev's documented example (0.84/0.159/0.001 -> ~0.596)
    assert confidence([0.84, 0.159, 0.001]) == pytest.approx(0.596, abs=0.01)


def test_answer_shapes_roundtrip():
    from typesafe_sdk._core.response_types import ScoreAnswer

    q = {"type": "score", "choices": [("0", "a"), ("1", "b")], "legend": ["a", {"k": 1}]}
    a = to_answer(q, [0.25, 0.75])
    decoded = msgspec.json.decode(msgspec.json.encode(a), type=ScoreAnswer)
    assert decoded.score == 0.75 and decoded.legend[1] == {"k": 1}


def test_indexed_format_with_mixed_types(tok):
    """Past ten questions answers are "q1yes q2A q3 4 ..."; every label type must keep one slot."""
    eng = Engine(Settings(), tok)
    qs = {}
    for i in range(12):
        if i % 3 == 0:
            qs[f"n{i}"] = {"type": "noul"}
        elif i % 3 == 1:
            qs[f"s{i}"] = {"type": "score", "criteria": [f"level {k}" for k in range(10)]}
        else:
            qs[f"c{i}"] = {"type": "choice", "criteria": {f"opt{k}": None for k in range(40)}}
    schema = eng.build_schema(qs)
    assert schema["format"] == "indexed"
    for g in eng.groups(schema["questions"], schema["format"]):
        eng.resolve_template(g, schema["format"])
