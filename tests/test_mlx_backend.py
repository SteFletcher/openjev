"""The MLX backend's wiring, with a stub in place of the model so it runs on any
machine. tests/test_mlx_model.py runs the same path against real weights."""
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from openjev import mlx_backend
from openjev.api import create_app
from openjev.config import Settings
from openjev.engine import PAD, TURN_CLOSE, Engine

from test_api import EXAMPLE, PNG, TOKENIZER


class StubRuntime:
    """Every slot reads 0.99 on its first label and splits the rest: sure enough
    that the engine does not re-read."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.reads = []

    def read(self, prompt, canvas, slots):
        self.reads.append({"prompt": prompt, "canvas": canvas, "slots": slots})
        out = []
        for s in slots:
            ids = s["label_ids"]
            out.append({i: math.log(0.99 if i == ids[0] else 0.01 / (len(ids) - 1)) for i in ids})
        return out

    def close(self):
        self.pool.shutdown()


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def client(tok, monkeypatch):
    monkeypatch.setattr(mlx_backend, "MlxRuntime", StubRuntime)
    with TestClient(create_app(Settings(backend="mlx", mlx_model="/models/dg"), tokenizer=tok)) as c:
        yield c


def test_vllm_stays_the_default(tok):
    assert Settings().backend == "vllm"
    with TestClient(create_app(Settings(), tokenizer=tok)) as c:
        assert type(c.app.state.engine) is Engine
        assert "diffusiongemma-26b" in [m["name"] for m in c.get("/v1/models").json()["models"]]


def test_a_read_goes_to_the_runtime(client):
    r = client.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 200, r.text
    engine = client.app.state.engine
    assert engine.runtime.model_path == "/models/dg"
    (read,) = engine.runtime.reads  # confident answers: one read, no re-reads
    a = r.json()["answers"]
    assert a["department"]["choice"] == "billing" and math.isclose(a["department"]["probabilities"]["billing"], 0.99)
    assert math.isclose(a["is_urgent"]["noul"], 0.99)
    assert math.isclose(a["frustration"]["score"], 0.005 * 1 + 0.005 * 2)
    # the prompt is the chat prompt, and it is what gets billed
    assert read["prompt"] == engine.chat_prompt_ids(client_sys_text(engine), EXAMPLE["state"])
    assert r.json()["usage"] == {"input_tokens": len(read["prompt"]), "output_tokens": 0}
    # the canvas is the template with noise only in the slots, closed and padded to a step
    template, slots = read_template(engine)
    canvas, positions = read["canvas"], {s["pos"] for s in slots}
    assert len(canvas) % engine.s.canvas_step == 0 and canvas[len(template)] == TURN_CLOSE
    assert all(canvas[i] == t for i, t in enumerate(template) if i not in positions)
    assert set(canvas[len(template) + 1:]) <= {PAD}


def client_sys_text(engine):
    schema = engine.build_schema(EXAMPLE["questions"])
    return engine.system_text(schema["questions"], schema["format"])


def read_template(engine):
    schema = engine.build_schema(EXAMPLE["questions"])
    return engine.resolve_template(schema["questions"], schema["format"])


def test_same_request_same_canvas(client):
    client.post("/v1/systemone", json=EXAMPLE)
    client.post("/v1/systemone", json=EXAMPLE)
    first, second = client.app.state.engine.runtime.reads
    assert first["canvas"] == second["canvas"]


def test_samples_and_sequential_work(client):
    r = client.post("/v1/systemone", json=dict(EXAMPLE, samples=3))
    assert r.status_code == 200 and len(client.app.state.engine.runtime.reads) == 3
    qs = {f"k{i}": {"type": "noul", "instructions": f"question {i}"} for i in range(24)}
    r = client.post("/v1/systemone", json={"state": "x", "model": "jev-latest", "questions": qs, "sequential": True})
    assert r.status_code == 200, r.text
    reads = client.app.state.engine.runtime.reads[3:]
    assert len(reads) > 1 and len(reads[1]["prompt"]) > len(reads[0]["prompt"])  # later chunks carry earlier answers


def test_unsupported_options_are_refused(client):
    for extra, field in [({"images": [f"data:image/png;base64,{PNG}"]}, "images"), ({"think": 64}, "think"), ({"steps": 2}, "steps")]:
        r = client.post("/v1/systemone", json=dict(EXAMPLE, **extra))
        assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", field], r.text
    assert client.app.state.engine.runtime.reads == []
    assert client.post("/v1/systemone", json=dict(EXAMPLE, steps=1, think=0)).status_code == 200


def test_long_prompts_are_refused(tok, monkeypatch):
    monkeypatch.setattr(mlx_backend, "MlxRuntime", StubRuntime)
    with TestClient(create_app(Settings(backend="mlx", mlx_max_prompt=200), tokenizer=tok)) as c:
        assert c.post("/v1/systemone", json=EXAMPLE).status_code == 200
        r = c.post("/v1/systemone", json=dict(EXAMPLE, state="word " * 400))
        assert r.status_code == 422 and "limit is 200" in r.json()["detail"][0]["msg"]


def test_no_text_generation(client):
    r = client.post("/v1/chat/completions", json={"model": "diffusiongemma-26b", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 501 and "MLX" in r.json()["error"]["message"]
    assert [m["name"] for m in client.get("/v1/models").json()["models"]] == ["openjev-latest", "openjev-0.1"]
