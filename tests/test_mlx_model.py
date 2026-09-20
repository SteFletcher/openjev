"""The MLX backend against real weights, on Apple silicon:

    OPENJEV_MLX_TEST_MODEL=~/models/diffusiongemma-26B-A4B-it-4bit pytest tests/test_mlx_model.py
"""
import os

import pytest
from fastapi.testclient import TestClient

MODEL = os.environ.get("OPENJEV_MLX_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set OPENJEV_MLX_TEST_MODEL to an MLX DiffusionGemma checkpoint")

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
    "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
}
STATES = {  # state -> (urgent, team, tone level)
    "Everything is down and we have a demo with our biggest client at noon.": (True, "outage", 2),
    "I was charged twice this month. Not urgent, just let me know when it's refunded. Thanks!": (False, "billing", 0),
    "Love the product. Any chance you could add a dark mode at some point?": (False, "feature", 0),
}


@pytest.fixture(scope="module")
def client():
    from openjev.api import create_app
    from openjev.config import Settings

    with TestClient(create_app(Settings(backend="mlx", mlx_model=os.path.expanduser(MODEL)))) as c:
        yield c


def ask(client, state, questions=QUESTIONS, **extra):
    r = client.post("/v1/systemone", json={"state": state, "model": "openjev-latest", "questions": questions, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_readme_example_and_friends(client):
    for state, (urgent, team, tone) in STATES.items():
        body = ask(client, state)
        a = body["answers"]
        assert (a["urgent"]["noul"] > 0.9) == urgent and (a["urgent"]["noul"] < 0.1) != urgent, (state, a["urgent"])
        assert a["team"]["choice"] == team and a["team"]["confidence"] > 0.9, (state, a["team"])
        assert abs(a["tone"]["score"] - tone) < 0.25, (state, a["tone"])
        assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-6
        assert body["usage"]["input_tokens"] > 100 and body["usage"]["output_tokens"] == 0


def test_same_request_same_answer(client):
    state = next(iter(STATES))
    assert ask(client, state) == ask(client, state)
    assert ask(client, state, samples=3) == ask(client, state, samples=3)


def test_a_cached_prefill_reads_the_same(client):
    """The decoder pass must leave the prompt's cache as it found it, or reusing
    it for re-reads and samples would change their answers."""
    engine = client.app.state.engine
    rt = engine.runtime
    schema = engine.build_schema(QUESTIONS)
    template, slots = engine.resolve_template(schema["questions"], schema["format"])
    prompt = engine.chat_prompt_ids(engine.system_text(schema["questions"], schema["format"]), "The invoice is wrong.")
    canvases = [engine.build_canvas(template, slots, seed) for seed in (1, 2, 3)]

    def fresh(canvas):
        rt.prefills.clear()
        return rt.read(prompt, canvas, slots)

    def reused():
        rt.prefills.clear()
        return [rt.read(prompt, c, slots) for c in canvases]

    cold = [rt.pool.submit(fresh, c).result() for c in canvases]
    assert rt.pool.submit(reused).result() == cold  # bitwise


def test_many_questions_chunk_and_run_in_sequence(client):
    qs = {f"k{i}": {"type": "noul", "instructions": f"The message mentions the number {i}"} for i in range(24)}
    state = "The numbers I care about are 3, 11 and 20."
    for extra in ({}, {"sequential": True}):
        a = ask(client, state, qs, **extra)["answers"]
        assert len(a) == 24 and all(0 <= v["noul"] <= 1 for v in a.values())


def test_refusals(client):
    r = client.post("/v1/systemone", json={"state": "x", "model": "openjev-latest", "questions": QUESTIONS, "think": 64})
    assert r.status_code == 400 and r.json()["detail"].startswith("think")
    r = client.post("/v1/chat/completions", json={"model": "diffusiongemma-26b", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 501


def test_the_prompt_cache_is_bounded_in_tokens(client):
    from openjev import mlx_backend

    rt = client.app.state.engine.runtime
    for i in range(12):
        ask(client, f"Order {i} arrived broken. " + "Please help. " * 600, {"refund": {"type": "noul", "instructions": "The customer wants a refund"}})
    assert 1 <= len(rt.prefills) < 12
    assert sum(map(len, rt.prefills)) <= mlx_backend.PROMPT_CACHE_TOKENS
