"""The worked examples, end to end through the real TypeSafe SDK and LangGraph.

A fake /v1/systemone server stands in for the model (answers come from keyword rules) and a
fake Claude stands in for the Anthropic API, so this runs offline and costs nothing.
"""
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("typesafe_sdk")
pytest.importorskip("langgraph")
pytest.importorskip("anthropic")

WORKED = Path(__file__).resolve().parent.parent / "examples" / "worked"
sys.path.insert(0, str(WORKED))

from common.llm import Claude, Reply  # noqa: E402
from common.s1 import client  # noqa: E402

NOUL = {
    "faithful": [(r"automatically, no approval|26 weeks at full pay", 0.06), (r"Probably", 0.55), (r".", 0.94)],
    "answers_question": [(r"can't help", 0.05), (r".", 0.95)],
    "personal_data": [(r"\w@\w", 0.97)],
    "person": [(r"@|O'Neill|Harbour Row", 0.96)],
    "special_category": [(r"diabetes", 0.95), (r"rash on arm", 0.52)],
    "secret": [(r"SECRET", 0.97)],
    "financial": [(r"4111", 0.97)],
    "actionable": [(r"healthcheck", 0.02), (r".", 0.97)],
    "needs_code": [(r"Exception", 0.9)],
    "adequate": [(r"cause unclear", 0.1), (r".", 0.93)],
}
SCORE = {
    "quality": [(r"automatically|can't help|@|26 weeks at full", 1), (r"Probably", 2), (r".", 4)],
    "complexity": [(r"MFA bypass", 4), (r"evictions", 3), (r"Exception", 2), (r"exited with code", 1)],
}


def answer(qid, q, text):
    first = lambda rules, default: next((v for rx, v in rules if re.search(rx, text, re.I)), default)
    if q["type"] == "noul":
        return {"type": "noul", "noul": first(NOUL[qid], 0.03)}
    k = len(q["criteria"])
    top = first(SCORE[qid], 0)
    p = [0.85 if i == top else 0.15 / (k - 1) for i in range(k)]
    return {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
            "probabilities": {str(i): v for i, v in enumerate(p)}, "confidence": 0.6}


class FakeSystemOne(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["state"] if isinstance(body["state"], str) else json.dumps(body["state"])
        out = {"model": "openjev-0.1", "usage": {"input_tokens": len(text) // 4, "output_tokens": 0},
               "answers": {k: answer(k, q, text) for k, q in body["questions"].items()}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def s1():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSystemOne)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mp = pytest.MonkeyPatch()
    mp.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    mp.setenv("TYPESAFE_API_KEY", "test")
    yield client()
    mp.undo()
    server.shutdown()


class FakeClaude:
    """Stands in for common.llm.Claude: same call shape, canned text per (tier, keyword)."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, tier, system, prompt):
        self.calls.append(tier)
        text = next((t for (tr, kw), t in self.replies.items() if tr == tier and kw in prompt), f"{tier} analysis")
        model = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5",
                 "fable": "claude-fable-5-1"}[tier]
        return Reply(text, model, 500, 200)


# --- evals -------------------------------------------------------------------

def test_offline_eval_matches_labels_and_gates(s1):
    from evals.run_eval import gate, run
    cases = [json.loads(line) for line in (WORKED / "evals" / "dataset.jsonl").read_text().splitlines()]
    report = run(s1, cases)
    outcomes = {r["id"]: r["outcome"] for r in report["rows"]}
    assert outcomes["late-taxi"] == "fail" and outcomes["prod-access"] == "fail"
    assert outcomes["dpia"] == "human" and outcomes["usb-drives"] == "pass"
    assert report["agreement"] == 1.0
    assert gate(report, min_pass=0.9, min_agreement=0.8) == [f"pass rate {report['pass_rate']:.2f} < 0.9"]


def test_guard_regenerates_with_the_judges_reasons(s1):
    from evals.guard import build_guard
    claude = FakeClaude({("sonnet", "reviewer rejected"): "Only after 21:00, for business reasons, with approval in advance.",
                         ("sonnet", "Question"): "Yes, any taxi after 8pm is covered automatically, no approval needed."})
    st = build_guard(s1, claude).invoke({"question": "Can I expense a taxi home after working late?",
                                         "context": "Taxis are reimbursable after 21:00 with approval."})
    assert st["outcome"] == "delivered" and st["attempts"] == 2
    assert st["answer"].startswith("Only after 21:00")


# --- PII ---------------------------------------------------------------------

def test_log_filter_redacts_quarantines_and_keeps_order(s1):
    from pii.detector import PiiDetector
    from pii.log_filter import filter_stream
    out, quarantine = io.StringIO(), io.StringIO()
    lines = (WORKED / "pii" / "sample.log").read_text().splitlines()
    counts = filter_stream(PiiDetector(s1), lines, out, quarantine, workers=4)
    shipped = out.getvalue().splitlines()
    assert counts == {"forward": 2, "redact": 5, "quarantine": 1}
    assert [s[:20] for s in shipped] == [line[:20] for line in lines if "rash on arm" not in line]
    assert "[EMAIL]" in shipped[1] and "[IPV4]" in shipped[1]
    assert 'note="[REDACTED:person]"' in shipped[2]
    assert "[CARD]" in shipped[4] and "[AWS_SECRET]" in shipped[5]
    assert "rash on arm" in quarantine.getvalue()


# --- routing -----------------------------------------------------------------

def test_router_routes_escalates_and_adds_up_cost(s1):
    from routing.router import build_router
    from routing.run_router import route_all
    claude = FakeClaude({("haiku", "exited with code"): "Exit 137. Cause unclear."})
    events = (WORKED / "routing" / "events.log").read_text().splitlines()
    results = {e.split()[1]: st for e, st in route_all(build_router(s1, claude), events)}
    assert results["healthcheck"]["models"] == [] and results["healthcheck"]["answer"] == "(no model called)"
    assert results["db-replica-2"]["models"] == ["claude-haiku-4-5"]
    assert results["payment-svc"]["models"] == ["claude-sonnet-5"]
    assert results["cron"]["models"] == ["claude-haiku-4-5", "claude-sonnet-5"]
    assert results["checkout"]["models"] == ["claude-opus-5"]
    assert results["auth"]["models"] == ["claude-fable-5-1"]
    assert results["cron"]["cost"] == pytest.approx(Reply("", "claude-haiku-4-5", 500, 200).cost
                                                    + Reply("", "claude-sonnet-5", 500, 200).cost)


# --- the Claude wrapper -----------------------------------------------------

def test_claude_uses_server_side_fallbacks_only_for_opus_and_fable():
    seen = []

    def create(**kw):
        seen.append(kw)
        return SimpleNamespace(stop_reason="end_turn", model=kw["model"], content=[SimpleNamespace(type="text", text="ok")],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    fake = SimpleNamespace(messages=SimpleNamespace(create=create), beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    claude = Claude(client=fake)
    assert claude("haiku", "s", "p").text == "ok"
    claude("fable", "s", "p")
    assert "fallbacks" not in seen[0]
    assert seen[1]["fallbacks"] == "default" and seen[1]["betas"] == ["server-side-fallback-2026-07-01"]


def test_claude_reports_refusals_without_reading_content():
    r = SimpleNamespace(stop_reason="refusal", model="claude-opus-5", content=[], usage=SimpleNamespace(input_tokens=0, output_tokens=0))
    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: r)))
    assert Claude(client=fake)("opus", "s", "p").refused


def test_site_shows_the_code_that_runs():
    import subprocess
    root = WORKED.parent.parent
    out = subprocess.run([sys.executable, str(root / "site" / "snippets.py"), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_generate_answers_feeds_run_eval(s1):
    from evals.generate_answers import answer_all
    from evals.run_eval import run
    questions = [json.loads(line) for line in (WORKED / "evals" / "questions.jsonl").read_text().splitlines()]
    claude = FakeClaude({("sonnet", "annual leave"): "Up to 5 days, used by 31 March."})
    answers = answer_all(claude, questions)
    assert [a["id"] for a in answers] == [q["id"] for q in questions] and all(a["answer"] for a in answers)
    report = run(s1, answers)
    assert report["agreement"] is None and report["cases"] == len(questions)   # no labels on fresh answers


def test_ci_example_and_vector_config_reference_files_that_exist():
    root = WORKED.parent.parent
    ci = (WORKED / "evals" / "eval-gate.yml").read_text()
    for path in re.findall(r"examples/worked/\S+\.(?:py|jsonl)", ci):
        assert (root / path).exists(), path
    assert (WORKED / "pii" / "vector.toml").exists()
