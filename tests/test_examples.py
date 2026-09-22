"""The enterprise examples run end to end against the offline mock, and the real client
sends only Jev's wire fields."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

from systemone import choice, confidence, noul, score, wire  # noqa: E402


def run(name):
    out = subprocess.run([sys.executable, str(EXAMPLES / name), "--mock"], capture_output=True, text=True,
                         cwd=EXAMPLES, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_evals_routes_each_case():
    out = run("evals.py")
    assert "pass rate 1/5" in out
    assert "late-taxi        generate -> judge -> regenerate -> judge -> deliver" in out
    assert "dpia             generate -> judge -> human" in out


def test_pii_redacts_what_regex_misses_and_quarantines_the_unsure():
    out = run("pii_in_logs.py")
    assert 'note="[REDACTED:person]"' in out
    assert 'notes="[REDACTED:special_category]"' in out
    assert "[CARD]" in out and "[AWS_SECRET]" in out and "[EMAIL]" in out
    assert "rash on arm" not in out.split("What reaches the log platform:")[1]
    assert "'quarantined for review': 1" in out


def test_model_routing_escalates_and_prices():
    out = run("model_routing.py")
    assert "haiku -> sonnet" in out
    assert " 1  none" in out and " 6  fable" in out
    assert "cheaper" in out


def test_wire_strips_mock_hints():
    qs = {"a": noul("x", true="t", false="f", mock=[("y", 0.9)]),
          "b": choice("x", {"p": "P", "q": None}, mock={"q": "z"}),
          "c": score("x", ["lo", "hi"], mock=[("z", 1)])}
    sent = wire(qs)
    assert all(not k.startswith("_") for q in sent.values() for k in q)
    assert sent["a"] == {"type": "noul", "instructions": "x", "criteria": {"true": "t", "false": "f"}}
    assert sent["b"]["criteria"] == {"p": "P", "q": None}


@pytest.mark.parametrize("p,expected", [([1.0, 0.0], 1.0), ([0.5, 0.5], 0.0), ([1.0], 1.0)])
def test_confidence_matches_server(p, expected):
    assert confidence(p) == pytest.approx(expected)
