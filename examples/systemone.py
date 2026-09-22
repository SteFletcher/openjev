"""A small stdlib client for POST /v1/systemone, question builders, and an offline mock.

Questions are plain dicts in Jev's wire shape. A question may carry a "_mock" hint that
the mock uses to fake an answer; the real client strips it before sending, so the model
never sees it.

    OPENJEV_BASE_URL   server to call (default http://127.0.0.1:8080)
    OPENJEV_API_KEY    bearer token, if the server needs one
    OPENJEV_MODEL      model name (default openjev-latest)
"""
import json
import math
import os
import re
import urllib.error
import urllib.request

BASE_URL = os.environ.get("OPENJEV_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.environ.get("OPENJEV_API_KEY", "")
MODEL = os.environ.get("OPENJEV_MODEL", "openjev-latest")


# --- question builders -------------------------------------------------------

def noul(instructions, true=None, false=None, mock=()):
    """Yes/no. mock: [(regex, p_yes), ...], first match wins, otherwise p_yes 0.03."""
    q = {"type": "noul", "instructions": instructions, "_mock": list(mock)}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


def choice(instructions, options, mock=None, default=None):
    """Pick one. options: {key: description}. mock: {key: regex}; default is the fallback key."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options),
            "_mock": dict(mock or {}), "_default": default or next(iter(options))}


def score(instructions, levels, mock=()):
    """A level on an ordered scale, lowest first. mock: [(regex, level), ...], first match wins."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels), "_mock": list(mock)}


def wire(questions):
    return {k: {f: v for f, v in q.items() if not f.startswith("_")} for k, q in questions.items()}


# --- answers ----------------------------------------------------------------

def confidence(p):
    """1 - H(p)/ln K, as the server computes it."""
    k = len(p)
    if k < 2:
        return 1.0
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def _peaked(k, top, p_top=0.9):
    rest = (1 - p_top) / (k - 1) if k > 1 else 0
    return [p_top if i == top else rest for i in range(k)]


def mock_answer(state, q):
    text = state if isinstance(state, str) else json.dumps(state)
    hit = lambda rx: re.search(rx, text, re.I | re.S)
    if q["type"] == "noul":
        p = next((p for rx, p in q["_mock"] if hit(rx)), 0.03)
        return {"type": "noul", "noul": p}
    if q["type"] == "choice":
        keys = list(q["criteria"])
        top = next((k for k, rx in q["_mock"].items() if hit(rx)), q["_default"])
        p = _peaked(len(keys), keys.index(top))
        return {"type": "choice", "choice": top, "probabilities": dict(zip(keys, p)), "confidence": confidence(p)}
    levels = q["criteria"]
    top = next((lvl for rx, lvl in q["_mock"] if hit(rx)), 0)
    p = _peaked(len(levels), top, 0.85)
    return {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
            "legend": {str(i): lv for i, lv in enumerate(levels)},
            "probabilities": {str(i): v for i, v in enumerate(p)}, "confidence": confidence(p)}


class SystemOne:
    """ask(state, questions) -> {question_id: answer}. Keeps a running token count."""

    def __init__(self, mock=False, base_url=BASE_URL, api_key=API_KEY, model=MODEL, timeout=60):
        self.mock, self.base_url, self.api_key, self.model, self.timeout = mock, base_url, api_key, model, timeout
        self.input_tokens = 0
        self.calls = 0

    def ask(self, state, questions, **options):
        self.calls += 1
        if self.mock:
            self.input_tokens += len(state if isinstance(state, str) else json.dumps(state)) // 4 + 20 * len(questions)
            return {k: mock_answer(state, q) for k, q in questions.items()}
        body = {"model": self.model, "state": state, "questions": wire(questions), **options}
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code} from {self.base_url}: {e.read().decode(errors='replace')}") from None
        except urllib.error.URLError as e:
            raise SystemExit(f"cannot reach {self.base_url} ({e.reason}). Start OpenJev, or pass --mock.") from None
        self.input_tokens += out.get("usage", {}).get("input_tokens", 0)
        return out["answers"]


def client_from_argv(argv):
    mock = "--mock" in argv
    if mock:
        print("(offline mock: answers come from keyword hints, not a model)\n")
    return SystemOne(mock=mock)
