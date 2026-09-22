# Enterprise examples

Three problems where a System One read (Jev, or OpenJev on your own hardware) does a better
job than the rule or the LLM call it replaces. Each is a small graph: nodes do the work, and
System One reads sit on the edges and decide where each item goes next. The write-up is at
<https://stefletcher.github.io/openjev/#enterprise>.

| Example | Problem | System One reads | Graph |
|---|---|---|---|
| [`evals.py`](evals.py) | Grading an internal policy assistant, offline and inline | faithful? answers the question? leaks personal data? quality 0–4 | generate → judge → deliver / regenerate / human |
| [`pii_in_logs.py`](pii_in_logs.py) | Personal data in application logs that regexes cannot see | person? special category? secret? card details? | scan_regex → read → forward / redact / quarantine |
| [`model_routing.py`](model_routing.py) | Sending each log event to the cheapest model that can deal with it | actionable? complexity 0–4? needs code? then: answer adequate? | route → drop / analyse(tier) → check → done / escalate |

Standard library only, Python 3.10 or later. `systemone.py` is the client and `graph.py` is a
forty-line graph runner, so the shape is visible without LangGraph or ADK. The System One
reads map directly onto their conditional edges.

```bash
python examples/evals.py --mock            # offline: answers come from keyword hints
python examples/pii_in_logs.py --mock
python examples/model_routing.py --mock

OPENJEV_BASE_URL=http://127.0.0.1:8080 python examples/evals.py    # a real OpenJev
OPENJEV_BASE_URL=https://api.codiv.ai OPENJEV_API_KEY=sk-codiv-... python examples/model_routing.py
```

The mock exists so the flow runs anywhere, and the tests use it. It is not a model. Run the
examples against a real server before you draw any conclusion about answer quality, and
build a labelled set for each decision before you trust a threshold.

For `pii_in_logs.py` in particular, use OpenJev on your own network. Sending raw logs to a
hosted API to find out whether they contain personal data is itself a disclosure.

The LLM calls in `evals.py` (regenerate) and `model_routing.py` (analyse) are stubbed with
canned answers, so the examples need no API keys. Replace them with real calls to the model
IDs in `model_routing.py`; the routing and checking stay the same.
