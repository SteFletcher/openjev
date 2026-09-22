# Worked examples

How the three enterprise use cases would be built for real: TypeSafe's SDK for the System One
reads (so the same code runs on Jev, hosted OpenJev or OpenJev on your own GPU),
[LangGraph](https://langchain-ai.github.io/langgraph/) for the graphs, and the Anthropic SDK
for the Claude calls. The write-up walks through each one step by step:
<https://stefletcher.github.io/openjev/#enterprise>.

```bash
pip install -e '.[examples]'                     # typesafe-sdk, langgraph, anthropic
export TYPESAFE_BASE_URL=http://127.0.0.1:8080   # or https://api.codiv.ai, or https://api.typesafe.ai
export TYPESAFE_API_KEY=...                      # any value for a local server without auth
export ANTHROPIC_API_KEY=...                     # only the guard and the router call Claude
```

| Use case | Files | Run |
|---|---|---|
| Evals | `evals/rubric.py` the rubric and verdict policy · `evals/run_eval.py` offline run with a CI gate · `evals/guard.py` the judge inline, as a LangGraph guard · `evals/eval-gate.yml` a GitHub Actions job · `evals/dataset.jsonl` eight labelled cases | `python examples/worked/evals/run_eval.py examples/worked/evals/dataset.jsonl --min-agreement 0.85` |
| PII in logs | `pii/detector.py` regexes, questions and the redact / quarantine / forward policy · `pii/log_filter.py` a concurrent, order-preserving stdin → stdout filter · `pii/sample.log` | `python examples/worked/pii/log_filter.py < examples/worked/pii/sample.log` |
| Model routing | `routing/router.py` route and check reads as LangGraph conditional edges · `routing/run_router.py` a CLI that streams events · `routing/events.log` | `python examples/worked/routing/run_router.py examples/worked/routing/events.log` |

`common/s1.py` builds the System One client from the environment. `common/llm.py` calls Claude
by tier (Haiku 4.5, Sonnet 5, Opus 5, Fable 5.1), uses server-side `fallbacks: "default"` for
Opus and Fable, checks `stop_reason` before reading content, and prices every reply.

`tests/test_worked.py` runs all three end to end against a fake System One server and a fake
Claude, so the plumbing is tested offline and for free. That says nothing about answer quality:
run against a real server and build a labelled set per decision before you trust a threshold.

The code blocks on the site are copied from the `# [snippet:…]` regions in these files by
`site/snippets.py`. Edit the code here, run `python site/snippets.py`, and a test keeps the two
in step.
