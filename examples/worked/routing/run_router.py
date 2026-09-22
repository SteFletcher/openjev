"""Route a file (or stdin) of log events, one per line, and report what each cost.

    python examples/worked/routing/run_router.py examples/worked/routing/events.log

Needs a System One server (TYPESAFE_BASE_URL) and Claude credentials (ANTHROPIC_API_KEY, or an
`ant auth login` profile). The Claude calls are real, so this spends a few cents.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm import Claude  # noqa: E402
from common.s1 import client  # noqa: E402
from routing.router import build_router  # noqa: E402

JEV_PRICE = 0.042  # $ per million input tokens


# [snippet:routing-run]
def route_all(graph, events):
    for event in events:
        st = graph.invoke({"event": event, "models": [], "cost": 0.0, "s1_tokens": 0})
        yield event, st
# [/snippet]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("events", nargs="?", type=argparse.FileType(), default=sys.stdin)
    args = ap.parse_args(argv)
    events = (line.strip() for line in args.events if line.strip())   # lazy: works on `tail -F`
    graph = build_router(client(), Claude())

    llm_total = s1_tokens = 0
    for event, st in route_all(graph, events):
        models = " -> ".join(st["models"]) or "none"
        llm_total += st["cost"]
        s1_tokens += st["s1_tokens"]
        print(f"{models:38} ${st['cost']:.5f}  {event[:70]}")
        print(f"{'':38}           {st['answer'][:160].replace(chr(10), ' ')}", flush=True)
    print(f"\nClaude ${llm_total:.4f}; System One {s1_tokens:,} input tokens "
          f"(${s1_tokens * JEV_PRICE / 1e6:.6f} at Jev's price, nothing extra on your own OpenJev).")


if __name__ == "__main__":
    main()
