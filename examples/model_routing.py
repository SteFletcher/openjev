"""Model routing for log events: send each one to the cheapest model that can handle it.

An ops assistant watches the error and alert stream. Most events need no model at all, a
few need a one-line summary, and a handful need real diagnosis. Sending everything to the
frontier model is slow and expensive; asking an LLM to do the routing costs seconds per
event. A System One read does the routing in one pass, and a second read checks the answer
and escalates one tier when it falls short.

    python examples/model_routing.py            # against a running OpenJev
    python examples/model_routing.py --mock     # offline

The graph, per event:

    route -> drop                                   not actionable
          -> analyse(tier) -> check -> done         answer is good enough
                                    -> escalate -> analyse(tier + 1) -> check ...

The LLM calls are stubbed so the example runs without API keys. Swap `analyse` for real
calls; the routing and checking do not change.
"""
import sys

from graph import Graph
from systemone import client_from_argv, noul, score

# list price per million tokens (input, output), Anthropic API, September 2026
TIERS = ["haiku", "sonnet", "opus", "fable"]
PRICE = {"haiku": (1, 5), "sonnet": (2, 10), "opus": (5, 25), "fable": (10, 50)}
MODEL_ID = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5", "fable": "claude-fable-5-1"}
OUTPUT_TOKENS = {"haiku": 200, "sonnet": 500, "opus": 1000, "fable": 1500}
SYSTEM_PROMPT_TOKENS = 600
JEV_PRICE = 0.042

EVENTS = [
    "INFO  healthcheck /ready 200 latency_ms=11",
    "WARN  db-replica-2 disk usage 91% on /var/lib/postgresql, growing 2%/hour",
    "ERROR payment-svc java.lang.NullPointerException at RefundHandler.java:88 "
    "(RefundHandler.apply <- RefundController.post); 212 occurrences in 10 min since 09:02",
    "ERROR cron nightly-export exited with code 137 after 41 min",
    "ERROR checkout p99 latency 4.2s (SLO 800ms); inventory-svc upstream timeouts; redis evictions spiking; "
    "pricing-svc deployed 08:55; 3 pods OOMKilled in cart-svc",
    "CRITICAL auth 3,000 successful logins from a new ASN in 5 min for admin accounts; MFA bypass flag set on 12 users",
]

ROUTE_QUESTIONS = {
    "actionable": noul(
        "Someone needs to act on or investigate this event.",
        mock=[(r"healthcheck", 0.02), (r".", 0.97)]),
    "complexity": score(
        "How much analysis does a useful answer to this event need?",
        ["routine: restate it in one line with the obvious next step",
         "simple: explain one known error and how to fix it",
         "moderate: diagnose one failing component from a stack trace or metrics",
         "hard: correlate several services or a timeline to find the root cause",
         "critical: security incident or possible data loss, needs careful high-stakes analysis"],
        mock=[(r"MFA bypass|successful logins", 4), (r"upstream .*timeouts|evictions", 3),
              (r"Exception", 2), (r"exited with code", 1), (r"disk usage", 0)]),
    "needs_code": noul(
        "A good answer needs someone to read or change application code.",
        mock=[(r"Exception|\.java", 0.9)]),
}

CHECK = {
    "adequate": noul(
        "The analysis names a probable cause and a concrete next step for this event.",
        mock=[(r"cause unclear", 0.1), (r".", 0.93)]),
}

# Stand-ins for the model's answer at each tier. Real code calls the model named in MODEL_ID.
STUB_ANSWERS = {
    ("haiku", 2): "Disk on db-replica-2 is at 91% and rising about 2%/hour. Expand the volume or prune WAL within four hours.",
    ("haiku", 4): "The job exited with code 137. Cause unclear.",
    ("sonnet", 4): "Exit 137 is SIGKILL, almost always the OOM killer. The export has grown past its 2 Gi limit: "
                   "raise the limit or stream the export in batches.",
    ("sonnet", 3): "RefundHandler.java:88 dereferences order.getPayment() for refunds created before payment settles. "
                   "Guard the null and return 409. The change started with the 09:00 deploy.",
    ("opus", 5): "The pricing-svc deploy at 08:55 doubled cart payload size, cart-svc pods hit their memory limit and "
                 "restarted, and the retries flooded inventory-svc and evicted redis. Roll back pricing-svc, then raise cart-svc limits.",
    ("fable", 6): "Treat as an active account takeover: revoke sessions for the affected admins, block the ASN, "
                  "disable the MFA bypass flag and start the incident process. Preserve auth logs.",
}


def cost(tier, event):
    tin = SYSTEM_PROMPT_TOKENS + len(event) // 4
    pin, pout = PRICE[tier]
    return (tin * pin + OUTPUT_TOKENS[tier] * pout) / 1e6


def build(s1):
    g = Graph("log-model-router")

    @g.node
    def route(st):
        a = s1.ask(st["event"], ROUTE_QUESTIONS)
        cx = a["complexity"]
        level = max(cx["probabilities"], key=cx["probabilities"].get)   # most likely level, not the mean
        tier = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}[int(level)]
        if a["needs_code"]["noul"] >= 0.5:
            tier = max(tier, 1)                  # code work starts at Sonnet
        if cx["confidence"] < 0.4:
            tier = min(tier + 1, len(TIERS) - 1)  # unsure how hard it is: err upwards
        st.update(actionable=a["actionable"]["noul"], tier=tier, spend=0.0, tiers_used=[])
        return st

    @g.node
    def analyse(st):
        t = TIERS[st["tier"]]
        st["tiers_used"].append(t)
        st["answer"] = STUB_ANSWERS.get((t, st["n"]), f"[{MODEL_ID[t]} analysis of event {st['n']}]")
        st["spend"] += cost(t, st["event"])
        return st

    @g.node
    def check(st):
        st["adequate"] = s1.ask({"event": st["event"], "analysis": st["answer"]}, CHECK)["adequate"]["noul"]
        return st

    @g.node
    def escalate(st):
        st["tier"] += 1
        return st

    @g.node
    def drop(st):
        st["answer"] = "(no model called)"

    @g.node
    def done(st):
        pass

    g.edge("route", lambda st: "analyse" if st["actionable"] >= 0.3 else "drop")
    g.edge("analyse", "check")
    g.edge("check", lambda st: "done" if st["adequate"] >= 0.7 or st["tier"] == len(TIERS) - 1 else "escalate")
    g.edge("escalate", "analyse")
    return g


def main():
    s1 = client_from_argv(sys.argv)
    g = build(s1)
    routed = baseline = 0.0
    print(f"{'#':>2}  {'models used':16} {'cost':>9}  answer")
    for n, event in enumerate(EVENTS, 1):
        st = g.run("route", {"event": event, "n": n})
        used = " -> ".join(st.get("tiers_used") or ["none"])
        routed += st.get("spend", 0.0)
        baseline += cost("opus", event)
        print(f"{n:>2}  {used:16} ${st.get('spend', 0.0):8.5f}  {st['answer'][:90]}")
    s1_cost = s1.input_tokens * JEV_PRICE / 1e6
    print(f"\nRouted: ${routed:.4f} in LLM calls + ${s1_cost:.6f} for {s1.calls} System One reads.")
    print(f"Everything to Opus 5: ${baseline:.4f}. Routing costs {100 * (routed + s1_cost) / baseline:.0f}% of that,")
    print("and sends the security incident to a stronger model than the default, for ~100 ms per event.\n")
    at_scale()


# A day of events is mostly noise and routine. Share of events by most likely route:
DAILY_EVENTS = 100_000
MIX = {"none": 0.60, "haiku": 0.30, "sonnet": 0.07, "opus": 0.025, "fable": 0.005}
TYPICAL_EVENT = "x" * 400   # ~100 tokens of log excerpt


def at_scale():
    s1_tokens = DAILY_EVENTS * (len(TYPICAL_EVENT) // 4 + 150) * 1.4   # route read on all, check read on the 40% analysed
    routed = sum(DAILY_EVENTS * share * cost(t, TYPICAL_EVENT) for t, share in MIX.items() if t != "none")
    routed += s1_tokens * JEV_PRICE / 1e6
    everything = DAILY_EVENTS * cost("opus", TYPICAL_EVENT)
    print(f"At {DAILY_EVENTS:,} events a day with the mix {MIX}:")
    print(f"  routed ${routed:,.0f}/day vs everything to Opus 5 ${everything:,.0f}/day "
          f"({everything / routed:.1f}x cheaper, ${(everything - routed) * 30:,.0f} a month).")


if __name__ == "__main__":
    main()
