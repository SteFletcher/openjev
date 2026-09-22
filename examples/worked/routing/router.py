"""Route each log event to the cheapest Claude model that can deal with it, check the answer,
and escalate one tier when it falls short.

route (System One) -> drop                                    nothing to do: no model called
                   -> analyse (Claude, tier) -> check (System One) -> done
                                                               -> escalate -> analyse ...
"""
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm import TIERS  # noqa: E402
from common.s1 import most_likely  # noqa: E402

SYSTEM = ("You are an SRE assistant. Given one log event or alert, say what most likely caused it "
          "and the next concrete step, in under 120 words. Say so plainly if you cannot tell.")

# [snippet:routing-questions]
ROUTE = {
    "actionable": {"type": "noul", "instructions": "Someone needs to act on or investigate this event."},
    "complexity": {
        "type": "score",
        "instructions": "How much analysis does a useful answer to this event need?",
        "criteria": [
            "routine: restate it in one line with the obvious next step",
            "simple: explain one known error and how to fix it",
            "moderate: diagnose one failing component from a stack trace or metrics",
            "hard: correlate several services or a timeline to find the root cause",
            "critical: security incident or possible data loss, needs careful high-stakes analysis",
        ],
    },
    "needs_code": {"type": "noul", "instructions": "A good answer needs someone to read or change application code."},
}
CHECK = {
    "adequate": {"type": "noul",
                 "instructions": "The analysis names a probable cause and a concrete next step for this event."},
}

LEVEL_TO_TIER = {0: "haiku", 1: "haiku", 2: "sonnet", 3: "opus", 4: "fable"}


def pick_tier(r):
    """Policy in code: most likely level -> tier, code work starts at Sonnet, unsure errs upwards."""
    cx = r.scores["complexity"]
    tier = TIERS.index(LEVEL_TO_TIER[int(most_likely(cx.probabilities))])
    if r.nouls["needs_code"].noul >= 0.5:
        tier = max(tier, TIERS.index("sonnet"))
    if cx.confidence < 0.4:
        tier = min(tier + 1, len(TIERS) - 1)
    return tier
# [/snippet]


# [snippet:routing-graph]
class RouteState(TypedDict, total=False):
    event: str
    actionable: float
    tier: int
    answer: str
    adequate: float
    models: Annotated[list, operator.add]     # every model that ran, appended by each analyse
    cost: Annotated[float, operator.add]      # dollars, summed across escalations
    s1_tokens: Annotated[int, operator.add]


def build_router(s1, claude, adequate_at=0.7, actionable_at=0.3):
    def route(st):
        r = s1.system_one(st["event"], ROUTE)
        return {"actionable": r.nouls["actionable"].noul, "tier": pick_tier(r),
                "s1_tokens": r.usage.input_tokens or 0}

    def analyse(st):
        reply = claude(TIERS[st["tier"]], SYSTEM, st["event"])
        return {"answer": reply.text, "models": [reply.model], "cost": reply.cost}

    def check(st):
        r = s1.system_one({"event": st["event"], "analysis": st["answer"]}, CHECK)
        return {"adequate": r.nouls["adequate"].noul, "s1_tokens": r.usage.input_tokens or 0}

    def after_check(st):
        at_top = st["tier"] == len(TIERS) - 1
        return "done" if st["adequate"] >= adequate_at or at_top else "escalate"

    g = StateGraph(RouteState)
    g.add_node("route", route)
    g.add_node("analyse", analyse)
    g.add_node("check", check)
    g.add_node("escalate", lambda st: {"tier": st["tier"] + 1})
    g.add_node("drop", lambda st: {"answer": "(no model called)"})
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda st: "analyse" if st["actionable"] >= actionable_at else "drop",
                            ["analyse", "drop"])
    g.add_edge("analyse", "check")
    g.add_conditional_edges("check", after_check, {"done": END, "escalate": "escalate"})
    g.add_edge("escalate", "analyse")
    g.add_edge("drop", END)
    return g.compile()
# [/snippet]
