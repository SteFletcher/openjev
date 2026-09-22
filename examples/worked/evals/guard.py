"""The same judge, inline: a LangGraph guard in front of a policy assistant.

    python examples/worked/evals/guard.py "Can I expense a taxi home after working late?" \
        --context "Taxis are reimbursable when travel after 21:00 is required for business reasons \
and approved in advance by a line manager."

generate (Sonnet) -> judge (System One) -> deliver
                                        -> regenerate (Sonnet, told what failed) -> judge ...
                                        -> human
"""
import argparse
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm import Claude  # noqa: E402
from common.s1 import client  # noqa: E402
from evals.rubric import judge  # noqa: E402

SYSTEM = ("You answer staff questions about company policy. Use only the policy text you are given. "
          "If it does not answer the question, say so. Never include names or contact details of staff.")
MAX_ATTEMPTS = 2


# [snippet:evals-guard]
class GuardState(TypedDict, total=False):
    question: str
    context: str
    answer: str
    attempts: int
    verdict: object
    outcome: str


def build_guard(s1, claude):
    def prompt(st, feedback=""):
        return f"Policy:\n{st['context']}\n\nQuestion: {st['question']}{feedback}"

    def generate(st):
        return {"answer": claude("sonnet", SYSTEM, prompt(st)).text, "attempts": 1}

    def judge_node(st):
        return {"verdict": judge(s1, st["question"], st["context"], st["answer"])}

    def regenerate(st):
        why = "; ".join(st["verdict"].reasons)
        feedback = f"\n\nA reviewer rejected this draft ({why}):\n{st['answer']}\nWrite a corrected answer."
        return {"answer": claude("sonnet", SYSTEM, prompt(st, feedback)).text, "attempts": st["attempts"] + 1}

    def after_judge(st):
        outcome = st["verdict"].outcome
        if outcome == "pass":
            return "deliver"
        if outcome == "fail" and st["attempts"] < MAX_ATTEMPTS:
            return "regenerate"
        return "human"

    g = StateGraph(GuardState)
    g.add_node("generate", generate)
    g.add_node("judge", judge_node)
    g.add_node("regenerate", regenerate)
    g.add_node("deliver", lambda st: {"outcome": "delivered"})
    g.add_node("human", lambda st: {"outcome": "queued for a person"})
    g.add_edge(START, "generate")
    g.add_edge("generate", "judge")
    g.add_conditional_edges("judge", after_judge, ["deliver", "regenerate", "human"])
    g.add_edge("regenerate", "judge")
    g.add_edge("deliver", END)
    g.add_edge("human", END)
    return g.compile()
# [/snippet]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Answer a policy question behind a System One judge.")
    ap.add_argument("question")
    ap.add_argument("--context", required=True)
    args = ap.parse_args(argv)
    st = build_guard(client(), Claude()).invoke({"question": args.question, "context": args.context})
    v = st["verdict"]
    print(f"{st['outcome']} after {st['attempts']} attempt(s): faithful {v.faithful:.2f}, "
          f"answers {v.answers_question:.2f}, personal data {v.personal_data:.2f}, quality {v.quality:.2f}")
    print(st["answer"])


if __name__ == "__main__":
    main()
