"""Answer every eval question with the current prompt, so the judge can grade this version.

    python examples/worked/evals/generate_answers.py examples/worked/evals/questions.jsonl > answers.jsonl
    python examples/worked/evals/run_eval.py answers.jsonl --min-pass 0.9

In CI this runs on every change to the prompt or model; run_eval.py then fails the build if
the answers got worse. Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm import Claude  # noqa: E402
from evals.guard import SYSTEM  # noqa: E402


def answer_all(claude, questions, tier="sonnet", workers=4):
    def one(q):
        prompt = f"Policy:\n{q['context']}\n\nQuestion: {q['question']}"
        return {**q, "answer": claude(tier, SYSTEM, prompt).text}
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, questions))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("questions", type=argparse.FileType())
    ap.add_argument("--tier", default="sonnet", choices=["haiku", "sonnet", "opus", "fable"])
    args = ap.parse_args(argv)
    questions = [json.loads(line) for line in args.questions if line.strip()]
    for row in answer_all(Claude(), questions, args.tier):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
