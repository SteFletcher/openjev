"""Offline eval run with a CI gate.

    python examples/worked/evals/run_eval.py examples/worked/evals/dataset.jsonl --out report.json \
        --min-pass 0.5 --min-agreement 0.85

Judges every case in parallel, compares the judge with the human labels where the dataset has
them, writes a JSON report, and exits non-zero if the gate fails, so a pipeline can block a
prompt or model change that makes answers worse.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.s1 import client  # noqa: E402
from evals.rubric import judge  # noqa: E402


# [snippet:evals-run]
def run(s1, cases, workers=8):
    with ThreadPoolExecutor(workers) as pool:   # reads are independent, so grade them in parallel
        verdicts = list(pool.map(lambda c: judge(s1, c["question"], c["context"], c["answer"]), cases))
    rows = [{"id": c["id"], "label": c.get("label"), **asdict(v)} for c, v in zip(cases, verdicts)]

    decided = [r for r in rows if r["outcome"] != "human" and r["label"]]
    labelled = [(v.p_good, r["label"] == "pass") for v, r in zip(verdicts, rows) if r["label"]]
    return {
        "cases": len(rows),
        "pass_rate": sum(r["outcome"] == "pass" for r in rows) / len(rows),
        "to_human": [r["id"] for r in rows if r["outcome"] == "human"],
        # how often the judge agrees with a person, on the cases it was willing to decide
        "agreement": sum(r["outcome"] == r["label"] for r in decided) / len(decided) if decided else None,
        # Brier score of the judge's P(good) against the labels: 0 is perfect, 0.25 is a coin toss
        "brier": sum((p - y) ** 2 for p, y in labelled) / len(labelled) if labelled else None,
        "rows": rows,
    }


def gate(report, min_pass, min_agreement):
    failures = []
    if report["pass_rate"] < min_pass:
        failures.append(f"pass rate {report['pass_rate']:.2f} < {min_pass}")
    if report["agreement"] is not None and report["agreement"] < min_agreement:
        failures.append(f"judge agreement {report['agreement']:.2f} < {min_agreement}")
    return failures
# [/snippet]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-pass", type=float, default=0.0)
    ap.add_argument("--min-agreement", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    report = run(client(), cases, args.workers)

    print(f"{'case':18} {'faithful':>8} {'answers':>8} {'PII':>6} {'quality':>8}  {'judge':6} label")
    for r in report["rows"]:
        print(f"{r['id']:18} {r['faithful']:8.2f} {r['answers_question']:8.2f} {r['personal_data']:6.2f} "
              f"{r['quality']:8.2f}  {r['outcome']:6} {r['label'] or '-'}")
    agreement = "n/a" if report["agreement"] is None else f"{report['agreement']:.2f}"
    brier = "n/a" if report["brier"] is None else f"{report['brier']:.3f}"
    print(f"\npass rate {report['pass_rate']:.2f}, agreement with labels {agreement}, Brier {brier}, "
          f"to a person: {', '.join(report['to_human']) or 'none'}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))

    failures = gate(report, args.min_pass, args.min_agreement)
    for f in failures:
        print(f"GATE FAILED: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
