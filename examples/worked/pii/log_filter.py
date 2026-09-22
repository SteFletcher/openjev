"""A streaming log filter: lines in on stdin, safe lines out on stdout, the rest to a quarantine file.

    python examples/worked/pii/log_filter.py --quarantine quarantine.log < examples/worked/pii/sample.log

    # in a pipeline, in front of whatever ships logs off the box
    kubectl logs -f deploy/support-api | python log_filter.py --quarantine /var/log/pii-review.log \
        | vector --config vector.toml

Reads run concurrently (bounded by --workers) and lines leave in the order they arrived.
Run it against OpenJev on your own network: sending raw logs to a hosted API to find out
whether they contain personal data is itself a disclosure.
"""
import argparse
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.s1 import client  # noqa: E402
from pii.detector import PiiDetector  # noqa: E402


# [snippet:pii-stream]
def filter_stream(detector, lines, out, quarantine, workers=16):
    """Check lines concurrently, emit them in order. Returns a count per route."""
    counts = Counter()
    pending = deque()

    def emit(future):
        d = future.result()
        counts[d.route] += 1
        (quarantine if d.route == "quarantine" else out).write(d.line + "\n")

    with ThreadPoolExecutor(workers) as pool:
        for line in lines:
            line = line.rstrip("\n")
            if not line:
                continue
            pending.append(pool.submit(detector.check, line))
            while len(pending) >= workers * 2:   # bounded: memory stays flat on an endless stream
                emit(pending.popleft())
        while pending:
            emit(pending.popleft())
    return counts
# [/snippet]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quarantine", type=Path, default=Path("quarantine.log"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--redact-at", type=float, default=0.8)
    ap.add_argument("--review-at", type=float, default=0.3)
    args = ap.parse_args(argv)
    detector = PiiDetector(client(), args.redact_at, args.review_at)
    with args.quarantine.open("a") as q:
        counts = filter_stream(detector, sys.stdin, sys.stdout, q, args.workers)
    print(f"pii-filter: {dict(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
