"""Finding personal data in logs before they leave the building.

Regexes are good at the structured cases: email addresses, card numbers, cloud keys. They
miss the ones that cause the incidents: a customer's name and address pasted into a
support note, a health condition in a claims comment, a search query that says more than
it should. A System One read answers "is there personal data here, and what kind" for
every line, with a probability you can put a policy on.

    python examples/pii_in_logs.py            # against a running OpenJev
    python examples/pii_in_logs.py --mock     # offline

Run this against OpenJev on your own network. Sending raw logs to a hosted API to find out
whether they contain personal data is itself a disclosure.

The graph, per line:

    scan_regex -> read_s1 -> forward                   nothing found
                          -> redact -> forward         found, by regex or with P >= 0.8
                          -> quarantine                0.3 <= P < 0.8, a person decides
"""
import re
import sys

from graph import Graph, path
from systemone import client_from_argv, noul

LOGS = [
    '2026-09-22T09:14:03Z INFO  checkout   order=48213 status=paid amount=42.10 GBP',
    '2026-09-22T09:14:07Z WARN  auth       failed login for user=j.patel@acme-retail.co.uk from 81.2.69.160',
    '2026-09-22T09:15:12Z ERROR support    ticket=7731 note="Customer Margaret O\'Neill says the parcel for 14 Harbour Row, Whitby went to a neighbour"',
    '2026-09-22T09:15:40Z INFO  claims     claim=C-2291 notes="claimant reports type 2 diabetes, requests a reasonable adjustment"',
    '2026-09-22T09:16:02Z DEBUG payments   gateway response {"card":"4111 1111 1111 1111","exp":"12/28"}',
    '2026-09-22T09:16:30Z DEBUG deploy     env loaded AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
    '2026-09-22T09:17:11Z INFO  search     query="rash on arm after new medication" session=ab12',
    '2026-09-22T09:17:15Z INFO  health     /ready ok latency_ms=12',
]

QUESTIONS = {
    "person": noul(
        "The log line identifies a person: a name, home address, phone number or email address.",
        mock=[(r"@|O'Neill|Harbour Row", 0.96)]),
    "special_category": noul(
        "The log line mentions a person's medical condition, symptoms or treatment, or their religion, "
        "ethnicity, sexuality, trade-union membership or biometrics (special-category data, UK GDPR Article 9).",
        mock=[(r"diabetes", 0.95), (r"rash on arm|medication", 0.52)]),
    "secret": noul(
        "The log line contains a credential or secret: a password, API key, access token or private key.",
        mock=[(r"SECRET|api[_-]?key|password=", 0.97)]),
    "financial": noul(
        "The log line contains a payment card number or a bank account number.",
        mock=[(r"\b(?:\d[ -]?){13,16}\b", 0.97)]),
}

PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "card": r"\b(?:\d[ -]?){13,19}\b",
    "aws_secret": r"(?<=AWS_SECRET_ACCESS_KEY=)\S+",
}

REDACT_AT, REVIEW_AT = 0.8, 0.3


def luhn(digits):
    d = [int(c) for c in re.sub(r"\D", "", digits)][::-1]
    return len(d) >= 13 and sum(x if i % 2 == 0 else (x * 2 - 9 if x > 4 else x * 2) for i, x in enumerate(d)) % 10 == 0


def build(s1):
    g = Graph("pii-in-logs")

    @g.node
    def scan_regex(st):
        hits = []
        for name, rx in PATTERNS.items():
            for m in re.finditer(rx, st["line"]):
                if name == "card" and not luhn(m.group()):
                    continue
                hits.append((name, m.span()))
        st["regex_hits"] = hits
        return st

    @g.node
    def read_s1(st):
        # the whole line, including free text the regexes cannot see into
        st["p"] = {k: a["noul"] for k, a in s1.ask(st["line"], QUESTIONS).items()}
        return st

    @g.node
    def redact(st):
        line = st["line"]
        for name, (a, b) in sorted(st["regex_hits"], key=lambda h: -h[1][0]):
            line = line[:a] + f"[{name.upper()}]" + line[b:]
        # a category with no span to cut: drop the free-text value rather than guess at a span
        found = [k for k, p in st["p"].items() if p >= REDACT_AT]
        if found and not st["regex_hits"]:
            line = re.sub(r'"[^"]*"', f'"[REDACTED:{",".join(found)}]"', line)
        st["line_out"] = line
        return st

    @g.node
    def forward(st):
        st["outcome"] = "forwarded"
        st.setdefault("line_out", st["line"])

    @g.node
    def quarantine(st):
        st["outcome"] = "quarantined for review"

    def route(st):
        top = max(st["p"].values())
        if st["regex_hits"] or top >= REDACT_AT:
            return "redact"
        return "quarantine" if top >= REVIEW_AT else "forward"

    g.edge("scan_regex", "read_s1")
    g.edge("read_s1", route)
    g.edge("redact", "forward")
    return g


def main():
    s1 = client_from_argv(sys.argv)
    g = build(s1)
    shipped, counts = [], {}
    print(f"{'#':>2}  {'regex':18} {'person':>6} {'art9':>5} {'secret':>6} {'card':>5}  route")
    for i, line in enumerate(LOGS, 1):
        st = g.run("scan_regex", {"line": line})
        p = st["p"]
        regex = ",".join(sorted({n for n, _ in st["regex_hits"]})) or "-"
        route = path(st).split(" -> ", 2)[2]
        print(f"{i:>2}  {regex:18} {p['person']:6.2f} {p['special_category']:5.2f} "
              f"{p['secret']:6.2f} {p['financial']:5.2f}  {route}")
        counts[st["outcome"]] = counts.get(st["outcome"], 0) + 1
        if st["outcome"] == "forwarded":
            shipped.append(st["line_out"])
    print("\nWhat reaches the log platform:\n")
    for line in shipped:
        print("  " + line)
    print(f"\n{counts}. Regex alone would have forwarded lines 3, 4 and 7 untouched.")
    print(f"{s1.calls} reads, {s1.input_tokens:,} input tokens.")


if __name__ == "__main__":
    main()
