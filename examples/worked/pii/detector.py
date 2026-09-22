"""Decide, per log line, whether it can leave the network as is, needs redacting, or needs a person."""
import re
from dataclasses import dataclass, field

# [snippet:pii-questions]
QUESTIONS = {
    "person": {"type": "noul", "instructions":
               "The log line identifies a person: a name, home address, phone number or email address."},
    "special_category": {"type": "noul", "instructions":
               "The log line mentions a person's medical condition, symptoms or treatment, or their religion, "
               "ethnicity, sexuality, trade-union membership or biometrics (special-category data, UK GDPR Article 9)."},
    "secret": {"type": "noul", "instructions":
               "The log line contains a credential or secret: a password, API key, access token or private key."},
    "financial": {"type": "noul", "instructions":
               "The log line contains a payment card number or a bank account number."},
}

# the structured cases stay with regexes: they are exact, free, and give a span to cut
PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "AWS_SECRET": re.compile(r"(?<=AWS_SECRET_ACCESS_KEY=)\S+"),
}
# [/snippet]

QUOTED = re.compile(r'"[^"]*"')


def luhn(text):
    d = [int(c) for c in re.sub(r"\D", "", text)][::-1]
    return len(d) >= 13 and sum(x if i % 2 == 0 else (x * 2 - 9 if x > 4 else x * 2) for i, x in enumerate(d)) % 10 == 0


@dataclass(frozen=True)
class Decision:
    route: str                      # "forward", "redact" or "quarantine"
    line: str                       # what may be shipped; the original line when quarantined
    p: dict = field(default_factory=dict)
    regex_hits: tuple = ()


# [snippet:pii-policy]
class PiiDetector:
    def __init__(self, s1, redact_at=0.8, review_at=0.3):
        self.s1, self.redact_at, self.review_at = s1, redact_at, review_at

    def check(self, line):
        hits = [(name, m.span()) for name, rx in PATTERNS.items() for m in rx.finditer(line)
                if name != "CARD" or luhn(m.group())]
        r = self.s1.system_one(line, QUESTIONS)          # the whole line, free text included
        p = {k: a.noul for k, a in r.nouls.items()}
        top = max(p.values())

        if hits or top >= self.redact_at:
            return Decision("redact", self._redact(line, hits, p), p, tuple(hits))
        if top >= self.review_at:
            return Decision("quarantine", line, p)           # unsure: a person decides
        return Decision("forward", line, p)

    def _redact(self, line, hits, p):
        for name, (a, b) in sorted(hits, key=lambda h: -h[1][0]):
            line = line[:a] + f"[{name}]" + line[b:]
        found = [k for k, v in p.items() if v >= self.redact_at]
        if found and not hits:
            # the model says *that* the line holds personal data, not *where*: drop the free text
            tag = f'"[REDACTED:{",".join(found)}]"'
            line = QUOTED.sub(tag, line) if QUOTED.search(line) else f"{line.split(' ', 1)[0]} [REDACTED:{','.join(found)}]"
        return line
# [/snippet]
