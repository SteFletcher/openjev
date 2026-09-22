"""Claude calls for the System 2 nodes: one function, a tier name in, text and cost out."""
from dataclasses import dataclass

import anthropic

MODELS = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5", "fable": "claude-fable-5-1"}
TIERS = list(MODELS)

# list price per million tokens (input, output), Anthropic API, September 2026.
# claude-opus-4-8 is here because it is a server-side fallback target.
PRICE = {"claude-haiku-4-5": (1, 5), "claude-sonnet-5": (2, 10), "claude-opus-5": (5, 25),
         "claude-opus-4-8": (5, 25), "claude-fable-5-1": (10, 50)}


@dataclass(frozen=True)
class Reply:
    text: str
    model: str            # the model that answered, which a fallback can change
    input_tokens: int
    output_tokens: int
    refused: bool = False

    @property
    def cost(self):
        pin, pout = PRICE.get(self.model, (0, 0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


# [snippet:claude-call]
class Claude:
    """claude(tier, system, prompt) -> Reply. Reads ANTHROPIC_API_KEY or an `ant auth login` profile."""

    def __init__(self, client=None, max_tokens=8000):
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens

    def __call__(self, tier, system, prompt):
        request = dict(model=MODELS[tier], max_tokens=self.max_tokens, system=system,
                       messages=[{"role": "user", "content": prompt}])
        if tier in ("opus", "fable"):
            # a declined request is re-run server-side on Anthropic's recommended fallback model
            r = self.client.beta.messages.create(**request, betas=["server-side-fallback-2026-07-01"],
                                                 fallbacks="default")
        else:
            r = self.client.messages.create(**request)
        if r.stop_reason == "refusal":   # check before reading content: a refusal can arrive with none
            return Reply("", r.model, r.usage.input_tokens, r.usage.output_tokens, refused=True)
        text = "".join(b.text for b in r.content if b.type == "text")
        return Reply(text, r.model, r.usage.input_tokens, r.usage.output_tokens)
# [/snippet]
