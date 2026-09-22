"""One System One client for every worked example, pointed at Jev or OpenJev by environment.

TypeSafe's own SDK works unchanged against OpenJev, so switching between hosted Jev, hosted
OpenJev (Codiv) and OpenJev on your own GPU is a change of base URL, not of code.
"""
import os

from typesafe_sdk import TypeSafeClient

# [snippet:s1-client]
# TYPESAFE_BASE_URL  https://api.typesafe.ai  -> Jev
#                    https://api.codiv.ai     -> hosted OpenJev
#                    http://127.0.0.1:8080    -> OpenJev on your own hardware (the default)
# TYPESAFE_API_KEY   the SDK requires one; any value works for a local server without auth
def client() -> TypeSafeClient:
    return TypeSafeClient(
        base_url=os.environ.get("TYPESAFE_BASE_URL", "http://127.0.0.1:8080"),
        api_key=os.environ.get("TYPESAFE_API_KEY", "local"),
        model=os.environ.get("TYPESAFE_MODEL", "jev-latest"),  # OpenJev accepts Jev's name as an alias
    )
# [/snippet]


def most_likely(probabilities):
    """The most likely key of a choice or score distribution."""
    return max(probabilities, key=probabilities.get)
