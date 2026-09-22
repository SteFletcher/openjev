"""Copy code regions from examples/ into site/index.html, so the page shows the code that runs.

A region in a source file is marked with comments:

    # [snippet:evals-rubric]
    ...
    # [/snippet]

and a place in the page with:

    <!-- snippet:evals-rubric examples/worked/evals/rubric.py -->...<!-- /snippet -->

    python site/snippets.py           # rewrite the page
    python site/snippets.py --check   # exit 1 if the page is out of date (used by the tests)
"""
import html
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "site" / "index.html"
REPO = "https://github.com/SteFletcher/openjev/blob/main/"
PLACE = re.compile(r"<!-- snippet:(?P<name>[\w-]+) (?P<path>\S+) -->.*?<!-- /snippet -->", re.S)
LANG = {".py": "python", ".yml": "yaml", ".yaml": "yaml", ".sh": "bash", ".jsonl": "json"}


def region(path, name):
    text = (ROOT / path).read_text()
    m = re.search(rf"^[ \t]*# \[snippet:{re.escape(name)}\][^\n]*\n(.*?)^[ \t]*# \[/snippet\]", text, re.S | re.M)
    if not m:
        raise SystemExit(f"{path}: no snippet region named {name!r}")
    return textwrap.dedent(m.group(1)).strip("\n")


def widget(name, path):
    code = html.escape(region(path, name), quote=False)
    lang = LANG.get(Path(path).suffix, "plaintext")
    return (f'<!-- snippet:{name} {path} -->\n'
            f'<div class="code"><div class="code-head"><a href="{REPO}{path}">{path}</a>'
            f'<button class="copy" type="button">Copy</button></div>'
            f'<pre><code class="language-{lang}">{code}</code></pre></div>\n'
            f'<!-- /snippet -->')


def render(page):
    return PLACE.sub(lambda m: widget(m["name"], m["path"]), page)


def main(argv):
    page = PAGE.read_text()
    new = render(page)
    if "--check" in argv:
        if new != page:
            print("site/index.html is out of date: run python site/snippets.py", file=sys.stderr)
            return 1
        return 0
    PAGE.write_text(new)
    print(f"{len(PLACE.findall(new))} snippets written to {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
