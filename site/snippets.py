"""Copy code regions from examples/ into site/index.html, so the page shows the code that runs.

A region in a source file is marked with comments:

    # [snippet:evals-rubric]
    ...
    # [/snippet]

and a place in the page with:

    <!-- snippet:evals-rubric examples/worked/evals/rubric.py -->...<!-- /snippet -->

It also rebuilds the table of contents between <!-- toc --> and <!-- /toc --> from every
h2-h4 heading with an id, so a new section can't be left out of it.

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


TOC = re.compile(r"<!-- toc -->.*?<!-- /toc -->", re.S)
HEADING = re.compile(r'<h([234])\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</h\1>', re.S)


def heading_text(inner):
    inner = re.sub(r'<span class="part-k">.*?</span>', "", inner)      # the Why / What / How kicker
    inner = re.sub(r'<span class="tag[^"]*">(.*?)</span>', r"(\1)", inner)
    return html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()


def toc(page):
    """Parts (h2 with a kicker) become columns; h3 and h4 nest under them. Other h2s go in the footer."""
    body = TOC.sub("", page)
    parts, extra = [], []
    for m in HEADING.finditer(body):
        level, hid, text = int(m[1]), m[2], heading_text(m[3])
        link = f'<a href="#{hid}">{html.escape(text, quote=False)}</a>'
        if level == 2:
            if 'class="part-h"' in m[0]:
                kicker = re.search(r'<span class="part-k">(.*?)</span>', m[3])[1]
                parts.append({"head": f'<a href="#{hid}">{kicker}</a>', "title": link, "items": []})
            else:
                extra.append(link)
        elif parts and level == 3:
            parts[-1]["items"].append([link, []])
        elif parts and parts[-1]["items"]:
            parts[-1]["items"][-1][1].append(link)
    cols = []
    for p in parts:
        lis = []
        for link, subs in p["items"]:
            sub = ("\n        <ol>" + "".join(f"<li>{s}</li>" for s in subs) + "</ol>") if subs else ""
            lis.append(f"      <li>{link}{sub}</li>")
        cols.append(f'    <div>\n      <p>{p["head"]}</p>\n      <p class="toc-part">{p["title"]}</p>\n'
                    f'      <ol>\n' + "\n".join(lis) + "\n      </ol>\n    </div>")
    foot = " · ".join(extra)
    return ('<!-- toc -->\n<nav class="toc" id="contents" aria-label="Contents">\n'
            '  <p class="toc-title">Contents</p>\n  <div class="toc-grid">\n' + "\n".join(cols) +
            f'\n  </div>\n  <p class="toc-foot">Also: {foot}</p>\n</nav>\n<!-- /toc -->')


def render(page):
    page = PLACE.sub(lambda m: widget(m["name"], m["path"]), page)
    return TOC.sub(lambda m: toc(page), page)


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
