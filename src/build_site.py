"""
Build the GitHub Pages site in docs/ from the artifact sources in web/.

The pages in web/ are written as artifact bodies - no <html>, <head> or <body>,
because the artifact host supplies those. For Pages they need a real document
wrapper, plus a nav bar tying the two demos together into one site.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DOCS = ROOT / "docs"
REPO = "https://github.com/realgauravvyas/flys-hash-function"

PAGES = [
    {
        "src": "index.html", "out": "index.html", "nav": "The Fly's Hash Function",
        "title": "The Fly's Hash Function",
        "desc": "The real fruit-fly mushroom body connectome running live as a "
                "locality-sensitive hash, benchmarked against the random matrix "
                "the literature has used in its place since 2017.",
        "assets": ["connectome.js"],
    },
    {
        "src": "sudoku.html", "out": "sudoku.html", "nav": "Can a Fly Solve Sudoku?",
        "title": "Can a Fly Solve Sudoku?",
        "desc": "The measured fruit-fly mushroom body deducing sudoku squares "
                "live - and stalling, because it has only one learned layer.",
        "assets": ["connectome.js", "sudoku_model.js"],
    },
]

NAV_CSS = """
.sitenav {
  display: flex; flex-wrap: wrap; gap: 2px 18px; align-items: center;
  max-width: 1180px; margin: 0 auto; padding: 11px 20px;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
  border-bottom: 1px solid #232B35;
}
.sitenav a { color: #6B7887; text-decoration: none; }
.sitenav a:hover { color: #DDE4EB; }
.sitenav a[aria-current="page"] { color: #3DE8A0; }
.sitenav .sep { color: #232B35; }
.sitenav .repo { margin-left: auto; }
"""


def nav_html(current):
    parts = []
    for p in PAGES:
        cur = ' aria-current="page"' if p["out"] == current else ""
        parts.append(f'<a href="{p["out"]}"{cur}>{p["nav"]}</a>')
    links = '<span class="sep">/</span>'.join(parts)
    return (f'<nav class="sitenav">{links}'
            f'<a class="repo" href="{REPO}">Source &amp; data &#8599;</a></nav>')


def wrap(page):
    body = (WEB / page["src"]).read_text(encoding="utf-8")

    # the <title> belongs in the head; the artifact body carries its own copy
    title_tag = f'<title>{page["title"]}</title>'
    body = body.replace(title_tag + "\n", "", 1)

    # inside the site the companion piece is a sibling page, not an artifact
    body = body.replace(
        "https://claude.ai/code/artifact/b4d0f375-0ac3-4f1d-b46b-60bea7d6e38b",
        "sudoku.html")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{title_tag}
<meta name="description" content="{page["desc"]}">
<meta property="og:title" content="{page["title"]}">
<meta property="og:description" content="{page["desc"]}">
<meta name="twitter:card" content="summary">
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #0A0C0F; }}
  img {{ max-width: 100%; }}
  [hidden] {{ display: none !important; }}
{NAV_CSS}</style>
</head>
<body>
{nav_html(page["out"])}
{body}
</body>
</html>
"""


def main():
    DOCS.mkdir(exist_ok=True)
    assets = set()
    for page in PAGES:
        out = DOCS / page["out"]
        html = wrap(page)
        assert f'<title>{page["title"]}</title>' in html.split("</head>")[0], page["out"]
        assert "claude.ai/code/artifact" not in html, page["out"]
        out.write_text(html, encoding="utf-8")
        print(f"  {out.name}  {out.stat().st_size / 1024:.0f} KB")
        assets.update(page["assets"])

    for a in sorted(assets):
        shutil.copy(WEB / a, DOCS / a)
        print(f"  {a}  {(DOCS / a).stat().st_size / 1024:.0f} KB")

    (DOCS / ".nojekyll").write_text("")
    print(f"\nsite built in {DOCS}")


if __name__ == "__main__":
    main()
