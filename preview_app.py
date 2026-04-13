#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

from flask import Flask, abort, render_template_string, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
AUTHORS_DIR = DOCS_DIR / "authors"
REPORTS_DIR = BASE_DIR / "pipeline_reports"

app = Flask(__name__)

PAGE_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VALIANT Wrapped Preview</title>
  <style>
    :root{
      --bg:#0a0f14; --bg2:#101820; --card:#121b23; --card2:#17232d;
      --ink:#ecf3f8; --muted:#99a9b8; --line:#233341;
      --green:#1ed760; --green2:#18b34e; --blue:#5cc8ff;
      --shadow:0 16px 40px rgba(0,0,0,.28); --radius:20px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(900px 420px at 10% 0%, rgba(30,215,96,.14), transparent 60%),
        radial-gradient(900px 420px at 100% 10%, rgba(92,200,255,.10), transparent 60%),
        linear-gradient(180deg,var(--bg),var(--bg2));
      color:var(--ink); min-height:100vh;
    }
    .wrap{max-width:1200px; margin:0 auto; padding:28px 20px 56px;}
    .hero{
      border:1px solid rgba(255,255,255,.08);
      background:linear-gradient(135deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
      border-radius:28px; padding:28px; box-shadow:var(--shadow);
    }
    .eyebrow{font-size:12px; letter-spacing:.14em; text-transform:uppercase; font-weight:800; color:#c6d2dd;}
    h1{margin:10px 0 10px; font-size:42px; line-height:1.02;}
    .lead{margin:0; color:#d4dde6; max-width:74ch; line-height:1.6;}
    .toolbar{margin-top:20px; display:grid; grid-template-columns:1fr auto auto; gap:12px; align-items:center;}
    input[type="search"]{
      width:100%; border:1px solid var(--line); background:#0d151c; color:var(--ink);
      border-radius:14px; padding:14px 16px; font-size:15px; outline:none;
    }
    .btn{
      display:inline-flex; align-items:center; justify-content:center; gap:8px;
      text-decoration:none; border:none; border-radius:14px; padding:13px 16px;
      font-size:14px; font-weight:800; cursor:pointer; transition:transform .12s ease, opacity .12s ease;
    }
    .btn:hover{transform:translateY(-1px)}
    .btn.green{background:linear-gradient(135deg,var(--green),var(--green2)); color:#04130a;}
    .btn.ghost{background:#13202a; color:#d8e3ec; border:1px solid var(--line);}
    .meta{margin-top:18px; display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); font-size:13px;}
    .pill{display:inline-flex; align-items:center; gap:8px; padding:8px 10px; border:1px solid var(--line); background:#101920; border-radius:999px;}
    .grid{margin-top:24px; display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px;}
    .card{
      background:linear-gradient(180deg, var(--card), var(--card2));
      border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:var(--shadow);
    }
    .name{font-size:19px; font-weight:900; margin:0 0 8px; color:#f4f8fb;}
    .sub{color:var(--muted); font-size:13px; line-height:1.5; min-height:40px;}
    .actions{margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;}
    .small{font-size:12px; color:var(--muted); margin-top:10px; word-break:break-all;}
    .empty{
      margin-top:24px; padding:24px; border:1px dashed var(--line); border-radius:20px;
      background:#0f171e; color:var(--muted);
    }
    .footer{margin-top:28px; color:var(--muted); font-size:13px;}
    @media (max-width:980px){.grid{grid-template-columns:1fr 1fr}.toolbar{grid-template-columns:1fr}}
    @media (max-width:640px){.grid{grid-template-columns:1fr}h1{font-size:34px}}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">VALIANT Wrapped Preview</div>
      <h1>Browse generated author pages</h1>
      <p class="lead">This preview app reads the HTML pages already generated in <code>docs/authors</code>. It does not rerun the pipeline.</p>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="Search by author name, label, or Scopus ID..." />
        {% if report_url %}<a class="btn ghost" href="{{ report_url }}">Open pipeline report</a>{% endif %}
        <a class="btn green" href="/refresh">Refresh list</a>
      </div>
      <div class="meta">
        <div class="pill">Authors found: {{ authors|length }}</div>
        <div class="pill">Docs dir: {{ docs_dir }}</div>
        <div class="pill">Authors dir: {{ authors_dir }}</div>
      </div>
    </section>

    {% if authors %}
      <section class="grid" id="authorGrid">
        {% for a in authors %}
          <article class="card author-card" data-search="{{ (a.display_name ~ ' ' ~ a.author_label ~ ' ' ~ a.scopus_id)|lower }}">
            <h2 class="name">{{ a.display_name }}</h2>
            <div class="sub">
              Label: {{ a.author_label }}<br>
              {% if a.scopus_id %}Scopus ID: {{ a.scopus_id }}{% endif %}
            </div>
            <div class="actions">
              <a class="btn green" href="{{ a.page_url }}" target="_blank" rel="noopener">Open page</a>
              <a class="btn ghost" href="{{ a.page_url }}">Open here</a>
            </div>
            <div class="small">{{ a.filename }}</div>
          </article>
        {% endfor %}
      </section>
    {% else %}
      <div class="empty">
        No author HTML pages were found in <code>{{ authors_dir }}</code>.
        Make sure your site generator has written files to <code>docs/authors/</code>.
      </div>
    {% endif %}

    <div class="footer">
      Run this app on the same machine where the <code>docs/</code> folder exists, then port-forward port 5001 to your browser.
    </div>
  </div>

  <script>
    const searchBox = document.getElementById('searchBox');
    const cards = Array.from(document.querySelectorAll('.author-card'));
    function applyFilter() {
      const q = (searchBox.value || '').trim().toLowerCase();
      for (const card of cards) {
        const hay = card.getAttribute('data-search') || '';
        card.style.display = (!q || hay.includes(q)) ? '' : 'none';
      }
    }
    if (searchBox) searchBox.addEventListener('input', applyFilter);
  </script>
</body>
</html>
'''

def display_name_from_label(label: str):
    parts = str(label).split("_")
    last = parts[0] if len(parts) > 0 else ""
    first = parts[1] if len(parts) > 1 else ""
    scopus_id = parts[-1] if len(parts) > 2 else ""
    pretty_first = first.replace("-", " ")
    pretty_last = last.replace("-", " ")
    display_name = f"{pretty_first} {pretty_last}".strip() or label
    return pretty_first, pretty_last, scopus_id, display_name

def find_report_url() -> Optional[str]:
    for path in [REPORTS_DIR / "pipeline_author_report.csv", REPORTS_DIR / "pipeline_summary.txt"]:
        if path.exists():
            return f"/reports/{path.name}"
    return None

def collect_authors():
    rows = []
    if not AUTHORS_DIR.exists():
        return rows
    for html_path in sorted(AUTHORS_DIR.glob("*.html")):
        author_label = html_path.stem
        _, _, scopus_id, display_name = display_name_from_label(author_label)
        rows.append({
            "author_label": author_label,
            "display_name": display_name,
            "scopus_id": scopus_id,
            "filename": html_path.name,
            "page_url": f"/authors/{html_path.name}",
        })
    return rows

@app.route("/")
def index():
    return render_template_string(
        PAGE_HTML,
        authors=collect_authors(),
        docs_dir=str(DOCS_DIR),
        authors_dir=str(AUTHORS_DIR),
        report_url=find_report_url(),
    )

@app.route("/refresh")
def refresh():
    return index()

@app.route("/authors/<path:filename>")
def serve_author_page(filename: str):
    path = AUTHORS_DIR / filename
    if not path.exists():
        abort(404)
    return send_from_directory(AUTHORS_DIR, filename)

@app.route("/reports/<path:filename>")
def serve_report(filename: str):
    path = REPORTS_DIR / filename
    if not path.exists():
        abort(404)
    return send_from_directory(REPORTS_DIR, filename)

@app.route("/assets/<path:subpath>")
def serve_assets(subpath: str):
    assets_dir = DOCS_DIR / "assets"
    target = assets_dir / subpath
    if not target.exists():
        abort(404)
    return send_from_directory(assets_dir, subpath)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
