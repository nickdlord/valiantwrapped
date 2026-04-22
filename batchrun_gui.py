#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

from flask import Flask, abort, redirect, render_template_string, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
AUTHORS_DIR = DOCS_DIR / "authors"
REPORTS_DIR = BASE_DIR / "pipeline_reports"

app = Flask(__name__)

PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VALIANT Wrapped Preview</title>
  <style>
    :root{
      --bg:#0a0a0a; --bg2:#101010; --card:#121212; --card2:#181818;
      --ink:#f5f5f5; --muted:#b3b3b3; --line:#2a2a2a;
      --green:#1db954; --green2:#1ed760;
      --shadow:0 16px 40px rgba(0,0,0,.28); --radius:20px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(900px 420px at 10% 0%, rgba(29,185,84,.14), transparent 60%),
        radial-gradient(900px 420px at 100% 10%, rgba(29,185,84,.08), transparent 60%),
        linear-gradient(180deg,var(--bg),var(--bg2));
      color:var(--ink); min-height:100vh;
    }
    .wrap{max-width:1200px; margin:0 auto; padding:28px 20px 56px;}
    .hero{
      border:1px solid rgba(255,255,255,.08);
      background:linear-gradient(135deg, rgba(29,185,84,.14), rgba(255,255,255,.02));
      border-radius:28px; padding:28px; box-shadow:var(--shadow);
    }
    .eyebrow{font-size:12px; letter-spacing:.14em; text-transform:uppercase; font-weight:800; color:#c6d2dd;}
    h1{margin:10px 0 10px; font-size:42px; line-height:1.02;}
    .lead{margin:0; color:#d4dde6; max-width:74ch; line-height:1.6;}
    .toolbar{margin-top:20px; display:grid; grid-template-columns:1fr auto auto; gap:12px; align-items:center;}
    input[type="search"]{
      width:100%; border:1px solid var(--line); background:#0d0d0d; color:var(--ink);
      border-radius:14px; padding:14px 16px; font-size:15px; outline:none;
    }
    .btn{
      display:inline-flex; align-items:center; justify-content:center; gap:8px;
      text-decoration:none; border:none; border-radius:14px; padding:13px 16px;
      font-size:14px; font-weight:800; cursor:pointer; transition:transform .12s ease, opacity .12s ease;
    }
    .btn:hover{transform:translateY(-1px)}
    .btn.green{background:linear-gradient(135deg,var(--green),var(--green2)); color:#04130a;}
    .btn.ghost{background:#161616; color:#d8e3ec; border:1px solid var(--line);}
    .meta{margin-top:18px; display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); font-size:13px;}
    .pill{display:inline-flex; align-items:center; gap:8px; padding:8px 10px; border:1px solid var(--line); background:#101010; border-radius:999px;}
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
      background:#0f0f0f; color:var(--muted);
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
      <p class="lead">This preview app reads the HTML pages already generated in <code>docs/authors/&lt;slug&gt;/index.html</code>. It does not rerun the pipeline.</p>
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
            <div class="small">{{ a.slug }}/index.html</div>
          </article>
        {% endfor %}
      </section>
    {% else %}
      <div class="empty">
        No author HTML pages were found in <code>{{ authors_dir }}</code>.
        Make sure your site generator has written files to <code>docs/authors/&lt;slug&gt;/index.html</code>.
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
"""

def display_name_from_label(label: str):
    parts = str(label).split("_")
    scopus_id = parts[-1] if len(parts) >= 3 and parts[-1].isdigit() else ""
    name_parts = parts[:-1] if scopus_id else parts
    if len(name_parts) >= 2:
        last = name_parts[0].replace("-", " ")
        first = " ".join(p.replace("-", " ") for p in name_parts[1:])
        display_name = f"{first} {last}".strip()
    else:
        display_name = label.replace("_", " ").replace("-", " ")
    return scopus_id, display_name

def find_report_url() -> Optional[str]:
    for path in [REPORTS_DIR / "pipeline_author_report.csv", REPORTS_DIR / "pipeline_summary.txt"]:
        if path.exists():
            return f"/reports/{path.name}"
    return None

def load_authors_json_map():
    path = DOCS_DIR / "data" / "authors.json"
    if not path.exists():
        return {}
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for row in data:
            label = str(row.get("author_label", "")).strip()
            if label:
                out[label] = row
        return out
    except Exception:
        return {}

def collect_authors():
    rows = []
    if not AUTHORS_DIR.exists():
        return rows

    authors_json_map = load_authors_json_map()

    for author_dir in sorted([p for p in AUTHORS_DIR.iterdir() if p.is_dir()]):
        index_path = author_dir / "index.html"
        if not index_path.exists():
            continue

        label = ""
        meta = None

        # Prefer matching from authors.json using href slug if possible
        for candidate_label, candidate in authors_json_map.items():
            href = str(candidate.get("href", "")).strip("/")
            if href.endswith(author_dir.name):
                label = candidate_label
                meta = candidate
                break

        if not label:
            # Fallback: try to infer from title or directory name
            label = author_dir.name
            meta = {}

        scopus_id = ""
        display_name = ""

        if meta:
            scopus_id = str(meta.get("scopus_id", "")).strip()
            display_name = str(meta.get("display_name", "")).strip()

        if not display_name:
            inferred_id, inferred_name = display_name_from_label(label)
            display_name = inferred_name
            scopus_id = scopus_id or inferred_id

        rows.append({
            "author_label": label,
            "display_name": display_name,
            "scopus_id": scopus_id,
            "slug": author_dir.name,
            "page_url": f"/authors/{author_dir.name}/",
        })

    rows.sort(key=lambda x: x["display_name"].lower())
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

@app.route("/authors/<slug>/")
def serve_author_page(slug: str):
    author_dir = AUTHORS_DIR / slug
    index_path = author_dir / "index.html"
    if not index_path.exists():
        abort(404)
    return send_from_directory(author_dir, "index.html")

@app.route("/authors/<slug>/index.html")
def serve_author_page_index(slug: str):
    return serve_author_page(slug)

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

@app.route("/index.html")
def serve_site_home():
    home = DOCS_DIR / "index.html"
    if not home.exists():
        abort(404)
    return send_from_directory(DOCS_DIR, "index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
