#!/usr/bin/env python3
"""
generate_valiantwrapped_site_withindex_compatible.py

Pipeline-tolerant VALIANT Wrapped site generator with a browse homepage.

What this version keeps from the original with-index script:
- Spotify-ish dark browse homepage / index
- Shared styling between browse page and author pages
- Safe album cover copying

What this version changes for compatibility with the current pipeline:
- Auto-detects common pipeline input locations instead of hard-coding one test file
- Accepts CLI overrides for all key inputs/outputs
- Supports optional expertise summaries from TXT files
- Supports optional recommendations if a recommendations CSV is present
- Generates both pretty folder-style author URLs and legacy .html redirects
- Tolerates a wider range of author key columns
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SUMMARY_CANDIDATES = [
    BASE_DIR / "author_summary_2025_present.csv",
    BASE_DIR / "author_summary_2025_present_test.csv",
    BASE_DIR / "outputs" / "author_summary_2025_present.csv",
]
DEFAULT_PERSONA_CANDIDATES = [
    BASE_DIR / "outputs" / "author_music_personas.csv",
    BASE_DIR / "author_music_personas.csv",
]
DEFAULT_EXPERTISE_DIR_CANDIDATES = [
    BASE_DIR / "author_expertise_txt",
    BASE_DIR / "outputs" / "author_expertise_txt",
]
DEFAULT_RECOMMENDATION_CANDIDATES = [
    BASE_DIR / "outputs" / "author_recommendations.csv",
    BASE_DIR / "outputs" / "paper_recommendations.csv",
    BASE_DIR / "author_recommendations.csv",
    BASE_DIR / "paper_recommendations.csv",
]
DEFAULT_COVER_DIR_CANDIDATES = [
    BASE_DIR / "outputs" / "album_covers",
    BASE_DIR / "album_covers",
]

DOCS_DIR = BASE_DIR / "docs"
AUTHOR_DIR = DOCS_DIR / "authors"
ASSETS_DIR = DOCS_DIR / "assets"
COVERS_DIR = ASSETS_DIR / "album_covers"
DATA_DIR = DOCS_DIR / "data"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def canonical_author_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    s = os.path.basename(s)
    for suffix in [".csv", ".txt", ".html"]:
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def prettify_label_token(token: str) -> str:
    token = token.replace("-", "-")
    token = token.replace("_", " ")
    token = re.sub(r"\s+", " ", token).strip()
    return token


def display_name_from_label(label: str) -> str:
    parts = [p for p in label.split("_") if p]
    if len(parts) >= 3 and parts[-1].isdigit():
        core = parts[:-1]
    else:
        core = parts
    if not core:
        return label
    return " ".join(prettify_label_token(p) for p in core)


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def safe_split_themes(x: object) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(t).strip() for t in arr if str(t).strip()]
        except Exception:
            pass
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("www."):
        return f"https://{url}"
    return url


def format_tracklist(tracklist: object) -> str:
    if tracklist is None or (isinstance(tracklist, float) and pd.isna(tracklist)):
        return ""
    raw = str(tracklist).strip()
    if not raw:
        return ""
    if "\n" in raw:
        tracks = [t.strip() for t in raw.splitlines() if t.strip()]
    else:
        for sep in [";", "|", ","]:
            if sep in raw:
                tracks = [t.strip() for t in raw.split(sep) if t.strip()]
                break
        else:
            tracks = [raw]

    cleaned = []
    for t in tracks:
        t = re.sub(r"^\s*(track\s*\d+)\s*[-:]\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*\d+\.\s*", "", t)
        cleaned.append(t.strip())

    items = []
    for i, t in enumerate(cleaned, start=1):
        safe = html_lib.escape(t)
        items.append(
            f"""
            <div class="track-row2">
              <div class="track-num2">{i:02d}</div>
              <div class="track-title2">{safe}</div>
            </div>
            """
        )
    return f'<div class="tracklist2">\n{"".join(items)}\n</div>'


def load_expertise_map(expertise_dir: Path | None) -> dict[str, str]:
    if expertise_dir is None or not expertise_dir.exists():
        return {}
    out: dict[str, str] = {}
    for path in sorted(expertise_dir.glob("*.txt")):
        out[canonical_author_label(path.name)] = path.read_text(encoding="utf-8", errors="replace").strip()
    return out


def select_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def build_author_label_column(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "author_label",
        "author_file",
        "author_id",
        "author",
        "label",
        "file",
    ]
    col = select_column(df, candidates)
    if col is None:
        raise ValueError(
            "Could not determine author key column. Expected one of: "
            + ", ".join(candidates)
        )
    return df[col].apply(canonical_author_label)


def rebuild_docs_folder(output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    docs_dir = output_dir
    author_dir = docs_dir / "authors"
    assets_dir = docs_dir / "assets"
    covers_dir = assets_dir / "album_covers"
    data_dir = docs_dir / "data"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    author_dir.mkdir(parents=True, exist_ok=True)
    covers_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir, author_dir, assets_dir, covers_dir, data_dir


BROWSE_CSS = r"""
:root{
  --bg:#0b0b0f;
  --panel:#12121a;
  --panel2:#171722;
  --text:#f2f2f2;
  --muted:#b7b7c7;
  --accent:#24d06f;
  --stroke:#2a2a3a;
  --shadow: 0 10px 30px rgba(0,0,0,.35);
  --radius: 18px;
  --radius2: 24px;
  --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;
  background:linear-gradient(180deg, #08080c 0%, var(--bg) 60%);
  color:var(--text);
  font-family:var(--font);
}
.app{display:grid;grid-template-columns:280px 1fr;min-height:100vh;}
.sidebar{padding:18px;background:linear-gradient(180deg, #0c0c12 0%, #07070b 100%);border-right:1px solid var(--stroke);}
.brand{display:flex;gap:12px;align-items:center;padding:12px;border-radius:var(--radius);background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);}
.logo{width:44px;height:44px;border-radius:14px;background:radial-gradient(circle at 30% 30%, var(--accent) 0%, rgba(36,208,111,.0) 55%),linear-gradient(135deg, #2c2cff 0%, #15151f 55%, #ff2c7a 110%);box-shadow:var(--shadow);}
.brandTitle{font-weight:700;letter-spacing:.2px;}.brandSub{font-size:12px;color:var(--muted);margin-top:2px;}
.nav{margin-top:16px;display:flex;flex-direction:column;gap:8px;}
.navItem{text-decoration:none;color:var(--muted);padding:10px 12px;border-radius:14px;border:1px solid transparent;display:block;}
.navItem:hover{background:rgba(255,255,255,0.04);color:var(--text);} .navItem.active{background:rgba(36,208,111,0.10);border-color:rgba(36,208,111,0.25);color:var(--text);}
.sidebarFooter{margin-top:18px;padding:12px;color:var(--muted);font-size:12px;}
.main{padding:18px;}
.topbar{display:flex;gap:12px;align-items:center;position:sticky;top:0;padding:12px 0 16px 0;background:linear-gradient(180deg, rgba(11,11,15,1) 0%, rgba(11,11,15,.75) 70%, rgba(11,11,15,0) 100%);backdrop-filter: blur(10px);z-index:5;}
.search,.filter{border-radius:999px;padding:12px 14px;border:1px solid var(--stroke);background:rgba(255,255,255,0.04);color:var(--text);outline:none;}
.search{flex:1}.search::placeholder{color:rgba(242,242,242,.45)}
.section{margin-top:12px;}.sectionTitle{font-size:22px;font-weight:800;margin:10px 0 14px 0;}
.grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(220px, 1fr));gap:14px;}
.card{background:linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);border:1px solid rgba(255,255,255,0.06);border-radius:var(--radius2);padding:14px;cursor:pointer;transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;text-decoration:none;color:inherit;}
.card:hover{transform:translateY(-2px);box-shadow:var(--shadow);border-color:rgba(255,255,255,0.14);}
.cover{width:100%;aspect-ratio:1/1;border-radius:18px;background:linear-gradient(135deg, rgba(36,208,111,.35), rgba(44,44,255,.25), rgba(255,44,122,.22));border:1px solid rgba(255,255,255,0.08);overflow:hidden;display:flex;align-items:center;justify-content:center;}
.cover img{width:100%;height:100%;object-fit:cover;display:block;}
.meta{margin-top:12px}.name{font-weight:800;font-size:16px;line-height:1.1}.sub{color:var(--muted);font-size:12px;margin-top:6px;line-height:1.3}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;}
.tag{font-size:11px;color:rgba(242,242,242,.85);padding:6px 10px;border-radius:999px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);}
.empty{color:var(--muted);padding:20px 4px;}.hidden{display:none;}
.authorHeader{padding:18px;border-radius:var(--radius2);border:1px solid rgba(255,255,255,0.06);background:linear-gradient(135deg, rgba(36,208,111,0.10), rgba(44,44,255,0.06), rgba(255,44,122,0.06));box-shadow:var(--shadow);}
.authorKicker{color:var(--muted);font-weight:800;letter-spacing:.08em;text-transform:uppercase;font-size:12px;}
.authorName{font-size:40px;font-weight:900;margin:10px 0 6px;line-height:1.05;}
.authorSub{color:var(--muted);margin:0;max-width:90ch;}
.statsGrid{display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:14px;}
.statCard,.contentCard{background:linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);border:1px solid rgba(255,255,255,0.06);border-radius:var(--radius2);padding:14px;}
.statLabel{color:var(--muted);font-size:12px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;}
.statValue{margin-top:10px;font-size:28px;font-weight:900;}.statSmall{margin-top:10px;color:var(--text);font-weight:700;line-height:1.35;}
.pill2{display:inline-block;margin-top:10px;padding:7px 10px;border-radius:999px;border:1px solid rgba(36,208,111,0.35);background:rgba(36,208,111,0.12);color:var(--text);font-weight:900;font-size:12px;}
.album-card2{border-radius:var(--radius2);padding:16px;border:1px solid rgba(255,255,255,0.06);background:linear-gradient(135deg, rgba(36,208,111,0.10), rgba(44,44,255,0.06), rgba(255,44,122,0.06));}
.artist2{font-size:22px;font-weight:900;margin:0 0 10px;} .bio2{margin:0 0 14px;color:var(--muted);line-height:1.55;white-space:normal;}
.album-title2{display:flex;align-items:center;gap:10px;font-weight:900;margin:0 0 12px;}.album-title2 span{color:var(--text);opacity:.92;}
.cover-wrap2{display:flex;justify-content:center;margin:14px 0;}
.album-cover2{width:340px;max-width:100%;height:auto;border-radius:18px;border:1px solid rgba(255,255,255,0.10);box-shadow:var(--shadow);background:rgba(255,255,255,0.06);}
.cover-placeholder2{width:340px;max-width:100%;border-radius:18px;border:1px dashed rgba(255,255,255,0.20);background:rgba(255,255,255,0.03);color:var(--muted);padding:18px;text-align:center;font-weight:900;}
.tracklist2{border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);background:rgba(0,0,0,0.25);}
.track-row2{display:flex;gap:14px;padding:12px 14px;align-items:center;border-bottom:1px solid rgba(255,255,255,0.08);} .track-row2:last-child{border-bottom:none;}
.track-num2{width:44px;color:rgba(242,242,242,.55);font-weight:900;font-variant-numeric:tabular-nums;} .track-title2{font-weight:700;}
.bodyText{color:var(--muted);line-height:1.65;} .rec-list{display:grid;gap:10px;} .rec-item{padding:12px 14px;border:1px solid rgba(255,255,255,0.08);border-radius:16px;background:rgba(0,0,0,0.20);} .rec-title{font-weight:800;} .rec-meta{color:var(--muted);font-size:12px;margin-top:6px;} .rec-link{display:inline-block;margin-top:8px;color:var(--text);font-weight:700;text-decoration:none;border-bottom:1px solid rgba(36,208,111,0.35);}
@media (max-width: 860px){.app{grid-template-columns:1fr;}.sidebar{position:sticky;top:0;z-index:10;}}
"""

BROWSE_JS = r"""
async function loadAuthors() {
  const res = await fetch("data/authors.json");
  if (!res.ok) throw new Error("Failed to load data/authors.json");
  return await res.json();
}
function uniq(arr) { return Array.from(new Set(arr)).sort((a,b) => a.localeCompare(b)); }
function norm(s) { return (s || "").toString().toLowerCase(); }
function matches(author, q, theme) {
  if (theme && !(author.themes || []).includes(theme)) return false;
  if (!q) return true;
  const hay = [author.display_name, author.author_label, author.artist_name, author.album_title, author.expertise_preview, (author.themes || []).join(" "), author.top_journal, author.top_paper].join(" ");
  return norm(hay).includes(norm(q));
}
function escapeHtml(str) {
  return (str || "").toString()
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}
function cardHTML(a) {
  const tags = (a.themes || []).slice(0, 3).map(t => `<span class="tag">${escapeHtml(t)}</span>`).join("");
  const sub = [a.artist_name ? `Artist: ${escapeHtml(a.artist_name)}` : "", a.album_title ? `Album: ${escapeHtml(a.album_title)}` : ""].filter(Boolean).join(" • ");
  const cover = a.cover_url ? `<img src="${a.cover_url}" alt="" loading="lazy" />` : "";
  return `<a class="card" href="${a.profile_url}"><div class="cover">${cover}</div><div class="meta"><div class="name">${escapeHtml(a.display_name || a.author_label)}</div><div class="sub">${sub || "&nbsp;"}</div><div class="tags">${tags}</div></div></a>`;
}
function render(authors, q, theme) {
  const grid = document.getElementById("grid");
  const empty = document.getElementById("empty");
  const filtered = authors.filter(a => matches(a, q, theme));
  grid.innerHTML = filtered.map(cardHTML).join("");
  empty.classList.toggle("hidden", filtered.length !== 0);
}
function fillThemeFilter(authors) {
  const sel = document.getElementById("themeFilter");
  const allThemes = uniq(authors.flatMap(a => a.themes || []).filter(Boolean));
  if (allThemes.length === 0) { sel.classList.add("hidden"); return; }
  for (const t of allThemes) {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t; sel.appendChild(opt);
  }
}
(async function init() {
  const authors = await loadAuthors();
  fillThemeFilter(authors);
  const search = document.getElementById("search");
  const themeFilter = document.getElementById("themeFilter");
  const rerender = () => render(authors, search.value, themeFilter.value);
  search.addEventListener("input", rerender);
  themeFilter.addEventListener("change", rerender);
  rerender();
})();
"""


def write_browse_assets(assets_dir: Path) -> None:
    (assets_dir / "styles.css").write_text(BROWSE_CSS.strip() + "\n", encoding="utf-8")
    (assets_dir / "app.js").write_text(BROWSE_JS.strip() + "\n", encoding="utf-8")


def write_browse_index(docs_dir: Path, generated_iso: str, title: str = "VALIANT Wrapped — Browse") -> None:
    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{html_lib.escape(title)}</title>
  <link rel="stylesheet" href="assets/styles.css" />
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="logo"></div>
        <div class="brandText">
          <div class="brandTitle">VALIANT Wrapped</div>
          <div class="brandSub">Browse profiles</div>
        </div>
      </div>
      <nav class="nav">
        <a class="navItem active" href="index.html">Home</a>
        <a class="navItem" href="#" onclick="return false;">Browse</a>
      </nav>
      <div class="sidebarFooter">
        Generated: {html_lib.escape(generated_iso)}<br>
        Published folder: {html_lib.escape(str(docs_dir))}
      </div>
    </aside>
    <main class="main">
      <header class="topbar">
        <input id="search" class="search" placeholder="Search authors, themes, albums…" />
        <select id="themeFilter" class="filter"><option value="">All themes</option></select>
      </header>
      <section class="section">
        <div class="sectionTitle">Authors</div>
        <div id="grid" class="grid"></div>
        <div id="empty" class="empty hidden">No matches. Try a different search.</div>
      </section>
    </main>
  </div>
  <script src="assets/app.js"></script>
</body>
</html>
"""
    (docs_dir / "index.html").write_text(index_html, encoding="utf-8")


def find_album_cover_source(author_label: str, album_covers_src_dir: Path) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = album_covers_src_dir / f"{author_label}{ext}"
        if path.exists():
            return path
    return None


def ensure_album_cover_in_docs(author_label: str, album_covers_src_dir: Path, covers_dir: Path, build_report: list[tuple[str, str, str]]) -> tuple[bool, str, str]:
    src = find_album_cover_source(author_label, album_covers_src_dir)
    if src is None:
        build_report.append((author_label, "missing_album_cover", f"no file in {album_covers_src_dir}"))
        return False, "", ""
    dst = covers_dir / src.name
    try:
        if (not dst.exists()) or (dst.stat().st_mtime < src.stat().st_mtime):
            shutil.copy2(src, dst)
    except Exception as exc:
        build_report.append((author_label, "album_cover_copy_failed", repr(exc)))
        return False, "", ""
    rel_from_author = f"../assets/album_covers/{html_lib.escape(dst.name)}"
    rel_from_root = f"assets/album_covers/{html_lib.escape(dst.name)}"
    return True, rel_from_author, rel_from_root


def album_cover_block(author_label: str, artist_name_raw: str, album_title_raw: str, album_covers_src_dir: Path, covers_dir: Path, build_report: list[tuple[str, str, str]]) -> str:
    ok, rel_from_author, _ = ensure_album_cover_in_docs(author_label, album_covers_src_dir, covers_dir, build_report)
    if not ok:
        return """<div class="cover-wrap2"><div class="cover-placeholder2">Album cover art not available yet.</div></div>"""
    alt = html_lib.escape(f"Album cover for {artist_name_raw} — {album_title_raw}".strip(" —"))
    return f"""<div class="cover-wrap2"><img class="album-cover2" src="{rel_from_author}" alt="{alt}" loading="lazy"></div>"""


def normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["author_label"] = build_author_label_column(out)
    colmap = {}
    for target, candidates in {
        "title": ["recommended_title", "title", "paper_title", "recommendation_title"],
        "authors": ["recommended_authors", "authors", "paper_authors"],
        "journal": ["journal", "source_title", "recommended_journal"],
        "year": ["year", "publication_year", "recommended_year"],
        "url": ["url", "google_url", "google_search_url", "link", "search_url"],
        "reason": ["reason", "why_recommended", "explanation"],
    }.items():
        colmap[target] = select_column(out, candidates)

    records = []
    for _, row in out.iterrows():
        author_label = row.get("author_label", "")
        if not author_label:
            continue
        records.append(
            {
                "author_label": author_label,
                "title": safe_text(row.get(colmap["title"], "")) if colmap["title"] else "",
                "authors": safe_text(row.get(colmap["authors"], "")) if colmap["authors"] else "",
                "journal": safe_text(row.get(colmap["journal"], "")) if colmap["journal"] else "",
                "year": safe_text(row.get(colmap["year"], "")) if colmap["year"] else "",
                "url": normalize_url(safe_text(row.get(colmap["url"], ""))) if colmap["url"] else "",
                "reason": safe_text(row.get(colmap["reason"], "")) if colmap["reason"] else "",
            }
        )
    return pd.DataFrame(records)


def recommendations_html(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    blocks = []
    for row in rows:
        title = html_lib.escape(row.get("title") or "Recommended paper")
        meta_bits = [row.get("authors", ""), row.get("journal", ""), row.get("year", "")]
        meta = " • ".join(html_lib.escape(bit) for bit in meta_bits if bit)
        reason = html_lib.escape(row.get("reason", ""))
        link = row.get("url", "")
        link_html = f'<a class="rec-link" href="{html_lib.escape(link)}" target="_blank" rel="noopener">Look it up on Google</a>' if link else ""
        reason_html = f'<div class="rec-meta">Why it fits: {reason}</div>' if reason else ""
        blocks.append(
            f"""
            <div class="rec-item">
              <div class="rec-title">{title}</div>
              <div class="rec-meta">{meta}</div>
              {reason_html}
              {link_html}
            </div>
            """
        )
    return f"""
    <section class="section">
      <div class="sectionTitle">Recommended Reading</div>
      <div class="contentCard">
        <div class="rec-list">{''.join(blocks)}</div>
      </div>
    </section>
    """


def expertise_html(expertise_text: str) -> str:
    if not expertise_text.strip():
        return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", expertise_text) if p.strip()]
    if not paras:
        paras = [expertise_text.strip()]
    body = "".join(f"<p class=\"bodyText\">{html_lib.escape(p).replace(chr(10), '<br>')}</p>" for p in paras)
    return f"""
    <section class="section">
      <div class="sectionTitle">Research Expertise</div>
      <div class="contentCard">{body}</div>
    </section>
    """


def persona_html(author_label: str, persona_row: pd.Series | None, album_covers_src_dir: Path, covers_dir: Path, build_report: list[tuple[str, str, str]]) -> str:
    if persona_row is None:
        build_report.append((author_label, "missing_persona_row", f"searched label={author_label}"))
        return "<div class='contentCard'><div class='empty'>No persona generated.</div></div>"

    status_val = str(persona_row.get("status", "")).strip()
    if status_val and status_val.lower() != "ok":
        build_report.append((author_label, "persona_status", status_val))

    artist_name_raw = safe_text(persona_row.get("artist_name", ""))
    album_title_raw = safe_text(persona_row.get("album_title", ""))
    persona_bio = html_lib.escape(safe_text(persona_row.get("persona_bio", ""))).replace("\n", "<br>")
    artist_name = html_lib.escape(artist_name_raw)
    album_title = html_lib.escape(album_title_raw)
    cover_block = album_cover_block(author_label, artist_name_raw, album_title_raw, album_covers_src_dir, covers_dir, build_report)
    tracklist_html = format_tracklist(persona_row.get("tracklist", ""))

    return f"""
    <div class="album-card2">
      <div class="artist2">{artist_name}</div>
      <p class="bio2">{persona_bio}</p>
      <div class="album-title2">Album: <span>{album_title}</span></div>
      {cover_block}
      {tracklist_html}
    </div>
    """


def write_author_pages(author_dir: Path, author_label: str, display_name: str, generated_iso: str, stats_html: str, expertise_section: str, persona_section: str, recommendations_section: str) -> None:
    title_safe = html_lib.escape(display_name)
    page_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_safe} • VALIANT Wrapped</title>
<link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand"><div class="logo"></div><div class="brandText"><div class="brandTitle">VALIANT Wrapped</div><div class="brandSub">Author profile</div></div></div>
      <nav class="nav">
        <a class="navItem" href="../index.html">Home</a>
        <a class="navItem active" href="./index.html">Profile</a>
        <a class="navItem" href="../index.html">Browse</a>
      </nav>
      <div class="sidebarFooter">Generated: {html_lib.escape(generated_iso)}</div>
    </aside>
    <main class="main">
      <header class="topbar">
        <div class="sectionTitle" style="margin:0;">Profile</div>
        <a class="navItem" style="margin-left:auto;" href="../index.html">← Back</a>
      </header>
      <section class="section">
        <div class="authorHeader">
          <div class="authorKicker">VALIANT Wrapped • 2025–Present</div>
          <div class="authorName">{title_safe}</div>
          <p class="authorSub">A year-in-review snapshot of publications, citations, expertise, and a fictional musical persona inspired by this author’s work.</p>
        </div>
      </section>
      {stats_html}
      {expertise_section}
      <section class="section"><div class="sectionTitle">Musical Persona</div>{persona_section}</section>
      {recommendations_section}
    </main>
  </div>
</body>
</html>
"""
    author_page_dir = author_dir / author_label
    author_page_dir.mkdir(parents=True, exist_ok=True)
    (author_page_dir / "index.html").write_text(page_html, encoding="utf-8")

    # Legacy redirect for old links expecting authors/<author>.html
    redirect_html = f"""<!doctype html><html><head><meta http-equiv="refresh" content="0; url=./{html_lib.escape(author_label)}/" /></head><body><a href="./{html_lib.escape(author_label)}/">Continue</a></body></html>"""
    (author_dir / f"{author_label}.html").write_text(redirect_html, encoding="utf-8")


def stats_section(row: pd.Series | None) -> str:
    if row is None:
        pub = cit = top_journal = top_paper = top_paper_cit = ""
    else:
        pub = safe_text(row.get("pub_count_2025_present", ""))
        cit = safe_text(row.get("citation_count_2025_present", ""))
        top_journal = safe_text(row.get("top_journal_2025_present", ""))
        top_paper = safe_text(row.get("top_paper_title_2025_present", ""))
        top_paper_cit = safe_text(row.get("top_paper_citations_2025_present", ""))
    return f"""
    <section class="section">
      <div class="sectionTitle">2025–Present Stats</div>
      <div class="statsGrid">
        <div class="statCard"><div class="statLabel">Publications (2025–Present)</div><div class="statValue">{html_lib.escape(pub)}</div></div>
        <div class="statCard"><div class="statLabel">Citations (2025–Present)</div><div class="statValue">{html_lib.escape(cit)}</div></div>
        <div class="statCard"><div class="statLabel">Top Journal</div><div class="statSmall">{html_lib.escape(top_journal)}</div></div>
        <div class="statCard"><div class="statLabel">Top Paper</div><div class="statSmall">{html_lib.escape(top_paper)}</div><div class="pill2">Citations: {html_lib.escape(top_paper_cit)}</div></div>
      </div>
    </section>
    """


def build_authors_json(authors: list[str], summary_by_label: dict[str, pd.Series], persona_by_label: dict[str, pd.Series], expertise_map: dict[str, str], summary_df: pd.DataFrame, album_covers_src_dir: Path, covers_dir: Path, data_dir: Path, build_report: list[tuple[str, str, str]]) -> None:
    theme_cols = [c for c in ["themes", "theme_list", "research_themes", "top_themes"] if c in summary_df.columns]
    records = []
    for author_label in authors:
        ok_cover, _rel_author, rel_root = ensure_album_cover_in_docs(author_label, album_covers_src_dir, covers_dir, build_report)
        row = summary_by_label.get(author_label)
        p = persona_by_label.get(author_label)
        themes = []
        if row is not None and theme_cols:
            for col in theme_cols:
                themes = safe_split_themes(row.get(col, ""))
                if themes:
                    break
        display_name = display_name_from_label(author_label)
        expertise_preview = safe_text(expertise_map.get(author_label, ""))[:180]
        records.append(
            {
                "author_label": author_label,
                "display_name": display_name,
                "artist_name": safe_text(p.get("artist_name", "")) if p is not None else "",
                "album_title": safe_text(p.get("album_title", "")) if p is not None else "",
                "themes": themes,
                "top_journal": safe_text(row.get("top_journal_2025_present", "")) if row is not None else "",
                "top_paper": safe_text(row.get("top_paper_title_2025_present", "")) if row is not None else "",
                "expertise_preview": expertise_preview,
                "profile_url": f"authors/{author_label}/",
                "cover_url": rel_root if ok_cover else "",
            }
        )
    records.sort(key=lambda r: (r.get("display_name") or r.get("author_label") or "").lower())
    (data_dir / "authors.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate VALIANT Wrapped site with browse index.")
    parser.add_argument("--summary-file", type=Path, default=None, help="Path to author_summary_2025_present.csv")
    parser.add_argument("--persona-file", type=Path, default=None, help="Path to outputs/author_music_personas.csv")
    parser.add_argument("--expertise-dir", type=Path, default=None, help="Directory containing per-author expertise TXT files")
    parser.add_argument("--recommendations-file", type=Path, default=None, help="Optional CSV of recommendations")
    parser.add_argument("--album-covers-dir", type=Path, default=None, help="Directory containing generated album covers")
    parser.add_argument("--output-dir", type=Path, default=DOCS_DIR, help="Docs/publish directory")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    summary_file = args.summary_file or first_existing(DEFAULT_SUMMARY_CANDIDATES)
    persona_file = args.persona_file or first_existing(DEFAULT_PERSONA_CANDIDATES)
    expertise_dir = args.expertise_dir or first_existing(DEFAULT_EXPERTISE_DIR_CANDIDATES)
    recommendations_file = args.recommendations_file or first_existing(DEFAULT_RECOMMENDATION_CANDIDATES)
    album_covers_src_dir = args.album_covers_dir or first_existing(DEFAULT_COVER_DIR_CANDIDATES)

    if summary_file is None:
        raise FileNotFoundError("Could not find summary file. Expected one of: " + ", ".join(str(p) for p in DEFAULT_SUMMARY_CANDIDATES))
    if persona_file is None:
        raise FileNotFoundError("Could not find persona CSV. Expected one of: " + ", ".join(str(p) for p in DEFAULT_PERSONA_CANDIDATES))
    if album_covers_src_dir is None:
        raise FileNotFoundError("Could not find album cover directory. Expected one of: " + ", ".join(str(p) for p in DEFAULT_COVER_DIR_CANDIDATES))

    docs_dir, author_dir, assets_dir, covers_dir, data_dir = rebuild_docs_folder(args.output_dir)
    write_browse_assets(assets_dir)

    generated_iso = now_iso()
    build_report: list[tuple[str, str, str]] = []

    summary_df = pd.read_csv(summary_file)
    summary_df["author_label"] = build_author_label_column(summary_df)

    persona_df = pd.read_csv(persona_file)
    persona_df["author_label"] = build_author_label_column(persona_df)

    expertise_map = load_expertise_map(expertise_dir)

    recommendation_map: dict[str, list[dict[str, str]]] = {}
    if recommendations_file and recommendations_file.exists():
        rec_df = normalize_recommendations(pd.read_csv(recommendations_file))
        if not rec_df.empty:
            recommendation_map = {
                label: grp.drop(columns=["author_label"]).to_dict(orient="records")
                for label, grp in rec_df.groupby("author_label")
            }

    summary_by_label = {row["author_label"]: row for _, row in summary_df.iterrows() if row.get("author_label")}
    persona_by_label = {row["author_label"]: row for _, row in persona_df.iterrows() if row.get("author_label")}

    authors = sorted(set(summary_by_label) | set(persona_by_label) | set(expertise_map) | set(recommendation_map))

    for author_label in authors:
        summary_row = summary_by_label.get(author_label)
        persona_row = persona_by_label.get(author_label)
        display_name = display_name_from_label(author_label)
        write_author_pages(
            author_dir=author_dir,
            author_label=author_label,
            display_name=display_name,
            generated_iso=generated_iso,
            stats_html=stats_section(summary_row),
            expertise_section=expertise_html(expertise_map.get(author_label, "")),
            persona_section=persona_html(author_label, persona_row, album_covers_src_dir, covers_dir, build_report),
            recommendations_section=recommendations_html(recommendation_map.get(author_label, [])),
        )

    build_authors_json(authors, summary_by_label, persona_by_label, expertise_map, summary_df, album_covers_src_dir, covers_dir, data_dir, build_report)
    write_browse_index(docs_dir, generated_iso)

    report_path = docs_dir / "build_report.csv"
    pd.DataFrame(build_report, columns=["author_label", "issue", "details"]).to_csv(report_path, index=False, encoding="utf-8")

    print("Site generation complete.")
    print(f"Summary input: {summary_file}")
    print(f"Persona input: {persona_file}")
    print(f"Expertise dir: {expertise_dir if expertise_dir else 'not found / skipped'}")
    print(f"Recommendations: {recommendations_file if recommendations_file else 'not found / skipped'}")
    print(f"Published to: {docs_dir}")
    print(f"Browse page: {docs_dir / 'index.html'}")
    print(f"Authors JSON: {data_dir / 'authors.json'}")
    print(f"Build report: {report_path}")


if __name__ == "__main__":
    main()
