#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def canonical_author_label(value: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().replace("\\", "/")
    s = os.path.basename(s)
    for ext in (".csv", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".html"):
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
            break
    return s.strip()


def parse_author_name(label: str):
    parts = str(label).split("_")
    last = parts[0] if len(parts) > 0 else ""
    first = parts[1] if len(parts) > 1 else ""
    scopus_id = parts[-1] if len(parts) > 2 else ""
    pretty_first = first.replace("-", " ")
    pretty_last = last.replace("-", " ")
    display_name = f"{pretty_first} {pretty_last}".strip() or label
    return pretty_first, pretty_last, scopus_id, display_name


def format_tracklist(tracklist) -> str:
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
    rows = []
    for i, t in enumerate(cleaned, start=1):
        rows.append(
            f'<div class="track-row"><div class="track-num">{i:02d}</div>'
            f'<div class="track-title">{html_lib.escape(t)}</div></div>'
        )
    return '<div class="tracklist">' + "".join(rows) + "</div>"


def read_expertise_text(expertise_dir: Path, author_label: str, build_report: List[Tuple[str, str, str]]) -> str:
    path = expertise_dir / f"{author_label}.txt"
    if not path.exists():
        build_report.append((author_label, "missing_expertise_txt", str(path)))
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        build_report.append((author_label, "expertise_read_failed", repr(e)))
        return ""


def find_album_cover_source(album_covers_src_dir: Path, author_label: str):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = album_covers_src_dir / f"{author_label}{ext}"
        if p.exists():
            return p
    return None


def copy_album_cover_into_docs(album_covers_src_dir: Path, asset_dir: Path, author_label: str, build_report):
    src = find_album_cover_source(album_covers_src_dir, author_label)
    if src is None:
        build_report.append((author_label, "missing_album_cover", f"no file in {album_covers_src_dir}"))
        return False, "", ""
    dst = asset_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        build_report.append((author_label, "album_cover_copy_failed", repr(e)))
        return False, "", ""
    if not dst.exists():
        build_report.append((author_label, "album_cover_copy_missing_dst", str(dst)))
        return False, "", ""
    return True, f"../assets/album_covers/{html_lib.escape(dst.name)}", f"assets/album_covers/{html_lib.escape(dst.name)}"


def _clean_block(text: str) -> str:
    return text.strip().strip('"').strip("'").strip()


def _extract_field(text: str, labels: List[str]) -> str:
    for label in labels:
        pattern = rf"(?ims)^\s*{re.escape(label)}\s*[:\-]\s*(.+?)(?=^\s*[A-Za-z][A-Za-z /_-]*\s*[:\-]|\Z)"
        m = re.search(pattern, text)
        if m:
            return _clean_block(m.group(1))
    return ""


def _extract_tracklist(text: str) -> str:
    block = _extract_field(text, ["tracklist", "tracks", "song list"])
    if block:
        return block

    lines = []
    capture = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"(?i)^\s*(tracklist|tracks|song list)\s*[:\-]?\s*$", stripped):
            capture = True
            continue
        if capture:
            if re.match(r"^[A-Za-z][A-Za-z /_-]*\s*[:\-]\s*", stripped):
                break
            if stripped:
                lines.append(stripped)
    return "\n".join(lines).strip()


def parse_persona_txt(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    data = {
        "author_label": canonical_author_label(path.name),
        "artist_name": "",
        "album_title": "",
        "persona_bio": "",
        "tracklist": "",
        "status": "ok",
    }

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            for key in data:
                if key in obj and obj[key] is not None:
                    data[key] = str(obj[key]).strip()
            if data["tracklist"] and isinstance(obj.get("tracklist"), list):
                data["tracklist"] = "\n".join(str(x).strip() for x in obj["tracklist"] if str(x).strip())
            return data
        except Exception:
            pass

    data["artist_name"] = _extract_field(text, ["artist_name", "artist name", "artist", "stage name"])
    data["album_title"] = _extract_field(text, ["album_title", "album title", "album"])
    data["persona_bio"] = _extract_field(text, ["persona_bio", "persona bio", "bio", "artist bio", "description"])
    data["tracklist"] = _extract_tracklist(text)

    if not data["tracklist"]:
        song_lines = []
        for line in text.splitlines():
            s = line.strip()
            if re.match(r"^(\d+\.\s+.+|[-*]\s+.+|track\s*\d+\s*[-:].+)$", s, flags=re.IGNORECASE):
                song_lines.append(s)
        if song_lines:
            data["tracklist"] = "\n".join(song_lines)

    if not any([data["artist_name"], data["album_title"], data["persona_bio"], data["tracklist"]]):
        data["status"] = "parse_failed"
    return data


def load_persona_rows(persona_file: Path | None, persona_dir: Path | None, build_report: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, str]]:
    persona_by_label: Dict[str, Dict[str, str]] = {}

    if persona_file:
        if not persona_file.exists():
            raise FileNotFoundError(f"Persona CSV not found: {persona_file}")
        persona_df = pd.read_csv(persona_file)
        if "author_label" not in persona_df.columns:
            raise ValueError("Persona CSV is missing required column: author_label")
        persona_df["author_label"] = persona_df["author_label"].apply(canonical_author_label)
        return {
            row["author_label"]: row
            for _, row in persona_df.iterrows()
            if row.get("author_label")
        }

    if persona_dir:
        if not persona_dir.exists():
            raise FileNotFoundError(f"Persona TXT directory not found: {persona_dir}")
        txt_files = sorted(persona_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No persona TXT files found in: {persona_dir}")
        for path in txt_files:
            parsed = parse_persona_txt(path)
            label = canonical_author_label(parsed.get("author_label", path.name))
            parsed["author_label"] = label
            if parsed.get("status") == "parse_failed":
                build_report.append((label, "persona_txt_parse_failed", str(path)))
            persona_by_label[label] = parsed
        return persona_by_label

    raise ValueError("Either --persona-file or --persona-dir is required")


PAGE_STYLE = r"""
<style>
:root{
  --bg:#000000;
  --bg2:#090909;
  --sidebar:#0b0b0b;
  --panel:#121212;
  --panel2:#181818;
  --card:#1a1a1a;
  --text:#ffffff;
  --muted:#b3b3b3;
  --line:rgba(255,255,255,.08);
  --accent:#1ed760;
  --accent2:#1db954;
  --shadow:0 12px 32px rgba(0,0,0,.35);
  --radius:18px;
  --font:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;
  color:var(--text);
  font-family:var(--font);
  background:linear-gradient(180deg, #111 0%, #000 28%, #000 100%);
}
.app{
  display:grid;
  grid-template-columns:280px 1fr;
  min-height:100vh;
}
.sidebar{
  background:linear-gradient(180deg, #0b0b0b 0%, #050505 100%);
  border-right:1px solid var(--line);
  padding:18px;
}
.brand{
  display:flex;
  gap:12px;
  align-items:center;
  padding:14px;
  border-radius:16px;
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.06);
}
.logo{
  width:46px;
  height:46px;
  border-radius:50%;
  background:
    radial-gradient(circle at 30% 30%, rgba(30,215,96,.95), rgba(30,215,96,.15) 45%, transparent 52%),
    linear-gradient(135deg, #1ed760 0%, #0d7f39 100%);
  box-shadow:0 0 24px rgba(30,215,96,.28);
}
.brandTitle{font-size:18px;font-weight:800;}
.brandSub{font-size:12px;color:var(--muted);margin-top:2px;}
.menu{
  margin-top:18px;
  display:flex;
  flex-direction:column;
  gap:8px;
}
.menu a,.menu .menu-static{
  display:flex;
  align-items:center;
  gap:10px;
  padding:12px 14px;
  border-radius:14px;
  color:var(--muted);
  text-decoration:none;
  border:1px solid transparent;
  background:transparent;
}
.menu a:hover{
  background:rgba(255,255,255,.05);
  color:var(--text);
}
.menu a.active{
  background:rgba(255,255,255,.08);
  border-color:rgba(255,255,255,.05);
  color:var(--text);
}
.sidebarNote{
  margin-top:20px;
  padding:12px 14px;
  color:var(--muted);
  font-size:12px;
  line-height:1.5;
  border-radius:14px;
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.05);
}
.main{
  padding:20px 22px 40px;
}
.hero{
  padding:28px;
  border-radius:24px;
  background:
    radial-gradient(900px 360px at 0% 0%, rgba(30,215,96,.26), transparent 55%),
    linear-gradient(180deg, #1a1a1a 0%, #121212 100%);
  border:1px solid var(--line);
  box-shadow:var(--shadow);
}
.kicker{
  color:#d9d9d9;
  font-weight:800;
  letter-spacing:.12em;
  text-transform:uppercase;
  font-size:12px;
}
h1{
  font-size:46px;
  line-height:1.02;
  margin:10px 0 10px;
}
.subtitle{
  color:var(--muted);
  max-width:78ch;
  line-height:1.6;
  margin:0;
}
.nav-actions{
  margin-top:18px;
  display:flex;
  gap:10px;
  flex-wrap:wrap;
}
.btn{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  text-decoration:none;
  border-radius:999px;
  padding:12px 18px;
  font-size:14px;
  font-weight:800;
}
.btn.primary{
  background:var(--accent);
  color:#000;
}
.btn.secondary{
  background:rgba(255,255,255,.08);
  color:var(--text);
  border:1px solid rgba(255,255,255,.08);
}
.section{
  margin-top:18px;
  padding:22px;
  border-radius:22px;
  background:linear-gradient(180deg, var(--panel2), var(--panel));
  border:1px solid var(--line);
  box-shadow:var(--shadow);
}
.section h2{
  margin:0 0 14px;
  color:var(--text);
  font-size:26px;
}
.stats-grid{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:14px;
}
.card{
  background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.025));
  border:1px solid rgba(255,255,255,.06);
  border-radius:18px;
  padding:16px;
}
.card .label{
  color:var(--muted);
  font-size:12px;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:.05em;
  margin-bottom:8px;
}
.card .value{
  font-size:28px;
  font-weight:900;
}
.card .small{
  color:var(--muted);
  line-height:1.55;
}
.card .small b{color:var(--text)}
.pill{
  display:inline-block;
  margin-top:8px;
  padding:6px 10px;
  border-radius:999px;
  background:rgba(30,215,96,.12);
  border:1px solid rgba(30,215,96,.22);
  color:#7bf2a8;
  font-weight:800;
  font-size:12px;
}
.album-card{
  border-radius:20px;
  padding:18px;
  background:
    radial-gradient(720px 240px at 0% 0%, rgba(30,215,96,.16), transparent 50%),
    linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.07);
}
.artist{
  font-size:24px;
  font-weight:900;
  margin:0 0 6px;
}
.album-submeta{
  color:var(--muted);
  font-size:14px;
  margin-bottom:12px;
}
.bio,.expertise{
  color:#d0d0d0;
  line-height:1.7;
  margin:0 0 14px;
}
.album-title{
  display:flex;
  align-items:center;
  gap:10px;
  margin:0 0 14px;
  font-size:18px;
  font-weight:900;
}
.album-title span{color:var(--accent)}
.cover-wrap{
  display:flex;
  justify-content:center;
  margin:14px 0 18px;
}
.album-cover{
  width:340px;
  max-width:100%;
  aspect-ratio:1 / 1;
  object-fit:cover;
  border-radius:18px;
  box-shadow:0 20px 44px rgba(0,0,0,.38);
  border:1px solid rgba(255,255,255,.12);
  background:#202020;
}
.cover-placeholder,.cover-placeholder-tile{
  display:flex;
  justify-content:center;
  align-items:center;
  background:linear-gradient(135deg, rgba(30,215,96,.18), rgba(255,255,255,.03));
  border:1px dashed rgba(255,255,255,.18);
  color:var(--muted);
}
.cover-placeholder{
  min-height:340px;
  border-radius:18px;
  padding:18px;
  text-align:center;
  font-weight:700;
}
.tracklist{
  border-radius:16px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,.08);
  background:rgba(0,0,0,.24);
}
.track-row{
  display:flex;
  gap:14px;
  padding:12px 14px;
  align-items:center;
  border-bottom:1px solid rgba(255,255,255,.06);
}
.track-row:last-child{border-bottom:none}
.track-num{
  width:42px;
  color:rgba(255,255,255,.58);
  font-weight:800;
  font-variant-numeric:tabular-nums;
}
.track-title{font-weight:650;}
.browse-header{
  display:flex;
  align-items:end;
  justify-content:space-between;
  gap:16px;
  margin-bottom:14px;
}
.browse-title{
  font-size:30px;
  font-weight:900;
}
.browse-subtitle{
  color:var(--muted);
  margin-top:4px;
}
.author-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:16px;
}
.author-card{
  display:block;
  text-decoration:none;
  color:inherit;
  background:linear-gradient(180deg, #1b1b1b 0%, #141414 100%);
  border:1px solid rgba(255,255,255,.06);
  border-radius:18px;
  padding:16px;
  transition:transform .12s ease, box-shadow .12s ease, background .12s ease;
  box-shadow:none;
}
.author-card:hover{
  transform:translateY(-2px);
  background:linear-gradient(180deg, #242424 0%, #181818 100%);
  box-shadow:var(--shadow);
}
.cover-tile,.cover-placeholder-tile{
  width:100%;
  aspect-ratio:1 / 1;
  border-radius:14px;
  overflow:hidden;
  margin-bottom:12px;
}
.cover-tile img{
  width:100%;
  height:100%;
  object-fit:cover;
  display:block;
}
.artist-name{
  font-size:18px;
  font-weight:800;
  line-height:1.2;
}
.author-meta{
  margin-top:6px;
  color:var(--text);
  font-size:14px;
  font-weight:600;
}
.album-meta{
  margin-top:4px;
  color:var(--muted);
  font-size:13px;
  line-height:1.4;
}
.footer-note{
  color:var(--muted);
  line-height:1.6;
}
@media (max-width: 980px){
  .app{grid-template-columns:1fr;}
  .sidebar{border-right:none;border-bottom:1px solid var(--line);}
}
@media (max-width: 860px){
  .stats-grid,.author-grid{grid-template-columns:1fr;}
  h1{font-size:38px;}
  .album-cover{width:320px;}
}
</style>
"""


def get_persona_display_fields(persona_row) -> Tuple[str, str, str]:
    if persona_row is None:
        return "", "", ""
    artist_name_raw = str(persona_row.get("artist_name", "") or "")
    album_title_raw = str(persona_row.get("album_title", "") or "")
    persona_bio_raw = str(persona_row.get("persona_bio", "") or "")
    return artist_name_raw, album_title_raw, persona_bio_raw


def render_sidebar(active: str, generated: str, home_href: str) -> str:
    home_class = "active" if active == "home" else ""
    browse_class = "active" if active == "browse" else ""
    return f'''
    <aside class="sidebar">
      <div class="brand">
        <div class="logo"></div>
        <div>
          <div class="brandTitle">VALIANT Wrapped</div>
          <div class="brandSub">Research, remixed</div>
        </div>
      </div>

      <nav class="menu">
        <a class="{home_class}" href="{home_href}">Home</a>
        <a class="{browse_class}" href="{home_href}">See all authors</a>
        <div class="menu-static">Spotify-style author browse</div>
      </nav>

      <div class="sidebarNote">
        Generated: {html_lib.escape(generated)}<br>
        Static site preview
      </div>
    </aside>
    '''


def generate_author_page(author_label, summary_row, persona_row, expertise_dir, album_covers_src_dir, author_dir, asset_dir, build_report):
    _, _, _, display_name = parse_author_name(author_label)

    pub = cit = top_journal = top_paper = top_paper_cit = ""
    if summary_row is None:
        build_report.append((author_label, "missing_summary_row", author_label))
    else:
        pub = summary_row.get("pub_count_2025_present", "")
        cit = summary_row.get("citation_count_2025_present", "")
        top_journal = summary_row.get("top_journal_2025_present", "")
        top_paper = summary_row.get("top_paper_title_2025_present", "")
        top_paper_cit = summary_row.get("top_paper_citations_2025_present", "")

    summary_html = f'''
    <div class="stats-grid">
      <div class="card"><div class="label">Publications (2025–Present)</div><div class="value">{html_lib.escape(str(pub))}</div></div>
      <div class="card"><div class="label">Citations (2025–Present)</div><div class="value">{html_lib.escape(str(cit))}</div></div>
      <div class="card"><div class="label">Top Journal</div><div class="small"><b>{html_lib.escape(str(top_journal or ""))}</b></div></div>
      <div class="card"><div class="label">Top Paper</div><div class="small"><b>{html_lib.escape(str(top_paper or ""))}</b><br><span class="pill">Citations: {html_lib.escape(str(top_paper_cit or ""))}</span></div></div>
    </div>
    '''

    expertise_text = read_expertise_text(expertise_dir, author_label, build_report)
    expertise_html = html_lib.escape(expertise_text).replace("\n", "<br>") if expertise_text else "No expertise summary generated."

    if persona_row is None:
        build_report.append((author_label, "missing_persona_row", author_label))
        persona_html = "<p class='footer-note'>No persona generated.</p>"
        hero_artist = ""
        hero_album = ""
    else:
        artist_name_raw, album_title_raw, persona_bio_raw = get_persona_display_fields(persona_row)
        hero_artist = html_lib.escape(artist_name_raw)
        hero_album = html_lib.escape(album_title_raw)
        persona_bio_html = html_lib.escape(persona_bio_raw).replace("\n", "<br>")

        ok, rel_path, _ = copy_album_cover_into_docs(album_covers_src_dir, asset_dir, author_label, build_report)
        if ok:
            alt_text = html_lib.escape((artist_name_raw + " — " + album_title_raw).strip(" —"))
            cover_block = f'<div class="cover-wrap"><img class="album-cover" src="{rel_path}" alt="{alt_text}" loading="lazy"></div>'
        else:
            cover_block = '<div class="cover-placeholder">Album cover art not available yet.</div>'

        persona_html = f'''
        <div class="album-card">
          <div class="artist">{html_lib.escape(artist_name_raw)}</div>
          <div class="album-submeta">Author persona inspired by {html_lib.escape(display_name)}</div>
          <p class="bio">{persona_bio_html}</p>
          <div class="album-title">Album: <span>{html_lib.escape(album_title_raw)}</span></div>
          {cover_block}
          {format_tracklist(persona_row.get("tracklist", ""))}
        </div>
        '''

    sidebar_html = render_sidebar(active="browse", generated=now_iso(), home_href="../index.html")
    hero_meta = ""
    if hero_artist or hero_album:
        hero_meta = f'<p class="subtitle" style="margin-top:12px;">Artist name: <strong>{hero_artist or "—"}</strong> &nbsp;•&nbsp; Album: <strong>{hero_album or "—"}</strong></p>'

    page_html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(display_name)} • VALIANT Wrapped</title>
<meta name="robots" content="index,follow">
{PAGE_STYLE}
</head>
<body>
<div class="app">
  {sidebar_html}
  <main class="main">
    <div class="hero">
      <div class="kicker">VALIANT Wrapped</div>
      <h1>{html_lib.escape(display_name)}</h1>
      <p class="subtitle">A year-in-review snapshot of publications, citations, expertise, and a fictional musical persona inspired by this research portfolio.</p>
      {hero_meta}
      <div class="nav-actions">
        <a class="btn primary" href="../index.html">See all authors</a>
      </div>
    </div>

    <section class="section">
      <h2>2025–Present Stats</h2>
      {summary_html}
    </section>

    <section class="section">
      <h2>Research Expertise</h2>
      <p class="expertise">{expertise_html}</p>
    </section>

    <section class="section">
      <h2>Musical Persona</h2>
      <p class="footer-note">To celebrate this work, we created a fictional musical persona inspired by the publication profile and research themes above.</p>
      {persona_html}
    </section>
  </main>
</div>
</body>
</html>
'''
    (author_dir / f"{author_label}.html").write_text(page_html, encoding="utf-8")


def build_index_card(label: str, persona_row, album_covers_src_dir: Path, asset_dir: Path, build_report) -> str:
    _, _, scopus_id, display_name = parse_author_name(label)
    artist_name_raw, album_title_raw, _ = get_persona_display_fields(persona_row)
    ok, _, rel_root = copy_album_cover_into_docs(album_covers_src_dir, asset_dir, label, build_report)

    if ok:
        cover_html = f'<div class="cover-tile"><img src="{rel_root}" alt="{html_lib.escape(display_name)} album cover" loading="lazy"></div>'
    else:
        cover_html = '<div class="cover-placeholder-tile">No cover</div>'

    return f'''
    <a class="author-card" href="authors/{label}.html">
      {cover_html}
      <div class="artist-name">{html_lib.escape(artist_name_raw or "Persona pending")}</div>
      <div class="author-meta">{html_lib.escape(display_name)}</div>
      <div class="album-meta">Album: {html_lib.escape(album_title_raw or "TBD")}<br>Scopus ID: {html_lib.escape(scopus_id)}</div>
    </a>
    '''


def generate_index_page(author_labels, persona_by_label, docs_dir, album_covers_src_dir, asset_dir, build_report):
    cards = []
    for label in sorted(author_labels, key=lambda x: parse_author_name(x)[3].lower()):
        cards.append(build_index_card(label, persona_by_label.get(label), album_covers_src_dir, asset_dir, build_report))

    sidebar_html = render_sidebar(active="home", generated=now_iso(), home_href="index.html")
    index_html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VALIANT Wrapped</title>
<meta name="robots" content="index,follow">
{PAGE_STYLE}
</head>
<body>
<div class="app">
  {sidebar_html}
  <main class="main">
    <div class="hero">
      <div class="kicker">VALIANT Wrapped</div>
      <h1>Browse author profiles</h1>
      <p class="subtitle">Explore research-inspired artist personas in a Spotify-style browse view. Authors are arranged alphabetically and each card highlights the artist name, author name, and album cover.</p>
      <div class="nav-actions">
        <a class="btn primary" href="#authors">Browse authors</a>
      </div>
    </div>

    <section class="section" id="authors">
      <div class="browse-header">
        <div>
          <div class="browse-title">All authors</div>
          <div class="browse-subtitle">Alphabetical by author name</div>
        </div>
      </div>
      <div class="author-grid">
        {"".join(cards)}
      </div>
    </section>
  </main>
</div>
</body>
</html>
'''
    (docs_dir / "index.html").write_text(index_html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-file", required=True)
    ap.add_argument("--persona-file", default="")
    ap.add_argument("--persona-dir", default="")
    ap.add_argument("--expertise-dir", required=True)
    ap.add_argument("--scopus-db", default="")
    ap.add_argument("--album-covers-dir", required=True)
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--base-url", default="")
    args = ap.parse_args()

    summary_file = Path(args.summary_file).resolve()
    persona_file = Path(args.persona_file).resolve() if args.persona_file else None
    persona_dir = Path(args.persona_dir).resolve() if args.persona_dir else None
    expertise_dir = Path(args.expertise_dir).resolve()
    album_covers_src_dir = Path(args.album_covers_dir).resolve()
    docs_dir = Path(args.docs_dir).resolve()
    author_dir = docs_dir / "authors"
    asset_dir = docs_dir / "assets" / "album_covers"

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_file}")

    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    author_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_file)
    summary_df["author_label"] = (
        summary_df["author_id"].apply(canonical_author_label)
        if "author_id" in summary_df.columns
        else summary_df["author_file"].apply(canonical_author_label)
    )

    build_report: List[Tuple[str, str, str]] = []
    persona_by_label = load_persona_rows(persona_file, persona_dir, build_report)

    summary_by_label = {row["author_label"]: row for _, row in summary_df.iterrows() if row.get("author_label")}
    author_labels = sorted(set(summary_by_label.keys()) | set(persona_by_label.keys()), key=lambda x: parse_author_name(x)[3].lower())

    for author_label in author_labels:
        generate_author_page(
            author_label,
            summary_by_label.get(author_label),
            persona_by_label.get(author_label),
            expertise_dir,
            album_covers_src_dir,
            author_dir,
            asset_dir,
            build_report,
        )

    generate_index_page(author_labels, persona_by_label, docs_dir, album_covers_src_dir, asset_dir, build_report)

    report_path = docs_dir / "build_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["author_label", "issue", "details"])
        for row in build_report:
            w.writerow(row)

    print("Site generation complete.")
    print(f"Published to: {docs_dir}")
    print(f"Index: {docs_dir / 'index.html'}")
    print(f"Author pages: {author_dir}")
    print(f"Build report: {report_path}")


if __name__ == "__main__":
    main()
