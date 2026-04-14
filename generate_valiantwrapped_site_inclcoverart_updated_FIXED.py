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
        rows.append(f'<div class="track-row"><div class="track-num">{i:02d}</div><div class="track-title">{html_lib.escape(t)}</div></div>')
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
        return False, ""
    dst = asset_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        build_report.append((author_label, "album_cover_copy_failed", repr(e)))
        return False, ""
    if not dst.exists():
        build_report.append((author_label, "album_cover_copy_missing_dst", str(dst)))
        return False, ""
    return True, f"../assets/album_covers/{html_lib.escape(dst.name)}"


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

    # Try JSON first in case the txt is serialized structured output.
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

    # Fallbacks for numbered/song lines anywhere in the file.
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
        persona_by_label = {
            row["author_label"]: row
            for _, row in persona_df.iterrows()
            if row.get("author_label")
        }
        return persona_by_label

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


PAGE_STYLE = '''
<style>
:root{
  --bg:#07130c; --bg2:#0d1f14; --card:#111a16; --card2:#16241d;
  --text:#edf7ef; --muted:#a9beb0; --line:rgba(255,255,255,.08);
  --accent:#1ed760; --accent2:#5cc8ff; --shadow:0 14px 36px rgba(0,0,0,.30); --radius:20px;
}
*{box-sizing:border-box}
body{
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  margin:0; color:var(--text);
  background:
    radial-gradient(1000px 520px at 10% 0%, rgba(30,215,96,.18), transparent 60%),
    radial-gradient(900px 520px at 100% 10%, rgba(92,200,255,.12), transparent 60%),
    linear-gradient(180deg,var(--bg),var(--bg2));
}
.container{max-width:1080px; margin:0 auto; padding:36px 20px 60px;}
.hero{
  position:relative; padding:28px; border-radius:var(--radius);
  background:linear-gradient(135deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border:1px solid var(--line); box-shadow:var(--shadow); overflow:hidden;
}
.hero::after{
  content:""; position:absolute; right:-140px; top:-140px; width:320px; height:320px;
  background:radial-gradient(circle at center, rgba(30,215,96,.28), transparent 62%);
}
.kicker{color:#cbe2d2; font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:12px;}
h1{font-size:46px; line-height:1.02; margin:10px 0 8px;}
.subtitle{color:var(--muted); margin:0; max-width:74ch; line-height:1.6;}
.section{
  margin-top:22px; padding:22px; border-radius:var(--radius);
  background:linear-gradient(180deg, var(--card), var(--card2));
  border:1px solid var(--line); box-shadow:var(--shadow);
}
.section h2{margin:0 0 14px; color:var(--accent); font-size:26px;}
.grid{display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px;}
.card{background:rgba(255,255,255,.03); border:1px solid var(--line); border-radius:16px; padding:16px;}
.card .label{color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px;}
.card .value{font-size:28px; font-weight:900;}
.card .small{color:var(--muted); line-height:1.5;}
.card .small b{color:var(--text)}
.pill{display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(30,215,96,.12); border:1px solid rgba(30,215,96,.22); color:var(--accent); font-weight:800; font-size:12px;}
.album-card{border-radius:18px; padding:18px; background:linear-gradient(135deg, rgba(30,215,96,.08), rgba(92,200,255,.08)); border:1px solid var(--line);}
.artist{font-size:22px; font-weight:900; margin:0 0 10px;}
.bio,.expertise{color:var(--muted); line-height:1.6; margin:0 0 14px;}
.album-title{display:flex; align-items:center; gap:10px; margin:0 0 12px; font-size:18px; font-weight:900;}
.album-title span{color:var(--accent2);}
.cover-wrap{display:flex; justify-content:center; margin:14px 0;}
.album-cover{width:340px; max-width:100%; height:auto; border-radius:16px; box-shadow:0 18px 45px rgba(0,0,0,.24); border:1px solid rgba(255,255,255,.14);}
.cover-placeholder{display:flex; justify-content:center; align-items:center; margin:14px 0; padding:18px; border-radius:16px; border:1px dashed rgba(169,190,176,.45); color:var(--muted); background:rgba(255,255,255,.03); font-weight:700;}
.tracklist{border-radius:14px; overflow:hidden; border:1px solid var(--line); background:rgba(0,0,0,.14);}
.track-row{display:flex; gap:14px; padding:12px 14px; align-items:center; border-bottom:1px solid rgba(255,255,255,.06);}
.track-row:last-child{border-bottom:none}
.track-num{width:44px; color:rgba(255,255,255,.58); font-weight:800; font-variant-numeric:tabular-nums;}
.track-title{font-weight:650;}
.nav-actions{margin-top:18px; display:flex; gap:10px; flex-wrap:wrap;}
.btn{display:inline-flex; align-items:center; justify-content:center; text-decoration:none; border:none; border-radius:14px; padding:12px 16px; font-size:14px; font-weight:800;}
.btn.primary{background:linear-gradient(135deg,var(--accent),#18b34e); color:#04130a;}
.footer-note{color:var(--muted); line-height:1.6;}
.index-grid{display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px;}
.author-card{display:block; text-decoration:none; color:inherit; background:linear-gradient(180deg, var(--card), var(--card2)); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:var(--shadow);}
.author-name{font-size:20px; font-weight:900; margin:0 0 8px;}
.author-meta{color:var(--muted); font-size:13px; line-height:1.5;}
@media (max-width: 860px){.grid,.index-grid{grid-template-columns:1fr;} h1{font-size:38px;} .album-cover{width:320px;}}
</style>
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
    <div class="grid">
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
    else:
        artist_name_raw = str(persona_row.get("artist_name", "") or "")
        album_title_raw = str(persona_row.get("album_title", "") or "")
        persona_bio_raw = str(persona_row.get("persona_bio", "") or "")

        persona_bio_html = html_lib.escape(persona_bio_raw).replace("\n", "<br>")
        ok, rel_path = copy_album_cover_into_docs(album_covers_src_dir, asset_dir, author_label, build_report)
        cover_block = f'<div class="cover-wrap"><img class="album-cover" src="{rel_path}" alt="{html_lib.escape((artist_name_raw + " " + album_title_raw).strip())}" loading="lazy"></div>' if ok else '<div class="cover-placeholder">Album cover art not available yet.</div>'
        persona_html = f'''
        <div class="album-card">
          <div class="artist">{html_lib.escape(artist_name_raw)}</div>
          <p class="bio">{persona_bio_html}</p>
          <div class="album-title">Album: <span>{html_lib.escape(album_title_raw)}</span></div>
          {cover_block}
          {format_tracklist(persona_row.get("tracklist", ""))}
        </div>
        '''

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
<div class="container">
  <div class="hero">
    <div class="kicker">VALIANT Wrapped</div>
    <h1>{html_lib.escape(display_name)}</h1>
    <p class="subtitle">A year-in-review snapshot of publications, citations, expertise, and a fictional musical persona inspired by this research portfolio.</p>
    <div class="nav-actions"><a class="btn primary" href="../index.html">Explore more authors</a></div>
  </div>
  <div class="section">
    <h2>2025–Present Stats</h2>
    {summary_html}
  </div>
  <div class="section">
    <h2>Research Expertise</h2>
    <p class="expertise">{expertise_html}</p>
  </div>
  <div class="section">
    <h2>Musical Persona</h2>
    <p class="footer-note">To celebrate this work, we created a fictional musical persona inspired by the publication profile and research themes above.</p>
    {persona_html}
  </div>
</div>
</body>
</html>
'''
    (author_dir / f"{author_label}.html").write_text(page_html, encoding="utf-8")


def generate_index_page(author_labels, docs_dir):
    cards = []
    for label in author_labels:
        _, _, scopus_id, display_name = parse_author_name(label)
        cards.append(f'''
        <a class="author-card" href="authors/{label}.html">
          <div class="author-name">{html_lib.escape(display_name)}</div>
          <div class="author-meta">Label: {html_lib.escape(label)}<br>Scopus ID: {html_lib.escape(scopus_id)}</div>
          <div class="nav-actions" style="margin-top:14px;"><span class="btn primary">Open profile</span></div>
        </a>
        ''')
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
<div class="container">
  <div class="hero">
    <div class="kicker">VALIANT Wrapped</div>
    <h1>Browse author profiles</h1>
    <p class="subtitle">Explore research-inspired music personas, publication highlights, and author pages generated from the pipeline.</p>
  </div>
  <div class="section">
    <h2>Authors</h2>
    <div class="index-grid">{"".join(cards)}</div>
  </div>
  <div class="section">
    <h2>Build notes</h2>
    <p class="footer-note">Generated: {html_lib.escape(now_iso())}<br>Published folder: {html_lib.escape(str(docs_dir))}</p>
  </div>
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
    summary_df["author_label"] = summary_df["author_id"].apply(canonical_author_label) if "author_id" in summary_df.columns else summary_df["author_file"].apply(canonical_author_label)

    build_report: List[Tuple[str, str, str]] = []
    persona_by_label = load_persona_rows(persona_file, persona_dir, build_report)

    summary_by_label = {row["author_label"]: row for _, row in summary_df.iterrows() if row.get("author_label")}
    author_labels = sorted(set(summary_by_label.keys()) | set(persona_by_label.keys()))

    for author_label in author_labels:
        generate_author_page(author_label, summary_by_label.get(author_label), persona_by_label.get(author_label), expertise_dir, album_covers_src_dir, author_dir, asset_dir, build_report)

    generate_index_page(author_labels, docs_dir)

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
