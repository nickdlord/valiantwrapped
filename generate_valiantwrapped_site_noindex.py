#!/usr/bin/env python3
"""
generate_valiantwrapped_site_noindex.py
GitHub Pages-friendly generator that creates ONLY individual author pages
(no public index page) and writes a URL manifest for sharing direct links.

Supports two input modes:

1) Batch CSV mode (original behavior)
   Example:
     python generate_valiantwrapped_site_noindex.py \
       --summary-file author_summary_2025_present.csv \
       --persona-file outputs/author_music_personas.csv

2) Single-author TXT mode (new; for pipeline Option B)
   Example:
     python generate_valiantwrapped_site_noindex.py \
       --summary-dir /tmp/.../summary_txt \
       --persona-dir /tmp/.../personas_txt \
       --album-covers-dir /tmp/.../album_covers \
       --docs-dir /tmp/.../docs \
       --author-label Landman_Bennett_16679175200
"""

import os
import re
import shutil
import argparse
from pathlib import Path
import html as html_lib
from datetime import datetime

import pandas as pd


# ----------------------------
# HELPERS
# ----------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def canonical_author_label(value: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().replace("\\", "/")
    s = os.path.basename(s)
    if s.lower().endswith(".csv"):
        s = s[:-4]
    if s.lower().endswith(".txt"):
        s = s[:-4]
    return s.strip()


def parse_author_name(label: str):
    parts = str(label).split("_")
    last = parts[0] if len(parts) > 0 else ""
    first = parts[1] if len(parts) > 1 else ""
    return first, last


def normalize_base_url(base_url: str) -> str:
    if not base_url:
        return ""
    return str(base_url).rstrip("/")


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


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
        t = re.sub(r"^\s*\d+[.)-]?\s*", "", t)
        t = re.sub(r"^\s*[-*•]+\s*", "", t)
        t = t.strip()
        if t:
            cleaned.append(t)

    if not cleaned:
        return ""

    items = []
    for i, t in enumerate(cleaned, start=1):
        safe = html_lib.escape(t)
        items.append(
            f"""
            <div class="track-row">
                <div class="track-num">{i:02d}</div>
                <div class="track-title">{safe}</div>
            </div>
            """
        )

    return f'<div class="tracklist">\n{"".join(items)}\n</div>'


def parse_summary_txt(summary_path: Path, author_label: str) -> dict:
    text = summary_path.read_text(encoding="utf-8", errors="replace")

    def grab(patterns, default=""):
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return default

    pub = grab([
        r"^\s*Publications\s*\(?(?:2025\s*[–-]\s*Present|2025\s*[–-]\s*2026)\)?\s*:\s*(.+)$",
        r"^\s*Publication\s+Count\s*:\s*(.+)$",
        r"^\s*Total\s+Publications\s*:\s*(.+)$",
        r"^\s*Publications\s*:\s*(.+)$",
    ])

    cit = grab([
        r"^\s*Citations\s*\(?(?:2025\s*[–-]\s*Present|2025\s*[–-]\s*2026)\)?\s*:\s*(.+)$",
        r"^\s*Citation\s+Count\s*:\s*(.+)$",
        r"^\s*Total\s+Citations\s*:\s*(.+)$",
        r"^\s*Citations\s*:\s*(.+)$",
    ])

    top_journal = grab([
        r"^\s*Top\s+Journal\s*:\s*(.+)$",
        r"^\s*Most\s+Common\s+Journal\s*:\s*(.+)$",
        r"^\s*Journal\s*:\s*(.+)$",
    ])

    top_paper = grab([
        r"^\s*Top\s+Paper\s*:\s*(.+)$",
        r"^\s*Most\s+Cited\s+Paper\s*:\s*(.+)$",
        r"^\s*Top\s+Paper\s+Title\s*:\s*(.+)$",
    ])

    top_paper_cit = grab([
        r"^\s*Top\s+Paper\s+Citations\s*:\s*(.+)$",
        r"^\s*Most\s+Cited\s+Paper\s+Citations\s*:\s*(.+)$",
        r"^\s*Paper\s+Citations\s*:\s*(.+)$",
    ])

    return {
        "author_label": canonical_author_label(author_label),
        "author_file": f"{canonical_author_label(author_label)}.txt",
        "pub_count_2025_present": pub,
        "citation_count_2025_present": cit,
        "top_journal_2025_present": top_journal,
        "top_paper_title_2025_present": top_paper,
        "top_paper_citations_2025_present": top_paper_cit,
        "summary_raw_text": text,
    }


def parse_persona_txt(persona_path: Path, author_label: str) -> dict:
    text = persona_path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]

    artist_name = ""
    album_title = ""
    persona_bio_lines = []
    track_lines = []

    in_tracks = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_tracks:
                track_lines.append("")
            else:
                persona_bio_lines.append("")
            continue

        artist_match = re.match(r"^\s*Artist\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if artist_match:
            artist_name = artist_match.group(1).strip()
            continue

        album_match = re.match(r"^\s*Album\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if album_match:
            album_title = album_match.group(1).strip()
            continue

        if re.match(r"^\s*(Tracklist|Tracks?)\s*:?\s*$", line, flags=re.IGNORECASE):
            in_tracks = True
            continue

        if re.match(r"^\s*(Bio|Persona\s+Bio|Description)\s*:\s*$", line, flags=re.IGNORECASE):
            in_tracks = False
            continue

        if re.match(r"^\s*(\d+[.)-]?\s+|[-*•]\s+)", line):
            in_tracks = True
            track_lines.append(stripped)
            continue

        if in_tracks:
            track_lines.append(stripped)
        else:
            # skip repeated labels in line form
            if not re.match(r"^\s*(Artist|Album|Tracklist|Tracks?|Bio|Persona\s+Bio|Description)\s*:", line, flags=re.IGNORECASE):
                persona_bio_lines.append(stripped)

    persona_bio = "\n".join([ln for ln in persona_bio_lines]).strip()
    persona_bio = re.sub(r"\n{3,}", "\n\n", persona_bio)

    cleaned_tracks = []
    for t in track_lines:
        t = t.strip()
        if not t:
            continue
        t = re.sub(r"^\s*\d+[.)-]?\s*", "", t)
        t = re.sub(r"^\s*[-*•]+\s*", "", t)
        if t:
            cleaned_tracks.append(t)

    return {
        "author_label": canonical_author_label(author_label),
        "artist_name": artist_name,
        "album_title": album_title,
        "persona_bio": persona_bio,
        "tracklist": "\n".join(cleaned_tracks),
        "status": "ok",
        "persona_raw_text": text,
    }


# ----------------------------
# STYLE
# ----------------------------

PAGE_STYLE = """
<style>
:root{
  --bg1:#f6f8ff;
  --bg2:#eef3ff;
  --card:#ffffff;
  --card2:#f3f6ff;
  --text:#1e293b;
  --muted:#64748b;
  --accent:#c5b358;
  --accent2:#6366f1;
  --shadow:0 10px 25px rgba(0,0,0,.08);
  --radius:18px;
}
* { box-sizing: border-box; }
body{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  margin:0;
  color:var(--text);
  background:
    radial-gradient(900px 600px at 10% 10%, rgba(99,102,241,.10), transparent 60%),
    radial-gradient(900px 600px at 90% 20%, rgba(197,179,88,.10), transparent 60%),
    linear-gradient(180deg, var(--bg1), var(--bg2));
}
.container{ max-width: 980px; margin: 0 auto; padding: 44px 20px 60px; }
.hero{
  position: relative; padding:28px; border-radius:var(--radius);
  background:linear-gradient(135deg, #ffffff, #f4f7ff);
  border:1px solid rgba(0,0,0,.06); box-shadow:var(--shadow); overflow: hidden;
}
.hero::after{
  content:""; position:absolute; right:-140px; top:-140px; width:320px; height:320px;
  background: radial-gradient(circle at center, rgba(56,189,248,.30), transparent 60%);
  transform: rotate(20deg);
}
.kicker{ color: var(--muted); font-weight:600; letter-spacing:.04em; text-transform:uppercase; font-size:12px; }
h1{ font-size:44px; margin:8px 0 6px; line-height:1.05; }
.subtitle{ color: var(--muted); font-size:15px; margin:0; max-width:70ch; }
.section{
  margin-top:22px; padding:22px; border-radius:var(--radius); background:var(--card);
  border:1px solid rgba(0,0,0,.05); box-shadow:var(--shadow);
}
.section h2{ margin:0 0 14px; color:var(--accent); font-size:26px; letter-spacing:.01em; }
.grid{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }
.card{ background:var(--card2); border:1px solid rgba(0,0,0,.05); border-radius:16px; padding:16px; }
.card .label{ color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; margin-bottom:8px; }
.card .value{ font-size:26px; font-weight:800; }
.card .small { color: var(--muted); }
.card .small b{ color:var(--text); font-weight:700; }
.pill{
  display:inline-block; padding:6px 10px; border-radius:999px;
  background: rgba(56,189,248,.14); border:1px solid rgba(56,189,248,.25);
  color: var(--accent); font-weight:700; font-size:12px;
}
.album-card{
  border-radius:var(--radius); padding:18px;
  background: linear-gradient(135deg, rgba(56,189,248,.12), rgba(167,139,250,.12));
  border: 1px solid rgba(255,255,255,.10);
}
.artist{ font-size:22px; font-weight:900; margin:0 0 10px; }
.bio{ color:var(--muted); line-height:1.55; margin:0 0 14px; }
.album-title{ display:flex; align-items:center; gap:10px; margin:0 0 12px; font-size:18px; font-weight:900; }
.album-title span{ color: var(--accent2); }
.cover-wrap{ display:flex; justify-content:center; margin:14px 0 14px; }
.album-cover{
  width:340px; max-width:100%; height:auto; border-radius:16px;
  box-shadow:0 18px 45px rgba(0,0,0,.18); border:1px solid rgba(255,255,255,.25);
  background: rgba(255,255,255,.25);
}
.cover-placeholder{
  display:flex; justify-content:center; align-items:center; margin:14px 0 14px; padding:16px;
  border-radius:16px; border:1px dashed rgba(100,116,139,.5); color:var(--muted);
  background: rgba(255,255,255,.35); font-weight:650;
}
.tracklist{
  border-radius:14px; overflow:hidden; border:1px solid rgba(255,255,255,.10);
  background: rgba(0,0,0,.12);
}
.track-row{ display:flex; gap:14px; padding:12px 14px; align-items:center; border-bottom:1px solid rgba(255,255,255,.08); }
.track-row:last-child{ border-bottom:none; }
.track-num{ width:44px; color:rgba(255,255,255,.65); font-weight:800; font-variant-numeric:tabular-nums; }
.track-title{ font-weight:650; }
.footer-note{ color:var(--muted); line-height:1.5; }
@media (max-width: 760px){
  .grid{ grid-template-columns: 1fr; }
  h1{ font-size:36px; }
  .album-cover{ width:320px; }
}
</style>
"""


# ----------------------------
# ARGPARSE
# ----------------------------

def build_parser():
    ap = argparse.ArgumentParser()

    # Original batch mode
    ap.add_argument("--summary-file", default="author_summary_2025_present.csv",
                    help="Path to metrics summary CSV (batch mode)")
    ap.add_argument("--persona-file", default=os.path.join("outputs", "author_music_personas.csv"),
                    help="Path to persona CSV (batch mode)")

    # New single-author mode
    ap.add_argument("--summary-dir", default="",
                    help="Folder containing per-author summary TXT files (single-author mode)")
    ap.add_argument("--persona-dir", default="",
                    help="Folder containing per-author persona TXT files (single-author mode)")
    ap.add_argument("--author-label", default="",
                    help="Author label like Last_First_ScopusID (single-author mode)")

    ap.add_argument("--album-covers-dir", default=os.path.join("outputs", "album_covers"),
                    help="Folder containing generated album cover images")
    ap.add_argument("--docs-dir", default="docs",
                    help="GitHub Pages publish directory")
    ap.add_argument("--base-url", default="",
                    help="Optional public site base URL for manifest")
    return ap


# ----------------------------
# MAIN
# ----------------------------

def main():
    args = build_parser().parse_args()

    base_dir = Path(__file__).resolve().parent
    docs_dir = resolve_path(base_dir, args.docs_dir)
    author_dir = docs_dir / "authors"
    asset_dir = docs_dir / "assets" / "album_covers"
    album_covers_src_dir = resolve_path(base_dir, args.album_covers_dir)
    base_url = normalize_base_url(args.base_url)

    single_mode = bool(args.summary_dir or args.persona_dir or args.author_label)

    if single_mode:
        if not (args.summary_dir and args.persona_dir and args.author_label):
            raise ValueError("Single-author mode requires --summary-dir, --persona-dir, and --author-label")

        author_label = canonical_author_label(args.author_label)
        summary_path = resolve_path(base_dir, os.path.join(args.summary_dir, f"{author_label}.txt"))
        persona_path = resolve_path(base_dir, os.path.join(args.persona_dir, f"{author_label}.txt"))

        if not summary_path.exists():
            raise FileNotFoundError(f"Summary TXT not found: {summary_path}")
        if not persona_path.exists():
            raise FileNotFoundError(f"Persona TXT not found: {persona_path}")

        summary_df = pd.DataFrame([parse_summary_txt(summary_path, author_label)])
        persona_df = pd.DataFrame([parse_persona_txt(persona_path, author_label)])
    else:
        summary_file = resolve_path(base_dir, args.summary_file)
        persona_file = resolve_path(base_dir, args.persona_file)

        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
        if not persona_file.exists():
            raise FileNotFoundError(f"Persona file not found: {persona_file}")

        summary_df = pd.read_csv(summary_file)
        persona_df = pd.read_csv(persona_file)

    # Rebuild docs/
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    author_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    build_report = []
    manifest_rows = []

    def find_album_cover_source(author_label: str):
        exts = [".png", ".jpg", ".jpeg", ".webp"]
        for ext in exts:
            p = album_covers_src_dir / f"{author_label}{ext}"
            if p.exists():
                return p
        return None

    def copy_album_cover_into_docs(author_label: str):
        src = find_album_cover_source(author_label)
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

        rel = f"../assets/album_covers/{html_lib.escape(dst.name)}"
        return True, rel

    def album_cover_block(author_label: str, artist_name_raw: str, album_title_raw: str) -> str:
        ok, rel_path = copy_album_cover_into_docs(author_label)
        if not ok:
            return """
            <div class="cover-placeholder">
              Album cover art not available yet.
            </div>
            """

        alt = html_lib.escape(f"Album cover for {artist_name_raw} — {album_title_raw}".strip(" —"))
        return f"""
          <div class="cover-wrap">
            <img class="album-cover" src="{rel_path}" alt="{alt}" loading="lazy">
          </div>
        """

    if "author_file" in summary_df.columns and "author_label" not in summary_df.columns:
        summary_df["author_label"] = summary_df["author_file"].apply(canonical_author_label)
    elif "author_label" in summary_df.columns:
        summary_df["author_label"] = summary_df["author_label"].apply(canonical_author_label)
    else:
        raise ValueError("Summary input is missing author identification columns")

    if "author_label" not in persona_df.columns:
        raise ValueError("Persona input is missing required column: author_label")
    persona_df["author_label"] = persona_df["author_label"].apply(canonical_author_label)

    summary_by_label = {
        row["author_label"]: row
        for _, row in summary_df.iterrows()
        if row.get("author_label")
    }
    persona_by_label = {
        row["author_label"]: row
        for _, row in persona_df.iterrows()
        if row.get("author_label")
    }

    def generate_author_page(author_label: str):
        author_label = canonical_author_label(author_label)
        first, last = parse_author_name(author_label)

        row = summary_by_label.get(author_label)
        if row is None:
            build_report.append((author_label, "missing_summary_row", f"searched label={author_label}"))
            pub = cit = top_journal = top_paper = top_paper_cit = ""
        else:
            pub = row.get("pub_count_2025_present", "")
            cit = row.get("citation_count_2025_present", "")
            top_journal = row.get("top_journal_2025_present", "")
            top_paper = row.get("top_paper_title_2025_present", "")
            top_paper_cit = row.get("top_paper_citations_2025_present", "")

        top_journal_safe = html_lib.escape(str(top_journal or ""))
        top_paper_safe = html_lib.escape(str(top_paper or ""))

        summary_html = f"""
        <div class="grid">
          <div class="card">
            <div class="label">Publications (2025–Present)</div>
            <div class="value">{html_lib.escape(str(pub or ''))}</div>
          </div>

          <div class="card">
            <div class="label">Citations (2025–Present)</div>
            <div class="value">{html_lib.escape(str(cit or ''))}</div>
          </div>

          <div class="card">
            <div class="label">Top Journal</div>
            <div class="small"><b>{top_journal_safe}</b></div>
          </div>

          <div class="card">
            <div class="label">Top Paper</div>
            <div class="small"><b>{top_paper_safe}</b><br>
            <span class="pill">Citations: {html_lib.escape(str(top_paper_cit or ''))}</span>
            </div>
          </div>
        </div>
        """

        p = persona_by_label.get(author_label)
        if p is None:
            build_report.append((author_label, "missing_persona_row", f"searched label={author_label}"))
            persona_html = "<p class='footer-note'>No persona generated.</p>"
        else:
            status_val = str(p.get("status", "")).strip()
            if status_val and status_val.lower() != "ok":
                build_report.append((author_label, "persona_status", status_val))

            artist_name_raw = str(p.get("artist_name", "") or "")
            album_title_raw = str(p.get("album_title", "") or "")
            artist_name = html_lib.escape(artist_name_raw)
            persona_bio = html_lib.escape(str(p.get("persona_bio", "") or "")).replace("\n", "<br>")
            album_title = html_lib.escape(album_title_raw)
            cover_block = album_cover_block(author_label, artist_name_raw, album_title_raw)
            tracklist_html = format_tracklist(p.get("tracklist", ""))

            persona_html = f"""
            <div class="album-card">
              <div class="artist">{artist_name}</div>
              <p class="bio">{persona_bio}</p>
              <div class="album-title">Album: <span>{album_title}</span></div>
              {cover_block}
              {tracklist_html}
            </div>
            """

        page_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{html_lib.escape((first + ' ' + last).strip())} • VALIANT Wrapped</title>
{PAGE_STYLE}
</head>
<body>
<div class="container">
  <div class="hero">
    <div class="kicker">VALIANT Wrapped • 2025–2026</div>
    <h1>{html_lib.escape((first + ' ' + last).strip())}</h1>
    <p class="subtitle">
      A year-in-review snapshot of publications, citations, and a fictional musical persona inspired by your work.
    </p>
  </div>

  <div class="section">
    <h2>2025–2026 Stats</h2>
    {summary_html}
  </div>

  <div class="section">
    <h2>Your Musical Persona</h2>
    <p class="footer-note">
      People like you are what make our discovery center so vibrant and unique.
      To celebrate your work, we created a fictional musical persona inspired by your publishing history.
    </p>
    {persona_html}
  </div>

  <div class="section">
    <h2>Thank You</h2>
    <p class="footer-note">
      Thank you for being part of our discovery center and for contributing to another incredible year of innovation and collaboration.
    </p>
  </div>
</div>
</body>
</html>
"""

        output_path = author_dir / f"{author_label}.html"
        output_path.write_text(page_html, encoding="utf-8")

        rel_url = f"authors/{author_label}.html"
        full_url = f"{base_url}/{rel_url}" if base_url else ""
        manifest_rows.append({
            "author_label": author_label,
            "first_name": first,
            "last_name": last,
            "display_name": f"{first} {last}".strip(),
            "relative_url": rel_url,
            "full_url": full_url,
        })

    authors = summary_df["author_label"].dropna().astype(str).unique()
    for author_label in authors:
        generate_author_page(author_label)

    report_path = docs_dir / "build_report.csv"
    report_df = pd.DataFrame(build_report, columns=["author_label", "issue", "details"])
    report_df.to_csv(report_path, index=False, encoding="utf-8")

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["last_name", "first_name", "author_label"])
    manifest_csv_path = docs_dir / "author_page_urls.csv"
    manifest_xlsx_path = docs_dir / "author_page_urls.xlsx"

    manifest_df.to_csv(manifest_csv_path, index=False, encoding="utf-8")
    try:
        manifest_df.to_excel(manifest_xlsx_path, index=False)
        wrote_xlsx = True
    except Exception as e:
        wrote_xlsx = False
        build_report.append(("__manifest__", "xlsx_write_failed", repr(e)))
        pd.DataFrame(build_report, columns=["author_label", "issue", "details"]).to_csv(report_path, index=False, encoding="utf-8")

    print("Site generation complete.")
    print(f"Published individual author pages to: {author_dir}")
    print("No index page was generated.")
    print(f"Direct-link CSV manifest: {manifest_csv_path}")
    if wrote_xlsx:
        print(f"Direct-link Excel manifest: {manifest_xlsx_path}")
    else:
        print("Excel manifest could not be written. CSV manifest was still created.")
    print(f"Build report: {report_path}")
    print(f"Album cover assets copied to: {asset_dir}")
    print(f"Run completed at: {now_iso()}")


if __name__ == "__main__":
    main()
