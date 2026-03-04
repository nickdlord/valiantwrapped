import os
import pandas as pd
from pathlib import Path
import html as html_lib

# ----------------------------
# PATHS (relative to this script, not your current working directory)
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent

SUMMARY_FILE = BASE_DIR / "author_summary_2025_present_test.csv"
PERSONA_FILE = BASE_DIR / "outputs" / "author_music_personas.csv"
EXPERTISE_DIR = BASE_DIR / "author_expertise_txt"

SITE_DIR = BASE_DIR / "site"
AUTHOR_DIR = SITE_DIR / "authors"
AUTHOR_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------

summary_df = pd.read_csv(SUMMARY_FILE)
persona_df = pd.read_csv(PERSONA_FILE)

build_report = []

# ----------------------------
# NORMALIZATION HELPERS
# ----------------------------


def canonical_author_label(value: str) -> str:
    """
    Convert values like:
      'Kim_Michael_58290603100.csv'
      'author_csvs/Kim_Michael_58290603100.csv'
    into:
      'Kim_Michael_58290603100'
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    s = os.path.basename(s)            # drop any folder path
    s = s.replace("\\", "/")           # safety on Windows paths
    if s.lower().endswith(".csv"):
        s = s[:-4]
    if s.lower().endswith(".txt"):
        s = s[:-4]
    return s.strip()


def parse_author_name(label: str):
    parts = label.split("_")
    last = parts[0] if len(parts) > 0 else ""
    first = parts[1] if len(parts) > 1 else ""
    return first, last


def format_tracklist(tracklist):
    if pd.isna(tracklist) or tracklist is None:
        return ""
    tracklist = str(tracklist).strip()
    if not tracklist:
        return ""

    for sep in [";", "|", ","]:
        if sep in tracklist:
            tracks = [t.strip() for t in tracklist.split(sep) if t.strip()]
            break
    else:
        tracks = [tracklist]

    items = "\n".join(f"<li>{html_lib.escape(t)}</li>" for t in tracks)
    return f"<ol>\n{items}\n</ol>"


def read_text_file_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

# ----------------------------
# PRE-NORMALIZE KEYS
# ----------------------------


# Canonical label for summary rows (derived from author_file)
summary_df["author_label"] = summary_df["author_file"].apply(
    canonical_author_label)

# Canonical label for persona rows (already base label, but normalize anyway)
if "author_label" not in persona_df.columns:
    raise ValueError("Persona CSV is missing required column: author_label")
persona_df["author_label"] = persona_df["author_label"].apply(
    canonical_author_label)

# Build quick lookup dicts for speed + robustness
summary_by_label = {row["author_label"]: row for _,
                    row in summary_df.iterrows() if row.get("author_label")}
persona_by_label = {row["author_label"]: row for _,
                    row in persona_df.iterrows() if row.get("author_label")}

# ----------------------------
# HTML STYLE
# ----------------------------

PAGE_STYLE = """
<style>
body {
    font-family: Arial, sans-serif;
    background:#0f172a;
    color:white;
    margin:0;
    padding:0;
}
.container {
    max-width:900px;
    margin:auto;
    padding:40px;
}
.section {
    background:#1e293b;
    padding:25px;
    margin-bottom:25px;
    border-radius:12px;
}
h1 {
    font-size:40px;
    margin-bottom: 8px;
}
h2 {
    color:#38bdf8;
    margin-top: 0px;
}
.stats {
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:15px;
}
.statbox {
    background:#334155;
    padding:15px;
    border-radius:10px;
    line-height: 1.35;
}
a { color:#38bdf8; }
</style>
"""

# ----------------------------
# PAGE GENERATION
# ----------------------------


def generate_author_page(author_label: str):
    author_label = canonical_author_label(author_label)
    first, last = parse_author_name(author_label)

    # -------- Summary --------
    row = summary_by_label.get(author_label)
    if row is None:
        build_report.append(
            (author_label, "missing_summary_row", f"searched label={author_label}"))
        summary_html = "<p>No summary data available.</p>"
    else:
        summary_html = f"""
        <div class="stats">
            <div class="statbox">
                Publications (2025–Present)<br>
                <b>{row.get("pub_count_2025_present", "")}</b>
            </div>
            <div class="statbox">
                Citations (2025–Present)<br>
                <b>{row.get("citation_count_2025_present", "")}</b>
            </div>
            <div class="statbox">
                Top Journal<br>
                <b>{row.get("top_journal_2025_present", "")}</b>
            </div>
            <div class="statbox">
                Top Paper<br>
                <b>{row.get("top_paper_title_2025_present", "")}</b><br>
                Citations: {row.get("top_paper_citations_2025_present", "")}
            </div>
        </div>
        """

    # -------- Expertise TXT --------
    expertise_path = EXPERTISE_DIR / f"{author_label}.txt"
    if expertise_path.exists():
        expertise_text = read_text_file_safely(expertise_path)
    else:
        expertise_text = "Expertise summary not found."
        build_report.append(
            (author_label, "missing_expertise_txt", f"expected {expertise_path}"))

    expertise_html = html_lib.escape(expertise_text).replace("\n", "<br>")

    # -------- Persona --------
    p = persona_by_label.get(author_label)
    if p is None:
        build_report.append(
            (author_label, "missing_persona_row", f"searched label={author_label}"))
        persona_html = "<p>No persona generated.</p>"
    else:
        status_val = str(p.get("status", "")).strip()
        if status_val and status_val.lower() != "ok":
            build_report.append((author_label, "persona_status", status_val))

        artist_name = html_lib.escape(str(p.get("artist_name", "") or ""))
        persona_bio = html_lib.escape(
            str(p.get("persona_bio", "") or "")).replace("\n", "<br>")
        album_title = html_lib.escape(str(p.get("album_title", "") or ""))
        tracklist_html = format_tracklist(p.get("tracklist", ""))

        persona_html = f"""
        <h3>{artist_name}</h3>
        <p>{persona_bio}</p>
        <h3>Album: {album_title}</h3>
        {tracklist_html}
        """

    # -------- Build Page --------
    page_html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{first} {last}</title>
{PAGE_STYLE}
</head>

<body>
<div class="container">
<h1>{first} {last}</h1>

<div class="section">
<h2>Introduction</h2>
<p>Dear {first},</p>
<p>The 2025–2026 academic year was truly unforgettable! Let's take a look at what you were up to:</p>
</div>

<div class="section">
<h2>2025–2026 Stats</h2>
{summary_html}
</div>

<div class="section">
<h2>Research Expertise</h2>
<p>{expertise_html}</p>
</div>

<div class="section">
<h2>Your Musical Persona</h2>
<p>
People like you are what make our discovery center so vibrant and unique.
The creativity and work that you do every day is truly instrumental to the
success of our discovery center.
</p>
<p>
So we thought it would only be fitting to come up with a unique musical
persona for you based on the entirety of your publishing history.
</p>
{persona_html}
</div>

<div class="section">
<h2>Thank You</h2>
<p>
Thank you for being part of our discovery center and for contributing to
another incredible year of innovation and collaboration.
</p>
</div>

</div>
</body>
</html>
"""

    output_path = AUTHOR_DIR / f"{author_label}.html"
    output_path.write_text(page_html, encoding="utf-8")

# ----------------------------
# GENERATE PAGES
# ----------------------------


authors = summary_df["author_label"].dropna().astype(str).unique()

for author_label in authors:
    generate_author_page(author_label)

# ----------------------------
# INDEX PAGE
# ----------------------------

links = ""
for a in authors:
    a = canonical_author_label(a)
    first, last = parse_author_name(a)
    links += f'<li><a href="authors/{a}.html">{first} {last}</a></li>\n'

index_html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VALIANT Wrapped</title>
{PAGE_STYLE}
</head>

<body>
<div class="container">
<h1>VALIANT Wrapped</h1>
<p>Internal index page (not intended for sharing). Use direct author links instead.</p>
<ul>
{links}
</ul>
</div>
</body>
</html>
"""

(SITE_DIR / "index.html").write_text(index_html, encoding="utf-8")

# ----------------------------
# BUILD REPORT
# ----------------------------

report_df = pd.DataFrame(build_report, columns=[
                         "author_label", "issue", "details"])
report_path = SITE_DIR / "build_report.csv"
report_df.to_csv(report_path, index=False, encoding="utf-8")

print("Site generation complete.")
print(f"Check {report_path} for issues.")
