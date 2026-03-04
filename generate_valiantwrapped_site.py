import os
import pandas as pd
from pathlib import Path

# ----------------------------
# INPUT FILES
# ----------------------------

SUMMARY_FILE = "author_summary_2025_present.csv"
PERSONA_FILE = "outputs/author_music_personas.csv"
EXPERTISE_DIR = "author_expertise_txt"

# ----------------------------
# OUTPUT
# ----------------------------

SITE_DIR = "site"
AUTHOR_DIR = os.path.join(SITE_DIR, "authors")

os.makedirs(AUTHOR_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------

summary_df = pd.read_csv(SUMMARY_FILE)
persona_df = pd.read_csv(PERSONA_FILE)

build_report = []

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------


def parse_author_name(label: str):
    parts = label.split("_")
    # Expected: Last_First_ID
    last = parts[0] if len(parts) > 0 else ""
    first = parts[1] if len(parts) > 1 else ""
    return first, last


def format_tracklist(tracklist):
    """Convert tracklist string into <ol><li>...</li></ol>."""
    if pd.isna(tracklist) or tracklist is None:
        return ""

    tracklist = str(tracklist).strip()
    if not tracklist:
        return ""

    # Try common separators; fall back to single item
    for sep in [";", "|", ","]:
        if sep in tracklist:
            tracks = [t.strip() for t in tracklist.split(sep) if t.strip()]
            break
    else:
        tracks = [tracklist]

    items = "\n".join(f"<li>{t}</li>" for t in tracks)
    return f"<ol>\n{items}\n</ol>"


def read_text_file_safely(path: str) -> str:
    """Read a text file as UTF-8, fallback to latin-1."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

# ----------------------------
# HTML TEMPLATES
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


def generate_author_page(author_label: str):
    first, last = parse_author_name(author_label)

    # ----------------------------
    # GET SUMMARY
    # ----------------------------
    # NOTE: your summary CSV header says: author_file, author_id, ...
    summary_row = summary_df[summary_df["author_file"] == author_label]

    if summary_row.empty:
        build_report.append(
            (author_label, "missing_summary_row_in_author_summary_2025_present"))
        summary_html = "<p>No summary data available.</p>"
    else:
        row = summary_row.iloc[0]
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

    # ----------------------------
    # GET EXPERTISE TXT
    # ----------------------------
    expertise_path = os.path.join(EXPERTISE_DIR, author_label + ".txt")

    if os.path.exists(expertise_path):
        expertise_text = read_text_file_safely(expertise_path)
    else:
        expertise_text = "Expertise summary not found."
        build_report.append((author_label, "missing_expertise_txt"))

    # ----------------------------
    # GET PERSONA
    # ----------------------------
    persona_row = persona_df[persona_df["author_label"] == author_label]

    if persona_row.empty:
        build_report.append(
            (author_label, "missing_persona_row_in_author_music_personas"))
        persona_html = "<p>No persona generated.</p>"
    else:
        p = persona_row.iloc[0]

        status_val = str(p.get("status", "")).strip()
        if status_val and status_val.lower() != "ok":
            build_report.append((author_label, f"persona_status:{status_val}"))

        artist_name = p.get("artist_name", "")
        persona_bio = p.get("persona_bio", "")
        album_title = p.get("album_title", "")
        tracklist = p.get("tracklist", "")

        tracklist_html = format_tracklist(tracklist)

        persona_html = f"""
        <h3>{artist_name}</h3>
        <p>{persona_bio}</p>

        <h3>Album: {album_title}</h3>
        {tracklist_html}
        """

    # ----------------------------
    # BUILD PAGE
    # ----------------------------
    # Include UTF-8 meta tag so browsers render special characters correctly
    html = f"""
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
<p>{expertise_text.replace("\n", "<br>")}</p>
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

    output_path = os.path.join(AUTHOR_DIR, author_label + ".html")
    # IMPORTANT: Write HTML as UTF-8 on Windows
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

# ----------------------------
# GENERATE PAGES
# ----------------------------


# Build author list primarily from summary file; later we can expand to union-of-sources if you want
authors = summary_df["author_file"].dropna().astype(str).unique()

for author in authors:
    generate_author_page(author)

# ----------------------------
# INDEX PAGE
# ----------------------------

links = ""
for a in authors:
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

with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(index_html)

# ----------------------------
# BUILD REPORT
# ----------------------------

report_df = pd.DataFrame(build_report, columns=["author_label", "issue"])
report_path = os.path.join(SITE_DIR, "build_report.csv")

# IMPORTANT: Write CSV as UTF-8
report_df.to_csv(report_path, index=False, encoding="utf-8")

print("Site generation complete.")
print(f"Check {report_path} for issues.")
