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

def parse_author_name(label):
    parts = label.split("_")
    last = parts[0]
    first = parts[1]
    return first, last

def format_tracklist(tracklist):
    if pd.isna(tracklist):
        return ""

    for sep in [";", "|", ","]:
        if sep in tracklist:
            tracks = [t.strip() for t in tracklist.split(sep)]
            break
    else:
        tracks = [tracklist]

    formatted = ""
    for i, t in enumerate(tracks, 1):
        formatted += f"<li>{t}</li>\n"

    return f"<ol>{formatted}</ol>"

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
}

h2 {
    color:#38bdf8;
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
}

</style>
"""

def generate_author_page(author_label):

    first, last = parse_author_name(author_label)

    # ----------------------------
    # GET SUMMARY
    # ----------------------------

    summary_row = summary_df[summary_df["author_file"] == author_label]

    if summary_row.empty:
        build_report.append((author_label, "missing_summary"))
        summary_html = "<p>No summary data available.</p>"
    else:
        row = summary_row.iloc[0]

        summary_html = f"""
        <div class="stats">
            <div class="statbox">
                Publications (2025-present)<br>
                <b>{row.pub_count_2025_present}</b>
            </div>

            <div class="statbox">
                Citations (2025-present)<br>
                <b>{row.citation_count_2025_present}</b>
            </div>

            <div class="statbox">
                Top Journal<br>
                <b>{row.top_journal_2025_present}</b>
            </div>

            <div class="statbox">
                Top Paper<br>
                <b>{row.top_paper_title_2025_present}</b><br>
                Citations: {row.top_paper_citations_2025_present}
            </div>
        </div>
        """

    # ----------------------------
    # GET EXPERTISE TXT
    # ----------------------------

    expertise_path = os.path.join(EXPERTISE_DIR, author_label + ".txt")

    if os.path.exists(expertise_path):
        with open(expertise_path) as f:
            expertise_text = f.read()
    else:
        expertise_text = "Expertise summary not found."
        build_report.append((author_label, "missing_expertise_txt"))

    # ----------------------------
    # GET PERSONA
    # ----------------------------

    persona_row = persona_df[persona_df["author_label"] == author_label]

    if persona_row.empty:
        build_report.append((author_label, "missing_persona_row"))
        persona_html = "<p>No persona generated.</p>"
    else:
        p = persona_row.iloc[0]

        if p.status != "ok":
            build_report.append((author_label, p.status))

        tracklist_html = format_tracklist(p.tracklist)

        persona_html = f"""
        <h3>{p.artist_name}</h3>
        <p>{p.persona_bio}</p>

        <h3>Album: {p.album_title}</h3>

        {tracklist_html}
        """

    # ----------------------------
    # BUILD PAGE
    # ----------------------------

    html = f"""
<html>
<head>
<title>{first} {last}</title>
{PAGE_STYLE}
</head>

<body>

<div class="container">

<h1>{first} {last}</h1>

<div class="section">
<h2>Introduction</h2>

<p>Dear {first},</p>

<p>The 2025–2026 academic year was truly unforgettable!
Let's take a look at what you were up to.</p>
</div>

<div class="section">
<h2>2025–2026 Stats</h2>
{summary_html}
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

    with open(output_path, "w") as f:
        f.write(html)

# ----------------------------
# GENERATE PAGES
# ----------------------------

authors = summary_df["author_file"].unique()

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
<html>
<head>
<title>VALIANT Wrapped</title>
{PAGE_STYLE}
</head>

<body>

<div class="container">

<h1>VALIANT Wrapped</h1>

<ul>
{links}
</ul>

</div>

</body>
</html>
"""

with open(os.path.join(SITE_DIR, "index.html"), "w") as f:
    f.write(index_html)

# ----------------------------
# BUILD REPORT
# ----------------------------

report_df = pd.DataFrame(build_report, columns=["author_label","issue"])

report_df.to_csv(os.path.join(SITE_DIR,"build_report.csv"),index=False)

print("Site generation complete.")
print("Check site/build_report.csv for issues.")
