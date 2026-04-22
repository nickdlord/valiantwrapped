# 🎧 VALIANT Wrapped

VALIANT Wrapped is a pipeline and web-based tool that transforms academic publication data into personalized, Spotify Wrapped–style researcher profiles.

## Features
- AI-generated expertise summaries
- Music personas
- Album cover generation
- Musician headshots
- Static website generation

## Installation
git clone https://github.com/nickdlord/valiantwrapped.git
cd valiantwrapped
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run GUI
python solorun_gui.py

Then open http://localhost:5000

## Batch Pipeline
python run_valiantwrapped_per_author_nosite.py \
  --project-root . \
  --author-csv-dir author_csvs \
  --expertise-txt-dir outputs/author_expertise_txt \
  --persona-txt-dir outputs/author_music_personas_txt \
  --album-covers-dir outputs/album_covers \
  --reports-dir pipeline_reports \
  --skip-existing

## View Site
cd docs
python -m http.server 8000
