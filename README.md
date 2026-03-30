# VALIANT Wrapped Localhost GUI

This is a cross-platform localhost web app for running the VALIANT Wrapped pipeline on a user's own computer.

## What it does

- Upload one or more Scopus CSV files
- Run the pipeline in single or batch mode
- Show live status updates
- Preview generated HTML author pages
- Download a ZIP of results
- Generate and download a manifest CSV for emailing teams

## Expected scripts in the same folder

- `scopus2txtsummary.py`
- `author_expertise_llama31_2.py`
- `author_persona_llama31.py`
- `generate_album_covers.py`
- `generate_valiantwrapped_site_noindex.py`
- `build_author_url_manifest.py`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Notes

- Uploaded CSVs are stored in a temporary run folder.
- Generated outputs for each run are isolated in a unique temp directory.
- The app serves generated author pages locally for preview.
- The ZIP download contains the generated `docs/` pages and album cover outputs.
