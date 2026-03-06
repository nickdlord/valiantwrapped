#!/usr/bin/env python3
"""
build_headshot_manifest.py

Purpose:
Build a master manifest that links each author's REDCap headshot file to their
Scopus ID and corresponding music persona TXT file so the downstream image
generation pipeline can process authors in batch.

Inputs:
1. recordname_scopusID_filename.csv
   - contains record_id, first_name, last_name, scopus, and original photo name
2. author_headshots/documents/
   - contains exported REDCap headshots named like "1_photo.jpg"
3. outputs/author_music_personas_txt/
   - contains one persona TXT per author, with filename containing the Scopus ID

Output:
- outputs/headshot_generation_manifest.csv

What the manifest contains:
- author identity fields
- source headshot path
- persona TXT path
- output PNG path
- status fields for missing files / skipped rows

Run:
  python build_headshot_manifest.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/lordnd/valiant_wrapped/valiantwrapped")

LOOKUP_CSV = BASE_DIR / "author_headshots" / "recordname_scopusID_filename.csv"
HEADSHOT_DIR = BASE_DIR / "author_headshots" / "documents"
PERSONA_DIR = BASE_DIR / "outputs" / "author_music_personas_txt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = OUTPUT_DIR / "headshot_generation_manifest.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def clean_text(value: object) -> str:
    """Convert a value to a stripped string safely."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_scopus(value: object) -> str:
    """
    Normalize Scopus IDs as strings.
    Prevents things like 56830058500.0 from appearing.
    """
    s = clean_text(value)
    if not s:
        return ""
    # Remove trailing .0 if pandas interpreted as float
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def build_expected_headshot_name(record_id: str, original_photo_name: str) -> str:
    """
    Build the exported REDCap filename from record_id and original extension.
    Example:
      record_id=7, original=Landman_Bennett1.jpg -> 7_photo.jpg
    """
    suffix = Path(original_photo_name).suffix
    if not suffix:
        suffix = ".jpg"
    return f"{record_id}_photo{suffix}"


def find_persona_file_for_scopus(scopus_id: str, persona_dir: Path) -> Optional[Path]:
    """
    Find persona TXT by Scopus ID.
    Expected pattern examples:
      Kim_Michael_58290603100.txt
      Huo_Yuankai_56830058500.txt
    """
    matches = list(persona_dir.glob(f"*_{scopus_id}.txt"))
    if not matches:
        return None
    if len(matches) > 1:
        # Prefer exact-ending match if somehow duplicates exist
        exact = [p for p in matches if p.stem.endswith(f"_{scopus_id}")]
        if exact:
            return sorted(exact)[0]
    return sorted(matches)[0]


def parse_persona_file(persona_path: Path) -> Tuple[str, str]:
    """
    Extract artist and album from persona TXT.
    Returns (artist_name, album_title).
    """
    artist = ""
    album = ""

    try:
        text = persona_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return artist, album

    artist_match = re.search(r"^\s*Artist:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    album_match = re.search(r"^\s*Album:\s*(.+?)\s*$", text, flags=re.MULTILINE)

    if artist_match:
        artist = artist_match.group(1).strip()
    if album_match:
        album = album_match.group(1).strip()

    return artist, album


def make_output_basename(last_name: str, first_name: str, scopus_id: str) -> str:
    """
    Build stable output basename like:
      Kim_Michael_58290603100
    """
    def slug_part(s: str) -> str:
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"[^A-Za-z0-9_]+", "", s)
        return s

    return f"{slug_part(last_name)}_{slug_part(first_name)}_{scopus_id}"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if not LOOKUP_CSV.exists():
        raise FileNotFoundError(f"Lookup CSV not found: {LOOKUP_CSV}")
    if not HEADSHOT_DIR.exists():
        raise FileNotFoundError(f"Headshot directory not found: {HEADSHOT_DIR}")
    if not PERSONA_DIR.exists():
        raise FileNotFoundError(f"Persona directory not found: {PERSONA_DIR}")

    df = pd.read_csv(LOOKUP_CSV)

    required_cols = {"record_id", "first_name", "last_name", "scopus", "photo"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Lookup CSV is missing required columns: {sorted(missing_cols)}"
        )

    rows = []

    for _, row in df.iterrows():
        record_id = clean_text(row["record_id"])
        first_name = clean_text(row["first_name"])
        last_name = clean_text(row["last_name"])
        scopus_id = safe_scopus(row["scopus"])
        original_photo_name = clean_text(row["photo"])

        status = "ready"
        notes = []

        if not record_id:
            status = "skip"
            notes.append("missing_record_id")

        if not scopus_id:
            status = "skip"
            notes.append("missing_scopus_id")

        if not original_photo_name:
            status = "skip"
            notes.append("missing_original_photo_name")

        expected_headshot_name = ""
        headshot_path = None

        if record_id and original_photo_name:
            expected_headshot_name = build_expected_headshot_name(
                record_id, original_photo_name
            )
            candidate = HEADSHOT_DIR / expected_headshot_name
            if candidate.exists():
                headshot_path = candidate
            else:
                # Fallback: search any file that starts with "{record_id}_photo"
                fallback_matches = sorted(HEADSHOT_DIR.glob(f"{record_id}_photo.*"))
                if fallback_matches:
                    headshot_path = fallback_matches[0]
                    expected_headshot_name = headshot_path.name
                else:
                    status = "missing_headshot"
                    notes.append("headshot_not_found")

        persona_path = None
        artist_name = ""
        album_title = ""

        if scopus_id:
            persona_path = find_persona_file_for_scopus(scopus_id, PERSONA_DIR)
            if persona_path is None:
                if status == "ready":
                    status = "missing_persona"
                notes.append("persona_not_found")
            else:
                artist_name, album_title = parse_persona_file(persona_path)

        output_basename = ""
        output_png = ""

        if first_name and last_name and scopus_id:
            output_basename = make_output_basename(last_name, first_name, scopus_id)
            output_png = str((OUTPUT_DIR / "musician_headshots" / f"{output_basename}.png"))

        rows.append(
            {
                "record_id": record_id,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": f"{first_name} {last_name}".strip(),
                "scopus_id": scopus_id,
                "original_photo_name": original_photo_name,
                "expected_headshot_name": expected_headshot_name,
                "headshot_path": str(headshot_path) if headshot_path else "",
                "persona_file": str(persona_path) if persona_path else "",
                "artist_name": artist_name,
                "album_title": album_title,
                "output_basename": output_basename,
                "output_image": output_png,
                "status": status,
                "notes": ";".join(notes),
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_df.sort_values(
        by=["status", "last_name", "first_name", "record_id"],
        inplace=True,
        na_position="last",
    )
    manifest_df.to_csv(MANIFEST_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    total = len(manifest_df)
    ready = (manifest_df["status"] == "ready").sum()
    missing_headshot = (manifest_df["status"] == "missing_headshot").sum()
    missing_persona = (manifest_df["status"] == "missing_persona").sum()
    skipped = (manifest_df["status"] == "skip").sum()

    print(f"Manifest written to: {MANIFEST_CSV}")
    print(f"Total rows: {total}")
    print(f"Ready: {ready}")
    print(f"Missing headshot: {missing_headshot}")
    print(f"Missing persona: {missing_persona}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
