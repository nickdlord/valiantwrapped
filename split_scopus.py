#!/usr/bin/env python3
"""
split_scopus_master_by_roster.py

Split a master Scopus export into one CSV per center author using a roster CSV.

Expected roster columns:
- first_name
- last_name
- scopus

Expected master Scopus column:
- Author(s) ID
  (fallbacks supported for some alternate Scopus-like exports)

Output filenames follow the canonical pipeline naming convention:
- Last_First_ScopusID.csv

A publication can correctly appear in multiple author CSVs when multiple center
authors coauthored the same paper.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

AUTHOR_ID_COL_CANDIDATES = [
    "Author(s) ID",
    "Authors with affiliations",
    "Authors",
    "Author Names",
]

REQUIRED_ROSTER_COLS = ["first_name", "last_name", "scopus"]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = str(path)
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def pick_existing_col(columns, candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_scopus_id(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    ids = re.findall(r"\d+", text)
    return ids[0] if ids else ""


def safe_filename_part(text: str) -> str:
    text = clean_text(text)
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9 _-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def canonical_author_label(last_name: str, first_name: str, scopus_id: str) -> str:
    return f"{safe_filename_part(last_name)}_{safe_filename_part(first_name)}_{normalize_scopus_id(scopus_id)}"


def parse_author_ids(raw: object) -> List[str]:
    text = clean_text(raw)
    if not text:
        return []
    ids = re.findall(r"\d+", text)
    seen: Set[str] = set()
    ordered: List[str] = []
    for scopus_id in ids:
        if scopus_id not in seen:
            seen.add(scopus_id)
            ordered.append(scopus_id)
    return ordered


def build_roster_map(roster_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    missing = [c for c in REQUIRED_ROSTER_COLS if c not in roster_df.columns]
    if missing:
        raise ValueError(
            f"Roster file is missing required columns: {missing}. "
            f"Expected at least: {REQUIRED_ROSTER_COLS}"
        )

    roster_map: Dict[str, Dict[str, str]] = {}
    duplicate_ids: Set[str] = set()
    duplicate_labels: Set[str] = set()
    labels_seen: Set[str] = set()

    for _, row in roster_df.iterrows():
        first_name = clean_text(row.get("first_name", ""))
        last_name = clean_text(row.get("last_name", ""))
        scopus_id = normalize_scopus_id(row.get("scopus", ""))

        if not scopus_id:
            continue

        label = canonical_author_label(last_name, first_name, scopus_id)

        if scopus_id in roster_map:
            duplicate_ids.add(scopus_id)
        if label in labels_seen:
            duplicate_labels.add(label)

        labels_seen.add(label)
        roster_map[scopus_id] = {
            "first_name": first_name,
            "last_name": last_name,
            "scopus": scopus_id,
            "label": label,
        }

    if duplicate_ids:
        raise ValueError(
            "Duplicate Scopus IDs found in roster: " + ", ".join(sorted(duplicate_ids))
        )
    if duplicate_labels:
        raise ValueError(
            "Duplicate canonical author labels found in roster: " + ", ".join(sorted(duplicate_labels))
        )
    if not roster_map:
        raise ValueError("No valid roster entries with Scopus IDs were found.")

    return roster_map


def split_master_by_roster(
    master_csv: str | Path,
    roster_csv: str | Path,
    output_dir: str | Path,
    clear_output_dir: bool = False,
    write_empty_files: bool = False,
) -> None:
    master_df = load_csv(master_csv)
    roster_df = load_csv(roster_csv)

    author_id_col = pick_existing_col(master_df.columns, AUTHOR_ID_COL_CANDIDATES)
    if not author_id_col:
        raise ValueError(
            f"Could not find an author ID column in the master export. Tried: {AUTHOR_ID_COL_CANDIDATES}"
        )

    roster_map = build_roster_map(roster_df)
    roster_ids = set(roster_map.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if clear_output_dir:
        for old_csv in output_path.glob("*.csv"):
            old_csv.unlink()

    matched_rows: Dict[str, List[dict]] = defaultdict(list)
    unmatched_rows = 0
    multi_match_rows = 0

    for _, row in master_df.iterrows():
        row_ids = set(parse_author_ids(row.get(author_id_col, "")))
        matched_ids = sorted(roster_ids.intersection(row_ids))

        if not matched_ids:
            unmatched_rows += 1
            continue

        if len(matched_ids) > 1:
            multi_match_rows += 1

        row_dict = row.to_dict()
        for scopus_id in matched_ids:
            matched_rows[scopus_id].append(row_dict)

    files_written = 0
    empty_authors = 0

    for scopus_id, author_meta in roster_map.items():
        rows = matched_rows.get(scopus_id, [])
        if not rows and not write_empty_files:
            empty_authors += 1
            continue

        out_file = output_path / f"{author_meta['label']}.csv"
        out_df = pd.DataFrame(rows, columns=master_df.columns) if rows else pd.DataFrame(columns=master_df.columns)
        out_df.to_csv(out_file, index=False, encoding="utf-8")
        files_written += 1

    print(f"Master rows scanned: {len(master_df)}")
    print(f"Roster authors found: {len(roster_map)}")
    print(f"Author CSVs written: {files_written}")
    print(f"Roster authors with no matched papers: {empty_authors}")
    print(f"Rows with no roster-author match: {unmatched_rows}")
    print(f"Rows matched to multiple roster authors: {multi_match_rows}")
    print(f"Author ID column used: {author_id_col}")
    print(f"Output folder: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", required=True, help="Path to master Scopus export CSV")
    ap.add_argument("--roster-csv", required=True, help="Path to roster CSV (e.g., scopusIDlist.csv)")
    ap.add_argument("--output-dir", default="author_csvs", help="Folder for per-author CSV outputs")
    ap.add_argument(
        "--clear-output-dir",
        action="store_true",
        help="Delete existing CSVs in the output directory before writing new ones",
    )
    ap.add_argument(
        "--write-empty-files",
        action="store_true",
        help="Also write empty CSVs for roster authors who have no matched papers",
    )
    args = ap.parse_args()

    split_master_by_roster(
        master_csv=args.master_csv,
        roster_csv=args.roster_csv,
        output_dir=args.output_dir,
        clear_output_dir=args.clear_output_dir,
        write_empty_files=args.write_empty_files,
    )


if __name__ == "__main__":
    main()
