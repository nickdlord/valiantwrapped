#!/usr/bin/env python3
"""
split_scopus_hybrid.py

Split a master Scopus export into one CSV per center author using a roster CSV.
Supports matching by either Scopus Author ID or ORCID from the roster.

Expected roster columns:
- first_name
- last_name
- scopus   (optional per row)
- orcid    (optional per row)
At least one of scopus/orcid must be present for a given author row.

Expected master Scopus columns:
- Scopus author identifiers from one of:
    * Author(s) ID
    * Authors with affiliations
    * Authors
    * Author Names
- ORCID identifiers from any column whose header contains "orcid"
  plus, when present, "Authors with affiliations" as a fallback text field.

Output filenames follow the canonical pipeline naming convention as closely as possible:
- Last_First_ScopusID.csv               when a Scopus ID exists
- Last_First_ORCID_0000_0000_....csv    when only ORCID exists

A publication can correctly appear in multiple author CSVs when multiple center
authors coauthored the same paper.
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

SCOPUS_AUTHOR_ID_COL_CANDIDATES = [
    "Author(s) ID",
    "Authors with affiliations",
    "Authors",
    "Author Names",
]

REQUIRED_NAME_COLS = ["first_name", "last_name"]
OPTIONAL_ID_COLS = ["scopus", "orcid"]
ORCID_PATTERN = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dXx]\b")


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

    # Treat common placeholder / missing-value forms as empty.
    text_lower = text.lower()
    if text_lower in {"0", "0.0", "nan", "none", "null", "na", "n/a", "missing"}:
        return ""

    if ORCID_PATTERN.search(text):
        # Prevent ORCIDs accidentally stored in the Scopus column from collapsing to 0000/0009.
        return ""

    ids = re.findall(r"\d{8,}", text)
    if ids:
        candidate = ids[0].lstrip("0")
        return candidate if candidate else ""

    ids = re.findall(r"\d+", text)
    if not ids:
        return ""

    candidate = ids[0].lstrip("0")
    return candidate if candidate else ""


def normalize_orcid(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    m = ORCID_PATTERN.search(text)
    if m:
        return m.group(0).upper()

    compact = re.sub(r"[^0-9Xx]", "", text)
    if len(compact) == 16:
        return f"{compact[0:4]}-{compact[4:8]}-{compact[8:12]}-{compact[12:16]}".upper()
    return ""


def safe_filename_part(text: str) -> str:
    text = clean_text(text)
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9 _-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def label_id_part(scopus_id: str, orcid: str) -> str:
    if scopus_id:
        return normalize_scopus_id(scopus_id)
    if orcid:
        return "ORCID_" + normalize_orcid(orcid).replace("-", "_")
    return "UNKNOWN_ID"


def canonical_author_label(last_name: str, first_name: str, scopus_id: str = "", orcid: str = "") -> str:
    return f"{safe_filename_part(last_name)}_{safe_filename_part(first_name)}_{label_id_part(scopus_id, orcid)}"


def parse_scopus_ids(raw: object) -> List[str]:
    text = clean_text(raw)
    if not text:
        return []
    ids = re.findall(r"\d{8,}", text)
    if not ids:
        ids = re.findall(r"\d+", text)
    seen: Set[str] = set()
    ordered: List[str] = []
    for scopus_id in ids:
        if scopus_id not in seen:
            seen.add(scopus_id)
            ordered.append(scopus_id)
    return ordered


def parse_orcids(raw: object) -> List[str]:
    text = clean_text(raw)
    if not text:
        return []
    found = [m.upper() for m in ORCID_PATTERN.findall(text)]
    seen: Set[str] = set()
    ordered: List[str] = []
    for orcid in found:
        if orcid not in seen:
            seen.add(orcid)
            ordered.append(orcid)
    return ordered


def build_roster_entries(roster_df: pd.DataFrame) -> List[Dict[str, str]]:
    missing = [c for c in REQUIRED_NAME_COLS if c not in roster_df.columns]
    if missing:
        raise ValueError(
            f"Roster file is missing required columns: {missing}. "
            f"Expected at least: {REQUIRED_NAME_COLS + OPTIONAL_ID_COLS}"
        )

    if not any(c in roster_df.columns for c in OPTIONAL_ID_COLS):
        raise ValueError(
            "Roster file must contain at least one identifier column: 'scopus' and/or 'orcid'."
        )

    entries: List[Dict[str, str]] = []
    duplicate_scopus_ids: Set[str] = set()
    duplicate_orcids: Set[str] = set()
    duplicate_labels: Set[str] = set()
    seen_scopus: Set[str] = set()
    seen_orcids: Set[str] = set()
    seen_labels: Set[str] = set()

    for _, row in roster_df.iterrows():
        first_name = clean_text(row.get("first_name", ""))
        last_name = clean_text(row.get("last_name", ""))
        scopus_id = normalize_scopus_id(row.get("scopus", "")) if "scopus" in roster_df.columns else ""
        orcid = normalize_orcid(row.get("orcid", "")) if "orcid" in roster_df.columns else ""

        if not (scopus_id or orcid):
            continue

        label = canonical_author_label(last_name, first_name, scopus_id=scopus_id, orcid=orcid)

        if scopus_id:
            if scopus_id in seen_scopus:
                duplicate_scopus_ids.add(scopus_id)
            seen_scopus.add(scopus_id)

        if orcid:
            if orcid in seen_orcids:
                duplicate_orcids.add(orcid)
            seen_orcids.add(orcid)

        if label in seen_labels:
            duplicate_labels.add(label)
        seen_labels.add(label)

        entries.append(
            {
                "first_name": first_name,
                "last_name": last_name,
                "scopus": scopus_id,
                "orcid": orcid,
                "label": label,
            }
        )

    if duplicate_scopus_ids:
        raise ValueError(
            "Duplicate Scopus IDs found in roster: " + ", ".join(sorted(duplicate_scopus_ids))
        )
    if duplicate_orcids:
        raise ValueError(
            "Duplicate ORCIDs found in roster: " + ", ".join(sorted(duplicate_orcids))
        )
    if duplicate_labels:
        raise ValueError(
            "Duplicate canonical author labels found in roster: " + ", ".join(sorted(duplicate_labels))
        )
    if not entries:
        raise ValueError("No valid roster entries with Scopus IDs and/or ORCIDs were found.")

    return entries


def build_lookup_maps(roster_entries: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    scopus_map: Dict[str, Dict[str, str]] = {}
    orcid_map: Dict[str, Dict[str, str]] = {}
    for entry in roster_entries:
        if entry["scopus"]:
            scopus_map[entry["scopus"]] = entry
        if entry["orcid"]:
            orcid_map[entry["orcid"]] = entry
    return scopus_map, orcid_map


def detect_orcid_columns(master_df: pd.DataFrame) -> List[str]:
    cols = [c for c in master_df.columns if "orcid" in c.lower()]
    if "Authors with affiliations" in master_df.columns and "Authors with affiliations" not in cols:
        cols.append("Authors with affiliations")
    return cols


def split_master_by_roster(
    master_csv: str | Path,
    roster_csv: str | Path,
    output_dir: str | Path,
    clear_output_dir: bool = False,
    write_empty_files: bool = False,
) -> None:
    master_df = load_csv(master_csv)
    roster_df = load_csv(roster_csv)

    scopus_author_id_col = pick_existing_col(master_df.columns, SCOPUS_AUTHOR_ID_COL_CANDIDATES)
    if not scopus_author_id_col:
        raise ValueError(
            f"Could not find a Scopus author ID column in the master export. Tried: {SCOPUS_AUTHOR_ID_COL_CANDIDATES}"
        )

    orcid_cols = detect_orcid_columns(master_df)

    roster_entries = build_roster_entries(roster_df)
    scopus_map, orcid_map = build_lookup_maps(roster_entries)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if clear_output_dir:
        for old_csv in output_path.glob("*.csv"):
            old_csv.unlink()

    matched_rows: Dict[str, List[dict]] = defaultdict(list)
    unmatched_rows = 0
    multi_match_rows = 0
    scopus_matched_rows = 0
    orcid_matched_rows = 0
    both_matched_rows = 0

    for _, row in master_df.iterrows():
        matched_labels: Set[str] = set()
        matched_by_scopus = False
        matched_by_orcid = False

        row_scopus_ids = set(parse_scopus_ids(row.get(scopus_author_id_col, "")))
        for scopus_id in sorted(row_scopus_ids):
            entry = scopus_map.get(scopus_id)
            if entry:
                matched_labels.add(entry["label"])
                matched_by_scopus = True

        row_orcids: Set[str] = set()
        for col in orcid_cols:
            row_orcids.update(parse_orcids(row.get(col, "")))
        for orcid in sorted(row_orcids):
            entry = orcid_map.get(orcid)
            if entry:
                matched_labels.add(entry["label"])
                matched_by_orcid = True

        if not matched_labels:
            unmatched_rows += 1
            continue

        if matched_by_scopus and matched_by_orcid:
            both_matched_rows += 1
        elif matched_by_scopus:
            scopus_matched_rows += 1
        elif matched_by_orcid:
            orcid_matched_rows += 1

        if len(matched_labels) > 1:
            multi_match_rows += 1

        row_dict = row.to_dict()
        for label in sorted(matched_labels):
            matched_rows[label].append(row_dict)

    files_written = 0
    empty_authors = 0

    for entry in roster_entries:
        rows = matched_rows.get(entry["label"], [])
        if not rows and not write_empty_files:
            empty_authors += 1
            continue

        out_file = output_path / f"{entry['label']}.csv"
        out_df = pd.DataFrame(rows, columns=master_df.columns) if rows else pd.DataFrame(columns=master_df.columns)
        out_df.to_csv(out_file, index=False, encoding="utf-8")
        files_written += 1

    scopus_authors = sum(1 for e in roster_entries if e["scopus"])
    orcid_authors = sum(1 for e in roster_entries if e["orcid"])
    orcid_only_authors = sum(1 for e in roster_entries if e["orcid"] and not e["scopus"])

    print(f"Master rows scanned: {len(master_df)}")
    print(f"Roster authors found: {len(roster_entries)}")
    print(f"Roster authors with Scopus IDs: {scopus_authors}")
    print(f"Roster authors with ORCIDs: {orcid_authors}")
    print(f"Roster authors with ORCID only: {orcid_only_authors}")
    print(f"Author CSVs written: {files_written}")
    print(f"Roster authors with no matched papers: {empty_authors}")
    print(f"Rows with no roster-author match: {unmatched_rows}")
    print(f"Rows matched to multiple roster authors: {multi_match_rows}")
    print(f"Rows matched by Scopus only: {scopus_matched_rows}")
    print(f"Rows matched by ORCID only: {orcid_matched_rows}")
    print(f"Rows matched by both Scopus and ORCID: {both_matched_rows}")
    print(f"Scopus author ID column used: {scopus_author_id_col}")
    print(f"ORCID columns used: {orcid_cols if orcid_cols else '(none detected)'}")
    print(f"Output folder: {output_path}")

    if orcid_map and not orcid_cols:
        print(
            "WARNING: Roster contains ORCID values, but no ORCID-like columns were detected in the master export. "
            "ORCID-only authors will not match unless the master CSV contains ORCID data."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", required=True, help="Path to master Scopus export CSV")
    ap.add_argument("--roster-csv", required=True, help="Path to roster CSV with first_name/last_name and scopus/orcid")
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
