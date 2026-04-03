#!/usr/bin/env python3
"""
scopus2txtsummary.py

Convert Scopus CSV export(s) into per-author TXT files that contain:
1) high-level metrics
2) compact publication records

Supports:
- single mode via --input-file
- batch mode via --input-dir

Output:
- one TXT file per author in --output-dir
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import re
import glob
import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd

YEAR_COLS = ["Year", "Publication Year", "Pub. Year"]
TITLE_COLS = ["Title", "Document Title", "Article Title"]
ABSTRACT_COLS = ["Abstract", "Description"]
JOURNAL_COLS = ["Source title", "Source Title", "Journal"]
CITES_COLS = ["Cited by", "Citations", "Citation count"]
KEYWORD_COLS = ["Author Keywords", "Indexed Keywords", "Keywords"]
DOCTYPE_COLS = ["Document Type", "Doc Type", "Type"]

def pick_existing_col(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

def safe_int_series(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def clean_text(x):
    s = "" if x is None else str(x)
    return re.sub(r"\s+", " ", s).strip()

def truncate(s: str, max_chars: int) -> str:
    s = clean_text(s)
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + "..."

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)

def resolve_input_paths(input_file: str, input_dir: str) -> List[Path]:
    if bool(input_file) == bool(input_dir):
        raise ValueError("Provide exactly one of --input-file or --input-dir")
    if input_file:
        p = Path(input_file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return [p]
    pdir = Path(input_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"Input directory not found: {pdir}")
    paths = sorted(Path(x) for x in glob.glob(str(pdir / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {pdir}")
    return paths

def format_record(row: pd.Series, abstract_chars: int) -> str:
    year = int(row["_year"]) if row.get("_year") is not None else -1
    cites = int(row["_cites"]) if row.get("_cites") is not None else 0
    title = truncate(row.get("_title", ""), 180)
    journal = truncate(row.get("_journal", ""), 90)
    keywords = truncate(row.get("_keywords", ""), 180)
    doctype = truncate(row.get("_doctype", ""), 60)
    abstract = truncate(row.get("_abstract", ""), abstract_chars)
    lines = []
    if year != -1:
        lines.append(f"Year: {year}")
    lines.append(f"Citations: {cites}")
    if doctype:
        lines.append(f"Type: {doctype}")
    if journal:
        lines.append(f"Venue: {journal}")
    if title:
        lines.append(f"Title: {title}")
    if keywords:
        lines.append(f"Keywords: {keywords}")
    if abstract:
        lines.append(f"Abstract: {abstract}")
    return "\n".join(lines)

def summarize_one_file(path: Path, year_cutoff: int, abstract_chars: int) -> Tuple[str, str]:
    filename = path.name
    author_id = path.stem
    df = load_csv(path)
    if df.empty:
        return author_id, f"""AUTHOR_ID: {author_id}
SOURCE_FILE: {filename}

METRICS:
Publications ({year_cutoff}-Present): 0
Citations ({year_cutoff}-Present): 0
Top Journal:
Top Paper:
Top Paper Citations: 0

PUBLICATION_RECORDS:
(none)
"""
    year_col = pick_existing_col(df.columns, YEAR_COLS)
    cites_col = pick_existing_col(df.columns, CITES_COLS)
    journal_col = pick_existing_col(df.columns, JOURNAL_COLS)
    title_col = pick_existing_col(df.columns, TITLE_COLS)
    abstract_col = pick_existing_col(df.columns, ABSTRACT_COLS)
    keyword_col = pick_existing_col(df.columns, KEYWORD_COLS)
    doctype_col = pick_existing_col(df.columns, DOCTYPE_COLS)
    if year_col is None:
        raise ValueError(f"{filename}: Could not find a Year column.")
    if cites_col is None:
        df["_citations"] = 0; cites_col = "_citations"
    if journal_col is None:
        df["_journal_fallback"] = ""; journal_col = "_journal_fallback"
    if title_col is None:
        df["_title_fallback"] = ""; title_col = "_title_fallback"
    if abstract_col is None:
        df["_abstract_fallback"] = ""; abstract_col = "_abstract_fallback"
    if keyword_col is None:
        df["_keywords_fallback"] = ""; keyword_col = "_keywords_fallback"
    if doctype_col is None:
        df["_doctype_fallback"] = ""; doctype_col = "_doctype_fallback"

    df["_year"] = safe_int_series(df[year_col]); df["_cites"] = safe_int_series(df[cites_col])
    df["_title"] = df[title_col].fillna(""); df["_abstract"] = df[abstract_col].fillna("")
    df["_journal"] = df[journal_col].fillna(""); df["_keywords"] = df[keyword_col].fillna("")
    df["_doctype"] = df[doctype_col].fillna("")
    df_recent = df[df["_year"] >= year_cutoff].copy()

    pub_count = int(len(df_recent)); cite_count = int(df_recent["_cites"].sum())
    top_journal = ""
    if pub_count > 0:
        journal_stats = (
            df_recent.groupby(journal_col, dropna=False)["_cites"]
            .agg(pub_count="size", cite_sum="sum").reset_index()
            .sort_values(["pub_count", "cite_sum"], ascending=[False, False])
        )
        top_journal = clean_text(journal_stats.iloc[0][journal_col]) if not journal_stats.empty else ""
    top_paper_title = ""; top_paper_cites = 0
    if pub_count > 0:
        df_recent_sorted = df_recent.sort_values(["_cites", "_year"], ascending=[False, False])
        top_paper_title = clean_text(df_recent_sorted.iloc[0][title_col])
        top_paper_cites = int(df_recent_sorted.iloc[0]["_cites"])

    df_records = df.sort_values(["_cites", "_year"], ascending=[False, False]).copy()
    records_text = "\n\n---\n\n".join(format_record(r, abstract_chars) for _, r in df_records.iterrows()) if len(df_records) else "(none)"
    text = f"""AUTHOR_ID: {author_id}
SOURCE_FILE: {filename}

METRICS:
Publications ({year_cutoff}-Present): {pub_count}
Citations ({year_cutoff}-Present): {cite_count}
Top Journal: {top_journal}
Top Paper: {top_paper_title}
Top Paper Citations: {top_paper_cites}

PUBLICATION_RECORDS:
{records_text}
"""
    return author_id, text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="")
    ap.add_argument("--input-dir", default="")
    ap.add_argument("--output-dir", default="outputs/summary_txt")
    ap.add_argument("--year-cutoff", type=int, default=2025)
    ap.add_argument("--abstract-chars", type=int, default=400)
    args = ap.parse_args()

    paths = resolve_input_paths(args.input_file, args.input_dir)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        author_id, text = summarize_one_file(path, args.year_cutoff, args.abstract_chars)
        out_path = output_dir / f"{author_id}.txt"
        out_path.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
