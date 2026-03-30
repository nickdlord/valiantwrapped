#!/usr/bin/env python3
"""
author_scopusmetrics_single.py

Takes one Scopus CSV for a single author and generates a readable TXT summary.

Run:
  python author_scopusmetrics_single.py --input-file my_scopus.csv

Output:
  author_summary.txt
"""

import os
import argparse
import pandas as pd


def pick_existing_col(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None


def safe_int_series(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def load_csv(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True, help="Path to Scopus CSV")
    ap.add_argument("--output-file", default="author_summary.txt", help="Output TXT file")
    ap.add_argument("--year-cutoff", type=int, default=2025)
    return ap


def main():
    args = build_parser().parse_args()

    input_file = args.input_file
    output_file = args.output_file
    year_cutoff = args.year_cutoff

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    filename = os.path.basename(input_file)
    author_id = os.path.splitext(filename)[0]

    df = load_csv(input_file)

    if df.empty:
        summary_text = f"""
Author ID: {author_id}

No publications were found in the provided file.
"""
    else:
        year_col = pick_existing_col(df.columns, ["Year", "Publication Year", "Pub. Year"])
        cites_col = pick_existing_col(df.columns, ["Cited by", "Citations", "Citation count"])
        journal_col = pick_existing_col(df.columns, ["Source title", "Journal", "Source Title"])
        title_col = pick_existing_col(df.columns, ["Title", "Document Title", "Article Title"])

        if year_col is None:
            raise ValueError("Could not find a Year column.")

        if cites_col is None:
            df["_citations"] = 0
            cites_col = "_citations"
        if journal_col is None:
            df["_journal"] = ""
            journal_col = "_journal"
        if title_col is None:
            df["_title"] = ""
            title_col = "_title"

        df["_year"] = safe_int_series(df[year_col])
        df["_cites"] = safe_int_series(df[cites_col])

        df_recent = df[df["_year"] >= year_cutoff].copy()

        pub_count = int(len(df_recent))
        cite_count = int(df_recent["_cites"].sum())

        # Top journal
        top_journal = ""
        if pub_count > 0:
            journal_stats = df_recent.groupby(journal_col, dropna=False)["_cites"].agg(
                pub_count="size",
                cite_sum="sum"
            ).reset_index()
            journal_stats = journal_stats.sort_values(
                ["pub_count", "cite_sum"], ascending=[False, False]
            )
            top_journal = str(journal_stats.iloc[0][journal_col]) if not journal_stats.empty else ""

        # Top paper
        top_paper_title = ""
        top_paper_cites = 0
        if pub_count > 0:
            df_recent_sorted = df_recent.sort_values(["_cites", "_year"], ascending=[False, False])
            top_paper_title = str(df_recent_sorted.iloc[0][title_col])
            top_paper_cites = int(df_recent_sorted.iloc[0]["_cites"])

        summary_text = f"""
==============================
VALIANT WRAPPED SUMMARY
==============================

Author ID: {author_id}

Publications (2025–Present): {pub_count}
Citations (2025–Present): {cite_count}

Top Journal:
{top_journal}

Top Paper:
{top_paper_title}

Top Paper Citations:
{top_paper_cites}
"""

    # Write TXT
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary_text.strip() + "\n")

    print(f"✅ Wrote summary: {output_file}")


if __name__ == "__main__":
    main()
