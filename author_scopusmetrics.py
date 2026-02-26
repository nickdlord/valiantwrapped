import os
import glob
import pandas as pd

# ----------------------------
# User settings
# ----------------------------
INPUT_DIR = "author_csvs"          # folder containing your per-author CSVs
OUTPUT_SUMMARY = "author_summary_2025_present.csv"
YEAR_CUTOFF = 2025

# ----------------------------
# Helper functions
# ----------------------------


def pick_existing_col(df_cols, candidates):
    """Return the first candidate column that exists in df_cols, else None."""
    for c in candidates:
        if c in df_cols:
            return c
    return None


def safe_int_series(s):
    """Convert a pandas Series to integers safely (coerce errors to 0)."""
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def main():
    csv_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in folder: {INPUT_DIR}")

    rows = []

    for path in csv_paths:
        filename = os.path.basename(path)
        # e.g., "AU_12345678900" or whatever your file is named
        author_id = os.path.splitext(filename)[0]

        try:
            df = pd.read_csv(
                path, dtype=str, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            # fallback if some files have non-utf8 characters
            df = pd.read_csv(
                path, dtype=str, encoding="latin-1", low_memory=False)

        if df.empty:
            rows.append({
                "author_file": filename,
                "author_id": author_id,
                "pub_count_2025_present": 0,
                "citation_count_2025_present": 0,
                "top_journal_2025_present": "",
                "top_paper_title_2025_present": "",
                "top_paper_citations_2025_present": 0,
            })
            continue

        # Column name guesses (Scopus exports can vary)
        year_col = pick_existing_col(
            df.columns, ["Year", "Publication Year", "Pub. Year"])
        cites_col = pick_existing_col(
            df.columns, ["Cited by", "Citations", "Citation count"])
        journal_col = pick_existing_col(
            df.columns, ["Source title", "Journal", "Source Title"])
        title_col = pick_existing_col(
            df.columns, ["Title", "Document Title", "Article Title"])

        if year_col is None:
            raise ValueError(
                f"{filename}: Could not find a Year column (tried Year / Publication Year / Pub. Year).")
        if cites_col is None:
            # If missing, treat citations as 0
            df["_citations"] = 0
            cites_col = "_citations"
        if journal_col is None:
            df["_journal"] = ""
            journal_col = "_journal"
        if title_col is None:
            df["_title"] = ""
            title_col = "_title"

        # Clean year + citations
        df["_year"] = safe_int_series(df[year_col])
        df["_cites"] = safe_int_series(df[cites_col])

        # Filter to 2025-present
        df_recent = df[df["_year"] >= YEAR_CUTOFF].copy()

        pub_count = int(len(df_recent))
        cite_count = int(df_recent["_cites"].sum())

        # Top journal: most publications; tie-breaker = most citations
        top_journal = ""
        if pub_count > 0:
            journal_stats = (
                df_recent.groupby(journal_col, dropna=False)
                .agg(pub_count=("__dummy__", "size") if "__dummy__" in df_recent.columns else (journal_col, "size"),
                     cite_sum=("_cites", "sum"))
                .reset_index()
            )
            # If groupby trick above complains, use a simpler form:
            # journal_stats = df_recent.groupby(journal_col, dropna=False)["_cites"].agg(pub_count="size", cite_sum="sum").reset_index()

            journal_stats = df_recent.groupby(journal_col, dropna=False)["_cites"].agg(
                pub_count="size", cite_sum="sum").reset_index()
            journal_stats = journal_stats.sort_values(
                ["pub_count", "cite_sum"], ascending=[False, False])
            top_journal = str(
                journal_stats.iloc[0][journal_col]) if not journal_stats.empty else ""

        # Top paper: highest citations; tie-breaker = most recent year
        top_paper_title = ""
        top_paper_cites = 0
        if pub_count > 0:
            df_recent_sorted = df_recent.sort_values(
                ["_cites", "_year"], ascending=[False, False])
            top_paper_title = str(df_recent_sorted.iloc[0][title_col])
            top_paper_cites = int(df_recent_sorted.iloc[0]["_cites"])

        rows.append({
            "author_file": filename,
            "author_id": author_id,
            "pub_count_2025_present": pub_count,
            "citation_count_2025_present": cite_count,
            "top_journal_2025_present": top_journal,
            "top_paper_title_2025_present": top_paper_title,
            "top_paper_citations_2025_present": top_paper_cites,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")
    print(f"✅ Wrote summary: {OUTPUT_SUMMARY}")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
