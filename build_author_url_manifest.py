#!/usr/bin/env python3
"""
build_author_url_manifest.py

Create a manifest file listing each author's:
- first name
- last name
- Scopus ID
- author label
- URL to their HTML page

Works from the generated HTML files in docs/authors/.

Examples:
  python build_author_url_manifest.py \
    --authors-dir docs/authors \
    --output-file docs/author_url_manifest.csv \
    --base-url https://nickdlord.github.io/valiantwrapped

  python build_author_url_manifest.py \
    --authors-dir docs/authors \
    --output-file docs/author_url_manifest.csv
"""

import os
import re
import csv
import glob
import argparse


def parse_author_label(label: str):
    """
    Expected format:
      Last_First_ScopusID
    Example:
      Kim_Michael_58290603100
    """
    parts = label.split("_")

    if len(parts) < 3:
        last_name = parts[0] if len(parts) > 0 else ""
        first_name = parts[1] if len(parts) > 1 else ""
        scopus_id = ""
        return first_name, last_name, scopus_id

    last_name = parts[0]
    first_name = parts[1]
    scopus_id = parts[-1]
    return first_name, last_name, scopus_id


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") if base_url else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--authors-dir",
        default="docs/authors",
        help="Folder containing per-author HTML pages",
    )
    ap.add_argument(
        "--output-file",
        default="docs/author_url_manifest.csv",
        help="Output CSV file path",
    )
    ap.add_argument(
        "--base-url",
        default="",
        help="Optional public base URL, e.g. https://nickdlord.github.io/valiantwrapped",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.authors_dir):
        raise FileNotFoundError(f"Authors directory not found: {args.authors_dir}")

    base_url = normalize_base_url(args.base_url)
    html_paths = sorted(glob.glob(os.path.join(args.authors_dir, "*.html")))

    if not html_paths:
        raise FileNotFoundError(f"No HTML files found in: {args.authors_dir}")

    rows = []

    for path in html_paths:
        filename = os.path.basename(path)
        author_label = os.path.splitext(filename)[0]

        first_name, last_name, scopus_id = parse_author_label(author_label)

        relative_url = f"authors/{filename}"
        full_url = f"{base_url}/{relative_url}" if base_url else relative_url

        rows.append({
            "first_name": first_name,
            "last_name": last_name,
            "scopus_id": scopus_id,
            "author_label": author_label,
            "url": full_url,
        })

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["first_name", "last_name", "scopus_id", "author_label", "url"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote manifest: {args.output_file}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
