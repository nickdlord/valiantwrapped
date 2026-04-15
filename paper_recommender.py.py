#!/usr/bin/env python3
"""
paper_recommender_from_expertise.py

Standalone recommendation step for VALIANT Wrapped.

Purpose
-------
Use per-author expertise summaries as the query text and search a center-wide
Scopus export for semantically similar papers using sentence embeddings.

Key behaviors
-------------
- Reads expertise TXT files produced by author_expertise_llama31_2.py
- Uses sentence-transformers/all-mpnet-base-v2 by default
- Searches only within the provided Scopus export CSV
- Excludes papers where the target author's Scopus ID appears in Author(s) ID
- Outputs clickable Google-search URLs for recommended paper titles
- Writes long-form CSV, per-author TXT files, and per-author HTML snippets with a button link
- Supports embedding cache to avoid recomputing the paper database every run

Typical usage
-------------
Batch mode:
    python paper_recommender_from_expertise.py \
      --expertise-dir author_expertise_txt \
      --scopus-db scopusexportALL_02252026.csv \
      --output-dir outputs/paper_recommendations

Single-author mode:
    python paper_recommender_from_expertise.py \
      --expertise-file author_expertise_txt/Landman_Bennett_16679175200.txt \
      --scopus-db scopusexportALL_02252026.csv \
      --output-dir outputs/paper_recommendations

Notes
-----
Expected expertise file naming convention:
    Last_First_ScopusID.txt

Expected Scopus CSV fields (flexibly resolved):
    - Title / Document Title / Article Title
    - Abstract / Description
    - Author(s) ID / Authors with affiliations / Authors / Author Names
    - Link / DOI / DOI link (optional, not required here)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

# Force UTF-8 console output where available
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_FUNNY_FALLBACK = (
    "Apparently your research fingerprint is so distinct that our algorithm "
    "couldn’t find a worthy in-house twin. In other words: too original to match cleanly."
)

TITLE_COLS = ["Title", "Document Title", "Article Title"]
ABSTRACT_COLS = ["Abstract", "Description"]
AUTHOR_ID_COLS = [
    "Author(s) ID",
    "Authors with affiliations",
    "Authors",
    "Author Names",
]
KEYWORD_COLS = ["Author Keywords", "Indexed Keywords", "Keywords"]
JOURNAL_COLS = ["Source title", "Source Title", "Journal"]
LINK_COLS = ["Link", "URL", "Scopus Link", "Page link"]
DOI_COLS = ["DOI", "doi", "DOI link"]


@dataclass
class PaperRecord:
    row_id: int
    title: str
    abstract: str
    keywords: str
    journal: str
    author_ids_raw: str
    link: str
    doi: str
    combined_text: str


@dataclass
class Recommendation:
    author_label: str
    rank: int
    title: str
    score: float
    google_url: str
    journal: str
    doi: str
    scopus_link: str


def pick_existing_col(cols: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = set(cols)
    for candidate in candidates:
        if candidate in col_set:
            return candidate
    return None


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_author_label(value: str) -> str:
    text = clean_text(value).replace("\\", "/")
    text = os.path.basename(text)
    if text.lower().endswith(".txt"):
        text = text[:-4]
    if text.lower().endswith(".csv"):
        text = text[:-4]
    return text


def extract_scopus_id_from_label(author_label: str) -> str:
    author_label = canonical_author_label(author_label)
    parts = author_label.split("_")
    if not parts:
        return ""
    last = parts[-1]
    return last if re.fullmatch(r"\d+", last) else ""


def load_text(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_query_text_from_expertise(raw_text: str) -> str:
    raw_text = raw_text.replace("\r\n", "\n")

    # Prefer explicit SUMMARY section written by author_expertise_llama31_2.py
    m = re.search(r"(?is)\bSUMMARY\s*:\s*(.+)$", raw_text)
    if m:
        summary = clean_text(m.group(1))
        if summary:
            return summary

    # Fallback: remove THEMES label noise and use the remaining text
    text = re.sub(r"(?im)^\s*THEMES\s*:\s*$", "", raw_text)
    text = re.sub(r"(?im)^\s*SUMMARY\s*:\s*$", "", text)
    text = clean_text(text)
    return text


def google_search_url(title: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(title)}"


def parse_author_ids(raw: str) -> List[str]:
    raw = clean_text(raw)
    if not raw:
        return []
    ids = re.findall(r"\d+", raw)
    seen = set()
    ordered = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def load_scopus_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def build_paper_records(scopus_df: pd.DataFrame) -> List[PaperRecord]:
    title_col = pick_existing_col(scopus_df.columns, TITLE_COLS)
    abstract_col = pick_existing_col(scopus_df.columns, ABSTRACT_COLS)
    author_id_col = pick_existing_col(scopus_df.columns, AUTHOR_ID_COLS)
    keyword_col = pick_existing_col(scopus_df.columns, KEYWORD_COLS)
    journal_col = pick_existing_col(scopus_df.columns, JOURNAL_COLS)
    link_col = pick_existing_col(scopus_df.columns, LINK_COLS)
    doi_col = pick_existing_col(scopus_df.columns, DOI_COLS)

    if not title_col:
        raise ValueError(
            f"Could not find a paper title column. Tried: {TITLE_COLS}"
        )

    records: List[PaperRecord] = []
    for idx, row in scopus_df.iterrows():
        title = clean_text(row.get(title_col, ""))
        if not title:
            continue
        abstract = clean_text(row.get(abstract_col, "")) if abstract_col else ""
        keywords = clean_text(row.get(keyword_col, "")) if keyword_col else ""
        journal = clean_text(row.get(journal_col, "")) if journal_col else ""
        author_ids_raw = clean_text(row.get(author_id_col, "")) if author_id_col else ""
        link = clean_text(row.get(link_col, "")) if link_col else ""
        doi = clean_text(row.get(doi_col, "")) if doi_col else ""

        parts = [f"Title: {title}"]
        if journal:
            parts.append(f"Venue: {journal}")
        if keywords:
            parts.append(f"Keywords: {keywords}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        combined = "\n".join(parts)

        records.append(
            PaperRecord(
                row_id=int(idx),
                title=title,
                abstract=abstract,
                keywords=keywords,
                journal=journal,
                author_ids_raw=author_ids_raw,
                link=link,
                doi=doi,
                combined_text=combined,
            )
        )

    if not records:
        raise ValueError("No usable paper rows were found in the Scopus export.")
    return records


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path_for(model_id: str, scopus_db: Path, cache_dir: Path) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    fp = file_fingerprint(scopus_db)
    return cache_dir / f"paper_embeddings__{safe_model}__{fp}.pkl"


def load_embedding_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. This recommender step is configured for GPU use on ACCRE."
        )

    device = "cuda"
    model = SentenceTransformer(model_id, device=device)
    return model


def encode_texts(model, texts: Sequence[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_or_create_paper_embedding_cache(
    model,
    model_id: str,
    scopus_db: Path,
    records: Sequence[PaperRecord],
    cache_dir: Path,
    batch_size: int,
    use_cache: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path_for(model_id, scopus_db, cache_dir)

    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        titles = payload.get("titles", [])
        if titles == [r.title for r in records]:
            return np.asarray(payload["embeddings"], dtype=np.float32)

    embeddings = encode_texts(model, [r.combined_text for r in records], batch_size=batch_size)

    if use_cache:
        payload = {
            "titles": [r.title for r in records],
            "embeddings": embeddings,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(payload, f)

    return embeddings


def cosine_scores(query_embedding: np.ndarray, paper_embeddings: np.ndarray) -> np.ndarray:
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]
    return np.dot(paper_embeddings, query_embedding)


def witty_fallback(author_label: str, custom_text: str) -> str:
    return custom_text.strip() if clean_text(custom_text) else DEFAULT_FUNNY_FALLBACK


def recommend_for_author(
    author_label: str,
    query_text: str,
    model,
    paper_records: Sequence[PaperRecord],
    paper_embeddings: np.ndarray,
    top_k: int,
    min_similarity: float,
) -> Tuple[List[Recommendation], Optional[str]]:
    author_label = canonical_author_label(author_label)
    scopus_id = extract_scopus_id_from_label(author_label)

    if not query_text.strip():
        return [], DEFAULT_FUNNY_FALLBACK

    query_embedding = encode_texts(model, [query_text], batch_size=1)[0]
    scores = cosine_scores(query_embedding, paper_embeddings)
    ranked_idx = np.argsort(scores)[::-1]

    recommendations: List[Recommendation] = []
    seen_titles = set()

    for idx in ranked_idx:
        record = paper_records[int(idx)]
        if record.title.lower() in seen_titles:
            continue

        if author_in_record(scopus_id, record):
            continue

        score = float(scores[int(idx)])
        if score < min_similarity:
            continue

        seen_titles.add(record.title.lower())
        recommendations.append(
            Recommendation(
                author_label=author_label,
                rank=len(recommendations) + 1,
                title=record.title,
                score=score,
                google_url=google_search_url(record.title),
                journal=record.journal,
                doi=record.doi,
                scopus_link=record.link,
            )
        )
        if len(recommendations) >= top_k:
            break

    if recommendations:
        return recommendations, None
    return [], DEFAULT_FUNNY_FALLBACK


def author_in_record(scopus_id: str, record: PaperRecord) -> bool:
    if not scopus_id:
        return False

    record_ids = parse_author_ids(record.author_ids_raw)
    if scopus_id in record_ids:
        return True

    raw = clean_text(record.author_ids_raw)
    return re.search(rf"(?<!\d){re.escape(scopus_id)}(?!\d)", raw) is not None


def write_author_html(
    path: Path,
    author_label: str,
    recommendations: Sequence[Recommendation],
    fallback_text: Optional[str],
) -> None:
    style = (
        "<style>"
        "body{font-family:Inter,Arial,sans-serif;margin:0;padding:0;background:#f8fafc;color:#0f172a;}"
        ".wrap{max-width:900px;margin:0 auto;padding:24px;}"
        ".card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:18px 20px;box-shadow:0 8px 24px rgba(15,23,42,.06);margin-bottom:14px;}"
        ".title{font-size:18px;font-weight:800;margin:0 0 10px;}"
        ".meta{font-size:13px;color:#475569;margin:0 0 12px;}"
        ".actions{margin-top:10px;}"
        ".button{display:inline-block;background:#2563eb;color:#fff;text-decoration:none;font-weight:700;padding:10px 14px;border-radius:999px;}"
        ".fallback{background:#fff7ed;border:1px solid #fdba74;color:#9a3412;border-radius:16px;padding:16px 18px;font-weight:600;}"
        ".small{font-size:12px;color:#64748b;}"
        "</style>"
    )

    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{html.escape(author_label)} Recommendations</title>',
        style,
        '</head><body><div class="wrap">',
        f'<h1>Related Papers for {html.escape(author_label)}</h1>',
    ]

    if recommendations:
        for rec in recommendations:
            journal = f'<p class="meta">Journal: {html.escape(rec.journal)}</p>' if rec.journal else ''
            parts.append(
                f'<div class="card"><h2 class="title">{rec.rank}. {html.escape(rec.title)}</h2>{journal}<p class="small">Similarity score: {rec.score:.4f}</p><div class="actions"><a class="button" href="{html.escape(rec.google_url)}" target="_blank" rel="noopener noreferrer">Look it up on Google!</a></div></div>'
            )
    else:
        parts.append(f'<div class="fallback">{html.escape(fallback_text or DEFAULT_FUNNY_FALLBACK)}</div>')

    parts.append('</div></body></html>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_author_txt(
    path: Path,
    author_label: str,
    recommendations: Sequence[Recommendation],
    fallback_text: Optional[str],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"AUTHOR_LABEL: {author_label}\n")
        if recommendations:
            f.write("RECOMMENDATIONS:\n")
            for rec in recommendations:
                f.write(f"{rec.rank}. {rec.title}\n")
                f.write(f"   LOOKUP_TEXT: Look it up on Google!\n")
                f.write(f"   GOOGLE_URL: {rec.google_url}\n")
                f.write(f"   SCORE: {rec.score:.4f}\n")
                if rec.journal:
                    f.write(f"   JOURNAL: {rec.journal}\n")
                if rec.doi:
                    f.write(f"   DOI: {rec.doi}\n")
                if rec.scopus_link:
                    f.write(f"   SCOPUS_LINK: {rec.scopus_link}\n")
        else:
            f.write("RECOMMENDATIONS:\n")
            f.write("(none)\n")
            f.write(f"FALLBACK_TEXT: {fallback_text or DEFAULT_FUNNY_FALLBACK}\n")


def list_expertise_paths(expertise_file: str, expertise_dir: str) -> List[Path]:
    if bool(expertise_file) == bool(expertise_dir):
        raise ValueError("Provide exactly one of --expertise-file or --expertise-dir")

    if expertise_file:
        path = Path(expertise_file)
        if not path.exists():
            raise FileNotFoundError(f"Expertise file not found: {path}")
        return [path]

    pdir = Path(expertise_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"Expertise directory not found: {pdir}")
    paths = sorted(pdir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in expertise directory: {pdir}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expertise-file", default="", help="Path to one expertise TXT file")
    ap.add_argument("--expertise-dir", default="", help="Directory of per-author expertise TXT files")
    ap.add_argument("--scopus-db", required=True, help="Path to the mass Scopus export CSV")
    ap.add_argument("--output-dir", default="outputs/paper_recommendations", help="Output folder")
    ap.add_argument("--model-id", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--recommendation-count", type=int, default=5)
    ap.add_argument("--min-similarity", type=float, default=0.30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-dir", default=".cache/paper_recommender")
    ap.add_argument("--no-cache", action="store_true", help="Disable paper embedding cache")
    ap.add_argument(
        "--fallback-text",
        default=DEFAULT_FUNNY_FALLBACK,
        help="Fallback text when no recommendation meets the similarity threshold",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    expertise_paths = list_expertise_paths(args.expertise_file, args.expertise_dir)
    scopus_db = Path(args.scopus_db)
    if not scopus_db.exists():
        raise FileNotFoundError(f"Scopus database CSV not found: {scopus_db}")

    output_dir = Path(args.output_dir)
    per_author_dir = output_dir / "per_author_txt"
    per_author_html_dir = output_dir / "per_author_html"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_author_dir.mkdir(parents=True, exist_ok=True)
    per_author_html_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Scopus database...")
    scopus_df = load_scopus_csv(scopus_db)
    paper_records = build_paper_records(scopus_df)
    print(f"Loaded {len(paper_records)} candidate papers from {scopus_db.name}")

    print(f"Loading embedding model: {args.model_id}")
    model = load_embedding_model(args.model_id)

    print("Preparing paper embeddings...")
    paper_embeddings = load_or_create_paper_embedding_cache(
        model=model,
        model_id=args.model_id,
        scopus_db=scopus_db,
        records=paper_records,
        cache_dir=Path(args.cache_dir),
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
    )

    csv_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for expertise_path in expertise_paths:
        author_label = canonical_author_label(expertise_path.stem)
        raw_text = load_text(expertise_path)
        query_text = extract_query_text_from_expertise(raw_text)

        print(f"\nRecommending papers for {author_label}...")
        recs, fallback = recommend_for_author(
            author_label=author_label,
            query_text=query_text,
            model=model,
            paper_records=paper_records,
            paper_embeddings=paper_embeddings,
            top_k=max(1, args.recommendation_count),
            min_similarity=args.min_similarity,
        )

        resolved_fallback = witty_fallback(author_label, args.fallback_text) if not recs else None

        write_author_txt(
            per_author_dir / f"{author_label}.txt",
            author_label=author_label,
            recommendations=recs,
            fallback_text=resolved_fallback,
        )
        write_author_html(
            per_author_html_dir / f"{author_label}.html",
            author_label=author_label,
            recommendations=recs,
            fallback_text=resolved_fallback,
        )

        if recs:
            summary_rows.append(
                {
                    "author_label": author_label,
                    "recommendation_count": len(recs),
                    "fallback_text": "",
                }
            )
            for rec in recs:
                csv_rows.append(
                    {
                        "author_label": rec.author_label,
                        "rank": rec.rank,
                        "title": rec.title,
                        "google_url": rec.google_url,
                        "lookup_text": "Look it up on Google!",
                        "score": rec.score,
                        "journal": rec.journal,
                        "doi": rec.doi,
                        "scopus_link": rec.scopus_link,
                    }
                )
        else:
            summary_rows.append(
                {
                    "author_label": author_label,
                    "recommendation_count": 0,
                    "fallback_text": witty_fallback(author_label, args.fallback_text),
                }
            )

    pd.DataFrame(csv_rows).to_csv(output_dir / "paper_recommendations_long.csv", index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(output_dir / "paper_recommendations_summary.csv", index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Per-author TXT outputs: {per_author_dir}")
    print(f"Per-author HTML outputs: {per_author_html_dir}")
    print(f"Long-form CSV: {output_dir / 'paper_recommendations_long.csv'}")
    print(f"Summary CSV: {output_dir / 'paper_recommendations_summary.csv'}")


if __name__ == "__main__":
    main()
