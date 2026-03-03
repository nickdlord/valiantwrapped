#!/usr/bin/env python3
"""
make_author_summaries.py

Per-author Scopus CSV -> 1–2 paragraph "academic journey + expertise" summary
using meta-llama/Meta-Llama-3.1-8B-Instruct on a GPU (Transformers).

Strategy: map -> reduce
- Convert each paper row into a compact "record"
- Chunk records to fit context
- MAP: extract themes + timeline notes per chunk
- REDUCE: synthesize into final 1–2 paragraph blurb

Inputs:
  A folder of CSVs, one CSV per author (filename used as author_id)

Outputs:
  - author_expertise_summaries.csv
  - optional per-author .txt files

Notes:
- Designed for HPC (ACCRE) single-GPU runs.
- Handles common Scopus column names, including the example schema
  (Title, Abstract, Year, Source title, Cited by, Author Keywords, etc.).
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Column candidates (Scopus exports vary)
# ----------------------------
YEAR_COLS = ["Year", "Publication Year", "Pub. Year"]
TITLE_COLS = ["Title", "Document Title", "Article Title"]
ABSTRACT_COLS = ["Abstract", "Description"]
JOURNAL_COLS = ["Source title", "Source Title", "Journal"]
CITES_COLS = ["Cited by", "Citations", "Citation count"]
KEYWORD_COLS = ["Author Keywords", "Indexed Keywords", "Keywords"]
DOCTYPE_COLS = ["Document Type", "Doc Type", "Type"]


# ----------------------------
# Basic helpers
# ----------------------------
def pick_existing_col(cols: pd.Index, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def clean_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def truncate(s: str, max_chars: int) -> str:
    s = clean_text(s)
    return s if len(s) <= max_chars else (s[:max_chars] + "...")


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def load_csv(path: str) -> pd.DataFrame:
    # Scopus exports sometimes have odd encodings
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def build_paper_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cols = df.columns
    year_col = pick_existing_col(cols, YEAR_COLS)
    title_col = pick_existing_col(cols, TITLE_COLS)
    abstract_col = pick_existing_col(cols, ABSTRACT_COLS)
    journal_col = pick_existing_col(cols, JOURNAL_COLS)
    cites_col = pick_existing_col(cols, CITES_COLS)
    kw_col = pick_existing_col(cols, KEYWORD_COLS)
    dtype_col = pick_existing_col(cols, DOCTYPE_COLS)

    out = df.copy()

    out["_year"] = safe_int_series(out[year_col]) if year_col else -1
    out["_title"] = out[title_col].fillna("") if title_col else ""
    out["_abstract"] = out[abstract_col].fillna("") if abstract_col else ""
    out["_journal"] = out[journal_col].fillna("") if journal_col else ""
    out["_cites"] = safe_int_series(out[cites_col]) if cites_col else 0
    out["_keywords"] = out[kw_col].fillna("") if kw_col else ""
    out["_doctype"] = out[dtype_col].fillna("") if dtype_col else ""

    colmap = {
        "year": year_col or "",
        "title": title_col or "",
        "abstract": abstract_col or "",
        "journal": journal_col or "",
        "cites": cites_col or "",
        "keywords": kw_col or "",
        "doctype": dtype_col or "",
    }
    return out, colmap


def format_record(row: pd.Series) -> str:
    year = int(row["_year"]) if row.get("_year") is not None else -1
    cites = int(row["_cites"]) if row.get("_cites") is not None else 0

    title = truncate(row.get("_title", ""), 180)
    journal = truncate(row.get("_journal", ""), 90)
    abstract = truncate(row.get("_abstract", ""), 650)
    keywords = truncate(row.get("_keywords", ""), 180)
    doctype = truncate(row.get("_doctype", ""), 60)

    parts = []
    if year != -1:
        parts.append(f"Year: {year}")
    parts.append(f"Citations: {cites}")
    if doctype:
        parts.append(f"Type: {doctype}")
    parts.append(f"Venue: {journal}" if journal else "Venue: (missing)")
    parts.append(f"Title: {title}" if title else "Title: (missing)")
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    return "\n".join(parts)


def chunk_records(records: List[str], tokenizer, max_input_tokens: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    for rec in records:
        rec_tokens = len(tokenizer.encode(rec, add_special_tokens=False)) + 8

        # If a single record is too big, hard truncate by chars
        if rec_tokens > max_input_tokens:
            rec = truncate(rec, 1500)
            rec_tokens = len(tokenizer.encode(rec, add_special_tokens=False)) + 8

        if current and (current_tokens + rec_tokens > max_input_tokens):
            chunks.append(current)
            current = [rec]
            current_tokens = rec_tokens
        else:
            current.append(rec)
            current_tokens += rec_tokens

    if current:
        chunks.append(current)

    return chunks


# ----------------------------
# Prompting
# ----------------------------
def make_map_messages(author_id: str, chunk_text: str) -> List[Dict[str, str]]:
    sys = (
        "You are a careful research analyst summarizing an author's publications.\n"
        "Hard rules:\n"
        "- Use ONLY the evidence provided.\n"
        "- Do NOT invent institutions, grants, methods, awards, roles, or claims.\n"
        "- If something is unknown, omit it.\n"
        "Output format:\n"
        "1) 4–8 bullet points of recurring research themes (methods + application areas).\n"
        "2) 2–4 bullet points describing timeline/evolution (early vs recent directions) if years allow.\n"
        "3) 1 sentence summarizing the overall research identity.\n"
    )
    user = (
        f"Author ID: {author_id}\n\n"
        "Publication records (each record is one paper):\n"
        f"{chunk_text}\n\n"
        "Task: Extract themes and any evidence-based evolution over time."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def make_reduce_messages(author_id: str, map_summaries: str) -> List[Dict[str, str]]:
    sys = (
        "You write concise website-ready research bios.\n"
        "Hard rules:\n"
        "- Use ONLY the evidence in the chunk summaries.\n"
        "- Do NOT invent facts.\n"
        "- Write 1–2 paragraphs, ~130–230 words total.\n"
        "- Must include (a) academic journey/evolution AND (b) primary expertise areas.\n"
        "- Professional, plain-language tone.\n"
    )
    user = (
        f"Author ID: {author_id}\n\n"
        "Chunk-level evidence summaries:\n"
        f"{map_summaries}\n\n"
        "Now synthesize into a 1–2 paragraph bio describing (1) evolution of research focus and "
        "(2) primary areas of expertise."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def generate_chat(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int,
                  temperature: float, top_p: float, repetition_penalty: float) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Some setups echo the prompt; remove it if detected
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    return text.strip()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="author_csvs", help="Folder containing per-author CSV files")
    ap.add_argument("--output-csv", default="author_expertise_summaries.csv", help="Output CSV path")
    ap.add_argument("--output-txt-dir", default="author_expertise_txt",
                    help="Folder for per-author .txt outputs; set to empty string to disable")
    ap.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max-papers", type=int, default=250, help="Cap papers per author (keeps runtime predictable)")
    ap.add_argument("--max-input-tokens", type=int, default=6000, help="Max tokens sent per chunk")
    ap.add_argument("--map-max-new", type=int, default=220, help="Max new tokens for map step")
    ap.add_argument("--reduce-max-new", type=int, default=240, help="Max new tokens for reduce step")
    ap.add_argument("--temperature", type=float, default=0.25, help="Lower = less creative")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {args.input_dir}")

    output_txt_dir = args.output_txt_dir.strip() if args.output_txt_dir else ""
    if output_txt_dir:
        os.makedirs(output_txt_dir, exist_ok=True)

    results = []

    for path in paths:
        fname = os.path.basename(path)
        author_id = os.path.splitext(fname)[0]
        print(f"\n--- Processing {author_id} ---")

        df = load_csv(path)
        if df.empty:
            summary = "No publications were found in the provided export."
            results.append({"author_id": author_id, "author_file": fname, "summary": summary})
            if output_txt_dir:
                with open(os.path.join(output_txt_dir, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(summary + "\n")
            continue

        df2, colmap = build_paper_frame(df)

        # Keep informative rows first: citations desc, then year desc
        df2 = df2.sort_values(["_cites", "_year"], ascending=[False, False]).head(args.max_papers)

        record_strings = [format_record(r) for _, r in df2.iterrows()]
        chunks = chunk_records(record_strings, tokenizer, args.max_input_tokens)

        # MAP
        map_outputs = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = "\n\n---\n\n".join(chunk)
            messages = make_map_messages(author_id, chunk_text)
            chunk_summary = generate_chat(
                model, tokenizer, messages,
                max_new_tokens=args.map_max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            map_outputs.append(f"Chunk {i}:\n{chunk_summary}")

        # REDUCE
        map_summaries_text = "\n\n".join(map_outputs)
        reduce_messages = make_reduce_messages(author_id, map_summaries_text)
        final_summary = generate_chat(
            model, tokenizer, reduce_messages,
            max_new_tokens=args.reduce_max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        results.append({
            "author_id": author_id,
            "author_file": fname,
            "summary": final_summary,
            "rows_used": len(df2),
            "detected_title_col": colmap["title"],
            "detected_abstract_col": colmap["abstract"],
            "detected_year_col": colmap["year"],
            "detected_journal_col": colmap["journal"],
            "detected_cites_col": colmap["cites"],
            "detected_keywords_col": colmap["keywords"],
        })

        if output_txt_dir:
            with open(os.path.join(output_txt_dir, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                f.write(final_summary + "\n")

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\n✅ Wrote: {args.output_csv}")
    if output_txt_dir:
        print(f"✅ Wrote per-author txt files to: {output_txt_dir}/")


if __name__ == "__main__":
    main()
