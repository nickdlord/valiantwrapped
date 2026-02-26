#!/usr/bin/env python3
"""
author_expertise_llama31.py

Reads per-author Scopus CSVs and generates a 1–2 paragraph expertise summary per author
using meta-llama/Meta-Llama-3.1-8B-Instruct (Transformers).

Approach: map-reduce summarization
- Create compact "paper records" from each row (title/journal/citations/keywords/abstract)
- Chunk records to fit the model context
- MAP: summarize themes per chunk
- REDUCE: synthesize chunk themes into final 1–2 paragraphs

Outputs:
- author_expertise_summaries.csv
- optional per-author text files in OUTPUT_TXT_DIR
"""

import os
import glob
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd

# ----------------------------
# USER SETTINGS
# ----------------------------
INPUT_DIR = "author_csvs"  # folder containing per-author CSVs
OUTPUT_CSV = "author_expertise_summaries.csv"
OUTPUT_TXT_DIR = "author_expertise_txt"  # set to None to disable .txt outputs

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Generation controls (keep fairly low-temp for factual tone)
MAX_NEW_TOKENS_MAP = 220
MAX_NEW_TOKENS_REDUCE = 220
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Chunking controls
MAX_INPUT_TOKENS = 6000  # keep prompt sizes manageable
MAX_PAPERS_PER_AUTHOR = 250  # cap to avoid huge prompts for prolific authors

# Column candidates (Scopus exports can vary)
YEAR_COLS = ["Year", "Publication Year", "Pub. Year"]
TITLE_COLS = ["Title", "Document Title", "Article Title"]
ABSTRACT_COLS = ["Abstract", "Description"]
JOURNAL_COLS = ["Source title", "Source Title", "Journal"]
CITES_COLS = ["Cited by", "Citations", "Citation count"]
KEYWORD_COLS = ["Author Keywords", "Indexed Keywords", "Keywords"]

# ----------------------------
# Helpers
# ----------------------------


def pick_existing_col(cols, candidates) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def truncate(x: str, max_chars: int) -> str:
    x = clean_text(x)
    return x if len(x) <= max_chars else (x[:max_chars] + "...")


def load_csv(path: str) -> pd.DataFrame:
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

    out = df.copy()

    # These are optional; we do NOT filter by year, but year helps summarize time evolution
    out["_year"] = safe_int_series(out[year_col]) if year_col else -1
    out["_title"] = out[title_col].fillna("") if title_col else ""
    out["_abstract"] = out[abstract_col].fillna("") if abstract_col else ""
    out["_journal"] = out[journal_col].fillna("") if journal_col else ""
    out["_cites"] = safe_int_series(out[cites_col]) if cites_col else 0
    out["_keywords"] = out[kw_col].fillna("") if kw_col else ""

    colmap = {
        "year_col": year_col or "",
        "title_col": title_col or "",
        "abstract_col": abstract_col or "",
        "journal_col": journal_col or "",
        "cites_col": cites_col or "",
        "kw_col": kw_col or "",
    }
    return out, colmap


def format_record(row: pd.Series) -> str:
    # Compact but informative per-paper record
    year = int(row["_year"]) if row["_year"] is not None else -1
    cites = int(row["_cites"]) if row["_cites"] is not None else 0

    title = truncate(row["_title"], 180)
    journal = truncate(row["_journal"], 90)
    abstract = truncate(row["_abstract"], 650)
    keywords = truncate(row["_keywords"], 180)

    parts = []
    if year != -1:
        parts.append(f"Year: {year}")
    parts.append(f"Citations: {cites}")
    parts.append(f"Journal: {journal}" if journal else "Journal: (missing)")
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
        rec_tokens = len(tokenizer.encode(rec, add_special_tokens=False)) + 10

        # If a single record is huge, hard truncate by characters
        if rec_tokens > max_input_tokens:
            rec = truncate(rec, 1500)
            rec_tokens = len(tokenizer.encode(
                rec, add_special_tokens=False)) + 10

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


def make_map_messages(author_id: str, chunk_text: str) -> List[Dict[str, str]]:
    sys = (
        "You are a careful research analyst. Summarize research themes from publication metadata.\n"
        "Rules:\n"
        "- Use ONLY the provided evidence.\n"
        "- Do NOT invent methods, grants, affiliations, institutions, or claims not supported.\n"
        "- Prefer recurring themes, application areas, and methods at a high level.\n"
        "- Output: 4–8 bullet themes + 1–2 sentences describing the overall focus.\n"
    )
    user = (
        f"Researcher ID: {author_id}\n\n"
        "Publication records:\n"
        f"{chunk_text}\n\n"
        "Task: Extract the main recurring research themes and problem areas."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def make_reduce_messages(author_id: str, map_summaries: str) -> List[Dict[str, str]]:
    sys = (
        "You write concise expertise blurbs for a research center website.\n"
        "Rules:\n"
        "- Use ONLY the evidence in the chunk summaries.\n"
        "- Do NOT invent facts.\n"
        "- Write 1–2 paragraphs (about 120–220 words total).\n"
        "- Emphasize the author's main research expertise and recurring themes.\n"
        "- Keep tone professional and plain-language.\n"
    )
    user = (
        f"Researcher ID: {author_id}\n\n"
        "Chunk-level theme summaries:\n"
        f"{map_summaries}\n\n"
        "Now synthesize into a 1–2 paragraph description of overall research expertise."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def generate_chat(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Some setups echo the prompt; remove if detected
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    return text.strip()

# ----------------------------
# Main
# ----------------------------


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer/model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    if torch.cuda.is_available():
        # bfloat16 works well on A100/H100; float16 otherwise
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {INPUT_DIR}")

    if OUTPUT_TXT_DIR:
        os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    results = []

    for path in paths:
        fname = os.path.basename(path)
        author_id = os.path.splitext(fname)[0]
        print(f"\n--- Processing {author_id} ({fname}) ---")

        df = load_csv(path)
        if df.empty:
            summary = "No publications were found in the provided export."
            results.append(
                {"author_id": author_id, "author_file": fname, "summary": summary})
            if OUTPUT_TXT_DIR:
                with open(os.path.join(OUTPUT_TXT_DIR, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(summary + "\n")
            continue

        df2, _ = build_paper_frame(df)

        # Sort so we keep the most informative rows first when capping
        # Priority: citations desc, then year desc (if available)
        df2 = df2.sort_values(["_cites", "_year"], ascending=[
                              False, False]).head(MAX_PAPERS_PER_AUTHOR)

        record_strings = [format_record(r) for _, r in df2.iterrows()]
        chunks = chunk_records(record_strings, tokenizer, MAX_INPUT_TOKENS)

        # MAP
        map_outputs = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = "\n\n---\n\n".join(chunk)
            messages = make_map_messages(author_id, chunk_text)
            chunk_summary = generate_chat(
                model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS_MAP)
            map_outputs.append(f"Chunk {i} summary:\n{chunk_summary}")

        # REDUCE
        map_summaries_text = "\n\n".join(map_outputs)
        reduce_messages = make_reduce_messages(author_id, map_summaries_text)
        final_summary = generate_chat(
            model, tokenizer, reduce_messages, max_new_tokens=MAX_NEW_TOKENS_REDUCE)

        results.append(
            {"author_id": author_id, "author_file": fname, "summary": final_summary})

        if OUTPUT_TXT_DIR:
            with open(os.path.join(OUTPUT_TXT_DIR, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                f.write(final_summary + "\n")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Wrote summaries CSV: {OUTPUT_CSV}")
    if OUTPUT_TXT_DIR:
        print(f"✅ Wrote per-author txt summaries to: {OUTPUT_TXT_DIR}/")


if __name__ == "__main__":
    main()
