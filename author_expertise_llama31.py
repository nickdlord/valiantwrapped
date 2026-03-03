#!/usr/bin/env python3
"""
makeAuthorSummaries.py

Similar workflow to the ES3890 MakeLLMOutline tutorial:
- Load a chat LLM once
- Loop over input files
- Generate text output

Input:  a folder of per-author Scopus CSVs (same schema)
Output: author_expertise_summaries.csv (+ optional per-author .txt files)

Notes:
- No year filtering.
- Uses a map->reduce approach so large author corpora fit reliably.

PATCH ADDED:
- MODEL_ID is now read from env (MODEL_ID) with a default fallback
- Gated-model auth support:
    - USE_HF_TOKEN=1 enables token=True for from_pretrained()
    - HF_TOKEN / HUGGINGFACE_HUB_TOKEN can be provided via environment
- Helpful debug prints to confirm which mode you’re in
"""

import os
import glob
import re
from typing import List, Dict, Optional

import pandas as pd

# ----------------------------
# SETTINGS (edit these)
# ----------------------------
INPUT_DIR = "author_csvs"
OUTPUT_CSV = "author_expertise_summaries.csv"
OUTPUT_TXT_DIR = "author_expertise_txt"  # set to None to disable

# NEW: model comes from environment if provided
MODEL_ID = os.getenv(
    "MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct").strip()

# NEW: gated-model auth toggle (set by your GUI/subprocess env)
USE_HF_TOKEN = os.getenv("USE_HF_TOKEN", "0") == "1"

# Generation behavior (lower temp = less “creative”)
MAP_MAX_NEW_TOKENS = 220
REDUCE_MAX_NEW_TOKENS = 240
TEMPERATURE = 0.25
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Prompt sizing / cost controls
MAX_INPUT_TOKENS_PER_CHUNK = 5500
MAX_ROWS_PER_AUTHOR = 250   # cap to avoid extremely large authors taking forever
ABSTRACT_CHAR_LIMIT = 700   # truncate long abstracts before tokenization

# Column candidates (Scopus exports vary)
TITLE_COLS = ["Title", "Document Title", "Article Title"]
ABSTRACT_COLS = ["Abstract", "Description"]
JOURNAL_COLS = ["Source title", "Source Title", "Journal"]
CITES_COLS = ["Cited by", "Citations", "Citation count"]
YEAR_COLS = ["Year", "Publication Year", "Pub. Year"]
KEYWORD_COLS = ["Author Keywords", "Indexed Keywords", "Keywords"]


# ----------------------------
# Utility helpers
# ----------------------------
def pick_col(cols, candidates) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def clean_text(x) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def truncate(s: str, n: int) -> str:
    s = clean_text(s)
    return s if len(s) <= n else s[:n] + "..."


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def build_records(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    title_col = pick_col(cols, TITLE_COLS)
    abs_col = pick_col(cols, ABSTRACT_COLS)
    jour_col = pick_col(cols, JOURNAL_COLS)
    cites_col = pick_col(cols, CITES_COLS)
    year_col = pick_col(cols, YEAR_COLS)
    kw_col = pick_col(cols, KEYWORD_COLS)

    out = df.copy()
    out["_title"] = out[title_col].fillna("") if title_col else ""
    out["_abstract"] = out[abs_col].fillna("") if abs_col else ""
    out["_journal"] = out[jour_col].fillna("") if jour_col else ""
    out["_cites"] = safe_int_series(out[cites_col]) if cites_col else 0
    out["_year"] = safe_int_series(out[year_col]) if year_col else -1
    out["_keywords"] = out[kw_col].fillna("") if kw_col else ""

    return out


def format_row_as_record(r: pd.Series) -> str:
    title = truncate(r["_title"], 200)
    journal = truncate(r["_journal"], 110)
    abstract = truncate(r["_abstract"], ABSTRACT_CHAR_LIMIT)
    keywords = truncate(r["_keywords"], 220)
    year = int(r["_year"]) if r["_year"] != -1 else None
    cites = int(r["_cites"])

    parts = []
    if year is not None:
        parts.append(f"Year: {year}")
    parts.append(f"Citations: {cites}")
    parts.append(f"Journal: {journal}" if journal else "Journal: (missing)")
    parts.append(f"Title: {title}" if title else "Title: (missing)")
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n".join(parts)


def chunk_records(records: List[str], tokenizer, max_tokens: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_tokens = 0

    for rec in records:
        rec_tokens = len(tokenizer.encode(rec, add_special_tokens=False)) + 8
        if rec_tokens > max_tokens:
            # hard truncate if needed
            rec = truncate(rec, 1500)
            rec_tokens = len(tokenizer.encode(
                rec, add_special_tokens=False)) + 8

        if cur and (cur_tokens + rec_tokens > max_tokens):
            chunks.append(cur)
            cur = [rec]
            cur_tokens = rec_tokens
        else:
            cur.append(rec)
            cur_tokens += rec_tokens

    if cur:
        chunks.append(cur)
    return chunks


# ----------------------------
# Prompt builders
# ----------------------------
def map_messages(author_id: str, chunk_text: str) -> List[Dict[str, str]]:
    system = (
        "You are a careful research analyst. Infer research themes from publication metadata.\n"
        "Rules:\n"
        "- Use ONLY the provided evidence.\n"
        "- Do NOT invent affiliations, grants, or specific claims not supported.\n"
        "- Output 4–8 bullet themes plus 1–2 sentences summarizing overall focus.\n"
    )
    user = (
        f"Researcher ID: {author_id}\n\n"
        "Publication records:\n"
        f"{chunk_text}\n\n"
        "Task: Extract the main recurring research themes, methods, and application areas."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reduce_messages(author_id: str, map_summaries: str) -> List[Dict[str, str]]:
    system = (
        "Write a concise expertise blurb for a research center website.\n"
        "Rules:\n"
        "- Use ONLY the evidence in the chunk summaries.\n"
        "- Do NOT invent facts.\n"
        "- Write 1–2 paragraphs (~120–220 words total).\n"
        "- Emphasize the author’s main research expertise and recurring themes.\n"
        "- Plain language, professional tone.\n"
    )
    user = (
        f"Researcher ID: {author_id}\n\n"
        "Chunk-level theme summaries:\n"
        f"{map_summaries}\n\n"
        "Now synthesize into a 1–2 paragraph description of overall research expertise."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------
# Generation
# ----------------------------
def generate(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text.strip()


def main():
    import sys
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # --- NEW: gated repo auth + env-driven model id ---
    print(f"Python: {sys.executable}")
    print(f"Loading model: {MODEL_ID}")
    print(f"USE_HF_TOKEN={int(USE_HF_TOKEN)}")

    # If USE_HF_TOKEN=1, Transformers will use HF_TOKEN/HUGGINGFACE_HUB_TOKEN from env.
    # Passing token=True is important for gated repos.
    token_arg = True if USE_HF_TOKEN else None

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        token=token_arg,
    )

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None  # safer on CPU / Windows

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        token=token_arg,
    )
    model.eval()
    # --- END NEW ---

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    if OUTPUT_TXT_DIR:
        os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    results = []

    for path in paths:
        fname = os.path.basename(path)
        author_id = os.path.splitext(fname)[0]
        print(f"\n--- {author_id} ---")

        df = load_csv(path)
        if df.empty:
            summary = "No publications were found in the provided export."
            results.append(
                {"author_id": author_id, "author_file": fname, "summary": summary})
            if OUTPUT_TXT_DIR:
                with open(os.path.join(OUTPUT_TXT_DIR, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(summary + "\n")
            continue

        df2 = build_records(df)

        # Keep the “most informative” rows first, then cap
        df2 = df2.sort_values(["_cites", "_year"], ascending=[
                              False, False]).head(MAX_ROWS_PER_AUTHOR)

        record_strings = [format_row_as_record(r) for _, r in df2.iterrows()]
        chunks = chunk_records(record_strings, tokenizer,
                               MAX_INPUT_TOKENS_PER_CHUNK)

        # MAP step
        map_out = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = "\n\n---\n\n".join(chunk)
            msg = map_messages(author_id, chunk_text)
            chunk_summary = generate(
                model, tokenizer, msg, max_new_tokens=MAP_MAX_NEW_TOKENS)
            map_out.append(f"Chunk {i}:\n{chunk_summary}")

        # REDUCE step
        combined = "\n\n".join(map_out)
        msg = reduce_messages(author_id, combined)
        final_summary = generate(
            model, tokenizer, msg, max_new_tokens=REDUCE_MAX_NEW_TOKENS)

        results.append(
            {"author_id": author_id, "author_file": fname, "summary": final_summary})

        if OUTPUT_TXT_DIR:
            with open(os.path.join(OUTPUT_TXT_DIR, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                f.write(final_summary + "\n")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Wrote: {OUTPUT_CSV}")
    if OUTPUT_TXT_DIR:
        print(f"✅ Wrote per-author .txt files to: {OUTPUT_TXT_DIR}/")


if __name__ == "__main__":
    main()
