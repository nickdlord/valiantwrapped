#!/usr/bin/env python3
"""
author_expertise_llama31_2.py (REWRITTEN)

Per-author Scopus CSV -> (A) recurring research themes + (B) 1–2 paragraph summary
using meta-llama/Meta-Llama-3.1-8B-Instruct on a GPU (Transformers).

Key changes vs your original:
- Output CSV contains ONLY: author_id, author_file, themes, summary
- Per-author TXT (optional) contains ONLY themes + summary
- Prompts explicitly forbid listing paper titles/citations/bibliographies
- Generation decoding returns ONLY generated tokens (prevents prompt/“assistant” leakage)
- Reduce max tokens increased to reduce cutoffs
- Input records are lighter (shorter abstracts) to save tokens

Run:
  python author_expertise_llama31_2.py --input-dir author_csvs_test
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


def format_record(row: pd.Series, abstract_chars: int) -> str:
    """
    Compact single-paper record for the model.
    Keep it tight to save tokens.
    """
    year = int(row["_year"]) if row.get("_year") is not None else -1
    cites = int(row["_cites"]) if row.get("_cites") is not None else 0

    title = truncate(row.get("_title", ""), 160)
    journal = truncate(row.get("_journal", ""), 80)
    keywords = truncate(row.get("_keywords", ""), 160)
    doctype = truncate(row.get("_doctype", ""), 50)
    abstract = truncate(row.get("_abstract", ""), abstract_chars)

    parts = []
    if year != -1:
        parts.append(f"Year: {year}")
    parts.append(f"Citations: {cites}")
    if doctype:
        parts.append(f"Type: {doctype}")
    if journal:
        parts.append(f"Venue: {journal}")
    if title:
        parts.append(f"Title: {title}")
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

        if rec_tokens > max_input_tokens:
            rec = truncate(rec, 1200)
            rec_tokens = len(tokenizer.encode(
                rec, add_special_tokens=False)) + 8

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


def parse_theme_bullets(text: str) -> List[str]:
    """
    Extract bullet lines from model output; keep only theme bullets.
    Accepts -, *, •, or numbered formats.
    """
    lines = [clean_text(x) for x in text.splitlines()]
    bullets = []
    for ln in lines:
        if re.match(r"^(\-|\*|•|\d+\)|\d+\.)\s+", ln):
            ln = re.sub(r"^(\-|\*|•|\d+\)|\d+\.)\s+", "", ln).strip()
            if ln:
                bullets.append(ln)
    # de-dup while preserving order
    seen = set()
    out = []
    for b in bullets:
        key = b.lower()
        if key not in seen:
            out.append(b)
            seen.add(key)
    return out


# ----------------------------
# Prompting
# ----------------------------
def make_map_messages(author_id: str, chunk_text: str) -> List[Dict[str, str]]:
    sys = (
        "You are a careful research analyst summarizing an author's publications.\n"
        "Hard rules:\n"
        "- Use ONLY the evidence provided.\n"
        "- Do NOT invent institutions, grants, awards, roles, or claims.\n"
        "- Do NOT list paper titles, DOIs, or full citations.\n"
        "- If something is unknown, omit it.\n\n"
        "Output format (STRICT):\n"
        "THEMES:\n"
        "- <theme 1>\n"
        "- <theme 2>\n"
        "- <theme 3>\n"
        "- <theme 4>\n"
        "(4–8 bullets total)\n"
    )
    user = (
        f"Author ID: {author_id}\n\n"
        "Publication records (each record is one paper):\n"
        f"{chunk_text}\n\n"
        "Task: Identify recurring research themes (methods + application areas)."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def make_reduce_messages(author_id: str, themes: List[str], evidence_notes: str) -> List[Dict[str, str]]:
    theme_block = "\n".join(
        [f"- {t}" for t in themes[:10]]) if themes else "- (No clear themes found.)"
    sys = (
        "You write concise website-ready research bios.\n"
        "Hard rules:\n"
        "- Use ONLY the evidence provided.\n"
        "- Do NOT invent facts.\n"
        "- Do NOT list paper titles, DOIs, or citations.\n"
        "- Write 1–2 paragraphs, 120–220 words total.\n"
        "- Must include (a) evolution of research focus over time IF supported by years, and "
        "(b) primary areas of expertise.\n"
        "- Plain-language, professional tone.\n"
    )
    user = (
        f"Author ID: {author_id}\n\n"
        "Recurring research themes (from prior analysis):\n"
        f"{theme_block}\n\n"
        "Evidence notes (chunk summaries; do not quote papers):\n"
        f"{evidence_notes}\n\n"
        "Now write the 1–2 paragraph bio. Do not include any paper lists."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def generate_chat(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    """
    Robust generation:
    - Decode only newly generated tokens to avoid prompt/role leakage.
    - Set pad_token_id to eos_token_id to silence warnings.
    """
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
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
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=False,
        )

    generated = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="author_csvs_test",
                    help="Folder containing per-author CSV files")
    ap.add_argument(
        "--output-csv", default="author_expertise_summaries.csv", help="Output CSV path")
    ap.add_argument(
        "--output-txt-dir",
        default="author_expertise_txt",
        help="Folder for per-author .txt outputs; set to empty string to disable",
    )
    ap.add_argument(
        "--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max-papers", type=int, default=250,
                    help="Cap papers per author (keeps runtime predictable)")
    ap.add_argument("--max-input-tokens", type=int,
                    default=6000, help="Max tokens sent per chunk")
    ap.add_argument("--map-max-new", type=int, default=200,
                    help="Max new tokens for map step")
    ap.add_argument("--reduce-max-new", type=int, default=512,
                    help="Max new tokens for reduce step")
    ap.add_argument("--temperature", type=float, default=0.25,
                    help="Lower = less creative")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--abstract-chars", type=int, default=260,
                    help="Max chars of abstract per paper record (token saver)")
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
            themes = []
            results.append(
                {"author_id": author_id, "author_file": fname, "themes": "", "summary": summary})
            if output_txt_dir:
                with open(os.path.join(output_txt_dir, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                    f.write("THEMES:\n")
                    f.write("(none)\n\n")
                    f.write("SUMMARY:\n")
                    f.write(summary + "\n")
            continue

        df2, _ = build_paper_frame(df)

        # Sort: citations desc, then year desc; cap papers
        df2 = df2.sort_values(["_cites", "_year"], ascending=[
                              False, False]).head(args.max_papers)

        record_strings = [format_record(
            r, abstract_chars=args.abstract_chars) for _, r in df2.iterrows()]
        chunks = chunk_records(record_strings, tokenizer,
                               args.max_input_tokens)

        # MAP: themes per chunk (tight output)
        theme_candidates: List[str] = []
        evidence_notes_parts: List[str] = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_text = "\n\n---\n\n".join(chunk)
            messages = make_map_messages(author_id, chunk_text)
            chunk_out = generate_chat(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.map_max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            evidence_notes_parts.append(f"Chunk {i} themes:\n{chunk_out}")
            theme_candidates.extend(parse_theme_bullets(chunk_out))

        # Consolidate themes (keep top N unique)
        themes = theme_candidates[:12]

        # REDUCE: 1–2 paragraphs (use themes + minimal evidence)
        evidence_notes = "\n\n".join(evidence_notes_parts)
        reduce_messages = make_reduce_messages(
            author_id, themes, evidence_notes)
        final_summary = generate_chat(
            model,
            tokenizer,
            reduce_messages,
            max_new_tokens=args.reduce_max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        themes_str = "; ".join(themes)  # CSV-friendly

        # OUTPUT: ONLY themes + summary
        results.append(
            {
                "author_id": author_id,
                "author_file": fname,
                "themes": themes_str,
                "summary": final_summary,
            }
        )

        if output_txt_dir:
            with open(os.path.join(output_txt_dir, f"{author_id}.txt"), "w", encoding="utf-8") as f:
                f.write("THEMES:\n")
                if themes:
                    for t in themes:
                        f.write(f"- {t}\n")
                else:
                    f.write("(none)\n")
                f.write("\nSUMMARY:\n")
                f.write(final_summary + "\n")

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\n✅ Wrote: {args.output_csv}")
    if output_txt_dir:
        print(f"✅ Wrote per-author txt files to: {output_txt_dir}/")


if __name__ == "__main__":
    main()
