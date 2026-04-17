#!/usr/bin/env python3
"""
author_expertise_llama31_2.py

Supports two input modes:

1) TXT summary mode
   Input from scopus2txtsummary.py:
   - single mode via --input-file
   - batch mode via --input-dir
   The script reads per-author TXT summaries and generates:
   - recurring research themes
   - 1–2 paragraph research summary

2) CSV mode
   Legacy support for per-author Scopus CSV files:
   - single mode via --input-file
   - batch mode via --input-dir

Outputs:
- optional CSV with author_id, author_file, themes, summary
- optional per-author TXT files with themes + summary
"""

from typing import List, Tuple
from pathlib import Path
import sys

# Force UTF-8 console output to avoid Windows cp1252 crashes on emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
    return s if len(s) <= max_chars else (s[:max_chars].rstrip() + "...")


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def load_text(path: str) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def resolve_input_paths(input_file: str, input_dir: str, allowed_exts: Tuple[str, ...]) -> List[str]:
    if bool(input_file) == bool(input_dir):
        raise ValueError("Provide exactly one of --input-file or --input-dir")

    if input_file:
        p = Path(input_file)

        # Normal success path
        if p.exists():
            ext = p.suffix.lower()
            if ext not in allowed_exts:
                raise ValueError(
                    f"Unsupported input file type: {ext}. Allowed: {allowed_exts}")
            return [str(p)]

        # Fallback 1: look for same stem in parent directory with allowed extensions
        parent = p.parent
        stem = p.stem.lower()
        if parent.exists():
            matches = []
            for ext in allowed_exts:
                for candidate in parent.glob(f"*{ext}"):
                    if candidate.stem.lower() == stem:
                        matches.append(candidate)

            if len(matches) == 1:
                print(
                    f"[WARN] Input file not found exactly; using matched file: {matches[0]}")
                return [str(matches[0])]

            # Fallback 2: if exactly one supported file exists in folder, use it
            folder_candidates = []
            for ext in allowed_exts:
                folder_candidates.extend(parent.glob(f"*{ext}"))

            if len(folder_candidates) == 1:
                print(
                    f"[WARN] Input file not found exactly; using only file in folder: {folder_candidates[0]}")
                return [str(folder_candidates[0])]

        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Directory mode
    pdir = Path(input_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    paths: List[Path] = []
    for ext in allowed_exts:
        paths.extend(sorted(pdir.glob(f"*{ext}")))

    if not paths:
        raise FileNotFoundError(
            f"No supported input files found in: {input_dir} (looked for {allowed_exts})"
        )

    return [str(p) for p in paths]


def infer_mode_from_paths(paths: List[str]) -> str:
    exts = {os.path.splitext(p)[1].lower() for p in paths}
    if exts == {".txt"}:
        return "txt"
    if exts == {".csv"}:
        return "csv"
    raise ValueError(
        f"Mixed input types are not supported in one run. Found extensions: {sorted(exts)}"
    )


# ----------------------------
# CSV processing helpers
# ----------------------------
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


def build_evidence_from_csv(path: str, max_papers: int, abstract_chars: int) -> Tuple[str, List[str]]:
    df = load_csv(path)
    if df.empty:
        return "No publications were found in the provided export.", []

    df2, _ = build_paper_frame(df)
    df2 = df2.sort_values(["_cites", "_year"], ascending=[
                          False, False]).head(max_papers)

    record_strings = [format_record(
        r, abstract_chars=abstract_chars) for _, r in df2.iterrows()]
    return "\n\n---\n\n".join(record_strings), record_strings


# ----------------------------
# TXT processing helpers
# ----------------------------
def parse_txt_summary_file(path: str) -> Tuple[str, str]:
    raw = load_text(path)
    author_id = os.path.splitext(os.path.basename(path))[0]

    m = re.search(r"AUTHOR_ID:\s*(.+)", raw)
    if m:
        author_id = clean_text(m.group(1))

    return author_id, raw.strip()


# ----------------------------
# Chunking / output parsing
# ----------------------------
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


def chunk_long_text(text: str, tokenizer, max_input_tokens: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else [""]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para, add_special_tokens=False)) + 8

        if para_tokens > max_input_tokens:
            para = truncate(para, 3000)
            para_tokens = len(tokenizer.encode(
                para, add_special_tokens=False)) + 8

        if current_parts and (current_tokens + para_tokens > max_input_tokens):
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = para_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def parse_theme_bullets(text: str) -> List[str]:
    lines = [clean_text(x) for x in text.splitlines()]
    bullets = []
    for ln in lines:
        if re.match(r"^(\-|\*|•|\d+\)|\d+\.)\s+", ln):
            ln = re.sub(r"^(\-|\*|•|\d+\)|\d+\.)\s+", "", ln).strip()
            if ln:
                bullets.append(ln)

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
        "- If something is unknown, omit it.\n"
        "- Do NOT assume, infer, or mention the author's gender.\n"
        "- Do NOT use gendered pronouns or gendered nouns.\n"
        "- If a reference to the author is needed, use only neutral phrasing such as "
        "'the author', 'the researcher', or 'they/their'.\n\n"
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
        "Publication evidence:\n"
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
        "- Must include (a) evolution of research focus over time IF supported by evidence, and "
        "(b) primary areas of expertise.\n"
        "- Plain-language, professional tone.\n"
        "- Do NOT assume, infer, or mention the author's gender.\n"
        "- Do NOT use gendered pronouns or gendered nouns such as he, she, his, her, himself, "
        "herself, chairman, or spokeswoman.\n"
        "- Use strictly gender-neutral language. If a reference to the author is needed, use only "
        "'the author', 'the researcher', or singular 'they/their'.\n"
        "- Avoid personal descriptors and avoid repeating the author's name unnecessarily.\n"
    )
    user = (
        f"Author ID: {author_id}\n\n"
        "Recurring research themes (from prior analysis):\n"
        f"{theme_block}\n\n"
        "Evidence notes (chunk summaries; do not quote papers):\n"
        f"{evidence_notes}\n\n"
        "Now write the 1–2 paragraph bio. Do not include any paper lists. "
        "Use fully gender-neutral language throughout."
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
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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



def load_model_and_tokenizer(model_id: str, allow_cpu_fallback: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not allow_cpu_fallback:
        raise RuntimeError(
            "No CUDA GPU detected for author_expertise_llama31_2.py. "
            "This script defaults to GPU-only because loading an 8B model on CPU/RAM "
            "often gets the process killed (exit code -9). "
            "Run it inside a GPU allocation or pass --allow-cpu-fallback if you really want CPU mode."
        )

    if has_cuda:
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        device_map = 'auto'
        print(f"CUDA detected: {torch.cuda.get_device_name(0)} | dtype={dtype}")
    else:
        dtype = torch.float16
        device_map = {'': 'cpu'}
        print('CUDA not detected. Falling back to CPU mode; this may be slow and memory-intensive.')

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model

# ----------------------------
# Per-author processing
# ----------------------------
def process_txt_input(
    path: str,
    tokenizer,
    model,
    max_input_tokens: int,
    map_max_new: int,
    reduce_max_new: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Dict[str, str]:
    fname = os.path.basename(path)
    author_id, source_text = parse_txt_summary_file(path)

    chunks = chunk_long_text(source_text, tokenizer, max_input_tokens)

    theme_candidates: List[str] = []
    evidence_notes_parts: List[str] = []

    for i, chunk_text in enumerate(chunks, start=1):
        messages = make_map_messages(author_id, chunk_text)
        chunk_out = generate_chat(
            model,
            tokenizer,
            messages,
            max_new_tokens=map_max_new,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        evidence_notes_parts.append(f"Chunk {i} themes:\n{chunk_out}")
        theme_candidates.extend(parse_theme_bullets(chunk_out))

    themes = theme_candidates[:12]
    evidence_notes = "\n\n".join(evidence_notes_parts)

    reduce_messages = make_reduce_messages(author_id, themes, evidence_notes)
    final_summary = generate_chat(
        model,
        tokenizer,
        reduce_messages,
        max_new_tokens=reduce_max_new,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    return {
        "author_id": author_id,
        "author_file": fname,
        "themes": "; ".join(themes),
        "summary": final_summary,
    }


def process_csv_input(
    path: str,
    tokenizer,
    model,
    max_papers: int,
    max_input_tokens: int,
    map_max_new: int,
    reduce_max_new: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    abstract_chars: int,
) -> Dict[str, str]:
    fname = os.path.basename(path)
    author_id = os.path.splitext(fname)[0]

    evidence_text, record_strings = build_evidence_from_csv(
        path, max_papers, abstract_chars)

    if not record_strings:
        return {
            "author_id": author_id,
            "author_file": fname,
            "themes": "",
            "summary": "No publications were found in the provided export.",
        }

    chunks = chunk_records(record_strings, tokenizer, max_input_tokens)

    theme_candidates: List[str] = []
    evidence_notes_parts: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        chunk_text = "\n\n---\n\n".join(chunk)
        messages = make_map_messages(author_id, chunk_text)
        chunk_out = generate_chat(
            model,
            tokenizer,
            messages,
            max_new_tokens=map_max_new,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        evidence_notes_parts.append(f"Chunk {i} themes:\n{chunk_out}")
        theme_candidates.extend(parse_theme_bullets(chunk_out))

    themes = theme_candidates[:12]
    evidence_notes = "\n\n".join(evidence_notes_parts)

    reduce_messages = make_reduce_messages(author_id, themes, evidence_notes)
    final_summary = generate_chat(
        model,
        tokenizer,
        reduce_messages,
        max_new_tokens=reduce_max_new,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    return {
        "author_id": author_id,
        "author_file": fname,
        "themes": "; ".join(themes),
        "summary": final_summary,
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # New flexible input handling
    ap.add_argument("--input-file", default="",
                    help="Path to one input file (.txt or .csv)")
    ap.add_argument("--input-dir", default="",
                    help="Folder containing input files (.txt or .csv)")

    # Keep backward compatibility
    ap.add_argument("--output-dir", default="",
                    help="Alias for --output-txt-dir")
    ap.add_argument(
        "--output-txt-dir",
        default="author_expertise_txt",
        help="Folder for per-author .txt outputs; set to empty string to disable",
    )
    ap.add_argument(
        "--output-csv",
        default="author_expertise_summaries.csv",
        help="Output CSV path; set to empty string to disable",
    )

    ap.add_argument(
        "--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max-papers", type=int, default=250,
                    help="Cap papers per author")
    ap.add_argument("--max-input-tokens", type=int,
                    default=6000, help="Max tokens per chunk")
    ap.add_argument("--map-max-new", type=int, default=200,
                    help="Max new tokens for map step")
    ap.add_argument("--reduce-max-new", type=int, default=512,
                    help="Max new tokens for reduce step")
    ap.add_argument("--temperature", type=float, default=0.25,
                    help="Lower = less creative")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--abstract-chars", type=int, default=260,
                    help="Max abstract chars per paper")
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="Allow CPU mode if no CUDA GPU is visible. Not recommended for 8B models.")
    args = ap.parse_args()

    allowed_exts = (".txt", ".csv")
    paths = resolve_input_paths(args.input_file, args.input_dir, allowed_exts)
    input_mode = infer_mode_from_paths(paths)

    output_txt_dir = args.output_dir.strip(
    ) if args.output_dir else args.output_txt_dir.strip()
    if output_txt_dir:
        os.makedirs(output_txt_dir, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(
        args.model_id,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )

    results: List[Dict[str, str]] = []

    for path in paths:
        print(f"\n--- Processing {os.path.basename(path)} ---")

        if input_mode == "txt":
            result = process_txt_input(
                path=path,
                tokenizer=tokenizer,
                model=model,
                max_input_tokens=args.max_input_tokens,
                map_max_new=args.map_max_new,
                reduce_max_new=args.reduce_max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            result = process_csv_input(
                path=path,
                tokenizer=tokenizer,
                model=model,
                max_papers=args.max_papers,
                max_input_tokens=args.max_input_tokens,
                map_max_new=args.map_max_new,
                reduce_max_new=args.reduce_max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                abstract_chars=args.abstract_chars,
            )

        results.append(result)

        if output_txt_dir:
            out_path = os.path.join(
                output_txt_dir, f"{result['author_id']}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("THEMES:\n")
                if result["themes"].strip():
                    for t in result["themes"].split("; "):
                        if t.strip():
                            f.write(f"- {t.strip()}\n")
                else:
                    f.write("(none)\n")
                f.write("\nSUMMARY:\n")
                f.write(result["summary"].strip() + "\n")
            print(f"✅ Wrote: {out_path}")

    if args.output_csv.strip():
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"✅ Wrote: {args.output_csv}")

    if output_txt_dir:
        print(f"✅ Wrote per-author txt files to: {output_txt_dir}")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
