#!/usr/bin/env python3
"""
author_persona_llama31.py

Reads per-author expertise summaries from TXT files produced by author_expertise_llama31_2.py:
  THEMES:
  - ...
  SUMMARY:
  ...

Generates a fictional musical persona per author as TXT files.

Supports:
- single mode via --input-file
- batch mode via --input-dir

Outputs:
- one TXT file per author in --output-dir
- optional CSV manifest (disabled by default)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.85
TOP_P = 0.92
REPETITION_PENALTY = 1.08
ENABLE_JSON_REPAIR = True

def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()

def resolve_input_paths(input_file: str, input_dir: str) -> List[str]:
    if bool(input_file) == bool(input_dir):
        raise ValueError("Provide exactly one of --input-file or --input-dir")
    if input_file:
        p = Path(input_file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        if p.suffix.lower() != ".txt":
            raise ValueError("author_persona_llama31.py expects TXT input files.")
        return [str(p)]
    pdir = Path(input_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"Input directory not found: {pdir}")
    files = sorted(glob.glob(str(pdir / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No TXT files found in: {pdir}")
    return files

def parse_themes_and_summary(file_text: str) -> Tuple[str, str]:
    text = file_text.replace("\r\n", "\n")
    if "SUMMARY:" not in text:
        return "", text.strip()
    before, after = text.split("SUMMARY:", 1)
    themes = before.replace("THEMES:", "").strip()
    summary = after.strip()
    return themes, summary

def extract_json_object(text: str) -> Optional[Dict]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    start = t.find("{"); end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = t[start:end+1].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError:
            return None

def validate_persona(obj: Dict) -> Tuple[bool, str]:
    for k in ["artist_name", "persona_bio", "album_title", "tracklist"]:
        if k not in obj:
            return False, f"Missing key: {k}"
    if not isinstance(obj["tracklist"], list):
        return False, "tracklist must be a list"
    obj["tracklist"] = [clean_text(str(t)) for t in obj["tracklist"] if clean_text(str(t))]
    if not (8 <= len(obj["tracklist"]) <= 12):
        return False, "tracklist must have 8–12 non-empty items"
    if len(clean_text(obj["persona_bio"])) < 100:
        return False, "persona_bio too short"
    return True, "ok"

def make_persona_messages(author_label: str, themes: str, summary: str):
    sys = (
        "You are a witty but respectful creative writer.\n"
        "Transform a researcher expertise summary into a fictional musical persona.\n\n"
        "Hard rules:\n"
        "- Output MUST be valid JSON only. No markdown. No commentary.\n"
        "- DO NOT mention real institutions, grants, paper titles, citation counts, or publication venues.\n"
        "- DO NOT fabricate specific real-world biographical facts.\n"
        "- Keep it fun, but not insulting.\n"
    )
    user = (
        f"AUTHOR LABEL: {author_label}\n\n"
        f"THEMES:\n{themes}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "Return EXACTLY this JSON schema:\n"
        "{\n"
        '  "artist_name": "string",\n'
        '  "persona_bio": "string",\n'
        '  "album_title": "string",\n'
        '  "tracklist": ["Track 01 - ...", "Track 02 - ..."]\n'
        "}\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def generate_chat(model, tokenizer, messages) -> str:
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

def repair_to_json(model, tokenizer, bad_text: str) -> str:
    messages = [
        {"role": "system", "content": "Return only valid JSON with keys artist_name, persona_bio, album_title, tracklist."},
        {"role": "user", "content": f"Convert this into valid JSON:\n\n{bad_text}"},
    ]
    return generate_chat(model, tokenizer, messages)

def write_persona_txt(out_path: str, artist_name: str, album_title: str, persona_bio: str, tracklist: List[str]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Artist: {artist_name}\n")
        f.write(f"Album: {album_title}\n\n")
        f.write(clean_text(persona_bio) + "\n\n")
        f.write("Tracklist:\n")
        for track in tracklist:
            f.write(clean_text(track) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="")
    ap.add_argument("--input-dir", default="")
    ap.add_argument("--output-dir", default="outputs/author_music_personas_txt", help="Folder for per-author persona TXT files")
    ap.add_argument("--output-csv", default="", help="Optional CSV manifest path; leave empty to disable")
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    files = resolve_input_paths(args.input_file, args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto")

    out_rows = []
    for idx, path in enumerate(files, start=1):
        author_label = Path(path).stem
        content = Path(path).read_text(encoding="utf-8")
        themes, summary = parse_themes_and_summary(content)
        themes = clean_text(themes)
        summary = clean_text(summary)
        if not summary:
            raise ValueError(f"{path}: summary was empty.")
        raw = generate_chat(model, tokenizer, make_persona_messages(author_label, themes, summary))
        obj = extract_json_object(raw)
        if obj is None and ENABLE_JSON_REPAIR:
            obj = extract_json_object(repair_to_json(model, tokenizer, raw))
        if obj is None:
            raise ValueError(f"{author_label}: model output could not be parsed as JSON.")
        ok, reason = validate_persona(obj)
        if not ok:
            raise ValueError(f"{author_label}: invalid persona output ({reason}).")
        out_path = os.path.join(args.output_dir, f"{author_label}.txt")
        write_persona_txt(out_path, obj["artist_name"], obj["album_title"], obj["persona_bio"], obj["tracklist"])
        out_rows.append({
            "author_label": author_label,
            "artist_name": clean_text(obj["artist_name"]),
            "album_title": clean_text(obj["album_title"]),
            "tracklist_count": len(obj["tracklist"]),
            "output_txt": out_path,
        })
        print(f"[{idx}/{len(files)}] Wrote: {out_path}")

    if args.output_csv.strip():
        pd.DataFrame(out_rows).to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"✅ Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
