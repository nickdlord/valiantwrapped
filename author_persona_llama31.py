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

This version uses plain-text model output rather than JSON.
Expected model output format:

Artist: ...
Album: ...

Bio:
...

Tracklist:
1. ...
2. ...
...
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.55
TOP_P = 0.90
REPETITION_PENALTY = 1.08


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
            raise ValueError(
                "author_persona_llama31.py expects TXT input files.")
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


def make_persona_messages(author_label: str, themes: str, summary: str):
    sys_msg = (
        "You are a witty but respectful creative writer.\n"
        "Transform a researcher expertise summary into a fictional musical persona.\n\n"
        "Hard rules:\n"
        "- Output plain text only.\n"
        "- Do NOT output JSON.\n"
        "- Do NOT output markdown code fences.\n"
        "- DO NOT mention real institutions, grants, paper titles, citation counts, or publication venues.\n"
        "- DO NOT fabricate specific real-world biographical facts.\n"
        "- Keep it fun, vivid, and respectful.\n"
        "- The bio should be one short paragraph, around 90-180 words.\n"
        "- Include 8 to 12 track titles.\n"
        "- Follow the exact output structure shown below.\n"
    )

    user_msg = (
        f"AUTHOR LABEL: {author_label}\n\n"
        f"THEMES:\n{themes}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "Return EXACTLY in this format:\n\n"
        "Artist: <artist name>\n"
        "Album: <album title>\n\n"
        "Bio:\n"
        "<one paragraph bio>\n\n"
        "Tracklist:\n"
        "1. <track title>\n"
        "2. <track title>\n"
        "3. <track title>\n"
        "4. <track title>\n"
        "5. <track title>\n"
        "6. <track title>\n"
        "7. <track title>\n"
        "8. <track title>\n"
        "9. <track title>\n"
        "10. <track title>\n"
    )

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def generate_chat(model, tokenizer, messages) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
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


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_persona_text(raw_text: str) -> Optional[Dict[str, object]]:
    """
    Expected format:

    Artist: ...
    Album: ...

    Bio:
    ...

    Tracklist:
    1. ...
    2. ...
    ...
    """
    if not raw_text:
        return None

    text = strip_code_fences(raw_text).replace("\r\n", "\n").strip()

    artist_match = re.search(r"(?im)^Artist:\s*(.+)$", text)
    album_match = re.search(r"(?im)^Album:\s*(.+)$", text)

    bio_match = re.search(
        r"(?is)^.*?^Bio:\s*(.*?)\s*^Tracklist:\s*",
        text,
        flags=re.MULTILINE,
    )

    tracklist_match = re.search(
        r"(?is)^.*?^Tracklist:\s*(.*)$",
        text,
        flags=re.MULTILINE,
    )

    if not artist_match or not album_match or not bio_match or not tracklist_match:
        return None

    artist_name = clean_text(artist_match.group(1))
    album_title = clean_text(album_match.group(1))
    persona_bio = clean_text(bio_match.group(1))
    track_blob = tracklist_match.group(1).strip()

    tracks: List[str] = []
    for line in track_blob.splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove numbering like "1. ", "01. ", "- ", "* "
        line = re.sub(r"^\s*(?:\d{1,2}[.)-]\s*|[-*]\s*)", "", line)
        line = clean_text(line)

        if line:
            tracks.append(line)

    if not artist_name or not album_title or not persona_bio:
        return None

    return {
        "artist_name": artist_name,
        "album_title": album_title,
        "persona_bio": persona_bio,
        "tracklist": tracks,
    }


def validate_persona(obj: Dict[str, object]) -> Tuple[bool, str]:
    artist_name = clean_text(obj.get("artist_name", ""))
    album_title = clean_text(obj.get("album_title", ""))
    persona_bio = clean_text(obj.get("persona_bio", ""))

    tracklist_raw = obj.get("tracklist", [])
    if not isinstance(tracklist_raw, list):
        return False, "tracklist must be a list"

    tracklist = [clean_text(str(t))
                 for t in tracklist_raw if clean_text(str(t))]
    obj["tracklist"] = tracklist

    if not artist_name:
        return False, "missing artist_name"
    if not album_title:
        return False, "missing album_title"
    if len(persona_bio) < 80:
        return False, "persona_bio too short"
    if not (8 <= len(tracklist) <= 12):
        return False, "tracklist must have 8-12 non-empty items"

    return True, "ok"


def repair_persona_text(model, tokenizer, bad_text: str) -> str:
    """
    Ask the model to reformat bad output into the required plain-text structure.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Reformat the provided content into the exact plain-text structure below.\n"
                "Do not output JSON. Do not use code fences.\n"
                "Output format:\n"
                "Artist: <artist name>\n"
                "Album: <album title>\n\n"
                "Bio:\n"
                "<one paragraph bio>\n\n"
                "Tracklist:\n"
                "1. <track title>\n"
                "2. <track title>\n"
                "...\n"
            ),
        },
        {
            "role": "user",
            "content": f"Reformat this into the required structure:\n\n{bad_text}",
        },
    ]
    return generate_chat(model, tokenizer, messages)


def write_persona_txt(
    out_path: str,
    artist_name: str,
    album_title: str,
    persona_bio: str,
    tracklist: List[str],
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Artist: {clean_text(artist_name)}\n")
        f.write(f"Album: {clean_text(album_title)}\n\n")
        f.write("Bio:\n")
        f.write(clean_text(persona_bio) + "\n\n")
        f.write("Tracklist:\n")
        for i, track in enumerate(tracklist, start=1):
            f.write(f"{i}. {clean_text(track)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="")
    ap.add_argument("--input-dir", default="")
    ap.add_argument(
        "--output-dir",
        default="outputs/author_music_personas_txt",
        help="Folder for per-author persona TXT files",
    )
    ap.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV manifest path; leave empty to disable",
    )
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    out_rows = []

    for idx, path in enumerate(files, start=1):
        author_label = Path(path).stem
        content = Path(path).read_text(encoding="utf-8", errors="replace")

        themes, summary = parse_themes_and_summary(content)
        themes = themes.strip()
        summary = clean_text(summary)

        if not summary:
            raise ValueError(f"{path}: summary was empty.")

        raw = generate_chat(
            model,
            tokenizer,
            make_persona_messages(author_label, themes, summary),
        )

        obj = parse_persona_text(raw)

        if obj is None:
            repaired = repair_persona_text(model, tokenizer, raw)
            obj = parse_persona_text(repaired)

        if obj is None:
            raise ValueError(
                f"{author_label}: model output could not be parsed in plain-text persona format."
            )

        ok, reason = validate_persona(obj)
        if not ok:
            raise ValueError(
                f"{author_label}: invalid persona output ({reason}).")

        out_path = os.path.join(args.output_dir, f"{author_label}.txt")
        write_persona_txt(
            out_path,
            obj["artist_name"],
            obj["album_title"],
            obj["persona_bio"],
            obj["tracklist"],
        )

        out_rows.append({
            "author_label": author_label,
            "artist_name": clean_text(obj["artist_name"]),
            "album_title": clean_text(obj["album_title"]),
            "tracklist_count": len(obj["tracklist"]),
            "output_txt": out_path,
        })

        print(f"[{idx}/{len(files)}] Wrote: {out_path}")

    if args.output_csv.strip():
        pd.DataFrame(out_rows).to_csv(
            args.output_csv, index=False, encoding="utf-8")
        print(f"✅ Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
