#!/usr/bin/env python3
"""
makeMusicPersonas.py

Reads per-author expertise summaries from TXT files in:
  author_expertise_txt/*.txt

Each TXT is expected to look like:
  THEMES:
  - ...
  SUMMARY:
  ...

Generates a fictional musical persona per author:
  1) persona_bio (career progression + themes as a music press bio)
  2) artist_name
  3) album_title + tracklist (8–12 tracks)

Outputs:
  outputs/author_music_personas.csv
  outputs/author_music_personas_txt/<author_label>.txt  (optional)

Key reliability features:
- Decodes ONLY newly generated tokens (prevents prompt/schema braces from breaking JSON parsing)
- Strict JSON-only output request + robust JSON extraction + optional repair pass
"""

import glob
import json
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd

# ----------------------------
# SETTINGS (edit these)
# ----------------------------
INPUT_DIR = "author_expertise_txt"

OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "author_music_personas.csv")

# Optional: write per-author persona TXT outputs (inside outputs/)
OUTPUT_TXT_DIR = os.path.join(
    OUTPUT_DIR, "author_music_personas_txt")  # set to None to disable

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Creative generation knobs (turn down if you want less flair)
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.85
TOP_P = 0.92
REPETITION_PENALTY = 1.08

# If JSON parse fails, do one "repair" attempt
ENABLE_JSON_REPAIR = True


# ----------------------------
# Helpers
# ----------------------------
def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def parse_themes_and_summary(file_text: str) -> Tuple[str, str]:
    """
    Extracts THEMES and SUMMARY from a text blob.
    If SUMMARY: is missing, treats entire content as summary.
    """
    text = file_text.replace("\r\n", "\n")

    if "SUMMARY:" not in text:
        return "", text.strip()

    before, after = text.split("SUMMARY:", 1)
    themes = before.replace("THEMES:", "").strip()
    summary = after.strip()
    return themes, summary


def extract_json_object(text: str) -> Optional[Dict]:
    """
    Find and parse the first JSON object in model output.
    Works even if the model adds a little extra text (we still try to carve out { ... }).
    """
    if not text:
        return None

    t = text.strip()
    # Remove fenced code blocks if present
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = t[start: end + 1].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Light repair: remove trailing commas
        candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError:
            return None


def validate_persona(obj: Dict) -> Tuple[bool, str]:
    required = ["artist_name", "persona_bio", "album_title", "tracklist"]
    for k in required:
        if k not in obj:
            return False, f"Missing key: {k}"

    if not isinstance(obj["tracklist"], list):
        return False, "tracklist must be a list"

    # Clean tracklist
    obj["tracklist"] = [clean_text(str(t))
                        for t in obj["tracklist"] if clean_text(str(t))]
    if not (8 <= len(obj["tracklist"]) <= 12):
        return False, "tracklist must have 8–12 non-empty items"

    if len(clean_text(obj["persona_bio"])) < 100:
        return False, "persona_bio too short"

    if len(clean_text(obj["artist_name"])) < 2:
        return False, "artist_name too short"

    if len(clean_text(obj["album_title"])) < 2:
        return False, "album_title too short"

    return True, "ok"


def make_persona_messages(author_label: str, themes: str, summary: str):
    sys = (
        "You are a witty but respectful creative writer.\n"
        "Transform a researcher expertise summary into a fictional musical persona.\n\n"
        "Hard rules:\n"
        "- Output MUST be valid JSON only. No markdown. No commentary.\n"
        "- DO NOT mention real institutions, grants, paper titles, citation counts, or publication venues.\n"
        "- DO NOT fabricate specific real-world biographical facts (schools, employers, dates, awards).\n"
        "- DO NOT include disclaimers like 'as an AI'.\n"
        "- Keep it fun, but not insulting.\n"
        "- Try to incorporate wordplay/puns based on the expertise when possible.\n"
    )

    user = (
        f"AUTHOR LABEL (file stem, may include real name): {author_label}\n\n"
        f"THEMES (may be empty):\n{themes}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "Return EXACTLY this JSON schema:\n"
        "{\n"
        '  "artist_name": "string (1–6 words, avoid using the author label verbatim; pun/transform if used)",\n'
        '  "persona_bio": "string (120–220 words, reads like a music press bio; implies career progression + signature themes)",\n'
        '  "album_title": "string (2–8 words)",\n'
        '  "tracklist": ["Track 01 - ...", "Track 02 - ...", ...]  // 8–12 tracks\n'
        "}\n\n"
        "Style guidance:\n"
        "- Map research themes to musical motifs (e.g., segmentation->splitting signals, diffusion->reverb, transformers->remix).\n"
        "- Track titles should be punchy, clever, and pun-friendly, but readable.\n"
        "- Avoid anything that reads like an academic CV.\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def generate_chat(model, tokenizer, messages) -> str:
    """
    IMPORTANT: decode only newly-generated tokens, not the prompt.
    This prevents prompt braces (JSON schema) from confusing JSON extraction.
    """
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

    gen_tokens = out[0][input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()


def repair_to_json(model, tokenizer, bad_text: str) -> str:
    sys = (
        "You are a strict JSON reformatter. "
        "Return ONLY valid JSON matching the required schema. No extra text."
    )
    user = (
        "Convert the following text into valid JSON with keys "
        "artist_name, persona_bio, album_title, tracklist (list of strings).\n\n"
        f"TEXT:\n{bad_text}"
    )
    messages = [{"role": "system", "content": sys},
                {"role": "user", "content": user}]
    return generate_chat(model, tokenizer, messages)


def safe_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ----------------------------
# Main
# ----------------------------
def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Ensure output folders exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if OUTPUT_TXT_DIR:
        os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    # Gather source files
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not files:
        raise RuntimeError(
            f"No TXT files found in: {INPUT_DIR} (expected {INPUT_DIR}/*.txt)")

    print(f"Found {len(files)} author summary TXT files in: {INPUT_DIR}/")

    # Load model
    print(f"Loading tokenizer/model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        device_note = f"CUDA detected, using dtype={dtype}"
    else:
        dtype = torch.float32
        device_note = "CUDA not detected, using CPU (this will be slow)"

    print(device_note)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )

    out_rows = []

    for idx, path in enumerate(files):
        author_label = safe_stem(path)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        themes, summary = parse_themes_and_summary(content)
        themes = clean_text(themes)
        summary = clean_text(summary)

        if not summary:
            out_rows.append(
                {
                    "author_label": author_label,
                    "artist_name": "",
                    "persona_bio": "",
                    "album_title": "",
                    "tracklist": "",
                    "status": "skipped_empty_summary",
                    "source_file": path,
                }
            )
            continue

        messages = make_persona_messages(
            author_label=author_label, themes=themes, summary=summary)
        raw = generate_chat(model, tokenizer, messages)

        obj = extract_json_object(raw)

        if obj is None and ENABLE_JSON_REPAIR:
            repaired = repair_to_json(model, tokenizer, raw)
            obj = extract_json_object(repaired)

        if obj is None:
            out_rows.append(
                {
                    "author_label": author_label,
                    "artist_name": "",
                    "persona_bio": "",
                    "album_title": "",
                    "tracklist": "",
                    "status": "json_parse_failed",
                    "source_file": path,
                }
            )
            print(f"[{idx+1}/{len(files)}] {author_label} -> json_parse_failed")
            continue

        ok, reason = validate_persona(obj)
        if not ok:
            out_rows.append(
                {
                    "author_label": author_label,
                    "artist_name": clean_text(obj.get("artist_name", "")),
                    "persona_bio": clean_text(obj.get("persona_bio", "")),
                    "album_title": clean_text(obj.get("album_title", "")),
                    "tracklist": json.dumps(obj.get("tracklist", []), ensure_ascii=False),
                    "status": f"invalid_output:{reason}",
                    "source_file": path,
                }
            )
            print(
                f"[{idx+1}/{len(files)}] {author_label} -> invalid_output:{reason}")
            continue

        tracklist_str = "\n".join(obj["tracklist"])

        out_rows.append(
            {
                "author_label": author_label,
                "artist_name": clean_text(obj["artist_name"]),
                "persona_bio": clean_text(obj["persona_bio"]),
                "album_title": clean_text(obj["album_title"]),
                "tracklist": tracklist_str,
                "status": "ok",
                "source_file": path,
            }
        )

        if OUTPUT_TXT_DIR:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", author_label)[:80]
            out_path = os.path.join(OUTPUT_TXT_DIR, f"{safe_name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Artist: {out_rows[-1]['artist_name']}\n")
                f.write(f"Album:  {out_rows[-1]['album_title']}\n\n")
                f.write(out_rows[-1]["persona_bio"] + "\n\n")
                f.write("Tracklist:\n" + tracklist_str + "\n")

        print(
            f"[{idx+1}/{len(files)}] {author_label} -> {out_rows[-1]['artist_name']}")

    pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Wrote personas CSV: {OUTPUT_CSV}")
    if OUTPUT_TXT_DIR:
        print(f"✅ Wrote per-author persona TXT files to: {OUTPUT_TXT_DIR}/")


if __name__ == "__main__":
    main()
