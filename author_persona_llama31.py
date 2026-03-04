#!/usr/bin/env python3
"""
makeMusicPersonas.py

Input:  author_expertise_summaries.csv (or similar) containing at least:
  - author_id (or author_name)
  - themes (optional)
  - summary (required)

Output: outputs/author_music_personas.csv with:
  author_label, artist_name, persona_bio, album_title, tracklist, status

Also (optional): outputs/author_music_personas_txt/<author_label>.txt

Notes:
- Uses an instruction-tuned LLM via transformers (default: Meta-Llama-3.1-8B-Instruct).
- Forces STRICT JSON output for reliable parsing.
- Includes guardrails: no paper titles, no citations, no institutions, no “as an AI…” etc.
"""

import json
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd

# ----------------------------
# SETTINGS (edit these)
# ----------------------------
INPUT_CSV = "author_expertise_summaries.csv"

OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "author_music_personas.csv")

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Creative generation knobs (turn down if you want less flair)
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.85
TOP_P = 0.92
REPETITION_PENALTY = 1.08

# Optional: write per-author txt outputs (inside outputs/)
OUTPUT_TXT_DIR = os.path.join(OUTPUT_DIR, "author_music_personas_txt")  # set to None to disable

# Column detection (your summarizer output may vary)
AUTHOR_ID_COL_CANDIDATES = ["author_id", "Author ID", "AU-ID", "author"]
AUTHOR_NAME_COL_CANDIDATES = ["author_name", "Author Name", "name"]
THEMES_COL_CANDIDATES = ["themes", "Themes", "THEMES"]
SUMMARY_COL_CANDIDATES = ["summary", "Summary", "SUMMARY"]


# ----------------------------
# Helpers
# ----------------------------
def pick_existing_col(cols, candidates) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def extract_json_object(text: str) -> Optional[Dict]:
    """
    Tries hard to find and parse the first JSON object in model output.
    """
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1].strip()
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

    if not isinstance(obj["tracklist"], list) or len(obj["tracklist"]) < 6:
        return False, "tracklist must be a list with at least 6 items"

    obj["tracklist"] = [clean_text(str(t)) for t in obj["tracklist"] if clean_text(str(t))]
    if len(obj["tracklist"]) < 6:
        return False, "tracklist items were empty after cleaning"

    if len(clean_text(obj["persona_bio"])) < 80:
        return False, "persona_bio too short"

    return True, "ok"


def make_persona_messages(author_label: str, themes: str, summary: str):
    sys = (
        "You are a witty but respectful creative writer.\n"
        "You will transform a researcher expertise summary into a fictional musical persona.\n"
        "Hard rules:\n"
        "- DO NOT mention real institutions, grants, paper titles, citation counts, or publication venues.\n"
        "- DO NOT fabricate specific real-world biographical facts (schools, employers, dates, awards).\n"
        "- DO NOT include disclaimers like 'as an AI'.\n"
        "- Keep it fun, but not insulting.\n"
        "- Incorporate wordplay/puns based on the expertise when possible.\n"
        "- Output MUST be valid JSON only (no markdown, no extra text).\n"
    )

    user = (
        f"AUTHOR LABEL: {author_label}\n\n"
        f"THEMES (may be empty):\n{themes}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "Create a fictional music persona with EXACTLY this JSON schema:\n"
        "{\n"
        '  "artist_name": "string (1–6 words)",\n'
        '  "persona_bio": "string (120–220 words, reads like a music press bio; show career progression + signature themes)",\n'
        '  "album_title": "string (2–8 words)",\n'
        '  "tracklist": ["Track 01 - ...", "Track 02 - ...", ...]  // 8–12 tracks total\n'
        "}\n\n"
        "Style guidance:\n"
        "- Bio should subtly map research themes to musical motifs (e.g., 'segmentation' -> 'splitting signals', 'diffusion' -> 'reverb', etc.).\n"
        "- Track titles should be punchy and pun-friendly, but still readable.\n"
        "- Avoid using the author’s real name in the artist_name (unless it’s clearly transformed/punned).\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def generate_chat(model, tokenizer, messages) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Remove prompt echo if present
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()

    return text.strip()


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Could not find INPUT_CSV: {INPUT_CSV}")

    # --- NEW: ensure output folders exist ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if OUTPUT_TXT_DIR:
        os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    cols = list(df.columns)

    author_id_col = pick_existing_col(cols, AUTHOR_ID_COL_CANDIDATES)
    author_name_col = pick_existing_col(cols, AUTHOR_NAME_COL_CANDIDATES)
    themes_col = pick_existing_col(cols, THEMES_COL_CANDIDATES)
    summary_col = pick_existing_col(cols, SUMMARY_COL_CANDIDATES)

    if not summary_col:
        raise ValueError(
            f"Could not find a summary column. Tried: {SUMMARY_COL_CANDIDATES}. "
            f"Found columns: {cols}"
        )

    label_col = author_id_col or author_name_col or summary_col

    print(f"Loading tokenizer/model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )

    out_rows = []

    for idx, row in df.iterrows():
        author_label = clean_text(row.get(label_col, f"row_{idx}")) or f"row_{idx}"
        themes = clean_text(row.get(themes_col, "")) if themes_col else ""
        summary = clean_text(row.get(summary_col, ""))

        if not summary:
            out_rows.append(
                {
                    "author_label": author_label,
                    "artist_name": "",
                    "persona_bio": "",
                    "album_title": "",
                    "tracklist": "",
                    "status": "skipped_empty_summary",
                }
            )
            continue

        messages = make_persona_messages(author_label=author_label, themes=themes, summary=summary)
        raw = generate_chat(model, tokenizer, messages)

        obj = extract_json_object(raw)
        if obj is None:
            out_rows.append(
                {
                    "author_label": author_label,
                    "artist_name": "",
                    "persona_bio": "",
                    "album_title": "",
                    "tracklist": "",
                    "status": "json_parse_failed",
                }
            )
            continue

        ok, reason = validate_persona(obj)
        if not ok:
            out_rows.append(
                {
                    "author_label": author_label,
                    "artist_name": obj.get("artist_name", ""),
                    "persona_bio": obj.get("persona_bio", ""),
                    "album_title": obj.get("album_title", ""),
                    "tracklist": json.dumps(obj.get("tracklist", []), ensure_ascii=False),
                    "status": f"invalid_output:{reason}",
                }
            )
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
            }
        )

        if OUTPUT_TXT_DIR:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", author_label)[:80]
            with open(os.path.join(OUTPUT_TXT_DIR, f"{safe_name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Artist: {out_rows[-1]['artist_name']}\n")
                f.write(f"Album:  {out_rows[-1]['album_title']}\n\n")
                f.write(out_rows[-1]["persona_bio"] + "\n\n")
                f.write("Tracklist:\n" + tracklist_str + "\n")

        print(f"[{idx+1}/{len(df)}] {author_label} -> {out_rows[-1]['artist_name']}")

    pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Wrote personas CSV: {OUTPUT_CSV}")
    if OUTPUT_TXT_DIR:
        print(f"✅ Wrote per-author TXT files to: {OUTPUT_TXT_DIR}/")


if __name__ == "__main__":
    main()
