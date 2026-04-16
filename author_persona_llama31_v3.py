#!/usr/bin/env python3
"""
author_persona_llama31_gender_cautious.py

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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.68
TOP_P = 0.92
REPETITION_PENALTY = 1.10
MAX_RETRIES = 2

GENERIC_PHRASES = [
    "turning data into melody",
    "blending science and sound",
    "genre-bending",
    "boundary-pushing",
    "sonic landscape",
    "ethereal soundscape",
    "merging precision with emotion",
    "translating research into rhythm",
    "where rigor meets rhythm",
    "beats and breakthroughs",
]

GENERIC_TRACK_WORDS = {
    "echo", "signal", "pulse", "spectrum", "rhythm", "frequency", "noise",
    "shadow", "light", "dream", "memory", "midnight", "machine", "algorithm",
    "horizon", "static", "gravity", "velocity", "blueprint", "circuit"
}


def clean_text(x: str) -> str:
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()


def canonical_author_label(value: str) -> str:
    text = clean_text(value).replace("\\", "/")
    text = os.path.basename(text)
    text = re.sub(r"\.(txt|csv)$", "", text, flags=re.IGNORECASE)
    return text


def normalize_for_similarity(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    a_norm = normalize_for_similarity(a)
    b_norm = normalize_for_similarity(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def source_has_explicit_gender(text: str) -> bool:
    t = f" {clean_text(text).lower()} "
    explicit_markers = [
        " he ", " him ", " his ",
        " she ", " her ", " hers ",
        " he/", " she/", " they/",
        " mr. ", " ms. ", " mrs. ", " miss ",
    ]
    return any(marker in t for marker in explicit_markers)


def neutralize_gendered_language(text: str) -> str:
    if not text:
        return text

    replacements = [
        (r"\bhe\b", "they"),
        (r"\bshe\b", "they"),
        (r"\bhim\b", "them"),
        (r"\bhis\b", "their"),
        (r"\bhers\b", "theirs"),
        (r"\bher\b", "their"),
        (r"\bfrontman\b", "frontperson"),
        (r"\bfrontwoman\b", "frontperson"),
        (r"\bsongstress\b", "singer"),
        (r"\bleading man\b", "lead performer"),
        (r"\bleading woman\b", "lead performer"),
        (r"\bking\b", "icon"),
        (r"\bqueen\b", "icon"),
        (r"\bmale vocalist\b", "vocalist"),
        (r"\bfemale vocalist\b", "vocalist"),
    ]

    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    out = re.sub(r"\b[Tt]hey is\b", "they are", out)
    out = re.sub(r"\b[Tt]hey was\b", "they were", out)
    out = re.sub(r"\b[Tt]hey has\b", "they have", out)
    out = re.sub(r"\b[Tt]hemself\b", "themselves", out)
    return out


def resolve_input_paths(input_file: str, input_dir: str) -> List[str]:
    if bool(input_file) == bool(input_dir):
        raise ValueError("Provide exactly one of --input-file or --input-dir")

    if input_file:
        p = Path(input_file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        if p.suffix.lower() != ".txt":
            raise ValueError(
                "author_persona_llama31_gender_cautious.py expects TXT input files.")
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


def parse_existing_persona_file(path: Path) -> Optional[Dict[str, object]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return parse_persona_text(raw)


def load_existing_personas(output_dir: str, exclude_author_label: str = "") -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    pdir = Path(output_dir)
    if not pdir.exists():
        return out

    for p in sorted(pdir.glob("*.txt")):
        label = canonical_author_label(p.stem)
        if exclude_author_label and label == canonical_author_label(exclude_author_label):
            continue
        obj = parse_existing_persona_file(p)
        if obj:
            obj = dict(obj)
            obj["author_label"] = label
            out.append(obj)
    return out


def prior_examples_text(priors: List[Dict[str, object]], max_examples: int = 8) -> str:
    if not priors:
        return "None yet."

    lines: List[str] = []
    for p in priors[-max_examples:]:
        lines.append(
            f"- {clean_text(p.get('author_label', 'unknown'))}: "
            f"Artist='{clean_text(p.get('artist_name', ''))}', "
            f"Album='{clean_text(p.get('album_title', ''))}'"
        )
    return "\n".join(lines)


def make_persona_messages(
    author_label: str,
    themes: str,
    summary: str,
    prior_personas: List[Dict[str, object]],
    attempt_num: int = 1,
    retry_reason: str = "",
):
    sys_msg = (
        "You are a witty but respectful creative writer.\n"
        "Transform a researcher's expertise summary into a fictional musical persona.\n\n"
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
        "- The output must feel specific to this author's themes, not generic.\n"
        "- The artist name, album title, and track titles should be clearly shaped by the themes and summary.\n"
        "- Avoid vague stock phrases like 'turning data into melody', 'blending science and sound', or generic futuristic music clichés.\n"
        "- Avoid repeating names, album concepts, and phrasing used for other authors.\n"
    )

    retry_block = ""
    if retry_reason:
        retry_block = (
            f"\nRETRY GUIDANCE:\n"
            f"- The previous attempt was rejected because: {retry_reason}\n"
            f"- Produce a more distinctive artist name, album concept, and bio.\n"
            f"- Use different wording, different imagery, and more theme-specific track titles.\n"
        )

    user_msg = (
        f"AUTHOR LABEL: {author_label}\n\n"
        f"THEMES:\n{themes}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "Previously accepted personas from other authors. Do NOT closely reuse these names or concepts:\n"
        f"{prior_examples_text(prior_personas)}\n"
        f"{retry_block}\n"
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
        messages, tokenize=False, add_generation_prompt=True
    )
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


def contains_generic_language(text: str) -> bool:
    t = normalize_for_similarity(text)
    return any(phrase in t for phrase in GENERIC_PHRASES)


def low_diversity_tracklist(tracklist: List[str]) -> bool:
    if not tracklist:
        return False
    lowered = [normalize_for_similarity(t) for t in tracklist]
    unique_ratio = len(set(lowered)) / max(1, len(lowered))
    if unique_ratio < 0.8:
        return True

    generic_count = 0
    for t in lowered:
        words = set(t.split())
        if words & GENERIC_TRACK_WORDS:
            generic_count += 1
    return generic_count >= max(5, len(tracklist) // 2)


def find_similarity_issue(
    candidate: Dict[str, object],
    prior_personas: List[Dict[str, object]],
) -> Tuple[bool, str]:
    cand_artist = clean_text(candidate.get("artist_name", ""))
    cand_album = clean_text(candidate.get("album_title", ""))
    cand_bio = clean_text(candidate.get("persona_bio", ""))

    if contains_generic_language(cand_artist) or contains_generic_language(cand_album) or contains_generic_language(cand_bio):
        return True, "it used overly generic persona wording"

    if low_diversity_tracklist(candidate.get("tracklist", [])):
        return True, "the tracklist felt too generic or repetitive"

    for prev in prior_personas:
        prev_artist = clean_text(prev.get("artist_name", ""))
        prev_album = clean_text(prev.get("album_title", ""))
        prev_bio = clean_text(prev.get("persona_bio", ""))

        if similarity(cand_artist, prev_artist) >= 0.88:
            return True, f"the artist name was too similar to an earlier one ({prev_artist})"

        if similarity(cand_album, prev_album) >= 0.88:
            return True, f"the album title was too similar to an earlier one ({prev_album})"

        if similarity(cand_bio, prev_bio) >= 0.82:
            return True, f"the bio was too similar to an earlier one ({prev_artist} / {prev_album})"

    return False, ""


def repair_persona_text(model, tokenizer, bad_text: str) -> str:
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


def generate_persona_with_dedup(
    model,
    tokenizer,
    author_label: str,
    themes: str,
    summary: str,
    prior_personas: List[Dict[str, object]],
    explicit_gender_supported: bool,
) -> Dict[str, object]:
    last_failure = ""

    for attempt in range(1, MAX_RETRIES + 2):
        raw = generate_chat(
            model,
            tokenizer,
            make_persona_messages(
                author_label=author_label,
                themes=themes,
                summary=summary,
                prior_personas=prior_personas,
                attempt_num=attempt,
                retry_reason=last_failure,
            ),
        )

        obj = parse_persona_text(raw)

        if obj is None:
            repaired = repair_persona_text(model, tokenizer, raw)
            obj = parse_persona_text(repaired)

        if obj is None:
            last_failure = "the output could not be parsed into the required plain-text format"
            continue

        if not explicit_gender_supported:
            obj["persona_bio"] = neutralize_gendered_language(
                clean_text(obj.get("persona_bio", "")))

        ok, reason = validate_persona(obj)
        if not ok:
            last_failure = f"invalid structure: {reason}"
            continue

        has_issue, issue_reason = find_similarity_issue(obj, prior_personas)
        if has_issue:
            last_failure = issue_reason
            continue

        return obj

    raise ValueError(
        f"{author_label}: could not produce a distinctive persona after retries ({last_failure or 'unknown reason'})."
    )


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
    accepted_personas = load_existing_personas(args.output_dir)

    for idx, path in enumerate(files, start=1):
        author_label = canonical_author_label(Path(path).stem)
        content = Path(path).read_text(encoding="utf-8", errors="replace")

        themes, summary = parse_themes_and_summary(content)
        themes = themes.strip()
        summary = clean_text(summary)
        explicit_gender_supported = source_has_explicit_gender(content)

        if not summary:
            raise ValueError(f"{path}: summary was empty.")

        prior_personas = load_existing_personas(
            args.output_dir, exclude_author_label=author_label)
        for in_memory in accepted_personas:
            if canonical_author_label(in_memory.get("author_label", "")) != author_label:
                prior_personas.append(in_memory)

        unique_priors = []
        seen_keys = set()
        for p in prior_personas:
            key = (
                normalize_for_similarity(clean_text(p.get("artist_name", ""))),
                normalize_for_similarity(clean_text(p.get("album_title", ""))),
                normalize_for_similarity(clean_text(
                    p.get("persona_bio", ""))[:120]),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                unique_priors.append(p)
        prior_personas = unique_priors

        obj = generate_persona_with_dedup(
            model=model,
            tokenizer=tokenizer,
            author_label=author_label,
            themes=themes,
            summary=summary,
            prior_personas=prior_personas,
            explicit_gender_supported=explicit_gender_supported,
        )

        out_path = os.path.join(args.output_dir, f"{author_label}.txt")
        write_persona_txt(
            out_path,
            obj["artist_name"],
            obj["album_title"],
            obj["persona_bio"],
            obj["tracklist"],
        )

        accepted_personas.append({
            "author_label": author_label,
            "artist_name": clean_text(obj["artist_name"]),
            "album_title": clean_text(obj["album_title"]),
            "persona_bio": clean_text(obj["persona_bio"]),
            "tracklist": list(obj["tracklist"]),
        })

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
            args.output_csv, index=False, encoding="utf-8"
        )
        print(f"✅ Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
