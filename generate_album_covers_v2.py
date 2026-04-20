#!/usr/bin/env python3
"""
generate_album_covers.py

Generate album cover PNG files from persona TXT files.

Supports:
- single mode via --input-file
- batch mode via --input-dir

Outputs:
- one PNG per author in --output-dir

Optional transient artifacts:
- themes.csv
- errors.csv

Use --cleanup-intermediate to delete the transient CSV files at the end of a successful run.
"""

import argparse
import csv
import gc
import glob
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
import torch
from diffusers import FluxPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"

WIDTH = 1024
HEIGHT = 1024
NUM_STEPS = 28
GUIDANCE = 4.0
THEME_MAX_NEW_TOKENS = 22
THEME_TEMPERATURE = 0.6
MAX_RETRIES_PER_AUTHOR = 2
RETRY_SLEEP_SECONDS = 3
MAX_BIO_CHARS = 4000
MAX_PROMPT_FIELD_CHARS = 180

FIELD_LABELS = {
    "artist": [
        "artist_name", "artist name", "artist", "band_name", "band name", "band",
        "stage name", "stage_name", "persona name", "persona_name", "fictional artist",
        "fictional artist name", "musician name", "performer", "alias",
    ],
    "album": [
        "album_title", "album title", "album", "record", "release", "debut album",
        "lp", "ep", "project title", "project_title", "mixtape", "soundtrack",
    ],
    "bio": [
        "persona_bio", "bio", "biography", "artist bio", "artist biography",
        "persona biography", "backstory", "about", "concept", "narrative", "story",
    ],
}

TRACK_HEADERS = ["tracklist", "tracks", "songs", "track list"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_row_csv(path: str, fieldnames, row: Dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def normalize_theme(theme: str) -> str:
    theme = (theme or "").strip().splitlines()[0].strip()
    theme = re.sub(r"\(.*?\)", "", theme).strip()
    theme = re.split(r"\s+\bor\b\s+", theme, maxsplit=1, flags=re.I)[0].strip()
    theme = re.split(r"\b(i chose|because|note:|return only)\b", theme, flags=re.I)[0].strip()
    theme = theme.strip(" \"'`.,:;—-")
    theme = re.sub(r"[^\w\s-]", "", theme)
    theme = re.sub(r"\s+", " ", theme).strip()
    return theme or "bold modern experimental album cover"


def truncate_clean(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def resolve_input_paths(input_file: str, input_dir: str) -> List[str]:
    if bool(input_file) == bool(input_dir):
        raise ValueError("Provide exactly one of --input-file or --input-dir")
    if input_file:
        p = Path(input_file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        if p.suffix.lower() != ".txt":
            raise ValueError("generate_album_covers.py expects TXT persona files.")
        return [str(p)]
    pdir = Path(input_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"Input directory not found: {pdir}")
    files = sorted(glob.glob(str(pdir / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No TXT files found in: {pdir}")
    return files


def extract_section_by_labels(text: str, labels: List[str]) -> Optional[str]:
    for label in labels:
        single_line = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text)
        if single_line:
            return single_line.group(1).strip()

    block_pattern = (
        r"(?ims)^\s*(%s)\s*:\s*(.+?)(?=^\s*[A-Za-z][\w /-]{1,40}\s*:\s*|\Z)"
        % "|".join(re.escape(label) for label in labels)
    )
    block_match = re.search(block_pattern, text)
    if block_match:
        return block_match.group(2).strip()
    return None


def infer_bio_block(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower().rstrip(":")
        if any(lowered == header for header in TRACK_HEADERS):
            break
        if re.match(r"^[*-]\s", stripped):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def extract_fields(text: str, fallback_artist: str) -> Tuple[str, str, str]:
    normalized = re.sub(r"\r\n", "\n", (text or "")).strip()
    artist = extract_section_by_labels(normalized, FIELD_LABELS["artist"])
    album = extract_section_by_labels(normalized, FIELD_LABELS["album"])
    bio = extract_section_by_labels(normalized, FIELD_LABELS["bio"])

    if not bio:
        bio = infer_bio_block(normalized)

    artist = truncate_clean(artist or fallback_artist, MAX_PROMPT_FIELD_CHARS)
    album = truncate_clean(album or "Untitled Album", MAX_PROMPT_FIELD_CHARS)
    bio = truncate_clean(bio or normalized, MAX_BIO_CHARS)
    return artist, album, bio


def load_llama_cpu(model_id: str):
    print(f"[init] Loading Llama tokenizer/model on CPU: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    print("[init] Llama loaded.", flush=True)
    return tokenizer, model


def generate_theme(tokenizer, model, bio: str) -> str:
    prompt = (
        "Summarize the following fictional musician biography into a short 5-10 word visual theme for album artwork.\n"
        "Return ONLY the theme words.\n\n"
        f"Biography:\n{bio}\n\nTheme:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=THEME_MAX_NEW_TOKENS,
        temperature=THEME_TEMPERATURE,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return normalize_theme(decoded.split("Theme:")[-1].strip())


def unload_llama(tokenizer, model):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_flux_pipeline_gpu(image_model: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. FLUX generation requires a GPU node.")

    print(f"[init] CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[init] Loading FLUX pipeline: {image_model}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipe = FluxPipeline.from_pretrained(image_model, torch_dtype=torch.bfloat16)
    print("[init] FLUX weights loaded. Moving pipeline to CUDA...", flush=True)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("[init] FLUX ready on CUDA.", flush=True)
    return pipe


def author_seed(author_label: str) -> int:
    return abs(hash(author_label)) % (2**31 - 1)


def build_image_prompt(artist: str, album: str, theme: str) -> str:
    return (
        f"Album cover artwork for the fictional music artist {artist}. "
        f"Album title: {album}. Theme: {theme}. "
        "Modern music album cover design, bold colors, dramatic lighting, "
        "cinematic composition, surreal artistic style, highly detailed, "
        "professional graphic design, square album art."
    )


def generate_cover(pipe, artist: str, album: str, theme: str, seed: Optional[int] = None):
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    return pipe(
        build_image_prompt(artist, album, theme),
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=NUM_STEPS,
        generator=generator,
    ).images[0]


def safe_remove(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def main():
    global NUM_STEPS

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="")
    ap.add_argument("--input-dir", default="")
    ap.add_argument("--output-dir", default="outputs/album_covers")
    ap.add_argument("--theme-csv", default="", help="Optional theme checkpoint CSV. Defaults to <output-dir>/themes.csv")
    ap.add_argument("--error-csv", default="", help="Optional error log CSV. Defaults to <output-dir>/errors.csv")
    ap.add_argument("--cleanup-intermediate", action="store_true", help="Delete transient CSV files at the end of a successful run")
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--image-model", default=IMAGE_MODEL)
    args = ap.parse_args()

    persona_files = resolve_input_paths(args.input_file, args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    theme_csv = args.theme_csv.strip() or os.path.join(args.output_dir, "themes.csv")
    error_csv = args.error_csv.strip() or os.path.join(args.output_dir, "errors.csv")

    existing_map: Dict[str, Dict[str, str]] = {}
    if os.path.exists(theme_csv):
        df = pd.read_csv(theme_csv)
        for _, row in df.iterrows():
            existing_map[str(row["author_label"])] = {
                "artist": str(row["artist"]),
                "album": str(row["album"]),
                "theme": str(row["theme"]),
            }

    print(f"[start] Persona files detected: {len(persona_files)}", flush=True)
    tokenizer, llama = load_llama_cpu(args.llm_model)
    for idx, file_path in enumerate(persona_files, start=1):
        author_label = Path(file_path).stem
        if author_label in existing_map:
            continue
        try:
            print(f"[theme] [{idx}/{len(persona_files)}] parsing {author_label}", flush=True)
            text = Path(file_path).read_text(encoding="utf-8")
            artist, album, bio = extract_fields(text, fallback_artist=author_label)
            print(
                f"[theme] artist={artist[:70]!r} album={album[:70]!r} bio_chars={len(bio)}",
                flush=True,
            )
            theme = generate_theme(tokenizer, llama, bio)
            existing_map[author_label] = {"artist": artist, "album": album, "theme": theme}
            append_row_csv(
                theme_csv,
                ["author_label", "artist", "album", "theme"],
                {
                    "author_label": author_label,
                    "artist": artist,
                    "album": album,
                    "theme": theme,
                },
            )
            print(f"[theme] {author_label} -> {theme}", flush=True)
        except Exception as exc:
            append_row_csv(
                error_csv,
                ["timestamp", "stage", "author_label", "error"],
                {
                    "timestamp": now_iso(),
                    "stage": "theme",
                    "author_label": author_label,
                    "error": repr(exc),
                },
            )
            print(f"[error] theme failed for {author_label}: {exc!r}", flush=True)
            raise

    print("[init] Theme extraction complete. Releasing Llama...", flush=True)
    unload_llama(tokenizer, llama)

    pipe = load_flux_pipeline_gpu(args.image_model)
    failed = 0
    for author_label, meta in sorted(existing_map.items()):
        out_path = os.path.join(args.output_dir, f"{author_label}.png")
        if os.path.exists(out_path):
            print(f"[skip] {author_label}", flush=True)
            continue

        seed = author_seed(author_label)
        success = False
        for attempt in range(1, MAX_RETRIES_PER_AUTHOR + 2):
            try:
                print(f"[img] {author_label} (attempt {attempt})", flush=True)
                print(
                    f"[img] prompt fields: artist={meta['artist'][:60]!r}, album={meta['album'][:60]!r}, theme={meta['theme'][:80]!r}",
                    flush=True,
                )
                img = generate_cover(pipe, meta["artist"], meta["album"], meta["theme"], seed=seed)
                print(f"[img] save -> {out_path}", flush=True)
                img.save(out_path)
                print(f"[img] done -> {out_path}", flush=True)
                success = True
                break
            except torch.OutOfMemoryError as exc:
                append_row_csv(
                    error_csv,
                    ["timestamp", "stage", "author_label", "error"],
                    {
                        "timestamp": now_iso(),
                        "stage": "image_oom",
                        "author_label": author_label,
                        "error": repr(exc),
                    },
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if NUM_STEPS > 22:
                    NUM_STEPS = 22
                print(f"[warn] OOM on {author_label}; retrying with NUM_STEPS={NUM_STEPS}", flush=True)
                time.sleep(RETRY_SLEEP_SECONDS)
            except Exception as exc:
                append_row_csv(
                    error_csv,
                    ["timestamp", "stage", "author_label", "error"],
                    {
                        "timestamp": now_iso(),
                        "stage": "image",
                        "author_label": author_label,
                        "error": repr(exc),
                    },
                )
                print(f"[error] image failed for {author_label} on attempt {attempt}: {exc!r}", flush=True)
                time.sleep(RETRY_SLEEP_SECONDS)
        if not success:
            failed += 1

    print(f"[done] image generation complete with failed={failed}", flush=True)
    if failed == 0 and args.cleanup_intermediate:
        safe_remove(theme_csv)
        if os.path.exists(error_csv) and os.path.getsize(error_csv) == 0:
            safe_remove(error_csv)
        print("Deleted transient intermediate CSV files.", flush=True)


if __name__ == "__main__":
    main()
