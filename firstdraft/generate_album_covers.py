#!/usr/bin/env python3
"""
generate_album_covers.py (SCALABLE / A6000 / NO CPU OFFLOAD / 1024x1024)

Two-phase pipeline designed for 103+ authors on ACCRE:

PHASE 1 (CPU)
 - Parse persona TXT files
 - Use Llama 3.1 on CPU to create short visual themes
 - Save/append themes to outputs/album_covers/themes.csv (checkpoint)

PHASE 2 (GPU)
 - Unload Llama
 - Load FLUX fully on GPU (NO CPU offload)
 - Generate 1024x1024 album covers, one at a time
 - Resume safely (skip existing images)
 - Per-author retries + error logging to outputs/album_covers/errors.csv

Input:
    /home/lordnd/valiant_wrapped/valiantwrapped/outputs/author_music_personas_txt/*.txt

Outputs:
    outputs/album_covers/<author_label>.png
    outputs/album_covers/themes.csv
    outputs/album_covers/errors.csv

Recommended long-run env var:
  export PYTORCH_ALLOC_CONF=expandable_segments:True
"""

import os
import re
import gc
import glob
import time
import csv
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import FluxPipeline

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

INPUT_DIR = "/home/lordnd/valiant_wrapped/valiantwrapped/outputs/author_music_personas_txt"
OUTPUT_DIR = "outputs/album_covers"
THEME_CSV = os.path.join(OUTPUT_DIR, "themes.csv")
ERROR_CSV = os.path.join(OUTPUT_DIR, "errors.csv")

# Models
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"

# Image settings (reduced for stability)
WIDTH = 1024
HEIGHT = 1024
NUM_STEPS = 28
GUIDANCE = 4.0

# Scalability / stability toggles
# Skip if outputs/album_covers/<author>.png exists
RESUME_SKIP_EXISTING_IMAGES = True
# If themes.csv exists, reuse and only compute missing
REUSE_EXISTING_THEMES_CSV = True
# Append each theme immediately (checkpoint)
SAVE_THEMES_INCREMENTALLY = True
# Slight VRAM reduction, safe default
ENABLE_ATTENTION_SLICING = True
SEED_PER_AUTHOR = True                      # Reproducible images per author_label

# Theme generation behavior
THEME_MAX_NEW_TOKENS = 22
THEME_TEMPERATURE = 0.6

# Robustness
MAX_RETRIES_PER_AUTHOR = 2                  # Retries for image generation only
RETRY_SLEEP_SECONDS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_row_csv(path: str, fieldnames, row: Dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def load_existing_themes() -> pd.DataFrame:
    if os.path.exists(THEME_CSV):
        return pd.read_csv(THEME_CSV)
    return pd.DataFrame(columns=["author_label", "artist", "album", "theme"])


def normalize_theme(theme: str) -> str:
    """
    Aggressively sanitize LLM output to keep prompts clean.
    """
    theme = (theme or "").strip()

    # Keep only first line
    theme = theme.splitlines()[0].strip()

    # Drop parentheticals and anything after common "option" wording
    theme = re.sub(r"\(.*?\)", "", theme).strip()
    theme = re.split(r"\s+\bor\b\s+", theme, maxsplit=1, flags=re.I)[0].strip()

    # Remove common chatter if it slips through
    theme = re.split(
        r"\b(i chose|was not chosen|because|note:|return only)\b", theme, flags=re.I)[0].strip()

    # Strip quotes/punct and collapse spaces
    theme = theme.strip(" \"'`.,:;—-")
    theme = re.sub(r"[^\w\s-]", "", theme)
    theme = re.sub(r"\s+", " ", theme).strip()

    if not theme:
        theme = "bold modern experimental album cover"
    return theme


# ------------------------------------------------
# FIELD EXTRACTION (ROBUST)
# ------------------------------------------------

def extract_fields(text: str, fallback_artist: str) -> Tuple[str, str, str]:
    """
    Robust extraction for loosely formatted persona TXT files.

    Returns: (artist, album, bio)
      - artist: falls back to filename-derived label
      - album:  falls back to "Untitled Album"
      - bio:    falls back to full text (minus obvious tracklist section if present)
    """
    t = (text or "").strip()
    t_norm = re.sub(r"\r\n", "\n", t)

    def find_labeled_value(labels):
        for lab in labels:
            m = re.search(
                rf"(?im)^\s*{re.escape(lab)}\s*:\s*(.+?)\s*$", t_norm)
            if m:
                return m.group(1).strip()
        return None

    # Artist
    artist = find_labeled_value([
        "artist_name", "artist name", "artist",
        "band_name", "band name", "band",
        "stage name", "stage_name", "project", "act"
    ])
    if not artist:
        m = re.search(
            r"(?im)^\s*(artist name|band name|artist|band)\s*[-—]\s*(.+?)\s*$", t_norm)
        if m:
            artist = m.group(2).strip()
    if not artist:
        artist = fallback_artist

    # Album
    album = find_labeled_value([
        "album_title", "album title", "album",
        "record", "release", "debut album", "lp", "ep"
    ])
    if not album:
        m = re.search(
            r"(?im)^\s*(album title|album|record|release)\s*[-—]\s*(.+?)\s*$", t_norm)
        if m:
            album = m.group(2).strip()
    if not album:
        album = "Untitled Album"

    # Bio
    bio = None
    m = re.search(r"(?is)^\s*(persona_bio|bio|biography)\s*:\s*(.+)$", t_norm)
    if m:
        bio = m.group(2).strip()
    if not bio:
        parts = re.split(r"(?im)^\s*(tracklist|tracks|songs)\s*:\s*$", t_norm)
        bio = parts[0].strip() if parts else t_norm
    if not (bio or "").strip():
        bio = t_norm

    return artist, album, bio


# ------------------------------------------------
# PHASE 1 — THEME GENERATION (LLAMA ON CPU)
# ------------------------------------------------

def load_llama_cpu():
    print("Loading Llama on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    return tokenizer, model


def generate_theme(tokenizer, model, bio: str) -> str:
    prompt = (
        "Summarize the following fictional musician biography into a short 5-10 word visual theme for album artwork.\n"
        "Return ONLY the theme words (no quotes, no extra commentary).\n\n"
        f"Biography:\n{bio}\n\n"
        "Theme:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")  # CPU
    outputs = model.generate(
        **inputs,
        max_new_tokens=THEME_MAX_NEW_TOKENS,
        temperature=THEME_TEMPERATURE,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = decoded.split("Theme:")[-1].strip()
    return normalize_theme(raw)


def unload_llama(tokenizer, model):
    print("Unloading Llama...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------
# PHASE 2 — IMAGE GENERATION (FLUX ON GPU, NO OFFLOAD)
# ------------------------------------------------

def load_flux_pipeline_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. FLUX generation requires a GPU node.")

    print("Loading FLUX pipeline (GPU-only, no CPU offload)...")

    # Speedups on Ampere (A6000)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipe = FluxPipeline.from_pretrained(
        IMAGE_MODEL,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if ENABLE_ATTENTION_SLICING:
        pipe.enable_attention_slicing()

    return pipe


def author_seed(author_label: str) -> int:
    return abs(hash(author_label)) % (2**31 - 1)


def build_image_prompt(artist: str, album: str, theme: str) -> str:
    return (
        f"Album cover artwork for the fictional music artist {artist}. "
        f"Album title: {album}. "
        f"Theme: {theme}. "
        "Modern music album cover design, bold colors, dramatic lighting, "
        "cinematic composition, surreal artistic style, highly detailed, "
        "professional graphic design, square album art."
    )


def generate_cover(pipe, artist: str, album: str, theme: str, seed: Optional[int] = None):
    prompt = build_image_prompt(artist, album, theme)

    generator = None
    if SEED_PER_AUTHOR and seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    return pipe(
        prompt,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=NUM_STEPS,
        generator=generator,
    ).images[0]


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():
    print("\n=== VALIANT Wrapped: Album Cover Generation (Scalable) ===\n")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GiB")
    else:
        print("GPU: not detected (this will fail during FLUX load).")

    if os.environ.get("PYTORCH_ALLOC_CONF", "") == "":
        print("\nTip (optional): For long runs, consider:")
        print("  export PYTORCH_ALLOC_CONF=expandable_segments:True\n")

    persona_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    print(f"Found {len(persona_files)} persona files in {INPUT_DIR}\n")

    # --------------------------
    # PHASE 1: THEMES (CPU)
    # --------------------------
    print("PHASE 1: Themes (CPU)\n")

    existing_df = load_existing_themes() if REUSE_EXISTING_THEMES_CSV else pd.DataFrame(
        columns=["author_label", "artist", "album", "theme"]
    )
    existing_map: Dict[str, Dict[str, str]] = {}
    if not existing_df.empty:
        for _, r in existing_df.iterrows():
            existing_map[str(r["author_label"])] = {
                "artist": str(r["artist"]),
                "album": str(r["album"]),
                "theme": str(r["theme"]),
            }
        print(f"Loaded {len(existing_map)} existing themes from {THEME_CSV}\n")

    tokenizer, llama = load_llama_cpu()

    theme_fieldnames = ["author_label", "artist", "album", "theme"]

    for file_path in persona_files:
        filename = os.path.basename(file_path)
        author_label = os.path.splitext(filename)[0]

        if author_label in existing_map:
            continue  # already themed

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            artist, album, bio = extract_fields(
                text, fallback_artist=author_label)
            if not (bio or "").strip():
                raise ValueError("Empty bio after parsing")

            theme = generate_theme(tokenizer, llama, bio)

            row = {"author_label": author_label,
                   "artist": artist, "album": album, "theme": theme}
            existing_map[author_label] = {
                "artist": artist, "album": album, "theme": theme}

            if SAVE_THEMES_INCREMENTALLY:
                append_row_csv(THEME_CSV, theme_fieldnames, row)

            print(f"[theme] {author_label} -> {theme}")

        except Exception as e:
            append_row_csv(
                ERROR_CSV,
                ["timestamp", "stage", "author_label", "error"],
                {"timestamp": now_iso(), "stage": "theme",
                 "author_label": author_label, "error": repr(e)},
            )
            print(f"[ERROR][theme] {author_label}: {e}")

    # If not writing incrementally, write full themes at end
    if not SAVE_THEMES_INCREMENTALLY:
        df_out = pd.DataFrame(
            [{"author_label": k, **v} for k, v in existing_map.items()],
            columns=["author_label", "artist", "album", "theme"],
        ).sort_values("author_label")
        df_out.to_csv(THEME_CSV, index=False)
        print(f"\nSaved themes to {THEME_CSV}\n")
    else:
        print(f"\nThemes checkpointed to {THEME_CSV}\n")

    unload_llama(tokenizer, llama)

    # --------------------------
    # PHASE 2: IMAGES (GPU)
    # --------------------------
    print("\nPHASE 2: Album Covers (GPU)\n")

    themes_df = pd.read_csv(THEME_CSV)
    print(f"Loaded {len(themes_df)} themes from {THEME_CSV}\n")

    pipe = load_flux_pipeline_gpu()

    created = 0
    skipped = 0
    failed = 0

    for _, row in themes_df.iterrows():
        author_label = str(row["author_label"])
        artist = str(row["artist"])
        album = str(row["album"])
        theme = str(row["theme"])

        out_path = os.path.join(OUTPUT_DIR, f"{author_label}.png")

        if RESUME_SKIP_EXISTING_IMAGES and os.path.exists(out_path):
            print(f"[skip] {author_label} (already exists)")
            skipped += 1
            continue

        seed = author_seed(author_label) if SEED_PER_AUTHOR else None

        attempt = 0
        success = False
        while attempt <= MAX_RETRIES_PER_AUTHOR:
            attempt += 1
            try:
                print(f"[img] {author_label} (attempt {attempt})")
                img = generate_cover(pipe, artist, album, theme, seed=seed)
                img.save(out_path)
                print(f"      saved -> {out_path}")
                created += 1
                success = True
                break

            except torch.OutOfMemoryError as e:
                # Clear cache and retry with safer settings
                append_row_csv(
                    ERROR_CSV,
                    ["timestamp", "stage", "author_label", "error"],
                    {"timestamp": now_iso(), "stage": "image_oom",
                     "author_label": author_label, "error": repr(e)},
                )
                print(f"[OOM] {author_label}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # On retry, reduce steps for a bit more headroom
                global NUM_STEPS
                if NUM_STEPS > 22:
                    NUM_STEPS = 22
                    print("      reducing num_inference_steps to 22 and retrying...")

                if attempt > MAX_RETRIES_PER_AUTHOR:
                    print("      giving up after retries.")
                else:
                    time.sleep(RETRY_SLEEP_SECONDS)

            except Exception as e:
                append_row_csv(
                    ERROR_CSV,
                    ["timestamp", "stage", "author_label", "error"],
                    {"timestamp": now_iso(), "stage": "image",
                     "author_label": author_label, "error": repr(e)},
                )
                print(f"[ERROR][img] {author_label}: {e}")
                if attempt > MAX_RETRIES_PER_AUTHOR:
                    print("      giving up after retries.")
                else:
                    time.sleep(RETRY_SLEEP_SECONDS)

        if not success:
            failed += 1

    print("\n=== Summary ===")
    print(f"Created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Failed : {failed}")

    print("\nDone. Covers are in outputs/album_covers/\n")
    if os.path.exists(ERROR_CSV):
        print(f"Any failures were logged to: {ERROR_CSV}")


if __name__ == "__main__":
    main()
