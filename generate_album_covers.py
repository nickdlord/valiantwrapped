#!/usr/bin/env python3
"""
generate_album_covers.py

Creates fictional album cover art from persona TXT files using:
1) meta-llama/Meta-Llama-3.1-8B-Instruct -> summarize persona bio into a 5–10 word theme
2) black-forest-labs/FLUX.1-dev -> generate album artwork

Input:
    /home/lordnd/valiant_wrapped/valiantwrapped/outputs/author_music_personas_txt/*.txt

Output:
    outputs/album_covers/*.png

Image size: 1536 x 1536
"""

import os
import glob
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import FluxPipeline

# ----------------------------
# PATH SETTINGS
# ----------------------------

INPUT_DIR = "/home/lordnd/valiant_wrapped/valiantwrapped/outputs/author_music_personas_txt"
OUTPUT_DIR = "outputs/album_covers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# MODEL SETTINGS
# ----------------------------

LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# HELPERS
# ----------------------------


def extract_fields(text: str, fallback_artist: str):
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
        # Matches: "Label: value" on one line
        for lab in labels:
            m = re.search(
                rf"(?im)^\s*{re.escape(lab)}\s*:\s*(.+?)\s*$", t_norm)
            if m:
                return m.group(1).strip()
        return None

    # Artist (lots of common variants)
    artist = find_labeled_value([
        "artist_name", "artist name", "artist",
        "band_name", "band name", "band",
        "stage name", "stage_name", "project", "act"
    ])

    # Also catch: "Artist Name - X" or "Artist Name — X"
    if not artist:
        m = re.search(
            r"(?im)^\s*(artist name|band name|artist|band)\s*[-—]\s*(.+?)\s*$", t_norm)
        if m:
            artist = m.group(2).strip()

    if not artist:
        artist = fallback_artist

    # Album title
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

    # Labeled bio block, if present
    m = re.search(r"(?is)^\s*(persona_bio|bio|biography)\s*:\s*(.+)$", t_norm)
    if m:
        bio = m.group(2).strip()

    # If no labeled bio, use everything except obvious tracklist section
    if not bio:
        # Split off tracklist section if present via heading lines like "Tracklist:" or "Tracks:"
        parts = re.split(r"(?im)^\s*(tracklist|tracks|songs)\s*:\s*$", t_norm)
        if len(parts) > 1:
            bio = parts[0].strip()
        else:
            bio = t_norm

    if not bio.strip():
        bio = t_norm

    return artist, album, bio


def generate_theme(bio: str) -> str:
    """
    Use the LLM to condense the bio into a 5–10 word theme suitable for an image prompt.
    Returns ONLY the theme.
    """
    prompt = (
        "Summarize the following fictional musician biography into a short 5-10 word theme "
        "describing the aesthetic or concept.\n\n"
        f"Biography:\n{bio}\n\n"
        "Return ONLY the theme (no quotes, no extra text).\n"
        "Theme:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=24,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the part after "Theme:"
    theme = full.split("Theme:")[-1].strip()

    # Safety cleanup if the model adds extra lines
    theme = theme.splitlines()[0].strip()

    # Hard fallback
    if not theme:
        theme = "bold modern experimental album cover"

    return theme


def generate_album_art(artist: str, album: str, theme: str):
    """
    Generate a 1536x1536 album cover image using FLUX.
    """
    prompt = (
        f"Album cover artwork for the fictional music artist {artist}. "
        f"Album title: {album}. "
        f"Theme: {theme}. "
        "Modern music album cover design, bold colors, dramatic lighting, "
        "cinematic composition, surreal artistic style, highly detailed, "
        "professional graphic design, square album art."
    )

    image = pipe(
        prompt,
        height=1536,
        width=1536,
        guidance_scale=4.0,
        num_inference_steps=30,
    ).images[0]

    return image


# ----------------------------
# LOAD LLM (theme generator)
# ----------------------------

print("Loading Llama model...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ----------------------------
# LOAD IMAGE MODEL (FLUX optimized for GPU)
# ----------------------------

print("Loading FLUX image model...")

# Faster matrix ops on Ampere+ GPUs (A6000/A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

pipe = FluxPipeline.from_pretrained(
    IMAGE_MODEL,
    torch_dtype=torch.bfloat16
).to(DEVICE)

# Slight VRAM reduction, stable default
pipe.enable_attention_slicing()

# ----------------------------
# MAIN LOOP
# ----------------------------

persona_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
print(f"Found {len(persona_files)} persona files")

for file_path in persona_files:
    filename = os.path.basename(file_path)
    # used for fallback artist and output name
    base = os.path.splitext(filename)[0]
    output_name = base + ".png"

    print(f"\nProcessing {filename}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    artist, album, bio = extract_fields(text, fallback_artist=base)

    if not bio or not bio.strip():
        print("Skipping (empty file)")
        continue

    # Create prompt theme via LLM
    theme = generate_theme(bio)
    print("Artist:", artist)
    print("Album:", album)
    print("Theme:", theme)

    # Generate image
    image = generate_album_art(artist, album, theme)

    # Save
    save_path = os.path.join(OUTPUT_DIR, output_name)
    image.save(save_path)
    print("Saved:", save_path)

print("\nDone generating album covers!")
