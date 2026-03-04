#!/usr/bin/env python3
"""
generate_album_covers.py

Creates fictional album cover art from persona TXT files using:

1) Meta-Llama-3.1-8B-Instruct -> summarize persona bio into theme
2) FLUX.1-dev -> generate album artwork

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

# Enable faster matrix operations on Ampere GPUs (A6000 / A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

pipe = FluxPipeline.from_pretrained(
    IMAGE_MODEL,
    torch_dtype=torch.bfloat16
)

pipe = pipe.to(DEVICE)

# Reduce VRAM usage slightly (safe default)
pipe.enable_attention_slicing()

# ----------------------------
# HELPERS
# ----------------------------


def extract_fields(text):

    artist = None
    album = None
    bio = None

    artist_match = re.search(r"artist_name\s*:\s*(.*)", text, re.I)
    album_match = re.search(r"album_title\s*:\s*(.*)", text, re.I)
    bio_match = re.search(r"persona_bio\s*:\s*(.*)", text, re.I | re.S)

    if artist_match:
        artist = artist_match.group(1).strip()

    if album_match:
        album = album_match.group(1).strip()

    if bio_match:
        bio = bio_match.group(1).strip()

    return artist, album, bio


def generate_theme(bio):

    prompt = f"""
Summarize the following fictional musician biography into a short
5-10 word theme describing the aesthetic or concept.

Biography:
{bio}

Theme:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    theme = text.split("Theme:")[-1].strip()

    return theme


def generate_album_art(artist, album, theme):

    prompt = f"""
Album cover artwork for the fictional music artist {artist}.
Album title: {album}.

Theme: {theme}.

Modern music album cover design, bold colors, dramatic lighting,
cinematic composition, surreal artistic style, highly detailed,
professional album artwork.
"""

    image = pipe(
        prompt,
        height=1536,
        width=1536,
        guidance_scale=4.0,
        num_inference_steps=30
    ).images[0]

    return image


# ----------------------------
# MAIN LOOP
# ----------------------------

persona_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

print(f"Found {len(persona_files)} persona files")

for file_path in persona_files:

    filename = os.path.basename(file_path)
    output_name = filename.replace(".txt", ".png")

    print(f"\nProcessing {filename}")

    with open(file_path, "r") as f:
        text = f.read()

    artist, album, bio = extract_fields(text)

    if not bio:
        print("Skipping (no bio found)")
        continue

    theme = generate_theme(bio)

    print("Theme:", theme)

    image = generate_album_art(artist, album, theme)

    save_path = os.path.join(OUTPUT_DIR, output_name)

    image.save(save_path)

    print("Saved:", save_path)

print("\nDone generating album covers!")
