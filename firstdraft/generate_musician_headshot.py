#!/usr/bin/env python3
"""
generate_musician_headshot.py

Generate a single musician-style portrait for one VALIANT Wrapped author by
using the author's real headshot as the img2img source and the fictional music
persona TXT as the stylistic prompt source.

Designed to fit the existing per-author VALIANT Wrapped pipeline:
- input identity convention: Last_First_ScopusID
- persona TXT location: outputs/author_music_personas_txt/<author_label>.txt
- headshot lookup CSV: scopusIDlist.csv (record_id, first_name, last_name, scopus)
- headshot images: author_headshots/documents/<record_id>_photo.*
- output image: outputs/musician_headshots/<author_label>.png

Example:
  python generate_musician_headshot.py \
      --author-label Kim_Michael_58290603100

  python generate_musician_headshot.py \
      --author-label Kim_Michael_58290603100 \
      --skip-existing

Notes:
- Default model is SDXL img2img for a simple first production version.
- Prompting is persona-driven but deterministic (no extra LLM call required).
- This script is single-author-first; batch mode can be added later by looping.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from PIL import Image, ImageOps


# ----------------------------
# Paths / defaults
# ----------------------------
DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_LOOKUP_CSV = "scopusIDlist.csv"
DEFAULT_HEADSHOT_DIR = "author_headshots/documents"
DEFAULT_PERSONA_DIR = "outputs/author_music_personas_txt"
DEFAULT_OUTPUT_DIR = "outputs/musician_headshots"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ----------------------------
# Basic helpers
# ----------------------------
def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def canonical_author_label(value: str) -> str:
    s = clean_text(value).replace("\\", "/")
    s = os.path.basename(s)
    for ext in (".txt", ".csv", ".png", ".jpg", ".jpeg", ".webp"):
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
            break
    return s.strip()


def extract_scopus_id(author_label: str) -> str:
    parts = canonical_author_label(author_label).split("_")
    if parts and parts[-1].isdigit():
        return parts[-1]
    raise ValueError(
        f"Could not extract Scopus ID from author label: {author_label!r}. "
        "Expected format like Last_First_12345678900"
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


# ----------------------------
# Persona parsing
# ----------------------------
@dataclass
class Persona:
    artist_name: str = ""
    album_title: str = ""
    bio: str = ""
    tracklist: List[str] = None

    def __post_init__(self) -> None:
        if self.tracklist is None:
            self.tracklist = []


def parse_persona_txt(path: Path) -> Persona:
    raw = read_text_file(path).replace("\r\n", "\n")
    artist = ""
    album = ""
    bio = ""
    tracks: List[str] = []

    m = re.search(r"(?im)^Artist:\s*(.+)$", raw)
    if m:
        artist = clean_text(m.group(1))

    m = re.search(r"(?im)^Album:\s*(.+)$", raw)
    if m:
        album = clean_text(m.group(1))

    m = re.search(r"(?is)^.*?^Bio:\s*(.*?)\s*^Tracklist:\s*(.*)$", raw, flags=re.MULTILINE)
    if m:
        bio = clean_text(m.group(1))
        track_blob = m.group(2)
        for line in track_blob.splitlines():
            line = re.sub(r"^\s*(?:\d{1,2}[.)-]\s*|[-*]\s*)", "", line).strip()
            line = clean_text(line)
            if line:
                tracks.append(line)
    else:
        m = re.search(r"(?is)^.*?^Bio:\s*(.+)$", raw, flags=re.MULTILINE)
        if m:
            bio = clean_text(m.group(1))

    return Persona(artist_name=artist, album_title=album, bio=bio, tracklist=tracks)


# ----------------------------
# Headshot lookup
# ----------------------------
def load_lookup_row(lookup_csv: Path, scopus_id: str) -> Dict[str, str]:
    df = read_csv_flexible(lookup_csv).fillna("")
    required = {"record_id", "first_name", "last_name", "scopus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lookup CSV missing required columns: {sorted(missing)}")

    df["scopus"] = df["scopus"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    match = df[df["scopus"] == str(scopus_id)]
    if match.empty:
        raise FileNotFoundError(f"No row found in {lookup_csv} for Scopus ID {scopus_id}")
    row = match.iloc[0]
    return {k: clean_text(row.get(k, "")) for k in df.columns}


def find_headshot(headshot_dir: Path, record_id: str) -> Path:
    matches = sorted(headshot_dir.glob(f"{record_id}_photo.*"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find headshot file for record_id={record_id} in {headshot_dir}"
        )
    return matches[0]


# ----------------------------
# Prompt building
# ----------------------------
GENRE_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(rap|rapper|hip hop|hip-hop|beats|swagger|street|flow)\b", "charismatic hip-hop artist"),
    (r"\b(rock|guitar|punk|grunge|garage|loud|rebellious)\b", "bold alt-rock musician"),
    (r"\b(jazz|improv|improvis|sax|brass|swing|bebop)\b", "sophisticated jazz performer"),
    (r"\b(folk|acoustic|lyrical|storytelling|warm|earthy)\b", "thoughtful indie folk singer-songwriter"),
    (r"\b(neon|futuristic|electronic|synth|digital|cyber|signal|machine|robotic)\b", "futuristic electronic musician"),
    (r"\b(pop|anthem|hook|glam|spotlight|starlight)\b", "modern pop star"),
    (r"\b(cinematic|orchestral|ambient|dream|ethereal|atmospheric)\b", "cinematic ambient artist"),
]

LIGHTING_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(neon|future|electronic|synth|cyber|digital)\b", "neon magenta and cyan concert lighting"),
    (r"\b(rock|punk|grunge|garage)\b", "dramatic stage lighting with strong shadows"),
    (r"\b(jazz|sophisticated|noir|late night)\b", "moody low-key club lighting"),
    (r"\b(folk|acoustic|warm|organic)\b", "warm golden spotlighting"),
    (r"\b(pop|glam|anthem)\b", "glossy arena-style lighting"),
]

WARDROBE_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(rock|punk|grunge|garage)\b", "dark stagewear, textured jacket, musician styling"),
    (r"\b(rap|rapper|hip hop|hip-hop|street|swagger)\b", "stylish streetwear, layered jewelry, performance styling"),
    (r"\b(jazz|swing|bebop|noir)\b", "tailored performance attire with a polished stage presence"),
    (r"\b(folk|acoustic|warm|earthy)\b", "elevated casual performance clothing with organic textures"),
    (r"\b(electronic|synth|digital|future|cyber)\b", "sleek futuristic performance clothing with subtle metallic accents"),
    (r"\b(pop|glam|spotlight)\b", "editorial pop-star styling with bold performance fashion"),
]

BACKGROUND_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(electronic|future|signal|machine|neural|brain|circuit|digital|synth)\b", "stylized concert backdrop with abstract light patterns"),
    (r"\b(rock|punk|garage|grunge)\b", "subtle backstage or stage-set atmosphere"),
    (r"\b(jazz|club|noir)\b", "tasteful club-performance background blur"),
    (r"\b(folk|organic|earthy|storytelling)\b", "soft textured studio backdrop"),
]

NEGATIVE_PROMPT = (
    "blurry, low resolution, distorted face, duplicate face, extra limbs, extra fingers, bad anatomy, "
    "crossed eyes, asymmetrical eyes, overexposed skin, deformed mouth, warped teeth, cartoon, illustration, anime, "
    "painting, watermark, text, logo, frame, collage, multiple people"
)


def first_matching_rule(text: str, rules: Sequence[Tuple[str, str]], default: str) -> str:
    for pattern, value in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return value
    return default


def build_visual_prompt(persona: Persona, author_label: str) -> Tuple[str, str, Dict[str, str]]:
    blob = " ".join([
        persona.artist_name,
        persona.album_title,
        persona.bio,
        " ".join(persona.tracklist or []),
        author_label,
    ])
    blob = clean_text(blob)

    genre = first_matching_rule(blob, GENRE_RULES, "stylish genre-bending musician")
    lighting = first_matching_rule(blob, LIGHTING_RULES, "cinematic editorial stage lighting")
    wardrobe = first_matching_rule(blob, WARDROBE_RULES, "fashionable performance wardrobe")
    background = first_matching_rule(blob, BACKGROUND_RULES, "clean dramatic studio backdrop")

    mood_words = []
    for token in ["confident", "charismatic", "cinematic", "editorial", "immersive", "moody", "bold"]:
        if token in blob.lower() and token not in mood_words:
            mood_words.append(token)
    if not mood_words:
        mood_words = ["confident", "cinematic", "charismatic"]

    prompt = (
        "editorial portrait of the same person as a "
        f"{genre}, {wardrobe}, {lighting}, {background}, "
        f"{', '.join(mood_words)} mood, professional music photography, album-promo portrait, "
        "high detail, realistic skin texture, recognizable face, tasteful stylization"
    )

    if persona.artist_name:
        prompt += f", artist persona inspired by '{persona.artist_name}'"
    if persona.album_title:
        prompt += f", visual tone inspired by the album '{persona.album_title}'"

    meta = {
        "genre": genre,
        "lighting": lighting,
        "wardrobe": wardrobe,
        "background": background,
        "mood": ", ".join(mood_words),
    }
    return prompt, NEGATIVE_PROMPT, meta


# ----------------------------
# Image prep
# ----------------------------
def load_pil_image(path: Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def pad_to_square(img: Image.Image, fill: Tuple[int, int, int] = (12, 12, 12)) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(img, (x, y))
    return canvas


def prepare_init_image(img: Image.Image, target_size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = pad_to_square(img)
    return img.resize((target_size, target_size), Image.Resampling.LANCZOS)


# ----------------------------
# Diffusers execution
# ----------------------------
def load_pipeline(model_id: str, dtype_str: str):
    import torch
    from diffusers import AutoPipelineForImage2Image

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. generate_musician_headshot.py is intended to run on a GPU node."
        )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.float16)

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else None,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe


def run_generation(
    pipe,
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> Image.Image:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=init_image,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    return result.images[0]


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate a musician-style portrait for one author.")
    ap.add_argument("--author-label", required=True, help="Canonical author label like Last_First_ScopusID")
    ap.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--lookup-csv", default=DEFAULT_LOOKUP_CSV)
    ap.add_argument("--headshot-dir", default=DEFAULT_HEADSHOT_DIR)
    ap.add_argument("--persona-dir", default=DEFAULT_PERSONA_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=0.42)
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--seed", type=int, default=3890)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--save-debug-txt", action="store_true", help="Write prompt/debug metadata next to the output PNG")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(args.project_root).resolve()
    lookup_csv = (project_root / args.lookup_csv).resolve()
    headshot_dir = (project_root / args.headshot_dir).resolve()
    persona_dir = (project_root / args.persona_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    author_label = canonical_author_label(args.author_label)
    scopus_id = extract_scopus_id(author_label)
    persona_path = persona_dir / f"{author_label}.txt"
    output_path = output_dir / f"{author_label}.png"

    ensure_dir(output_dir)

    if args.skip_existing and output_path.exists():
        print(f"Skipping existing portrait: {output_path}")
        return

    if not lookup_csv.exists():
        raise FileNotFoundError(f"Lookup CSV not found: {lookup_csv}")
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona TXT not found: {persona_path}")
    if not headshot_dir.exists():
        raise FileNotFoundError(f"Headshot directory not found: {headshot_dir}")

    lookup_row = load_lookup_row(lookup_csv, scopus_id)
    record_id = clean_text(lookup_row.get("record_id"))
    headshot_path = find_headshot(headshot_dir, record_id)
    persona = parse_persona_txt(persona_path)

    prompt, negative_prompt, meta = build_visual_prompt(persona, author_label)

    print(f"Author: {author_label}")
    print(f"Scopus ID: {scopus_id}")
    print(f"Record ID: {record_id}")
    print(f"Headshot: {headshot_path}")
    print(f"Persona: {persona_path}")
    print(f"Output: {output_path}")
    print(f"Prompt genre cue: {meta['genre']}")

    init_image = prepare_init_image(load_pil_image(headshot_path), args.image_size)
    pipe = load_pipeline(args.model_id, args.dtype)
    final_image = run_generation(
        pipe=pipe,
        init_image=init_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    final_image.save(output_path)
    print(f"✅ Wrote: {output_path}")

    if args.save_debug_txt:
        debug_path = output_dir / f"{author_label}__prompt_debug.txt"
        debug_text = (
            f"AUTHOR_LABEL: {author_label}\n"
            f"SCOPUS_ID: {scopus_id}\n"
            f"RECORD_ID: {record_id}\n"
            f"HEADSHOT_PATH: {headshot_path}\n"
            f"PERSONA_PATH: {persona_path}\n"
            f"OUTPUT_PATH: {output_path}\n\n"
            f"ARTIST: {persona.artist_name}\n"
            f"ALBUM: {persona.album_title}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"NEGATIVE_PROMPT:\n{negative_prompt}\n\n"
            f"META:\n{meta}\n"
        )
        debug_path.write_text(debug_text, encoding="utf-8")
        print(f"✅ Wrote: {debug_path}")


if __name__ == "__main__":
    main()
