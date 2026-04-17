#!/usr/bin/env python3
"""
generate_musician_headshot.py

Generate a single musician-style portrait for one VALIANT Wrapped author by
using the author's real headshot as the img2img source and the fictional music
persona TXT as the stylistic prompt source.

Updated behavior:
- Supports --input-file <persona_txt> so you do NOT have to pass --author-label
- Still supports --author-label for backwards compatibility
- Prompting is intentionally more transformative and theatrical
- Defaults push farther away from the original faculty-headshot look

Examples:
  python generate_musician_headshot.py \
      --input-file outputs/author_music_personas_txt/Kim_Michael_58290603100.txt

  python generate_musician_headshot.py \
      --input-file outputs/author_music_personas_txt/Kim_Michael_58290603100.txt \
      --project-root . \
      --save-debug-txt

  python generate_musician_headshot.py \
      --author-label Kim_Michael_58290603100 \
      --project-root . \
      --strength 0.88
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


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_LOOKUP_CSV = "scopusIDlist.csv"
DEFAULT_HEADSHOT_DIR = "author_headshots/documents"
DEFAULT_PERSONA_DIR = "outputs/author_music_personas_txt"
DEFAULT_OUTPUT_DIR = "outputs/musician_headshots"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


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


GENRE_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(rap|rapper|hip hop|hip-hop|beats|swagger|street|flow|hustle)\b", "swagger-heavy hip-hop star"),
    (r"\b(rock|guitar|punk|grunge|garage|loud|rebellious|riot)\b", "ferocious alt-rock frontperson"),
    (r"\b(jazz|improv|improvis|sax|brass|swing|bebop|blue note)\b", "late-night jazz icon"),
    (r"\b(folk|acoustic|lyrical|storytelling|warm|earthy|campfire)\b", "mythic indie folk singer-songwriter"),
    (r"\b(neon|futuristic|electronic|synth|digital|cyber|signal|machine|robotic|neural|brain)\b", "futuristic synth-pop visionary"),
    (r"\b(pop|anthem|hook|glam|spotlight|starlight|chart)\b", "high-glam arena pop headliner"),
    (r"\b(cinematic|orchestral|ambient|dream|ethereal|atmospheric)\b", "cinematic dream-pop artist"),
]

LIGHTING_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(neon|future|electronic|synth|cyber|digital|signal|neural|brain)\b", "electric cyan and magenta concert lighting with hard contrast"),
    (r"\b(rock|punk|grunge|garage|riot)\b", "harsh stage spotlights, smoky shadows, live-show intensity"),
    (r"\b(jazz|swing|bebop|late night|noir)\b", "low-key moody club lighting with dramatic pools of light"),
    (r"\b(folk|acoustic|earthy|warm|storytelling)\b", "golden cinematic lighting with intimate shadow falloff"),
    (r"\b(pop|glam|spotlight|chart)\b", "glossy high-production arena lighting with polished highlights"),
]

WARDROBE_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(rock|punk|grunge|garage|riot)\b", "black leather, layered textures, dramatic stage styling"),
    (r"\b(rap|rapper|hip hop|hip-hop|street|swagger|flow)\b", "luxury streetwear, layered chains, bold performance styling"),
    (r"\b(jazz|swing|bebop|noir)\b", "sharp tailored performance fashion with nightclub flair"),
    (r"\b(folk|acoustic|earthy|organic)\b", "romantic bohemian stagewear with textured fabrics"),
    (r"\b(electronic|synth|digital|future|cyber|neural|brain)\b", "sleek futuristic stagewear with metallic details and experimental fashion"),
    (r"\b(pop|glam|spotlight|anthem)\b", "high-fashion pop-star wardrobe, bold silhouette, statement styling"),
]

BACKGROUND_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(electronic|future|signal|machine|neural|brain|circuit|digital|synth|cyber)\b", "surreal neon stage environment with abstract light architecture"),
    (r"\b(rock|punk|garage|grunge|riot)\b", "smoky arena-stage atmosphere with gritty performance energy"),
    (r"\b(jazz|club|noir|late night)\b", "velvet-dark club scene with blurred lights"),
    (r"\b(folk|organic|earthy|storytelling)\b", "cinematic textured backdrop with mythic atmosphere"),
    (r"\b(pop|glam|spotlight)\b", "editorial backstage-meets-stage fantasy setting"),
]

MAKEUP_ACCESSORY_RULES: Sequence[Tuple[str, str]] = [
    (r"\b(rock|punk|grunge|garage|riot)\b", "smudged stage makeup, rings, dramatic styling accents"),
    (r"\b(rap|rapper|hip hop|hip-hop|street|swagger)\b", "performance jewelry, tinted glasses, striking accessories"),
    (r"\b(electronic|synth|digital|future|cyber|neural)\b", "avant-garde makeup accents and futuristic accessories"),
    (r"\b(pop|glam|spotlight)\b", "editorial glam makeup and bold accessories"),
]

NEGATIVE_PROMPT = (
    "blurry, low resolution, distorted face, duplicate face, extra limbs, extra fingers, bad anatomy, "
    "crossed eyes, asymmetrical eyes, warped teeth, plain office portrait, academic headshot, corporate headshot, "
    "passport photo, linkedin photo, hospital portrait, lab coat portrait, neutral business attire, bland background, "
    "cartoon, illustration, anime, painting, watermark, text, logo, frame, collage, multiple people"
)


def first_matching_rule(text: str, rules: Sequence[Tuple[str, str]], default: str) -> str:
    for pattern, value in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return value
    return default


def infer_mood_words(blob: str) -> List[str]:
    moods = []
    candidates = [
        ("rebellious", r"\b(rebellious|riot|punk|grunge)\b"),
        ("electrifying", r"\b(electronic|synth|neon|cyber|signal|future)\b"),
        ("swaggering", r"\b(rap|rapper|swagger|flow|street)\b"),
        ("mythic", r"\b(folk|earthy|storytelling|dream)\b"),
        ("nocturnal", r"\b(noir|late night|club|jazz)\b"),
        ("glamorous", r"\b(pop|glam|spotlight|starlight|chart)\b"),
        ("cinematic", r"\b(cinematic|orchestral|ambient|dream|atmospheric)\b"),
    ]
    blob_lower = blob.lower()
    for word, pattern in candidates:
        if re.search(pattern, blob_lower, flags=re.IGNORECASE):
            moods.append(word)
    if not moods:
        moods = ["cinematic", "theatrical", "charismatic"]
    return moods[:4]


def build_visual_prompt(persona: Persona, author_label: str) -> Tuple[str, str, Dict[str, str]]:
    blob = " ".join([
        persona.artist_name,
        persona.album_title,
        persona.bio,
        " ".join(persona.tracklist or []),
        author_label,
    ])
    blob = clean_text(blob)

    genre = first_matching_rule(blob, GENRE_RULES, "theatrical genre-bending music icon")
    lighting = first_matching_rule(blob, LIGHTING_RULES, "high-contrast cinematic stage lighting")
    wardrobe = first_matching_rule(blob, WARDROBE_RULES, "dramatic performance wardrobe")
    background = first_matching_rule(blob, BACKGROUND_RULES, "surreal concert-photo backdrop")
    accents = first_matching_rule(blob, MAKEUP_ACCESSORY_RULES, "bold styling accents and performance accessories")
    mood_words = infer_mood_words(blob)

    prompt = (
        "dramatic fictional reimagining of a musician persona, "
        f"{genre}, {wardrobe}, {accents}, {lighting}, {background}, "
        f"{', '.join(mood_words)} mood, highly stylized editorial music portrait, "
        "album-shoot aesthetic, transformed stage presence, bold fashion energy, "
        "striking visual reinvention, photorealistic, high detail, rich contrast, shallow depth of field"
    )

    if persona.artist_name:
        prompt += f", inspired by the stage identity '{persona.artist_name}'"
    if persona.album_title:
        prompt += f", visual world inspired by the album '{persona.album_title}'"

    meta = {
        "genre": genre,
        "lighting": lighting,
        "wardrobe": wardrobe,
        "background": background,
        "accents": accents,
        "mood": ", ".join(mood_words),
    }
    return prompt, NEGATIVE_PROMPT, meta


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

    kwargs = {
        "torch_dtype": torch_dtype,
        "use_safetensors": True,
    }
    if torch_dtype == torch.float16:
        kwargs["variant"] = "fp16"

    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
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


def resolve_persona_inputs(
    project_root: Path,
    persona_dir_arg: str,
    input_file: str,
    input_dir: str,
    author_label: str,
) -> List[Tuple[Path, str]]:
    persona_dir = (project_root / persona_dir_arg).resolve()

    has_input_file = bool(clean_text(input_file))
    has_input_dir = bool(clean_text(input_dir))
    has_author_label = bool(clean_text(author_label))
    provided = int(has_input_file) + int(has_input_dir) + int(has_author_label)
    if provided != 1:
        raise ValueError("Provide exactly one of --input-file, --input-dir, or --author-label")

    if has_input_file:
        p = Path(input_file)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Persona TXT not found: {p}")
        if p.suffix.lower() != ".txt":
            raise ValueError("--input-file must point to a .txt persona file")
        return [(p, canonical_author_label(p.stem))]

    if has_input_dir:
        pdir = Path(input_dir)
        if not pdir.is_absolute():
            pdir = (project_root / pdir).resolve()
        if not pdir.exists():
            raise FileNotFoundError(f"Persona directory not found: {pdir}")
        persona_paths = sorted(pdir.glob("*.txt"))
        if not persona_paths:
            raise FileNotFoundError(f"No persona TXT files found in: {pdir}")
        return [(p, canonical_author_label(p.stem)) for p in persona_paths]

    label = canonical_author_label(author_label)
    return [((persona_dir / f"{label}.txt").resolve(), label)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate a musician-style portrait for one author.")
    ap.add_argument("--input-file", default="", help="Persona TXT file path; preferred for per-author pipeline use")
    ap.add_argument("--input-dir", default="", help="Directory of persona TXT files for multi-author runs")
    ap.add_argument("--author-label", default="", help="Backwards-compatible author label like Last_First_ScopusID")
    ap.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--lookup-csv", default=DEFAULT_LOOKUP_CSV)
    ap.add_argument("--headshot-dir", default=DEFAULT_HEADSHOT_DIR)
    ap.add_argument("--persona-dir", default=DEFAULT_PERSONA_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=0.82)
    ap.add_argument("--guidance-scale", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=40)
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
    output_dir = (project_root / args.output_dir).resolve()

    persona_items = resolve_persona_inputs(
        project_root=project_root,
        persona_dir_arg=args.persona_dir,
        input_file=args.input_file,
        input_dir=args.input_dir,
        author_label=args.author_label,
    )

    ensure_dir(output_dir)

    if not lookup_csv.exists():
        raise FileNotFoundError(f"Lookup CSV not found: {lookup_csv}")
    if not headshot_dir.exists():
        raise FileNotFoundError(f"Headshot directory not found: {headshot_dir}")

    pipe = None
    total = len(persona_items)
    built = 0
    skipped = 0
    failed = 0

    for idx, (persona_path, author_label) in enumerate(persona_items, start=1):
        output_path = output_dir / f"{author_label}.png"

        try:
            if not persona_path.exists():
                raise FileNotFoundError(f"Persona TXT not found: {persona_path}")

            if args.skip_existing and output_path.exists():
                print(f"[{idx}/{total}] Skipping existing portrait: {output_path}")
                skipped += 1
                continue

            scopus_id = extract_scopus_id(author_label)
            lookup_row = load_lookup_row(lookup_csv, scopus_id)
            record_id = clean_text(lookup_row.get("record_id"))
            headshot_path = find_headshot(headshot_dir, record_id)
            persona = parse_persona_txt(persona_path)
            prompt, negative_prompt, meta = build_visual_prompt(persona, author_label)

            print(f"[{idx}/{total}] Author: {author_label}")
            print(f"  Scopus ID: {scopus_id}")
            print(f"  Record ID: {record_id}")
            print(f"  Headshot: {headshot_path}")
            print(f"  Persona: {persona_path}")
            print(f"  Output: {output_path}")
            print(f"  Prompt genre cue: {meta['genre']}")
            print(f"  Transformation strength: {args.strength}")

            init_image = prepare_init_image(load_pil_image(headshot_path), args.image_size)
            if pipe is None:
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
            print(f"  ✅ Wrote: {output_path}")
            built += 1

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
                print(f"  ✅ Wrote: {debug_path}")

        except Exception as exc:
            failed += 1
            print(f"[{idx}/{total}] ❌ Failed for {author_label}: {exc}")

    print(f"Done. Built: {built} | Skipped existing: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()
