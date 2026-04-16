#!/usr/bin/env python3
"""
generate_valiantwrapped_site_v3.py

Build a static Spotify-inspired VALIANT Wrapped website from outputs produced by
previous pipeline steps.

Expected inputs by default
--------------------------
- author_summary_2025_present.csv
- outputs/author_expertise_txt/*.txt
- outputs/author_music_personas_txt/*.txt
- outputs/album_covers/*.png
- scopusexportALL.csv (optional but recommended if generating recommendations on the fly)

Output
------
A static site folder (default: docs/) containing:
- index.html
- authors/<author>/index.html
- data/authors.json
- assets/styles.css
- assets/app.js
- assets/album_covers/*
- assets/musician_headshots/*

Notes
-----
- This script does NOT split the master Scopus export into per-author CSVs.
- It can optionally generate paper recommendations during site generation using
  the master Scopus export and expertise summaries.
- Missing sections degrade gracefully.
- Album cover failures get a witty placeholder tile instead of a broken image.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import pickle
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_TAGLINE = "We all know AI hallucinates — so we decided to make it sing."
DEFAULT_CLOSER = "Thanks for an amazing and productive year. We can't wait to see what next year sounds like."
DEFAULT_RECOMMENDATION_FALLBACK = (
    "Our recommender listened carefully, nodded thoughtfully, and still decided this author is simply too original to match neatly."
)

TITLE_COLS = ["Title", "Document Title", "Article Title"]
ABSTRACT_COLS = ["Abstract", "Description"]
KEYWORD_COLS = ["Author Keywords", "Indexed Keywords", "Keywords"]
JOURNAL_COLS = ["Source title", "Source Title", "Journal"]
CITES_COLS = ["Cited by", "Citations", "Citation count"]
AUTHOR_ID_COLS = [
    "Author(s) ID", "Authors with affiliations", "Authors", "Author Names"]
DOI_COLS = ["DOI", "doi", "DOI link"]
LINK_COLS = ["Link", "URL", "Scopus Link", "Page link"]

PLACEHOLDER_LINES = [
    "Cover art missing.",
    "Apparently the AI dropped the album before it dropped the cover.",
]

SECTION_KICKERS = {
    "publications": "Research in Rotation",
    "persona": "Your Musical Alter Ego",
    "album": "Latest Album",
    "recommendations": "Suggested for you",
    "browse": "Browse",
    "about": "About the project",
    "closing": "Finale",
    "share": "Share",
}


@dataclass
class Metrics:
    pub_count: int = 0
    citation_count: int = 0
    top_journal: str = ""
    top_paper_title: str = ""
    top_paper_citations: int = 0


@dataclass
class Persona:
    artist_name: str = ""
    album_title: str = ""
    bio: str = ""
    tracklist: List[str] = field(default_factory=list)


@dataclass
class Recommendation:
    rank: int
    title: str
    google_url: str
    score: Optional[float] = None
    journal: str = ""
    doi: str = ""
    scopus_link: str = ""


@dataclass
class AuthorRecord:
    author_label: str
    display_name: str
    scopus_id: str
    expertise_summary: str = ""
    metrics: Metrics = field(default_factory=Metrics)
    lifetime_pub_count: int = 0
    lifetime_citation_count: int = 0
    persona: Persona = field(default_factory=Persona)
    cover_filename: str = ""
    musician_portrait_filename: str = ""
    recommendations: List[Recommendation] = field(default_factory=list)
    recommendation_fallback: str = ""
    share_card_filename: str = ""


@dataclass
class PaperRecord:
    title: str
    abstract: str
    keywords: str
    journal: str
    author_ids_raw: str
    doi: str
    scopus_link: str
    combined_text: str


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_author_label(value: object) -> str:
    text = clean_text(value).replace("\\", "/")
    text = os.path.basename(text)
    text = re.sub(r"\.(txt|csv|png|jpg|jpeg|webp)$",
                  "", text, flags=re.IGNORECASE)
    return text


def normalized_author_key(value: object) -> str:
    text = canonical_author_label(value)
    if not text:
        return ""
    text = text.replace("-", "_").replace(" ", "_")
    parts = [p for p in re.split(r"_+", text) if p]
    if parts and re.fullmatch(r"\d+", parts[-1]):
        parts = parts[:-1]
    return "_".join(part.lower() for part in parts)


def slugify(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "author"


def pick_existing_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(cols)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def infer_display_name(author_label: str) -> str:
    parts = canonical_author_label(author_label).split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        name_parts = parts[:-1]
    else:
        name_parts = parts
    if len(name_parts) >= 2:
        last = name_parts[0]
        first = " ".join(name_parts[1:])
        return f"{first.replace('_', ' ')} {last.replace('_', ' ')}".strip()
    return canonical_author_label(author_label).replace("_", " ")


def extract_scopus_id(author_label: str) -> str:
    parts = canonical_author_label(author_label).split("_")
    if parts and re.fullmatch(r"\d+", parts[-1]):
        return parts[-1]
    return ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)


def parse_expertise_txt(path: Path) -> str:
    raw = read_text_file(path).replace("\r\n", "\n").strip()
    if not raw:
        return ""

    summary_match = re.search(r"(?is)\bSUMMARY\s*:\s*(.+)$", raw)
    if summary_match:
        summary = clean_text(summary_match.group(1))
        if summary:
            return summary

    text = re.sub(r"(?im)^\s*THEMES\s*:\s*$", "", raw)
    text = re.sub(r"(?im)^\s*SUMMARY\s*:\s*$", "", text)
    text = re.sub(r"(?im)^\s*THEMES\s*:\s*", "", text)
    text = re.sub(r"(?im)^\s*SUMMARY\s*:\s*", "", text)
    text = clean_text(text)
    return text


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

    m = re.search(r"(?is)^.*?^Bio:\s*(.*?)\s*^Tracklist:\s*(.*)$",
                  raw, flags=re.MULTILINE)
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


def parse_recommendations_txt(path: Path) -> Tuple[List[Recommendation], str]:
    raw = read_text_file(path).replace("\r\n", "\n")
    lines = raw.splitlines()
    recommendations: List[Recommendation] = []
    fallback = ""

    current: Dict[str, str] = {}
    current_rank: Optional[int] = None
    current_title: str = ""

    def flush_current() -> None:
        nonlocal current, current_rank, current_title, recommendations
        if current_rank is None or not current_title:
            current = {}
            current_rank = None
            current_title = ""
            return
        score = None
        if current.get("SCORE"):
            try:
                score = float(current["SCORE"])
            except ValueError:
                score = None
        google_url = current.get("GOOGLE_URL", "")
        if not google_url and current_title:
            google_url = google_search_url(current_title)

        recommendations.append(
            Recommendation(
                rank=current_rank,
                title=current_title,
                google_url=google_url,
                score=score,
                journal=current.get("JOURNAL", ""),
                doi=current.get("DOI", ""),
                scopus_link=current.get("SCOPUS_LINK", ""),
            )
        )
        current = {}
        current_rank = None
        current_title = ""

    for line in lines:
        stripped = line.strip()

        m = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if m:
            flush_current()
            current_rank = int(m.group(1))
            current_title = clean_text(m.group(2))
            continue

        m = re.match(r"^\s*([A-Z_]+):\s*(.*)$", line)
        if m and current_rank is not None:
            current[m.group(1)] = clean_text(m.group(2))
            continue

        m = re.match(r"^FALLBACK_TEXT:\s*(.+)$", stripped)
        if m:
            fallback = clean_text(m.group(1))
            continue

    flush_current()
    return recommendations, fallback


def _safe_int(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def build_lifetime_metrics_map(author_csv_dir: Path) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    if not author_csv_dir.exists():
        return out

    for csv_path in sorted(author_csv_dir.glob("*.csv")):
        author_label = canonical_author_label(csv_path.stem)
        if not author_label:
            continue
        try:
            df = read_csv_flexible(csv_path).fillna("")
        except Exception:
            continue

        cite_col = pick_existing_col(df.columns, CITES_COLS)
        pub_count = len(df.index)
        citation_count = 0
        if cite_col:
            citation_count = int(pd.to_numeric(
                df[cite_col], errors="coerce").fillna(0).sum())

        out[author_label] = (pub_count, citation_count)

    return out


def parse_metrics_csv(path: Path) -> Dict[str, Metrics]:
    df = read_csv_flexible(path).fillna("")
    out: Dict[str, Metrics] = {}
    for _, row in df.iterrows():
        raw_label = clean_text(row.get("author_id") or row.get("author_file"))
        author_label = canonical_author_label(raw_label)
        if not author_label:
            continue
        out[author_label] = Metrics(
            pub_count=_safe_int(row.get("pub_count_2025_present")),
            citation_count=_safe_int(row.get("citation_count_2025_present")),
            top_journal=clean_text(row.get("top_journal_2025_present")),
            top_paper_title=clean_text(
                row.get("top_paper_title_2025_present")),
            top_paper_citations=_safe_int(
                row.get("top_paper_citations_2025_present")),
        )
    return out


def html_escape(text: object) -> str:
    return html.escape(clean_text(text))


def wrap_text_for_draw(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = clean_text(text).split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_multiline_block(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, fill, x: int, y: int, max_width: int, line_gap: int = 8, max_lines: Optional[int] = None) -> int:
    lines = wrap_text_for_draw(draw, text, font, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip(' .') + '…'
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        current_y += (bbox[3] - bbox[1]) + line_gap
    return current_y


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
        ]
    else:
        candidates = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()




def fit_font_to_width(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int = 18, bold: bool = False) -> ImageFont.ImageFont:
    text = clean_text(text) or " "
    for size in range(start_size, min_size - 1, -2):
        font = load_font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
    return load_font(min_size, bold=bold)


def fit_font_for_multiline(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int = 18, bold: bool = False, max_lines: int = 3) -> ImageFont.ImageFont:
    text = clean_text(text) or " "
    for size in range(start_size, min_size - 1, -2):
        font = load_font(size, bold=bold)
        lines = wrap_text_for_draw(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return font
    return load_font(min_size, bold=bold)

def open_and_fill_square(path: Optional[Path], size: int) -> Image.Image:
    if path and path.exists():
        try:
            img = Image.open(path).convert('RGB')
            w, h = img.size
            scale = max(size / max(w, 1), size / max(h, 1))
            resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            left = max(0, (resized.width - size) // 2)
            top = max(0, (resized.height - size) // 2)
            return resized.crop((left, top, left + size, top + size))
        except Exception:
            pass
    placeholder = Image.new('RGB', (size, size), (24, 24, 24))
    draw = ImageDraw.Draw(placeholder)
    draw.ellipse((size * 0.22, size * 0.22, size * 0.78, size * 0.78), outline=(29, 185, 84), width=8)
    draw.ellipse((size * 0.43, size * 0.43, size * 0.57, size * 0.57), fill=(29, 185, 84))
    return placeholder


def add_round_corners(image: Image.Image, radius: int) -> Image.Image:
    image = image.convert('RGBA')
    mask = Image.new('L', image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, image.size[0], image.size[1]), radius=radius, fill=255)
    image.putalpha(mask)
    return image



def build_share_card(author: AuthorRecord, output_path: Path, cover_src_dir: Path, musician_headshot_src_dir: Path, project_title: str) -> None:
    size = 1400
    card = Image.new('RGB', (size, size), (8, 8, 8))
    draw = ImageDraw.Draw(card)

    # Spotify-Wrapped-inspired background layers
    draw.rounded_rectangle((34, 34, size - 34, size - 34), radius=48, fill=(17, 17, 17), outline=(42, 42, 42), width=2)
    draw.ellipse((1040, -170, 1620, 360), fill=(18, 92, 52))
    draw.ellipse((830, -110, 1320, 280), fill=(34, 34, 34))
    draw.ellipse((-170, 920, 320, 1440), fill=(42, 18, 82))
    draw.ellipse((1020, 1080, 1500, 1560), fill=(16, 64, 34))
    draw.rounded_rectangle((70, 70, size - 70, size - 70), radius=42, outline=(56, 56, 56), width=1)

    album_path = cover_src_dir / author.cover_filename if author.cover_filename else None
    portrait_path = musician_headshot_src_dir / author.musician_portrait_filename if author.musician_portrait_filename else None

    portrait = add_round_corners(open_and_fill_square(portrait_path, 232), 28)
    album = add_round_corners(open_and_fill_square(album_path, 300), 30)
    card.paste(portrait, (96, 148), portrait)
    card.paste(album, (96, 702), album)

    # Brand / header
    font_brand = load_font(32, bold=True)
    font_eyebrow = load_font(23, bold=True)
    font_section = load_font(24, bold=True)
    font_small = load_font(22, bold=False)
    font_stat_label = load_font(20, bold=False)

    draw.text((96, 84), project_title, font=font_brand, fill=(30, 215, 96))
    draw.text((96, 118), 'Research, remixed.', font=font_eyebrow, fill=(240, 240, 240))

    share_label = '2025 Wrapped Share Card'
    share_bbox = draw.textbbox((0, 0), share_label, font=font_small)
    share_w = share_bbox[2] - share_bbox[0]
    draw.text((size - 96 - share_w, 86), share_label, font=font_small, fill=(185, 185, 185))

    # Top title band, kept clear of artwork and decorative circles
    title_left = 360
    title_top = 138
    title_width = 760
    display_font = fit_font_for_multiline(draw, author.display_name, title_width, 68, min_size=34, bold=True, max_lines=2)
    title_end_y = draw_multiline_block(draw, author.display_name, display_font, (250, 250, 250), title_left, title_top, title_width, line_gap=6, max_lines=2)

    subtitle_font = load_font(24, bold=False)
    subtitle_text = 'Your VALIANT Wrapped stage entrance'
    subtitle_y = title_end_y + 10
    draw.text((title_left, subtitle_y), subtitle_text, font=subtitle_font, fill=(174, 174, 174))

    # Artist block starts below title block so long names do not collide
    artist_box_top = max(220, subtitle_y + 26)
    artist_box_bottom = 628
    artist_box = (360, artist_box_top, 1304, artist_box_bottom)
    draw.rounded_rectangle(artist_box, radius=34, fill=(20, 20, 20), outline=(52, 52, 52), width=2)
    draw.text((388, artist_box_top + 28), 'YOUR MUSICAL ALTER EGO', font=font_section, fill=(30, 215, 96))

    artist_name = author.persona.artist_name or 'Still waiting on the stage name reveal'
    artist_name_font = fit_font_for_multiline(draw, artist_name, 620, 54, min_size=26, bold=True, max_lines=2)
    artist_name_y = artist_box_top + 72
    artist_name_end = draw_multiline_block(draw, artist_name, artist_name_font, (245, 245, 245), 388, artist_name_y, 620, line_gap=6, max_lines=2)

    bio_text = author.persona.bio or 'This artist bio is fashionably late, but the research still made the lineup.'
    bio_font = fit_font_for_multiline(draw, bio_text, 620, 24, min_size=16, bold=False, max_lines=4)
    draw.text((388, artist_name_end + 12), 'Bio', font=font_small, fill=(30, 215, 96))
    bio_end = draw_multiline_block(draw, bio_text, bio_font, (205, 205, 205), 388, artist_name_end + 46, 620, line_gap=6, max_lines=4)

    # Stats row adapts to bio height and stays inside the card
    stat_y = min(max(bio_end + 18, artist_box_top + 232), artist_box_bottom - 90)
    stats = [
        ('Papers', str(author.metrics.pub_count) if author.metrics.pub_count > 0 else '—'),
        ('Citations', str(author.metrics.citation_count) if author.metrics.citation_count > 0 else '—'),
        ('Top cites', str(author.metrics.top_paper_citations) if author.metrics.top_paper_citations > 0 else '—'),
    ]
    stat_x = 388
    stat_gap = 146
    for idx, (label, value) in enumerate(stats):
        left = stat_x + idx * stat_gap
        draw.rounded_rectangle((left, stat_y, left + 128, stat_y + 70), radius=22, fill=(29, 29, 29))
        value_font = fit_font_to_width(draw, value, 92, 31, min_size=20, bold=True)
        draw.text((left + 15, stat_y + 10), value, font=value_font, fill=(245, 245, 245))
        draw.text((left + 15, stat_y + 41), label, font=font_stat_label, fill=(173, 173, 173))

    journal = clean_text(author.metrics.top_journal) or 'Still warming up'
    journal_title_x = 860
    journal_top = stat_y + 4
    journal_font = fit_font_for_multiline(draw, journal, 250, 22, min_size=16, bold=False, max_lines=2)
    draw.text((journal_title_x, journal_top), 'Favorite journal', font=font_stat_label, fill=(173, 173, 173))
    draw_multiline_block(draw, journal, journal_font, (235, 235, 235), journal_title_x, journal_top + 24, 250, line_gap=4, max_lines=2)

    # Lower band with album + tracklist side by side
    lower_box = (70, 652, 1330, 1142)
    draw.rounded_rectangle(lower_box, radius=38, fill=(19, 19, 19), outline=(52, 52, 52), width=2)
    draw.text((96, 678), 'NOW SPINNING', font=font_section, fill=(30, 215, 96))
    draw.text((748, 678), 'TRACKLIST', font=font_section, fill=(30, 215, 96))

    album_name = author.persona.album_title or 'Untitled drop'
    album_name_font = fit_font_for_multiline(draw, album_name, 560, 38, min_size=22, bold=True, max_lines=2)
    draw_multiline_block(draw, album_name, album_name_font, (242, 242, 242), 418, 718, 560, line_gap=6, max_lines=2)
    draw.text((418, 806), 'Album cover', font=font_small, fill=(173, 173, 173))

    draw.rounded_rectangle((86, 694, 640, 1110), radius=32, fill=(23, 23, 23))
    card.paste(album, (96, 742), album)

    tracks = list(author.persona.tracklist[:8]) if author.persona.tracklist else []
    if not tracks:
        tracks = ['No tracklist found. Even fictional artists miss deadlines sometimes.']
    track_y = 726
    row_h = 42
    for i, track in enumerate(tracks, start=1):
        prefix = f'{i:02d}. ' if author.persona.tracklist else ''
        line = prefix + clean_text(track)
        row_top = track_y - 4
        row_bottom = track_y + row_h - 4
        draw.rounded_rectangle((748, row_top, 1274, row_bottom), radius=16, fill=(28, 28, 28))
        track_font = fit_font_for_multiline(draw, line, 486, 22, min_size=16, bold=False, max_lines=1)
        draw_multiline_block(draw, line, track_font, (234, 234, 234), 766, track_y + 1, 486, line_gap=4, max_lines=1)
        track_y += row_h

    # Footer closer / encore line, right brand kept inside boundary
    footer_top = 1172
    footer_bottom = size - 90
    draw.rounded_rectangle((70, footer_top, size - 70, footer_bottom), radius=30, fill=(24, 24, 24), outline=(52, 52, 52), width=2)
    encore_text = 'Thanks for an amazing and productive year. We can’t wait for the encore.'
    encore_font = fit_font_for_multiline(draw, encore_text, 900, 33, min_size=22, bold=True, max_lines=2)
    encore_end = draw_multiline_block(draw, encore_text, encore_font, (245, 245, 245), 98, footer_top + 22, 900, line_gap=6, max_lines=2)
    footer_line = f"{author.lifetime_pub_count or author.metrics.pub_count} total papers • {author.lifetime_citation_count or author.metrics.citation_count} total citations"
    draw.text((98, encore_end + 8), footer_line, font=font_small, fill=(175, 175, 175))

    footer_brand = 'VALIANT Wrapped'
    footer_brand_font = fit_font_to_width(draw, footer_brand, 300, 30, min_size=22, bold=True)
    brand_bbox = draw.textbbox((0, 0), footer_brand, font=footer_brand_font)
    brand_w = brand_bbox[2] - brand_bbox[0]
    draw.text((size - 98 - brand_w, footer_top + 34), footer_brand, font=footer_brand_font, fill=(30, 215, 96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(output_path, format='PNG')

def has_current_year_stats(metrics: Metrics) -> bool:
    return any([
        metrics.pub_count > 0,
        metrics.citation_count > 0,
        bool(clean_text(metrics.top_journal)),
        bool(clean_text(metrics.top_paper_title)),
        metrics.top_paper_citations > 0,
    ])


def current_year_stats_intro(author: AuthorRecord) -> str:
    if has_current_year_stats(author.metrics):
        return "A good Wrapped looks back before it looks ahead. Consider this the annual academic mirror check."
    return (
        "No 2025–2026 stats made the charts this time, but every great artist has an acoustic interlude. "
        "The lab lights are still on, the creative engine is still humming, and the next hit could already be in the mix."
    )


def top_paper_line(author: AuthorRecord) -> str:
    if clean_text(author.metrics.top_paper_title):
        return author.metrics.top_paper_title
    if has_current_year_stats(author.metrics):
        return "No top paper was available from the current metrics file."
    return "No lead single dropped in the current academic year, but the discography is far from over."


def should_show_stats_grid(metrics: Metrics) -> bool:
    return any([
        metrics.pub_count > 0,
        metrics.citation_count > 0,
        bool(clean_text(metrics.top_journal)),
        metrics.top_paper_citations > 0,
    ])


def behind_the_scenes_line(author: AuthorRecord) -> str:
    facts = [
        "Fun fact: Chappell Roan spent years building a devoted underground following before suddenly becoming one of pop's fastest-rising names.",
        "Fun fact: Lizzo had been performing and releasing music for years before \"Truth Hurts\" exploded and turned a long grind into a breakthrough moment.",
        "Fun fact: Glass Animals were already well into their career before \"Heat Waves\" slowly caught fire and became a massive global hit.",
        "Fun fact: Doja Cat had been releasing music online for years before \"Say So\" launched her into full-on pop-star orbit.",
        "Fun fact: Billy Strings toured relentlessly in the bluegrass world before his wider breakout brought years of groundwork into the spotlight.",
    ]
    key = clean_text(author.author_label) or clean_text(
        author.display_name) or "author"
    idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(facts)
    return facts[idx]


def metric_chip(label: str, value: str) -> str:
    return (
        '<div class="metric-chip">'
        f'<div class="metric-label">{html_escape(label)}</div>'
        f'<div class="metric-value">{html_escape(value)}</div>'
        '</div>'
    )


def cover_markup(author: AuthorRecord, depth_prefix: str = "") -> str:
    if author.cover_filename:
        src = f"{depth_prefix}assets/album_covers/{author.cover_filename}"
        alt = f"Album cover for {author.display_name}"
        return f'<img class="album-cover" src="{html_escape(src)}" alt="{html_escape(alt)}">'

    line1, line2 = PLACEHOLDER_LINES
    return (
        '<div class="album-cover placeholder-cover">'
        '<div class="placeholder-vinyl"></div>'
        f'<div class="placeholder-line">{html_escape(line1)}</div>'
        f'<div class="placeholder-subline">{html_escape(line2)}</div>'
        '</div>'
    )


def musician_portrait_markup(author: AuthorRecord, depth_prefix: str = "") -> str:
    if author.musician_portrait_filename:
        src = f"{depth_prefix}assets/musician_headshots/{author.musician_portrait_filename}"
        alt = f"Musician portrait for {author.display_name}"
        return f'<img class="musician-portrait" src="{html_escape(src)}" alt="{html_escape(alt)}">'
    return '<div class="empty-state">Musician portrait not found yet. The artist is still getting ready backstage.</div>'


def recommendations_markup(author: AuthorRecord) -> str:
    if author.recommendations:
        items = []
        for rec in author.recommendations:
            meta_parts = []
            if rec.journal:
                meta_parts.append(rec.journal)
            if rec.score is not None:
                meta_parts.append(f"Similarity {rec.score:.2f}")
            meta = " • ".join(meta_parts)

            items.append(
                '<article class="rec-card">'
                f'<div class="rec-rank">#{rec.rank}</div>'
                '<div class="rec-main">'
                f'<h3>"{html_escape(rec.title)}"</h3>'
                f'<p class="rec-meta">{html_escape(meta)}</p>'
                '</div>'
                f'<a class="google-btn" href="{html_escape(rec.google_url)}" target="_blank" rel="noopener noreferrer">Google it</a>'
                '</article>'
            )
        return "\n".join(items)

    fallback = author.recommendation_fallback or DEFAULT_RECOMMENDATION_FALLBACK
    return f'<div class="empty-state">{html_escape(fallback)}</div>'


def tracklist_markup(tracks: Sequence[str]) -> str:
    if not tracks:
        return '<div class="empty-state">No tracklist found. Even fictional artists miss deadlines sometimes.</div>'
    items = [
        (
            '<li class="track-item">'
            f'<span class="track-number">{i:02d}</span>'
            f'<span class="track-title">{html_escape(track)}</span>'
            '</li>'
        )
        for i, track in enumerate(tracks, start=1)
    ]
    return (
        '<div class="tracklist-card">'
        '<div class="tracklist-header">'
        '<div class="tracklist-eyebrow">Now spinning</div>'
        '<div class="tracklist-heading">Tracklist</div>'
        '</div>'
        f'<ol class="tracklist">{"".join(items)}</ol>'
        '</div>'
    )


def home_card_markup(author: AuthorRecord) -> str:
    slug = slugify(author.author_label)
    subtitle = author.persona.artist_name or "Still waiting on the stage name reveal"
    album = author.persona.album_title or "Untitled drop"
    art = (
        musician_portrait_markup(author)
        if author.musician_portrait_filename
        else cover_markup(author)
    )
    return (
        f'<a class="author-card" href="authors/{slug}/">'
        f'{art}'
        '<div class="author-card-body">'
        f'<div class="author-name">{html_escape(author.display_name)}</div>'
        f'<div class="author-subtitle">{html_escape(subtitle)}</div>'
        f'<div class="author-album">{html_escape(album)}</div>'
        f'<div class="author-metrics-line">{author.lifetime_pub_count or author.metrics.pub_count} total papers • {author.lifetime_citation_count or author.metrics.citation_count} total citations</div>'
        '</div>'
        '</a>'
    )


def author_page_markup(author: AuthorRecord, project_title: str, tagline: str, closer: str) -> str:
    title = f"{author.display_name} — {project_title}"
    persona_artist = author.persona.artist_name or "TBD on streaming platforms"
    persona_album = author.persona.album_title or "Untitled Album"
    persona_bio = author.persona.bio or "This artist bio is fashionably late, but the research still made the lineup."

    show_stats_grid = should_show_stats_grid(author.metrics)
    show_top_paper = bool(clean_text(author.metrics.top_paper_title))
    stats_intro = current_year_stats_intro(author)

    metric_items: List[str] = []
    if author.metrics.pub_count > 0:
        metric_items.append(metric_chip("Papers published",
                            str(author.metrics.pub_count)))
    if author.metrics.citation_count > 0:
        metric_items.append(metric_chip(
            "Citations", str(author.metrics.citation_count)))
    if clean_text(author.metrics.top_journal):
        metric_items.append(metric_chip(
            "Favorite journal", author.metrics.top_journal))
    if author.metrics.top_paper_citations > 0:
        metric_items.append(metric_chip(
            "Top paper citations", str(author.metrics.top_paper_citations)))

    if not show_stats_grid and not show_top_paper:
        stats_html = f'<div class="come-up-line">{html_escape(behind_the_scenes_line(author))}</div>'
    else:
        metrics_html = f'<div class="metrics-grid">{"".join(metric_items)}</div>' if metric_items else ""
        top_paper_html = ""
        if show_top_paper:
            top_paper_html = (
                '<div class="top-paper-block">'
                '<div class="metric-label">Top paper</div>'
                f'<div class="top-paper-title">{html_escape(author.metrics.top_paper_title)}</div>'
                '</div>'
            )
        stats_html = f"{metrics_html}{top_paper_html}"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <link rel="stylesheet" href="../../assets/styles.css">
</head>
<body class="site-body">
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">{html_escape(project_title)}</div>
      <div class="sidebar-tag">Research, remixed.</div>
      <nav class="sidebar-nav">
        <a href="../../index.html" class="nav-link active">← Back to browse</a>
        <a href="#publications" class="nav-link">Academic year stats</a>
        <a href="#persona" class="nav-link">Artist persona</a>
        <a href="#album" class="nav-link">Album drop</a>
        <a href="#recommendations" class="nav-link">Recommended papers</a>
        <a href="#closing" class="nav-link">Encore</a>
      </nav>
    </aside>

    <main class="main-panel">
      <section class="hero author-hero">
        <div class="hero-copy hero-copy-only">
          <div class="eyebrow">{html_escape(project_title)}</div>
          <h1>{html_escape(author.display_name)}</h1>
          <div id="publications" class="hero-stats-block">
            <div class="hero-stats-title">2025–2026 Academic Year Stats</div>
            <p class="section-copy hero-stats-copy">{html_escape(stats_intro)}</p>
            {stats_html}
          </div>
        </div>
      </section>

      <section id="persona" class="content-card">
        <h2>Your Musical Alter Ego</h2>
        <p class="section-copy">Generated from the themes, patterns, and publication trail in your real Scopus data — then remixed for your amusement.</p>
        <div class="persona-hero-grid">
          <div class="persona-portrait-wrap">{musician_portrait_markup(author, depth_prefix='../../')}</div>
          <div>
            <div class="persona-title persona-title-large">{html_escape(persona_artist)}</div>
            <p>{html_escape(persona_bio)}</p>
          </div>
        </div>
      </section>

      <section id="album" class="content-card album-section">
        <div class="section-kicker">{html_escape(SECTION_KICKERS["album"])}</div>
        <h2>{html_escape(persona_album)}</h2>
        <div class="album-grid">
          <div class="album-art-wrap">{cover_markup(author, depth_prefix='../../')}</div>
          <div class="tracklist-wrap">{tracklist_markup(author.persona.tracklist)}</div>
        </div>
      </section>

      <section id="recommendations" class="content-card">
        <div class="section-kicker">{html_escape(SECTION_KICKERS["recommendations"])}</div>
        <h2>Recommended Papers</h2>
        <p class="section-copy">A collection of papers from fellow VALIANT peers that we thought might belong in your intellectual queue.</p>
        <div class="recommendation-list">{recommendations_markup(author)}</div>
      </section>

      <section id="share" class="content-card">
        <div class="section-kicker">{html_escape(SECTION_KICKERS["share"])}</div>
        <h2>Share to social media</h2>
        <p class="section-copy">Download your custom share card and take your VALIANT Wrapped on tour.</p>
        <div class="share-actions">
          <a class="browse-btn" href="../../assets/share_cards/{html_escape(author.share_card_filename)}" download>Download share card</a>
          <button class="share-btn" type="button" onclick="copyShareLink()">Copy page link</button>
        </div>
      </section>

      <section id="closing" class="content-card closing-card">
        <div class="section-kicker">{html_escape(SECTION_KICKERS["closing"])}</div>
        <h2>Encore</h2>
        <p>{html_escape(closer)}</p>
        <a href="../../index.html" class="browse-btn">Back to browse</a>
      </section>
    </main>
  </div>
</body>
</html>
"""


def homepage_markup(authors: Sequence[AuthorRecord], project_title: str, tagline: str, closer: str) -> str:
    card_html = "\n".join(home_card_markup(author) for author in authors)
    author_data = [
        {
            "label": a.author_label,
            "display_name": a.display_name,
            "artist_name": a.persona.artist_name,
            "album_title": a.persona.album_title,
            "pub_count": a.metrics.pub_count,
            "citation_count": a.metrics.citation_count,
            "lifetime_pub_count": a.lifetime_pub_count,
            "lifetime_citation_count": a.lifetime_citation_count,
            "href": f"authors/{slugify(a.author_label)}/",
            "share_card_filename": a.share_card_filename,
        }
        for a in authors
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(project_title)}</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body class="site-body">
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">{html_escape(project_title)}</div>
      <div class="sidebar-tag">Research, remixed.</div>
      <nav class="sidebar-nav">
        <a href="#browse" class="nav-link active">Browse authors</a>
        <a href="#about" class="nav-link">About</a>
        <a href="#closing" class="nav-link">Encore</a>
      </nav>
    </aside>

    <main class="main-panel">
      <section class="hero">
        <div class="hero-copy">
          <div class="eyebrow">Annual research remix</div>
          <h1>{html_escape(project_title)}</h1>
          <p class="lede">{html_escape(tagline)}</p>
          <p class="support">A playful, Spotify-inspired look at our center's authors, their 2025-present publication highlights, fictional artist personas, imaginary albums, and recommended reads from across the lab ecosystem.</p>
        </div>
        <div class="hero-glow"></div>
      </section>

      <section id="browse" class="content-card">
        <div class="section-kicker">{html_escape(SECTION_KICKERS["browse"])}</div>
        <h2>All authors</h2>
        <div class="browse-toolbar">
          <input id="authorSearch" class="search-input" type="text" placeholder="Search authors, artist names, or albums" autocomplete="off">
          <div class="browse-count"><span id="visibleCount">{len(authors)}</span> authors in rotation</div>
        </div>
        <div id="authorGrid" class="author-grid">{card_html}</div>
      </section>

      <section id="about" class="content-card">
        <div class="section-kicker">{html_escape(SECTION_KICKERS["about"])}</div>
        <h2>What is VALIANT Wrapped?</h2>
        <p>Part publication showcase, part creative experiment, part affectionate roast of generative AI. Each profile pulls together real research outputs and then lets the hallucinations take the aux cord for the persona, album, and tracklist.</p>
      </section>

      <section id="closing" class="content-card closing-card">
        <div class="section-kicker">{html_escape(SECTION_KICKERS["closing"])}</div>
        <h2>Encore</h2>
        <p>{html_escape(closer)}</p>
      </section>
    </main>
  </div>

  <script id="authorData" type="application/json">{json.dumps(author_data, ensure_ascii=False)}</script>
  <script src="assets/app.js"></script>
</body>
</html>
"""


def styles_css() -> str:
    return """
:root {
  --bg: #0a0a0a;
  --panel: #121212;
  --panel-2: #181818;
  --line: #2a2a2a;
  --text: #f5f5f5;
  --muted: #b3b3b3;
  --green: #1db954;
  --green-2: #1ed760;
  --chip: #202020;
  --shadow: 0 18px 40px rgba(0,0,0,.35);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body.site-body {
  margin: 0;
  background:
    radial-gradient(circle at top right, rgba(29,185,84,.16), transparent 24%),
    linear-gradient(180deg, #070707 0%, #0f0f0f 100%);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: inherit; text-decoration: none; }
.app-shell { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
.sidebar {
  position: sticky; top: 0; height: 100vh; padding: 28px 22px;
  border-right: 1px solid rgba(255,255,255,.06);
  background: linear-gradient(180deg, rgba(18,18,18,.98), rgba(10,10,10,.98));
}
.brand { font-size: 1.8rem; font-weight: 800; letter-spacing: -.03em; }
.sidebar-tag { margin-top: 8px; color: var(--muted); font-size: .95rem; }
.sidebar-nav { margin-top: 36px; display: grid; gap: 8px; }
.nav-link { padding: 12px 14px; border-radius: 14px; color: var(--muted); transition: .18s ease; }
.nav-link:hover, .nav-link.active { background: rgba(255,255,255,.06); color: var(--text); }
.main-panel { padding: 28px; }
.hero {
  position: relative; overflow: hidden;
  background:
    radial-gradient(circle at top right, rgba(29,185,84,.10), transparent 22%),
    radial-gradient(circle at bottom left, rgba(255,255,255,.06), transparent 26%),
    linear-gradient(145deg, rgba(24,24,24,.98) 0%, rgba(14,14,14,.98) 55%, rgba(10,10,10,.98) 100%);
  border: 1px solid rgba(255,255,255,.08); border-radius: 28px;
  padding: 38px; box-shadow: var(--shadow);
}
.author-hero { display: block; background:
  radial-gradient(circle at top center, rgba(255,255,255,.05), transparent 24%),
  linear-gradient(180deg, rgba(22,22,22,.98) 0%, rgba(12,12,12,.98) 100%); }
.hero-copy-only { max-width: 100%; }
.hero-no-art .hero-copy { max-width: 980px; }
.hero-copy { max-width: 850px; }
.hero-stats-block { margin-top: 22px; }
.hero-stats-title { font-size: 1.15rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: 8px; }
.hero-stats-copy { max-width: 760px; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .78rem; color: var(--green-2); font-weight: 700; }
.hero h1 { margin: 8px 0 14px; font-size: clamp(2.4rem, 5vw, 4.4rem); line-height: .95; letter-spacing: -.04em; }
.lede { font-size: 1.16rem; line-height: 1.55; max-width: 760px; color: #f0f0f0; }
.support { color: var(--muted); max-width: 720px; line-height: 1.6; }
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 22px; }
.pill { padding: 9px 14px; border-radius: 999px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.08); font-size: .92rem; }
.hero-portrait { display: flex; align-items: stretch; justify-content: center; }
.hero-portrait .musician-portrait,
.hero-portrait .empty-state {
  width: 100%;
  max-width: 360px;
}
.hero-glow {
  position: absolute; right: -80px; top: -80px; width: 240px; height: 240px; border-radius: 50%;
  background: radial-gradient(circle, rgba(29,185,84,.38), transparent 65%); filter: blur(12px);
}
.content-card {
  margin-top: 24px; background:
    radial-gradient(circle at top right, rgba(255,255,255,.025), transparent 24%),
    rgba(18,18,18,.94);
  border: 1px solid rgba(255,255,255,.06); border-radius: 24px;
  padding: 28px; box-shadow: var(--shadow);
}
.section-kicker { color: var(--green-2); text-transform: uppercase; letter-spacing: .12em; font-size: .76rem; font-weight: 700; margin-bottom: 6px; }
.content-card h2 { margin: 0 0 12px; font-size: 1.7rem; letter-spacing: -.03em; }
.section-copy { color: var(--muted); margin-top: 0; }
.browse-toolbar { display: flex; gap: 14px; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-top: 18px; margin-bottom: 18px; }
.search-input {
  min-width: min(100%, 420px); width: 420px; background: #0f0f0f;
  border: 1px solid rgba(255,255,255,.08); color: var(--text); padding: 14px 16px; border-radius: 14px; outline: none;
}
.search-input:focus { border-color: rgba(29,185,84,.7); box-shadow: 0 0 0 4px rgba(29,185,84,.12); }
.browse-count { color: var(--muted); font-size: .95rem; }
.author-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 18px; }
.author-card {
  background: var(--panel-2); border: 1px solid rgba(255,255,255,.06);
  border-radius: 20px; overflow: hidden; transition: transform .18s ease, border-color .18s ease, background .18s ease;
}
.author-card:hover { transform: translateY(-3px); border-color: rgba(29,185,84,.55); background: #1c1c1c; }
.author-card-body { padding: 14px 14px 18px; }
.author-name { font-weight: 800; line-height: 1.25; }
.author-subtitle, .author-album, .author-metrics-line, .rec-meta { color: var(--muted); }
.author-subtitle { margin-top: 6px; font-size: .94rem; }
.author-album { margin-top: 4px; font-size: .92rem; }
.author-metrics-line { margin-top: 10px; font-size: .88rem; }
.album-cover { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; background: linear-gradient(135deg, #161616, #0c0c0c); }
.musician-portrait { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; border-radius: 22px; background: linear-gradient(135deg, #161616, #0c0c0c); }
.placeholder-cover {
  position: relative; display: grid; place-items: center; align-content: center; gap: 10px; padding: 18px; text-align: center;
  background: radial-gradient(circle at top right, rgba(29,185,84,.26), transparent 28%), linear-gradient(140deg, #171717, #090909);
}
.placeholder-vinyl {
  width: 90px; height: 90px; border-radius: 50%;
  background: radial-gradient(circle at center, #0d0d0d 0 14px, #1db954 15px 19px, #0d0d0d 20px 100%);
  box-shadow: inset 0 0 0 8px rgba(255,255,255,.03);
}
.placeholder-line { font-weight: 800; }
.placeholder-subline { font-size: .9rem; color: var(--muted); max-width: 20ch; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 18px; }
.metric-chip { background: var(--chip); border: 1px solid rgba(255,255,255,.06); border-radius: 18px; padding: 16px; }
.metric-label { color: var(--muted); font-size: .84rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value, .persona-title, .top-paper-title { margin-top: 8px; font-size: 1.05rem; font-weight: 700; line-height: 1.35; }
.top-paper-block, .summary-block { margin-top: 18px; padding: 18px; border-radius: 18px; background: var(--chip); border: 1px solid rgba(255,255,255,.06); }
.persona-grid, .album-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.persona-grid-single { grid-template-columns: minmax(0, 1fr); max-width: 420px; }
.persona-hero-grid { display: grid; grid-template-columns: minmax(260px, 360px) 1fr; gap: 22px; align-items: start; }
.persona-portrait-wrap { max-width: 360px; }
.album-section .album-art-wrap { max-width: 420px; }
.tracklist-wrap { display: flex; }
.tracklist-card {
  width: 100%;
  background:
    radial-gradient(circle at top right, rgba(29,185,84,.16), transparent 28%),
    linear-gradient(160deg, #171717, #101010);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 22px;
  padding: 20px 22px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
}
.tracklist-header { margin-bottom: 14px; }
.tracklist-eyebrow {
  color: var(--green-2);
  text-transform: uppercase;
  letter-spacing: .12em;
  font-size: .72rem;
  font-weight: 800;
}
.tracklist-heading {
  margin-top: 6px;
  font-size: 1.35rem;
  font-weight: 800;
  letter-spacing: -.03em;
}
.tracklist {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 10px;
}
.track-item {
  display: grid;
  grid-template-columns: 52px 1fr;
  gap: 12px;
  align-items: center;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255,255,255,.035);
  border: 1px solid rgba(255,255,255,.05);
}
.track-item:hover {
  background: rgba(29,185,84,.08);
  border-color: rgba(29,185,84,.24);
}
.track-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  border-radius: 999px;
  background: rgba(29,185,84,.14);
  color: var(--green-2);
  font-size: .9rem;
  font-weight: 800;
  letter-spacing: .04em;
}
.track-title {
  font-size: 1rem;
  font-weight: 700;
  line-height: 1.35;
}
.rec-card {
  display: grid; grid-template-columns: auto 1fr auto; gap: 16px; align-items: center;
  padding: 16px 18px; border-radius: 18px; background: var(--chip);
  border: 1px solid rgba(255,255,255,.06); margin-bottom: 12px;
}
.rec-rank {
  width: 44px; height: 44px; border-radius: 50%; display: grid; place-items: center;
  background: rgba(29,185,84,.14); color: var(--green-2); font-weight: 800;
}
.rec-main h3 { margin: 0; line-height: 1.35; }
.secondary-link { color: var(--green-2); font-size: .9rem; font-weight: 700; }
.google-btn, .browse-btn {
  display: inline-flex; align-items: center; justify-content: center; padding: 12px 16px;
  border-radius: 999px; background: var(--green); color: #08130b; font-weight: 800; white-space: nowrap;
}
.google-btn:hover, .browse-btn:hover { background: var(--green-2); }
.empty-state {
  background: rgba(255,255,255,.04); border: 1px dashed rgba(255,255,255,.12);
  border-radius: 18px; padding: 18px; color: var(--muted);
}
.share-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 18px;
}
.share-btn {
  display: inline-flex; align-items: center; justify-content: center; padding: 12px 16px;
  border-radius: 999px; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08);
  color: var(--text); font-weight: 800; white-space: nowrap; cursor: pointer;
}
.share-btn:hover { background: rgba(255,255,255,.10); }
.come-up-line {
  margin-top: 18px;
  padding: 18px 20px;
  border-radius: 18px;
  background:
    radial-gradient(circle at top right, rgba(29,185,84,.08), transparent 28%),
    linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08);
  color: #f3f3f3;
  font-size: 1.02rem;
  line-height: 1.6;
  max-width: 900px;
}
.closing-card { background: linear-gradient(135deg, rgba(29,185,84,.18), rgba(255,255,255,.02)); }
@media (max-width: 980px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid rgba(255,255,255,.06); }
  .author-hero, .album-grid, .persona-grid, .persona-hero-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .main-panel { padding: 16px; }
  .hero, .content-card { padding: 20px; border-radius: 20px; }
  .rec-card { grid-template-columns: 1fr; align-items: start; }
  .search-input { width: 100%; min-width: 100%; }
}
"""


def app_js() -> str:
    return """
(function () {
  const input = document.getElementById('authorSearch');
  const grid = document.getElementById('authorGrid');
  const countEl = document.getElementById('visibleCount');
  if (!input || !grid || !countEl) return;
  const cards = Array.from(grid.querySelectorAll('.author-card'));
  function normalize(text) { return (text || '').toLowerCase().trim(); }
  function filterCards() {
    const q = normalize(input.value);
    let visible = 0;
    cards.forEach((card) => {
      const haystack = normalize(card.textContent);
      const show = !q || haystack.includes(q);
      card.style.display = show ? '' : 'none';
      if (show) visible += 1;
    });
    countEl.textContent = String(visible);
  }
  input.addEventListener('input', filterCards);
})();

function copyShareLink() {
  const url = window.location.href;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(url).then(() => {
      window.alert('Page link copied to clipboard');
    }).catch(() => {
      window.prompt('Copy this page link:', url);
    });
  } else {
    window.prompt('Copy this page link:', url);
  }
}
window.copyShareLink = copyShareLink;
"""


def build_site(
    output_dir: Path,
    authors: Sequence[AuthorRecord],
    cover_src_dir: Path,
    musician_headshot_src_dir: Path,
    project_title: str,
    tagline: str,
    closer: str,
) -> None:
    ensure_dir(output_dir)
    assets_dir = output_dir / "assets"
    author_pages_dir = output_dir / "authors"
    album_out_dir = assets_dir / "album_covers"
    musician_headshot_out_dir = assets_dir / "musician_headshots"
    share_card_out_dir = assets_dir / "share_cards"
    data_dir = output_dir / "data"

    ensure_dir(assets_dir)
    ensure_dir(author_pages_dir)
    ensure_dir(album_out_dir)
    ensure_dir(musician_headshot_out_dir)
    ensure_dir(share_card_out_dir)
    ensure_dir(data_dir)

    if cover_src_dir.exists():
        for cover_path in cover_src_dir.glob("*.png"):
            shutil.copy2(cover_path, album_out_dir / cover_path.name)

    if musician_headshot_src_dir.exists():
        for portrait_path in musician_headshot_src_dir.glob("*.*"):
            if portrait_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                shutil.copy2(
                    portrait_path, musician_headshot_out_dir / portrait_path.name)

    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    (assets_dir / "styles.css").write_text(styles_css(), encoding="utf-8")
    (assets_dir / "app.js").write_text(app_js(), encoding="utf-8")
    (data_dir / "authors.json").write_text(
        json.dumps(
            [
                {
                    "author_label": a.author_label,
                    "display_name": a.display_name,
                    "artist_name": a.persona.artist_name,
                    "album_title": a.persona.album_title,
                    "pub_count": a.metrics.pub_count,
                    "citation_count": a.metrics.citation_count,
                    "lifetime_pub_count": a.lifetime_pub_count,
                    "lifetime_citation_count": a.lifetime_citation_count,
                    "has_expertise_summary": bool(a.expertise_summary),
                    "recommendation_count": len(a.recommendations),
                    "share_card_filename": a.share_card_filename,
                }
                for a in authors
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        homepage_markup(authors, project_title, tagline, closer),
        encoding="utf-8",
    )

    for author in authors:
        slug = slugify(author.author_label)
        author.share_card_filename = f"{slug}.png"
        build_share_card(
            author=author,
            output_path=share_card_out_dir / author.share_card_filename,
            cover_src_dir=cover_src_dir,
            musician_headshot_src_dir=musician_headshot_src_dir,
            project_title=project_title,
        )
        author_dir = author_pages_dir / slug
        ensure_dir(author_dir)
        (author_dir / "index.html").write_text(
            author_page_markup(author, project_title, tagline, closer),
            encoding="utf-8",
        )


def extract_query_text_from_expertise(summary: str) -> str:
    return clean_text(summary)


def parse_author_ids(raw: str) -> List[str]:
    raw = clean_text(raw)
    if not raw:
        return []
    ids = re.findall(r"\d+", raw)
    seen = set()
    ordered = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def google_search_url(title: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(title)}"


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path_for(model_id: str, scopus_db: Path, cache_dir: Path) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    fp = file_fingerprint(scopus_db)
    return cache_dir / f"paper_embeddings__{safe_model}__{fp}.pkl"


def build_paper_records(scopus_df: pd.DataFrame) -> List[PaperRecord]:
    title_col = pick_existing_col(scopus_df.columns, TITLE_COLS)
    abstract_col = pick_existing_col(scopus_df.columns, ABSTRACT_COLS)
    keyword_col = pick_existing_col(scopus_df.columns, KEYWORD_COLS)
    journal_col = pick_existing_col(scopus_df.columns, JOURNAL_COLS)
    author_id_col = pick_existing_col(scopus_df.columns, AUTHOR_ID_COLS)
    doi_col = pick_existing_col(scopus_df.columns, DOI_COLS)
    link_col = pick_existing_col(scopus_df.columns, LINK_COLS)

    if not title_col:
        raise ValueError(
            f"Could not find a title column in {scopus_df.columns.tolist()}")

    records: List[PaperRecord] = []
    for _, row in scopus_df.iterrows():
        title = clean_text(row.get(title_col, ""))
        if not title:
            continue
        abstract = clean_text(row.get(abstract_col, "")
                              ) if abstract_col else ""
        keywords = clean_text(row.get(keyword_col, "")) if keyword_col else ""
        journal = clean_text(row.get(journal_col, "")) if journal_col else ""
        author_ids_raw = clean_text(
            row.get(author_id_col, "")) if author_id_col else ""
        doi = clean_text(row.get(doi_col, "")) if doi_col else ""
        scopus_link = clean_text(row.get(link_col, "")) if link_col else ""
        parts = [f"Title: {title}"]
        if journal:
            parts.append(f"Venue: {journal}")
        if keywords:
            parts.append(f"Keywords: {keywords}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        combined = "\n".join(parts)
        records.append(
            PaperRecord(
                title=title,
                abstract=abstract,
                keywords=keywords,
                journal=journal,
                author_ids_raw=author_ids_raw,
                doi=doi,
                scopus_link=scopus_link,
                combined_text=combined,
            )
        )
    if not records:
        raise ValueError(
            "No usable paper rows were found in the Scopus export.")
    return records


def load_embedding_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_id, device=device)


def encode_texts(model, texts: Sequence[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_or_create_paper_embedding_cache(
    model,
    model_id: str,
    scopus_db: Path,
    records: Sequence[PaperRecord],
    cache_dir: Path,
    batch_size: int,
    use_cache: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path_for(model_id, scopus_db, cache_dir)

    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        titles = payload.get("titles", [])
        if titles == [r.title for r in records]:
            return np.asarray(payload["embeddings"], dtype=np.float32)

    embeddings = encode_texts(
        model, [r.combined_text for r in records], batch_size=batch_size)

    if use_cache:
        payload = {
            "titles": [r.title for r in records],
            "embeddings": embeddings,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(payload, f)

    return embeddings


def cosine_scores(query_embedding: np.ndarray, paper_embeddings: np.ndarray) -> np.ndarray:
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]
    return np.dot(paper_embeddings, query_embedding)


def author_in_record(scopus_id: str, record: PaperRecord) -> bool:
    if not scopus_id:
        return False
    record_ids = parse_author_ids(record.author_ids_raw)
    if scopus_id in record_ids:
        return True
    raw = clean_text(record.author_ids_raw)
    return re.search(rf"(?<!\d){re.escape(scopus_id)}(?!\d)", raw) is not None


def recommend_for_author(
    author: AuthorRecord,
    model,
    paper_records: Sequence[PaperRecord],
    paper_embeddings: np.ndarray,
    top_k: int,
    min_similarity: float,
) -> Tuple[List[Recommendation], str]:
    query_text = extract_query_text_from_expertise(author.expertise_summary)
    if not query_text:
        return [], DEFAULT_RECOMMENDATION_FALLBACK

    query_embedding = encode_texts(model, [query_text], batch_size=1)[0]
    scores = cosine_scores(query_embedding, paper_embeddings)
    ranked_idx = np.argsort(scores)[::-1]

    recommendations: List[Recommendation] = []
    seen_titles = set()

    for idx in ranked_idx:
        record = paper_records[int(idx)]
        key = record.title.lower()
        if key in seen_titles:
            continue
        if author_in_record(author.scopus_id, record):
            continue

        score = float(scores[int(idx)])
        if score < min_similarity:
            continue

        seen_titles.add(key)
        recommendations.append(
            Recommendation(
                rank=len(recommendations) + 1,
                title=record.title,
                google_url=google_search_url(record.title),
                score=score,
                journal=record.journal,
                doi=record.doi,
                scopus_link=record.scopus_link,
            )
        )
        if len(recommendations) >= top_k:
            break

    if recommendations:
        return recommendations, ""
    return [], DEFAULT_RECOMMENDATION_FALLBACK


def write_recommendations_txt(path: Path, author: AuthorRecord, recs: Sequence[Recommendation], fallback_text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"AUTHOR_LABEL: {author.author_label}\n")
        if recs:
            f.write("RECOMMENDATIONS:\n")
            for rec in recs:
                f.write(f"{rec.rank}. {clean_text(rec.title)}\n")
                f.write(f"   LOOKUP_TEXT: Google it\n")
                f.write(f"   GOOGLE_URL: {clean_text(rec.google_url)}\n")
                if rec.score is not None:
                    f.write(f"   SCORE: {rec.score:.4f}\n")
                if rec.journal:
                    f.write(f"   JOURNAL: {clean_text(rec.journal)}\n")
                if rec.doi:
                    f.write(f"   DOI: {clean_text(rec.doi)}\n")
                if rec.scopus_link:
                    f.write(f"   SCOPUS_LINK: {clean_text(rec.scopus_link)}\n")
        else:
            f.write("RECOMMENDATIONS:\n")
            f.write("(none)\n")
            f.write(
                f"FALLBACK_TEXT: {clean_text(fallback_text or DEFAULT_RECOMMENDATION_FALLBACK)}\n")


def build_recommendations_from_scopus(
    authors: Sequence[AuthorRecord],
    scopus_db: Path,
    model_id: str,
    recommendation_count: int,
    min_similarity: float,
    batch_size: int,
    cache_dir: Path,
    no_cache: bool,
) -> Dict[str, Tuple[List[Recommendation], str]]:
    if not scopus_db.exists():
        raise FileNotFoundError(f"Scopus database CSV not found: {scopus_db}")

    scopus_df = read_csv_flexible(scopus_db)
    paper_records = build_paper_records(scopus_df)
    model = load_embedding_model(model_id)
    paper_embeddings = load_or_create_paper_embedding_cache(
        model=model,
        model_id=model_id,
        scopus_db=scopus_db,
        records=paper_records,
        cache_dir=cache_dir,
        batch_size=batch_size,
        use_cache=not no_cache,
    )

    out: Dict[str, Tuple[List[Recommendation], str]] = {}
    for author in authors:
        recs, fallback = recommend_for_author(
            author=author,
            model=model,
            paper_records=paper_records,
            paper_embeddings=paper_embeddings,
            top_k=max(1, recommendation_count),
            min_similarity=min_similarity,
        )
        out[author.author_label] = (recs, fallback)
    return out


def list_normalized_file_map(path_glob: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in path_glob:
        exact_key = canonical_author_label(p.stem)
        normalized_key = normalized_author_key(p.stem)
        if exact_key and exact_key not in out:
            out[exact_key] = p
        if normalized_key and normalized_key not in out:
            out[normalized_key] = p
    return out


def collect_authors(
    metrics_csv: Path,
    expertise_dir: Path,
    persona_dir: Path,
    cover_dir: Path,
    musician_headshots_dir: Path,
    recommendations_dir: Path,
    author_csv_dir: Path,
    verbose: bool = False,
) -> List[AuthorRecord]:
    metrics_map = parse_metrics_csv(
        metrics_csv) if metrics_csv.exists() else {}
    lifetime_metrics_map = build_lifetime_metrics_map(author_csv_dir)

    expertise_map: Dict[str, str] = {}
    if expertise_dir.exists():
        for p in sorted(expertise_dir.glob("*.txt")):
            expertise_map[canonical_author_label(
                p.stem)] = parse_expertise_txt(p)

    persona_map: Dict[str, Persona] = {}
    if persona_dir.exists():
        for p in sorted(persona_dir.glob("*.txt")):
            persona_map[canonical_author_label(p.stem)] = parse_persona_txt(p)

    rec_map: Dict[str, Tuple[List[Recommendation], str]] = {}
    if recommendations_dir.exists():
        for p in sorted(recommendations_dir.glob("*.txt")):
            rec_map[canonical_author_label(
                p.stem)] = parse_recommendations_txt(p)

    cover_map = list_normalized_file_map(
        cover_dir.glob("*.png")) if cover_dir.exists() else {}
    portrait_map = (
        list_normalized_file_map(
            p for p in musician_headshots_dir.glob("*.*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if musician_headshots_dir.exists()
        else {}
    )

    # Build the page list only from core record sources so normalized fallback keys
    # used for portrait/cover lookup do not create duplicate author pages.
    source_labels = sorted(set(metrics_map) | set(lifetime_metrics_map) | set(
        expertise_map) | set(persona_map) | set(rec_map))

    # If none of the core text/metrics inputs are present, fall back to image-derived labels.
    if not source_labels:
        source_labels = sorted({
            key for key in set(cover_map) | set(portrait_map)
            if key == canonical_author_label(key)
        })

    # Prefer the most complete canonical label for each normalized author key.
    def label_priority(label: str) -> tuple[int, int, str]:
        canonical = canonical_author_label(label)
        has_scopus = 1 if extract_scopus_id(canonical) else 0
        return (has_scopus, len(canonical), canonical)

    preferred_label_by_normalized: Dict[str, str] = {}
    for label in source_labels:
        normalized_label = normalized_author_key(label)
        current = preferred_label_by_normalized.get(normalized_label)
        if current is None or label_priority(label) > label_priority(current):
            preferred_label_by_normalized[normalized_label] = label

    labels = sorted(set(preferred_label_by_normalized.values()))

    authors: List[AuthorRecord] = []
    for label in labels:
        normalized_label = normalized_author_key(label)
        persona = persona_map.get(label) or persona_map.get(
            normalized_label, Persona())
        recommendations, fallback = rec_map.get(
            label) or rec_map.get(normalized_label) or ([], "")
        cover_path = cover_map.get(label) or cover_map.get(normalized_label)
        portrait_path = portrait_map.get(
            label) or portrait_map.get(normalized_label)
        lifetime_pub_count, lifetime_citation_count = lifetime_metrics_map.get(
            label) or lifetime_metrics_map.get(normalized_label) or (0, 0)
        authors.append(
            AuthorRecord(
                author_label=label,
                display_name=infer_display_name(label),
                scopus_id=extract_scopus_id(label),
                expertise_summary=expertise_map.get(
                    label) or expertise_map.get(normalized_label, ""),
                metrics=metrics_map.get(label) or metrics_map.get(
                    normalized_label, Metrics()),
                lifetime_pub_count=lifetime_pub_count,
                lifetime_citation_count=lifetime_citation_count,
                persona=persona,
                cover_filename=cover_path.name if cover_path else "",
                musician_portrait_filename=portrait_path.name if portrait_path else "",
                recommendations=recommendations,
                recommendation_fallback=fallback,
            )
        )

    authors.sort(key=lambda a: (a.display_name.split()
                 [-1].lower(), a.display_name.lower()))

    if verbose:
        print(f"[collect] metrics labels: {len(metrics_map)}")
        print(f"[collect] expertise dir: {expertise_dir}")
        print(
            f"[collect] lifetime metrics labels: {len(lifetime_metrics_map)}")
        print(f"[collect] expertise labels: {len(expertise_map)}")
        print(f"[collect] persona labels: {len(persona_map)}")
        print(f"[collect] recommendation labels: {len(rec_map)}")
        print(f"[collect] cover labels: {len(cover_map)}")
        print(f"[collect] portrait labels: {len(portrait_map)}")
        print(f"[collect] canonical page labels: {len(labels)}")
        print(f"[collect] total assembled authors: {len(authors)}")

        missing_expertise = [
            a.author_label for a in authors if not a.expertise_summary]
        missing_recs = [
            a.author_label for a in authors if not a.recommendations]
        missing_portraits = [
            a.author_label for a in authors if not a.musician_portrait_filename]
        if missing_expertise:
            print(
                f"[collect] authors missing expertise summaries: {len(missing_expertise)}")
        if missing_recs:
            print(
                f"[collect] authors missing recommendations: {len(missing_recs)}")
        if missing_portraits:
            print(
                f"[collect] authors missing musician portraits: {len(missing_portraits)}")

    return authors


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build the VALIANT Wrapped static website.")
    ap.add_argument("--scopus-db", default="",
                    help="Master Scopus export used for optional on-the-fly paper recommendations.")
    ap.add_argument("--metrics-csv", default="author_summary_2025_present.csv")
    ap.add_argument("--author-csv-dir", default="author_csvs",
                    help="Directory of per-author CSVs used to compute lifetime paper and citation totals for the index page.")
    ap.add_argument("--expertise-dir", default="outputs/author_expertise_txt")
    ap.add_argument("--persona-dir",
                    default="outputs/author_music_personas_txt")
    ap.add_argument("--cover-dir", default="outputs/album_covers")
    ap.add_argument("--musician-headshots-dir",
                    default="outputs/musician_headshots")
    ap.add_argument("--recommendations-dir",
                    default="outputs/paper_recommendations/per_author_txt")
    ap.add_argument("--output-dir", default="docs")
    ap.add_argument("--project-title", default="VALIANT Wrapped")
    ap.add_argument("--tagline", default=DEFAULT_TAGLINE)
    ap.add_argument("--closer", default=DEFAULT_CLOSER)

    ap.add_argument(
        "--generate-recommendations",
        action="store_true",
        help="Generate recommendations during site build using --scopus-db and expertise summaries."
    )
    ap.add_argument("--recommendation-model-id",
                    default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--recommendation-count", type=int, default=5)
    ap.add_argument("--min-similarity", type=float, default=0.30)
    ap.add_argument("--recommendation-batch-size", type=int, default=64)
    ap.add_argument("--recommendation-cache-dir",
                    default=".cache/paper_recommender")
    ap.add_argument("--no-recommendation-cache", action="store_true")
    ap.add_argument(
        "--write-generated-recommendations",
        action="store_true",
        help="When generating recommendations on the fly, also save per-author TXT outputs into --recommendations-dir."
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print counts and diagnostics for collected inputs."
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    metrics_csv = Path(args.metrics_csv)
    author_csv_dir = Path(args.author_csv_dir)
    expertise_dir = Path(args.expertise_dir)
    persona_dir = Path(args.persona_dir)
    cover_dir = Path(args.cover_dir)
    musician_headshots_dir = Path(args.musician_headshots_dir)
    recommendations_dir = Path(args.recommendations_dir)
    output_dir = Path(args.output_dir)
    scopus_db = Path(args.scopus_db) if args.scopus_db else None

    authors = collect_authors(
        metrics_csv=metrics_csv,
        expertise_dir=expertise_dir,
        persona_dir=persona_dir,
        cover_dir=cover_dir,
        musician_headshots_dir=musician_headshots_dir,
        recommendations_dir=recommendations_dir,
        author_csv_dir=author_csv_dir,
        verbose=args.verbose,
    )

    if not authors:
        raise RuntimeError(
            "No authors could be assembled from the provided inputs. "
            "Check that your metrics, expertise, persona, recommendation, and cover outputs exist."
        )

    if args.generate_recommendations:
        if not scopus_db or not scopus_db.exists():
            raise FileNotFoundError(
                "--generate-recommendations was requested, but --scopus-db was missing or not found."
            )

        if args.verbose:
            print("[recs] generating recommendations from Scopus during site build...")

        generated = build_recommendations_from_scopus(
            authors=authors,
            scopus_db=scopus_db,
            model_id=args.recommendation_model_id,
            recommendation_count=args.recommendation_count,
            min_similarity=args.min_similarity,
            batch_size=args.recommendation_batch_size,
            cache_dir=Path(args.recommendation_cache_dir),
            no_cache=args.no_recommendation_cache,
        )

        if args.write_generated_recommendations:
            ensure_dir(recommendations_dir)

        for author in authors:
            recs, fallback = generated.get(
                author.author_label, ([], DEFAULT_RECOMMENDATION_FALLBACK))
            author.recommendations = recs
            author.recommendation_fallback = fallback or DEFAULT_RECOMMENDATION_FALLBACK

            if args.write_generated_recommendations:
                write_recommendations_txt(
                    recommendations_dir / f"{author.author_label}.txt",
                    author=author,
                    recs=recs,
                    fallback_text=author.recommendation_fallback,
                )

    build_site(
        output_dir=output_dir,
        authors=authors,
        cover_src_dir=cover_dir,
        musician_headshot_src_dir=musician_headshots_dir,
        project_title=args.project_title,
        tagline=args.tagline,
        closer=args.closer,
    )

    summary_count = sum(1 for a in authors if a.expertise_summary)
    rec_count = sum(1 for a in authors if a.recommendations)

    print(
        f"Built VALIANT Wrapped site for {len(authors)} authors in: {output_dir}")
    print(f"Homepage: {output_dir / 'index.html'}")
    print(f"Authors with research summaries: {summary_count}/{len(authors)}")
    print(f"Authors with recommendations: {rec_count}/{len(authors)}")


if __name__ == "__main__":
    main()
