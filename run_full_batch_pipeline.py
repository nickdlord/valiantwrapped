#!/usr/bin/env python3
"""
run_full_batch_pipeline.py

Per-author master pipeline for VALIANT Wrapped.

Behavior:
- One CSV per author is the starting unit.
- For each author, run the full pipeline in order:
    1) scopus2txtsummary.py
    2) author_expertise_llama31_2.py
    3) author_persona_llama31.py
    4) generate_album_covers.py
    5) revised GitHub Pages site generator
- Then move to the next author.

Design goals:
- Use subprocesses to call existing stage scripts.
- Support skip-existing/resume at every stage.
- Failure at any stage blocks downstream stages for that author.
- Support optional selected-authors file for reruns.
- Write consolidated reports and stage logs to a separate reports folder.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set


# ----------------------------
# Helpers
# ----------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def canonical_author_label(path_or_label: str) -> str:
    s = str(path_or_label).strip().replace("\\", "/")
    s = os.path.basename(s)
    for ext in (".csv", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".html"):
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
            break
    return s.strip()


def read_selected_authors(path: Path) -> Set[str]:
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = canonical_author_label(line)
        if s:
            out.add(s)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def list_author_csvs(author_csv_dir: Path) -> Dict[str, Path]:
    files = sorted(author_csv_dir.glob("*.csv"))
    return {canonical_author_label(p.name): p for p in files}


def stage_log_paths(reports_dir: Path, stage_name: str, author_label: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_author = canonical_author_label(author_label)
    return (
        reports_dir / "logs" / f"{timestamp}_{safe_author}_{stage_name}.out",
        reports_dir / "logs" / f"{timestamp}_{safe_author}_{stage_name}.err",
    )


def run_subprocess(
    cmd: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
    cwd: Optional[Path] = None,
) -> int:
    ensure_dir(stdout_path.parent)
    ensure_dir(stderr_path.parent)

    with open(stdout_path, "w", encoding="utf-8") as fout, open(stderr_path, "w", encoding="utf-8") as ferr:
        fout.write(f"[START] {now_iso()}\n")
        fout.write("COMMAND:\n")
        fout.write(" ".join(cmd) + "\n\n")
        fout.flush()

        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            stdout=fout,
            stderr=ferr,
            text=True,
        )

        fout.write(f"\n[END] {now_iso()} | returncode={proc.returncode}\n")
        return proc.returncode


# ----------------------------
# Data model
# ----------------------------

STAGES = ["summary_txt", "expertise_txt", "persona_txt", "album_cover", "site"]


@dataclass
class AuthorRun:
    author_label: str
    selected: bool = True
    stage_status: Dict[str, str] = field(
        default_factory=lambda: {s: "not_started" for s in STAGES})
    stage_detail: Dict[str, str] = field(
        default_factory=lambda: {s: "" for s in STAGES})

    def mark(self, stage: str, status: str, detail: str = "") -> None:
        self.stage_status[stage] = status
        self.stage_detail[stage] = detail

    def overall_status(self) -> str:
        for s in STAGES:
            if self.stage_status[s] in {"failed", "blocked"}:
                return "incomplete"
        if all(self.stage_status[s] in {"built", "skipped_existing"} for s in STAGES):
            return "complete"
        return "partial"


# ----------------------------
# Output checks
# ----------------------------

def summary_txt_path(summary_txt_dir: Path, author: str) -> Path:
    return summary_txt_dir / f"{author}.txt"


def expertise_txt_path(expertise_txt_dir: Path, author: str) -> Path:
    return expertise_txt_dir / f"{author}.txt"


def persona_txt_path(persona_txt_dir: Path, author: str) -> Path:
    return persona_txt_dir / f"{author}.txt"


def album_cover_path(album_covers_dir: Path, author: str) -> Path:
    return album_covers_dir / f"{author}.png"


def site_path(docs_dir: Path, author: str) -> Path:
    return docs_dir / "authors" / author / "index.html"


# ----------------------------
# Reporting
# ----------------------------

def write_consolidated_report(path: Path, authors: Dict[str, AuthorRun]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "author_label",
        "selected",
        "overall_status",
    ]
    for stage in STAGES:
        fieldnames.append(f"{stage}_status")
        fieldnames.append(f"{stage}_detail")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for author_label in sorted(authors):
            row = authors[author_label]
            out = {
                "author_label": row.author_label,
                "selected": row.selected,
                "overall_status": row.overall_status(),
            }
            for stage in STAGES:
                out[f"{stage}_status"] = row.stage_status[stage]
                out[f"{stage}_detail"] = row.stage_detail[stage]
            w.writerow(out)


def write_summary_report(path: Path, authors: Dict[str, AuthorRun]) -> None:
    ensure_dir(path.parent)
    total = len(authors)
    complete = sum(1 for a in authors.values()
                   if a.overall_status() == "complete")
    incomplete = total - complete

    lines = [
        f"Run completed: {now_iso()}",
        f"Total authors considered: {total}",
        f"Complete: {complete}",
        f"Incomplete: {incomplete}",
        "",
    ]
    for stage in STAGES:
        counts: Dict[str, int] = {}
        for a in authors.values():
            counts[a.stage_status[stage]] = counts.get(
                a.stage_status[stage], 0) + 1
        lines.append(f"{stage}:")
        for k in sorted(counts):
            lines.append(f"  {k}: {counts[k]}")
        lines.append("")

    write_text(path, "\n".join(lines).rstrip() + "\n")


# ----------------------------
# Per-author stage execution
# ----------------------------

def run_author_pipeline(
    *,
    author: str,
    author_csv_path: Path,
    authors: Dict[str, AuthorRun],
    python_exe: str,
    scopus_script: Path,
    expertise_script: Path,
    persona_script: Path,
    album_script: Path,
    site_script: Path,
    summary_txt_dir: Path,
    expertise_txt_dir: Path,
    persona_txt_dir: Path,
    album_covers_dir: Path,
    docs_dir: Path,
    reports_dir: Path,
    base_url: str,
    skip_existing: bool,
    year_cutoff: int,
    summary_abstract_chars: int,
    expertise_model_id: str,
    expertise_max_input_tokens: int,
    expertise_map_max_new: int,
    expertise_reduce_max_new: int,
    expertise_temperature: float,
    expertise_top_p: float,
    expertise_repetition_penalty: float,
    expertise_abstract_chars: int,
    persona_model_id: str,
    album_llm_model: str,
    album_image_model: str,
) -> None:
    row = authors[author]

    ensure_dir(summary_txt_dir)
    ensure_dir(expertise_txt_dir)
    ensure_dir(persona_txt_dir)
    ensure_dir(album_covers_dir)
    ensure_dir(docs_dir)

    summary_path = summary_txt_path(summary_txt_dir, author)
    expertise_path = expertise_txt_path(expertise_txt_dir, author)
    persona_path = persona_txt_path(persona_txt_dir, author)
    cover_path = album_cover_path(album_covers_dir, author)
    final_site_path = site_path(docs_dir, author)

    # ---------------- Stage 1 ----------------
    if skip_existing and summary_path.exists():
        row.mark("summary_txt", "skipped_existing", str(summary_path))
    else:
        stdout_path, stderr_path = stage_log_paths(
            reports_dir, "stage1_summary_txt", author)
        cmd = [
            python_exe,
            str(scopus_script),
            "--input-file", str(author_csv_path),
            "--output-dir", str(summary_txt_dir),
            "--year-cutoff", str(year_cutoff),
            "--abstract-chars", str(summary_abstract_chars),
        ]
        returncode = run_subprocess(cmd, stdout_path, stderr_path)
        if summary_path.exists():
            row.mark("summary_txt", "built", str(summary_path))
        else:
            row.mark(
                "summary_txt",
                "failed",
                f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}",
            )
            for stage in ["expertise_txt", "persona_txt", "album_cover", "site"]:
                row.mark(stage, "blocked", "upstream failure at summary_txt")
            return

    # ---------------- Stage 2 ----------------
    if skip_existing and expertise_path.exists():
        row.mark("expertise_txt", "skipped_existing", str(expertise_path))
    else:
        stdout_path, stderr_path = stage_log_paths(
            reports_dir, "stage2_expertise_txt", author)
        cmd = [
            python_exe,
            str(expertise_script),
            "--input-file", str(summary_path),
            "--output-txt-dir", str(expertise_txt_dir),
            "--output-csv", str(reports_dir /
                                "author_expertise_summaries.csv"),
            "--model-id", expertise_model_id,
            "--max-input-tokens", str(expertise_max_input_tokens),
            "--map-max-new", str(expertise_map_max_new),
            "--reduce-max-new", str(expertise_reduce_max_new),
            "--temperature", str(expertise_temperature),
            "--top-p", str(expertise_top_p),
            "--repetition-penalty", str(expertise_repetition_penalty),
            "--abstract-chars", str(expertise_abstract_chars),
        ]
        returncode = run_subprocess(cmd, stdout_path, stderr_path)
        if expertise_path.exists():
            row.mark("expertise_txt", "built", str(expertise_path))
        else:
            row.mark(
                "expertise_txt",
                "failed",
                f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}",
            )
            for stage in ["persona_txt", "album_cover", "site"]:
                row.mark(stage, "blocked", "upstream failure at expertise_txt")
            return

    # ---------------- Stage 3 ----------------
    if skip_existing and persona_path.exists():
        row.mark("persona_txt", "skipped_existing", str(persona_path))
    else:
        stdout_path, stderr_path = stage_log_paths(
            reports_dir, "stage3_persona_txt", author)
        cmd = [
            python_exe,
            str(persona_script),
            "--input-file", str(expertise_path),
            "--output-dir", str(persona_txt_dir),
            "--output-csv", str(reports_dir / "author_music_personas.csv"),
            "--model-id", persona_model_id,
        ]
        returncode = run_subprocess(cmd, stdout_path, stderr_path)
        if persona_path.exists():
            row.mark("persona_txt", "built", str(persona_path))
        else:
            row.mark(
                "persona_txt",
                "failed",
                f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}",
            )
            for stage in ["album_cover", "site"]:
                row.mark(stage, "blocked", "upstream failure at persona_txt")
            return

    # ---------------- Stage 4 ----------------
    if skip_existing and cover_path.exists():
        row.mark("album_cover", "skipped_existing", str(cover_path))
    else:
        stdout_path, stderr_path = stage_log_paths(
            reports_dir, "stage4_album_covers", author)
        cmd = [
            python_exe,
            str(album_script),
            "--input-file", str(persona_path),
            "--output-dir", str(album_covers_dir),
            "--llm-model", album_llm_model,
            "--image-model", album_image_model,
        ]
        returncode = run_subprocess(cmd, stdout_path, stderr_path)
        if cover_path.exists():
            row.mark("album_cover", "built", str(cover_path))
        else:
            row.mark(
                "album_cover",
                "failed",
                f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}",
            )
            row.mark("site", "blocked", "upstream failure at album_cover")
            return

    # ---------------- Stage 5 ----------------
    if skip_existing and final_site_path.exists():
        row.mark("site", "skipped_existing", str(final_site_path))
    else:
        stdout_path, stderr_path = stage_log_paths(
            reports_dir, "stage5_site", author)
        cmd = [
            python_exe,
            str(site_script),
            "--summary-dir", str(expertise_txt_dir),
            "--persona-dir", str(persona_txt_dir),
            "--author-label", author,
            "--album-covers-dir", str(album_covers_dir),
            "--docs-dir", str(docs_dir),
            "--base-url", base_url,
            "--skip-existing",
        ]
        returncode = run_subprocess(cmd, stdout_path, stderr_path)
        if final_site_path.exists():
            row.mark("site", "built", str(final_site_path))
        else:
            row.mark(
                "site",
                "failed",
                f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}",
            )


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--python-exe", default=sys.executable,
                    help="Python interpreter for subprocesses")

    ap.add_argument("--author-csv-dir", default="author_csvs")
    ap.add_argument("--summary-txt-dir", default="outputs/summary_txt")
    ap.add_argument("--expertise-txt-dir",
                    default="outputs/author_expertise_txt")
    ap.add_argument("--persona-txt-dir",
                    default="outputs/author_music_personas_txt")
    ap.add_argument("--album-covers-dir", default="outputs/album_covers")
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--reports-dir", default="pipeline_reports")
    ap.add_argument("--base-url", required=True)

    ap.add_argument("--selected-authors-file", default="",
                    help="Optional text file with one author label per line")
    ap.add_argument("--skip-existing", action="store_true")

    ap.add_argument("--scopus-script", default="scopus2txtsummary.py")
    ap.add_argument("--expertise-script",
                    default="author_expertise_llama31_2.py")
    ap.add_argument("--persona-script", default="author_persona_llama31.py")
    ap.add_argument("--album-script", default="generate_album_covers.py")
    ap.add_argument("--site-script", default="generate_site_github.py")

    ap.add_argument("--year-cutoff", type=int, default=2025)
    ap.add_argument("--summary-abstract-chars", type=int, default=400)

    ap.add_argument("--expertise-model-id",
                    default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--expertise-max-input-tokens", type=int, default=6000)
    ap.add_argument("--expertise-map-max-new", type=int, default=200)
    ap.add_argument("--expertise-reduce-max-new", type=int, default=512)
    ap.add_argument("--expertise-temperature", type=float, default=0.25)
    ap.add_argument("--expertise-top-p", type=float, default=0.9)
    ap.add_argument("--expertise-repetition-penalty", type=float, default=1.1)
    ap.add_argument("--expertise-abstract-chars", type=int, default=260)

    ap.add_argument("--persona-model-id",
                    default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    ap.add_argument("--album-llm-model",
                    default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--album-image-model",
                    default="black-forest-labs/FLUX.1-dev")

    return ap


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = build_parser().parse_args()

    author_csv_dir = Path(args.author_csv_dir).resolve()
    summary_txt_dir = Path(args.summary_txt_dir).resolve()
    expertise_txt_dir = Path(args.expertise_txt_dir).resolve()
    persona_txt_dir = Path(args.persona_txt_dir).resolve()
    album_covers_dir = Path(args.album_covers_dir).resolve()
    docs_dir = Path(args.docs_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()

    scopus_script = Path(args.scopus_script).resolve()
    expertise_script = Path(args.expertise_script).resolve()
    persona_script = Path(args.persona_script).resolve()
    album_script = Path(args.album_script).resolve()
    site_script = Path(args.site_script).resolve()

    ensure_dir(reports_dir / "logs")

    source_csvs = list_author_csvs(author_csv_dir)
    if not source_csvs:
        raise FileNotFoundError(f"No author CSVs found in: {author_csv_dir}")

    selected = sorted(source_csvs.keys())
    if args.selected_authors_file:
        wanted = read_selected_authors(
            Path(args.selected_authors_file).resolve())
        selected = [a for a in selected if a in wanted]

    if not selected:
        raise ValueError("No authors selected for this run.")

    authors: Dict[str, AuthorRun] = {
        author: AuthorRun(author_label=author, selected=True)
        for author in selected
    }

    run_meta = [
        f"run_started: {now_iso()}",
        f"python_exe: {args.python_exe}",
        f"author_csv_dir: {author_csv_dir}",
        f"selected_authors_count: {len(selected)}",
        f"skip_existing: {args.skip_existing}",
        f"base_url: {args.base_url}",
        f"scopus_script: {scopus_script}",
        f"expertise_script: {expertise_script}",
        f"persona_script: {persona_script}",
        f"album_script: {album_script}",
        f"site_script: {site_script}",
        f"mode: per_author_full_pipeline",
    ]
    write_text(reports_dir / "run_metadata.txt", "\n".join(run_meta) + "\n")

    for idx, author in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] Processing {author}")
        run_author_pipeline(
            author=author,
            author_csv_path=source_csvs[author],
            authors=authors,
            python_exe=args.python_exe,
            scopus_script=scopus_script,
            expertise_script=expertise_script,
            persona_script=persona_script,
            album_script=album_script,
            site_script=site_script,
            summary_txt_dir=summary_txt_dir,
            expertise_txt_dir=expertise_txt_dir,
            persona_txt_dir=persona_txt_dir,
            album_covers_dir=album_covers_dir,
            docs_dir=docs_dir,
            reports_dir=reports_dir,
            base_url=args.base_url,
            skip_existing=args.skip_existing,
            year_cutoff=args.year_cutoff,
            summary_abstract_chars=args.summary_abstract_chars,
            expertise_model_id=args.expertise_model_id,
            expertise_max_input_tokens=args.expertise_max_input_tokens,
            expertise_map_max_new=args.expertise_map_max_new,
            expertise_reduce_max_new=args.expertise_reduce_max_new,
            expertise_temperature=args.expertise_temperature,
            expertise_top_p=args.expertise_top_p,
            expertise_repetition_penalty=args.expertise_repetition_penalty,
            expertise_abstract_chars=args.expertise_abstract_chars,
            persona_model_id=args.persona_model_id,
            album_llm_model=args.album_llm_model,
            album_image_model=args.album_image_model,
        )
        write_consolidated_report(
            reports_dir / "pipeline_author_report.csv", authors)

    write_summary_report(reports_dir / "pipeline_summary.txt", authors)

    print("Per-author master batch pipeline complete.")
    print(f"Author report: {reports_dir / 'pipeline_author_report.csv'}")
    print(f"Summary report: {reports_dir / 'pipeline_summary.txt'}")
    print(f"Logs: {reports_dir / 'logs'}")


if __name__ == "__main__":
    main()
