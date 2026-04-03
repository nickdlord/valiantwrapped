#!/usr/bin/env python3
"""
run_full_batch_pipeline.py

Master batch pipeline for VALIANT Wrapped.

Design goals:
- One CSV per author is the starting unit.
- Run stages in order across the eligible author set.
- Use subprocesses to call existing stage scripts.
- Support skip-existing/resume at every stage.
- Failure at any stage blocks downstream stages for that author.
- Support optional selected-authors file for reruns.
- Write consolidated reports and stage logs to a separate reports folder.
- Default to one sequential ACCRE GPU job.

Expected stages:
1) scopus2txtsummary.py
2) author_expertise_llama31_2.py
3) author_persona_llama31.py
4) generate_album_covers.py
5) revised GitHub Pages site generator

Notes:
- Stage 2/3/4 are run in batch mode against temporary input directories that contain
  only currently eligible authors. This preserves stage-by-stage batching while still
  allowing author-level blocking and reruns.
- Stage 5 is run per author because the revised site generator supports direct
  author-label builds safely. Do NOT use the older site generator that wipes docs/.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
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


def safe_link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def list_author_csvs(author_csv_dir: Path) -> Dict[str, Path]:
    files = sorted(author_csv_dir.glob("*.csv"))
    return {canonical_author_label(p.name): p for p in files}


def stage_log_paths(reports_dir: Path, stage_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        reports_dir / "logs" / f"{timestamp}_{stage_name}.out",
        reports_dir / "logs" / f"{timestamp}_{stage_name}.err",
    )


def run_subprocess(cmd: Sequence[str], stdout_path: Path, stderr_path: Path, cwd: Optional[Path] = None) -> int:
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


def write_author_labels_file(path: Path, labels: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")


# ----------------------------
# Data model
# ----------------------------

STAGES = ["summary_txt", "expertise_txt", "persona_txt", "album_cover", "site"]

@dataclass
class AuthorRun:
    author_label: str
    selected: bool = True
    stage_status: Dict[str, str] = field(default_factory=lambda: {s: "not_started" for s in STAGES})
    stage_detail: Dict[str, str] = field(default_factory=lambda: {s: "" for s in STAGES})

    def mark(self, stage: str, status: str, detail: str = "") -> None:
        self.stage_status[stage] = status
        self.stage_detail[stage] = detail

    def succeeded_through(self, stage: str) -> bool:
        status = self.stage_status.get(stage, "")
        return status in {"built", "skipped_existing"}

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
# Temp input preparation
# ----------------------------

def make_temp_inputs(stage_root: Path, stage_name: str, authors: Sequence[str], source_lookup: Dict[str, Path]) -> Path:
    temp_dir = stage_root / stage_name
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for author in authors:
        src = source_lookup[author]
        dst = temp_dir / src.name
        safe_link_or_copy(src, dst)

    return temp_dir


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
    complete = sum(1 for a in authors.values() if a.overall_status() == "complete")
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
            counts[a.stage_status[stage]] = counts.get(a.stage_status[stage], 0) + 1
        lines.append(f"{stage}:")
        for k in sorted(counts):
            lines.append(f"  {k}: {counts[k]}")
        lines.append("")

    write_text(path, "\n".join(lines).rstrip() + "\n")


# ----------------------------
# Stage runners
# ----------------------------

def run_stage_1_summary(
    python_exe: str,
    script_path: Path,
    author_csv_dir: Path,
    summary_txt_dir: Path,
    reports_dir: Path,
    temp_root: Path,
    authors: Dict[str, AuthorRun],
    selected_authors: List[str],
    year_cutoff: int,
    abstract_chars: int,
    skip_existing: bool,
) -> None:
    ensure_dir(summary_txt_dir)

    source_lookup = list_author_csvs(author_csv_dir)

    to_build: List[str] = []
    for author in selected_authors:
        out_path = summary_txt_path(summary_txt_dir, author)
        if skip_existing and out_path.exists():
            authors[author].mark("summary_txt", "skipped_existing", str(out_path))
        else:
            if author not in source_lookup:
                authors[author].mark("summary_txt", "failed", f"missing_source_csv in {author_csv_dir}")
            else:
                to_build.append(author)

    if not to_build:
        return

    temp_input_dir = make_temp_inputs(temp_root, "summary_inputs", to_build, source_lookup)
    stdout_path, stderr_path = stage_log_paths(reports_dir, "stage1_summary_txt")

    cmd = [
        python_exe,
        str(script_path),
        "--input-dir", str(temp_input_dir),
        "--output-dir", str(summary_txt_dir),
        "--year-cutoff", str(year_cutoff),
        "--abstract-chars", str(abstract_chars),
    ]
    returncode = run_subprocess(cmd, stdout_path, stderr_path)

    for author in to_build:
        out_path = summary_txt_path(summary_txt_dir, author)
        if out_path.exists():
            authors[author].mark("summary_txt", "built", str(out_path))
        else:
            detail = f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}"
            authors[author].mark("summary_txt", "failed", detail)


def run_stage_2_expertise(
    python_exe: str,
    script_path: Path,
    summary_txt_dir: Path,
    expertise_txt_dir: Path,
    expertise_csv_path: Path,
    reports_dir: Path,
    temp_root: Path,
    authors: Dict[str, AuthorRun],
    max_input_tokens: int,
    map_max_new: int,
    reduce_max_new: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    abstract_chars: int,
    model_id: str,
    skip_existing: bool,
) -> None:
    ensure_dir(expertise_txt_dir)
    ensure_dir(expertise_csv_path.parent)

    eligible = [a for a, r in authors.items() if r.succeeded_through("summary_txt")]
    source_lookup = {a: summary_txt_path(summary_txt_dir, a) for a in eligible if summary_txt_path(summary_txt_dir, a).exists()}

    to_build: List[str] = []
    for author in eligible:
        out_path = expertise_txt_path(expertise_txt_dir, author)
        if skip_existing and out_path.exists():
            authors[author].mark("expertise_txt", "skipped_existing", str(out_path))
        else:
            if author not in source_lookup:
                authors[author].mark("expertise_txt", "failed", f"missing_summary_txt in {summary_txt_dir}")
            else:
                to_build.append(author)

    blocked = [a for a, r in authors.items() if not r.succeeded_through("summary_txt")]
    for author in blocked:
        if authors[author].stage_status["expertise_txt"] == "not_started":
            authors[author].mark("expertise_txt", "blocked", "upstream failure at summary_txt")

    if not to_build:
        return

    temp_input_dir = make_temp_inputs(temp_root, "expertise_inputs", to_build, source_lookup)
    stdout_path, stderr_path = stage_log_paths(reports_dir, "stage2_expertise_txt")

    cmd = [
        python_exe,
        str(script_path),
        "--input-dir", str(temp_input_dir),
        "--output-txt-dir", str(expertise_txt_dir),
        "--output-csv", str(expertise_csv_path),
        "--model-id", model_id,
        "--max-input-tokens", str(max_input_tokens),
        "--map-max-new", str(map_max_new),
        "--reduce-max-new", str(reduce_max_new),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--repetition-penalty", str(repetition_penalty),
        "--abstract-chars", str(abstract_chars),
    ]
    returncode = run_subprocess(cmd, stdout_path, stderr_path)

    for author in to_build:
        out_path = expertise_txt_path(expertise_txt_dir, author)
        if out_path.exists():
            authors[author].mark("expertise_txt", "built", str(out_path))
        else:
            detail = f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}"
            authors[author].mark("expertise_txt", "failed", detail)


def run_stage_3_persona(
    python_exe: str,
    script_path: Path,
    expertise_txt_dir: Path,
    persona_txt_dir: Path,
    persona_csv_path: Path,
    reports_dir: Path,
    temp_root: Path,
    authors: Dict[str, AuthorRun],
    model_id: str,
    skip_existing: bool,
) -> None:
    ensure_dir(persona_txt_dir)
    ensure_dir(persona_csv_path.parent)

    eligible = [a for a, r in authors.items() if r.succeeded_through("expertise_txt")]
    source_lookup = {a: expertise_txt_path(expertise_txt_dir, a) for a in eligible if expertise_txt_path(expertise_txt_dir, a).exists()}

    to_build: List[str] = []
    for author in eligible:
        out_path = persona_txt_path(persona_txt_dir, author)
        if skip_existing and out_path.exists():
            authors[author].mark("persona_txt", "skipped_existing", str(out_path))
        else:
            if author not in source_lookup:
                authors[author].mark("persona_txt", "failed", f"missing_expertise_txt in {expertise_txt_dir}")
            else:
                to_build.append(author)

    blocked = [a for a, r in authors.items() if not r.succeeded_through("expertise_txt")]
    for author in blocked:
        if authors[author].stage_status["persona_txt"] == "not_started":
            authors[author].mark("persona_txt", "blocked", "upstream failure at expertise_txt")

    if not to_build:
        return

    temp_input_dir = make_temp_inputs(temp_root, "persona_inputs", to_build, source_lookup)
    stdout_path, stderr_path = stage_log_paths(reports_dir, "stage3_persona_txt")

    cmd = [
        python_exe,
        str(script_path),
        "--input-dir", str(temp_input_dir),
        "--output-dir", str(persona_txt_dir),
        "--output-csv", str(persona_csv_path),
        "--model-id", model_id,
    ]
    returncode = run_subprocess(cmd, stdout_path, stderr_path)

    for author in to_build:
        out_path = persona_txt_path(persona_txt_dir, author)
        if out_path.exists():
            authors[author].mark("persona_txt", "built", str(out_path))
        else:
            detail = f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}"
            authors[author].mark("persona_txt", "failed", detail)


def run_stage_4_album_covers(
    python_exe: str,
    script_path: Path,
    persona_txt_dir: Path,
    album_covers_dir: Path,
    reports_dir: Path,
    temp_root: Path,
    authors: Dict[str, AuthorRun],
    llm_model: str,
    image_model: str,
    skip_existing: bool,
) -> None:
    ensure_dir(album_covers_dir)

    eligible = [a for a, r in authors.items() if r.succeeded_through("persona_txt")]
    source_lookup = {a: persona_txt_path(persona_txt_dir, a) for a in eligible if persona_txt_path(persona_txt_dir, a).exists()}

    to_build: List[str] = []
    for author in eligible:
        out_path = album_cover_path(album_covers_dir, author)
        if skip_existing and out_path.exists():
            authors[author].mark("album_cover", "skipped_existing", str(out_path))
        else:
            if author not in source_lookup:
                authors[author].mark("album_cover", "failed", f"missing_persona_txt in {persona_txt_dir}")
            else:
                to_build.append(author)

    blocked = [a for a, r in authors.items() if not r.succeeded_through("persona_txt")]
    for author in blocked:
        if authors[author].stage_status["album_cover"] == "not_started":
            authors[author].mark("album_cover", "blocked", "upstream failure at persona_txt")

    if not to_build:
        return

    temp_input_dir = make_temp_inputs(temp_root, "album_cover_inputs", to_build, source_lookup)
    stdout_path, stderr_path = stage_log_paths(reports_dir, "stage4_album_covers")

    cmd = [
        python_exe,
        str(script_path),
        "--input-dir", str(temp_input_dir),
        "--output-dir", str(album_covers_dir),
        "--llm-model", llm_model,
        "--image-model", image_model,
    ]
    returncode = run_subprocess(cmd, stdout_path, stderr_path)

    for author in to_build:
        out_path = album_cover_path(album_covers_dir, author)
        if out_path.exists():
            authors[author].mark("album_cover", "built", str(out_path))
        else:
            detail = f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}"
            authors[author].mark("album_cover", "failed", detail)


def run_stage_5_site(
    python_exe: str,
    site_script: Path,
    expertise_txt_dir: Path,
    persona_txt_dir: Path,
    album_covers_dir: Path,
    docs_dir: Path,
    reports_dir: Path,
    authors: Dict[str, AuthorRun],
    base_url: str,
    skip_existing: bool,
) -> None:
    ensure_dir(docs_dir)

    eligible = [a for a, r in authors.items() if r.succeeded_through("album_cover")]

    blocked = [a for a, r in authors.items() if not r.succeeded_through("album_cover")]
    for author in blocked:
        if authors[author].stage_status["site"] == "not_started":
            authors[author].mark("site", "blocked", "upstream failure at album_cover")

    for author in eligible:
        out_path = site_path(docs_dir, author)
        if skip_existing and out_path.exists():
            authors[author].mark("site", "skipped_existing", str(out_path))
            continue

        summary_path = expertise_txt_path(expertise_txt_dir, author)
        persona_path = persona_txt_path(persona_txt_dir, author)
        cover_path = album_cover_path(album_covers_dir, author)

        if not summary_path.exists():
            authors[author].mark("site", "failed", f"missing_expertise_txt in {expertise_txt_dir}")
            continue
        if not persona_path.exists():
            authors[author].mark("site", "failed", f"missing_persona_txt in {persona_txt_dir}")
            continue
        if not cover_path.exists():
            authors[author].mark("site", "failed", f"missing_album_cover in {album_covers_dir}")
            continue

        stdout_path, stderr_path = stage_log_paths(reports_dir, f"stage5_site_{author}")
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
        if out_path.exists():
            authors[author].mark("site", "built", str(out_path))
        else:
            detail = f"missing_expected_output after returncode={returncode}; see logs {stdout_path.name}, {stderr_path.name}"
            authors[author].mark("site", "failed", detail)


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--python-exe", default=sys.executable, help="Python interpreter for subprocesses")

    ap.add_argument("--author-csv-dir", default="author_csvs")
    ap.add_argument("--summary-txt-dir", default="outputs/summary_txt")
    ap.add_argument("--expertise-txt-dir", default="outputs/author_expertise_txt")
    ap.add_argument("--persona-txt-dir", default="outputs/author_music_personas_txt")
    ap.add_argument("--album-covers-dir", default="outputs/album_covers")
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--reports-dir", default="pipeline_reports")
    ap.add_argument("--base-url", required=True)

    ap.add_argument("--selected-authors-file", default="", help="Optional text file with one author label per line")
    ap.add_argument("--skip-existing", action="store_true")

    ap.add_argument("--scopus-script", default="scopus2txtsummary.py")
    ap.add_argument("--expertise-script", default="author_expertise_llama31_2.py")
    ap.add_argument("--persona-script", default="author_persona_llama31.py")
    ap.add_argument("--album-script", default="generate_album_covers.py")
    ap.add_argument("--site-script", default="generate_site_github.py",
                    help="Path to revised per-author GitHub Pages site generator")

    ap.add_argument("--year-cutoff", type=int, default=2025)
    ap.add_argument("--summary-abstract-chars", type=int, default=400)

    ap.add_argument("--expertise-model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--expertise-max-input-tokens", type=int, default=6000)
    ap.add_argument("--expertise-map-max-new", type=int, default=200)
    ap.add_argument("--expertise-reduce-max-new", type=int, default=512)
    ap.add_argument("--expertise-temperature", type=float, default=0.25)
    ap.add_argument("--expertise-top-p", type=float, default=0.9)
    ap.add_argument("--expertise-repetition-penalty", type=float, default=1.1)
    ap.add_argument("--expertise-abstract-chars", type=int, default=260)

    ap.add_argument("--persona-model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    ap.add_argument("--album-llm-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--album-image-model", default="black-forest-labs/FLUX.1-dev")

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
    temp_root = reports_dir / "_temp_stage_inputs"
    ensure_dir(temp_root)

    source_csvs = list_author_csvs(author_csv_dir)
    if not source_csvs:
        raise FileNotFoundError(f"No author CSVs found in: {author_csv_dir}")

    selected = sorted(source_csvs.keys())
    if args.selected_authors_file:
        wanted = read_selected_authors(Path(args.selected_authors_file).resolve())
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
    ]
    write_text(reports_dir / "run_metadata.txt", "\n".join(run_meta) + "\n")

    # Stage 1
    run_stage_1_summary(
        python_exe=args.python_exe,
        script_path=scopus_script,
        author_csv_dir=author_csv_dir,
        summary_txt_dir=summary_txt_dir,
        reports_dir=reports_dir,
        temp_root=temp_root,
        authors=authors,
        selected_authors=selected,
        year_cutoff=args.year_cutoff,
        abstract_chars=args.summary_abstract_chars,
        skip_existing=args.skip_existing,
    )
    write_consolidated_report(reports_dir / "pipeline_author_report.csv", authors)

    # Stage 2
    run_stage_2_expertise(
        python_exe=args.python_exe,
        script_path=expertise_script,
        summary_txt_dir=summary_txt_dir,
        expertise_txt_dir=expertise_txt_dir,
        expertise_csv_path=reports_dir / "author_expertise_summaries.csv",
        reports_dir=reports_dir,
        temp_root=temp_root,
        authors=authors,
        max_input_tokens=args.expertise_max_input_tokens,
        map_max_new=args.expertise_map_max_new,
        reduce_max_new=args.expertise_reduce_max_new,
        temperature=args.expertise_temperature,
        top_p=args.expertise_top_p,
        repetition_penalty=args.expertise_repetition_penalty,
        abstract_chars=args.expertise_abstract_chars,
        model_id=args.expertise_model_id,
        skip_existing=args.skip_existing,
    )
    write_consolidated_report(reports_dir / "pipeline_author_report.csv", authors)

    # Stage 3
    run_stage_3_persona(
        python_exe=args.python_exe,
        script_path=persona_script,
        expertise_txt_dir=expertise_txt_dir,
        persona_txt_dir=persona_txt_dir,
        persona_csv_path=reports_dir / "author_music_personas.csv",
        reports_dir=reports_dir,
        temp_root=temp_root,
        authors=authors,
        model_id=args.persona_model_id,
        skip_existing=args.skip_existing,
    )
    write_consolidated_report(reports_dir / "pipeline_author_report.csv", authors)

    # Stage 4
    run_stage_4_album_covers(
        python_exe=args.python_exe,
        script_path=album_script,
        persona_txt_dir=persona_txt_dir,
        album_covers_dir=album_covers_dir,
        reports_dir=reports_dir,
        temp_root=temp_root,
        authors=authors,
        llm_model=args.album_llm_model,
        image_model=args.album_image_model,
        skip_existing=args.skip_existing,
    )
    write_consolidated_report(reports_dir / "pipeline_author_report.csv", authors)

    # Stage 5
    run_stage_5_site(
        python_exe=args.python_exe,
        site_script=site_script,
        expertise_txt_dir=expertise_txt_dir,
        persona_txt_dir=persona_txt_dir,
        album_covers_dir=album_covers_dir,
        docs_dir=docs_dir,
        reports_dir=reports_dir,
        authors=authors,
        base_url=args.base_url,
        skip_existing=args.skip_existing,
    )
    write_consolidated_report(reports_dir / "pipeline_author_report.csv", authors)

    write_summary_report(reports_dir / "pipeline_summary.txt", authors)

    print("Master batch pipeline complete.")
    print(f"Author report: {reports_dir / 'pipeline_author_report.csv'}")
    print(f"Summary report: {reports_dir / 'pipeline_summary.txt'}")
    print(f"Logs: {reports_dir / 'logs'}")


if __name__ == "__main__":
    main()
