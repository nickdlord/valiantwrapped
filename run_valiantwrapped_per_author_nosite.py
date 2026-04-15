#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

STAGES = ["summary_txt", "expertise_txt", "persona_txt", "album_cover"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def canonical_author_label(value: str) -> str:
    s = str(value).strip().replace("\\", "/")
    s = os.path.basename(s)
    for ext in (".csv", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".html"):
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
            break
    return s.strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_selected_authors(path: Path) -> Set[str]:
    labels: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        label = canonical_author_label(line)
        if label:
            labels.add(label)
    return labels


def list_author_csvs(author_csv_dir: Path) -> Dict[str, Path]:
    return {canonical_author_label(p.name): p for p in sorted(author_csv_dir.glob("*.csv"))}


def stage_log_paths(reports_dir: Path, author_label: str, stage_name: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_author = canonical_author_label(author_label) if author_label else "__global__"
    return (
        reports_dir / "logs" / f"{stamp}_{safe_author}_{stage_name}.out",
        reports_dir / "logs" / f"{stamp}_{safe_author}_{stage_name}.err",
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


def summary_txt_path(summary_txt_dir: Path, author: str) -> Path:
    return summary_txt_dir / f"{author}.txt"


def expertise_txt_path(expertise_txt_dir: Path, author: str) -> Path:
    return expertise_txt_dir / f"{author}.txt"


def persona_txt_path(persona_txt_dir: Path, author: str) -> Path:
    return persona_txt_dir / f"{author}.txt"


def album_cover_path(album_covers_dir: Path, author: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = album_covers_dir / f"{author}{ext}"
        if p.exists():
            return p
    return album_covers_dir / f"{author}.png"


@dataclass
class AuthorRun:
    author_label: str
    stage_status: Dict[str, str] = field(default_factory=lambda: {s: "not_started" for s in STAGES})
    stage_detail: Dict[str, str] = field(default_factory=lambda: {s: "" for s in STAGES})

    def mark(self, stage: str, status: str, detail: str = "") -> None:
        self.stage_status[stage] = status
        self.stage_detail[stage] = detail

    def overall_status(self) -> str:
        for stage in STAGES:
            if self.stage_status[stage] in {"failed", "blocked"}:
                return "incomplete"
        if all(self.stage_status[s] in {"built", "skipped_existing"} for s in STAGES):
            return "complete"
        return "partial"


def write_consolidated_report(path: Path, authors: Dict[str, AuthorRun]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["author_label", "overall_status"]
    for stage in STAGES:
        fieldnames.extend([f"{stage}_status", f"{stage}_detail"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label in sorted(authors):
            row = authors[label]
            out = {"author_label": label, "overall_status": row.overall_status()}
            for stage in STAGES:
                out[f"{stage}_status"] = row.stage_status[stage]
                out[f"{stage}_detail"] = row.stage_detail[stage]
            w.writerow(out)


def run_global_metrics(
    python_exe: str,
    metrics_script: Path,
    project_root: Path,
    reports_dir: Path,
    cached_copy: Path,
    skip_existing: bool,
) -> Path:
    if skip_existing and cached_copy.exists():
        return cached_copy

    stdout_path, stderr_path = stage_log_paths(reports_dir, "__global__", "author_scopusmetrics")
    returncode = run_subprocess([python_exe, str(metrics_script)], stdout_path, stderr_path, cwd=project_root)
    generated = project_root / "author_summary_2025_present.csv"
    if returncode != 0 or not generated.exists():
        raise RuntimeError(
            f"author_scopusmetrics.py failed. See {stdout_path.name} and {stderr_path.name}."
        )
    shutil.copy2(generated, cached_copy)
    return cached_copy


def run_per_author_pipeline(
    author: str,
    csv_path: Path,
    authors: Dict[str, AuthorRun],
    python_exe: str,
    project_root: Path,
    reports_dir: Path,
    summary_txt_dir: Path,
    expertise_txt_dir: Path,
    persona_txt_dir: Path,
    album_covers_dir: Path,
    scopus2txt_script: Path,
    expertise_script: Path,
    persona_script: Path,
    album_script: Path,
    skip_existing: bool,
) -> None:
    row = authors[author]

    summary_path = summary_txt_path(summary_txt_dir, author)
    expertise_path = expertise_txt_path(expertise_txt_dir, author)
    persona_path = persona_txt_path(persona_txt_dir, author)
    cover_path = album_cover_path(album_covers_dir, author)

    # 1) scopus2txtsummary
    if skip_existing and summary_path.exists():
        row.mark("summary_txt", "skipped_existing", str(summary_path))
    else:
        stdout_path, stderr_path = stage_log_paths(reports_dir, author, "summary_txt")
        cmd = [python_exe, str(scopus2txt_script), "--input-file", str(csv_path), "--output-dir", str(summary_txt_dir)]
        rc = run_subprocess(cmd, stdout_path, stderr_path, cwd=project_root)
        if rc == 0 and summary_path.exists():
            row.mark("summary_txt", "built", str(summary_path))
        else:
            row.mark("summary_txt", "failed", f"See {stdout_path.name} / {stderr_path.name}")
            for s in ["expertise_txt", "persona_txt", "album_cover"]:
                row.mark(s, "blocked", "upstream failure at summary_txt")
            return

    # 2) expertise
    if skip_existing and expertise_path.exists():
        row.mark("expertise_txt", "skipped_existing", str(expertise_path))
    else:
        per_author_csv = reports_dir / "per_author_csv" / "expertise" / f"{author}.csv"
        ensure_dir(per_author_csv.parent)
        stdout_path, stderr_path = stage_log_paths(reports_dir, author, "expertise_txt")
        cmd = [
            python_exe,
            str(expertise_script),
            "--input-file", str(summary_path),
            "--output-txt-dir", str(expertise_txt_dir),
            "--output-csv", str(per_author_csv),
        ]
        rc = run_subprocess(cmd, stdout_path, stderr_path, cwd=project_root)
        if rc == 0 and expertise_path.exists():
            row.mark("expertise_txt", "built", str(expertise_path))
        else:
            row.mark("expertise_txt", "failed", f"See {stdout_path.name} / {stderr_path.name}")
            for s in ["persona_txt", "album_cover"]:
                row.mark(s, "blocked", "upstream failure at expertise_txt")
            return

    # 3) persona
    if skip_existing and persona_path.exists():
        row.mark("persona_txt", "skipped_existing", str(persona_path))
    else:
        per_author_csv = reports_dir / "per_author_csv" / "persona" / f"{author}.csv"
        ensure_dir(per_author_csv.parent)
        stdout_path, stderr_path = stage_log_paths(reports_dir, author, "persona_txt")
        cmd = [
            python_exe,
            str(persona_script),
            "--input-file", str(expertise_path),
            "--output-dir", str(persona_txt_dir),
            "--output-csv", str(per_author_csv),
        ]
        rc = run_subprocess(cmd, stdout_path, stderr_path, cwd=project_root)
        if rc == 0 and persona_path.exists():
            row.mark("persona_txt", "built", str(persona_path))
        else:
            row.mark("persona_txt", "failed", f"See {stdout_path.name} / {stderr_path.name}")
            row.mark("album_cover", "blocked", "upstream failure at persona_txt")
            return

    # 4) album cover
    if skip_existing and cover_path.exists():
        row.mark("album_cover", "skipped_existing", str(cover_path))
    else:
        stdout_path, stderr_path = stage_log_paths(reports_dir, author, "album_cover")
        cmd = [
            python_exe,
            str(album_script),
            "--input-file", str(persona_path),
            "--output-dir", str(album_covers_dir),
        ]
        rc = run_subprocess(cmd, stdout_path, stderr_path, cwd=project_root)
        cover_path = album_cover_path(album_covers_dir, author)
        if rc == 0 and cover_path.exists():
            row.mark("album_cover", "built", str(cover_path))
        else:
            row.mark("album_cover", "failed", f"See {stdout_path.name} / {stderr_path.name}")
            return


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--author-csv-dir", default="author_csvs")
    ap.add_argument("--summary-txt-dir", default="outputs/summary_txt")
    ap.add_argument("--expertise-txt-dir", default="outputs/author_expertise_txt")
    ap.add_argument("--persona-txt-dir", default="outputs/author_music_personas_txt")
    ap.add_argument("--album-covers-dir", default="outputs/album_covers")
    ap.add_argument("--reports-dir", default="pipeline_reports")
    ap.add_argument("--selected-authors-file", default="")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--scopus-db", default="")
    ap.add_argument("--scopus2txt-script", default="scopus2txtsummary.py")
    ap.add_argument("--metrics-script", default="author_scopusmetrics.py")
    ap.add_argument("--expertise-script", default="author_expertise_llama31_2.py")
    ap.add_argument("--persona-script", default="author_persona_llama31.py")
    ap.add_argument("--album-script", default="generate_album_covers.py")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(args.project_root).resolve()
    author_csv_dir = (project_root / args.author_csv_dir).resolve()
    summary_txt_dir = (project_root / args.summary_txt_dir).resolve()
    expertise_txt_dir = (project_root / args.expertise_txt_dir).resolve()
    persona_txt_dir = (project_root / args.persona_txt_dir).resolve()
    album_covers_dir = (project_root / args.album_covers_dir).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()

    scopus2txt_script = (project_root / args.scopus2txt_script).resolve()
    metrics_script = (project_root / args.metrics_script).resolve()
    expertise_script = (project_root / args.expertise_script).resolve()
    persona_script = (project_root / args.persona_script).resolve()
    album_script = (project_root / args.album_script).resolve()

    ensure_dir(reports_dir / "logs")
    ensure_dir(summary_txt_dir)
    ensure_dir(expertise_txt_dir)
    ensure_dir(persona_txt_dir)
    ensure_dir(album_covers_dir)

    source_csvs = list_author_csvs(author_csv_dir)
    if not source_csvs:
        raise FileNotFoundError(f"No CSV files found in {author_csv_dir}")

    selected = sorted(source_csvs.keys())
    if args.selected_authors_file:
        selected_set = read_selected_authors((project_root / args.selected_authors_file).resolve())
        selected = [a for a in selected if a in selected_set]
    if not selected:
        raise ValueError("No authors selected for this run.")

    authors: Dict[str, AuthorRun] = {author: AuthorRun(author) for author in selected}

    meta_lines = [
        f"run_started: {now_iso()}",
        f"project_root: {project_root}",
        f"selected_authors_count: {len(selected)}",
        f"skip_existing: {args.skip_existing}",
        f"scopus_db: {args.scopus_db or '(not used by this runner)'}",
        "mode: per_author_content_generation_no_site_build",
        f"outputs_ready_for_site_generation: author_summary_2025_present.csv, {expertise_txt_dir}, {persona_txt_dir}, {album_covers_dir}",
    ]
    (reports_dir / "run_metadata.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    # global metrics stage (kept because the site generator uses this later)
    cached_metrics_csv = reports_dir / "author_summary_2025_present.csv"
    metrics_ok = True
    try:
        run_global_metrics(
            python_exe=args.python_exe,
            metrics_script=metrics_script,
            project_root=project_root,
            reports_dir=reports_dir,
            cached_copy=cached_metrics_csv,
            skip_existing=args.skip_existing,
        )
    except Exception as exc:
        metrics_ok = False
        (reports_dir / "metrics_stage_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    # per-author stages
    for idx, author in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {author}")
        run_per_author_pipeline(
            author=author,
            csv_path=source_csvs[author],
            authors=authors,
            python_exe=args.python_exe,
            project_root=project_root,
            reports_dir=reports_dir,
            summary_txt_dir=summary_txt_dir,
            expertise_txt_dir=expertise_txt_dir,
            persona_txt_dir=persona_txt_dir,
            album_covers_dir=album_covers_dir,
            scopus2txt_script=scopus2txt_script,
            expertise_script=expertise_script,
            persona_script=persona_script,
            album_script=album_script,
            skip_existing=args.skip_existing,
        )
        write_consolidated_report(reports_dir / "pipeline_author_report.csv", authors)

    summary_lines = [
        f"Run completed: {now_iso()}",
        f"Total authors considered: {len(selected)}",
        f"Complete: {sum(1 for a in authors.values() if a.overall_status() == 'complete')}",
        f"Incomplete: {sum(1 for a in authors.values() if a.overall_status() != 'complete')}",
        f"Metrics stage status: {'complete' if metrics_ok else 'failed'}",
        "Site generation: not run by this script",
        "Paper recommendation: not run by this script",
        f"Metrics CSV available at project root: {project_root / 'author_summary_2025_present.csv'}",
        f"Cached metrics CSV copy: {cached_metrics_csv}",
    ]
    (reports_dir / "pipeline_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_consolidated_report(reports_dir / "pipeline_author_report.csv", authors)

    print("Runner complete.")
    print("This script does not run site generation.")
    print("This script does not run paper recommendation.")
    if metrics_ok:
        print(f"Metrics CSV: {project_root / 'author_summary_2025_present.csv'}")
    else:
        print(f"Metrics stage failed. See: {reports_dir / 'metrics_stage_error.txt'}")
    print(f"Report: {reports_dir / 'pipeline_author_report.csv'}")
    print(f"Summary: {reports_dir / 'pipeline_summary.txt'}")


if __name__ == "__main__":
    main()
