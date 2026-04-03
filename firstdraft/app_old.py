#!/usr/bin/env python3
"""
app.py

Localhost GUI for VALIANT Wrapped.

Features:
- Upload one or many Scopus CSV files
- Auto-select single vs batch (or let user choose)
- Run pipeline in background with progress updates
- Preview generated author pages
- Download results as ZIP
- Generate and download author manifest CSV

Assumes pipeline scripts live in the same folder as this file.
"""

from __future__ import annotations

import os
import uuid
import shutil
import zipfile
import tempfile
import threading
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    abort,
)
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
RUNS_ROOT = Path(tempfile.gettempdir()) / "valiantwrapped_gui_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv"}

app = Flask(__name__, template_folder="templates", static_folder="static")


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def safe_author_label_from_name(filename: str) -> str:
    return Path(filename).stem


def parse_author_label(label: str) -> tuple[str, str, str]:
    parts = str(label).split("_")
    last = parts[0] if len(parts) > 0 else ""
    first = parts[1] if len(parts) > 1 else ""
    scopus_id = parts[-1] if len(parts) > 2 else ""
    return first, last, scopus_id


@dataclass
class RunState:
    run_id: str
    mode_requested: str
    mode_effective: str = ""
    status: str = "queued"      # queued/running/success/error
    step_index: int = 0
    step_total: int = 6
    current_step: str = "Queued"
    progress_pct: int = 0
    error: str = ""
    logs: list[str] = field(default_factory=list)
    work_dir: str = ""
    uploads_dir: str = ""
    summary_dir: str = ""
    expertise_dir: str = ""
    persona_dir: str = ""
    album_covers_dir: str = ""
    docs_dir: str = ""
    zip_path: str = ""
    manifest_path: str = ""
    author_labels: list[str] = field(default_factory=list)
    result_pages: list[dict] = field(default_factory=list)


RUNS: dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()


def append_log(run: RunState, text: str) -> None:
    run.logs.append(text)


def set_step(run: RunState, idx: int, label: str) -> None:
    run.step_index = idx
    run.current_step = label
    run.progress_pct = int((idx - 1) / run.step_total * 100)


def finalize_progress(run: RunState) -> None:
    run.progress_pct = 100


def run_subprocess(run: RunState, cmd: list[str], label: str) -> None:
    append_log(run, f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        append_log(run, proc.stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}.")


def build_zip(run: RunState) -> str:
    zip_path = Path(run.work_dir) / "valiantwrapped_results.zip"
    docs_dir = Path(run.docs_dir)
    covers_dir = Path(run.album_covers_dir)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if docs_dir.exists():
            for p in docs_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(
                        Path("docs") / p.relative_to(docs_dir)))
        if covers_dir.exists():
            for p in covers_dir.rglob("*"):
                if p.is_file():
                    zf.write(
                        p,
                        arcname=str(Path("outputs") / "album_covers" /
                                    p.relative_to(covers_dir)),
                    )
    run.zip_path = str(zip_path)
    return str(zip_path)


def collect_result_pages(run: RunState) -> list[dict]:
    author_dir = Path(run.docs_dir) / "authors"
    rows: list[dict] = []
    if not author_dir.exists():
        return rows

    for html_path in sorted(author_dir.glob("*.html")):
        author_label = html_path.stem
        first, last, scopus_id = parse_author_label(author_label)
        rows.append(
            {
                "author_label": author_label,
                "first_name": first,
                "last_name": last,
                "scopus_id": scopus_id,
                "display_name": f"{first} {last}".strip() or author_label,
                "page_url": f"/runs/{run.run_id}/docs/authors/{html_path.name}",
            }
        )
    run.result_pages = rows
    return rows


def execute_pipeline(run: RunState, uploaded_files: list[Path]) -> None:
    python_exe = os.environ.get("PYTHON", os.sys.executable)

    try:
        run.status = "running"
        run.mode_effective = "single" if len(
            uploaded_files) == 1 and run.mode_requested != "batch" else "batch"
        if run.mode_requested == "single":
            run.mode_effective = "single"
        if run.mode_requested == "batch":
            run.mode_effective = "batch"

        run.author_labels = [safe_author_label_from_name(
            p.name) for p in uploaded_files]

        set_step(run, 1, "Preparing uploads")
        append_log(run, f"Run ID: {run.run_id}")
        append_log(run, f"Requested mode: {run.mode_requested}")
        append_log(run, f"Effective mode: {run.mode_effective}")
        append_log(run, f"Uploaded files: {len(uploaded_files)}")

        if run.mode_effective == "single":
            input_flag = ["--input-file", str(uploaded_files[0])]
            author_label = safe_author_label_from_name(uploaded_files[0].name)
        else:
            input_flag = ["--input-dir", run.uploads_dir]
            author_label = ""

        set_step(run, 2, "Step 1/5 · Reading Scopus CSVs")
        run_subprocess(
            run,
            [
                python_exe,
                "scopus2txtsummary.py",
                *input_flag,
                "--output-dir",
                run.summary_dir,
                "--year-cutoff",
                "2025",
            ],
            "scopus2txtsummary.py",
        )

        set_step(run, 3, "Step 2/5 · Generating expertise summaries")
        expertise_input = (
            ["--input-file", str(Path(run.summary_dir) /
                                 f"{author_label}.txt")]
            if run.mode_effective == "single"
            else ["--input-dir", run.summary_dir]
        )
        run_subprocess(
            run,
            [
                python_exe,
                "author_expertise_llama31_2.py",
                *expertise_input,
                "--output-dir",
                run.expertise_dir,
            ],
            "author_expertise_llama31_2.py",
        )

        set_step(run, 4, "Step 3/5 · Creating music personas")
        persona_input = (
            ["--input-file", str(Path(run.expertise_dir) /
                                 f"{author_label}.txt")]
            if run.mode_effective == "single"
            else ["--input-dir", run.expertise_dir]
        )
        run_subprocess(
            run,
            [
                python_exe,
                "author_persona_llama31.py",
                *persona_input,
                "--output-dir",
                run.persona_dir,
            ],
            "author_persona_llama31.py",
        )

        set_step(run, 5, "Step 4/5 · Generating album covers")
        cover_input = (
            ["--input-file", str(Path(run.persona_dir) /
                                 f"{author_label}.txt")]
            if run.mode_effective == "single"
            else ["--input-dir", run.persona_dir]
        )
        run_subprocess(
            run,
            [
                python_exe,
                "generate_album_covers.py",
                *cover_input,
                "--output-dir",
                run.album_covers_dir,
            ],
            "generate_album_covers.py",
        )

        set_step(run, 6, "Step 5/5 · Building HTML pages")
        site_cmd = [
            python_exe,
            "generate_valiantwrapped_site_noindex.py",
            "--summary-dir",
            run.summary_dir,
            "--persona-dir",
            run.persona_dir,
            "--album-covers-dir",
            run.album_covers_dir,
            "--docs-dir",
            run.docs_dir,
        ]
        if run.mode_effective == "single":
            site_cmd.extend(["--author-label", author_label])

        run_subprocess(
            run, site_cmd, "generate_valiantwrapped_site_noindex.py")

        collect_result_pages(run)
        build_zip(run)

        run.status = "success"
        finalize_progress(run)
        append_log(run, "Pipeline finished successfully.")

    except Exception as e:
        run.status = "error"
        run.error = str(e)
        append_log(run, f"ERROR: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    uploaded = request.files.getlist("files")
    mode_requested = (request.form.get("run_mode") or "auto").strip().lower()

    files = [f for f in uploaded if f and f.filename]
    if not files:
        return jsonify({"ok": False, "error": "Please upload at least one CSV file."}), 400

    for f in files:
        if not allowed_file(f.filename):
            return jsonify({"ok": False, "error": f"Unsupported file type: {f.filename}"}), 400

    if mode_requested not in {"auto", "single", "batch"}:
        return jsonify({"ok": False, "error": "Invalid run mode."}), 400

    if len(files) == 1 and mode_requested == "auto":
        mode_requested = "single"
    elif len(files) > 1 and mode_requested == "auto":
        mode_requested = "batch"

    run_id = uuid.uuid4().hex[:12]
    work_dir = RUNS_ROOT / run_id
    uploads_dir = work_dir / "uploads"
    outputs_dir = work_dir / "outputs"
    summary_dir = outputs_dir / "summary_txt"
    expertise_dir = outputs_dir / "expertise_txt"
    persona_dir = outputs_dir / "personas_txt"
    album_covers_dir = outputs_dir / "album_covers"
    docs_dir = work_dir / "docs"

    for p in [uploads_dir, summary_dir, expertise_dir, persona_dir, album_covers_dir, docs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for f in files:
        filename = secure_filename(f.filename)
        if not filename:
            continue
        out_path = uploads_dir / filename
        f.save(out_path)
        saved_paths.append(out_path)

    run = RunState(
        run_id=run_id,
        mode_requested=mode_requested,
        work_dir=str(work_dir),
        uploads_dir=str(uploads_dir),
        summary_dir=str(summary_dir),
        expertise_dir=str(expertise_dir),
        persona_dir=str(persona_dir),
        album_covers_dir=str(album_covers_dir),
        docs_dir=str(docs_dir),
    )

    with RUNS_LOCK:
        RUNS[run_id] = run

    t = threading.Thread(target=execute_pipeline,
                         args=(run, saved_paths), daemon=True)
    t.start()

    return jsonify(
        {
            "ok": True,
            "run_id": run_id,
            "mode_requested": mode_requested,
        }
    )


@app.route("/api/status/<run_id>")
def api_status(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"ok": False, "error": "Run not found."}), 404

    return jsonify(
        {
            "ok": True,
            "run_id": run.run_id,
            "status": run.status,
            "mode_requested": run.mode_requested,
            "mode_effective": run.mode_effective,
            "step_index": run.step_index,
            "step_total": run.step_total,
            "current_step": run.current_step,
            "progress_pct": run.progress_pct,
            "error": run.error,
            "logs": run.logs[-200:],
            "result_pages": run.result_pages,
            "zip_download_url": f"/api/download-zip/{run.run_id}" if run.zip_path else "",
            "manifest_download_url": f"/api/manifest/{run.run_id}" if run.manifest_path else "",
        }
    )


@app.route("/api/manifest/<run_id>", methods=["POST", "GET"])
def api_manifest(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"ok": False, "error": "Run not found."}), 404

    docs_authors = Path(run.docs_dir) / "authors"
    if not docs_authors.exists():
        return jsonify({"ok": False, "error": "No generated author pages found yet."}), 400

    manifest_path = Path(run.docs_dir) / "author_url_manifest.csv"
    python_exe = os.environ.get("PYTHON", os.sys.executable)

    try:
        proc = subprocess.run(
            [
                python_exe,
                "build_author_url_manifest.py",
                "--authors-dir",
                str(docs_authors),
                "--output-file",
                str(manifest_path),
            ],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout:
            append_log(run, proc.stdout.strip())
        if proc.returncode != 0:
            raise RuntimeError(
                f"Manifest generation failed with exit code {proc.returncode}.")
        run.manifest_path = str(manifest_path)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify(
        {
            "ok": True,
            "download_url": f"/api/download-manifest/{run.run_id}",
        }
    )


@app.route("/api/download-manifest/<run_id>")
def api_download_manifest(run_id: str):
    run = RUNS.get(run_id)
    if not run or not run.manifest_path or not Path(run.manifest_path).exists():
        abort(404)
    return send_file(run.manifest_path, as_attachment=True)


@app.route("/api/download-zip/<run_id>")
def api_download_zip(run_id: str):
    run = RUNS.get(run_id)
    if not run or not run.zip_path or not Path(run.zip_path).exists():
        abort(404)
    return send_file(run.zip_path, as_attachment=True)


@app.route("/runs/<run_id>/docs/<path:subpath>")
def serve_generated_docs(run_id: str, subpath: str):
    run = RUNS.get(run_id)
    if not run:
        abort(404)
    docs_dir = Path(run.docs_dir)
    target = docs_dir / subpath
    if not target.exists():
        abort(404)
    return send_from_directory(docs_dir, subpath)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
