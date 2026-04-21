#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
RUNS_ROOT = Path(tempfile.gettempdir()) / "valiantwrapped_single_gui_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

SCOPUS2TXT_SCRIPT = BASE_DIR / "scopus2txtsummary.py"
EXPERTISE_SCRIPT = BASE_DIR / "author_expertise_llama31_v4.py"
PERSONA_SCRIPT = BASE_DIR / "author_persona_llama31_v4.py"
ALBUM_SCRIPT = BASE_DIR / "generate_album_covers.py"
HEADSHOT_SCRIPT = BASE_DIR / "generate_musician_headshot_v10.py"
SITE_SCRIPT = BASE_DIR / "generate_valiantwrapped_site_solo_revised.py"

ALLOWED_CSV_EXTENSIONS = {".csv"}
ALLOWED_HEADSHOT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PAGE_HTML = r'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VALIANT Wrapped Studio</title>
  <style>
    :root{
      --bg:#070707; --panel:#121212; --panel2:#181818; --line:#2a2a2a; --text:#f5f5f5; --muted:#b3b3b3;
      --green:#1db954; --green2:#1ed760; --danger:#ef4444; --shadow:0 20px 48px rgba(0,0,0,.35);
    }
    *{box-sizing:border-box}
    body{
      margin:0; color:var(--text); font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:
        radial-gradient(circle at top right, rgba(29,185,84,.18), transparent 24%),
        radial-gradient(circle at left bottom, rgba(255,255,255,.06), transparent 20%),
        linear-gradient(180deg, #050505 0%, #0b0b0b 100%);
      min-height:100vh;
    }
    .wrap{max-width:1180px; margin:0 auto; padding:28px 18px 48px;}
    .hero{background:linear-gradient(145deg, rgba(24,24,24,.98), rgba(10,10,10,.98)); border:1px solid rgba(255,255,255,.08); border-radius:28px; box-shadow:var(--shadow); padding:30px;}
    .eyebrow{text-transform:uppercase; letter-spacing:.12em; color:var(--green2); font-size:.76rem; font-weight:800}
    h1{margin:8px 0 12px; font-size:clamp(2.2rem, 5vw, 4rem); line-height:.95; letter-spacing:-.04em; max-width:13ch}
    .lede{max-width:72ch; color:#ededed; line-height:1.6; margin:0}
    .layout{display:grid; grid-template-columns:minmax(340px, 470px) 1fr; gap:22px; margin-top:22px}
    .card{background:var(--panel); border:1px solid rgba(255,255,255,.06); border-radius:24px; box-shadow:var(--shadow); padding:24px}
    .card h2{margin:0 0 14px; font-size:1.5rem; letter-spacing:-.03em}
    .helper{color:var(--muted); line-height:1.6; font-size:.95rem}
    .field{display:flex; flex-direction:column; gap:8px; margin-bottom:14px}
    label{font-size:.9rem; font-weight:700; color:#ebebeb}
    input[type=text], input[type=file]{width:100%; border-radius:14px; border:1px solid rgba(255,255,255,.10); background:#0f0f0f; color:var(--text); padding:14px 14px; font-size:15px}
    input[type=file]{padding:12px}
    .mini{font-size:.82rem; color:var(--muted)}
    .actions{display:flex; gap:10px; flex-wrap:wrap; margin-top:18px}
    button, a.btn{display:inline-flex; align-items:center; justify-content:center; padding:13px 16px; border-radius:999px; border:none; cursor:pointer; font-weight:800; text-decoration:none; transition:.15s ease; font-size:.95rem}
    button:hover, a.btn:hover{transform:translateY(-1px)}
    button.primary{background:var(--green); color:#08130b}
    button.secondary, a.btn.secondary{background:rgba(255,255,255,.08); color:var(--text); border:1px solid rgba(255,255,255,.08)}
    button.danger{background:rgba(239,68,68,.12); color:#fecaca; border:1px solid rgba(239,68,68,.26)}
    button:disabled{opacity:.55; cursor:not-allowed; transform:none}
    .status-pill{display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px; background:rgba(255,255,255,.08); color:#e5e7eb; font-weight:800; font-size:.8rem}
    .status-pill.running{background:rgba(29,185,84,.14); color:#bbf7d0}
    .status-pill.success{background:rgba(34,197,94,.16); color:#bbf7d0}
    .status-pill.error{background:rgba(239,68,68,.16); color:#fecaca}
    .status-pill.cancelled{background:rgba(245,158,11,.16); color:#fde68a}
    .progress{margin-top:14px; width:100%; height:12px; border-radius:999px; overflow:hidden; background:#202020}
    .progress > div{height:100%; width:0%; background:linear-gradient(90deg, var(--green2), var(--green)); transition:width .3s ease}
    .logs{margin-top:14px; background:#0a0a0a; color:#dbfce7; border-radius:18px; padding:14px; min-height:340px; max-height:500px; overflow:auto; white-space:pre-wrap; font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; border:1px solid rgba(255,255,255,.05)}
    .result-card{margin-top:16px; background:var(--panel2); border:1px solid rgba(255,255,255,.06); border-radius:20px; padding:18px}
    .result-title{font-size:1.1rem; font-weight:800}
    .result-meta{margin-top:4px; color:var(--muted); font-size:.9rem}
    .result-actions{display:flex; gap:10px; flex-wrap:wrap; margin-top:14px}
    .two-col{display:grid; grid-template-columns:1fr 1fr; gap:12px}
    @media (max-width: 980px){ .layout{grid-template-columns:1fr} .two-col{grid-template-columns:1fr} }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">VALIANT Wrapped</div>
      <h1>Generate one personalized VALIANT Wrapped profile.</h1>
      <p class="lede">Upload a Scopus CSV export and a real headshot, add the person’s name and Scopus ID, and the app will run the single-author pipeline end to end using the latest scripts.</p>
    </section>

    <div class="layout">
      <section class="card">
        <h2>Required inputs</h2>
        <p class="helper">This version is strictly single-person only. All five fields below are required.</p>
        <form id="runForm" enctype="multipart/form-data">
          <div class="two-col">
            <div class="field">
              <label for="first_name">First Name</label>
              <input id="first_name" name="first_name" type="text" required>
            </div>
            <div class="field">
              <label for="last_name">Last Name</label>
              <input id="last_name" name="last_name" type="text" required>
            </div>
          </div>
          <div class="field">
            <label for="scopus_id">Scopus ID</label>
            <input id="scopus_id" name="scopus_id" type="text" required>
            <div class="mini">Used to build the canonical author label and the temporary headshot lookup record.</div>
          </div>
          <div class="field">
            <label for="scopus_csv">Scopus CSV export</label>
            <input id="scopus_csv" name="scopus_csv" type="file" accept=".csv" required>
          </div>
          <div class="field">
            <label for="headshot">Headshot image</label>
            <input id="headshot" name="headshot" type="file" accept=".jpg,.jpeg,.png,.webp" required>
          </div>
          <div class="actions">
            <button class="primary" id="runBtn" type="submit">Run VALIANT Wrapped</button>
            <button class="danger" id="stopBtn" type="button" disabled>Stop run</button>
            <button class="secondary" id="resetBtn" type="button">Reset</button>
          </div>
        </form>
      </section>

      <section class="card">
        <h2>Pipeline status</h2>
        <div class="status-pill" id="statusPill">QUEUED</div>
        <div style="margin-top:12px; font-size:1.15rem; font-weight:800" id="currentStep">Waiting to start</div>
        <div class="helper" style="margin-top:6px" id="runMeta">No active run.</div>
        <div class="progress"><div id="progressBar"></div></div>
        <div class="logs" id="logs">Ready.</div>
        <div id="resultArea"></div>
      </section>
    </div>
  </div>

<script>
(() => {
  let currentRunId = null;
  let pollTimer = null;

  const form = document.getElementById('runForm');
  const runBtn = document.getElementById('runBtn');
  const stopBtn = document.getElementById('stopBtn');
  const resetBtn = document.getElementById('resetBtn');
  const statusPill = document.getElementById('statusPill');
  const currentStep = document.getElementById('currentStep');
  const runMeta = document.getElementById('runMeta');
  const progressBar = document.getElementById('progressBar');
  const logs = document.getElementById('logs');
  const resultArea = document.getElementById('resultArea');

  function setStatus(status, label) {
    statusPill.textContent = label || String(status || 'queued').toUpperCase();
    statusPill.className = 'status-pill' + (status ? ' ' + status : '');
  }

  function resetUi() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    currentRunId = null;
    form.reset();
    setStatus('', 'QUEUED');
    currentStep.textContent = 'Waiting to start';
    runMeta.textContent = 'No active run.';
    progressBar.style.width = '0%';
    logs.textContent = 'Ready.';
    resultArea.innerHTML = '';
    runBtn.disabled = false;
    stopBtn.disabled = true;
  }

  function renderResult(data) {
    if (!data || !data.site_url) {
      resultArea.innerHTML = '';
      return;
    }
    const html = `
      <div class="result-card">
        <div class="result-title">${data.display_name || 'Generated site ready'}</div>
        <div class="result-meta">Author label: ${data.author_label || ''}</div>
        <div class="result-actions">
          <a class="btn secondary" href="${data.site_url}" target="_blank" rel="noopener">Open generated site</a>
          <a class="btn secondary" href="${data.author_page_url || data.site_url}" target="_blank" rel="noopener">Open author page</a>
        </div>
      </div>`;
    resultArea.innerHTML = html;
  }

  async function pollStatus() {
    if (!currentRunId) return;
    const res = await fetch('/api/status/' + encodeURIComponent(currentRunId));
    const data = await res.json();
    if (!res.ok || !data.ok) {
      setStatus('error', 'ERROR');
      currentStep.textContent = data.error || 'Status check failed';
      runMeta.textContent = 'Could not retrieve run status.';
      runBtn.disabled = false;
      stopBtn.disabled = true;
      if (pollTimer) clearInterval(pollTimer);
      return;
    }
    setStatus(data.status, String(data.status || 'queued').toUpperCase());
    currentStep.textContent = data.current_step || 'Working';
    runMeta.textContent = data.run_id ? `Run ID: ${data.run_id}` : 'Working';
    progressBar.style.width = String(data.progress_pct || 0) + '%';
    logs.textContent = (data.logs || []).join('\n\n') || 'Working...';
    logs.scrollTop = logs.scrollHeight;
    renderResult(data.result || null);

    if (['success', 'error', 'cancelled'].includes(data.status)) {
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = null;
      runBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (currentRunId) {
      alert('A run is already active.');
      return;
    }
    const fd = new FormData(form);
    runBtn.disabled = true;
    stopBtn.disabled = true;
    setStatus('running', 'RUNNING');
    currentStep.textContent = 'Submitting run';
    runMeta.textContent = 'Starting pipeline...';
    logs.textContent = 'Submitting run...';
    resultArea.innerHTML = '';
    progressBar.style.width = '0%';

    try {
      const res = await fetch('/api/run', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to start run.');
      currentRunId = data.run_id;
      stopBtn.disabled = false;
      await pollStatus();
      pollTimer = setInterval(pollStatus, 2000);
    } catch (err) {
      alert(err.message || String(err));
      runBtn.disabled = false;
      stopBtn.disabled = true;
      setStatus('error', 'ERROR');
      currentStep.textContent = 'Could not start run';
    }
  });

  stopBtn.addEventListener('click', async () => {
    if (!currentRunId) return;
    const proceed = confirm('Stop the current run?');
    if (!proceed) return;
    try {
      const res = await fetch('/api/cancel/' + encodeURIComponent(currentRunId), { method: 'POST' });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to stop run.');
      setStatus('cancelled', 'STOPPING');
      currentStep.textContent = 'Stopping run...';
      stopBtn.disabled = true;
    } catch (err) {
      alert(err.message || String(err));
    }
  });

  resetBtn.addEventListener('click', () => {
    if (currentRunId) {
      const proceed = confirm('Reset the form? Stop any active run first.');
      if (!proceed) return;
    }
    resetUi();
  });
})();
</script>
</body>
</html>
'''


app = Flask(__name__)


@dataclass
class RunState:
    run_id: str
    status: str = "queued"
    current_step: str = "Queued"
    progress_pct: int = 0
    logs: list[str] = field(default_factory=list)
    error: str = ""
    work_dir: str = ""
    csv_path: str = ""
    headshot_path: str = ""
    author_label: str = ""
    display_name: str = ""
    site_url: str = ""
    author_page_url: str = ""
    cancel_requested: bool = False
    current_pid: Optional[int] = None


RUNS: dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()


def allowed_extension(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


def slug_piece(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    text = text.replace("_", " ")
    text = re.sub(r"[^A-Za-z0-9\- ]+", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def canonical_author_label(first_name: str, last_name: str, scopus_id: str) -> str:
    first = slug_piece(first_name)
    last = slug_piece(last_name)
    sid = re.sub(r"\D+", "", (scopus_id or "").strip())
    if not first or not last or not sid:
        raise ValueError("First Name, Last Name, and Scopus ID are required.")
    return f"{last}_{first}_{sid}"


def display_name(first_name: str, last_name: str) -> str:
    return f"{(first_name or '').strip()} {(last_name or '').strip()}".strip()


def append_log(run: RunState, text: str) -> None:
    run.logs.append(text)


def set_step(run: RunState, label: str, progress_pct: int) -> None:
    run.current_step = label
    run.progress_pct = progress_pct


def cancel_process_group(pid: Optional[int]) -> None:
    if not pid:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        pass


def run_subprocess(run: RunState, cmd: list[str], cwd: Path) -> None:
    if run.cancel_requested:
        raise RuntimeError("Run cancelled by user.")

    append_log(run, "$ " + " ".join(cmd))
    kwargs = {
        "cwd": str(cwd),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(cmd, **kwargs)
    run.current_pid = proc.pid

    try:
        assert proc.stdout is not None
        while True:
            if run.cancel_requested:
                cancel_process_group(run.current_pid)
                raise RuntimeError("Run cancelled by user.")
            line = proc.stdout.readline()
            if line:
                append_log(run, line.rstrip())
            elif proc.poll() is not None:
                break
        remainder = proc.stdout.read()
        if remainder:
            for line in remainder.splitlines():
                append_log(run, line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Command failed with exit code {code}.")
    finally:
        run.current_pid = None
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


def write_lookup_csv(path: Path, first_name: str, last_name: str, scopus_id: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "first_name", "last_name", "scopus", "orcid"])
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "1",
                "first_name": first_name.strip(),
                "last_name": last_name.strip(),
                "scopus": re.sub(r"\D+", "", scopus_id.strip()),
                "orcid": "",
            }
        )


def write_zero_metrics_csv(path: Path, author_label: str) -> None:
    """Create a placeholder metrics CSV for compatibility with the solo site generator.

    The solo site now computes lifetime paper/citation totals from the uploaded
    per-author Scopus CSV in --author-csv-dir, but it still accepts --metrics-csv
    as part of the shared collector flow. We therefore write a single zero row so
    the file exists without reintroducing the academic-year stats pipeline.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "author_id",
                "author_file",
                "pub_count_2025_present",
                "citation_count_2025_present",
                "top_journal_2025_present",
                "top_paper_title_2025_present",
                "top_paper_citations_2025_present",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "author_id": author_label,
                "author_file": f"{author_label}.csv",
                "pub_count_2025_present": 0,
                "citation_count_2025_present": 0,
                "top_journal_2025_present": "",
                "top_paper_title_2025_present": "",
                "top_paper_citations_2025_present": 0,
            }
        )


def build_author_page_url(run: RunState) -> str:
    # Solo site generator writes the artist page directly to site/index.html.
    return f"/runs/{run.run_id}/site/"


def execute_pipeline(run: RunState, first_name: str, last_name: str, scopus_id: str) -> None:
    python_exe = os.environ.get("PYTHON", os.sys.executable)
    work_dir = Path(run.work_dir)

    author_csv_dir = work_dir / "author_csvs"
    headshot_dir = work_dir / "author_headshots" / "documents"
    outputs_dir = work_dir / "outputs"
    summary_dir = outputs_dir / "summary_txt"
    expertise_dir = outputs_dir / "author_expertise_txt"
    persona_dir = outputs_dir / "author_music_personas_txt"
    album_dir = outputs_dir / "album_covers"
    musician_headshots_dir = outputs_dir / "musician_headshots"
    recommendations_dir = outputs_dir / "paper_recommendations" / "per_author_txt"
    site_dir = work_dir / "site"
    lookup_csv = work_dir / "lookup.csv"
    metrics_csv = work_dir / "author_summary_2025_present.csv"

    for p in [author_csv_dir, headshot_dir, summary_dir, expertise_dir, persona_dir, album_dir, musician_headshots_dir, recommendations_dir, site_dir]:
        p.mkdir(parents=True, exist_ok=True)

    try:
        run.status = "running"
        set_step(run, "Preparing files", 5)
        append_log(run, f"Run ID: {run.run_id}")
        append_log(run, f"Author label: {run.author_label}")
        write_lookup_csv(lookup_csv, first_name, last_name, scopus_id)
        write_zero_metrics_csv(metrics_csv, run.author_label)

        summary_txt = summary_dir / f"{run.author_label}.txt"
        expertise_txt = expertise_dir / f"{run.author_label}.txt"
        persona_txt = persona_dir / f"{run.author_label}.txt"

        set_step(run, "Converting Scopus CSV to text summary", 15)
        run_subprocess(
            run,
            [
                python_exe,
                str(SCOPUS2TXT_SCRIPT),
                "--input-file",
                str(Path(run.csv_path)),
                "--output-dir",
                str(summary_dir),
            ],
            cwd=work_dir,
        )
        if not summary_txt.exists():
            raise FileNotFoundError(f"Expected summary TXT was not created: {summary_txt}")

        set_step(run, "Generating expertise summary", 32)
        run_subprocess(
            run,
            [
                python_exe,
                str(EXPERTISE_SCRIPT),
                "--input-file",
                str(summary_txt),
                "--output-txt-dir",
                str(expertise_dir),
                "--output-csv",
                str(outputs_dir / "author_expertise_summaries.csv"),
            ],
            cwd=work_dir,
        )
        if not expertise_txt.exists():
            raise FileNotFoundError(f"Expected expertise TXT was not created: {expertise_txt}")

        set_step(run, "Generating music persona", 48)
        run_subprocess(
            run,
            [
                python_exe,
                str(PERSONA_SCRIPT),
                "--input-file",
                str(expertise_txt),
                "--output-dir",
                str(persona_dir),
            ],
            cwd=work_dir,
        )
        if not persona_txt.exists():
            raise FileNotFoundError(f"Expected persona TXT was not created: {persona_txt}")

        set_step(run, "Generating album cover", 64)
        run_subprocess(
            run,
            [
                python_exe,
                str(ALBUM_SCRIPT),
                "--input-file",
                str(persona_txt),
                "--output-dir",
                str(album_dir),
                "--cleanup-intermediate",
            ],
            cwd=work_dir,
        )

        set_step(run, "Generating musician headshot", 80)
        run_subprocess(
            run,
            [
                python_exe,
                str(HEADSHOT_SCRIPT),
                "--input-file",
                str(persona_txt),
                "--project-root",
                str(work_dir),
                "--lookup-csv",
                lookup_csv.name,
                "--headshot-dir",
                str(Path("author_headshots") / "documents"),
                "--output-dir",
                str(Path("outputs") / "musician_headshots"),
            ],
            cwd=work_dir,
        )

        set_step(run, "Building solo VALIANT Wrapped site", 93)
        run_subprocess(
            run,
            [
                python_exe,
                str(SITE_SCRIPT),
                "--metrics-csv",
                str(metrics_csv),
                "--author-csv-dir",
                str(author_csv_dir),
                "--expertise-dir",
                str(expertise_dir),
                "--persona-dir",
                str(persona_dir),
                "--cover-dir",
                str(album_dir),
                "--musician-headshots-dir",
                str(musician_headshots_dir),
                "--recommendations-dir",
                str(recommendations_dir),
                "--output-dir",
                str(site_dir),
                "--project-title",
                "VALIANT Wrapped",
                "--tagline",
                "We all know AI hallucinates — so we decided to make it sing.",
                "--closer",
                "Thanks for being part of the sound. We are already queued up for what you drop next.",
                "--verbose",
            ],
            cwd=work_dir,
        )

        site_index = site_dir / "index.html"
        if not site_index.exists():
            raise FileNotFoundError(f"Expected solo artist page was not created: {site_index}")

        run.site_url = f"/runs/{run.run_id}/site/"
        append_log(run, "Solo site generator completed: opening site root will go directly to the artist page.")
        run.author_page_url = build_author_page_url(run)
        set_step(run, "Complete", 100)
        run.status = "success"
        append_log(run, "Pipeline finished successfully.")
    except Exception as exc:
        if run.cancel_requested or "cancelled by user" in str(exc).lower():
            run.status = "cancelled"
            run.error = "Run cancelled by user."
            set_step(run, "Run cancelled", min(run.progress_pct, 99))
            append_log(run, "Run cancelled by user.")
        else:
            run.status = "error"
            run.error = str(exc)
            set_step(run, "Run failed", min(max(run.progress_pct, 1), 99))
            append_log(run, f"ERROR: {exc}")
    finally:
        run.current_pid = None


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/api/run", methods=["POST"])
def api_run():
    first_name = (request.form.get("first_name") or "").strip()
    last_name = (request.form.get("last_name") or "").strip()
    scopus_id = (request.form.get("scopus_id") or "").strip()
    scopus_csv = request.files.get("scopus_csv")
    headshot = request.files.get("headshot")

    if not first_name or not last_name or not scopus_id:
        return jsonify({"ok": False, "error": "First Name, Last Name, and Scopus ID are required."}), 400
    if not scopus_csv or not scopus_csv.filename:
        return jsonify({"ok": False, "error": "Scopus CSV upload is required."}), 400
    if not headshot or not headshot.filename:
        return jsonify({"ok": False, "error": "Headshot upload is required."}), 400
    if not allowed_extension(scopus_csv.filename, ALLOWED_CSV_EXTENSIONS):
        return jsonify({"ok": False, "error": "Scopus upload must be a .csv file."}), 400
    if not allowed_extension(headshot.filename, ALLOWED_HEADSHOT_EXTENSIONS):
        return jsonify({"ok": False, "error": "Headshot must be a .jpg, .jpeg, .png, or .webp file."}), 400

    required_scripts = [
        SCOPUS2TXT_SCRIPT,
        EXPERTISE_SCRIPT,
        PERSONA_SCRIPT,
        ALBUM_SCRIPT,
        HEADSHOT_SCRIPT,
        SITE_SCRIPT,
    ]
    missing = [str(p) for p in required_scripts if not p.exists()]
    if missing:
        return jsonify({"ok": False, "error": f"Missing required script(s): {', '.join(missing)}"}), 500

    try:
        author_label = canonical_author_label(first_name, last_name, scopus_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    run_id = uuid.uuid4().hex[:12]
    work_dir = RUNS_ROOT / run_id
    author_csv_dir = work_dir / "author_csvs"
    headshot_dir = work_dir / "author_headshots" / "documents"
    author_csv_dir.mkdir(parents=True, exist_ok=True)
    headshot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = author_csv_dir / f"{author_label}.csv"
    csv_path.write_bytes(scopus_csv.read())

    headshot_ext = Path(secure_filename(headshot.filename)).suffix.lower() or ".jpg"
    headshot_path = headshot_dir / f"1_photo{headshot_ext}"
    headshot_path.write_bytes(headshot.read())

    run = RunState(
        run_id=run_id,
        work_dir=str(work_dir),
        csv_path=str(csv_path),
        headshot_path=str(headshot_path),
        author_label=author_label,
        display_name=display_name(first_name, last_name),
    )
    append_log(run, "Created single-author run workspace.")

    with RUNS_LOCK:
        RUNS[run_id] = run

    thread = threading.Thread(
        target=execute_pipeline,
        args=(run, first_name, last_name, scopus_id),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "run_id": run_id})


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
            "current_step": run.current_step,
            "progress_pct": run.progress_pct,
            "logs": run.logs[-500:],
            "error": run.error,
            "result": (
                {
                    "author_label": run.author_label,
                    "display_name": run.display_name,
                    "site_url": run.site_url,
                    "author_page_url": run.author_page_url,
                }
                if run.status == "success"
                else None
            ),
        }
    )


@app.route("/api/cancel/<run_id>", methods=["POST"])
def api_cancel(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"ok": False, "error": "Run not found."}), 404
    run.cancel_requested = True
    cancel_process_group(run.current_pid)
    append_log(run, "Cancellation requested.")
    return jsonify({"ok": True})


@app.route("/runs/<run_id>/site")
@app.route("/runs/<run_id>/site/")
def run_site_root(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        abort(404)
    return redirect(f"/runs/{run_id}/site/index.html")


@app.route("/runs/<run_id>/site/<path:subpath>")
def serve_run_site(run_id: str, subpath: str):
    run = RUNS.get(run_id)
    if not run:
        abort(404)
    site_dir = Path(run.work_dir) / "site"
    requested = site_dir / subpath
    if requested.is_dir():
        subpath = str(Path(subpath) / "index.html")
    return send_from_directory(site_dir, subpath)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
