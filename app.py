#!/usr/bin/env python3
"""
app.py

Localhost GUI for VALIANT Wrapped.

Features:
- Upload one or many Scopus CSV files
- Required author identity fields for single-author runs
- Auto-select single vs batch (or let user choose)
- Run pipeline in background with progress updates
- Stop/kill active runs
- Preview generated author pages
- Download results as ZIP
- Generate and download author manifest CSV
- Generate social-share PNG cards for download

Assumes pipeline scripts live in the same folder as this file.
"""

from __future__ import annotations

import io
import os
import re
import uuid
import zipfile
import tempfile
import threading
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    send_from_directory,
    abort,
    render_template_string,
)
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
RUNS_ROOT = Path(tempfile.gettempdir()) / "valiantwrapped_gui_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv"}

app = Flask(__name__)


PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VALIANT Wrapped</title>
  <style>
    :root{
      --bg1:#0b1020; --bg2:#171c34; --card:#ffffff; --card2:#f5f7ff; --ink:#12203a;
      --muted:#64748b; --accent:#7c3aed; --accent2:#06b6d4; --gold:#c5b358; --danger:#dc2626;
      --ok:#16a34a; --shadow:0 16px 40px rgba(0,0,0,.18); --radius:22px;
    }
    *{box-sizing:border-box}
    body{
      margin:0; font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:#e5e7eb;
      background:
        radial-gradient(1000px 500px at 10% 0%, rgba(124,58,237,.35), transparent 60%),
        radial-gradient(900px 500px at 100% 10%, rgba(6,182,212,.22), transparent 60%),
        linear-gradient(180deg,var(--bg1),var(--bg2));
      min-height:100vh;
    }
    .wrap{max-width:1180px; margin:0 auto; padding:32px 20px 64px;}
    .hero{
      position:relative; overflow:hidden; border-radius:28px; padding:32px;
      background:linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
      border:1px solid rgba(255,255,255,.10); box-shadow:var(--shadow);
      backdrop-filter: blur(14px);
    }
    .hero::after{
      content:""; position:absolute; right:-140px; top:-140px; width:320px; height:320px;
      background:radial-gradient(circle at center, rgba(197,179,88,.28), transparent 62%);
      pointer-events:none;
    }
    .eyebrow{font-size:12px; letter-spacing:.14em; text-transform:uppercase; font-weight:800; color:#cbd5e1;}
    h1{margin:10px 0 12px; font-size:48px; line-height:1.02; max-width:14ch;}
    .lead{margin:0; max-width:74ch; color:#dbe4f3; line-height:1.6; font-size:17px;}
    .hook{
      margin-top:18px; display:inline-block; padding:12px 16px; border-radius:999px;
      background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.12);
      font-weight:700; color:#fff;
    }
    .layout{display:grid; grid-template-columns: 1.02fr .98fr; gap:22px; margin-top:24px;}
    .panel{
      background:var(--card); color:var(--ink); border-radius:var(--radius); box-shadow:var(--shadow);
      padding:22px; border:1px solid rgba(0,0,0,.04);
    }
    .panel h2{margin:0 0 16px; font-size:24px; color:#1e293b}
    .subtle{color:var(--muted); line-height:1.55;}
    .grid3{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px;}
    .field{display:flex; flex-direction:column; gap:6px; margin-bottom:14px;}
    label{font-size:13px; font-weight:800; color:#334155;}
    input[type=text], select, input[type=file]{
      width:100%; border:1px solid #d8deea; border-radius:14px; padding:13px 14px; font-size:15px;
      background:#fff; color:#0f172a;
    }
    input[type=file]{padding:11px 12px; background:#f8fbff}
    .hint{font-size:12px; color:#64748b; margin-top:4px;}
    .actions{display:flex; gap:10px; flex-wrap:wrap; margin-top:16px;}
    button{
      appearance:none; border:none; border-radius:14px; padding:13px 16px; font-size:15px; font-weight:800;
      cursor:pointer; transition:transform .12s ease, opacity .12s ease, box-shadow .12s ease;
    }
    button:hover{transform:translateY(-1px)}
    button.primary{background:linear-gradient(135deg, var(--accent), #4f46e5); color:#fff; box-shadow:0 12px 30px rgba(99,102,241,.28)}
    button.secondary{background:#edf2ff; color:#312e81}
    button.danger{background:#fee2e2; color:#991b1b}
    button.ghost{background:#f8fafc; color:#334155}
    button:disabled{opacity:.55; cursor:not-allowed; transform:none}
    .status-card{background:var(--card2); border-radius:18px; padding:18px; border:1px solid #e6ebf4;}
    .pill{display:inline-flex; align-items:center; gap:8px; padding:7px 10px; border-radius:999px; font-weight:800; font-size:12px; background:#e2e8f0; color:#334155}
    .pill.running{background:#e0f2fe; color:#075985}
    .pill.success{background:#dcfce7; color:#166534}
    .pill.error{background:#fee2e2; color:#991b1b}
    .pill.cancelled{background:#fef3c7; color:#92400e}
    .progress{margin-top:14px; width:100%; height:14px; background:#dfe7f3; border-radius:999px; overflow:hidden}
    .progress > div{height:100%; width:0%; background:linear-gradient(90deg,var(--accent2),var(--accent)); transition:width .3s ease}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    .logs{
      margin-top:14px; background:#0f172a; color:#dbeafe; border-radius:16px; padding:14px; height:260px; overflow:auto;
      font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap;
    }
    .results{margin-top:18px; display:grid; gap:12px}
    .result-card{border:1px solid #e6ebf4; border-radius:18px; padding:16px; background:#fff}
    .result-head{display:flex; justify-content:space-between; gap:10px; align-items:flex-start; flex-wrap:wrap}
    .result-title{font-size:18px; font-weight:900; color:#0f172a}
    .result-meta{color:#64748b; font-size:13px}
    .result-actions{display:flex; gap:8px; flex-wrap:wrap; margin-top:12px}
    a.btn{
      text-decoration:none; display:inline-flex; align-items:center; justify-content:center; border-radius:12px;
      padding:11px 13px; font-weight:800; font-size:14px; background:#eff6ff; color:#1d4ed8;
    }
    .share-note{margin-top:10px; color:#155e75; background:#ecfeff; border:1px solid #a5f3fc; border-radius:14px; padding:10px 12px; display:none}
    .share-note.show{display:block}
    .mini{font-size:12px; color:#64748b}
    @media (max-width: 980px){
      .layout{grid-template-columns:1fr}
      .grid3{grid-template-columns:1fr}
      h1{font-size:38px}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">VALIANT Wrapped</div>
      <h1>Turn your research record into a gloriously over-the-top AI music persona.</h1>
      <p class="lead">We all know AI hallucinates a little — so why not let it go all the way? Upload your Scopus export, spin it into stats, album art, and a fictional artist identity, then download a social-ready card built for sharing.</p>
      <div class="hook">Academic output, but make it album-drop energy.</div>
    </section>

    <div class="layout">
      <section class="panel">
        <h2>Start a run</h2>
        <p class="subtle">For single-author runs, enter the name exactly how you want it encoded into the pipeline label. Multi-word last names are supported.</p>
        <form id="runForm">
          <div class="grid3">
            <div class="field">
              <label for="first_name">First name <span class="mini">(required for single mode)</span></label>
              <input id="first_name" name="first_name" type="text" placeholder="Andre" />
            </div>
            <div class="field">
              <label for="last_name">Last name <span class="mini">(required for single mode)</span></label>
              <input id="last_name" name="last_name" type="text" placeholder="da Silva Hucke" />
            </div>
            <div class="field">
              <label for="scopus_id">Scopus ID <span class="mini">(required for single mode)</span></label>
              <input id="scopus_id" name="scopus_id" type="text" placeholder="58290603100" />
            </div>
          </div>
          <div class="field">
            <label for="run_mode">Run mode</label>
            <select id="run_mode" name="run_mode">
              <option value="auto">Auto-detect</option>
              <option value="single">Single author</option>
              <option value="batch">Batch</option>
            </select>
            <div class="hint">Single mode uses the fields above to create the canonical author label. Batch mode ignores them.</div>
          </div>
          <div class="field">
            <label for="files">Scopus CSV upload</label>
            <input id="files" name="files" type="file" accept=".csv" multiple />
          </div>
          <div class="actions">
            <button class="primary" id="runBtn" type="submit">Run VALIANT Wrapped</button>
            <button class="danger" id="killBtn" type="button" disabled>STOP / KILL RUN</button>
            <button class="ghost" id="resetBtn" type="button">Reset</button>
          </div>
        </form>
      </section>

      <section class="panel">
        <h2>Pipeline status</h2>
        <div class="status-card">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap;">
            <div>
              <div class="pill" id="statusPill">Queued</div>
              <div style="margin-top:10px; font-weight:800; font-size:18px; color:#0f172a;" id="currentStep">Waiting to start</div>
              <div class="subtle" style="margin-top:6px;" id="runMeta">No active run.</div>
            </div>
          </div>
          <div class="progress"><div id="progressBar"></div></div>
          <div class="logs mono" id="logs">Ready.</div>
        </div>
        <div class="results" id="results"></div>
      </section>
    </div>
  </div>

  <script>
    let currentRunId = null;
    let statusTimer = null;

    const form = document.getElementById('runForm');
    const runBtn = document.getElementById('runBtn');
    const killBtn = document.getElementById('killBtn');
    const resetBtn = document.getElementById('resetBtn');
    const resultsEl = document.getElementById('results');
    const logsEl = document.getElementById('logs');
    const progressBar = document.getElementById('progressBar');
    const statusPill = document.getElementById('statusPill');
    const currentStep = document.getElementById('currentStep');
    const runMeta = document.getElementById('runMeta');

    function escapeHtml(str) {
      return (str || '').replace(/[&<>'\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c] || c));
    }

    function setStatus(status, text) {
      statusPill.textContent = text || status;
      statusPill.className = 'pill ' + (status || '');
    }

    function clearPolling() {
      if (statusTimer) {
        clearInterval(statusTimer);
        statusTimer = null;
      }
    }

    function resetUiOnly() {
      clearPolling();
      currentRunId = null;
      form.reset();
      resultsEl.innerHTML = '';
      logsEl.textContent = 'Ready.';
      progressBar.style.width = '0%';
      currentStep.textContent = 'Waiting to start';
      runMeta.textContent = 'No active run.';
      setStatus('', 'Queued');
      killBtn.disabled = true;
      runBtn.disabled = false;
    }

    function renderResults(resultPages, runId) {
      if (!resultPages || !resultPages.length) {
        resultsEl.innerHTML = '';
        return;
      }
      resultsEl.innerHTML = '<h2 style="margin:0 0 2px; color:#0f172a;">Results</h2>' + resultPages.map(row => {
        const label = escapeHtml(row.author_label || '');
        const name = escapeHtml(row.display_name || row.author_label || 'Author');
        const scopus = escapeHtml(row.scopus_id || '');
        const pageUrl = row.page_url || '#';
        const pngUrl = `/api/download-social-card/${runId}/${encodeURIComponent(row.author_label)}`;
        return `
          <div class="result-card">
            <div class="result-head">
              <div>
                <div class="result-title">${name}</div>
                <div class="result-meta">Label: ${label}${scopus ? ' · Scopus ID: ' + scopus : ''}</div>
              </div>
            </div>
            <div class="result-actions">
              <a class="btn" href="${pageUrl}" target="_blank" rel="noopener">Open page</a>
              <button class="secondary share-btn" type="button" data-run-id="${runId}" data-author-label="${label}">Share to social</button>
            </div>
            <div class="share-note" id="share-note-${CSS.escape(row.author_label)}"></div>
          </div>
        `;
      }).join('');

      document.querySelectorAll('.share-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
          const runId = btn.dataset.runId;
          const authorLabel = btn.dataset.authorLabel;
          btn.disabled = true;
          btn.textContent = 'Generating PNG...';
          try {
            const res = await fetch(`/api/social-card/${runId}/${encodeURIComponent(authorLabel)}`, { method: 'POST' });
            const data = await res.json();
            if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to generate social card.');
            const note = document.getElementById(`share-note-${CSS.escape(authorLabel)}`);
            note.classList.add('show');
            note.innerHTML = `PNG download is available below for you to share on social media of your choice! <div style="margin-top:8px;"><a class="btn" href="${data.download_url}">Download social PNG</a></div>`;
            btn.textContent = 'PNG ready';
          } catch (err) {
            alert(err.message || String(err));
            btn.disabled = false;
            btn.textContent = 'Share to social';
          }
        });
      });
    }

    async function pollStatus() {
      if (!currentRunId) return;
      const res = await fetch(`/api/status/${currentRunId}`);
      const data = await res.json();
      if (!res.ok || !data.ok) {
        clearPolling();
        setStatus('error', 'Error');
        currentStep.textContent = 'Status check failed';
        runMeta.textContent = data.error || 'Unknown error';
        killBtn.disabled = true;
        runBtn.disabled = false;
        return;
      }
      setStatus(data.status, (data.status || 'queued').toUpperCase());
      currentStep.textContent = data.current_step || 'Working';
      runMeta.textContent = `Run ID: ${data.run_id} · Mode: ${data.mode_effective || data.mode_requested || 'n/a'}`;
      progressBar.style.width = `${data.progress_pct || 0}%`;
      logsEl.textContent = (data.logs || []).join('\n\n') || 'Working...';
      logsEl.scrollTop = logsEl.scrollHeight;
      renderResults(data.result_pages || [], data.run_id);

      if (['success', 'error', 'cancelled'].includes(data.status)) {
        clearPolling();
        killBtn.disabled = true;
        runBtn.disabled = false;
        if (data.status === 'error') {
          currentStep.textContent = data.error || 'Run failed';
        }
        if (data.status === 'cancelled') {
          currentStep.textContent = 'Run stopped.';
        }
      }
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (currentRunId) {
        alert('A run is already active. Stop it or reset the page before starting another.');
        return;
      }

      const fd = new FormData(form);
      runBtn.disabled = true;
      killBtn.disabled = true;
      resultsEl.innerHTML = '';
      logsEl.textContent = 'Submitting run...';
      currentStep.textContent = 'Submitting run';
      runMeta.textContent = 'Starting pipeline...';
      progressBar.style.width = '0%';

      try {
        const res = await fetch('/api/run', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to start run.');
        currentRunId = data.run_id;
        killBtn.disabled = false;
        setStatus('running', 'RUNNING');
        await pollStatus();
        statusTimer = setInterval(pollStatus, 2000);
      } catch (err) {
        alert(err.message || String(err));
        runBtn.disabled = false;
        killBtn.disabled = true;
        setStatus('error', 'Error');
        currentStep.textContent = 'Could not start run';
      }
    });

    killBtn.addEventListener('click', async () => {
      if (!currentRunId) return;
      if (!confirm('Stop the current run? Any in-progress step will be terminated.')) return;
      try {
        const res = await fetch(`/api/cancel/${currentRunId}`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'Failed to stop run.');
        setStatus('cancelled', 'STOPPING');
        currentStep.textContent = 'Stopping run...';
        killBtn.disabled = true;
      } catch (err) {
        alert(err.message || String(err));
      }
    });

    resetBtn.addEventListener('click', () => {
      if (currentRunId) {
        const proceed = confirm('Reset the page? If a run is active, stop it first.');
        if (!proceed) return;
      }
      resetUiOnly();
    });
  </script>
</body>
</html>
"""


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def slug_piece(value: str) -> str:
    s = re.sub(r"\s+", " ", (value or "").strip())
    s = s.replace("_", " ")
    s = re.sub(r"[^A-Za-z0-9\- ]+", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def build_author_label(first_name: str, last_name: str, scopus_id: str) -> str:
    first = slug_piece(first_name)
    last = slug_piece(last_name)
    sid = re.sub(r"\D+", "", (scopus_id or "").strip())
    if not first or not last or not sid:
        raise ValueError("First name, last name, and Scopus ID are required.")
    return f"{last}_{first}_{sid}"


def display_name_from_label(label: str) -> tuple[str, str, str, str]:
    parts = str(label).split("_")
    last = parts[0] if len(parts) > 0 else ""
    first = parts[1] if len(parts) > 1 else ""
    scopus_id = parts[-1] if len(parts) > 2 else ""
    pretty_first = first.replace("-", " ")
    pretty_last = last.replace("-", " ")
    display_name = f"{pretty_first} {pretty_last}".strip() or label
    return pretty_first, pretty_last, scopus_id, display_name


@dataclass
class RunState:
    run_id: str
    mode_requested: str
    mode_effective: str = ""
    status: str = "queued"      # queued/running/success/error/cancelled
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
    first_name: str = ""
    last_name: str = ""
    scopus_id: str = ""
    primary_author_label: str = ""
    cancel_requested: bool = False
    social_cards_dir: str = ""
    current_pid: Optional[int] = None


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


def cancel_run_process(run: RunState) -> None:
    pid = run.current_pid
    if not pid:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(pid, 15)
    except ProcessLookupError:
        pass
    except Exception as exc:
        append_log(run, f"WARN: could not terminate process group cleanly: {exc}")


def run_subprocess(run: RunState, cmd: list[str], label: str) -> None:
    if run.cancel_requested:
        raise RuntimeError("Run cancelled by user.")

    append_log(run, f"$ {' '.join(cmd)}")

    creationflags = 0
    popen_kwargs = {
        "cwd": str(BASE_DIR),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(cmd, **popen_kwargs)
    run.current_pid = proc.pid

    captured: list[str] = []
    try:
        assert proc.stdout is not None
        while True:
            if run.cancel_requested:
                cancel_run_process(run)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise RuntimeError("Run cancelled by user.")

            line = proc.stdout.readline()
            if line:
                captured.append(line.rstrip())
            elif proc.poll() is not None:
                break

        remainder = proc.stdout.read()
        if remainder:
            captured.extend(remainder.splitlines())

        if captured:
            append_log(run, "\n".join(captured))

        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {proc.returncode}.")
    finally:
        run.current_pid = None
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


def build_zip(run: RunState) -> str:
    zip_path = Path(run.work_dir) / "valiantwrapped_results.zip"
    docs_dir = Path(run.docs_dir)
    covers_dir = Path(run.album_covers_dir)
    social_dir = Path(run.social_cards_dir)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if docs_dir.exists():
            for p in docs_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(Path("docs") / p.relative_to(docs_dir)))
        if covers_dir.exists():
            for p in covers_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(Path("outputs") / "album_covers" / p.relative_to(covers_dir)))
        if social_dir.exists():
            for p in social_dir.rglob("*.png"):
                zf.write(p, arcname=str(Path("outputs") / "social_cards" / p.relative_to(social_dir)))
    run.zip_path = str(zip_path)
    return str(zip_path)


def collect_result_pages(run: RunState) -> list[dict]:
    author_dir = Path(run.docs_dir) / "authors"
    rows: list[dict] = []
    if not author_dir.exists():
        return rows

    for html_path in sorted(author_dir.glob("*.html")):
        author_label = html_path.stem
        first, last, scopus_id, display_name = display_name_from_label(author_label)
        rows.append(
            {
                "author_label": author_label,
                "first_name": first,
                "last_name": last,
                "scopus_id": scopus_id,
                "display_name": display_name,
                "page_url": f"/runs/{run.run_id}/docs/authors/{html_path.name}",
            }
        )
    run.result_pages = rows
    return rows


def read_persona_fields(run: RunState, author_label: str) -> tuple[str, str]:
    persona_path = Path(run.persona_dir) / f"{author_label}.txt"
    artist_name = ""
    album_title = ""
    if not persona_path.exists():
        return artist_name, album_title
    try:
        text = persona_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return artist_name, album_title
    for line in text.splitlines():
        if not artist_name:
            m = re.match(r"^\s*Artist\s*:\s*(.+)$", line, flags=re.I)
            if m:
                artist_name = m.group(1).strip()
        if not album_title:
            m = re.match(r"^\s*Album\s*:\s*(.+)$", line, flags=re.I)
            if m:
                album_title = m.group(1).strip()
    return artist_name, album_title


def read_summary_stats(run: RunState, author_label: str) -> dict[str, str]:
    summary_path = Path(run.summary_dir) / f"{author_label}.txt"
    stats = {"publications": "", "citations": "", "top_journal": ""}
    if not summary_path.exists():
        return stats
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "publications": [r"^\s*Publications\s*\([^\)]*\)\s*:\s*(.+)$", r"^\s*Publications\s*:\s*(.+)$"],
        "citations": [r"^\s*Citations\s*\([^\)]*\)\s*:\s*(.+)$", r"^\s*Citations\s*:\s*(.+)$"],
        "top_journal": [r"^\s*Top\s+Journal\s*:\s*(.+)$"],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text, flags=re.I | re.M)
            if m:
                stats[key] = m.group(1).strip()
                break
    return stats


def rounded_image(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, img.size[0], img.size[1]), radius=radius, fill=255)
    out = Image.new("RGBA", img.size)
    out.paste(img, (0, 0), mask)
    return out


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ])
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if len(lines) == max_lines and draw.textlength(lines[-1], font=font) > max_width:
        while lines[-1] and draw.textlength(lines[-1] + "…", font=font) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines


def generate_social_card(run: RunState, author_label: str) -> Path:
    Path(run.social_cards_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(run.social_cards_dir) / f"{author_label}.png"
    if out_path.exists():
        return out_path

    first, last, scopus_id, display_name = display_name_from_label(author_label)
    stats = read_summary_stats(run, author_label)
    artist_name, album_title = read_persona_fields(run, author_label)

    canvas = Image.new("RGBA", (1200, 630), (10, 16, 32, 255))
    draw = ImageDraw.Draw(canvas)

    # Background gradients / shapes
    for i in range(630):
        blend = i / 629
        r = int(11 * (1 - blend) + 23 * blend)
        g = int(16 * (1 - blend) + 28 * blend)
        b = int(32 * (1 - blend) + 52 * blend)
        draw.line((0, i, 1200, i), fill=(r, g, b, 255))

    draw.ellipse((780, -90, 1220, 330), fill=(124, 58, 237, 90))
    draw.ellipse((880, 250, 1280, 640), fill=(6, 182, 212, 70))
    draw.rounded_rectangle((36, 36, 1164, 594), radius=34, outline=(255, 255, 255, 36), width=1)

    title_font = load_font(54, bold=True)
    subtitle_font = load_font(22, bold=False)
    meta_font = load_font(18, bold=True)
    body_font = load_font(32, bold=True)
    small_font = load_font(18, bold=False)
    kicker_font = load_font(20, bold=True)

    draw.text((52, 48), "VALIANT WRAPPED", font=kicker_font, fill=(210, 220, 240, 230))
    draw.text((52, 82), display_name, font=title_font, fill=(255, 255, 255, 255))
    draw.text((52, 148), "Your research, reimagined as an AI-generated music persona.", font=subtitle_font, fill=(217, 226, 242, 230))

    # Album cover
    cover_path = None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = Path(run.album_covers_dir) / f"{author_label}{ext}"
        if candidate.exists():
            cover_path = candidate
            break
    if cover_path:
        try:
            cover = Image.open(cover_path).convert("RGBA").resize((330, 330))
            cover = rounded_image(cover, 26)
            canvas.alpha_composite(cover, (58, 226))
        except Exception:
            draw.rounded_rectangle((58, 226, 388, 556), radius=26, fill=(255, 255, 255, 25), outline=(255, 255, 255, 60))
            draw.text((96, 376), "Album art\nunavailable", font=body_font, fill=(235, 241, 255, 220))
    else:
        draw.rounded_rectangle((58, 226, 388, 556), radius=26, fill=(255, 255, 255, 25), outline=(255, 255, 255, 60))
        draw.text((96, 376), "Album art\nunavailable", font=body_font, fill=(235, 241, 255, 220))

    # Text block
    x0 = 430
    y0 = 230
    draw.text((x0, y0), artist_name or "Fictional artist persona", font=body_font, fill=(255, 255, 255, 255))
    if album_title:
        draw.text((x0, y0 + 46), f"Album: {album_title}", font=subtitle_font, fill=(197, 179, 88, 255))

    stat_box_y = y0 + 110
    stat_fill = (255, 255, 255, 20)
    stat_outline = (255, 255, 255, 40)
    boxes = [
        (x0, stat_box_y, x0 + 190, stat_box_y + 92, "Publications", stats.get("publications") or "—"),
        (x0 + 206, stat_box_y, x0 + 396, stat_box_y + 92, "Citations", stats.get("citations") or "—"),
    ]
    for left, top, right, bottom, label, value in boxes:
        draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=stat_fill, outline=stat_outline)
        draw.text((left + 16, top + 14), label, font=small_font, fill=(201, 210, 228, 230))
        draw.text((left + 16, top + 42), str(value), font=body_font, fill=(255, 255, 255, 255))

    journal = stats.get("top_journal") or ""
    journal_lines = wrap_text(draw, f"Top Journal: {journal}" if journal else "Top Journal: —", subtitle_font, 660, 2)
    y = stat_box_y + 128
    for line in journal_lines:
        draw.text((x0, y), line, font=subtitle_font, fill=(227, 233, 246, 235))
        y += 28

    footer = f"Scopus ID: {scopus_id}" if scopus_id else "Generated with VALIANT Wrapped"
    draw.text((x0, 542), footer, font=small_font, fill=(185, 196, 218, 220))

    canvas.convert("RGB").save(out_path, format="PNG")
    return out_path


def execute_pipeline(run: RunState, uploaded_files: list[Path]) -> None:
    python_exe = os.environ.get("PYTHON", os.sys.executable)

    try:
        run.status = "running"
        run.mode_effective = "single" if len(uploaded_files) == 1 and run.mode_requested != "batch" else "batch"
        if run.mode_requested == "single":
            run.mode_effective = "single"
        if run.mode_requested == "batch":
            run.mode_effective = "batch"

        if run.mode_effective == "single":
            if not (run.first_name and run.last_name and run.scopus_id):
                raise RuntimeError("First name, last name, and Scopus ID are required for single-author runs.")
            author_label = run.primary_author_label or build_author_label(run.first_name, run.last_name, run.scopus_id)
            input_flag = ["--input-file", str(uploaded_files[0])]
            run.author_labels = [author_label]
        else:
            input_flag = ["--input-dir", run.uploads_dir]
            author_label = ""
            run.author_labels = [Path(p).stem for p in uploaded_files]

        set_step(run, 1, "Preparing uploads")
        append_log(run, f"Run ID: {run.run_id}")
        append_log(run, f"Requested mode: {run.mode_requested}")
        append_log(run, f"Effective mode: {run.mode_effective}")
        append_log(run, f"Uploaded files: {len(uploaded_files)}")
        if run.mode_effective == "single":
            append_log(run, f"Canonical author label: {author_label}")

        set_step(run, 2, "Step 1/5 · Reading Scopus CSVs")
        scopus_cmd = [
            python_exe,
            "scopus2txtsummary.py",
            *input_flag,
            "--output-dir",
            run.summary_dir,
            "--year-cutoff",
            "2025",
        ]
        if run.mode_effective == "single":
            scopus_cmd.extend([
                "--first-name", run.first_name,
                "--last-name", run.last_name,
                "--scopus-id", run.scopus_id,
            ])
        run_subprocess(run, scopus_cmd, "scopus2txtsummary.py")

        set_step(run, 3, "Step 2/5 · Generating expertise summaries")
        expertise_input = (["--input-file", str(Path(run.summary_dir) / f"{author_label}.txt")]
                           if run.mode_effective == "single"
                           else ["--input-dir", run.summary_dir])
        run_subprocess(
            run,
            [python_exe, "author_expertise_llama31_2.py", *expertise_input, "--output-dir", run.expertise_dir],
            "author_expertise_llama31_2.py",
        )

        set_step(run, 4, "Step 3/5 · Creating music personas")
        persona_input = (["--input-file", str(Path(run.expertise_dir) / f"{author_label}.txt")]
                         if run.mode_effective == "single"
                         else ["--input-dir", run.expertise_dir])
        run_subprocess(
            run,
            [python_exe, "author_persona_llama31.py", *persona_input, "--output-dir", run.persona_dir],
            "author_persona_llama31.py",
        )

        set_step(run, 5, "Step 4/5 · Generating album covers")
        cover_input = (["--input-file", str(Path(run.persona_dir) / f"{author_label}.txt")]
                       if run.mode_effective == "single"
                       else ["--input-dir", run.persona_dir])
        run_subprocess(
            run,
            [python_exe, "generate_album_covers.py", *cover_input, "--output-dir", run.album_covers_dir],
            "generate_album_covers.py",
        )

        set_step(run, 6, "Step 5/5 · Building HTML pages")
        site_cmd = [
            python_exe,
            "generate_valiantwrapped_site_noindex.py",
            "--summary-dir", run.summary_dir,
            "--persona-dir", run.persona_dir,
            "--album-covers-dir", run.album_covers_dir,
            "--docs-dir", run.docs_dir,
        ]
        if run.mode_effective == "single":
            site_cmd.extend(["--author-label", author_label])
        run_subprocess(run, site_cmd, "generate_valiantwrapped_site_noindex.py")

        collect_result_pages(run)
        build_zip(run)

        if run.cancel_requested:
            run.status = "cancelled"
            run.current_step = "Run stopped."
            append_log(run, "Run cancelled by user.")
        else:
            run.status = "success"
            finalize_progress(run)
            append_log(run, "Pipeline finished successfully.")

    except Exception as e:
        if run.cancel_requested or "cancelled by user" in str(e).lower():
            run.status = "cancelled"
            run.current_step = "Run stopped."
            append_log(run, f"CANCELLED: {e}")
        else:
            run.status = "error"
            run.error = str(e)
            append_log(run, f"ERROR: {e}")
    finally:
        run.current_pid = None


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/api/run", methods=["POST"])
def api_run():
    uploaded = request.files.getlist("files")
    mode_requested = (request.form.get("run_mode") or "auto").strip().lower()
    first_name = (request.form.get("first_name") or "").strip()
    last_name = (request.form.get("last_name") or "").strip()
    scopus_id = (request.form.get("scopus_id") or "").strip()

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

    primary_author_label = ""
    if mode_requested == "single":
        if not first_name or not last_name or not scopus_id:
            return jsonify({"ok": False, "error": "First name, last name, and Scopus ID are required for single-author runs."}), 400
        try:
            primary_author_label = build_author_label(first_name, last_name, scopus_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    run_id = uuid.uuid4().hex[:12]
    work_dir = RUNS_ROOT / run_id
    uploads_dir = work_dir / "uploads"
    outputs_dir = work_dir / "outputs"
    summary_dir = outputs_dir / "summary_txt"
    expertise_dir = outputs_dir / "expertise_txt"
    persona_dir = outputs_dir / "personas_txt"
    album_covers_dir = outputs_dir / "album_covers"
    social_cards_dir = outputs_dir / "social_cards"
    docs_dir = work_dir / "docs"

    for p in [uploads_dir, summary_dir, expertise_dir, persona_dir, album_covers_dir, social_cards_dir, docs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for idx, f in enumerate(files, start=1):
        filename = secure_filename(f.filename)
        if not filename:
            continue
        if mode_requested == "single":
            filename = f"{primary_author_label}.csv"
        elif (uploads_dir / filename).exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{idx}{suffix}"
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
        social_cards_dir=str(social_cards_dir),
        first_name=first_name,
        last_name=last_name,
        scopus_id=scopus_id,
        primary_author_label=primary_author_label,
    )

    with RUNS_LOCK:
        RUNS[run_id] = run

    t = threading.Thread(target=execute_pipeline, args=(run, saved_paths), daemon=True)
    t.start()

    return jsonify({"ok": True, "run_id": run_id, "mode_requested": mode_requested})


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


@app.route("/api/cancel/<run_id>", methods=["POST"])
def api_cancel(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"ok": False, "error": "Run not found."}), 404
    if run.status not in {"queued", "running"}:
        return jsonify({"ok": False, "error": f"Run is already {run.status}."}), 400
    run.cancel_requested = True
    append_log(run, "Cancellation requested by user.")
    cancel_run_process(run)
    return jsonify({"ok": True, "status": "cancelling"})


@app.route("/api/social-card/<run_id>/<path:author_label>", methods=["POST"])
def api_social_card(run_id: str, author_label: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"ok": False, "error": "Run not found."}), 404
    if run.status != "success":
        return jsonify({"ok": False, "error": "Social cards are only available after a successful run."}), 400

    author_label = Path(author_label).name
    known = {row["author_label"] for row in run.result_pages}
    if author_label not in known:
        return jsonify({"ok": False, "error": "Author result not found for this run."}), 404

    try:
        out_path = generate_social_card(run, author_label)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Could not generate social card: {exc}"}), 500

    return jsonify({
        "ok": True,
        "download_url": f"/api/download-social-card/{run_id}/{author_label}",
        "path": str(out_path),
    })


@app.route("/api/download-social-card/<run_id>/<path:author_label>")
def api_download_social_card(run_id: str, author_label: str):
    run = RUNS.get(run_id)
    if not run:
        abort(404)
    author_label = Path(author_label).name
    path = Path(run.social_cards_dir) / f"{author_label}.png"
    if not path.exists():
        try:
            path = generate_social_card(run, author_label)
        except Exception:
            abort(404)
    return send_file(path, as_attachment=True)


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
            [python_exe, "build_author_url_manifest.py", "--authors-dir", str(docs_authors), "--output-file", str(manifest_path)],
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
            raise RuntimeError(f"Manifest generation failed with exit code {proc.returncode}.")
        run.manifest_path = str(manifest_path)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "download_url": f"/api/download-manifest/{run.run_id}"})


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
