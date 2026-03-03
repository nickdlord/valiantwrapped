#!/usr/bin/env python3
"""
Gradio UI for VALIANT Wrapped (LLAMA + metrics) with blocking run + live logs.

Adds:
- Runs author_scopusmetrics.py to produce metrics CSV
- Runs author_expertise_llama31.py to produce LLM summaries CSV
- Displays a merged table (summary + metrics)

Run:
    source .venv/Scripts/activate
    python gradio_llama_gui.py
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Generator

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "author_csvs"

# LLM output (from author_expertise_llama31.py)
LLM_OUTPUT_CSV = ROOT / "author_expertise_summaries.csv"
LLM_SCRIPT = ROOT / "author_expertise_llama31.py"

# Metrics output (from author_scopusmetrics.py)
# IMPORTANT: set this to whatever your metrics script writes.
METRICS_OUTPUT_CSV = ROOT / "author_summary_2025_present.csv"
METRICS_SCRIPT = ROOT / "author_scopusmetrics.py"

LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# ----------------------------
# File handling
# ----------------------------
def save_uploaded(files: list[gr.File] | None) -> str:
    if not files:
        return "No files selected."
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    for f in files:
        src = Path(f.name)
        if src.suffix.lower() != ".csv":
            skipped += 1
            continue
        dst = INPUT_DIR / src.name
        dst.write_bytes(src.read_bytes())
        saved += 1
    msg = f"Saved {saved} CSV file(s) into {INPUT_DIR}."
    if skipped:
        msg += f" Skipped {skipped} non-CSV file(s)."
    return msg


def input_stats() -> str:
    files = sorted(INPUT_DIR.glob("*.csv"))
    rows = 0
    for p in files:
        try:
            with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                r = csv.reader(fh)
                next(r, None)
                for _ in r:
                    rows += 1
        except Exception:
            pass
    return f"Input files: {len(files)} | Total rows: {rows}"


# ----------------------------
# Output loading + merging
# ----------------------------
def load_llm_df() -> pd.DataFrame:
    if not LLM_OUTPUT_CSV.exists():
        return pd.DataFrame(columns=["author_id", "author_file", "summary"])
    try:
        df = pd.read_csv(LLM_OUTPUT_CSV)
        # normalize expected cols
        for c in ["author_id", "author_file", "summary"]:
            if c not in df.columns:
                df[c] = ""
        return df[["author_id", "author_file", "summary"]]
    except Exception:
        return pd.DataFrame(columns=["author_id", "author_file", "summary"])


def load_metrics_df() -> pd.DataFrame:
    """
    Expects metrics file from author_scopusmetrics.py, typically:
    author_file, author_id, pub_count_2025_present, citation_count_2025_present,
    top_journal_2025_present, top_paper_title_2025_present, top_paper_citations_2025_present
    """
    if not METRICS_OUTPUT_CSV.exists():
        return pd.DataFrame(columns=[
            "author_file",
            "author_id",
            "pub_count_2025_present",
            "citation_count_2025_present",
            "top_journal_2025_present",
            "top_paper_title_2025_present",
            "top_paper_citations_2025_present",
        ])
    try:
        df = pd.read_csv(METRICS_OUTPUT_CSV)
        # Ensure expected columns exist; fill missing with blanks/0
        expected = [
            "author_file",
            "author_id",
            "pub_count_2025_present",
            "citation_count_2025_present",
            "top_journal_2025_present",
            "top_paper_title_2025_present",
            "top_paper_citations_2025_present",
        ]
        for c in expected:
            if c not in df.columns:
                df[c] = 0 if "count" in c or "citations" in c else ""
        return df[expected]
    except Exception:
        return pd.DataFrame(columns=[
            "author_file",
            "author_id",
            "pub_count_2025_present",
            "citation_count_2025_present",
            "top_journal_2025_present",
            "top_paper_title_2025_present",
            "top_paper_citations_2025_present",
        ])


def load_combined_df() -> pd.DataFrame:
    llm = load_llm_df()
    metrics = load_metrics_df()

    # Merge strategy:
    # Prefer joining on author_id (most stable). Fall back on author_file if needed.
    if not llm.empty and not metrics.empty:
        if "author_id" in llm.columns and "author_id" in metrics.columns:
            out = pd.merge(metrics, llm, on="author_id", how="outer", suffixes=("_metrics", "_llm"))
            # If both have author_file, pick the non-empty one
            if "author_file_metrics" in out.columns and "author_file_llm" in out.columns:
                out["author_file"] = out["author_file_metrics"].fillna("").astype(str)
                mask = out["author_file"].str.strip().eq("")
                out.loc[mask, "author_file"] = out.loc[mask, "author_file_llm"].fillna("").astype(str)
                out = out.drop(columns=["author_file_metrics", "author_file_llm"])
        else:
            # fallback: merge on author_file if author_id missing
            out = pd.merge(metrics, llm, on="author_file", how="outer")
    elif not metrics.empty:
        out = metrics.copy()
        out["summary"] = ""
    else:
        out = llm.copy()
        # add metrics columns for consistent display
        out["pub_count_2025_present"] = 0
        out["citation_count_2025_present"] = 0
        out["top_journal_2025_present"] = ""
        out["top_paper_title_2025_present"] = ""
        out["top_paper_citations_2025_present"] = 0

    # nice column order for UI
    cols = [
        "author_id",
        "author_file",
        "pub_count_2025_present",
        "citation_count_2025_present",
        "top_journal_2025_present",
        "top_paper_title_2025_present",
        "top_paper_citations_2025_present",
        "summary",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = "" if "top_" in c or c in ["author_file", "summary"] else 0

    return out[cols]


# ----------------------------
# Subprocess runner helper
# ----------------------------
def _run_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict,
    log_text: str,
    refresh_df_fn,
) -> Generator[tuple[str, str, pd.DataFrame], None, int]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_text += line
        yield "Running...", log_text, refresh_df_fn()
    rc = proc.wait()
    log_text += f"\n[run] completed with exit code {rc}\n"
    yield ("Done" if rc == 0 else f"Failed (exit code {rc})"), log_text, refresh_df_fn()
    return rc


# ----------------------------
# Pipeline runner (metrics + llama)
# ----------------------------
def run_pipeline(
    hf_token: str,
    max_authors: int,
    max_rows_per_author: int,
    max_tokens_per_chunk: int,
    map_tokens: int,
    reduce_tokens: int,
) -> Generator[tuple[str, str, pd.DataFrame], None, None]:
    """
    Runs:
      1) Metrics script (author_scopusmetrics.py)
      2) Llama summary script (author_expertise_llama31.py)
    Then displays merged results.
    """

    # --- Step 0: basic input validation
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        yield "Failed", f"No input CSVs found in {INPUT_DIR}. Upload/save author CSVs first.\n", load_combined_df()
        return

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_text = "[run] Starting...\n"
    log_text += f"[run] Using interpreter: {sys.executable}\n"
    yield "Running...", log_text, load_combined_df()

    # --- Step 1: run metrics (no token required)
    log_text += "\n[step 1/2] Running metrics: author_scopusmetrics.py\n"
    yield "Running...", log_text, load_combined_df()

    metrics_cmd = [sys.executable, str(METRICS_SCRIPT)]
    proc = subprocess.Popen(
        metrics_cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        log_text += line
        yield "Running...", log_text, load_combined_df()

    metrics_rc = proc.wait()
    log_text += f"\n[metrics] completed with exit code {metrics_rc}\n"
    if metrics_rc != 0:
        yield f"Failed (metrics exit {metrics_rc})", log_text, load_combined_df()
        return

    # --- Step 2: run llama summaries (token gating retained)
    token = (hf_token or "").strip()
    if not token:
        log_text += (
            "\n[step 2/2] Llama summaries require an HF Token for gated repos.\n"
            "Paste a READ token (hf_...) into the HF Token box.\n"
        )
        yield "Failed (no token)", log_text, load_combined_df()
        return

    env["MODEL_ID"] = LLAMA_MODEL_ID
    env["USE_HF_TOKEN"] = "1"
    env["HF_TOKEN"] = token
    env["HUGGINGFACE_HUB_TOKEN"] = token

    # Pipeline knobs for your LLM script (kept as-is)
    env["MAX_AUTHORS"] = str(int(max_authors))
    env["MAX_ROWS_PER_AUTHOR"] = str(int(max_rows_per_author))
    env["MAX_INPUT_TOKENS_PER_CHUNK"] = str(int(max_tokens_per_chunk))
    env["MAP_MAX_NEW_TOKENS"] = str(int(map_tokens))
    env["REDUCE_MAX_NEW_TOKENS"] = str(int(reduce_tokens))

    log_text += "\n[step 2/2] Running Llama summaries: author_expertise_llama31.py\n"
    log_text += f"[run] MODEL_ID={env['MODEL_ID']}\n"
    log_text += "[run] USE_HF_TOKEN=1 (HF_TOKEN provided; hidden)\n"
    yield "Running...", log_text, load_combined_df()

    llama_cmd = [sys.executable, str(LLM_SCRIPT)]
    proc = subprocess.Popen(
        llama_cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        log_text += line
        yield "Running...", log_text, load_combined_df()

    llama_rc = proc.wait()
    log_text += f"\n[llama] completed with exit code {llama_rc}\n"

    if llama_rc != 0 and ("401" in log_text or "gated repo" in log_text.lower()):
        log_text += (
            "\n[hint] Still seeing 401 / gated repo?\n"
            "  1) Confirm your HF account has access to the model (request/accept access on the model page).\n"
            "  2) Confirm your token is valid and has READ permissions.\n"
            "  3) Confirm your pipeline uses token=True when USE_HF_TOKEN=1.\n"
        )

    status = "Done" if llama_rc == 0 else f"Failed (llama exit {llama_rc})"
    yield status, log_text, load_combined_df()


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title="VALIANT Wrapped - Llama + Metrics Runner (Gradio)") as demo:
    gr.Markdown("# VALIANT Wrapped Runner (Llama 3.1 Instruct + Scopus Metrics)")
    gr.Markdown(
        f"**Model (forced):** `{LLAMA_MODEL_ID}`\n\n"
        "This runner executes:\n"
        "1) `author_scopusmetrics.py` (metrics)\n"
        "2) `author_expertise_llama31.py` (LLM summaries)\n\n"
        "Paste your Hugging Face token below (READ scope is enough). "
        "This model is gated, so you also need approved access on your HF account."
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            run_btn = gr.Button("Run Metrics + Llama Pipeline", variant="primary")
            refresh_btn = gr.Button("Refresh")
            status = gr.Textbox(label="Run status", value="Idle", interactive=False)
            stats_box = gr.Textbox(label="Input stats", value=input_stats(), interactive=False)

            with gr.Accordion("Auth (Required for Llama Step)", open=True):
                hf_token = gr.Textbox(
                    label="HF Token",
                    value="",
                    type="password",
                    placeholder="hf_********************************",
                    info="Required for gated Meta-Llama models on Hugging Face.",
                )

            with gr.Accordion("Run Settings (LLM Step)", open=True):
                max_authors = gr.Number(label="MAX_AUTHORS (0=all)", value=1, precision=0)
                max_rows = gr.Number(label="MAX_ROWS_PER_AUTHOR", value=80, precision=0)
                chunk_toks = gr.Number(label="MAX_INPUT_TOKENS_PER_CHUNK", value=2500, precision=0)
                map_toks = gr.Number(label="MAP_MAX_NEW_TOKENS", value=96, precision=0)
                reduce_toks = gr.Number(label="REDUCE_MAX_NEW_TOKENS", value=128, precision=0)

            with gr.Accordion("Input Files", open=False):
                upload = gr.File(label="Upload CSVs", file_count="multiple", file_types=[".csv"])
                upload_btn = gr.Button("Save CSVs")
                upload_msg = gr.Textbox(label="Upload status", value="", interactive=False)

            with gr.Accordion("Logs", open=False):
                logs = gr.Textbox(label="Live logs", value="", lines=20, interactive=False)

        with gr.Column(scale=3):
            out_df = gr.Dataframe(
                label="Output Preview (Metrics + Summary)",
                value=load_combined_df(),
                wrap=True,
                interactive=False,
                max_height=700,
            )

    upload_btn.click(fn=save_uploaded, inputs=[upload], outputs=[upload_msg]).then(
        fn=input_stats, outputs=[stats_box]
    ).then(fn=load_combined_df, outputs=[out_df])

    refresh_btn.click(fn=input_stats, outputs=[stats_box]).then(
        fn=load_combined_df, outputs=[out_df]
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[hf_token, max_authors, max_rows, chunk_toks, map_toks, reduce_toks],
        outputs=[status, logs, out_df],
    ).then(fn=input_stats, outputs=[stats_box])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
