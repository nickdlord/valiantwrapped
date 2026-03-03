#!/usr/bin/env python3
"""
Gradio UI for VALIANT Wrapped (LLAMA-only) with blocking run + live logs.

Key changes vs your old GUI:
- Forces MODEL_ID to Meta Llama 3.1 Instruct (no Qwen option)
- Adds an HF Token field and REQUIRED gating-aware behavior
- Uses sys.executable for the subprocess (prevents VS Code running system Python)
- Passes HF_TOKEN + USE_HF_TOKEN=1 into the subprocess env

IMPORTANT:
1) You must have access approved for the gated repo on Hugging Face:
   meta-llama/Meta-Llama-3.1-8B-Instruct
2) Your pipeline (author_expertise_llama31.py) must load MODEL_ID from env
   and use token=True when USE_HF_TOKEN=1. See patch at bottom.

Run (Windows Git Bash):
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
OUTPUT_CSV = ROOT / "author_expertise_summaries.csv"
SCRIPT = ROOT / "author_expertise_llama31.py"

LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


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


def load_output_df() -> pd.DataFrame:
    if not OUTPUT_CSV.exists():
        return pd.DataFrame(columns=["author_id", "author_file", "summary"])
    try:
        return pd.read_csv(OUTPUT_CSV)
    except Exception:
        return pd.DataFrame(columns=["author_id", "author_file", "summary"])


def run_pipeline(
    hf_token: str,
    max_authors: int,
    max_rows_per_author: int,
    max_tokens_per_chunk: int,
    map_tokens: int,
    reduce_tokens: int,
) -> Generator[tuple[str, str, pd.DataFrame], None, None]:
    """
    Runs the pipeline as a subprocess and streams stdout to the UI.

    HF token is REQUIRED for gated Llama repos. If empty, we fail fast with
    a helpful message instead of a long traceback.
    """
    token = (hf_token or "").strip()
    if not token:
        msg = (
            "Missing HF Token.\n\n"
            "This model is gated on Hugging Face. Paste a READ token (starts with `hf_...`) "
            "into the HF Token box, and make sure your HF account has been granted access "
            "to Meta Llama 3.1 Instruct.\n"
        )
        yield "Failed (no token)", msg, load_output_df()
        return

    env = os.environ.copy()

    # Force LLAMA
    env["MODEL_ID"] = LLAMA_MODEL_ID
    env["USE_HF_TOKEN"] = "1"
    env["HF_TOKEN"] = token
    env["HUGGINGFACE_HUB_TOKEN"] = token  # common alt var

    # Pipeline knobs
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["MAX_AUTHORS"] = str(int(max_authors))
    env["MAX_ROWS_PER_AUTHOR"] = str(int(max_rows_per_author))
    env["MAX_INPUT_TOKENS_PER_CHUNK"] = str(int(max_tokens_per_chunk))
    env["MAP_MAX_NEW_TOKENS"] = str(int(map_tokens))
    env["REDUCE_MAX_NEW_TOKENS"] = str(int(reduce_tokens))

    # Use the *current* interpreter (the one running Gradio)
    cmd = [sys.executable, str(SCRIPT)]

    log_text = "[run] Starting...\n"
    log_text += f"[run] Using interpreter: {sys.executable}\n"
    log_text += f"[run] MODEL_ID={env['MODEL_ID']}\n"
    log_text += "[run] USE_HF_TOKEN=1 (HF_TOKEN provided; hidden)\n"
    yield "Running...", log_text, load_output_df()

    proc = subprocess.Popen(
        cmd,
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
        yield "Running...", log_text, load_output_df()

    rc = proc.wait()
    status = "Done" if rc == 0 else f"Failed (exit code {rc})"
    log_text += f"\n[run] completed with exit code {rc}\n"

    if rc != 0 and ("401" in log_text or "gated repo" in log_text.lower()):
        log_text += (
            "\n[hint] Still seeing 401 / gated repo?\n"
            "  1) Confirm your HF account has access to the model (request/accept access on the model page).\n"
            "  2) Confirm your token is valid and has READ permissions.\n"
            "  3) Confirm your pipeline uses token=True when USE_HF_TOKEN=1.\n"
        )

    yield status, log_text, load_output_df()


with gr.Blocks(title="VALIANT Wrapped - Llama Runner (Gradio)") as demo:
    gr.Markdown("# VALIANT Wrapped Runner (Llama 3.1 Instruct)")
    gr.Markdown(
        f"**Model (forced):** `{LLAMA_MODEL_ID}`\n\n"
        "Paste your Hugging Face token below (READ scope is enough). "
        "This model is gated, so you also need approved access on your HF account."
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            run_btn = gr.Button("Run Llama Pipeline", variant="primary")
            refresh_btn = gr.Button("Refresh")
            status = gr.Textbox(label="Run status",
                                value="Idle", interactive=False)
            stats_box = gr.Textbox(label="Input stats",
                                   value=input_stats(), interactive=False)

            with gr.Accordion("Auth (Required)", open=True):
                hf_token = gr.Textbox(
                    label="HF Token",
                    value="",
                    type="password",
                    placeholder="hf_********************************",
                    info="Required for gated Meta-Llama models on Hugging Face.",
                )

            with gr.Accordion("Run Settings", open=True):
                max_authors = gr.Number(
                    label="MAX_AUTHORS (0=all)", value=1, precision=0)
                max_rows = gr.Number(
                    label="MAX_ROWS_PER_AUTHOR", value=80, precision=0)
                chunk_toks = gr.Number(
                    label="MAX_INPUT_TOKENS_PER_CHUNK", value=2500, precision=0)
                map_toks = gr.Number(
                    label="MAP_MAX_NEW_TOKENS", value=96, precision=0)
                reduce_toks = gr.Number(
                    label="REDUCE_MAX_NEW_TOKENS", value=128, precision=0)

            with gr.Accordion("Input Files", open=False):
                upload = gr.File(label="Upload CSVs",
                                 file_count="multiple", file_types=[".csv"])
                upload_btn = gr.Button("Save CSVs")
                upload_msg = gr.Textbox(
                    label="Upload status", value="", interactive=False)

            with gr.Accordion("Logs", open=False):
                logs = gr.Textbox(label="Live logs", value="",
                                  lines=20, interactive=False)

        with gr.Column(scale=3):
            out_df = gr.Dataframe(
                label="Output Preview",
                value=load_output_df(),
                wrap=True,
                interactive=False,
                max_height=700,
            )

    upload_btn.click(fn=save_uploaded, inputs=[upload], outputs=[upload_msg]).then(
        fn=input_stats, outputs=[stats_box]
    ).then(fn=load_output_df, outputs=[out_df])

    refresh_btn.click(fn=input_stats, outputs=[stats_box]).then(
        fn=load_output_df, outputs=[out_df])

    run_btn.click(
        fn=run_pipeline,
        inputs=[hf_token, max_authors, max_rows,
                chunk_toks, map_toks, reduce_toks],
        outputs=[status, logs, out_df],
    ).then(fn=input_stats, outputs=[stats_box])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
