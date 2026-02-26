#!/usr/bin/env python3
"""Gradio UI for VALIANT Wrapped with blocking run + live logs.

Run:
    . .venv/bin/activate
    python gradio_wait_gui.py
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from typing import Generator

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "author_csvs"
OUTPUT_CSV = ROOT / "author_expertise_summaries.csv"
SCRIPT = ROOT / "author_expertise_llama31.py"


def save_uploaded(files: list[gr.File] | None) -> str:
    if not files:
        return "No files selected."
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        src = Path(f.name)
        if src.suffix.lower() != ".csv":
            continue
        dst = INPUT_DIR / src.name
        dst.write_bytes(src.read_bytes())
        saved += 1
    return f"Saved {saved} CSV file(s) into {INPUT_DIR}."


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
    model_id: str,
    max_authors: int,
    max_rows_per_author: int,
    max_tokens_per_chunk: int,
    map_tokens: int,
    reduce_tokens: int,
) -> Generator[tuple[str, str, pd.DataFrame], None, None]:
    env = os.environ.copy()
    env["MODEL_ID"] = model_id.strip()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["MAX_AUTHORS"] = str(max_authors)
    env["MAX_ROWS_PER_AUTHOR"] = str(max_rows_per_author)
    env["MAX_INPUT_TOKENS_PER_CHUNK"] = str(max_tokens_per_chunk)
    env["MAP_MAX_NEW_TOKENS"] = str(map_tokens)
    env["REDUCE_MAX_NEW_TOKENS"] = str(reduce_tokens)

    cmd = ["python", str(SCRIPT)]
    log_text = "[run] Starting...\n"
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
    yield status, log_text, load_output_df()


with gr.Blocks(title="VALIANT Wrapped - Gradio Runner") as demo:
    gr.Markdown("# VALIANT Wrapped Runner (Gradio)")
    gr.Markdown("Run waits for completion and updates output table when finished.")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            run_btn = gr.Button("Run Llama Pipeline", variant="primary")
            refresh_btn = gr.Button("Refresh")
            status = gr.Textbox(label="Run status", value="Idle", interactive=False)
            stats_box = gr.Textbox(label="Input stats", value=input_stats(), interactive=False)

            with gr.Accordion("Run Settings", open=True):
                model_id = gr.Textbox(label="Model ID", value=os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))
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
                label="Output Preview",
                value=load_output_df(),
                wrap=True,
                interactive=False,
                max_height=700,
            )

    upload_btn.click(fn=save_uploaded, inputs=[upload], outputs=[upload_msg]).then(
        fn=input_stats, outputs=[stats_box]
    ).then(fn=load_output_df, outputs=[out_df])

    refresh_btn.click(fn=input_stats, outputs=[stats_box]).then(fn=load_output_df, outputs=[out_df])

    run_btn.click(
        fn=run_pipeline,
        inputs=[model_id, max_authors, max_rows, chunk_toks, map_toks, reduce_toks],
        outputs=[status, logs, out_df],
    ).then(fn=input_stats, outputs=[stats_box])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
