#!/usr/bin/env python3
"""
run_valiantwrapped_pipeline.py

One command runner that chains the TXT-based pipeline end-to-end and optionally
deletes intermediate TXT folders after they are no longer needed.

Pipeline:
1) scopus2txtsummary.py        CSV  -> summary TXT
2) author_expertise_llama31_2.py  summary TXT -> expertise TXT
3) author_persona_llama31.py   expertise TXT -> persona TXT
4) generate_album_covers.py    persona TXT -> PNG covers

Example:
  python run_valiantwrapped_pipeline.py \
    --input-file author_csvs/Landman_Bennett_16679175200.csv \
    --work-dir /tmp/valiantwrapped_run \
    --cleanup-intermediate
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)

def remove_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
        print(f"Deleted: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="")
    ap.add_argument("--input-dir", default="")
    ap.add_argument("--work-dir", default="outputs/pipeline_run")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--year-cutoff", type=int, default=2025)
    ap.add_argument("--keep-summary-txt", action="store_true")
    ap.add_argument("--keep-expertise-txt", action="store_true")
    ap.add_argument("--keep-persona-txt", action="store_true")
    ap.add_argument("--cleanup-intermediate", action="store_true")
    args = ap.parse_args()

    if bool(args.input_file) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --input-file or --input-dir")

    root = Path(__file__).resolve().parent
    work = Path(args.work_dir)
    summary_dir = work / "summary_txt"
    expertise_dir = work / "expertise_txt"
    persona_dir = work / "persona_txt"
    cover_dir = work / "album_covers"
    work.mkdir(parents=True, exist_ok=True)

    scopus_cmd = [args.python_bin, str(root / "scopus2txtsummary.py"), "--output-dir", str(summary_dir), "--year-cutoff", str(args.year_cutoff)]
    expertise_cmd = [args.python_bin, str(root / "author_expertise_llama31_2.py"), "--output-dir", str(expertise_dir), "--output-csv", ""]
    persona_cmd = [args.python_bin, str(root / "author_persona_llama31.py"), "--output-dir", str(persona_dir), "--output-csv", ""]
    covers_cmd = [args.python_bin, str(root / "generate_album_covers.py"), "--output-dir", str(cover_dir)]

    if args.input_file:
        scopus_cmd += ["--input-file", args.input_file]
    else:
        scopus_cmd += ["--input-dir", args.input_dir]

    expertise_cmd += ["--input-dir", str(summary_dir)]
    persona_cmd += ["--input-dir", str(expertise_dir)]
    covers_cmd += ["--input-dir", str(persona_dir)]

    run_cmd(scopus_cmd)
    run_cmd(expertise_cmd)
    run_cmd(persona_cmd)
    run_cmd(covers_cmd)

    if args.cleanup_intermediate:
        if not args.keep_summary_txt:
            remove_dir(summary_dir)
        if not args.keep_expertise_txt:
            remove_dir(expertise_dir)
        if not args.keep_persona_txt:
            remove_dir(persona_dir)

    print("\nPipeline completed successfully.")
    print(f"Final album covers: {cover_dir}")

if __name__ == "__main__":
    main()
