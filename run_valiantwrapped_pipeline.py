#!/usr/bin/env python3
"""
run_valiantwrapped_pipeline.py

Run the full TXT-based VALIANT Wrapped pipeline in either:
- single mode
- batch mode

Pipeline:
1. scopus2txtsummary.py          (CSV -> summary TXT)
2. author_expertise_llama31_2.py (summary TXT -> expertise TXT)
3. author_persona_llama31.py     (expertise TXT -> persona TXT)
4. generate_album_covers.py      (persona TXT -> album cover image)
5. generate_valiantwrapped_site_noindex.py (TXT + images -> HTML)

Optional:
- --cleanup-intermediate removes intermediate TXT folders after final outputs are generated
"""

import argparse
import os
import shutil
import subprocess
import sys


def run(cmd):
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 80 + "\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def author_label_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def cleanup_folder(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"í·¹ Deleted intermediate folder: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "batch"], required=True)

    ap.add_argument("--input-file", default="", help="Single Scopus CSV input file")
    ap.add_argument("--input-dir", default="", help="Folder of Scopus CSV input files")

    ap.add_argument("--year-cutoff", type=int, default=2025)

    ap.add_argument("--summary-dir", default="outputs/summary_txt")
    ap.add_argument("--expertise-dir", default="outputs/expertise_txt")
    ap.add_argument("--persona-dir", default="outputs/personas_txt")
    ap.add_argument("--album-covers-dir", default="outputs/album_covers")
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--base-url", default="")

    ap.add_argument("--text-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--image-model", default="black-forest-labs/FLUX.1-dev")

    ap.add_argument(
        "--cleanup-intermediate",
        action="store_true",
        help="Delete intermediate TXT output folders after final outputs are generated",
    )

    args = ap.parse_args()
    py = sys.executable

    if args.mode == "single":
        if not args.input_file:
            raise ValueError("--input-file is required for single mode")
        if not args.input_file.lower().endswith(".csv"):
            raise ValueError("--input-file must be a Scopus CSV file")
        author_label = author_label_from_path(args.input_file)
        scopus_input_flag = ["--input-file", args.input_file]
        summary_input_flag = ["--input-file", os.path.join(args.summary_dir, f"{author_label}.txt")]
        expertise_input_flag = ["--input-file", os.path.join(args.expertise_dir, f"{author_label}.txt")]
        persona_input_flag = ["--input-file", os.path.join(args.persona_dir, f"{author_label}.txt")]
    else:
        if not args.input_dir:
            raise ValueError("--input-dir is required for batch mode")
        scopus_input_flag = ["--input-dir", args.input_dir]
        summary_input_flag = ["--input-dir", args.summary_dir]
        expertise_input_flag = ["--input-dir", args.expertise_dir]
        persona_input_flag = ["--input-dir", args.persona_dir]
        author_label = ""

    # Step 1: CSV -> summary TXT
    run([
        py, "scopus2txtsummary.py",
        *scopus_input_flag,
        "--output-dir", args.summary_dir,
        "--year-cutoff", str(args.year_cutoff),
    ])

    # Step 2: summary TXT -> expertise TXT
    run([
        py, "author_expertise_llama31_2.py",
        *summary_input_flag,
        "--output-dir", args.expertise_dir,
        "--model-id", args.text_model,
    ])

    # Step 3: expertise TXT -> persona TXT
    run([
        py, "author_persona_llama31.py",
        *expertise_input_flag,
        "--output-dir", args.persona_dir,
        "--model-id", args.text_model,
    ])

    # Step 4: persona TXT -> album covers
    run([
        py, "generate_album_covers.py",
        *persona_input_flag,
        "--output-dir", args.album_covers_dir,
        "--llm-model", args.text_model,
        "--image-model", args.image_model,
    ])

    # Step 5: build HTML pages
    site_cmd = [
        py, "generate_valiantwrapped_site_noindex.py",
        "--summary-dir", args.summary_dir,
        "--persona-dir", args.persona_dir,
        "--album-covers-dir", args.album_covers_dir,
        "--docs-dir", args.docs_dir,
    ]
    if args.base_url:
        site_cmd.extend(["--base-url", args.base_url])
    if args.mode == "single":
        site_cmd.extend(["--author-label", author_label])

    run(site_cmd)

    # Optional cleanup
    if args.cleanup_intermediate:
        cleanup_folder(args.summary_dir)
        cleanup_folder(args.expertise_dir)
        cleanup_folder(args.persona_dir)

    print("\nâœ… Pipeline complete.")


if __name__ == "__main__":
    main()
