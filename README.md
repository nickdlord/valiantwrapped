# VALIANT Wrapped

VALIANT Wrapped is a prototype pipeline for turning Scopus publication exports into concise, website-ready summaries of researcher expertise.

Given per-author Scopus CSV exports, the project:

* Generates **1–2 paragraph expertise blurbs** per author using **Meta-Llama-3.1-8B-Instruct**
* Optionally outputs **per-author text files** for easy copy/paste
* Produces a simple **2025–present publication/citation summary** per author (counts + top journal + top cited paper)

This is designed to support scalable, automated “faculty/researcher profile” generation for a research center.

---

## What’s in this repo

### Key Scripts

**`author_expertise_llama31.py`**
Reads per-author Scopus CSVs and generates a 120–220 word expertise description per author using a **map-reduce summarization** approach (chunk → theme summaries → final synthesis).

**`author_summary_2025_present.py`**
Reads per-author Scopus CSVs and calculates simple 2025–present metrics:

* Publication count
* Total citations
* Top journal (by publications; tie-breaker by citations)
* Top paper title + citations (by citations; tie-breaker by year)

---

## Expected Folder Structure

```
VALIANT-Wrapped/
├── author_csvs/                      # Input: per-author Scopus CSVs (usually NOT committed)
├── author_expertise_llama31.py       # LLM expertise summarizer
├── author_summary_2025_present.py    # 2025+ metrics summarizer
├── author_expertise_txt/             # Output: per-author summaries (auto-created)
├── author_expertise_summaries.csv    # Output: 1–2 paragraph summaries
├── author_summary_2025_present.csv   # Output: 2025+ metrics summary
└── README.md
```

> Tip: If `author_csvs/` contains sensitive or large data, add it to `.gitignore`.

---

## Requirements

* Python 3.9+ recommended
* GPU recommended for Llama 3.1 inference
* Libraries:

  * `pandas`
  * `torch`
  * `transformers`
  * `accelerate` (recommended when using `device_map="auto"`)

---

## Setup

### 1. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -U pip
pip install pandas torch transformers accelerate
```

> On HPC/GPU clusters (e.g., ACCRE), use the environment-specific PyTorch install recommended by the cluster if needed.

---

## Run the Pipeline

### A) Generate Expertise Summaries (LLM)

This script reads all CSVs in `author_csvs/` and outputs:

* `author_expertise_summaries.csv`
* `author_expertise_txt/<author_id>.txt` (optional)

```bash
python author_expertise_llama31.py
```

### Key Configuration Settings (Inside Script)

* `INPUT_DIR = "author_csvs"`
* `OUTPUT_CSV = "author_expertise_summaries.csv"`
* `OUTPUT_TXT_DIR = "author_expertise_txt"`
* `MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"`
* `MAX_PAPERS_PER_AUTHOR = 250`
* `MAX_INPUT_TOKENS = 6000`

### Output Format

* 1–2 paragraphs
* Professional tone
* Plain-language
* Grounded only in publication evidence (no invented facts)

---

### B) Generate 2025–Present Metrics Summary

This script reads all CSVs in `author_csvs/` and outputs:

* `author_summary_2025_present.csv`

```bash
python author_summary_2025_present.py
```

### Key Configuration Settings (Inside Script)

* `INPUT_DIR = "author_csvs"`
* `YEAR_CUTOFF = 2025`
* `OUTPUT_SUMMARY = "author_summary_2025_present.csv"`

---

## Notes on Data & Privacy

Scopus exports may contain large text fields (abstracts) and/or data you may not want to publish publicly.

Recommended approach:

* Keep raw inputs in `author_csvs/`
* Add `author_csvs/` and large outputs to `.gitignore`
* Commit only scripts, documentation, and small example outputs (if permitted)

---

## Roadmap / Next Steps

* Add keyword/theme extraction and ranking
* Add cross-author topic clustering
* Build a lightweight web interface to browse summaries
* Add reproducible `requirements.txt`
* Add a small demo dataset (if permitted)

---

## Credits

Built as part of the VALIANT Discovery Center “Wrapped”-style research summary prototype.

---

If you’d like, I can also:

* Add a **professional GitHub badge section**
* Add a short **Architecture Overview diagram**
* Tighten this into a more “AI product portfolio” style README**
* Or rewrite it to sound more like a polished research software project**

Just tell me which direction you want this repo to signal professionally.

