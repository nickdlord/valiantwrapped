# VALIANT Wrapped

### Scopus CSV → Llama 3.1 → Research Expertise Summaries

VALIANT Wrapped is a prototype pipeline for transforming Scopus publication exports into concise, website-ready summaries of researcher expertise.

Given per-author Scopus CSV exports, the system:

* Generates **1–2 paragraph expertise blurbs** using `meta-llama/Meta-Llama-3.1-8B-Instruct`
* Outputs structured `.csv` and optional `.txt` summaries
* Produces optional **2025–present publication metrics**
* Runs locally on an ACCRE GPU node using `transformers`

This project supports scalable, automated research profile generation for the VALIANT Discovery Center and similar research environments.

---

# 📂 Project Structure

```id="merged-structure"
VALIANT-Wrapped/
│
├── author_csvs/                      # Per-author Scopus CSV files (not typically committed)
│     ├── 5719....csv
│     ├── 5720....csv
│
├── author_expertise_llama31.py       # Main LLM summarization script
├── author_summary_2025_present.py    # Optional 2025+ metrics summary
│
├── author_expertise_summaries.csv    # Output: 1–2 paragraph summaries
├── author_expertise_txt/             # Optional per-author text outputs
├── author_summary_2025_present.csv   # Output: 2025+ metrics
│
└── README.md
```

> Recommended: add `author_csvs/` to `.gitignore` if files are large or sensitive.

---

# ⚙️ What the LLM Script Does

For each author CSV:

1. Loads the file
2. Extracts structured metadata:

   * Title
   * Abstract
   * Journal
   * Year
   * Citation count
   * Keywords (if available)
3. Sorts papers by citation count (most informative first)
4. Caps extremely large author corpora
5. Uses a **map → reduce summarization strategy**
6. Writes:

   * `author_expertise_summaries.csv`
   * Optional `.txt` summary per author

No year filtering is applied for expertise summaries — the full publication record is used.

---

# 🧠 Why Map → Reduce?

Instead of sending an entire publication corpus into a single prompt (which can cause):

* Token overflow
* Hallucination
* Weak synthesis

The pipeline:

1. **Chunks publications**
2. **Extracts research themes per chunk (Map step)**
3. **Synthesizes chunk themes into a final expertise summary (Reduce step)**

This produces:

* More stable outputs
* Better thematic synthesis
* Lower hallucination risk
* Stronger performance for prolific authors

---

# 🧭 Workflow Summary

```
Scopus Export
    ↓
Split by Author
    ↓
Per-author CSVs
    ↓
author_expertise_llama31.py
    ↓
Llama 3.1 (Map → Reduce)
    ↓
1–2 Paragraph Expertise Summary
```

---

# 🧪 Model Used

```
meta-llama/Meta-Llama-3.1-8B-Instruct
```

Loaded via:

```python
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
```

The model is loaded once and reused across all authors for efficiency.

---

# 🖥 Running on ACCRE

## 1️⃣ Start an Interactive GPU Session

```bash
salloc --gres=gpu:1 --mem=64G --time=04:00:00
```

(or your preferred allocation command)

---

## 2️⃣ Activate Your Environment

```bash
conda activate huggingface
```

If needed:

```bash
pip install -U pandas transformers accelerate torch
```

---

## 3️⃣ (Recommended) Set Hugging Face Cache to Nobackup

To avoid filling your home directory:

```bash
mkdir -p /nobackup/$USER/hf
export HF_HOME=/nobackup/$USER/hf
```

---

## 4️⃣ Run the Script

```bash
python author_expertise_llama31.py
```

---

# 🧪 Local GUI Run Notes

If you have a local GUI script in your working directory (for example `gui_app.py`), run:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install pandas torch transformers accelerate
python gui_app.py
```

Typical local URL:

```text
http://127.0.0.1:8501
```

Recommended first test run (small workload):

```bash
export MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
export MAX_AUTHORS=1
export MAX_ROWS_PER_AUTHOR=80
export MAX_INPUT_TOKENS_PER_CHUNK=2500
export MAP_MAX_NEW_TOKENS=96
export REDUCE_MAX_NEW_TOKENS=128
```

---

# 🛠 Debug Notes (Hugging Face + GPU)

If model loading fails, check these common cases:

* `403 Forbidden` / `gated repo`:
  Use an approved token for Meta Llama or switch to an open model (for example `Qwen/Qwen2.5-7B-Instruct`).
* `Temporary failure in name resolution`:
  DNS/network issue to `huggingface.co`; retry after connectivity is restored.
* `CUDA out of memory`:
  Lower `MAX_*` settings and set:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Useful terminal checks:

```bash
hf auth login
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

---

# 🎛 Adjustable Settings (Inside Script)

You may tune:

```python
MAX_PAPERS_PER_AUTHOR = 250
MAX_INPUT_TOKENS = 6000
TEMPERATURE = 0.3
MAX_NEW_TOKENS_MAP = 220
MAX_NEW_TOKENS_REDUCE = 220
```

### If summaries feel:

* Too generic → increase `MAX_NEW_TOKENS_REDUCE`
* Too verbose → decrease `MAX_NEW_TOKENS_REDUCE`
* Too creative → lower `TEMPERATURE`
* Too slow → lower `MAX_PAPERS_PER_AUTHOR`

---

# 📊 Optional 2025–Present Metrics Script

The secondary script generates:

* Publication count (2025–present)
* Total citations (2025–present)
* Top journal (by publications, tie-break by citations)
* Top paper (by citations, tie-break by year)

Run:

```bash
python author_summary_2025_present.py
```

Outputs:

```
author_summary_2025_present.csv
```

---

# 📤 Output Files

## Main Expertise File

```
author_expertise_summaries.csv
```

Columns:

* `author_id`
* `author_file`
* `summary`

---

## Optional Text Folder

```
author_expertise_txt/
```

Contains:

```
5719....txt
5720....txt
...
```

Each file contains the final 1–2 paragraph expertise summary.

---

# 🧾 Requirements

* Python 3.9+
* GPU recommended
* Libraries:

  * pandas
  * torch
  * transformers
  * accelerate (recommended)

---

# 🔐 Data & Privacy Notes

Scopus exports may contain large text fields and institutional data.

Recommended practice:

* Keep raw CSVs out of version control
* Use `.gitignore` for:

  * `author_csvs/`
  * Large output files
  * Model cache directories

---

# 🚀 Future Enhancements

* Add structured JSON output for web ingestion
* Add all-time vs recent summaries
* Add keyword extraction layer before LLM
* Cross-author topic clustering
* Convert to CLI tool with arguments
* Add batch logging + progress tracking
* Generate persona visuals using image models (e.g., FLUX)
* Build lightweight web interface for browsing summaries

---

# 🎯 Project Context

Built as part of the **VALIANT Discovery Center “Wrapped”-style research profile prototype**, demonstrating:

* Research data parsing
* LLM integration
* Prompt engineering
* Map-reduce summarization
* GPU inference workflow
* Scalable academic profile automation
