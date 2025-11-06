# Korean OCR

A collection of scripts and utilities to run an OCR pipeline tailored for Korean e-commerce product images, plus evaluation tools to compare OCR outputs against provided ground-truth text. The project was developed from a competition-style dataset (Chung Innovation Challenge) and includes a robust evaluation script that handles layout differences (line merges/splits/reordering).

## Table of contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Highlight](#highlight)
- [Evaluation](#evaluation)
- [Pipeline](#pipeline)
- [Workflow](#workflow)
- [Files](#files)
- [Technologies and techniques](#technologies-and-techniques)
- [Known limitations & notes](#known-limitations--notes)

## Overview

This repository provides a practical OCR pipeline optimized for photographic and complex-layout images containing Korean text. It focuses on preprocessing, chunked OCR with overlap and deduplication, plus a robust evaluation suite that reports character- and line-level metrics.

Key capabilities:
- Chunked OCR with overlapping windows to handle long/tall images
- Adaptive preprocessing (denoise, upscale, contrast adjust, sharpen)
- Deduplication across chunk overlaps using similarity + confidence heuristics
- Pairwise and batch evaluation tools with aggregated metrics

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the OCR pipeline (from project root):

```bash
# From project root (macOS / zsh)
python3 ocr_main.py
```

Run evaluation (pairwise runner + summary):

```bash
python3 ocr_eval_recursive.py
python3 ocr_eval_metrics.py
```

Quick single-file evaluation (development):

```bash
python3 ocr_eval_20250903.py ORIGINAL_TEXT/texts_hyecho/TAF20161_00.txt.text submission/texts_hyecho/TAF20161_00.txt -v
```

Notes:
- `easyocr` downloads models on first run. For GPU acceleration install a matching PyTorch build; the reader will fall back to CPU if GPU init fails.
- Configuration options for input/output folders, thresholds, and debug flags live in `OCRConfig` inside `ocr_main.py`.

## Highlight

- Practical pipeline tuned for Korean product images.
- Overlap-based chunking + conservative deduplication to reduce duplicated lines.
- Configurable preprocessing pipeline (text-size-aware upscaling, contrast heuristics).
- Evaluation tooling for per-file and batch metrics (CER, line-level precision/recall/F1, balanced score).

## Evaluation

`ocr_eval_metrics.py` parses `evaluation_results.txt` and computes micro-averaged character-level CER and line-level precision/recall/F1 and a final balanced score defined as sqrt(Char Accuracy * Line F1).

Example aggregated summary (from a sample run)

```text
📊 Overall OCR Evaluation Summary
=====================================
Blocks parsed:                17

Character-level (micro)
	GT chars:                   5751
	OCR chars:                  5669
	Total edit errors:          1458
	CER:                        25.35%
	Accuracy:                   74.65%
	Precision (est):            84.67%
	Recall (est):               83.46%
	F1 (est):                   84.06%

Line-level (micro)
	GT lines:                   317
	OCR lines:                  470
	Aligned/matched lines:      284
	Precision:                  60.43%
	Recall:                     89.59%
	F1:                         72.17%
-------------------------------------
🏆 Overall Score (Balanced):     73.40
	(Calculated as sqrt(Char Accuracy * Line F1))
-------------------------------------
```

## Pipeline

Principal scripts and roles:

- `ocr_main.py` — OCR pipeline.
- `ocr_eval_20250903.py` — Robust evaluation for a single GT/OCR pair (normalization, approximate substring matching, multi-line matching, error scoring).
- `ocr_eval_recursive.py` — Parallel runner that executes the single-file evaluator across many file pairs and aggregates outputs into `evaluation_results.txt`.
- `ocr_eval_metrics.py` — Reads `evaluation_results.txt` and produces a human-readable overall summary.

High-level flow inside `ocr_main.py`:

1. Read batch (traverse input image folders)
2. Open each image file
3. Chunk image vertically into overlapping windows
4. Preprocess each chunk (resize, denoise, upscale, contrast adjust, sharpen)
5. Run OCR (EasyOCR) on preprocessed chunk
6. Map detected bounding boxes back to original coordinates and group detections into logical lines (by Y-coordinate)
7. Deduplicate overlapping lines across adjacent chunks using similarity + confidence heuristics
8. Flatten grouped lines and save output to `submission/`

## Workflow

1. Prepare dataset
   - Place images in `OFFICIAL_TEST/` under dataset sub-folders (e.g., `images_hyecho`, `images_lfmall1`, ...).
   - Ground-truth text files live in `ORIGINAL_TEXT/` under corresponding `texts_*` folders.

2. Run OCR pipeline
   - The main entrypoint for OCR is `ocr_main.py`. It processes images from `OCRConfig.input_dir` (default `OFFICIAL_TEST`) and writes `.txt` text files under `submission/` (default `OCRConfig.output_dir`).

3. Evaluate OCR outputs
   - `ocr_eval_recursive.py` pairs GT files (from `ORIGINAL_TEXT/`) with OCR outputs (from `submission/`) and invokes `ocr_eval_20250903.py` for each pair while collecting outputs into `evaluation_results.txt`.
   - `ocr_eval_metrics.py` reads `evaluation_results.txt` and computes aggregated character- and line-level statistics plus the final balanced score.

## Files

Top-level files and purpose:

- `ocr_main.py` — main OCR pipeline and configuration (`OCRConfig`).
- `ocr_eval_20250903.py` — single-pair evaluation utility.
- `ocr_eval_recursive.py` — parallel evaluation runner.
- `ocr_eval_metrics.py` — metrics aggregator and summary reporter.
- `requirements.txt` — Python dependencies.
- `OFFICIAL_TEST/`, `ORIGINAL_IMAGE/`, `ORIGINAL_TEXT/` — example dataset folders (images and GT text).
- `submission/` — default output folder for OCR results.
- `otherVersion/debug_chunks/` — debug images (when enabled).

## Technologies and techniques

- Python 3
- OpenCV (`cv2`) for image I/O and preprocessing (thresholding, resizing, filtering, morphology)
- EasyOCR as the OCR engine (multi-language reader; Korean + English by default)
- NumPy for numeric image operations
- `difflib.SequenceMatcher` for line-similarity checks used in deduplication
- A custom matching algorithm in `ocr_eval_20250903.py` using approximate substring matching, greedy selection and bounded backtracking to find optimal line matches and minimize a total error score
- `concurrent.futures` (ThreadPool / ProcessPool) used by `ocr_eval_recursive.py` to run evaluations in parallel and aggregate results

## Known limitations & notes

- EasyOCR performance depends heavily on the environment (GPU vs CPU). `OCRConfig.use_gpu` controls GPU usage; the reader will fall back to CPU if GPU init fails.
- The evaluation ground truth provided in `ORIGINAL_TEXT/` was generated with an LLM (Gemini 2.5 Pro) and may contain errors; treat results as relative rather than absolute.
- Spelling correction or language-model-based postprocessing is not included by default; a spellfixer could be added as an optional postprocessing step.
- The deduplication heuristics (SequenceMatcher + confidence) are conservative and tuned for overlapping chunking. You may need to adjust `overlap` and `lcs_threshold` for different datasets.


*Last Updated: October 2025*
