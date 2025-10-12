# Korean OCR (kaggle-ocr-korean)

A collection of scripts and utilities used to run an OCR pipeline targeted at Korean e-commerce images and to evaluate OCR outputs against provided ground-truth text. The project was developed from a competition-style dataset (Chung Innovation Challenge) and an provided evaluation tool (`ocr_eval_20250903.py`) that robustly compares OCR output to ground truth while handling layout differences (line merges/splits/reordering).

## Project topic and goals

- Topic: Korean OCR for product images collected from multiple e-commerce websites.
- Goal: Provide a practical OCR pipeline tailored to Korean text in photographic and complex layout images, plus a robust evaluation tool to measure character- and line-level accuracy while being tolerant to layout inconsistencies.

This repo contains:
- `ocr_main.py` — OCR pipeline (preprocessing, chunking, deduplication, EasyOCR integration, batch runner).
- `ocr_eval_20250903.py` — Robust OCR evaluation script (normalization, approximate substring matching, multi-line matching, and error scoring).
- `ocr_eval_recursive.py` — Parallel runner that executes `ocr_eval_20250903.py` across many file pairs and aggregates results into `evaluation_results.txt`.
- `ocr_eval_metrics.py` — Summarizes `evaluation_results.txt` into a human-readable overall evaluation summary.
- `OFFICIAL_TEST/`, `ORIGINAL_IMAGE/`, `ORIGINAL_TEXT/` — Example dataset folders (images and GT text).

> Note: Large/generated files and folders listed in `.gitignore` were intentionally skipped during analysis.

## Requirements & Quick start

Install minimal dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- `easyocr` requires additional model downloads on first run. If you plan to use GPU acceleration, install PyTorch with the appropriate CUDA support (`pip install torch torchvision` from the official install instructions). The scripts will fall back to CPU if GPU init fails.
- This `requirements.txt` lists the main runtime dependencies used across the scripts (EasyOCR, OpenCV, NumPy). Add other tools as needed.

## How to run

1. OCR pipeline (single-machine):

```bash
# From project root (macOS / zsh)
python3 ocr_main.py
```

- Edit `OCRConfig` inside `ocr_main.py` to change input/output folders, debug flags, or thresholds.
- Debug images are saved under `otherVersion/debug_chunks/` when enabled.

2. Evaluation (pairwise + aggregate):

```bash
# Run parallel evaluation runner (this will call ocr_eval_20250903.py for each GT/OCR pair)
python3 ocr_eval_recursive.py

# Summarize results
python3 ocr_eval_metrics.py
```

3. Quick single-file evaluation (for development):

```bash
python3 ocr_eval_20250903.py ORIGINAL_TEXT/texts_hyecho/TAF20161_00.txt.text submission/texts_hyecho/TAF20161_00.txt -v
```

## Workflow

1. Prepare dataset
	- Place images in `OFFICIAL_TEST/` under dataset sub-folders (e.g., `images_hyecho`, `images_lfmall1`, ...).
	- Ground-truth text files live in `ORIGINAL_TEXT/` under corresponding `texts_*` folders.

2. Run OCR pipeline
	- The main entrypoint for OCR is `ocr_main.py`. It processes images from `OCRConfig.input_dir` (default `OFFICIAL_TEST`) and writes `.txt` text files under `submission/` (default `OCRConfig.output_dir`).

3. Evaluate OCR outputs
	- `ocr_eval_recursive.py` is the parallel runner: it pairs GT files (from `ORIGINAL_TEXT/`) with OCR outputs (from `submission/`) and invokes `ocr_eval_20250903.py` for each pair while collecting each run's output into `evaluation_results.txt`.
	- `ocr_eval_metrics.py` reads `evaluation_results.txt` (produced by the recursive runner) and computes the aggregated character-level and line-level statistics and the final balanced score.

## `ocr_main.py` workflow

1. Read batch (traverse input image folders)
2. Open each image file
3. Chunk image vertically into overlapping windows
4. Preprocess each chunk (resize, denoise, upscale, contrast adjust, sharpen)
5. Run OCR (EasyOCR) on preprocessed chunk
6. Map detected bounding boxes back to original coordinates and group detections into logical lines (by Y-coordinate)
7. Deduplicate overlapping lines across adjacent chunks using similarity + confidence heuristics
8. Flatten grouped lines and save output to `submission/`



## Detailed `ocr_main.py` workflow (step-by-step)

`ocr_main.py` is implemented with modular classes and runs as follows:

1. OCRConfig (dataclass)
	- Central configuration object controlling directories, preprocessing parameters, chunking behavior, deduplication thresholds, OCR languages and thresholds, debugging flags, and other heuristics.

2. BatchProcessor
	- Entrypoint for batch processing. It:
	  - Clears the `submission` output folder (unless you change the code).
	  - Optionally clears debug image folder when `enable_debug_images=True`.
	  - Initializes the `OCRProcessor` (which in turn initializes EasyOCR Reader).
	  - Walks the `input_dir` recursively, finds image files, and calls `_process_single_file` for each one.
	  - `_process_single_file` maps image folder names from `images_*` to `texts_*` to create matching output paths and saves OCR results as one text line per detected logical token.

3. OCRProcessor
	- Manages the actual reading of images, preprocessing, chunking, OCR calls, and line-level deduplication.

	- initialize_reader():
	  - Initializes EasyOCR Reader with configured languages and GPU preference. Falls back to CPU mode if GPU initialization fails.

	- process_image(img_path):
	  - Reads the image in grayscale.
	  - Splits the image vertically into overlapping chunks using `_generate_chunks`. The default chunk height is 900 px and overlap is 15 px.
	  - For each chunk:
		 - Calls ImagePreprocessor.preprocess to run the preprocessing pipeline.
		 - Runs `reader.readtext` (EasyOCR) on the preprocessed chunk.
		 - Optionally writes debug images with detection boxes if `enable_debug_images` is set.
		 - If deduplication is disabled (`enable_deduplication=False`), each detection above a confidence threshold becomes its own output line.
		 - If deduplication is enabled, the processor:
			- Maps bounding boxes from processed chunk coordinates back to original coordinates, computes an absolute Y coordinate for each detection, and groups detections by approximate Y into logical lines using `_group_detections_into_lines`.
			- For the first line in a new chunk that falls within the top overlapping region, performs deduplication against the last logical line in the previous chunk using an LCS-like SequenceMatcher-based similarity and average-confidence heuristics in `_deduplicate_with_lcs`.
			- Deduplication strategy returns one of: keep both lines, remove the previous chunk's last line, or remove the current chunk's first line based on similarity and confidence.
		 - The processor accumulates logical lines (each a list of text tokens) across chunks.
	  - After processing chunks, the processor flattens grouped lines into a single list of text lines and returns it.

4. ImagePreprocessor
	- Responsible for per-chunk image transformations: validating image, optional resizing for very large images, noise reduction (bilateral filter), upscaling small text (adaptive using estimated smallest text height), contrast adjustments based on image brightness/contrast statistics, and sharpening.
	- Uses ImageAnalyzer to calculate `mean`, `std`, `bright_ratio`, `dark_ratio` and to estimate text heights via morphological connected components on thresholded images.
	- Contrast heuristics select alpha multipliers tailored for bright/dark/low-contrast images to maximize legibility for OCR.

5. Output
	- The flattened list of extracted text lines is saved to a `.txt` file in `submission/` preserving the relative folder structure (mapping `images_*` -> `texts_*`).

6. Tips & configuration
	- Tune `OCRConfig` parameters at the top of `ocr_main.py` for different datasets (chunk_height, overlap, confidence_threshold, enable_deduplication, etc.).
	- Enable `enable_debug_images` to save visual debug outputs per chunk which helps diagnose bounding boxes and dedupe issues.

## Technologies and techniques used

- Python 3 (scripts are written as single-file CLI utilities)
- OpenCV (cv2) for image I/O and preprocessing (thresholding, resizing, filtering, morphology)
- EasyOCR as the OCR engine (multi-language reader; Korean and English config used by default)
- NumPy for numeric image operations
- difflib.SequenceMatcher for lightweight line-similarity checks used in deduplication
- A custom matching algorithm in `ocr_eval_20250903.py` using approximate substring matching, greedy selection and bounded backtracking to find optimal line matches and minimize a total error score
- Multiprocessing/ThreadPool (via `concurrent.futures`) used by `ocr_eval_recursive.py` to run evaluations in parallel and aggregate results

## Evaluation: output format

- `ocr_eval_metrics.py` parses `evaluation_results.txt` and computes micro-averaged character-level CER and line-level precision/recall/F1 and a final balanced score defined as sqrt(Char Accuracy * Line F1).

Example aggregated summary (from a sample run)

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

## Known limitations & notes

- EasyOCR performance depends heavily on the environment (GPU vs CPU). `OCRConfig.use_gpu` controls GPU usage; the reader will fall back to CPU if GPU init fails.
- The evaluation ground truth provided in `ORIGINAL_TEXT/` was generated with an LLM (Gemini 2.5 Pro) and may contain errors; treat results as relative rather than absolute.
- Spelling correction or language-model-based postprocessing is not included by default; a spellfixer could be added as an optional postprocessing step.
- The deduplication heuristics (SequenceMatcher + confidence) are conservative and tuned for overlapping chunking. You may need to adjust `overlap` and `lcs_threshold` for different datasets.


## 👤 Author

**Truong Trung Bao (Brian)**  
Now: Computer Science @ HCMUT-VNU → Software Engineering Honours @ UQ  

---

## 📜 License

Educational purposes - Free to use with attribution

---

*Last Updated: October 2025*