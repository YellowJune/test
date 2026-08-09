# Reproducibility protocol

1. Install `requirements.txt` in a fresh Python 3.10+ environment.
2. Set `OPENBLAS_NUM_THREADS=4` to match the timing environment.
3. Run `bash scripts/run_all.sh` from the project root.
4. Confirm that four unit tests pass.
5. Confirm row counts: 600 in `suite_20seed.jsonl`, 150 in `mlp_10seed.jsonl`, 20 in `shape_capacity_20seed.jsonl`, and 200 capacity rows inside `stress_20seed.json`.
6. Compare generated `results/summary/*.csv` and `figures/*.pdf` with the packaged files.
7. Build `paper/main.tex` with `latexmk -pdf`; `pdfinfo` must report at least 12 pages.

Randomness is fully seeded. The Digits dataset ships with scikit-learn; Shape Stream is generated deterministically. No internet, GPU, pretrained model, or external checkpoint is required.

The JSONL writer appends each run immediately, so a preempted job retains completed records. Use `--overwrite` only when intentionally starting a clean suite.
