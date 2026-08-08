#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

cd "$project_root"

"$python_bin" experiments/run_suite.py \
  --streams split_digits rotated_digits permuted_digits corrupted_digits shape_stream \
  --methods vanish unwrapped functional_l2 replay param_null joint_refit \
  --seeds 20 --overwrite --output results/raw/suite_20seed.jsonl

"$python_bin" experiments/run_mlp_baselines.py \
  --streams split_digits rotated_digits permuted_digits corrupted_digits shape_stream \
  --methods mlp_ft mlp_replay mlp_joint --seeds 10 --epochs 35 --buffer-size 200 \
  --overwrite --output results/raw/mlp_10seed.jsonl

"$python_bin" experiments/stress_tests.py \
  --capacity-seeds 20 --output results/raw/stress_20seed.json

"$python_bin" experiments/run_suite.py \
  --streams shape_stream --methods vanish --seeds 20 --width 4096 --gamma 0.004 \
  --overwrite --output results/raw/shape_capacity_20seed.jsonl

"$python_bin" -m unittest discover -s tests -v
"$python_bin" experiments/summarize.py
MPLBACKEND=Agg "$python_bin" experiments/make_figures.py

if command -v latexmk >/dev/null 2>&1; then
  (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
fi

