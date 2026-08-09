#!/usr/bin/env python3
"""Conventional MLP fine-tuning, replay, and joint-training baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from datasets import STREAMS  # noqa: E402
from run_suite import DEFAULTS, metrics_from_matrix  # noqa: E402


def make_mlp(seed: int, input_dim: int):
    hidden = 256 if input_dim <= 64 else 384
    return MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=2e-3,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        warm_start=False,
    )


def update_buffer(seen_x, seen_y, budget: int, seed: int):
    x = np.concatenate(seen_x)
    y = np.concatenate(seen_y)
    if len(x) <= budget:
        return x, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    indices = []
    for cls in classes:
        candidates = np.flatnonzero(y == cls)
        quota = max(1, budget // len(classes))
        indices.extend(rng.choice(candidates, min(quota, len(candidates)), replace=False))
    indices = np.asarray(indices[:budget], dtype=int)
    return x[indices], y[indices]


def train_epochs(model, x, y, classes, epochs):
    for epoch in range(epochs):
        if epoch == 0 and not hasattr(model, "classes_"):
            model.partial_fit(x, y, classes=classes)
        else:
            model.partial_fit(x, y)


def run(stream_name: str, method: str, seed: int, epochs: int, buffer_size: int):
    stream = STREAMS[stream_name](seed, cap=DEFAULTS[stream_name]["cap"])
    classes = np.arange(stream.classes)
    model = make_mlp(seed, stream.train[0][0].shape[1])
    seen_x, seen_y = [], []
    matrix = np.full((len(stream.train), len(stream.train)), np.nan)
    max_drift = 0.0
    start = time.perf_counter()
    for task_id, (x_new, y_new) in enumerate(stream.train):
        old_x = np.concatenate(seen_x) if seen_x else None
        before = model.predict_proba(old_x) if old_x is not None else None
        seen_x.append(x_new)
        seen_y.append(y_new)
        if method == "mlp_joint":
            model = make_mlp(seed * 101 + task_id, x_new.shape[1])
            train_epochs(model, np.concatenate(seen_x), np.concatenate(seen_y), classes, epochs)
        elif method == "mlp_replay":
            bx, by = update_buffer(seen_x[:-1], seen_y[:-1], buffer_size, seed * 101 + task_id) if task_id else (np.empty((0, x_new.shape[1])), np.empty(0, dtype=int))
            tx = np.concatenate((x_new, bx))
            ty = np.concatenate((y_new, by))
            train_epochs(model, tx, ty, classes, epochs)
        else:
            train_epochs(model, x_new, y_new, classes, epochs)
        if old_x is not None:
            max_drift = max(max_drift, float(np.max(np.abs(model.predict_proba(old_x) - before))))
        for eval_id in range(task_id + 1):
            xe, ye = stream.test[eval_id]
            matrix[task_id, eval_id] = np.mean(model.predict(xe) == ye)
    elapsed = time.perf_counter() - start
    parameter_bytes = int(sum(x.nbytes for x in model.coefs_) + sum(x.nbytes for x in model.intercepts_))
    return {
        "stream": stream_name,
        "method": method,
        "seed": seed,
        "tasks": len(stream.train),
        "train_examples": int(sum(len(x) for x, _ in stream.train)),
        "elapsed_s": elapsed,
        "parameter_bytes": parameter_bytes,
        "max_anchor_drift": max_drift,
        "accuracy_matrix": matrix.tolist(),
        "diagnostics": [],
        "config": {"epochs_per_task": epochs, "buffer_size": buffer_size},
        **metrics_from_matrix(matrix),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", choices=sorted(STREAMS), default=sorted(STREAMS))
    parser.add_argument("--methods", nargs="+", choices=("mlp_ft", "mlp_replay", "mlp_joint"), default=("mlp_ft", "mlp_replay", "mlp_joint"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--buffer-size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/mlp.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    for stream in args.streams:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for method in args.methods:
                result = run(stream, method, seed, args.epochs, args.buffer_size)
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
                print(json.dumps({
                    "stream": stream,
                    "seed": seed,
                    "method": method,
                    "final_acc": round(result["final_average_accuracy"], 5),
                    "forget": round(result["mean_forgetting"], 5),
                    "drift": f"{result['max_anchor_drift']:.2e}",
                    "seconds": round(result["elapsed_s"], 3),
                }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

