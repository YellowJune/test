#!/usr/bin/env python3
"""Run seeded VANISH and matched functional baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from datasets import STREAMS  # noqa: E402
from vanish.core import AdditiveModel, RandomFeatureMap, fit_patch, one_hot, ridge_fit  # noqa: E402


DEFAULTS = {
    "split_digits": {"width": 768, "ridge": 1e-4, "gamma": 0.03, "cap": 220},
    "rotated_digits": {"width": 1024, "ridge": 1e-4, "gamma": 0.06, "cap": 180},
    "permuted_digits": {"width": 1024, "ridge": 1e-4, "gamma": 0.06, "cap": 180},
    "corrupted_digits": {"width": 1024, "ridge": 1e-4, "gamma": 0.03, "cap": 180},
    "shape_stream": {"width": 1024, "ridge": 1e-4, "gamma": 0.015, "cap": 240},
}


def metrics_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    tasks = matrix.shape[0]
    diagonal = np.diag(matrix)
    final = matrix[-1, :]
    old = slice(0, max(tasks - 1, 1))
    maxima = np.nanmax(matrix, axis=0)
    lower = matrix[np.tril_indices(tasks)]
    return {
        "final_average_accuracy": float(np.nanmean(final)),
        "average_incremental_accuracy": float(np.nanmean(lower)),
        "mean_new_task_accuracy": float(np.nanmean(diagonal)),
        "backward_transfer": float(np.nanmean(final[old] - diagonal[old])) if tasks > 1 else 0.0,
        "mean_forgetting": float(np.nanmean(maxima[old] - final[old])) if tasks > 1 else 0.0,
        "worst_forgetting": float(np.nanmax(maxima[old] - final[old])) if tasks > 1 else 0.0,
    }


class JointRefit:
    """Full-data oracle using the exact same cumulative random feature maps."""

    def __init__(self, classes: int, ridge: float):
        self.classes = classes
        self.ridge = ridge
        self.maps: list[RandomFeatureMap] = []
        self.coefficient: np.ndarray | None = None

    def add_map(self, fmap: RandomFeatureMap):
        self.maps.append(fmap)

    def features(self, x: np.ndarray):
        return np.concatenate([fmap(x) for fmap in self.maps], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray):
        phi = self.features(x)
        self.coefficient = ridge_fit(phi, one_hot(y, self.classes), self.ridge)

    def logits(self, x: np.ndarray):
        return self.features(x) @ self.coefficient

    def predict(self, x: np.ndarray):
        return np.argmax(self.logits(x), axis=1)

    def parameter_bytes(self):
        maps = sum(m.weight.nbytes + m.bias.nbytes for m in self.maps)
        return int(maps + (0 if self.coefficient is None else self.coefficient.nbytes))


def run_additive(stream_name: str, method: str, seed: int, config: dict) -> dict:
    stream = STREAMS[stream_name](seed, cap=config["cap"])
    model = AdditiveModel(stream.classes)
    protected: list[np.ndarray] = []
    matrix = np.full((len(stream.train), len(stream.train)), np.nan)
    diagnostics = []
    max_cumulative_drift = 0.0
    start = time.perf_counter()
    for task_id, (x_new, y_new) in enumerate(stream.train):
        old_x = np.concatenate(protected, axis=0) if protected else None
        before = model.logits(old_x) if old_x is not None else None
        diag = fit_patch(
            model,
            x_new,
            y_new,
            old_x,
            mode=method,
            width=config["width"],
            ridge=config["ridge"],
            gamma=config["gamma"],
            seed=seed * 10007 + task_id * 97 + 13,
            functional_weight=config.get("functional_weight", 10.0),
        )
        if old_x is not None:
            cumulative = float(np.max(np.abs(model.logits(old_x) - before)))
            max_cumulative_drift = max(max_cumulative_drift, cumulative)
        else:
            cumulative = 0.0
        diag["cumulative_anchor_drift"] = cumulative
        diagnostics.append(diag)
        protected.append(x_new)
        for eval_id in range(task_id + 1):
            x_eval, y_eval = stream.test[eval_id]
            matrix[task_id, eval_id] = np.mean(model.predict(x_eval) == y_eval)
    elapsed = time.perf_counter() - start
    result = {
        "stream": stream_name,
        "method": method,
        "seed": seed,
        "tasks": len(stream.train),
        "train_examples": int(sum(len(x) for x, _ in stream.train)),
        "elapsed_s": elapsed,
        "parameter_bytes": model.parameter_bytes(),
        "max_anchor_drift": max_cumulative_drift,
        "accuracy_matrix": matrix.tolist(),
        "diagnostics": diagnostics,
        "config": dict(config),
        **metrics_from_matrix(matrix),
    }
    return result


def run_joint(stream_name: str, seed: int, config: dict) -> dict:
    stream = STREAMS[stream_name](seed, cap=config["cap"])
    model = JointRefit(stream.classes, config["ridge"])
    seen_x, seen_y = [], []
    matrix = np.full((len(stream.train), len(stream.train)), np.nan)
    protected_x = None
    previous_logits = None
    max_drift = 0.0
    start = time.perf_counter()
    for task_id, (x_new, y_new) in enumerate(stream.train):
        if seen_x:
            protected_x = np.concatenate(seen_x, axis=0)
            previous_logits = model.logits(protected_x)
        model.add_map(RandomFeatureMap.create(x_new.shape[1], config["width"], seed * 10007 + task_id * 97 + 13))
        seen_x.append(x_new)
        seen_y.append(y_new)
        model.fit(np.concatenate(seen_x), np.concatenate(seen_y))
        if protected_x is not None:
            max_drift = max(max_drift, float(np.max(np.abs(model.logits(protected_x) - previous_logits))))
        for eval_id in range(task_id + 1):
            x_eval, y_eval = stream.test[eval_id]
            matrix[task_id, eval_id] = np.mean(model.predict(x_eval) == y_eval)
    elapsed = time.perf_counter() - start
    return {
        "stream": stream_name,
        "method": "joint_refit",
        "seed": seed,
        "tasks": len(stream.train),
        "train_examples": int(sum(len(x) for x, _ in stream.train)),
        "elapsed_s": elapsed,
        "parameter_bytes": model.parameter_bytes(),
        "max_anchor_drift": max_drift,
        "accuracy_matrix": matrix.tolist(),
        "diagnostics": [],
        "config": dict(config),
        **metrics_from_matrix(matrix),
    }


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", choices=sorted(STREAMS), default=sorted(STREAMS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("vanish", "unwrapped", "functional_l2", "replay", "param_null", "joint_refit"),
        default=("vanish", "unwrapped", "functional_l2", "replay", "param_null", "joint_refit"),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--width", type=int, help="Override the stream-specific feature width")
    parser.add_argument("--gamma", type=float, help="Override the stream-specific RBF gamma")
    parser.add_argument("--ridge", type=float, help="Override the stream-specific ridge coefficient")
    parser.add_argument("--cap", type=int, help="Override the stream-specific per-task training cap")
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/suite.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.output.exists():
        args.output.unlink()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    for stream in args.streams:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for method in args.methods:
                config = dict(DEFAULTS[stream])
                for key in ("width", "gamma", "ridge", "cap"):
                    value = getattr(args, key)
                    if value is not None:
                        config[key] = value
                result = run_joint(stream, seed, config) if method == "joint_refit" else run_additive(stream, method, seed, config)
                append_jsonl(args.output, result)
                short = {
                    "stream": stream,
                    "seed": seed,
                    "method": method,
                    "final_acc": round(result["final_average_accuracy"], 5),
                    "forget": round(result["mean_forgetting"], 5),
                    "drift": f"{result['max_anchor_drift']:.2e}",
                    "seconds": round(result["elapsed_s"], 3),
                }
                print(json.dumps(short, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
