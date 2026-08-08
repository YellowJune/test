#!/usr/bin/env python3
"""CIFAR-100 class-incremental kill-gate on frozen ResNet-18 embeddings.

The script is intended for a CPU GitHub Actions runner.  It extracts ImageNet
pretrained ResNet-18 features once, then runs seed-matched functional methods
under an identical protected/replay memory budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_suite import metrics_from_matrix  # noqa: E402
from vanish.core import AdditiveModel, fit_patch  # noqa: E402


def extract_embeddings(data_root: Path, cache: Path, batch_size: int) -> tuple[np.ndarray, ...]:
    if cache.exists():
        saved = np.load(cache)
        return saved["x_train"], saved["y_train"], saved["x_test"], saved["y_test"]

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    train = datasets.CIFAR100(data_root, train=True, download=True, transform=transform)
    test = datasets.CIFAR100(data_root, train=False, download=True, transform=transform)
    backbone = models.resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    def encode(dataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        features, labels = [], []
        with torch.inference_mode():
            for image, target in loader:
                features.append(backbone(image).cpu().numpy().astype(np.float32))
                labels.append(target.numpy().astype(np.int64))
        x = np.concatenate(features).astype(np.float64)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x, np.concatenate(labels)

    x_train, y_train = encode(train)
    x_test, y_test = encode(test)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return x_train, y_train, x_test, y_test


def balanced_indices(y: np.ndarray, classes: np.ndarray, per_class: int, rng: np.random.Generator):
    out = []
    for cls in classes:
        candidates = np.flatnonzero(y == cls)
        out.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
    out = np.asarray(out, dtype=int)
    rng.shuffle(out)
    return out


def run_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    method: str,
    tasks: int,
    buffer_size: int,
    train_per_class: int,
    width: int,
    ridge: float,
    gamma: float,
):
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(100)
    blocks = np.array_split(class_order, tasks)
    anchors_per_class = max(1, buffer_size // 100)
    model = AdditiveModel(100)
    protected_x: list[np.ndarray] = []
    matrix = np.full((tasks, tasks), np.nan)
    diagnostics = []
    max_drift = 0.0
    start = time.perf_counter()

    for task_id, classes in enumerate(blocks):
        task_rng = np.random.default_rng(seed * 1009 + task_id * 97 + 31)
        selected = balanced_indices(y_train, classes, train_per_class, task_rng)
        x_new, y_new = x_train[selected], y_train[selected]
        old_x = np.concatenate(protected_x) if protected_x else None
        before = model.logits(old_x) if old_x is not None else None
        diag = fit_patch(
            model,
            x_new,
            y_new,
            old_x,
            mode=method,
            width=width,
            ridge=ridge,
            gamma=gamma,
            seed=seed * 10007 + task_id * 97 + 13,
            functional_weight=10.0,
        )
        if old_x is not None:
            drift = float(np.max(np.abs(model.logits(old_x) - before)))
            max_drift = max(max_drift, drift)
        else:
            drift = 0.0
        diag["cumulative_anchor_drift"] = drift
        diagnostics.append(diag)

        anchor_idx = balanced_indices(y_train, classes, anchors_per_class, task_rng)
        protected_x.append(x_train[anchor_idx])
        for eval_id in range(task_id + 1):
            mask = np.isin(y_test, blocks[eval_id])
            matrix[task_id, eval_id] = np.mean(model.predict(x_test[mask]) == y_test[mask])

    elapsed = time.perf_counter() - start
    return {
        "benchmark": "cifar100_resnet18_imagenet1k",
        "stream": f"cifar100_{tasks}task",
        "method": method,
        "seed": seed,
        "tasks": tasks,
        "buffer_size": buffer_size,
        "anchors_per_class": anchors_per_class,
        "train_per_class": train_per_class,
        "width": width,
        "ridge": ridge,
        "gamma": gamma,
        "elapsed_s": elapsed,
        "parameter_bytes": model.parameter_bytes(),
        "max_anchor_drift": max_drift,
        "accuracy_matrix": matrix.tolist(),
        "diagnostics": diagnostics,
        "class_order": class_order.tolist(),
        **metrics_from_matrix(matrix),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/cifar100_resnet18_64px.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/cifar100_resnet.jsonl")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("vanish", "unwrapped", "functional_l2", "replay", "param_null"),
        choices=("vanish", "unwrapped", "functional_l2", "replay", "param_null"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = extract_embeddings(args.data_root, args.cache, args.batch_size)
    print(json.dumps({"feature_shape": list(x_train.shape), "test_shape": list(x_test.shape)}), flush=True)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for method in args.methods:
            result = run_one(
                x_train,
                y_train,
                x_test,
                y_test,
                seed=seed,
                method=method,
                tasks=args.tasks,
                buffer_size=args.buffer_size,
                train_per_class=args.train_per_class,
                width=args.width,
                ridge=args.ridge,
                gamma=args.gamma,
            )
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(json.dumps({
                "seed": seed,
                "method": method,
                "final_acc": round(result["final_average_accuracy"], 5),
                "forget": round(result["mean_forgetting"], 5),
                "drift": f"{result['max_anchor_drift']:.2e}",
                "seconds": round(result["elapsed_s"], 2),
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
