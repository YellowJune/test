#!/usr/bin/env python3
"""Aggregate seeded runs, confidence intervals, and paired tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "final_average_accuracy",
    "average_incremental_accuracy",
    "mean_new_task_accuracy",
    "backward_transfer",
    "mean_forgetting",
    "worst_forgetting",
    "max_anchor_drift",
    "elapsed_s",
    "parameter_bytes",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return mean, sd, mean - half, mean + half


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=ROOT / "results/raw/suite_20seed.jsonl")
    parser.add_argument("--mlp", type=Path, default=ROOT / "results/raw/mlp_10seed.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/summary")
    args = parser.parse_args()
    records = read_jsonl(args.suite) + read_jsonl(args.mlp)
    if not records:
        raise SystemExit("No experiment records found")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["stream"], record["method"])].append(record)

    summary_rows = []
    for (stream, method), group in sorted(grouped.items()):
        row = {"stream": stream, "method": method, "n": len(group)}
        for metric in METRICS:
            mean, sd, low, high = mean_ci([float(r[metric]) for r in group])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    write_csv(args.output / "summary_long.csv", summary_rows)

    paper_rows = []
    for row in summary_rows:
        paper_rows.append(
            {
                "stream": row["stream"],
                "method": row["method"],
                "n": row["n"],
                "final_acc_pct": 100.0 * row["final_average_accuracy_mean"],
                "final_acc_ci95_pct": 50.0 * (row["final_average_accuracy_ci95_high"] - row["final_average_accuracy_ci95_low"]),
                "forget_pct": 100.0 * row["mean_forgetting_mean"],
                "forget_ci95_pct": 50.0 * (row["mean_forgetting_ci95_high"] - row["mean_forgetting_ci95_low"]),
                "anchor_drift": row["max_anchor_drift_mean"],
                "seconds": row["elapsed_s_mean"],
                "memory_mib": row["parameter_bytes_mean"] / (1024.0 * 1024.0),
            }
        )
    write_csv(args.output / "paper_table.csv", paper_rows)

    # Pair by stream and seed. VANISH has 20 seeds; conventional MLPs have 10.
    by_key = {(r["stream"], r["method"], int(r["seed"])): r for r in records}
    test_specs = [
        ("final_average_accuracy", "greater"),
        ("mean_forgetting", "less"),
        ("max_anchor_drift", "less"),
    ]
    test_rows = []
    streams = sorted({r["stream"] for r in records})
    methods = sorted({r["method"] for r in records if r["method"] != "vanish"})
    for stream in streams:
        for baseline in methods:
            seeds = sorted(
                {seed for (s, m, seed) in by_key if s == stream and m == "vanish"}
                & {seed for (s, m, seed) in by_key if s == stream and m == baseline}
            )
            if len(seeds) < 5:
                continue
            for metric, alternative in test_specs:
                a = np.asarray([by_key[(stream, "vanish", seed)][metric] for seed in seeds], dtype=float)
                b = np.asarray([by_key[(stream, baseline, seed)][metric] for seed in seeds], dtype=float)
                try:
                    test = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                test_rows.append(
                    {
                        "stream": stream,
                        "baseline": baseline,
                        "metric": metric,
                        "alternative": alternative,
                        "n_pairs": len(seeds),
                        "vanish_mean": float(np.mean(a)),
                        "baseline_mean": float(np.mean(b)),
                        "mean_difference": float(np.mean(a - b)),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )
    adjusted = holm_adjust([r["p_value"] for r in test_rows])
    for row, p_adj in zip(test_rows, adjusted):
        row["holm_p"] = p_adj
    write_csv(args.output / "paired_tests.csv", test_rows)

    matrices = {
        f"{r['stream']}::{r['method']}::{r['seed']}": r["accuracy_matrix"] for r in records
    }
    (args.output / "accuracy_matrices.json").write_text(
        json.dumps(matrices, indent=2), encoding="utf-8"
    )

    macro_rows = []
    for method in sorted({r["method"] for r in records}):
        method_rows = [r for r in paper_rows if r["method"] == method]
        macro_rows.append(
            {
                "method": method,
                "streams": len(method_rows),
                "macro_final_acc_pct": float(np.mean([r["final_acc_pct"] for r in method_rows])),
                "macro_forget_pct": float(np.mean([r["forget_pct"] for r in method_rows])),
                "geometric_anchor_drift": float(
                    np.exp(np.mean(np.log([max(r["anchor_drift"], 1e-16) for r in method_rows])))
                ),
            }
        )
    write_csv(args.output / "macro_summary.csv", macro_rows)
    print(json.dumps({"records": len(records), "groups": len(summary_rows), "paired_tests": len(test_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
