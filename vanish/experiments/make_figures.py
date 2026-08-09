#!/usr/bin/env python3
"""Create publication figures from immutable raw result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "vanish": "#0072B2",
    "unwrapped": "#D55E00",
    "functional_l2": "#CC79A7",
    "replay": "#E69F00",
    "param_null": "#009E73",
    "joint_refit": "#4D4D4D",
    "mlp_ft": "#8C564B",
    "mlp_replay": "#F0A202",
    "mlp_joint": "#6A5ACD",
}
LABELS = {
    "vanish": "VANISH",
    "unwrapped": "Unwrapped",
    "functional_l2": r"Functional $L_2$",
    "replay": "Feature replay",
    "param_null": "Parameter null",
    "joint_refit": "Joint refit",
    "mlp_ft": "MLP fine-tune",
    "mlp_replay": "MLP replay",
    "mlp_joint": "MLP joint",
}
STREAM_LABELS = {
    "split_digits": "Split Digits",
    "rotated_digits": "Rotated Digits",
    "permuted_digits": "Permuted Digits",
    "corrupted_digits": "Corrupted Digits",
    "shape_stream": "Shape Stream",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def overview(out: Path):
    fig, ax = plt.subplots(figsize=(7.15, 2.15))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.30, 1.05, 2.55, 1.75, "1  Train any patch", "$u_t(x)$\nany optimizer / module", "#E8F1FA"),
        (3.35, 1.05, 2.75, 1.75, "2  Annihilate", "$u_t-k_{xS}K_{SS}^{-1}u_t(S)$\nin function space", "#D7ECFA"),
        (6.75, 1.05, 2.55, 1.75, "3  Exact invariant", "$\\mathcal{A}_{S}u_t(S)=0$\nfor the finite patch", "#DDF3E7"),
        (9.95, 1.05, 2.65, 1.75, "4  Add safely", "$f_t=f_{t-1}+\\mathcal{A}_{S}u_t$\nold values unchanged", "#F4E9D8"),
    ]
    for x, y, w, h, title, formula, color in boxes:
        rect = mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=color, edgecolor="#333333", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + 1.29, title, ha="center", va="center", weight="bold", fontsize=8.5)
        ax.text(x + w / 2, y + 0.64, formula, ha="center", va="center", fontsize=7.5, linespacing=1.45)
    for x0, x1 in ((2.85, 3.35), (6.1, 6.75), (9.3, 9.95)):
        ax.annotate("", xy=(x1 - 0.07, 1.93), xytext=(x0 + 0.07, 1.93), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))
    ax.text(4.72, 3.38, "protected functionals", ha="center", weight="bold", color="#005A8D")
    ax.annotate(r"values, derivatives, moments $\ell_1,\ldots,\ell_m$", xy=(4.72, 2.87), xytext=(4.72, 3.15), ha="center", arrowprops=dict(arrowstyle="-|>", color="#005A8D"))
    ax.text(8.0, 0.50, "finite-update identity (not a first-order promise)", ha="center", color="#007A53", weight="bold")
    save(fig, out, "fig1_overview")


def pareto(records: list[dict], out: Path):
    streams = list(STREAM_LABELS)
    methods = ["vanish", "unwrapped", "functional_l2", "replay", "param_null", "joint_refit", "mlp_ft", "mlp_replay", "mlp_joint"]
    fig, axes = plt.subplots(1, 5, figsize=(7.15, 2.55), sharey=True)
    for ax, stream in zip(axes, streams):
        subset = [r for r in records if r["stream"] == stream]
        for method in methods:
            group = [r for r in subset if r["method"] == method]
            if not group:
                continue
            x = 100 * np.mean([r["final_average_accuracy"] for r in group])
            y = np.mean([max(r["max_anchor_drift"], 1e-16) for r in group])
            ax.scatter(x, y, s=34 if method == "vanish" else 20, marker="*" if method == "vanish" else "o", color=COLORS[method], edgecolor="white", linewidth=0.35, zorder=4)
        ax.set_yscale("log")
        ax.set_title(STREAM_LABELS[stream])
        ax.set_xlabel("final acc. (%)")
        ax.grid(axis="y", alpha=0.2, which="both")
    axes[0].set_ylabel("maximum protected-output drift")
    handles = [mpl.lines.Line2D([], [], linestyle="", marker="*" if m == "vanish" else "o", color=COLORS[m], label=LABELS[m], markersize=7 if m == "vanish" else 4.5) for m in methods]
    fig.subplots_adjust(bottom=0.32, top=0.79, left=0.08, right=0.995, wspace=0.25)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=5, frameon=False)
    fig.suptitle("Accuracy--invariance frontier across 600 functional and 150 MLP runs", y=0.97, weight="bold")
    save(fig, out, "fig2_pareto")


def accuracy_bars(records: list[dict], out: Path):
    methods = ["vanish", "param_null", "functional_l2", "replay", "unwrapped", "joint_refit", "mlp_ft", "mlp_replay", "mlp_joint"]
    streams = list(STREAM_LABELS)
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 4.35), sharex=True)
    x = np.arange(len(streams))
    width = 0.087
    for i, method in enumerate(methods):
        means, cis, forget = [], [], []
        for stream in streams:
            values = [r for r in records if r["stream"] == stream and r["method"] == method]
            a = np.asarray([r["final_average_accuracy"] for r in values])
            f = np.asarray([r["mean_forgetting"] for r in values])
            means.append(100 * np.mean(a))
            cis.append(100 * 1.96 * np.std(a, ddof=1) / np.sqrt(len(a)))
            forget.append(100 * np.mean(f))
        offset = (i - (len(methods) - 1) / 2) * width
        axes[0].bar(x + offset, means, width, yerr=cis, color=COLORS[method], label=LABELS[method], linewidth=0, error_kw=dict(lw=0.55, capsize=1.2))
        axes[1].bar(x + offset, forget, width, color=COLORS[method], linewidth=0)
    axes[0].set_ylabel("final average accuracy (%)")
    axes[1].set_ylabel("mean forgetting (points)")
    axes[1].set_xticks(x, [STREAM_LABELS[s] for s in streams], rotation=12, ha="right")
    axes[0].grid(axis="y", alpha=0.18)
    axes[1].grid(axis="y", alpha=0.18)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False)
    fig.subplots_adjust(hspace=0.18)
    save(fig, out, "fig3_accuracy_forgetting")


def matrices(records: list[dict], out: Path):
    chosen = []
    for method in ("unwrapped", "functional_l2", "param_null", "vanish"):
        chosen.append(next(r for r in records if r["stream"] == "split_digits" and r["method"] == method and r["seed"] == 0))
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 1.78), constrained_layout=True)
    for ax, record in zip(axes, chosen):
        matrix = np.asarray(record["accuracy_matrix"], dtype=float) * 100
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        for i in range(matrix.shape[0]):
            for j in range(i + 1):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="white" if matrix[i, j] < 60 else "black", fontsize=6.2)
        ax.set_title(LABELS[record["method"]])
        ax.set_xlabel("evaluated task")
        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(5), range(1, 6))
    axes[0].set_ylabel("after learning task")
    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label("accuracy (%)")
    save(fig, out, "fig4_accuracy_matrices")


def mechanism(stress: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45))
    rows = stress["finite_update"]["rows"]
    scale = np.asarray([r["scale"] for r in rows])
    tangent = np.asarray([r["tangent_projection_drift"] for r in rows])
    vanish = np.asarray([r["vanish_drift"] for r in rows])
    probe = np.asarray([r["vanish_probe_rms"] for r in rows])
    axes[0].loglog(scale, tangent, "o-", ms=3, color=COLORS["param_null"], label="tangent-null drift")
    axes[0].loglog(scale, vanish, "o-", ms=3, color=COLORS["vanish"], label="VANISH drift")
    axes[0].loglog(scale, probe, "--", color="#777777", label="VANISH off-anchor RMS")
    axes[0].set_xlabel(r"finite parameter displacement $\|\Delta\theta\|$")
    axes[0].set_ylabel("absolute update")
    axes[0].set_title("First-order safety fails at finite scale")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False)

    d = stress["derivative_constraints"]
    x = np.asarray(d["x"])
    axes[1].plot(x, d["raw"], color="#999999", lw=1.2, label="raw update")
    axes[1].plot(x, d["projected"], color=COLORS["vanish"], lw=1.5, label="value+derivative annihilated")
    axes[1].scatter(d["sites"], np.zeros(len(d["sites"])), color="#111111", marker="x", zorder=5, label="protected jets")
    axes[1].axhline(0, color="#BBBBBB", lw=0.6)
    axes[1].set_xlabel("input")
    axes[1].set_ylabel("function increment")
    axes[1].set_title("One operator protects values and slopes")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    save(fig, out, "fig5_mechanism")


def capacity_scaling(stress: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
    rows = stress["capacity"]
    for method in ("vanish", "param_null"):
        widths = sorted({r["width"] for r in rows if r["method"] == method})
        mean = [100 * np.mean([r["final_average_accuracy"] for r in rows if r["method"] == method and r["width"] == w]) for w in widths]
        sd = [100 * np.std([r["final_average_accuracy"] for r in rows if r["method"] == method and r["width"] == w], ddof=1) for w in widths]
        axes[0].plot(widths, mean, "o-", color=COLORS[method], label=LABELS[method])
        axes[0].fill_between(widths, np.asarray(mean) - np.asarray(sd), np.asarray(mean) + np.asarray(sd), color=COLORS[method], alpha=0.14)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("random-feature width")
    axes[0].set_ylabel("final accuracy (%)")
    axes[0].set_title("Plasticity under saturated parameter rank")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    scaling = stress["scaling"]
    anchors = np.asarray([r["anchors"] for r in scaling])
    seconds = np.asarray([r["factor_s"] for r in scaling])
    state = np.asarray([r["state_bytes"] / (1024 * 1024) for r in scaling])
    axes[1].loglog(anchors, seconds, "o-", color=COLORS["vanish"], label="factor + solve time")
    twin = axes[1].twinx()
    twin.loglog(anchors, state, "s--", color="#D55E00", label="state")
    axes[1].set_xlabel("protected anchors")
    axes[1].set_ylabel("time (s)", color=COLORS["vanish"])
    twin.set_ylabel("state (MiB)", color="#D55E00")
    axes[1].set_title("Exact dense reference implementation")
    axes[1].grid(alpha=0.2, which="both")
    handles = axes[1].get_lines() + twin.get_lines()
    axes[1].legend(handles, [h.get_label() for h in handles], frameon=False, loc="upper left")
    save(fig, out, "fig6_capacity_scaling")


def learning_curves(records: list[dict], out: Path):
    streams = ["split_digits", "rotated_digits", "permuted_digits", "corrupted_digits", "shape_stream"]
    methods = ["vanish", "unwrapped", "functional_l2", "replay", "param_null", "joint_refit"]
    fig, axes = plt.subplots(1, 5, figsize=(7.15, 2.35), sharey=True)
    for ax, stream in zip(axes, streams):
        for method in methods:
            group = [r for r in records if r["stream"] == stream and r["method"] == method]
            curves = []
            for r in group:
                m = np.asarray(r["accuracy_matrix"], dtype=float)
                curves.append([100 * np.nanmean(m[t, : t + 1]) for t in range(len(m))])
            curves = np.asarray(curves)
            ax.plot(range(1, curves.shape[1] + 1), curves.mean(0), color=COLORS[method], lw=1.35 if method == "vanish" else 0.9, label=LABELS[method])
        ax.set_title(STREAM_LABELS[stream])
        ax.set_xlabel("tasks seen")
        ax.set_xticks(range(1, curves.shape[1] + 1))
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("seen-task accuracy (%)")
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], label=LABELS[m], lw=1.5) for m in methods]
    fig.subplots_adjust(bottom=0.27, top=0.86, left=0.08, right=0.995, wspace=0.24)
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.015), frameon=False)
    save(fig, out, "fig7_learning_curves")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=ROOT / "results/raw/suite_20seed.jsonl")
    parser.add_argument("--mlp", type=Path, default=ROOT / "results/raw/mlp_10seed.jsonl")
    parser.add_argument("--stress", type=Path, default=ROOT / "results/raw/stress_20seed.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_style()
    records = read_jsonl(args.suite) + read_jsonl(args.mlp)
    stress = json.loads(args.stress.read_text(encoding="utf-8"))
    overview(args.output)
    pareto(records, args.output)
    accuracy_bars(records, args.output)
    matrices(records, args.output)
    mechanism(stress, args.output)
    capacity_scaling(stress, args.output)
    learning_curves(records, args.output)
    print(json.dumps({"figures": 7, "formats": ["pdf", "png"], "records": len(records)}))


if __name__ == "__main__":
    main()
