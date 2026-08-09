#!/usr/bin/env python3
"""Mechanism-level falsification tests for VANISH."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from datasets import split_digits  # noqa: E402
from run_suite import metrics_from_matrix  # noqa: E402
from vanish.core import AdditiveModel, KernelSystem, fit_patch, rbf_kernel  # noqa: E402


def mlp_unpack(theta: np.ndarray, d: int, h: int):
    p = 0
    w1 = theta[p : p + d * h].reshape(d, h)
    p += d * h
    b1 = theta[p : p + h]
    p += h
    w2 = theta[p : p + h]
    p += h
    b2 = theta[p]
    return w1, b1, w2, b2


def mlp_value(theta: np.ndarray, x: np.ndarray, d: int, h: int):
    w1, b1, w2, b2 = mlp_unpack(theta, d, h)
    return np.tanh(x @ w1 + b1) @ w2 + b2


def mlp_jacobian(theta: np.ndarray, x: np.ndarray, d: int, h: int):
    w1, b1, w2, _ = mlp_unpack(theta, d, h)
    hidden = np.tanh(x @ w1 + b1)
    local = (1.0 - hidden * hidden) * w2
    jw1 = np.einsum("nd,nh->ndh", x, local).reshape(len(x), d * h)
    return np.concatenate((jw1, local, hidden, np.ones((len(x), 1))), axis=1)


def finite_update_stress(seed: int = 0):
    rng = np.random.default_rng(seed)
    d, h, n = 3, 12, 18
    p = d * h + h + h + 1
    theta = rng.normal(0.0, 0.35, p)
    protected = rng.normal(size=(n, d))
    probe = rng.normal(size=(128, d))
    jac = mlp_jacobian(theta, protected, d, h)
    direction = rng.normal(size=p)
    jj = jac @ jac.T
    direction = direction - jac.T @ np.linalg.solve(jj + 1e-12 * np.eye(n), jac @ direction)
    direction /= np.linalg.norm(direction)
    tangent_residual = float(np.max(np.abs(jac @ direction)))
    system = KernelSystem(protected, gamma=0.7, base_jitter=1e-13)
    scales = np.logspace(-5, 2, 22)
    rows = []
    f0_old = mlp_value(theta, protected, d, h)
    f0_probe = mlp_value(theta, probe, d, h)
    for scale in scales:
        shifted = theta + scale * direction
        update_old = mlp_value(shifted, protected, d, h) - f0_old
        update_probe = mlp_value(shifted, probe, d, h) - f0_probe
        correction = system.solve(update_old)
        vanish_old = update_old - rbf_kernel(protected, protected, 0.7) @ correction
        vanish_probe = update_probe - rbf_kernel(probe, protected, 0.7) @ correction
        rows.append(
            {
                "scale": float(scale),
                "tangent_projection_drift": float(np.max(np.abs(update_old))),
                "vanish_drift": float(np.max(np.abs(vanish_old))),
                "vanish_probe_rms": float(np.sqrt(np.mean(vanish_probe * vanish_probe))),
            }
        )
    return {"tangent_residual": tangent_residual, "rows": rows}


def derivative_annihilation(seed: int = 0):
    rng = np.random.default_rng(seed)
    gamma = 1.3
    sites = np.array([-1.2, -0.2, 0.8, 1.5])
    delta = sites[:, None] - sites[None, :]
    k = np.exp(-gamma * delta * delta)
    g_ee = k
    g_ed = 2.0 * gamma * delta * k
    g_de = -2.0 * gamma * delta * k
    g_dd = (2.0 * gamma - 4.0 * gamma * gamma * delta * delta) * k
    gram = np.block([[g_ee, g_ed], [g_de, g_dd]])
    x = np.linspace(-2.2, 2.2, 700)
    dx = x[:, None] - sites[None, :]
    base_rep = np.exp(-gamma * dx * dx)
    deriv_rep = 2.0 * gamma * dx * base_rep
    representers = np.concatenate((base_rep, deriv_rep), axis=1)
    phase = rng.uniform(-0.5, 0.5)
    update = 0.8 * np.sin(2.7 * x + phase) + 0.3 * np.cos(5.1 * x)
    update_at = 0.8 * np.sin(2.7 * sites + phase) + 0.3 * np.cos(5.1 * sites)
    derivative_at = 0.8 * 2.7 * np.cos(2.7 * sites + phase) - 0.3 * 5.1 * np.sin(5.1 * sites)
    coefficients = np.linalg.solve(gram, np.concatenate((update_at, derivative_at)))
    projected = update - representers @ coefficients
    # Audit values and derivatives by applying the constraints algebraically.
    constraint_residual = np.concatenate((update_at, derivative_at)) - gram @ coefficients
    return {
        "x": x.tolist(),
        "raw": update.tolist(),
        "projected": projected.tolist(),
        "sites": sites.tolist(),
        "max_value_or_derivative_residual": float(np.max(np.abs(constraint_residual))),
        "projected_rms_away": float(np.sqrt(np.mean(projected * projected))),
    }


class ConditionalKernelModel:
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.blocks = []
        self.all_x = None

    def predict(self, x: np.ndarray):
        value = np.zeros((len(x), 1))
        for block in self.blocks:
            raw = rbf_kernel(x, block["new"], self.gamma)
            if block["old"] is not None:
                raw = raw - rbf_kernel(x, block["old"], self.gamma) @ block["coupling"]
            value += raw @ block["coefficient"]
        return value

    def add(self, x_new: np.ndarray, y_new: np.ndarray):
        old = self.all_x
        residual = y_new - self.predict(x_new)
        if old is None:
            conditional = rbf_kernel(x_new, x_new, self.gamma)
            coupling = None
        else:
            old_system = KernelSystem(old, self.gamma, base_jitter=1e-13)
            coupling = old_system.solve(rbf_kernel(old, x_new, self.gamma))
            conditional = rbf_kernel(x_new, x_new, self.gamma) - rbf_kernel(x_new, old, self.gamma) @ coupling
        coefficient = np.linalg.solve(conditional + 1e-12 * np.eye(len(x_new)), residual)
        self.blocks.append({"new": x_new.copy(), "old": None if old is None else old.copy(), "coupling": coupling, "coefficient": coefficient})
        self.all_x = x_new.copy() if old is None else np.concatenate((old, x_new))


def order_invariance(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, 40)[:, None]
    x[:, 0] += rng.normal(0.0, 2e-4, len(x))
    y = (np.sin(3.0 * x[:, 0]) + 0.2 * np.cos(7.0 * x[:, 0]))[:, None]
    blocks = [np.arange(i, i + 10) for i in range(0, 40, 10)]
    probe = np.linspace(-2.0, 2.0, 500)[:, None]
    gamma = 30.0
    kernel = rbf_kernel(x, x, gamma)
    offline = rbf_kernel(probe, x, gamma) @ np.linalg.solve(kernel + 1e-12 * np.eye(len(x)), y)
    errors = []
    for order in itertools.permutations(range(4)):
        model = ConditionalKernelModel(gamma)
        for block_id in order:
            idx = blocks[block_id]
            model.add(x[idx], y[idx])
        errors.append(float(np.max(np.abs(model.predict(probe) - offline))))
    return {"permutations": 24, "max_error": max(errors), "median_error": float(np.median(errors)), "errors": errors}


def capacity_stress(seeds: int = 5):
    rows = []
    for seed in range(seeds):
        stream = split_digits(seed, cap=140)
        for width in (16, 32, 64, 128, 256):
            for method in ("vanish", "param_null"):
                model = AdditiveModel(stream.classes)
                protected = []
                matrix = np.full((5, 5), np.nan)
                ranks = []
                drifts = []
                for task_id, (xn, yn) in enumerate(stream.train):
                    old = np.concatenate(protected) if protected else None
                    diag = fit_patch(
                        model,
                        xn,
                        yn,
                        old,
                        mode=method,
                        width=width,
                        ridge=1e-4,
                        gamma=0.03,
                        seed=seed * 10007 + task_id * 97 + 13,
                    )
                    ranks.append(diag["effective_rank"])
                    drifts.append(diag["anchor_patch_max"])
                    protected.append(xn)
                    for eval_id in range(task_id + 1):
                        xe, ye = stream.test[eval_id]
                        matrix[task_id, eval_id] = np.mean(model.predict(xe) == ye)
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "width": width,
                        "feature_dimension": 2 * width,
                        "protected_anchors": int(sum(len(x) for x, _ in stream.train[:-1])),
                        "final_available_rank": float(ranks[-1]),
                        "max_anchor_drift": float(max(drifts)),
                        **metrics_from_matrix(matrix),
                    }
                )
    return rows


def scaling_stress(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for count in (25, 50, 100, 200, 400, 800, 1200):
        x = rng.normal(size=(count, 32))
        values = rng.normal(size=(count, 8))
        start = time.perf_counter()
        system = KernelSystem(x, gamma=0.04)
        coefficient = system.solve(values)
        elapsed = time.perf_counter() - start
        drift = values - rbf_kernel(x, x, 0.04) @ coefficient
        rows.append(
            {
                "anchors": count,
                "factor_s": elapsed,
                "max_drift": float(np.max(np.abs(drift))),
                "state_bytes": int(x.nbytes + coefficient.nbytes),
                **system.audit(),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/stress.json")
    parser.add_argument("--capacity-seeds", type=int, default=5)
    args = parser.parse_args()
    result = {
        "finite_update": finite_update_stress(),
        "derivative_constraints": derivative_annihilation(),
        "order_invariance": order_invariance(),
        "capacity": capacity_stress(args.capacity_seeds),
        "scaling": scaling_stress(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "finite_max_vanish_drift": max(r["vanish_drift"] for r in result["finite_update"]["rows"]),
        "finite_max_tangent_drift": max(r["tangent_projection_drift"] for r in result["finite_update"]["rows"]),
        "derivative_residual": result["derivative_constraints"]["max_value_or_derivative_residual"],
        "order_max_error": result["order_invariance"]["max_error"],
        "capacity_rows": len(result["capacity"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
