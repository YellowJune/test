"""Core operators for VANISH.

The operator implemented here is

    A_S u(x) = u(x) - k(x,S) K(S,S)^{-1} u(S),

which makes an arbitrary finite function update exactly zero on a protected
anchor set S (up to floating-point linear-solve error).  The update generator
used in the experiments is a deterministic random-feature neural module; the
annihilator itself is agnostic to how the generator is parameterized or fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def one_hot(y: np.ndarray, classes: int, low: float = -1.0) -> np.ndarray:
    out = np.full((len(y), classes), low, dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def squared_distances(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(z * z, axis=1)[None, :]
        - 2.0 * (x @ z.T),
        0.0,
    )


def rbf_kernel(x: np.ndarray, z: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-float(gamma) * squared_distances(x, z))


class KernelSystem:
    """Numerically guarded SPD kernel solve with explicit audit metadata."""

    def __init__(self, anchors: np.ndarray, gamma: float, base_jitter: float = 1e-12):
        self.anchors = np.asarray(anchors, dtype=np.float64)
        self.gamma = float(gamma)
        self.kernel = rbf_kernel(self.anchors, self.anchors, self.gamma)
        self.jitter = float(base_jitter)
        identity = np.eye(len(self.kernel))
        self.factor = None
        for _ in range(12):
            try:
                self.factor = cho_factor(
                    self.kernel + self.jitter * identity,
                    lower=True,
                    check_finite=False,
                )
                break
            except np.linalg.LinAlgError:
                self.jitter *= 10.0
        if self.factor is None:
            raise np.linalg.LinAlgError("Protected-anchor kernel is not numerically factorizable")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self.factor, rhs, check_finite=False)

    def interpolation(self, x: np.ndarray) -> np.ndarray:
        """Return B such that B @ v interpolates anchor values v at x."""
        return self.solve(rbf_kernel(self.anchors, x, self.gamma)).T

    def audit(self) -> dict[str, float]:
        # Computing an exact condition number for every run is expensive.  The
        # interpolation residual is the directly relevant numerical certificate.
        residual = self.kernel @ self.solve(np.eye(len(self.kernel))) - np.eye(len(self.kernel))
        return {
            "jitter": self.jitter,
            "inverse_residual_max": float(np.max(np.abs(residual))),
        }


@dataclass(frozen=True)
class RandomFeatureMap:
    weight: np.ndarray
    bias: np.ndarray

    @classmethod
    def create(cls, input_dim: int, width: int, seed: int) -> "RandomFeatureMap":
        rng = np.random.default_rng(seed)
        weight = rng.normal(0.0, math.sqrt(2.0 / input_dim), size=(input_dim, width))
        bias = rng.uniform(-math.pi, math.pi, size=width)
        return cls(weight=weight, bias=bias)

    @property
    def output_dim(self) -> int:
        return 2 * self.weight.shape[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        pre = np.asarray(x, dtype=np.float64) @ self.weight + self.bias
        return np.concatenate((np.tanh(pre), np.cos(pre)), axis=1) / math.sqrt(
            self.weight.shape[1]
        )


@dataclass
class Patch:
    feature_map: RandomFeatureMap
    coefficient: np.ndarray
    anchors: np.ndarray | None
    correction: np.ndarray | None
    gamma: float
    name: str

    def raw(self, x: np.ndarray) -> np.ndarray:
        return self.feature_map(x) @ self.coefficient

    def predict(self, x: np.ndarray) -> np.ndarray:
        value = self.raw(x)
        if self.anchors is not None:
            value = value - rbf_kernel(x, self.anchors, self.gamma) @ self.correction
        return value

    def parameter_bytes(self) -> int:
        total = self.feature_map.weight.nbytes + self.feature_map.bias.nbytes + self.coefficient.nbytes
        if self.anchors is not None:
            total += self.anchors.nbytes + self.correction.nbytes
        return int(total)


class AdditiveModel:
    def __init__(self, classes: int):
        self.classes = int(classes)
        self.patches: list[Patch] = []

    def logits(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), self.classes), dtype=np.float64)
        for patch in self.patches:
            out += patch.predict(x)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(x), axis=1)

    def parameter_bytes(self) -> int:
        return sum(p.parameter_bytes() for p in self.patches)


def ridge_fit(z: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    n, d = z.shape
    ridge = float(ridge)
    if d <= n:
        return np.linalg.solve(z.T @ z + ridge * np.eye(d), z.T @ target)
    return z.T @ np.linalg.solve(z @ z.T + ridge * np.eye(n), target)


def _fit_parameter_nullspace(
    phi_new: np.ndarray,
    phi_old: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(phi_old) == 0:
        return ridge_fit(phi_new, target, ridge), {"available_rank": float(phi_new.shape[1])}
    gram = phi_old @ phi_old.T
    factor = None
    jitter = 1e-10
    eye = np.eye(len(gram))
    for _ in range(10):
        try:
            factor = cho_factor(gram + jitter * eye, lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if factor is None:
        raise np.linalg.LinAlgError("Null-space Gram matrix failed")
    cross = phi_new @ phi_old.T
    z = phi_new - cho_solve(factor, cross.T, check_finite=False).T @ phi_old
    coefficient = ridge_fit(z, target, ridge)
    # The fitted coefficient must itself lie in the protected nullspace.
    coefficient = coefficient - phi_old.T @ cho_solve(
        factor, phi_old @ coefficient, check_finite=False
    )
    rank_old = np.linalg.matrix_rank(phi_old, tol=1e-8)
    return coefficient, {
        "available_rank": float(max(phi_old.shape[1] - rank_old, 0)),
        "projection_jitter": jitter,
    }


def fit_patch(
    model: AdditiveModel,
    x_new: np.ndarray,
    y_new: np.ndarray,
    protect_x: np.ndarray | None,
    *,
    mode: str,
    width: int,
    ridge: float,
    gamma: float,
    seed: int,
    functional_weight: float = 10.0,
) -> dict[str, float]:
    """Fit one neural increment under the selected preservation mechanism."""
    valid = {"vanish", "unwrapped", "functional_l2", "replay", "param_null"}
    if mode not in valid:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(valid)}")
    fmap = RandomFeatureMap.create(x_new.shape[1], width, seed)
    phi_new = fmap(x_new)
    target = one_hot(y_new, model.classes) - model.logits(x_new)
    phi_old = fmap(protect_x) if protect_x is not None and len(protect_x) else None
    anchors = None
    correction = None
    numerical = {}

    if mode == "vanish" and phi_old is not None:
        system = KernelSystem(protect_x, gamma)
        interpolation = system.interpolation(x_new)
        annihilated = phi_new - interpolation @ phi_old
        coefficient = ridge_fit(annihilated, target, ridge)
        correction = system.solve(phi_old @ coefficient)
        anchors = protect_x.copy()
        numerical = system.audit()
        effective_rank = np.linalg.matrix_rank(annihilated, tol=1e-8)
    elif mode in {"functional_l2", "replay"} and phi_old is not None:
        weight = 1.0 if mode == "replay" else functional_weight
        z = np.concatenate((phi_new, math.sqrt(weight) * phi_old), axis=0)
        t = np.concatenate((target, np.zeros((len(phi_old), model.classes))), axis=0)
        coefficient = ridge_fit(z, t, ridge)
        effective_rank = np.linalg.matrix_rank(z, tol=1e-8)
    elif mode == "param_null" and phi_old is not None:
        coefficient, numerical = _fit_parameter_nullspace(phi_new, phi_old, target, ridge)
        effective_rank = int(numerical["available_rank"])
    else:
        coefficient = ridge_fit(phi_new, target, ridge)
        effective_rank = np.linalg.matrix_rank(phi_new, tol=1e-8)

    patch = Patch(
        feature_map=fmap,
        coefficient=coefficient,
        anchors=anchors,
        correction=correction,
        gamma=float(gamma),
        name=mode,
    )
    model.patches.append(patch)
    anchor_drift = 0.0
    if protect_x is not None and len(protect_x):
        anchor_drift = float(np.max(np.abs(patch.predict(protect_x))))
    fit_error = model.logits(x_new) - one_hot(y_new, model.classes)
    return {
        "anchor_patch_max": anchor_drift,
        "new_fit_rmse": float(np.sqrt(np.mean(fit_error * fit_error))),
        "effective_rank": float(effective_rank),
        "patch_bytes": float(patch.parameter_bytes()),
        **numerical,
    }

