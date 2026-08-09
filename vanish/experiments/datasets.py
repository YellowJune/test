"""Deterministic visual continual-learning streams used in the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class TaskStream:
    name: str
    train: list[tuple[np.ndarray, np.ndarray]]
    test: list[tuple[np.ndarray, np.ndarray]]
    classes: int
    input_shape: tuple[int, ...]


def _stratified_cap(x: np.ndarray, y: np.ndarray, cap: int | None, rng: np.random.Generator):
    if cap is None or len(x) <= cap:
        return x.copy(), y.copy()
    classes = np.unique(y)
    quota = np.full(len(classes), cap // len(classes), dtype=int)
    quota[: cap % len(classes)] += 1
    indices = []
    for cls, count in zip(classes, quota):
        available = np.flatnonzero(y == cls)
        indices.extend(rng.choice(available, size=min(count, len(available)), replace=False).tolist())
    indices = np.array(indices, dtype=int)
    rng.shuffle(indices)
    return x[indices], y[indices]


def _digits_base(seed: int):
    data = load_digits()
    x = data.images.astype(np.float64) / 16.0
    y = data.target.astype(int)
    return train_test_split(x, y, train_size=0.70, random_state=seed, stratify=y)


def split_digits(seed: int, cap: int | None = 220) -> TaskStream:
    x_train, x_test, y_train, y_test = _digits_base(seed)
    rng = np.random.default_rng(seed + 101)
    train, test = [], []
    for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)):
        tr = np.isin(y_train, (a, b))
        te = np.isin(y_test, (a, b))
        xt, yt = _stratified_cap(x_train[tr], y_train[tr], cap, rng)
        train.append((xt.reshape(len(xt), -1), yt))
        test.append((x_test[te].reshape(np.sum(te), -1), y_test[te]))
    return TaskStream("split_digits", train, test, 10, (8, 8))


def rotated_digits(seed: int, cap: int | None = 180) -> TaskStream:
    x_train, x_test, y_train, y_test = _digits_base(seed)
    rng = np.random.default_rng(seed + 211)
    train, test = [], []
    for angle in (-30.0, -15.0, 0.0, 15.0, 30.0):
        tr = rotate(x_train, angle, axes=(1, 2), reshape=False, order=1, mode="constant")
        te = rotate(x_test, angle, axes=(1, 2), reshape=False, order=1, mode="constant")
        xt, yt = _stratified_cap(tr, y_train, cap, rng)
        train.append((xt.reshape(len(xt), -1), yt))
        test.append((te.reshape(len(te), -1), y_test.copy()))
    return TaskStream("rotated_digits", train, test, 10, (8, 8))


def permuted_digits(seed: int, cap: int | None = 180) -> TaskStream:
    x_train, x_test, y_train, y_test = _digits_base(seed)
    rng = np.random.default_rng(seed + 307)
    flat_train = x_train.reshape(len(x_train), -1)
    flat_test = x_test.reshape(len(x_test), -1)
    train, test = [], []
    for _ in range(5):
        permutation = rng.permutation(flat_train.shape[1])
        xt, yt = _stratified_cap(flat_train[:, permutation], y_train, cap, rng)
        train.append((xt, yt))
        test.append((flat_test[:, permutation], y_test.copy()))
    return TaskStream("permuted_digits", train, test, 10, (8, 8))


def corrupted_digits(seed: int, cap: int | None = 180) -> TaskStream:
    x_train, x_test, y_train, y_test = _digits_base(seed)
    rng = np.random.default_rng(seed + 409)
    transforms = []
    transforms.append((x_train.copy(), x_test.copy()))
    transforms.append(
        (
            np.clip(x_train + rng.normal(0.0, 0.18, x_train.shape), 0.0, 1.0),
            np.clip(x_test + rng.normal(0.0, 0.18, x_test.shape), 0.0, 1.0),
        )
    )
    transforms.append((gaussian_filter(x_train, sigma=(0, 0.65, 0.65)), gaussian_filter(x_test, sigma=(0, 0.65, 0.65))))
    occ_train, occ_test = x_train.copy(), x_test.copy()
    occ_train[:, 2:5, 2:5] = 0.0
    occ_test[:, 2:5, 2:5] = 0.0
    transforms.append((occ_train, occ_test))
    transforms.append((np.clip(1.25 * x_train - 0.1, 0.0, 1.0), np.clip(1.25 * x_test - 0.1, 0.0, 1.0)))
    train, test = [], []
    for tr, te in transforms:
        xt, yt = _stratified_cap(tr, y_train, cap, rng)
        train.append((xt.reshape(len(xt), -1), yt))
        test.append((te.reshape(len(te), -1), y_test.copy()))
    return TaskStream("corrupted_digits", train, test, 10, (8, 8))


def _draw_shapes(count: int, seed: int, size: int = 16):
    rng = np.random.default_rng(seed)
    images = np.zeros((count, size, size), dtype=np.float64)
    labels = np.arange(count) % 4
    yy, xx = np.mgrid[:size, :size]
    for i, label in enumerate(labels):
        cx = rng.integers(5, size - 5)
        cy = rng.integers(5, size - 5)
        radius = rng.integers(3, 6)
        thickness = rng.integers(1, 3)
        if label == 0:  # ring
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            mask = np.abs(dist - radius) <= thickness / 1.5
        elif label == 1:  # square frame
            outer = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
            inner = (np.abs(xx - cx) < radius - thickness) & (np.abs(yy - cy) < radius - thickness)
            mask = outer & ~inner
        elif label == 2:  # triangle
            dy = yy - (cy - radius)
            half = np.maximum(dy / 2.0, 0.0)
            outer = (dy >= 0) & (dy <= 2 * radius) & (np.abs(xx - cx) <= half)
            inner = (dy >= thickness) & (dy <= 2 * radius - thickness) & (np.abs(xx - cx) < np.maximum(half - thickness, 0))
            mask = outer & ~inner
        else:  # cross
            mask = ((np.abs(xx - cx) <= thickness) & (np.abs(yy - cy) <= radius)) | (
                (np.abs(yy - cy) <= thickness) & (np.abs(xx - cx) <= radius)
            )
        images[i, mask] = rng.uniform(0.75, 1.0)
        images[i] = np.clip(images[i] + rng.normal(0.0, 0.05, (size, size)), 0.0, 1.0)
    order = rng.permutation(count)
    return images[order], labels[order]


def shape_stream(seed: int, cap: int | None = 240) -> TaskStream:
    base_train, y_train = _draw_shapes(1200, seed + 503)
    base_test, y_test = _draw_shapes(600, seed + 509)
    rng = np.random.default_rng(seed + 521)
    train, test = [], []
    for angle, noise in ((0.0, 0.0), (20.0, 0.0), (-20.0, 0.0), (0.0, 0.12)):
        tr = rotate(base_train, angle, axes=(1, 2), reshape=False, order=1, mode="constant")
        te = rotate(base_test, angle, axes=(1, 2), reshape=False, order=1, mode="constant")
        if noise:
            tr = np.clip(tr + rng.normal(0.0, noise, tr.shape), 0.0, 1.0)
            te = np.clip(te + rng.normal(0.0, noise, te.shape), 0.0, 1.0)
        xt, yt = _stratified_cap(tr, y_train, cap, rng)
        train.append((xt.reshape(len(xt), -1), yt))
        test.append((te.reshape(len(te), -1), y_test.copy()))
    return TaskStream("shape_stream", train, test, 4, (16, 16))


STREAMS = {
    "split_digits": split_digits,
    "rotated_digits": rotated_digits,
    "permuted_digits": permuted_digits,
    "corrupted_digits": corrupted_digits,
    "shape_stream": shape_stream,
}

