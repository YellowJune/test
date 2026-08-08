"""Executable numerical contracts for the VANISH reference implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from stress_tests import derivative_annihilation, finite_update_stress, order_invariance  # noqa: E402
from vanish.core import KernelSystem, RandomFeatureMap, rbf_kernel  # noqa: E402


class TestFunctionalAnnihilation(unittest.TestCase):
    def test_arbitrary_vector_update_is_zero_at_anchors(self):
        rng = np.random.default_rng(20260808)
        anchors = rng.normal(size=(45, 7))
        probe = rng.normal(size=(25, 7))
        fmap = RandomFeatureMap.create(7, 96, 17)
        coefficient = rng.normal(size=(fmap.output_dim, 5))
        raw_anchor = fmap(anchors) @ coefficient
        raw_probe = fmap(probe) @ coefficient
        system = KernelSystem(anchors, gamma=0.18)
        correction = system.solve(raw_anchor)
        safe_anchor = raw_anchor - rbf_kernel(anchors, anchors, 0.18) @ correction
        safe_probe = raw_probe - rbf_kernel(probe, anchors, 0.18) @ correction
        self.assertLess(float(np.max(np.abs(safe_anchor))), 1e-8)
        self.assertGreater(float(np.sqrt(np.mean(safe_probe**2))), 1e-3)

    def test_finite_update_beats_tangent_promise(self):
        result = finite_update_stress(0)
        self.assertLess(result["tangent_residual"], 1e-8)
        self.assertGreater(max(r["tangent_projection_drift"] for r in result["rows"]), 1.0)
        self.assertLess(max(r["vanish_drift"] for r in result["rows"]), 1e-8)

    def test_values_and_derivatives(self):
        result = derivative_annihilation(0)
        self.assertLess(result["max_value_or_derivative_residual"], 1e-10)
        self.assertGreater(result["projected_rms_away"], 1e-3)

    def test_hard_interpolation_is_order_equivalent(self):
        result = order_invariance(0)
        self.assertEqual(result["permutations"], 24)
        self.assertLess(result["max_error"], 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
