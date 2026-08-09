# VANISH

VANISH (Value-Annihilating Neural Increments for Stable History) is a reference implementation of exact finite-update preservation by function-space annihilation.

For protected anchors `S`, a completed function update `u` is converted to

```text
A_S u(x) = u(x) - k(x,S) K(S,S)^(-1) u(S).
```

The converted update is zero on `S` up to the audited linear-solve residual. The raw patch may be produced by any optimizer or model; the preservation identity is applied to its realized output rather than to a first-order parameter direction.

## Reproduced headline results

- 600 matched functional runs: 5 streams × 20 seeds × 6 methods.
- 150 conventional MLP runs: 5 streams × 10 seeds × 3 methods.
- 200 capacity-stress rows and four executable mechanism contracts.
- VANISH macro final accuracy: **90.0479%**.
- Matched feature replay macro final accuracy: **90.0481%**.
- Geometric protected-output drift: **2.00e-10** (VANISH) versus **2.30e-1** (feature replay).
- Finite nonlinear tangent-null drift: **55.2465**; wrapped drift: **1.1998e-11**.
- Mixed value/derivative residual: **2.6645e-15**.
- Maximum error versus offline interpolation over all 24 task orders: **6.5810e-13**.

All values above are generated from files in `results/raw/`; derived CSVs are in `results/summary/`.

## Quick verification

The packaged artifact already contains raw outputs. To verify the numerical contracts and rebuild the summaries and figures:

```bash
python -m unittest discover -s tests -v
python experiments/summarize.py
MPLBACKEND=Agg python experiments/make_figures.py
```

To rerun every experiment and rebuild the paper:

```bash
bash scripts/run_all.sh
```

The full suite is CPU-only and needs no dataset download. Scikit-learn Digits and deterministic synthetic shapes are generated locally.

## Layout

```text
src/vanish/core.py              preservation operator and neural patches
experiments/datasets.py         deterministic visual task streams
experiments/run_suite.py        matched functional experiments
experiments/run_mlp_baselines.py conventional neural baselines
experiments/stress_tests.py     nonlinear, derivative, order, rank, scaling tests
experiments/summarize.py        confidence intervals and paired statistics
experiments/make_figures.py     vector and raster publication figures
tests/test_core.py              executable theorem contracts
results/raw/                    immutable run-level records
results/summary/                generated CSV and matrix summaries
figures/                        generated PDF and PNG figures
paper/                          IEEE-compatible two-column source and PDF
```

## Contract boundary

VANISH exactly preserves the declared linear functionals. Point anchors, derivative observations, and moments are examples. The paper's power-function theorem gives a quantitative off-anchor bound when the raw update belongs to the RKHS. Test-distribution accuracy remains an empirical quantity and is reported separately from anchor drift.

## Citation

See `CITATION.cff`. Author: JunHyun Kim, Independent Researcher.

## License

MIT. See `LICENSE`.

