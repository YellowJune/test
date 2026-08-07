# Trainability Quotient Model Conversion Benchmark

Reference artifact for **Models Are Not Functions: Measuring Trainability Preservation Across Model Conversions**.

The benchmark evaluates two different questions for a model conversion:

1. **Prediction fidelity:** how close are the converted model's current outputs?
2. **Trainability fidelity:** under a declared continuation contract, how close are the *future function updates*?

Primary trajectory score:

`TQ_traj = 1 / (1 + D_func)`

where `D_func` is the RMS discrepancy between source and converted function-update trajectories after subtracting each artifact's own initial output.

Files:
- `BENCHMARK_SPEC.md`: protocol, tiers, and validity rules.
- `tq_benchmark.py`: reference scorer.
- `spec.json`: machine-readable benchmark metadata.
- `leaderboard.csv`: paper baselines from Qwen/Llama quantization, pruning, distillation, and full-parameter continuation.

The score is **contract-dependent**. A result must state checkpoint, conversion, optimizer/state, trainable parameter scope, data split/order, held-out probes, and horizon. `TQ_traj` is an empirical benchmark score, not the paper's continuous/discrete information-dimension Trainability Quotients.

## Minimal scoring input

```json
{
  "source_outputs": [[0.0, 1.0], [0.1, 1.2], [0.2, 1.3]],
  "converted_outputs": [[0.0, 1.0], [0.09, 1.19], [0.19, 1.31]]
}
```

Run:

```bash
python tq_benchmark.py trajectory.json
```
