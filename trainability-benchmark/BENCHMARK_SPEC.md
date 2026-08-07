# Trainability Quotient (TQ) Model Conversion Benchmark v1

A benchmark instance is a 5-tuple

`(source artifact, conversion C, continuation contract, held-out probes, horizon)`.

The benchmark deliberately reports **prediction fidelity and trainability fidelity separately**.

## Required fields

1. **Source artifact**: exact model/checkpoint identifier and revision when available.
2. **Conversion**: deterministic conversion code/configuration (e.g. INT8 group size 64, magnitude pruning 40%, adapter merge, distillation recipe).
3. **Continuation contract**: optimizer/update law, learning rate, trainable parameter scope, data ordering, and state/provenance available after conversion.
4. **Held-out probes**: fixed inputs on which function-space updates are observed.
5. **Horizon**: declared checkpoint set K (for example {1,5,10,20}).

## Primary metrics

For held-out model outputs `f^S_t` and `f^C_t`, define update trajectories

`u^S_t = f^S_t - f^S_0`, `u^C_t = f^C_t - f^C_0`.

Then

`D_func = sqrt(sum_t ||u^C_t-u^S_t||^2 / (sum_t ||u^S_t||^2 + eps))`

and

`TQ_traj = 1/(1+D_func)`.

`TQ_traj=1` means identical measured function-update trajectories under the declared continuation contract. It is **not** an information dimension and is not claimed to be invariant to dataset, optimizer, probes, or horizon.

Always report alongside it:
- initial prediction relative error,
- initial gradient cosine when available,
- source/converted final loss,
- conversion compression statistics,
- mean/std across genuinely different data splits or seeds.

## Benchmark tiers

- **Tier A - exact-equivalence stress**: present predictor is exactly identical (e.g. LoRA merge/gauge, function-preserving reparameterization).
- **Tier B - approximate conversion**: quantization, pruning, distillation; report initial prediction error explicitly.
- **Tier C - full-parameter continuation**: all model parameters are trainable after conversion.
- **Tier D - scale stress**: >=3B full causal-LM execution or >=7B long continuation.
- **Tier E - optimizer portability**: repeat a fixed conversion under multiple optimizer/update symmetries.

## Validity rules

- A numerical replicate must change the declared random split/seed in a way that changes data order or stochastic state; duplicate executions are not independent replicates.
- CI wrappers must propagate Python failure (`set -euo pipefail`) and require a result artifact.
- Results must distinguish full-parameter fine-tuning from full-model loss with a restricted trainable subspace.
- Failed/partial runs are not silently averaged into the leaderboard.

The accompanying `tq_benchmark.py` is the reference implementation of the primary trajectory score.
