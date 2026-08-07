from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("real_llm_test", HERE / "real_llm_test.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["qwen", "llama", "mistral"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The original protocol calls seed_all internally with a fixed protocol seed.
    # Override that hook so every matrix job preserves the protocol but changes
    # LoRA initialization/gauge randomness reproducibly.
    def seeded(_ignored=0):
        torch.manual_seed(args.seed)
    mod.seed_all = seeded

    if args.target == "qwen":
        result = mod.full_lm_test("qwen", "Qwen/Qwen2.5-0.5B-Instruct", out)
    elif args.target == "llama":
        result = mod.full_lm_test("llama", "unsloth/Llama-3.2-1B-Instruct", out)
    else:
        result = mod.mistral_real_tensor_test(out)

    result["seed"] = args.seed
    path = out / f"{args.target}_seed{args.seed}_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"RESULT_FILE={path}")


if __name__ == "__main__":
    main()
