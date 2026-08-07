from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 2))

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. Neural networks learn from examples.",
    "Paris is the capital of France, and Tokyo is the capital of Japan.",
    "A compact model conversion can preserve predictions while changing future learning dynamics.",
    "Optimization follows gradients, but parameterization can alter the induced update in function space.",
    "Low rank adaptation represents a weight update as a product of two smaller trainable matrices.",
    "Machine learning systems should record enough provenance to reproduce future training behavior.",
]


def seed_all(seed: int = 0):
    torch.manual_seed(seed)


def get_parent_qproj(model):
    # Qwen2 and Llama/Mistral-family Hugging Face models expose this path.
    layers = model.model.layers
    parent = layers[0].self_attn
    return parent, "q_proj", parent.q_proj


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, A=None, B=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        if A is None:
            A = torch.randn(rank, base.in_features, dtype=base.weight.dtype) * 0.01
        if B is None:
            B = torch.zeros(base.out_features, rank, dtype=base.weight.dtype)
        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())

    def forward(self, x):
        # Avoid forming the dense delta during ordinary training.
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)

    @torch.no_grad()
    def delta(self):
        return self.B @ self.A


def canonical_from_delta(delta: torch.Tensor, rank: int):
    # Exact dense SVD is intentionally used at the resume boundary. This is a
    # one-time conversion cost and makes the sidecar definition self-contained.
    U, S, Vh = torch.linalg.svd(delta.double(), full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank].clamp_min(1e-18), Vh[:rank]
    root = torch.sqrt(S)
    B0 = U * root.unsqueeze(0)
    A0 = root.unsqueeze(1) * Vh
    return B0, A0


def build_sidecar(A: torch.Tensor, B: torch.Tensor):
    delta = (B.double() @ A.double()).contiguous()
    B0, A0 = canonical_from_delta(delta, A.shape[0])
    R = torch.linalg.lstsq(B0, B.double()).solution
    H = R @ R.T
    H = 0.5 * (H + H.T)
    return delta, H


def resume_from_sidecar(delta: torch.Tensor, H: torch.Tensor, dtype):
    rank = H.shape[0]
    B0, A0 = canonical_from_delta(delta, rank)
    evals, Q = torch.linalg.eigh(H)
    evals = evals.clamp_min(1e-18)
    R = (Q * torch.sqrt(evals).unsqueeze(0)) @ Q.T
    Rinv = (Q * (1.0 / torch.sqrt(evals)).unsqueeze(0)) @ Q.T
    B = B0 @ R
    A = Rinv @ A0
    return A.to(dtype), B.to(dtype)


def gauge_factors(A: torch.Tensor, B: torch.Tensor, condition: float = 4.0):
    r = A.shape[0]
    M = torch.randn(r, r, dtype=A.dtype)
    U, _, Vh = torch.linalg.svd(M)
    scales = torch.logspace(-math.log10(condition) / 2, math.log10(condition) / 2, r, dtype=A.dtype)
    R = U @ torch.diag(scales) @ Vh
    Bg = B @ R
    Ag = torch.linalg.solve(R, A)
    return Ag, Bg


def encode_batches(tokenizer, max_len=24):
    batches = []
    for text in PROMPTS:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
        ids = enc["input_ids"]
        if ids.shape[1] < 4:
            continue
        batches.append(ids)
    return batches


def lm_step(model, params, ids, lr):
    for p in params:
        if p.grad is not None:
            p.grad = None
    out = model(input_ids=ids, labels=ids, use_cache=False)
    loss = out.loss
    loss.backward()
    with torch.no_grad():
        for p in params:
            p.add_(p.grad, alpha=-lr)
    return float(loss.detach())


@torch.no_grad()
def probe_logits(model, ids, tail=4):
    out = model(input_ids=ids, use_cache=False).logits[:, -tail:, :].float().cpu()
    return out


def full_lm_test(target: str, model_id: str, out_dir: Path):
    seed_all(123)
    rank = 8
    lr = 0.02
    warm_steps = 2
    cont_steps = 3
    print(f"[load] {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    parent, name, base = get_parent_qproj(model)
    base_weight = base.weight.detach().clone()
    base_bias = base.bias.detach().clone() if base.bias is not None else None
    wrapper = LoRALinear(base, rank)
    setattr(parent, name, wrapper)

    batches = encode_batches(tokenizer)
    assert len(batches) >= warm_steps + cont_steps + 1

    warm_losses = []
    for i in range(warm_steps):
        warm_losses.append(lm_step(model, [wrapper.A, wrapper.B], batches[i], lr))

    with torch.no_grad():
        A0, B0 = gauge_factors(wrapper.A.detach().clone(), wrapper.B.detach().clone(), condition=4.0)
        wrapper.A.copy_(A0)
        wrapper.B.copy_(B0)

    t0 = time.perf_counter()
    delta0, H = build_sidecar(wrapper.A.detach(), wrapper.B.detach())
    sidecar_build_s = time.perf_counter() - t0
    sidecar_scalars = rank * (rank + 1) // 2
    factor_scalars = rank * (base.in_features + base.out_features)

    # SOURCE: continue factor-SGD from the gauged source factors.
    source_A, source_B = wrapper.A.detach().clone(), wrapper.B.detach().clone()
    source_losses = []
    for j in range(cont_steps):
        source_losses.append(lm_step(model, [wrapper.A, wrapper.B], batches[warm_steps + j], lr))
    source_delta = wrapper.delta().detach().double().cpu()
    source_logits = probe_logits(model, batches[-1])

    # SIDECAR: reset to a representative reconstructed only from merged delta + H.
    t0 = time.perf_counter()
    Ar, Br = resume_from_sidecar(delta0, H, wrapper.A.dtype)
    resume_s = time.perf_counter() - t0
    with torch.no_grad():
        wrapper.A.copy_(Ar)
        wrapper.B.copy_(Br)
    side_losses = []
    for j in range(cont_steps):
        side_losses.append(lm_step(model, [wrapper.A, wrapper.B], batches[warm_steps + j], lr))
    side_delta = wrapper.delta().detach().double().cpu()
    side_logits = probe_logits(model, batches[-1])

    # NAIVE MERGED DENSE SGD: same current inference weight, but discard factor semantics.
    setattr(parent, name, base)
    with torch.no_grad():
        base.weight.copy_(base_weight + delta0.to(base_weight.dtype))
        if base_bias is not None:
            base.bias.copy_(base_bias)
    base.weight.requires_grad_(True)
    naive_losses = []
    for j in range(cont_steps):
        naive_losses.append(lm_step(model, [base.weight], batches[warm_steps + j], lr))
    naive_delta = (base.weight.detach().double().cpu() - base_weight.double().cpu())
    naive_logits = probe_logits(model, batches[-1])

    source_norm = source_delta.norm() + 1e-18
    source_logit_norm = source_logits.norm() + 1e-18
    result = {
        "target": target,
        "model_id": model_id,
        "test_type": "full_pretrained_causal_lm_cross_entropy",
        "rank": rank,
        "lr": lr,
        "warm_steps": warm_steps,
        "continuation_steps": cont_steps,
        "q_proj_shape": list(base_weight.shape),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "warm_losses": warm_losses,
        "source_losses": source_losses,
        "sidecar_losses": side_losses,
        "naive_losses": naive_losses,
        "sidecar_delta_relerr": float((side_delta - source_delta).norm() / source_norm),
        "naive_delta_relerr": float((naive_delta - source_delta).norm() / source_norm),
        "sidecar_logit_relerr": float((side_logits - source_logits).norm() / source_logit_norm),
        "naive_logit_relerr": float((naive_logits - source_logits).norm() / source_logit_norm),
        "sidecar_build_seconds": sidecar_build_s,
        "sidecar_resume_seconds": resume_s,
        "sidecar_scalars": sidecar_scalars,
        "factor_scalars": factor_scalars,
        "sidecar_fraction_of_factors": sidecar_scalars / factor_scalars,
        "actual_pretrained_weights_loaded": True,
        "actual_tokenized_text_used": True,
        "actual_language_model_loss_used": True,
    }
    return result


def rmsnorm(x, weight, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def locate_mistral_tensors(repo_id: str):
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    wm = index["weight_map"]
    keys = {
        "embed": "model.embed_tokens.weight",
        "norm": "model.layers.0.input_layernorm.weight",
        "q": "model.layers.0.self_attn.q_proj.weight",
    }
    shards = {k: wm[v] for k, v in keys.items()}
    local = {}
    for shard in sorted(set(shards.values())):
        local[shard] = hf_hub_download(repo_id, shard)
    tensors = {}
    for short, full_key in keys.items():
        shard = shards[short]
        with safe_open(local[shard], framework="pt", device="cpu") as f:
            tensors[short] = f.get_tensor(full_key).float().contiguous()
    return tensors, shards


def local_loss_step(A, B, W, hidden, lr):
    A = A.detach().clone().requires_grad_(True)
    B = B.detach().clone().requires_grad_(True)
    q = F.linear(hidden, W) + F.linear(F.linear(hidden, A), B)
    # A deterministic, weight-dependent local objective on real pretrained activations.
    loss = 0.5 * q.float().pow(2).mean()
    gA, gB = torch.autograd.grad(loss, (A, B))
    with torch.no_grad():
        A -= lr * gA
        B -= lr * gB
    return A.detach(), B.detach(), float(loss.detach())


def dense_local_step(W_eff, hidden, lr):
    Wv = W_eff.detach().clone().requires_grad_(True)
    q = F.linear(hidden, Wv)
    loss = 0.5 * q.float().pow(2).mean()
    (gW,) = torch.autograd.grad(loss, Wv)
    with torch.no_grad():
        Wv -= lr * gW
    return Wv.detach(), float(loss.detach())


def mistral_real_tensor_test(out_dir: Path):
    seed_all(321)
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    rank = 8
    lr = 0.05
    warm_steps = 2
    cont_steps = 4
    print(f"[load selected real tensors] {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    tensors, shards = locate_mistral_tensors(repo_id)
    E, norm_w, W = tensors["embed"], tensors["norm"], tensors["q"]
    batches = encode_batches(tokenizer, max_len=24)

    hidden_batches = []
    for ids in batches:
        h = F.embedding(ids, E)
        h = rmsnorm(h, norm_w)
        hidden_batches.append(h.detach())

    A = torch.randn(rank, W.shape[1]) * 0.01
    B = torch.zeros(W.shape[0], rank)
    warm_losses = []
    for i in range(warm_steps):
        A, B, loss = local_loss_step(A, B, W, hidden_batches[i], lr)
        warm_losses.append(loss)
    A, B = gauge_factors(A, B, condition=4.0)

    t0 = time.perf_counter()
    delta0, H = build_sidecar(A, B)
    build_s = time.perf_counter() - t0

    As, Bs = A.clone(), B.clone()
    source_losses = []
    for j in range(cont_steps):
        As, Bs, loss = local_loss_step(As, Bs, W, hidden_batches[warm_steps + j], lr)
        source_losses.append(loss)
    source_delta = (Bs.double() @ As.double())

    t0 = time.perf_counter()
    Ar, Br = resume_from_sidecar(delta0, H, torch.float32)
    resume_s = time.perf_counter() - t0
    side_losses = []
    for j in range(cont_steps):
        Ar, Br, loss = local_loss_step(Ar, Br, W, hidden_batches[warm_steps + j], lr)
        side_losses.append(loss)
    side_delta = Br.double() @ Ar.double()

    Wn = W + delta0.float()
    naive_losses = []
    for j in range(cont_steps):
        Wn, loss = dense_local_step(Wn, hidden_batches[warm_steps + j], lr)
        naive_losses.append(loss)
    naive_delta = Wn.double() - W.double()

    probe_h = hidden_batches[-1]
    with torch.no_grad():
        source_q = F.linear(probe_h, W) + F.linear(F.linear(probe_h, As), Bs)
        side_q = F.linear(probe_h, W) + F.linear(F.linear(probe_h, Ar), Br)
        naive_q = F.linear(probe_h, Wn)
    dnorm = source_delta.norm() + 1e-18
    qnorm = source_q.norm() + 1e-18
    sidecar_scalars = rank * (rank + 1) // 2
    factor_scalars = rank * (W.shape[0] + W.shape[1])
    return {
        "target": "mistral",
        "model_id": repo_id,
        "test_type": "real_pretrained_embedding_rmsnorm_qproj_local_objective",
        "rank": rank,
        "lr": lr,
        "warm_steps": warm_steps,
        "continuation_steps": cont_steps,
        "q_proj_shape": list(W.shape),
        "downloaded_shards": shards,
        "warm_losses": warm_losses,
        "source_losses": source_losses,
        "sidecar_losses": side_losses,
        "naive_losses": naive_losses,
        "sidecar_delta_relerr": float((side_delta - source_delta).norm() / dnorm),
        "naive_delta_relerr": float((naive_delta - source_delta).norm() / dnorm),
        "sidecar_local_q_relerr": float((side_q - source_q).norm() / qnorm),
        "naive_local_q_relerr": float((naive_q - source_q).norm() / qnorm),
        "sidecar_build_seconds": build_s,
        "sidecar_resume_seconds": resume_s,
        "sidecar_scalars": sidecar_scalars,
        "factor_scalars": factor_scalars,
        "sidecar_fraction_of_factors": sidecar_scalars / factor_scalars,
        "actual_pretrained_weights_loaded": True,
        "actual_tokenized_text_used": True,
        "actual_language_model_loss_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["qwen", "llama", "mistral"], required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.target == "qwen":
        result = full_lm_test("qwen", "Qwen/Qwen2.5-0.5B-Instruct", out_dir)
    elif args.target == "llama":
        # Open mirror of Meta Llama 3.2 1B weights; avoids gated-token requirements on CI.
        result = full_lm_test("llama", "unsloth/Llama-3.2-1B-Instruct", out_dir)
    else:
        result = mistral_real_tensor_test(out_dir)

    path = out_dir / f"{args.target}_real_llm_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"RESULT_FILE={path}")


if __name__ == "__main__":
    main()
