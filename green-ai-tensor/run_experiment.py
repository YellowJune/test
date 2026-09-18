import argparse, copy, csv, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


class TuckerConv2d(nn.Module):
    def __init__(self, conv, rank_in, rank_out):
        super().__init__()
        W = conv.weight.detach().float().cpu()
        o, i, kh, kw = W.shape
        ri, ro = max(1, min(rank_in, i)), max(1, min(rank_out, o))
        Uo, _, _ = torch.linalg.svd(W.reshape(o, -1), full_matrices=False)
        Ui, _, _ = torch.linalg.svd(W.permute(1,0,2,3).reshape(i, -1), full_matrices=False)
        Uo, Ui = Uo[:, :ro].contiguous(), Ui[:, :ri].contiguous()
        core = torch.einsum("or,oihw->rihw", Uo, W)
        core = torch.einsum("ij,rjhw->rihw", Ui.T, core).contiguous()
        self.a = nn.Conv2d(i, ri, 1, bias=False)
        self.b = nn.Conv2d(ri, ro, (kh,kw), stride=conv.stride, padding=conv.padding,
                           dilation=conv.dilation, bias=False)
        self.c = nn.Conv2d(ro, o, 1, bias=(conv.bias is not None))
        with torch.no_grad():
            self.a.weight.copy_(Ui.T[:, :, None, None])
            self.b.weight.copy_(core)
            self.c.weight.copy_(Uo[:, :, None, None])
            if conv.bias is not None:
                self.c.bias.copy_(conv.bias.detach().cpu())

    def forward(self, x):
        return self.c(self.b(self.a(x)))


def eligible(conv):
    return isinstance(conv, nn.Conv2d) and conv.groups == 1 and conv.kernel_size != (1,1) and conv.in_channels >= 8 and conv.out_channels >= 8


def est_conv_params(conv, alpha):
    if not eligible(conv):
        return sum(p.numel() for p in conv.parameters(recurse=False))
    o, i, kh, kw = conv.weight.shape
    ri = max(1, min(i, int(round(i*alpha))))
    ro = max(1, min(o, int(round(o*alpha))))
    return i*ri + ri*ro*kh*kw + ro*o + (o if conv.bias is not None else 0)


def est_model_params(model, alpha):
    total = 0
    for m in model.modules():
        if len(list(m.children())) == 0:
            if isinstance(m, nn.Conv2d):
                total += est_conv_params(m, alpha)
            else:
                total += sum(p.numel() for p in m.parameters(recurse=False))
    return total


def find_alpha(model, target_remaining):
    base = count_params(model)
    lo, hi, best = 0.02, 1.0, (1e9, 1.0)
    for _ in range(50):
        mid = (lo + hi)/2
        ratio = est_model_params(model, mid)/base
        err = abs(ratio-target_remaining)
        if err < best[0]: best = (err, mid)
        if ratio > target_remaining: hi = mid
        else: lo = mid
    return best[1]


def replace_convs(module, alpha):
    for name, child in list(module.named_children()):
        if eligible(child):
            ri = max(1, min(child.in_channels, int(round(child.in_channels*alpha))))
            ro = max(1, min(child.out_channels, int(round(child.out_channels*alpha))))
            setattr(module, name, TuckerConv2d(child, ri, ro))
        else:
            replace_convs(child, alpha)


def compress(model, reduction):
    out = copy.deepcopy(model).cpu()
    alpha = find_alpha(out, 1.0-reduction)
    replace_convs(out, alpha)
    return out, alpha


def make_loaders(data_dir, train_bs, eval_bs, workers):
    mean=(0.4914,0.4822,0.4465); std=(0.2023,0.1994,0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean,std)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    tr = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    te = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return (
        DataLoader(tr, batch_size=train_bs, shuffle=True, num_workers=workers),
        DataLoader(te, batch_size=eval_bs, shuffle=False, num_workers=workers),
        te
    )


def finetune(model, loader, device, epochs, lr):
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs))
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); total=0.0; n=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y); loss.backward(); opt.step()
            total += loss.item()*y.numel(); n += y.numel()
        sched.step()
        print(f"finetune epoch {ep+1}/{epochs}: loss={total/n:.4f}")
    return model


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval().to(device); c=n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        c += (model(x).argmax(1)==y).sum().item(); n += y.numel()
    return 100.0*c/n


def rapl_energy_file():
    roots = list(Path("/sys/class/powercap").glob("**/energy_uj")) if Path("/sys/class/powercap").exists() else []
    for p in roots:
        try:
            name=(p.parent/"name").read_text().strip().lower()
        except Exception:
            name=""
        if "package" in name or "pkg" in name:
            return p
    return roots[0] if roots else None


def read_uj(p):
    try: return int(p.read_text().strip())
    except Exception: return None


@torch.no_grad()
def benchmark(model, dataset, device, n_images=2000, warmup=200):
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    model.eval().to(device)
    it=iter(loader)
    for _ in range(warmup):
        try: x,_=next(it)
        except StopIteration: it=iter(loader); x,_=next(it)
        _=model(x.to(device))
    energy_path=rapl_energy_file() if device.type=="cpu" else None
    e0=read_uj(energy_path) if energy_path else None
    t0=time.perf_counter(); done=0
    it=iter(loader)
    while done<n_images:
        try: x,_=next(it)
        except StopIteration: it=iter(loader); x,_=next(it)
        _=model(x.to(device)); done += x.shape[0]
    t1=time.perf_counter()
    e1=read_uj(energy_path) if energy_path else None
    latency_ms=(t1-t0)*1000.0/done
    joules=float("nan")
    if e0 is not None and e1 is not None and e1>=e0:
        joules=((e1-e0)/1e6)/done
    return latency_ms, joules, str(energy_path) if energy_path else ""


def save_csv(path, rows):
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",default="outputs")
    ap.add_argument("--data-dir",default="data")
    ap.add_argument("--finetune-epochs",type=int,default=5)
    ap.add_argument("--train-batch-size",type=int,default=256)
    ap.add_argument("--eval-batch-size",type=int,default=512)
    ap.add_argument("--benchmark-images",type=int,default=2000)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--grid-gco2-kwh",type=float,default=0.0)
    args=ap.parse_args()

    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cpu")
    print("Loading pretrained CIFAR-10 ResNet20...")
    base=torch.hub.load("chenyaofo/pytorch-cifar-models","cifar10_resnet20",pretrained=True,trust_repo=True)
    train_loader,test_loader,test_ds=make_loaders(args.data_dir,args.train_batch_size,args.eval_batch_size,args.workers)

    models={"Original":copy.deepcopy(base).cpu()}
    alphas={"Original":1.0}
    for label,red in [("50% tensor decomposition",0.50),("75% over-compression",0.75)]:
        m,a=compress(base,red)
        print(label, "alpha=",a, "params=",count_params(m))
        m=finetune(m,train_loader,device,args.finetune_epochs,0.01)
        models[label]=m.cpu(); alphas[label]=a

    base_params=count_params(models["Original"])
    rows=[]
    for label,m in models.items():
        acc=accuracy(m,test_loader,device)
        lat,j,rapl=benchmark(m,test_ds,device,args.benchmark_images)
        carbon=float("nan")
        if args.grid_gco2_kwh>0 and not math.isnan(j):
            carbon=(j/3.6e6)*args.grid_gco2_kwh*1000.0
        rows.append({
            "model":label,
            "params":count_params(m),
            "params_m":count_params(m)/1e6,
            "remaining_params_pct":100*count_params(m)/base_params,
            "accuracy_pct":acc,
            "latency_ms_per_inf":lat,
            "energy_j_per_inf":j,
            "carbon_mgco2_per_inf":carbon,
            "rank_alpha":alphas[label],
            "energy_source":rapl if rapl else "N/A"
        })
        print(rows[-1])

    base_e=rows[0]["energy_j_per_inf"]
    for r in rows:
        r["energy_ratio_pct"]=(100*r["energy_j_per_inf"]/base_e) if not math.isnan(base_e) and not math.isnan(r["energy_j_per_inf"]) else float("nan")
    save_csv(out/"results.csv",rows)

    md=["# Green AI Tensor-Decomposition Results","",
        "|Model|Params (M)|Remaining params|Accuracy|Latency (ms/inf)|Energy (J/inf)|",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        e="N/A" if math.isnan(r["energy_j_per_inf"]) else f'{r["energy_j_per_inf"]:.6f}'
        md.append(f'|{r["model"]}|{r["params_m"]:.3f}|{r["remaining_params_pct"]:.1f}%|{r["accuracy_pct"]:.2f}%|{r["latency_ms_per_inf"]:.4f}|{e}|')
    md += ["","Energy is reported only when Linux RAPL is exposed by the runner; otherwise it is intentionally N/A.",
           "Baseline weights are loaded from chenyaofo/pytorch-cifar-models via torch.hub."]
    (out/"report.md").write_text("\n".join(md),encoding="utf-8")
    (out/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
