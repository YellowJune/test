from __future__ import annotations
import argparse, gc, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4, os.cpu_count() or 2))
MID='Qwen/Qwen2.5-7B-Instruct'

class LoRAQ(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8):
        super().__init__(); self.base=base; self.r=r
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A=nn.Parameter(torch.randn(r, base.in_features, dtype=torch.float32)*0.01)
        self.B=nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
    def forward(self,x):
        y=self.base(x)
        z=F.linear(F.linear(x.float(), self.A), self.B)
        return y+z.to(y.dtype)
    @torch.no_grad()
    def delta(self): return self.B.detach().float()@self.A.detach().float()

def seed_all(seed): random.seed(seed); torch.manual_seed(seed)
def qproj_parent(m): return m.model.layers[0].self_attn

def data(tok, seed, max_len=12):
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train')
    texts=[x['text'].strip() for x in ds if len(x['text'].strip())>120]
    random.Random(seed).shuffle(texts); seq=[]
    for t in texts:
        ids=tok(t,return_tensors='pt',truncation=True,max_length=max_len,add_special_tokens=True)['input_ids']
        if ids.shape[1]>=8: seq.append(ids)
        if len(seq)>=16: break
    return seq[:12], seq[12:16]

def deterministic_factor(delta,r,seed=777,oversample=4):
    g=torch.Generator().manual_seed(seed)
    omega=torch.randn(delta.shape[1],r+oversample,generator=g,dtype=torch.float32)
    Q=torch.linalg.qr(delta.float()@omega,mode='reduced').Q
    C=Q.T@delta.float(); U,S,Vh=torch.linalg.svd(C,full_matrices=False)
    root=torch.sqrt(S[:r].clamp_min(1e-15))
    return (Q@U[:,:r])*root.unsqueeze(0), root.unsqueeze(1)*Vh[:r]

def sidecar(A,B):
    d=B.float()@A.float(); B0,A0=deterministic_factor(d,A.shape[0])
    R=torch.linalg.lstsq(B0.double(),B.float().double()).solution
    H=R@R.T; return d,0.5*(H+H.T)

def resume(delta,H):
    B0,A0=deterministic_factor(delta,H.shape[0])
    e,Q=torch.linalg.eigh(H.double()); e=e.clamp_min(1e-15)
    R=(Q*torch.sqrt(e).unsqueeze(0))@Q.T
    Ri=(Q*(1/torch.sqrt(e)).unsqueeze(0))@Q.T
    return (Ri@A0.double()).float(), (B0.double()@R).float()

def gauge(A,B,cond=6.0):
    r=A.shape[0]; M=torch.randn(r,r,dtype=torch.float32)
    U,_,Vh=torch.linalg.svd(M)
    s=torch.logspace(-math.log10(cond)/2,math.log10(cond)/2,r)
    R=U@torch.diag(s)@Vh
    return torch.linalg.solve(R,A.float()), B.float()@R

@torch.no_grad()
def set_ab(w,A,B): w.A.copy_(A); w.B.copy_(B)
@torch.no_grad()
def probe(m,x): return m(input_ids=x,use_cache=False).logits[:,-2:,:].float().cpu()

def grad_step(m,w,x,A,B,lr):
    set_ab(w,A,B)
    if w.A.grad is not None: w.A.grad=None
    if w.B.grad is not None: w.B.grad=None
    loss=m(input_ids=x,labels=x,use_cache=False).loss
    loss.backward()
    with torch.no_grad():
        An=w.A-lr*w.A.grad; Bn=w.B-lr*w.B.grad
    w.A.grad=None; w.B.grad=None
    return An.detach().clone(), Bn.detach().clone(), float(loss.detach())

def state_metrics(As,Bs,Ac,Bc):
    ds=Bs@As; dc=Bc@Ac
    return float((dc-ds).norm()/(ds.norm()+1e-12))

def run(seed=0,steps=200,lr=0.015,rank=8):
    seed_all(seed); t0=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(MID,use_fast=True)
    train,held=data(tok,seed)
    m=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager')
    load_s=time.perf_counter()-t0; m.eval(); m.config.use_cache=False
    for p in m.parameters(): p.requires_grad_(False)
    parent=qproj_parent(m); base=parent.q_proj; w=LoRAQ(base,rank); parent.q_proj=w
    # Warm the actual pretrained LM so both low-rank factors are active.
    A=w.A.detach().clone(); B=w.B.detach().clone(); warm=[]
    for t in range(2): A,B,l=grad_step(m,w,train[t],A,B,lr); warm.append(l)
    A,B=gauge(A,B); d0,H=sidecar(A,B); Ar,Br=resume(d0,H)
    initial_resume=float(((Br@Ar)-(B@A)).norm()/((B@A).norm()+1e-12))
    As,Bs=A.clone(),B.clone(); Ac,Bc=Ar.clone(),Br.clone()
    cps=sorted(set([0,1,10,50,100,steps])); records={}
    set_ab(w,As,Bs); psrc0=probe(m,held[0]); set_ab(w,Ac,Bc); pcar0=probe(m,held[0])
    records['0']={'delta_relerr':state_metrics(As,Bs,Ac,Bc),'probe_logit_relerr':float((pcar0-psrc0).norm()/(psrc0.norm()+1e-12))}
    src_losses=[]; car_losses=[]
    ttrain=time.perf_counter()
    for t in range(1,steps+1):
        x=train[(t-1)%len(train)]
        As,Bs,ls=grad_step(m,w,x,As,Bs,lr)
        Ac,Bc,lc=grad_step(m,w,x,Ac,Bc,lr)
        src_losses.append(ls); car_losses.append(lc)
        if t in cps:
            set_ab(w,As,Bs); ps=probe(m,held[0]); set_ab(w,Ac,Bc); pc=probe(m,held[0])
            records[str(t)]={'delta_relerr':state_metrics(As,Bs,Ac,Bc),'probe_logit_relerr':float((pc-ps).norm()/(ps.norm()+1e-12)),'source_loss':ls,'sidecar_loss':lc,'loss_absdiff':abs(ls-lc)}
    train_s=time.perf_counter()-ttrain
    npars=sum(p.numel() for p in m.parameters())
    out={'model_id':MID,'seed':seed,'test_type':'actual_pretrained_7B_bfloat16_full_causal_lm_loss_200step_lora_sidecar_continuation','parameter_count':int(npars),'rank':rank,'q_proj_shape':list(base.weight.shape),'steps':steps,'lr':lr,'load_seconds':load_s,'training_seconds':train_s,'warm_losses':warm,'initial_sidecar_reconstruction_relerr':initial_resume,'checkpoints':records,'source_losses':src_losses,'sidecar_losses':car_losses,'final_delta_relerr':records[str(steps)]['delta_relerr'],'final_probe_logit_relerr':records[str(steps)]['probe_logit_relerr'],'max_loss_absdiff':float(max(abs(a-b) for a,b in zip(src_losses,car_losses))),'sidecar_scalars':rank*(rank+1)//2,'factor_scalars':rank*(base.in_features+base.out_features),'actual_pretrained_weights_loaded':True,'actual_public_dataset_used':True,'actual_tokenized_text_used':True,'actual_full_causal_lm_loss_backward_used':True,'all_base_parameters_frozen':True,'trainable_target':'layer0_q_proj_rank8_LoRA','note':'Full 7.61B model participates in each causal-LM forward/backward; only the rank-8 layer-0 q-proj adapter is updated, so this is long end-to-end loss continuation rather than full-parameter 7B finetuning.'}
    del m; gc.collect(); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--steps',type=int,default=200); ap.add_argument('--out',required=True); a=ap.parse_args()
    o=Path(a.out);o.mkdir(parents=True,exist_ok=True); r=run(a.seed,a.steps); p=o/f'qwen7b_long_seed{a.seed}.json';p.write_text(json.dumps(r,indent=2),encoding='utf-8'); print(json.dumps(r,indent=2)); print('RESULT_FILE='+str(p))
if __name__=='__main__': main()
