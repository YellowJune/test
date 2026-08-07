from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4,os.cpu_count() or 2))
MID='ministral/Ministral-3b-instruct'
PROMPTS=[
 'A trainable model is characterized by predictions and by how optimization changes those predictions.',
 'Low rank adaptation modifies a pretrained projection using two trainable factor matrices.',
 'Future learning trajectories can diverge after an inference preserving conversion.',
 'Optimization geometry depends on the parameterization used to represent the same function.',
 'Model conversion should be evaluated by both inference fidelity and continuation fidelity.'
]

def seed_all(s): torch.manual_seed(s)
def qproj(m): return m.model.layers[0].self_attn.q_proj
class LoRA(nn.Module):
 def __init__(self,base,r=8):
  super().__init__(); self.base=base; self.r=r
  for p in base.parameters(): p.requires_grad_(False)
  self.A=nn.Parameter(torch.randn(r,base.in_features,dtype=base.weight.dtype)*0.01)
  self.B=nn.Parameter(torch.zeros(base.out_features,r,dtype=base.weight.dtype))
 def forward(self,x): return self.base(x)+F.linear(F.linear(x,self.A),self.B)
 @torch.no_grad()
 def delta(self): return self.B.float()@self.A.float()

def deterministic_factor(delta,r,seed=777,oversample=4):
 g=torch.Generator().manual_seed(seed); omega=torch.randn(delta.shape[1],r+oversample,generator=g,dtype=torch.float32)
 Q=torch.linalg.qr(delta.float()@omega,mode='reduced').Q; C=Q.T@delta.float(); U,S,Vh=torch.linalg.svd(C,full_matrices=False); root=torch.sqrt(S[:r].clamp_min(1e-15)); return (Q@U[:,:r])*root.unsqueeze(0),root.unsqueeze(1)*Vh[:r]
def sidecar(A,B):
 d=B.float()@A.float(); B0,A0=deterministic_factor(d,A.shape[0]); R=torch.linalg.lstsq(B0.double(),B.float().double()).solution; H=R@R.T; return d,0.5*(H+H.T)
def resume(delta,H,dtype):
 B0,A0=deterministic_factor(delta,H.shape[0]); e,Q=torch.linalg.eigh(H); e=e.clamp_min(1e-15); R=(Q*torch.sqrt(e).unsqueeze(0))@Q.T; Ri=(Q*(1/torch.sqrt(e)).unsqueeze(0))@Q.T; return (Ri@A0.double()).to(dtype),(B0.double()@R).to(dtype)
def gauge(A,B,cond=4):
 r=A.shape[0]; M=torch.randn(r,r,dtype=torch.float32); U,_,Vh=torch.linalg.svd(M); s=torch.logspace(-math.log10(cond)/2,math.log10(cond)/2,r); R=U@torch.diag(s)@Vh; return torch.linalg.solve(R,A.float()).to(A.dtype),(B.float()@R).to(B.dtype)
def ids(tok,t): return tok(t,return_tensors='pt',truncation=True,max_length=12,add_special_tokens=True)['input_ids']
def step(m,ps,x,lr):
 for p in ps:
  if p.grad is not None:p.grad=None
 loss=m(input_ids=x,labels=x,use_cache=False).loss; loss.backward()
 with torch.no_grad():
  for p in ps:p.add_(p.grad,alpha=-lr)
 return float(loss.detach())
@torch.no_grad()
def probe(m,x): return m(input_ids=x,use_cache=False).logits[:,-2:,:].float().cpu()

def run(seed=0,cont=10):
 seed_all(seed); tok=AutoTokenizer.from_pretrained(MID,use_fast=True); batches=[ids(tok,t) for t in PROMPTS]
 tload=time.perf_counter(); m=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager'); load_s=time.perf_counter()-tload; m.eval(); m.config.use_cache=False
 for p in m.parameters():p.requires_grad_(False)
 parent=m.model.layers[0].self_attn; base=parent.q_proj; w=LoRA(base,8); parent.q_proj=w
 lr=0.02
 warm=[step(m,[w.A,w.B],batches[0],lr)]
 with torch.no_grad(): A,B=gauge(w.A.detach(),w.B.detach()); w.A.copy_(A);w.B.copy_(B)
 d0,H=sidecar(w.A.detach(),w.B.detach()); source_losses=[]
 for t in range(cont):source_losses.append(step(m,[w.A,w.B],batches[1+t%3],lr))
 source_delta=w.delta().cpu(); source_logits=probe(m,batches[-1])
 Ar,Br=resume(d0,H,w.A.dtype)
 with torch.no_grad():w.A.copy_(Ar);w.B.copy_(Br)
 side_losses=[]
 for t in range(cont):side_losses.append(step(m,[w.A,w.B],batches[1+t%3],lr))
 side_delta=w.delta().cpu(); side_logits=probe(m,batches[-1]); dnorm=source_delta.norm()+1e-12; lnorm=source_logits.norm()+1e-12
 return {'model_id':MID,'seed':seed,'test_type':'full_pretrained_3B_bfloat16_causal_lm_cross_entropy','parameter_count':int(sum(p.numel() for p in m.parameters())),'q_proj_shape':list(base.weight.shape),'dtype':'bfloat16','load_seconds':load_s,'warm_losses':warm,'continuation_steps':cont,'source_losses':source_losses,'sidecar_losses':side_losses,'sidecar_delta_relerr':float((side_delta-source_delta).norm()/dnorm),'sidecar_logit_relerr':float((side_logits-source_logits).norm()/lnorm),'sidecar_scalars':36,'factor_scalars':8*(base.in_features+base.out_features),'actual_pretrained_weights_loaded':True,'actual_tokenized_text_used':True,'actual_language_model_loss_used':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=0);ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);r=run(a.seed);p=o/f'ministral3b_seed{a.seed}.json';p.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2));print('RESULT_FILE='+str(p))
if __name__=='__main__':main()
