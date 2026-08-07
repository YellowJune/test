from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4, os.cpu_count() or 2))

PROMPTS=[
 'A trainable model artifact must specify how future evidence changes its parameters.',
 'Optimization geometry can change when a model is converted between equivalent representations.',
 'Low rank adapters are often merged into dense weights before deployment.',
 'Adaptive optimizers keep hidden state that may matter when training resumes.',
 'Mixture of experts models route each token to a small subset of expert networks.',
 'Knowledge distillation transfers information from a teacher to a smaller student.'
]

def seed_all(s): torch.manual_seed(s)

def encode(tok,max_len=24):
 out=[]
 for t in PROMPTS:
  ids=tok(t,return_tensors='pt',truncation=True,max_length=max_len)['input_ids']
  if ids.shape[1]>=4: out.append(ids)
 return out

def get_q(model): return model.model.layers[0].self_attn.q_proj

def dense_grad_full(model,qmod,ids):
 qmod.weight.requires_grad_(True)
 if qmod.weight.grad is not None: qmod.weight.grad=None
 out=model(input_ids=ids,labels=ids,use_cache=False)
 out.loss.backward()
 g=qmod.weight.grad.detach().float().cpu().clone()
 qmod.weight.grad=None
 qmod.weight.requires_grad_(False)
 return g,float(out.loss.detach())

def rmsnorm(x,w,eps=1e-5):
 x=x.float(); return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*w.float()

def mistral_real_gradients(seed):
 repo='mistralai/Mistral-7B-Instruct-v0.3'
 tok=AutoTokenizer.from_pretrained(repo,use_fast=True)
 idxp=hf_hub_download(repo,'model.safetensors.index.json')
 wm=json.load(open(idxp))['weight_map']
 keys=['model.embed_tokens.weight','model.layers.0.input_layernorm.weight','model.layers.0.self_attn.q_proj.weight']
 local={}
 for sh in sorted({wm[k] for k in keys}): local[sh]=hf_hub_download(repo,sh)
 tens={}
 for k in keys:
  with safe_open(local[wm[k]],framework='pt',device='cpu') as f: tens[k]=f.get_tensor(k).float().contiguous()
 E=tens[keys[0]]; nw=tens[keys[1]]; W=tens[keys[2]]
 grads=[]; losses=[]
 for ids in encode(tok):
  h=rmsnorm(F.embedding(ids,E),nw).detach()
  Wv=W.detach().clone().requires_grad_(True)
  q=F.linear(h,Wv)
  loss=.5*q.pow(2).mean(); (g,)=torch.autograd.grad(loss,Wv)
  grads.append(g.detach().cpu()); losses.append(float(loss.detach()))
  if len(grads)>=5: break
 return grads,losses,list(W.shape),repo

def full_lm_gradients(model_id):
 tok=AutoTokenizer.from_pretrained(model_id,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager')
 model.config.use_cache=False; model.eval()
 for p in model.parameters(): p.requires_grad_(False)
 q=get_q(model)
 grads=[]; losses=[]
 for ids in encode(tok):
  g,l=dense_grad_full(model,q,ids); grads.append(g); losses.append(l)
  if len(grads)>=5: break
 shape=list(q.weight.shape); n=int(sum(p.numel() for p in model.parameters()))
 del model; gc.collect()
 return grads,losses,shape,model_id,n

def orthogonal(r,gen):
 M=torch.randn(r,r,generator=gen,dtype=torch.float64); Q,_=torch.linalg.qr(M); return Q

def init_factors(dout,din,r,gen):
 A=torch.randn(r,din,generator=gen,dtype=torch.float64)*.01
 B=torch.randn(dout,r,generator=gen,dtype=torch.float64)*.01
 return A,B

def polar(g):
 U,_,Vh=torch.linalg.svd(g,full_matrices=False); return U@Vh

def inv_quarter(S,eps=1e-8):
 e,Q=torch.linalg.eigh(.5*(S+S.T)); e=e.clamp_min(eps); return (Q*(e.pow(-.25)).unsqueeze(0))@Q.T

def step(method,A,B,state,G,lr=.002,b1=.9,b2=.999,eps=1e-8,wd=.01):
 gB=G@A.T; gA=B.T@G
 if method=='sgd':
  dA,dB=gA,gB
 elif method in ('momentum','nesterov'):
  mA=state.get('mA',torch.zeros_like(A)); mB=state.get('mB',torch.zeros_like(B))
  mA=b1*mA+gA; mB=b1*mB+gB; state['mA']=mA; state['mB']=mB
  if method=='momentum': dA,dB=mA,mB
  else: dA,dB=gA+b1*mA,gB+b1*mB
 elif method=='global_rms':
  dA=gA/(gA.pow(2).mean().sqrt()+eps); dB=gB/(gB.pow(2).mean().sqrt()+eps)
 elif method=='adamw':
  mA=state.get('mA',torch.zeros_like(A)); mB=state.get('mB',torch.zeros_like(B)); vA=state.get('vA',torch.zeros_like(A)); vB=state.get('vB',torch.zeros_like(B))
  mA=b1*mA+(1-b1)*gA; mB=b1*mB+(1-b1)*gB; vA=b2*vA+(1-b2)*gA.square(); vB=b2*vB+(1-b2)*gB.square()
  state.update(mA=mA,mB=mB,vA=vA,vB=vB)
  dA=mA/(vA.sqrt()+eps)+wd*A; dB=mB/(vB.sqrt()+eps)+wd*B
 elif method=='lion':
  mA=state.get('mA',torch.zeros_like(A)); mB=state.get('mB',torch.zeros_like(B))
  uA=(b1*mA+(1-b1)*gA).sign(); uB=(b1*mB+(1-b1)*gB).sign()
  state['mA']=b2*mA+(1-b2)*gA; state['mB']=b2*mB+(1-b2)*gB; dA=uA+wd*A; dB=uB+wd*B
 elif method=='muon_core':
  dA=polar(gA); dB=polar(gB)
 elif method=='shampoo_core':
  RA=gA@gA.T; RB=gB.T@gB
  dA=inv_quarter(RA+eps*torch.eye(RA.shape[0],dtype=RA.dtype))@gA
  dB=gB@inv_quarter(RB+eps*torch.eye(RB.shape[0],dtype=RB.dtype))
 else: raise ValueError(method)
 return A-lr*dA,B-lr*dB,state

def replay(grads,shape,seed):
 dout,din=shape; r=8; gen=torch.Generator().manual_seed(5000+seed)
 A0,B0=init_factors(dout,din,r,gen); st={}
 for G in grads[:2]: A0,B0,st=step('sgd',A0,B0,st,G.double(),lr=.01)
 Q=orthogonal(r,gen); A1=Q.T@A0; B1=B0@Q
 methods=['sgd','momentum','nesterov','global_rms','adamw','lion','muon_core','shampoo_core']
 out=[]
 for m in methods:
  A,B=A0.clone(),B0.clone(); Ap,Bp=A1.clone(),B1.clone(); s={}; sp={}; start=B@A
  for G in grads[2:]:
   G=G.double(); A,B,s=step(m,A,B,s,G); Ap,Bp,sp=step(m,Ap,Bp,sp,G)
  D=B@A; Dp=Bp@Ap; denom=(D-start).norm()+1e-18; disc=float((Dp-D).norm()/denom)
  reset_dim=r*(r+1)//2 if m in ['sgd','momentum','nesterov','global_rms','muon_core','shampoo_core'] else r*r
  factor_scalars=r*(din+dout)
  if m in ('sgd','global_rms','muon_core','shampoo_core'): state_buffers=0
  elif m in ('momentum','nesterov','lion'): state_buffers=factor_scalars
  else: state_buffers=2*factor_scalars
  stateful_upper=reset_dim+state_buffers
  out.append(dict(optimizer=m,discrepancy=disc,reset_portability_dim=reset_dim,state_buffer_scalars=state_buffers,stateful_portability_upper_scalars=stateful_upper,factor_scalars=factor_scalars,reset_fraction=reset_dim/factor_scalars,stateful_fraction=stateful_upper/max(factor_scalars,1)))
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--target',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--output',required=True); args=ap.parse_args(); seed_all(100+args.seed)
 if args.target=='qwen': grads,losses,shape,mid,n=full_lm_gradients('Qwen/Qwen2.5-0.5B-Instruct'); test='full_lm_ce'
 elif args.target=='llama': grads,losses,shape,mid,n=full_lm_gradients('unsloth/Llama-3.2-1B-Instruct'); test='full_lm_ce'
 elif args.target=='mistral': grads,losses,shape,mid=mistral_real_gradients(args.seed); n=None; test='real_embedding_rmsnorm_q_energy'
 else: raise ValueError(args.target)
 result=dict(target=args.target,seed=args.seed,model_id=mid,test_type=test,q_proj_shape=shape,parameter_count=n,gradient_losses=losses,actual_pretrained_weights_loaded=True,optimizer_results=replay(grads,shape,args.seed))
 od=Path(args.output); od.mkdir(parents=True,exist_ok=True); p=od/f'portability_{args.target}_seed{args.seed}.json'; p.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
