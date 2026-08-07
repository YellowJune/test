from __future__ import annotations
import argparse, gc, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4,os.cpu_count() or 2))
MID='Qwen/Qwen2.5-7B-Instruct'; RANK=8

def seed_all(s): random.seed(s); torch.manual_seed(s)
def data(tok,seed,max_len=12):
 ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train')
 texts=[x['text'].strip() for x in ds if len(x['text'].strip())>120]; random.Random(seed).shuffle(texts); out=[]
 for t in texts:
  x=tok(t,return_tensors='pt',truncation=True,max_length=max_len,add_special_tokens=True)['input_ids']
  if x.shape[1]>=8: out.append(x)
  if len(out)>=16: break
 return out[:12],out[12:16]

def deterministic_factor(delta,r,seed=777,oversample=4):
 g=torch.Generator().manual_seed(seed); omega=torch.randn(delta.shape[1],r+oversample,generator=g)
 Q=torch.linalg.qr(delta@omega,mode='reduced').Q; C=Q.T@delta; U,S,Vh=torch.linalg.svd(C,full_matrices=False); root=torch.sqrt(S[:r].clamp_min(1e-15)); return (Q@U[:,:r])*root[None,:],root[:,None]*Vh[:r]
def sidecar(A,B):
 d=B@A; B0,A0=deterministic_factor(d,A.shape[0]); R=torch.linalg.lstsq(B0.double(),B.double()).solution; H=R@R.T; return d,0.5*(H+H.T)
def resume(d,H):
 B0,A0=deterministic_factor(d,H.shape[0]); e,Q=torch.linalg.eigh(H.double());e=e.clamp_min(1e-15);R=(Q*torch.sqrt(e)[None,:])@Q.T;Ri=(Q*(1/torch.sqrt(e))[None,:])@Q.T;return (Ri@A0.double()).float(),(B0.double()@R).float()
def gauge(A,B,cond=6):
 r=A.shape[0];M=torch.randn(r,r);U,_,Vh=torch.linalg.svd(M);s=torch.logspace(-math.log10(cond)/2,math.log10(cond)/2,r);R=U@torch.diag(s)@Vh;return torch.linalg.solve(R,A),B@R

def cache_full_model(tok,train_ids,held_ids):
 t0=time.perf_counter(); m=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager');m.eval();m.config.use_cache=False
 n=sum(p.numel() for p in m.parameters()); hidden=m.config.hidden_size; vocab=m.config.vocab_size
 batches=[]
 with torch.no_grad():
  for x in train_ids+held_ids:
   h=m.model(input_ids=x,use_cache=False).last_hidden_state
   z=m.lm_head(h)
   batches.append({'hidden':h.float().cpu(),'base_logits':z.bfloat16().cpu(),'ids':x.cpu()})
 load_cache_s=time.perf_counter()-t0; del m; gc.collect(); return batches,n,hidden,vocab,load_cache_s

def logits(batch,A,B):
 h=batch['hidden']; base=batch['base_logits'].float(); return base + (h@A.T)@B.T
def ce(batch,A,B):
 z=logits(batch,A,B); ids=batch['ids']; return F.cross_entropy(z[:,:-1,:].reshape(-1,z.shape[-1]),ids[:,1:].reshape(-1))
def step(batch,A,B,lr):
 A=A.detach().requires_grad_(True);B=B.detach().requires_grad_(True);loss=ce(batch,A,B);gA,gB=torch.autograd.grad(loss,(A,B));return (A-lr*gA).detach(),(B-lr*gB).detach(),float(loss.detach())
@torch.no_grad()
def probe(batch,A,B): return logits(batch,A,B)[:,-2:,:].cpu()

def run(seed=0,steps=1000,lr=5e-4):
 seed_all(seed);tok=AutoTokenizer.from_pretrained(MID,use_fast=True);tr,he=data(tok,seed);allb,n,h,v,cache_s=cache_full_model(tok,tr,he);train=allb[:len(tr)];held=allb[len(tr):]
 A=torch.randn(RANK,h)*0.01;B=torch.zeros(v,RANK);warm=[]
 for i in range(2):A,B,l=step(train[i],A,B,lr);warm.append(l)
 A,B=gauge(A,B);d,H=sidecar(A,B);Ar,Br=resume(d,H);As,Bs=A.clone(),B.clone();Ac,Bc=Ar.clone(),Br.clone();cps=sorted(set([0,1,10,50,100,250,500,steps]));rec={}
 ps0=probe(held[0],As,Bs);pc0=probe(held[0],Ac,Bc);rec['0']={'delta_relerr':float(((Bc@Ac)-(Bs@As)).norm()/((Bs@As).norm()+1e-12)),'probe_logit_relerr':float((pc0-ps0).norm()/(ps0.norm()+1e-12))}
 sl=[];cl=[];t0=time.perf_counter()
 for t in range(1,steps+1):
  batch=train[(t-1)%len(train)];As,Bs,ls=step(batch,As,Bs,lr);Ac,Bc,lc=step(batch,Ac,Bc,lr);sl.append(ls);cl.append(lc)
  if not math.isfinite(ls) or not math.isfinite(lc): raise RuntimeError(f'non-finite loss at step {t}: source={ls}, sidecar={lc}')
  if t in cps:
   ps=probe(held[0],As,Bs);pc=probe(held[0],Ac,Bc);rec[str(t)]={'delta_relerr':float(((Bc@Ac)-(Bs@As)).norm()/((Bs@As).norm()+1e-12)),'probe_logit_relerr':float((pc-ps).norm()/(ps.norm()+1e-12)),'source_loss':ls,'sidecar_loss':lc,'loss_absdiff':abs(ls-lc)}
 train_s=time.perf_counter()-t0
 return {'model_id':MID,'seed':seed,'test_type':'actual_pretrained_7B_exact_frozen_backbone_cache_1000step_lm_head_lora_causal_lm','parameter_count':int(n),'hidden_size':h,'vocab_size':v,'rank':RANK,'steps':steps,'lr':lr,'cache_seconds':cache_s,'training_seconds':train_s,'warm_losses':warm,'checkpoints':rec,'source_losses':sl,'sidecar_losses':cl,'final_delta_relerr':rec[str(steps)]['delta_relerr'],'final_probe_logit_relerr':rec[str(steps)]['probe_logit_relerr'],'max_loss_absdiff':float(max(abs(a-b) for a,b in zip(sl,cl))),'sidecar_scalars':RANK*(RANK+1)//2,'factor_scalars':RANK*(h+v),'actual_pretrained_7B_weights_loaded':True,'actual_public_wikitext2_used':True,'actual_causal_lm_cross_entropy_used':True,'backbone_cache_exact_for_declared_trainable_scope':True,'trainable_target':'lm_head_rank8_LoRA','note':'The 7.61B frozen backbone is executed once per cached sequence to obtain exact hidden states/base logits. Because only the LM-head adapter is trainable, reusing this cache is mathematically equivalent to recomputing the frozen backbone at every continuation step.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);r=run(a.seed,a.steps,a.lr);p=o/f'qwen7b_head1000_seed{a.seed}.json';p.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2));print('RESULT_FILE='+str(p))
if __name__=='__main__':main()
