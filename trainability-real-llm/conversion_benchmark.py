from __future__ import annotations

import argparse, gc, json, math, os, random
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4, os.cpu_count() or 2))

PROMPTS=[
 'A model conversion can preserve predictions while changing how the model learns from the next batch.',
 'Optimization in parameter space induces a trajectory in function space.',
 'Quantization reduces numerical precision and pruning removes selected parameters.',
 'Neural language models adapt to new data through gradient based optimization.',
 'The capital of France is Paris and the capital of Japan is Tokyo.',
 'Low rank adaptation changes a pretrained model using a small trainable update.',
 'Future learning behavior is part of a trainable artifact contract.',
 'Machine learning evaluation should distinguish inference fidelity from continuation fidelity.'
]
MODEL_IDS={'qwen':'Qwen/Qwen2.5-0.5B-Instruct','llama':'unsloth/Llama-3.2-1B-Instruct'}

def seed_all(seed): random.seed(seed); torch.manual_seed(seed)
def encode(tok,max_len=16):
 out=[]
 for t in PROMPTS:
  ids=tok(t,return_tensors='pt',truncation=True,max_length=max_len,add_special_tokens=True)['input_ids']
  if ids.shape[1]>=5: out.append(ids)
 return out
def qproj(model): return model.model.layers[0].self_attn.q_proj
def freeze_except_q(model):
 for p in model.parameters(): p.requires_grad_(False)
 q=qproj(model); q.weight.requires_grad_(True); return q

def group_quantize_dequantize_(w,bits,group=64):
 qmax=2**(bits-1)-1
 with torch.no_grad():
  m=w.view(w.shape[0],-1)
  for j in range(0,m.shape[1],group):
   x=m[:,j:j+group]; scale=x.abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/qmax
   x.copy_((x/scale).round().clamp(-qmax,qmax)*scale)
def quantize_model_(model,bits):
 count=0
 for mod in model.modules():
  if isinstance(mod,nn.Linear): group_quantize_dequantize_(mod.weight,bits); count+=mod.weight.numel()
 return count
def prune_model_(model,sparsity):
 count=zero=0
 with torch.no_grad():
  for mod in model.modules():
   if not isinstance(mod,nn.Linear): continue
   w=mod.weight; flat=w.abs().float().reshape(-1); k=max(1,min(flat.numel(),int(round(sparsity*flat.numel())))); thr=torch.kthvalue(flat,k).values; mask=w.abs()>thr.to(w.dtype)
   zero+=int((~mask).sum()); count+=w.numel(); w.mul_(mask)
 return count,zero
def convert_(model,conversion):
 if conversion=='int8': return {'converted_linear_scalars':quantize_model_(model,8)}
 if conversion=='int4': return {'converted_linear_scalars':quantize_model_(model,4)}
 if conversion.startswith('prune'):
  s=float(conversion.replace('prune',''))/100.; n,z=prune_model_(model,s); return {'converted_linear_scalars':n,'actual_zero_fraction':z/max(n,1)}
 raise ValueError(conversion)
@torch.no_grad()
def probe(model,ids,tail=2): return model(input_ids=ids,use_cache=False).logits[:,-tail:,:].float().cpu()
@torch.no_grad()
def eval_loss(model,batches): return float(sum(float(model(input_ids=x,labels=x,use_cache=False).loss) for x in batches)/len(batches))
def grad_only(model,q,ids):
 if q.weight.grad is not None:q.weight.grad=None
 loss=model(input_ids=ids,labels=ids,use_cache=False).loss; loss.backward(); g=q.weight.grad.detach().float().cpu().clone(); q.weight.grad=None; return float(loss.detach()),g
def step(model,q,ids,lr):
 if q.weight.grad is not None:q.weight.grad=None
 loss=model(input_ids=ids,labels=ids,use_cache=False).loss; loss.backward()
 with torch.no_grad():q.weight.add_(q.weight.grad,alpha=-lr)
 q.weight.grad=None; return float(loss.detach())

def trajectory(target,seed,conversion,steps=20,lr=5e-4):
 seed_all(seed); mid=MODEL_IDS[target]; tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
 batches=encode(tok); order=list(range(len(batches))); random.Random(seed).shuffle(order); batches=[batches[i] for i in order]; train=batches[:6]; held=batches[6:]
 checkpoints=sorted(set([0,1,5,10,steps]))
 def run(convert=False):
  model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager'); model.config.use_cache=False; model.eval(); metadata={}
  if convert:metadata=convert_(model,conversion)
  q=freeze_except_q(model); p0=probe(model,held[0]); l0=eval_loss(model,held); _,g0=grad_only(model,q,train[0]); logits={0:p0}; losses={0:l0}; train_losses=[]
  for t in range(1,steps+1):
   train_losses.append(step(model,q,train[(t-1)%len(train)],lr))
   if t in checkpoints:logits[t]=probe(model,held[0]);losses[t]=eval_loss(model,held)
  final=losses[steps]; npars=sum(p.numel() for p in model.parameters()); del model; gc.collect(); return {'logits':logits,'losses':losses,'grad':g0,'train_losses':train_losses,'initial':p0,'initial_loss':l0,'final_loss':final,'parameter_count':npars,'metadata':metadata}
 src=run(False); conv=run(True); init_pred=float((conv['initial']-src['initial']).norm()/(src['initial'].norm()+1e-12)); gdiff=float((conv['grad']-src['grad']).norm()/(src['grad'].norm()+1e-12)); gcos=float(torch.dot(conv['grad'].reshape(-1),src['grad'].reshape(-1))/(conv['grad'].norm()*src['grad'].norm()+1e-12))
 num=den=lnum=lden=0.; per={}
 for t in checkpoints[1:]:
  ds=src['logits'][t]-src['logits'][0]; dc=conv['logits'][t]-conv['logits'][0]; n=float((dc-ds).pow(2).sum()); d=float(ds.pow(2).sum()); num+=n;den+=d
  ls=src['losses'][t]-src['losses'][0]; lc=conv['losses'][t]-conv['losses'][0]; lnum+=(lc-ls)**2;lden+=ls**2;per[str(t)]={'source_eval_loss':src['losses'][t],'converted_eval_loss':conv['losses'][t],'function_update_relerr':math.sqrt(n/(d+1e-18))}
 dfunc=math.sqrt(num/(den+1e-18)); dloss=math.sqrt(lnum/(lden+1e-18)); tq=1/(1+dfunc); recovery=None
 for t in checkpoints:
  if conv['losses'][t]<=src['final_loss']*1.01:recovery=t;break
 return {'target':target,'model_id':mid,'seed':seed,'prompt_permutation':order,'train_prompt_indices':order[:6],'heldout_prompt_indices':order[6:],'conversion':conversion,'steps':steps,'lr':lr,'test_type':'full_pretrained_checkpoint_conversion_plus_causal_lm_continuation','parameter_count':src['parameter_count'],'initial_prediction_relerr':init_pred,'gradient_relerr_t0':gdiff,'gradient_cosine_t0':gcos,'trajectory_function_divergence':dfunc,'trajectory_loss_divergence':dloss,'trainability_quotient_TQ':tq,'recovery_checkpoint':recovery,'source_initial_eval_loss':src['initial_loss'],'converted_initial_eval_loss':conv['initial_loss'],'source_final_eval_loss':src['final_loss'],'converted_final_eval_loss':conv['final_loss'],'source_train_losses':src['train_losses'],'converted_train_losses':conv['train_losses'],'checkpoints':per,'conversion_metadata':conv['metadata'],'actual_pretrained_weights_loaded':True,'actual_tokenized_text_used':True,'actual_language_model_loss_used':True,'primary_metric_definition':'TQ=1/(1+D_func), D_func=RMS discrepancy between converted and source function-space update trajectories normalized by source update energy after subtracting each artifact initial output.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target',choices=list(MODEL_IDS),required=True);ap.add_argument('--conversion',choices=['int8','int4','prune20','prune40'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);r=trajectory(a.target,a.seed,a.conversion,a.steps);p=o/f'{a.target}_{a.conversion}_seed{a.seed}.json';p.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2));print('RESULT_FILE='+str(p))
if __name__=='__main__':main()
