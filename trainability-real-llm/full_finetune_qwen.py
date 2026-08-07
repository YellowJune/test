from __future__ import annotations
import argparse,gc,json,math,os,random
from pathlib import Path
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
os.environ.setdefault('TOKENIZERS_PARALLELISM','false');torch.set_num_threads(min(4,os.cpu_count() or 2))
MID='Qwen/Qwen2.5-0.5B-Instruct'
def qdq_(w,bits=4,group=64):
 qmax=2**(bits-1)-1
 with torch.no_grad():
  m=w.view(w.shape[0],-1)
  for j in range(0,m.shape[1],group):
   x=m[:,j:j+group];s=x.abs().amax(1,keepdim=True).clamp_min(1e-12)/qmax;x.copy_((x/s).round().clamp(-qmax,qmax)*s)
def convert_(m,c):
 with torch.no_grad():
  for mod in m.modules():
   if not isinstance(mod,nn.Linear):continue
   if c=='int4':qdq_(mod.weight,4)
   else:
    f=mod.weight.abs().float().reshape(-1);k=max(1,int(.4*f.numel()));thr=torch.kthvalue(f,k).values;mod.weight.mul_(mod.weight.abs()>thr.to(mod.weight.dtype))
def data(tok,seed):
 ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train');texts=[x['text'].strip() for x in ds if len(x['text'].strip())>80];r=random.Random(seed);r.shuffle(texts);seq=[]
 for t in texts:
  ids=tok(t,return_tensors='pt',truncation=True,max_length=32,add_special_tokens=True)['input_ids']
  if ids.shape[1]>=24:seq.append(ids)
  if len(seq)>=20:break
 return seq[:16],seq[16:20]
@torch.no_grad()
def probe(m,x):return m(input_ids=x,use_cache=False).logits[:,-2:,:].float().cpu()
@torch.no_grad()
def ev(m,xs):return float(sum(float(m(input_ids=x,labels=x,use_cache=False).loss) for x in xs)/len(xs))
def step(m,x,lr):
 m.zero_grad(set_to_none=True);loss=m(input_ids=x,labels=x,use_cache=False).loss;loss.backward()
 with torch.no_grad():
  for p in m.parameters():
   if p.grad is not None:p.add_(p.grad,alpha=-lr)
 m.zero_grad(set_to_none=True);return float(loss.detach())
def one(seed,conv,steps=50,lr=1e-5):
 random.seed(seed);torch.manual_seed(seed);tok=AutoTokenizer.from_pretrained(MID,use_fast=True);train,held=data(tok,seed);cps=sorted(set([0,1,10,25,steps]))
 def run(do_conv):
  m=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager');m.config.use_cache=False;m.train();
  if do_conv:convert_(m,conv)
  p0=probe(m,held[0]);l0=ev(m,held);logs={0:p0};losses={0:l0};tr=[]
  for t in range(1,steps+1):
   tr.append(step(m,train[(t-1)%len(train)],lr))
   if t in cps:logs[t]=probe(m,held[0]);losses[t]=ev(m,held)
  n=sum(p.numel() for p in m.parameters());del m;gc.collect();return p0,l0,logs,losses,tr,n
 s0,sl0,sl,sloss,strain,n=run(False);c0,cl0,cl,closs,ctrain,_=run(True);num=den=lnum=lden=0.;per={}
 for t in cps[1:]:
  a=sl[t]-sl[0];b=cl[t]-cl[0];nn=float((b-a).pow(2).sum());dd=float(a.pow(2).sum());num+=nn;den+=dd;x=sloss[t]-sloss[0];y=closs[t]-closs[0];lnum+=(y-x)**2;lden+=x*x;per[str(t)]={'source_eval_loss':sloss[t],'converted_eval_loss':closs[t],'function_update_relerr':math.sqrt(nn/(dd+1e-18))}
 D=math.sqrt(num/(den+1e-18));return {'model_id':MID,'seed':seed,'conversion':conv,'test_type':'full_pretrained_full_parameter_wikitext2_causal_lm_continuation','dataset':'Salesforce/wikitext:wikitext-2-raw-v1','parameter_count':n,'steps':steps,'lr':lr,'initial_prediction_relerr':float((c0-s0).norm()/(s0.norm()+1e-12)),'trajectory_function_divergence':D,'trajectory_loss_divergence':math.sqrt(lnum/(lden+1e-18)),'trainability_quotient_TQ':1/(1+D),'source_initial_eval_loss':sl0,'converted_initial_eval_loss':cl0,'source_final_eval_loss':sloss[steps],'converted_final_eval_loss':closs[steps],'source_train_losses':strain,'converted_train_losses':ctrain,'checkpoints':per,'all_model_parameters_updated':True,'actual_pretrained_weights_loaded':True,'actual_causal_lm_loss_used':True,'actual_public_dataset_used':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--conversion',choices=['int4','prune40'],required=True);ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);r=one(a.seed,a.conversion);p=o/f'qwen_full_{a.conversion}_seed{a.seed}.json';p.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2));print('RESULT_FILE='+str(p))
if __name__=='__main__':main()
