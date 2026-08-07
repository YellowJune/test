from __future__ import annotations
import argparse, copy, gc, json, os, random, math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
torch.set_num_threads(min(4,os.cpu_count() or 2))
MID='Qwen/Qwen2.5-0.5B-Instruct'
TEXTS=[
 'Knowledge distillation transfers behavior from a teacher to a smaller student.',
 'A language model predicts the next token from its preceding context.',
 'Model compression can preserve outputs on a calibration distribution.',
 'Optimization changes a model after new training evidence arrives.',
 'Future fine tuning can expose differences hidden at conversion time.',
 'Trainability is a property of how a model responds to optimization.',
 'The trainability quotient compares update trajectories in function space.',
 'A student can imitate a teacher while retaining different internal geometry.'
]

def seed_all(s):random.seed(s);torch.manual_seed(s)
def enc(tok,text,max_len=16):return tok(text,return_tensors='pt',truncation=True,max_length=max_len,add_special_tokens=True)['input_ids']
def qproj(m):return m.model.layers[0].self_attn.q_proj
def freeze_q(m):
 for p in m.parameters():p.requires_grad_(False)
 q=qproj(m);q.weight.requires_grad_(True);return q
@torch.no_grad()
def tail_logits(m,ids,tail=2):return m(input_ids=ids,use_cache=False).logits[:,-tail:,:].float().cpu()
@torch.no_grad()
def eval_loss(m,batches):return float(sum(float(m(input_ids=x,labels=x,use_cache=False).loss) for x in batches)/len(batches))
def ce_step(m,q,ids,lr):
 if q.weight.grad is not None:q.weight.grad=None
 loss=m(input_ids=ids,labels=ids,use_cache=False).loss;loss.backward()
 with torch.no_grad():q.weight.add_(q.weight.grad,alpha=-lr)
 q.weight.grad=None;return float(loss.detach())
def build_student(teacher,nlayers):
 cfg=copy.deepcopy(teacher.config);cfg.num_hidden_layers=nlayers;student=AutoModelForCausalLM.from_config(cfg).float();student.eval();student.config.use_cache=False
 student.model.embed_tokens.load_state_dict(teacher.model.embed_tokens.state_dict());student.model.norm.load_state_dict(teacher.model.norm.state_dict())
 if hasattr(student,'lm_head') and hasattr(teacher,'lm_head'):student.lm_head.load_state_dict(teacher.lm_head.state_dict())
 Lt=len(teacher.model.layers);ids=torch.linspace(0,Lt-1,nlayers).round().long().tolist()
 for i,j in enumerate(ids):student.model.layers[i].load_state_dict(teacher.model.layers[j].state_dict(),strict=True)
 return student,ids
def distill_step(student,q,ids,target_logits,lr,T=2.):
 if q.weight.grad is not None:q.weight.grad=None
 z=student(input_ids=ids,use_cache=False).logits[:,-target_logits.shape[1]:,:].float();target=target_logits.to(z.device);loss=F.kl_div(F.log_softmax(z/T,-1),F.softmax(target/T,-1),reduction='batchmean')*(T*T);loss.backward()
 with torch.no_grad():q.weight.add_(q.weight.grad,alpha=-lr)
 q.weight.grad=None;return float(loss.detach())

def run(seed,nlayers,distill_steps=32,cont_steps=10,lr=7e-4):
 seed_all(seed);tok=AutoTokenizer.from_pretrained(MID,use_fast=True);order=list(range(len(TEXTS)));random.Random(seed).shuffle(order);cal_idx=order[:4];fut_idx=order[4:];cal=[enc(tok,TEXTS[i]) for i in cal_idx];fut=[enc(tok,TEXTS[i]) for i in fut_idx]
 teacher=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager');teacher.eval();teacher.config.use_cache=False
 for p in teacher.parameters():p.requires_grad_(False)
 with torch.no_grad():targets=[tail_logits(teacher,x) for x in cal]
 student,selected=build_student(teacher,nlayers);teacher_initial=tail_logits(teacher,fut[-1]);teacher_initial_loss=eval_loss(teacher,fut);del teacher;gc.collect()
 q=freeze_q(student);distill=[]
 for t in range(distill_steps):distill.append(distill_step(student,q,cal[t%len(cal)],targets[t%len(cal)],lr))
 student_initial=tail_logits(student,fut[-1]);student_initial_loss=eval_loss(student,fut);cps=[0,1,5,cont_steps];slog={0:student_initial};sloss={0:student_initial_loss};student_train=[]
 for t in range(1,cont_steps+1):
  student_train.append(ce_step(student,q,fut[(t-1)%len(fut)],lr))
  if t in cps:slog[t]=tail_logits(student,fut[-1]);sloss[t]=eval_loss(student,fut)
 del student;gc.collect()
 teacher=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager');teacher.eval();teacher.config.use_cache=False;tq=freeze_q(teacher);tlog={0:teacher_initial};tloss={0:teacher_initial_loss};teacher_train=[]
 for t in range(1,cont_steps+1):
  teacher_train.append(ce_step(teacher,tq,fut[(t-1)%len(fut)],lr))
  if t in cps:tlog[t]=tail_logits(teacher,fut[-1]);tloss[t]=eval_loss(teacher,fut)
 num=den=lnum=lden=0.;per={}
 for t in cps[1:]:
  dt=tlog[t]-tlog[0];ds=slog[t]-slog[0];n=float((ds-dt).pow(2).sum());d=float(dt.pow(2).sum());num+=n;den+=d;a=tloss[t]-tloss[0];b=sloss[t]-sloss[0];lnum+=(b-a)**2;lden+=a*a;per[str(t)]={'teacher_eval_loss':tloss[t],'student_eval_loss':sloss[t],'function_update_relerr':math.sqrt(n/(d+1e-18))}
 dfunc=math.sqrt(num/(den+1e-18));dloss=math.sqrt(lnum/(lden+1e-18));score=1/(1+dfunc)
 return {'seed':seed,'prompt_permutation':order,'calibration_prompt_indices':cal_idx,'future_prompt_indices':fut_idx,'teacher_model':MID,'conversion':'layer_drop_plus_logit_distillation','teacher_layers':int(teacher.config.num_hidden_layers),'student_layers':nlayers,'compression_fraction_layers':1-nlayers/int(teacher.config.num_hidden_layers),'selected_teacher_layers':selected,'distillation_steps':distill_steps,'continuation_steps':cont_steps,'lr':lr,'distillation_losses':distill,'initial_prediction_relerr':float((student_initial-teacher_initial).norm()/(teacher_initial.norm()+1e-12)),'teacher_initial_eval_loss':teacher_initial_loss,'student_initial_eval_loss':student_initial_loss,'teacher_final_eval_loss':tloss[cont_steps],'student_final_eval_loss':sloss[cont_steps],'trajectory_function_divergence':dfunc,'trajectory_loss_divergence':dloss,'trainability_quotient_TQ':score,'teacher_train_losses':teacher_train,'student_train_losses':student_train,'checkpoints':per,'actual_pretrained_teacher_loaded':True,'actual_teacher_logits_used_for_distillation':True,'actual_causal_lm_continuation_used':True,'primary_metric_definition':'TQ=1/(1+D_func), comparing teacher and student function-space update trajectories after subtracting each artifact initial output.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--student-layers',type=int,choices=[12,18,22],required=True);ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);r=run(a.seed,a.student_layers);p=o/f'qwen_distill_L{a.student_layers}_seed{a.seed}.json';p.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2));print('RESULT_FILE='+str(p))
if __name__=='__main__':main()
