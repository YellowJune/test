from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

def trajectory_divergence(source_outputs, converted_outputs, eps=1e-18):
    s=np.asarray(source_outputs,dtype=np.float64); c=np.asarray(converted_outputs,dtype=np.float64)
    if s.shape != c.shape or s.shape[0] < 2:
        raise ValueError('source and converted trajectories must have identical shape with T>=2')
    us=s[1:]-s[0]; uc=c[1:]-c[0]
    return float(math.sqrt(np.sum((uc-us)**2)/(np.sum(us**2)+eps)))

def tq_traj(source_outputs, converted_outputs, eps=1e-18):
    d=trajectory_divergence(source_outputs,converted_outputs,eps)
    return {'D_func':d,'TQ_traj':1.0/(1.0+d)}

def initial_prediction_relerr(source0, converted0, eps=1e-18):
    s=np.asarray(source0,dtype=np.float64); c=np.asarray(converted0,dtype=np.float64)
    return float(np.linalg.norm(c-s)/(np.linalg.norm(s)+eps))

def score_file(path):
    o=json.loads(Path(path).read_text()); r=tq_traj(o['source_outputs'],o['converted_outputs'])
    r['initial_prediction_relerr']=initial_prediction_relerr(o['source_outputs'][0],o['converted_outputs'][0])
    if 'source_losses' in o and 'converted_losses' in o:
        sl=np.asarray(o['source_losses'],dtype=np.float64); cl=np.asarray(o['converted_losses'],dtype=np.float64)
        if sl.shape==cl.shape and sl.size: r['final_loss_gap']=float(cl[-1]-sl[-1])
    return r

def main():
    ap=argparse.ArgumentParser(description='Reference scorer for the Trainability Quotient benchmark'); ap.add_argument('trajectory_json'); a=ap.parse_args(); print(json.dumps(score_file(a.trajectory_json),indent=2))
if __name__=='__main__': main()
