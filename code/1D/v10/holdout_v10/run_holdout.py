import csv, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import capexp_v10 as base

OUT=Path(__file__).resolve().parent
# Frozen v10 winners (one configuration per target, depth, and activation).
# PreCap values come from v10 final winners; SiLU/GELU are selected from v10
# by per-target/depth minimum median over original seeds 0-9.
src=[]
for f in sorted(ROOT.glob('v10_results_shard*.csv')):
    src.extend(csv.DictReader(open(f, newline='')))
fields=['target','act','c','a','gamma','soft_s','depth','arch','init_k','lr','seed','timestamp','test_loss','bad','fail_step','duration_sec']
def num(r,k): return float(r[k])
def frozen(act,target,depth):
    candidates=[r for r in src if r['act']==act and r['target']==target and int(r['depth'])==depth and r['arch']=='res_inv']
    groups={}
    for r in candidates:
        key=tuple(r[k] for k in ['c','a','gamma','soft_s','init_k','lr'])
        groups.setdefault(key,[]).append(num(r,'test_loss'))
    best=min(groups, key=lambda k: sorted(groups[k])[len(groups[k])//2])
    c,a,g,s,k,lr=map(float,best)
    return dict(target=target,act=act,c=c,a=a,gamma=g,soft_s=s,depth=depth,arch='res_inv',init_k=k,lr=lr)
configs=[]
for target in ['sin3_quad','sin_mix','abs_wave']:
 for depth in [32,64]:
  for act in ['precap','silu','gelu']:
   configs.append(frozen(act,target,depth))
allruns=[]
for cfg in configs:
 for seed in range(10,20):
  z=dict(cfg); z['seed']=seed; allruns.append(z)
shard=int(sys.argv[1]); nshards=int(sys.argv[2])
runs=allruns[shard::nshards]
out=OUT/f'holdout_results_shard{shard}.csv'
existing=set()
if out.exists():
 for r in csv.DictReader(open(out, newline='')):
  existing.add(tuple(r[k] for k in ['target','act','c','a','gamma','soft_s','depth','arch','init_k','lr','seed']))
new=not out.exists()
with open(out,'a',newline='') as fh:
 w=csv.DictWriter(fh,fieldnames=fields)
 if new: w.writeheader(); fh.flush()
 for i,cfg in enumerate(runs,1):
  key=tuple(str(cfg[k]) for k in ['target','act','c','a','gamma','soft_s','depth','arch','init_k','lr','seed'])
  if key in existing: continue
  row=base.run(cfg)
  w.writerow({k:row.get(k,'') for k in fields}); fh.flush()
  print(f'[{i}/{len(runs)}] {cfg["target"]} {cfg["act"]} d={cfg["depth"]} seed={cfg["seed"]} test={row["test_loss"]:.5g} bad={row["bad"]}', flush=True)
