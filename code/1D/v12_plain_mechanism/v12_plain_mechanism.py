import argparse,csv,os,sys,time
from datetime import datetime
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(os.path.dirname(HERE),'v11'))
import capexp_v11_proto as P
SEEDS=list(range(200,210)); TARGETS=['sin3_quad','sin_mix','abs_wave']; DEPTHS=[4,8,16]
REGIMES=[('base',(0.5,0.001)),('high_lr',(0.5,0.003)),('large_init',(1.0,0.001))]
BASE=dict(c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.25)
PLAIN=dict(c=0.0,a=1.0,gamma=0.0,soft_s=1.0,knee=0.75,mix_beta=0.25)
ACTS={'precap':BASE,'silu':PLAIN,'gelu':PLAIN}
FIELDS=P.KEYS+['timestamp','steps','bad','test_loss','grad_final','hmax_final','cap_final','hmax_peak']
def configs():
 o=[]
 for seed in SEEDS:
  for target in TARGETS:
   for depth in DEPTHS:
    for arch in ['plain','res_inv']:
     for reg,(ik,lr) in REGIMES:
      for act,pr in ACTS.items(): o.append(dict(target=target,act=act,depth=depth,arch=arch,init_k=ik,lr=lr,seed=seed,regime=reg,**pr))
 return o
def key(c): return tuple(str(c[k]) for k in P.KEYS)
def done(p):
 if not os.path.exists(p): return set()
 with open(p,newline='') as f:return {tuple(r[k] for k in P.KEYS) for r in csv.DictReader(f)}
def worker(s,n):
 out=f'{HERE}/results/v12_w{s}.csv'; mine=[c for i,c in enumerate(configs()) if i%n==s]; have=done(out); j=len(have)
 for c in mine:
  if key(c) in have:continue
  t=time.time(); curve,final,bad,fail,row=P.run(c); last=curve[-1] if curve else {}
  row.update(grad_final=last.get('grad_norm',''),hmax_final=last.get('act_abs_max',''),cap_final=last.get('cap_frac',''),hmax_peak=max([x['act_abs_max'] for x in curve],default=''))
  P.append_csv(out,row,FIELDS);j+=1
  with open(f'{HERE}/logs/progress.log','a') as f:f.write(f"{datetime.now():%F %T} w{s} {j}/{len(mine)} {c['target']} {c['act']} d={c['depth']} {c['arch']} {c['regime']} seed={c['seed']} test={final:.4g} bad={bad} {time.time()-t:.0f}s\n")
def merge():
 rows=[]
 for i in range(7):
  p=f'{HERE}/results/v12_w{i}.csv'
  if os.path.exists(p):
   with open(p,newline='') as f:rows+=list(csv.DictReader(f))
 exp=len(configs()); ks=[tuple(r[k] for k in P.KEYS) for r in rows]
 with open(f'{HERE}/v12_results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 msg=f'rows={len(rows)} expected={exp} unique={len(set(ks))} bad={sum(r["bad"]=="True" for r in rows)}';print(msg);open(f'{HERE}/logs/integrity.txt','w').write(msg+'\n');return len(rows)==exp and len(set(ks))==exp
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['worker','merge','count']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--n',type=int,default=7);a=ap.parse_args()
 if a.cmd=='worker':worker(a.shard,a.n)
 elif a.cmd=='merge':sys.exit(0 if merge() else 1)
 else:
  from collections import Counter
  cs=configs();print('total',len(cs),'unique',len({key(c) for c in cs}));print(Counter(c['depth'] for c in cs));print([sum(i%7==s for i in range(len(cs))) for s in range(7)])
