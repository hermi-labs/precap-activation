import argparse, csv, os, sys, time
from datetime import datetime
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import capexp_v11_proto as P

SEEDS=list(range(100,110))
PAIRS=[(0.5,0.001),(1.0,0.001),(1.0,0.003)]
BASE=dict(c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.25)
PLAIN=dict(c=0.0,a=1.0,gamma=0.0,soft_s=1.0,knee=0.75,mix_beta=0.25)
ACTS={'precap':BASE,'hybrid_gate':dict(BASE,mix_beta=0.30),'silu':PLAIN,'gelu':PLAIN}
TARGETS=['sin3_quad','sin_mix','abs_wave']
GROUPS=[(d,a,PAIRS) for d in (4,8,16) for a in ('plain','res_inv')]+[(d,'res_inv',[(0.5,0.001)]) for d in (32,64)]
FIELDS=P.KEYS+['timestamp','steps','bad','test_loss','grad_final','hmax_final','cap_final','hmax_peak']

def configs():
    out=[]
    for seed in SEEDS:
        for t in TARGETS:
            for depth,arch,pairs in GROUPS:
                for act,pr in ACTS.items():
                    for ik,lr in pairs:
                        out.append(dict(target=t,act=act,depth=depth,arch=arch,init_k=ik,lr=lr,seed=seed,**pr))
    return out

def key(cfg): return tuple(str(cfg[k]) for k in P.KEYS)

def done_keys(path):
    if not os.path.exists(path): return set()
    with open(path,newline='') as f: return {tuple(r[k] for k in P.KEYS) for r in csv.DictReader(f)}

def worker(shard,n):
    out=os.path.join(HERE,'results',f'confirm_w{shard}.csv')
    mine=[c for i,c in enumerate(configs()) if i%n==shard]
    done=done_keys(out); total=len(mine); j=len(done)
    plog=os.path.join(HERE,'logs','confirm_progress.log')
    for cfg in mine:
        if key(cfg) in done: continue
        t0=time.time()
        curve,final,bad,fail,row=P.run(cfg)
        last=curve[-1] if curve else {}
        row.update(grad_final=last.get('grad_norm',''),hmax_final=last.get('act_abs_max',''),cap_final=last.get('cap_frac',''),hmax_peak=max([c['act_abs_max'] for c in curve],default=''))
        P.append_csv(out,row,FIELDS); j+=1
        with open(plog,'a') as f: f.write(f"{datetime.now():%F %T} w{shard} {j}/{total} {cfg['target']} {cfg['act']} d={cfg['depth']} {cfg['arch']} seed={cfg['seed']} test={final:.4g} bad={bad} {time.time()-t0:.0f}s\n")

def merge():
    rows=[]
    for i in range(7):
        p=os.path.join(HERE,'results',f'confirm_w{i}.csv')
        if os.path.exists(p):
            with open(p,newline='') as f: rows+=list(csv.DictReader(f))
    exp=len(configs()); keys=[tuple(r[k] for k in P.KEYS) for r in rows]
    outp=os.path.join(os.path.dirname(HERE),'v11_confirm_results.csv')
    with open(outp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    msg=f"rows={len(rows)} expected={exp} unique={len(set(keys))} bad={sum(r['bad']=='True' for r in rows)}"
    print(msg); open(os.path.join(HERE,'logs','confirm_integrity.txt'),'w').write(msg+'\n')
    return len(rows)==exp and len(set(keys))==exp

def selftest():
    cs=[c for c in configs() if c['depth']==4 and c['arch']=='plain' and c['target']=='sin3_quad' and c['seed']==100 and (c['init_k'],c['lr'])==(0.5,0.001)]
    for c in cs:
        t=time.time(); _,final,bad,_,row=P.run(c); print('SELFTEST',c['act'],f'test={final:.4g}',f'bad={bad}',f'{time.time()-t:.1f}s',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['worker','merge','selftest','count']); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--n',type=int,default=7)
    a=ap.parse_args()
    if a.cmd=='worker': worker(a.shard,a.n)
    elif a.cmd=='merge': sys.exit(0 if merge() else 1)
    elif a.cmd=='selftest': selftest()
    else:
        cs=configs(); print('total',len(cs),'unique',len({key(c) for c in cs}))
        from collections import Counter
        print(Counter(c['act'] for c in cs)); print(Counter(c['depth'] for c in cs)); print([sum(1 for i in range(len(cs)) if i%7==s) for s in range(7)])
