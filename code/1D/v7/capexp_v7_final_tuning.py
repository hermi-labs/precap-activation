import argparse, csv, glob, os, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capexp_v6_stability as v6

BASE = {'silu': nn.SiLU, 'gelu': nn.GELU, 'relu': nn.ReLU}
_orig = v6.make_act
def make_act(kind, c, in_scale, gamma, p, soft_s):
    if kind in BASE:
        return BASE[kind]()
    return _orig(kind, c, in_scale, gamma, p, soft_s)
v6.make_act = make_act

KEYS = ['target','act','c','depth','init','arch','lr','seed','in_scale','gamma','p','soft_s']
T2 = ['sin3_quad','sin_mix']
T3 = ['sin3_quad','sin_mix','abs_wave']
STAGES = {
 'A': dict(depths=[16,32], targets=T2, lrs=[1e-3], inits=['default','small'], archs=['plain','res_inv'],
           cs_cap=[0.5,0.7,1.0,1.5], ins=[1.0,1.25,1.5], cs_pre=[0.5,0.7,1.0], gammas=[0.0,0.01], softs=[1.0,1.5]),
 'B': dict(depths=[64], targets=T2, lrs=[1e-3], inits=['default','small'], archs=['plain','res_inv'],
           cs_cap=[0.5,1.0], ins=[1.0,1.5], cs_pre=[0.5,1.0], gammas=[0.0,0.01], softs=[1.0]),
 'C': dict(depths=[16,32], targets=T3, lrs=[3e-4,1e-3,3e-3], inits=['default','small'], archs=['plain'],
           cs_cap=[0.5,1.0], ins=[1.0,1.5], cs_pre=[0.5,1.0], gammas=[0.0,0.01], softs=[1.0]),
 'D': dict(depths=[16,32], targets=T2, lrs=[1e-3], inits=['small'], archs=['plain'],
           cs_cap=[0.5,1.0], ins=[1.0,1.5], cs_pre=[0.5,1.0], gammas=[0.0,0.01], softs=[1.0]),
}

def build_jobs(stage, seeds):
    S = STAGES[stage]
    jobs = []
    for seed in range(seeds):
        for target in S['targets']:
            for depth in S['depths']:
                for lr in S['lrs']:
                    for init in S['inits']:
                        for arch in S['archs']:
                            b = dict(target=target, depth=depth, init=init, arch=arch, lr=lr, seed=seed)
                            for c in S['cs_cap']:
                                for s in S['ins']:
                                    jobs.append(dict(b, act='capexp', c=c, in_scale=s, gamma=0.0, p=1, soft_s=1.0))
                            for c in S['cs_pre']:
                                for s in S['ins']:
                                    for g in S['gammas']:
                                        for ss in S['softs']:
                                            jobs.append(dict(b, act='precap', c=c, in_scale=s, gamma=g, p=2, soft_s=ss))
                            for k in BASE:
                                jobs.append(dict(b, act=k, c=0.0, in_scale=1.0, gamma=0.0, p=1, soft_s=1.0))
    return jobs

def jkey(cfg):
    return tuple(str(cfg[k]) for k in KEYS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=list(STAGES), default='A')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out-prefix', default='experiments_v7')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    jobs = build_jobs(a.stage, a.seeds)
    if a.smoke:
        seen = set()
        for cfg in jobs:
            if cfg['act'] in seen:
                continue
            seen.add(cfg['act'])
            r = v6.train_one(cfg, a.steps, a.width, a.device)
            print('smoke', a.stage, cfg['act'], round(r['test_loss'], 4), r['nan_or_inf'])
        print('jobs stage', a.stage, 'seeds', a.seeds, len(jobs))
        return
    done = set()
    for f in glob.glob('%s_%s_shard*.csv' % (a.out_prefix, a.stage)):
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                done.add(tuple(r[k] for k in KEYS))
    mine = [j for j in jobs[a.shard::a.nshards] if jkey(j) not in done]
    out = '%s_%s_shard%d.csv' % (a.out_prefix, a.stage, a.shard)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    print('shard', a.shard, 'stage', a.stage, 'todo', len(mine), flush=True)
    with open(out, 'a', newline='') as fh:
        w = None
        for i, cfg in enumerate(mine, 1):
            r = v6.train_one(cfg, a.steps, a.width, a.device)
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if new:
                    w.writeheader()
            w.writerow(r)
            fh.flush()
            print('[%d/%d] %s d=%s %s lr=%s c=%s in=%s seed=%s test=%.4f bad=%s' % (
                i, len(mine), cfg['act'], cfg['depth'], cfg['arch'], cfg['lr'], cfg['c'],
                cfg['in_scale'], cfg['seed'], r['test_loss'], r['nan_or_inf']), flush=True)

if __name__ == '__main__':
    main()
