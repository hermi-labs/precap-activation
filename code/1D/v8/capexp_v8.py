import argparse, csv, glob, math, os, sys, time
from datetime import datetime
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capexp_v6_stability as v6

BASE = {'silu': nn.SiLU, 'gelu': nn.GELU, 'relu': nn.ReLU}
KEYS = ['target','act','c','in_scale','gamma','soft_s','depth','arch','init_k','lr','clip','warmup','seed']
T2 = ['sin3_quad','sin_mix']
T3 = ['sin3_quad','sin_mix','abs_wave']


def mk_act(cfg):
    k = cfg['act']
    if k in BASE:
        return BASE[k]()
    if k == 'capexp':
        return v6.CapExp(c=cfg['c'], in_scale=cfg['in_scale'])
    if k == 'precap':
        return v6.PreCap(c=cfg['c'], gamma=cfg['gamma'], p=2, soft_s=cfg['soft_s'], in_scale=cfg['in_scale'])
    raise ValueError(k)


def build(cfg, width):
    layers = [nn.Linear(1, width), mk_act(cfg)]
    for _ in range(cfg['depth'] - 1):
        layers.append(v6.Block(width, mk_act(cfg), arch=cfg['arch']))
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=cfg['init_k'] / math.sqrt(m.in_features))
            nn.init.zeros_(m.bias)
    return model


def train_one(cfg, steps, width):
    torch.manual_seed(cfg['seed'])
    f = v6.TARGETS[cfg['target']]
    xtr = torch.linspace(-3, 3, 512).unsqueeze(1); ytr = f(xtr)
    xte = torch.linspace(-3.5, 3.5, 400).unsqueeze(1); yte = f(xte)
    model = build(cfg, width)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    maxg, bad, fail_step = 0.0, False, -1
    clipv = cfg['clip'] if cfg['clip'] > 0 else float('inf')
    t0 = time.time()
    for step in range(1, steps + 1):
        if cfg['warmup'] > 0:
            for g in opt.param_groups:
                g['lr'] = cfg['lr'] * min(1.0, step / cfg['warmup'])
        opt.zero_grad()
        loss = ((model(xtr) - ytr) ** 2).mean()
        if not torch.isfinite(loss):
            bad, fail_step = True, step
            break
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clipv))
        if not math.isfinite(gn):
            bad, fail_step = True, step
            break
        maxg = max(maxg, gn)
        opt.step()
    with torch.no_grad():
        fin = float('nan') if bad else float(((model(xtr) - ytr) ** 2).mean().item())
        te = float('nan') if bad else float(((model(xte) - yte) ** 2).mean().item())
    r = {'timestamp': datetime.now().isoformat(timespec='seconds')}
    r.update({k: cfg[k] for k in KEYS})
    r.update({'steps': steps, 'loss_final': fin, 'test_loss': te, 'max_grad': maxg, 'nan_or_inf': bad, 'fail_step': fail_step, 'duration_sec': time.time() - t0})
    return r


def base_cfg(**kw):
    d = dict(gamma=0.0, soft_s=1.0, c=0.0, in_scale=1.0, init_k=0.5, lr=1e-3, clip=0.0, warmup=0)
    d.update(kw)
    return d


def build_jobs(stage, seeds):
    J = []
    for seed in range(seeds):
        if stage == 'E':
            for t in T3:
                for d in [16, 32]:
                    for arch in ['plain', 'res_inv']:
                        for c in [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]:
                            for a in [1.0, 1.1]:
                                J.append(base_cfg(target=t, act='capexp', depth=d, arch=arch, c=c, in_scale=a, seed=seed))
        elif stage == 'F':
            for t in T2:
                for d in [32, 64]:
                    for c in [1.0, 1.5]:
                        for a in [1.0, 1.5]:
                            for k in [0.25, 0.5]:
                                for clip in [0.0, 1.0]:
                                    for wu in [0, 200]:
                                        J.append(base_cfg(target=t, act='capexp', depth=d, arch='plain', c=c, in_scale=a, init_k=k, clip=clip, warmup=wu, seed=seed))
        elif stage == 'G':
            for t in T3:
                for d in [16, 32, 64]:
                    for arch in ['plain', 'res_inv']:
                        for lr in [3e-4, 1e-3, 3e-3]:
                            for k in [0.5, 1.0]:
                                b = dict(target=t, depth=d, arch=arch, lr=lr, init_k=k, seed=seed)
                                for c in [0.8, 1.0]:
                                    J.append(base_cfg(act='capexp', c=c, **b))
                                J.append(base_cfg(act='precap', c=0.5, in_scale=1.5, gamma=0.0, soft_s=1.5, **b))
                                J.append(base_cfg(act='precap', c=1.0, in_scale=1.0, gamma=0.01, soft_s=1.5, **b))
                                for a in BASE:
                                    J.append(base_cfg(act=a, **b))
    return J


def jkey(cfg):
    return tuple(str(cfg[k]) for k in KEYS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['E', 'F', 'G'], default='E')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(1)
    jobs = build_jobs(a.stage, a.seeds)
    if a.smoke:
        seen = set()
        for cfg in jobs:
            sig = (cfg['act'], cfg['clip'], cfg['warmup'])
            if sig in seen: continue
            seen.add(sig)
            r = train_one(cfg, a.steps, a.width)
            print('smoke', a.stage, sig, round(r['test_loss'], 4), r['nan_or_inf'])
        print('jobs', a.stage, 'seeds', a.seeds, len(jobs))
        return
    done = set()
    for fn in glob.glob('experiments_v8_%s_shard*.csv' % a.stage):
        with open(fn, newline='') as fh:
            for r in csv.DictReader(fh):
                done.add(tuple(r[k] for k in KEYS))
    mine = [j for j in jobs[a.shard::a.nshards] if jkey(j) not in done]
    out = 'experiments_v8_%s_shard%d.csv' % (a.stage, a.shard)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    print('shard', a.shard, 'stage', a.stage, 'todo', len(mine), flush=True)
    with open(out, 'a', newline='') as fh:
        w = None
        for i, cfg in enumerate(mine, 1):
            r = train_one(cfg, a.steps, a.width)
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if new: w.writeheader()
            w.writerow(r); fh.flush()
            print('[%d/%d] %s %s d=%s %s c=%s a=%s k=%s lr=%s clip=%s wu=%s seed=%s test=%.4f bad=%s' % (
                i, len(mine), cfg['target'], cfg['act'], cfg['depth'], cfg['arch'], cfg['c'], cfg['in_scale'],
                cfg['init_k'], cfg['lr'], cfg['clip'], cfg['warmup'], cfg['seed'], r['test_loss'], r['nan_or_inf']), flush=True)


if __name__ == '__main__':
    main()
