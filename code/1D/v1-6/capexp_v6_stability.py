import argparse, csv, math, time
from datetime import datetime
import torch
import torch.nn as nn


def capexp(x, c=0.5):
    return torch.where(x <= c, torch.exp(torch.clamp(x, max=c)) - 1, math.exp(c) - 1 + math.exp(c) * (x - c))


class CapExp(nn.Module):
    def __init__(self, c=0.5, in_scale=1.0):
        super().__init__()
        self.c = c
        self.in_scale = in_scale
    def forward(self, x):
        return capexp(self.in_scale * x, self.c)


class PreCap(nn.Module):
    def __init__(self, c=0.5, gamma=0.03, p=3, soft_s=1.0, in_scale=1.0):
        super().__init__()
        self.c = c
        self.gamma = gamma
        self.p = p
        self.soft_s = soft_s
        self.in_scale = in_scale
    def forward(self, x):
        z = self.in_scale * x
        u = z + self.gamma * (z ** self.p)
        u = self.soft_s * torch.tanh(u / self.soft_s)
        return capexp(u, self.c)


class Block(nn.Module):
    def __init__(self, width, act, arch='plain'):
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.act = act
        self.arch = arch
    def forward(self, x):
        y = self.act(self.lin(x))
        if self.arch == 'plain':
            return y
        if self.arch == 'res_inv':
            return x + y / math.sqrt(x.shape[-1])
        if self.arch == 'res_zero':
            return x + 0.1 * y
        raise ValueError(self.arch)


TARGETS = {
    'sin3_quad': lambda x: torch.sin(3 * x) + 0.3 * x ** 2,
    'sin_mix': lambda x: torch.sin(2 * x) + 0.5 * torch.sin(5 * x),
    'abs_wave': lambda x: torch.abs(x) * torch.sin(2 * x),
}


def make_act(kind, c, in_scale, gamma, p, soft_s):
    if kind == 'capexp':
        return CapExp(c=c, in_scale=in_scale)
    if kind == 'precap':
        return PreCap(c=c, gamma=gamma, p=p, soft_s=soft_s, in_scale=in_scale)
    raise ValueError(kind)


def build_model(kind, c, depth, width, init, arch, in_scale, gamma, p, soft_s, dev):
    act0 = make_act(kind, c, in_scale, gamma, p, soft_s)
    layers = [nn.Linear(1, width), act0]
    for _ in range(depth - 1):
        layers.append(Block(width, make_act(kind, c, in_scale, gamma, p, soft_s), arch=arch))
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == 'small':
                nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))
            elif init == 'default':
                pass
            else:
                raise ValueError(init)
            nn.init.zeros_(m.bias)
    return model.to(dev)


def grad_norm(model):
    t = 0.0
    for p in model.parameters():
        if p.grad is not None:
            t += float(p.grad.norm().item() ** 2)
    return t ** 0.5


def train_one(cfg, steps, width, dev):
    torch.manual_seed(cfg['seed'])
    f = TARGETS[cfg['target']]
    xtr = torch.linspace(-3, 3, 512, device=dev).unsqueeze(1)
    ytr = f(xtr)
    xte = torch.linspace(-3.5, 3.5, 400, device=dev).unsqueeze(1)
    yte = f(xte)
    model = build_model(cfg['act'], cfg['c'], cfg['depth'], width, cfg['init'], cfg['arch'], cfg['in_scale'], cfg['gamma'], cfg['p'], cfg['soft_s'], dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_100 = loss_200 = loss_400 = float('nan')
    maxg, bad = 0.0, False
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = ((model(xtr) - ytr) ** 2).mean()
        if not torch.isfinite(loss):
            bad = True
            break
        loss.backward()
        g = grad_norm(model)
        if not math.isfinite(g):
            bad = True
            break
        maxg = max(maxg, g)
        opt.step()
        if step == 100: loss_100 = float(loss.item())
        if step == 200: loss_200 = float(loss.item())
        if step == 400: loss_400 = float(loss.item())
    if dev == 'mps':
        torch.mps.synchronize()
    with torch.no_grad():
        final = float('nan') if bad else float(((model(xtr) - ytr) ** 2).mean().item())
        test = float('nan') if bad else float(((model(xte) - yte) ** 2).mean().item())
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'device': dev,
        'target': cfg['target'],
        'act': cfg['act'],
        'c': cfg['c'],
        'depth': cfg['depth'],
        'init': cfg['init'],
        'arch': cfg['arch'],
        'lr': cfg['lr'],
        'seed': cfg['seed'],
        'in_scale': cfg['in_scale'],
        'gamma': cfg['gamma'],
        'p': cfg['p'],
        'soft_s': cfg['soft_s'],
        'loss_100': loss_100,
        'loss_200': loss_200,
        'loss_400': loss_400,
        'loss_final': final,
        'test_loss': test,
        'max_grad': maxg,
        'nan_or_inf': bad,
        'duration_sec': time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='experiments_v6.csv')
    ap.add_argument('--device', default='auto', choices=['auto','cpu','mps'])
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--targets', nargs='+', default=['sin3_quad','sin_mix'])
    ap.add_argument('--depths', type=int, nargs='+', default=[8,16,32])
    ap.add_argument('--acts', nargs='+', default=['capexp','precap'])
    ap.add_argument('--cs', type=float, nargs='+', default=[0.5,0.3])
    ap.add_argument('--inits', nargs='+', default=['default','small'])
    ap.add_argument('--archs', nargs='+', default=['res_zero','res_inv','plain'])
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-3])
    ap.add_argument('--in-scales', type=float, nargs='+', default=[1.0,0.75,0.5])
    ap.add_argument('--gammas', type=float, nargs='+', default=[0.0,0.01,0.03])
    ap.add_argument('--ps', type=int, nargs='+', default=[2,3])
    ap.add_argument('--soft-ss', type=float, nargs='+', default=[1.0,1.5])
    a = ap.parse_args()

    dev = 'mps' if a.device == 'auto' and torch.backends.mps.is_available() else ('cpu' if a.device == 'auto' else a.device)

    jobs = []
    for seed in range(a.seeds):
        for target in a.targets:
            for depth in a.depths:
                for init in a.inits:
                    for arch in a.archs:
                        for c in a.cs:
                            for lr in a.lrs:
                                for in_scale in a.in_scales:
                                    jobs.append(dict(target=target, act='capexp', c=c, depth=depth, init=init, arch=arch, lr=lr, seed=seed,
                                                     in_scale=in_scale, gamma=0.0, p=1, soft_s=1.0))
                                for gamma in a.gammas:
                                    for p in a.ps:
                                        for soft_s in a.soft_ss:
                                            for in_scale in a.in_scales:
                                                jobs.append(dict(target=target, act='precap', c=c, depth=depth, init=init, arch=arch, lr=lr, seed=seed,
                                                                 in_scale=in_scale, gamma=gamma, p=p, soft_s=soft_s))

    with open(a.out, 'w', newline='') as fh:
        writer = None
        for i, cfg in enumerate(jobs, 1):
            row = train_one(cfg, a.steps, a.width, dev)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            print(f"{i}/{len(jobs)} {cfg['target']} {cfg['act']} depth={cfg['depth']} arch={cfg['arch']} c={cfg['c']} scale={cfg['in_scale']} g={cfg['gamma']} p={cfg['p']} s={cfg['soft_s']} test={row['test_loss']:.4g} bad={row['nan_or_inf']}")


if __name__ == '__main__':
    main()
