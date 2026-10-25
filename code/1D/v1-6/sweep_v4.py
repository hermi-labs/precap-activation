import argparse, csv, math, os, time
from datetime import datetime
import torch
import torch.nn as nn

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def capexp(x, c=0.5):
    xc = torch.clamp(x, max=c)
    return torch.where(x <= c, torch.exp(xc) - 1, math.exp(c) - 1 + math.exp(c) * (x - c))

class CapExp(nn.Module):
    def __init__(self, c=0.5):
        super().__init__(); self.c = c
    def forward(self, x): return capexp(x, self.c)

TARGETS = {
    "sin3_quad": lambda x: torch.sin(3 * x) + 0.3 * x ** 2,
    "sin_mix":   lambda x: torch.sin(2 * x) + 0.5 * torch.sin(5 * x),
    "abs_wave":  lambda x: torch.abs(x) * torch.sin(2 * x),
    "tanh_step": lambda x: torch.tanh(4 * x) + 0.2 * x,
    "cubic_sin": lambda x: 0.1 * x ** 3 + torch.sin(4 * x),
}

def make_act(name, c):
    if name == "capexp": return CapExp(c)
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]()

class ResBlock(nn.Module):
    def __init__(self, width, act, c):
        super().__init__()
        self.lin = nn.Linear(width, width); self.act = make_act(act, c)
    def forward(self, x): return x + self.act(self.lin(x))

def build(act, c, depth, width, init, residual):
    layers = [nn.Linear(1, width), make_act(act, c)]
    for _ in range(depth - 1):
        if residual: layers.append(ResBlock(width, act, c))
        else: layers += [nn.Linear(width, width), make_act(act, c)]
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)
    if init != "default":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if init == "he": nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif init == "xavier": nn.init.xavier_normal_(m.weight)
                elif init == "small": nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)
    return model.to(DEV)

def grad_norm(model):
    t = 0.0
    for p in model.parameters():
        if p.grad is not None: t += float(p.grad.norm().item() ** 2)
    return t ** 0.5

def run(cfg, steps, width):
    torch.manual_seed(cfg["seed"])
    f = TARGETS[cfg["target"]]
    xtr = torch.linspace(-3, 3, 512, device=DEV).unsqueeze(1); ytr = f(xtr)
    xte = torch.linspace(-3.5, 3.5, 400, device=DEV).unsqueeze(1); yte = f(xte)
    model = build(cfg["act"], cfg["c"], cfg["depth"], width, cfg["init"], cfg["residual"])
    opt = torch.optim.Adam(model.parameters(), cfg["lr"])
    maxg, bad = 0.0, False
    marks = {100: float("nan"), 200: float("nan"), 400: float("nan"), steps: float("nan")}
    t0 = time.time()
    for s in range(1, steps + 1):
        opt.zero_grad()
        loss = ((model(xtr) - ytr) ** 2).mean()
        if not torch.isfinite(loss): bad = True; break
        loss.backward(); g = grad_norm(model)
        if not math.isfinite(g): bad = True; break
        maxg = max(maxg, g); opt.step()
        if s in marks: marks[s] = loss.item()
    if DEV == "mps": torch.mps.synchronize()
    with torch.no_grad():
        test = float("nan") if bad else ((model(xte) - yte) ** 2).mean().item()
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "device": DEV,
            "target": cfg["target"], "act": cfg["act"], "c": cfg["c"], "depth": cfg["depth"],
            "init": cfg["init"], "residual": cfg["residual"], "lr": cfg["lr"], "seed": cfg["seed"],
            "loss_100": marks[100], "loss_200": marks[200], "loss_400": marks[400],
            "loss_final": marks[steps], "test_loss": test, "max_grad": maxg,
            "nan_or_inf": bad, "duration_sec": time.time() - t0}

KEYS = ("target", "act", "c", "depth", "init", "residual", "lr", "seed")
def key(d): return tuple(str(d[k]) for k in KEYS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments_v4.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depths", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--cs", type=float, nargs="+", default=[0.3, 0.5, 0.75])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3])
    ap.add_argument("--targets", nargs="+", default=["sin3_quad", "sin_mix", "abs_wave"])
    ap.add_argument("--inits", nargs="+", default=["default", "he"])
    a = ap.parse_args()

    grid = []
    for seed in range(a.seeds):
        for tg in a.targets:
            for depth in a.depths:
                for init in a.inits:
                    for res in (False, True):
                        for lr in a.lrs:
                            base = dict(seed=seed, target=tg, depth=depth, init=init, residual=res, lr=lr)
                            for act in ("relu", "gelu", "silu"):
                                grid.append({**base, "act": act, "c": ""})
                            for c in a.cs:
                                grid.append({**base, "act": "capexp", "c": c})

    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, newline="") as f:
            for r in csv.DictReader(f): done.add(key(r))
    todo = [g for g in grid if key(g) not in done]
    print(f"Device: {DEV} | {len(todo)} offen von {len(grid)}", flush=True)

    for i, cfg in enumerate(todo, 1):
        row = run(cfg, a.steps, a.width)
        new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
        with open(a.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row); fh.flush()
        print(f"{i}/{len(todo)} {cfg['target']} {cfg['act']} c={cfg['c']} L={cfg['depth']} "
              f"init={cfg['init']} res={cfg['residual']} s={cfg['seed']} "
              f"-> test {row['test_loss']:.5f} | {row['duration_sec']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
