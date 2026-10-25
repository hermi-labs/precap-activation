import argparse, csv, gc, math, os, sys, time
from datetime import datetime
import torch
import torch.nn as nn

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
}

def make_act(name, c):
    if name == "capexp": return CapExp(c)
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]()

class ResBlock(nn.Module):
    # x + alpha * act(W x); alpha = linearer Stabilitaetsfaktor
    def __init__(self, width, act, c, mode, depth):
        super().__init__()
        self.lin = nn.Linear(width, width); self.act = make_act(act, c)
        if mode == "res_zero":   self.alpha = nn.Parameter(torch.zeros(1))
        elif mode == "res_inv":  self.register_buffer("alpha", torch.tensor(1.0 / math.sqrt(depth)))
        else:                    self.register_buffer("alpha", torch.tensor(1.0))
    def forward(self, x): return x + self.alpha * self.act(self.lin(x))

def build(act, c, depth, width, init, arch):
    layers = [nn.Linear(1, width), make_act(act, c)]
    for i in range(depth - 1):
        if arch.startswith("res"):
            layers.append(ResBlock(width, act, c, arch, depth))
        else:
            layers += [nn.Linear(width, width), make_act(act, c)]
            if arch == "ln4" and (i + 2) % 4 == 0: layers.append(nn.LayerNorm(width))
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)
    if init != "default":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if init == "he": nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif init == "small": nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)
    return model

def run(cfg, steps, width, dev):
    torch.manual_seed(cfg["seed"])
    f = TARGETS[cfg["target"]]
    xtr = torch.linspace(-3, 3, 512, device=dev).unsqueeze(1); ytr = f(xtr)
    xte = torch.linspace(-3.5, 3.5, 400, device=dev).unsqueeze(1); yte = f(xte)
    model = build(cfg["act"], cfg["c"], cfg["depth"], width, cfg["init"], cfg["arch"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), cfg["lr"])
    maxg, bad = 0.0, False
    marks = {100: float("nan"), 200: float("nan"), 400: float("nan"), steps: float("nan")}
    t0 = time.time()
    for s in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = ((model(xtr) - ytr) ** 2).mean()
        lv = loss.item()
        if not math.isfinite(lv) or lv > 1e12: bad = True; break
        loss.backward()
        g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        if not math.isfinite(g): bad = True; break
        maxg = max(maxg, g); opt.step()
        if s in marks: marks[s] = lv
    with torch.no_grad():
        test = float("nan") if bad else ((model(xte) - yte) ** 2).mean().item()
    alphas = [m.alpha.item() for m in model.modules() if isinstance(m, ResBlock)]
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "device": dev,
           "target": cfg["target"], "act": cfg["act"], "c": cfg["c"], "depth": cfg["depth"],
           "init": cfg["init"], "arch": cfg["arch"], "lr": cfg["lr"], "seed": cfg["seed"],
           "loss_100": marks[100], "loss_200": marks[200], "loss_400": marks[400],
           "loss_final": marks[steps], "test_loss": test, "max_grad": maxg,
           "nan_or_inf": bad, "alpha_mean": (sum(alphas) / len(alphas)) if alphas else float("nan"),
           "duration_sec": time.time() - t0}
    del model, opt, xtr, ytr, xte, yte
    gc.collect()
    if dev == "mps": torch.mps.empty_cache()
    return row

KEYS = ("target", "act", "c", "depth", "init", "arch", "lr", "seed")
def key(d): return tuple(str(d[k]) for k in KEYS)

def bench(width):
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        cfg = dict(target="sin3_quad", act="silu", c=float("nan"), depth=16, init="default", arch="plain", lr=1e-3, seed=0)
        t = time.time(); run(cfg, 200, width, dev); dt = time.time() - t
        print(f"{dev}: {dt:.1f}s fuer 200 Steps, Tiefe 16 -> ca. {dt / 200 * 800 / 16:.2f} s pro Layer bei 800 Steps")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments_v5.csv")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 8, 16, 32, 64])
    ap.add_argument("--cs", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--acts", nargs="+", default=["relu", "gelu", "silu", "capexp"])
    ap.add_argument("--inits", nargs="+", default=["default", "small", "he"])
    ap.add_argument("--archs", nargs="+", default=["plain", "res_inv", "res_zero"])
    ap.add_argument("--targets", nargs="+", default=["sin3_quad"])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("K", "N"))
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--sec-per-layer", type=float, default=1.0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    if a.bench: bench(a.width); return
    dev = a.device
    if dev == "auto": dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = a.out if a.shard[1] == 1 else a.out.replace(".csv", f"_s{a.shard[0]}.csv")

    acts = []
    for n in a.acts:
        if n == "capexp": acts += [("capexp", c) for c in a.cs]
        else: acts.append((n, float("nan")))
    grid = [dict(target=t, act=n, c=c, depth=d, init=i, arch=ar, lr=a.lr, seed=s)
            for s in range(a.seeds) for t in a.targets for d in a.depths
            for ar in a.archs for i in a.inits for (n, c) in acts]
    grid = [g for k, g in enumerate(grid) if k % a.shard[1] == a.shard[0]]

    done = set()
    if os.path.exists(out):
        with open(out) as fh: done = {key(r) for r in csv.DictReader(fh)}
    todo = [g for g in grid if key(g) not in done]
    est = sum(g["depth"] for g in todo) * a.sec_per_layer * a.steps / 800
    print(f"[{dev}] Grid {len(grid)} | fertig {len(grid) - len(todo)} | offen {len(todo)} | Schaetzung ca. {est / 3600:.1f} h", flush=True)
    if a.dry: return

    new = 0
    for n, cfg in enumerate(todo, 1):
        while os.path.exists("PAUSE"): time.sleep(5)
        row = run(cfg, a.steps, a.width, dev)
        write_header = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
        new += 1
        print(f"{n}/{len(todo)} {cfg['arch']} L={cfg['depth']} {cfg['act']} c={cfg['c']} {cfg['init']} s={cfg['seed']} "
              f"test={row['test_loss']:.4g} bad={row['nan_or_inf']} {row['duration_sec']:.0f}s", flush=True)
        if a.max_runs and new >= a.max_runs and n < len(todo):
            print("Neustart zur Speicher-Bereinigung ...", flush=True); sys.exit(75)

if __name__ == "__main__":
    main()
