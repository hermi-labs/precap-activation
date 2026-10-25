import argparse, csv, math, time
from datetime import datetime

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


def capexp(x, c=0.5):
    xc = torch.clamp(x, max=c)
    return torch.where(x <= c, torch.exp(xc) - 1, math.exp(c) - 1 + math.exp(c) * (x - c))


class CapExp(nn.Module):
    def __init__(self, c=0.5):
        super().__init__()
        self.c = c
    def forward(self, x):
        return capexp(x, self.c)


class RegulatedPreAct(nn.Module):
    def __init__(self, c=0.5, gamma=0.03, p=3, scale=1.0, mode="softclip"):
        super().__init__()
        self.c = c
        self.gamma = gamma
        self.p = p
        self.scale = scale
        self.mode = mode
    def forward(self, x):
        u = x + self.gamma * (x ** self.p)
        if self.mode == "softclip":
            u = self.scale * torch.tanh(u / self.scale)
        elif self.mode == "linear":
            u = self.scale * u
        elif self.mode == "none":
            pass
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return capexp(u, self.c)


TARGETS = {
    "sin3_quad": lambda x: torch.sin(3 * x) + 0.3 * x ** 2,
    "sin_mix": lambda x: torch.sin(2 * x) + 0.5 * torch.sin(5 * x),
    "abs_wave": lambda x: torch.abs(x) * torch.sin(2 * x),
}


def make_act(name, c, gamma, p, scale, mode):
    if name == "capexp":
        return CapExp(c)
    if name == "precap":
        return RegulatedPreAct(c=c, gamma=gamma, p=p, scale=scale, mode=mode)
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]()


def build(act, c, depth, width, init, gamma, p, scale, mode, dev):
    layers = [nn.Linear(1, width), make_act(act, c, gamma, p, scale, mode)]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), make_act(act, c, gamma, p, scale, mode)]
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)
    if init != "default":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if init == "small":
                    nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))
                elif init == "he":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    return model.to(dev)


def grad_norm(model):
    t = 0.0
    for p in model.parameters():
        if p.grad is not None:
            t += float(p.grad.norm().item() ** 2)
    return t ** 0.5


def run(cfg, steps=800, width=64, dev="cpu"):
    torch.manual_seed(cfg["seed"])
    f = TARGETS[cfg["target"]]
    xtr = torch.linspace(-3, 3, 512, device=dev).unsqueeze(1)
    ytr = f(xtr)
    xte = torch.linspace(-3.5, 3.5, 400, device=dev).unsqueeze(1)
    yte = f(xte)
    model = build(cfg["act"], cfg["c"], cfg["depth"], width, cfg["init"], cfg["gamma"], cfg["p"], cfg["scale"], cfg["mode"], dev)
    opt = torch.optim.Adam(model.parameters(), cfg["lr"])
    maxg, bad = 0.0, False
    t0 = time.time()
    for _ in range(steps):
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
    if dev == "mps":
        torch.mps.synchronize()
    with torch.no_grad():
        test = float("nan") if bad else ((model(xte) - yte) ** 2).mean().item()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": dev,
        "target": cfg["target"],
        "act": cfg["act"],
        "c": cfg["c"],
        "depth": cfg["depth"],
        "init": cfg["init"],
        "gamma": cfg["gamma"],
        "p": cfg["p"],
        "scale": cfg["scale"],
        "mode": cfg["mode"],
        "lr": cfg["lr"],
        "seed": cfg["seed"],
        "test_loss": test,
        "max_grad": maxg,
        "nan_or_inf": bad,
        "duration_sec": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="precap_results.csv")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--targets", nargs="+", default=["sin3_quad"])
    ap.add_argument("--inits", nargs="+", default=["default", "small"])
    ap.add_argument("--cs", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1])
    ap.add_argument("--ps", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5])
    ap.add_argument("--modes", nargs="+", default=["softclip"])
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    if torch is None:
        raise SystemExit("PyTorch ist in diesem Python nicht installiert. Bitte mit deiner .venv starten.")

    dev = a.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = []
    for seed in range(a.seeds):
        for target in a.targets:
            for depth in a.depths:
                for init in a.inits:
                    for c in a.cs:
                        rows.append(dict(target=target, act="capexp", c=c, depth=depth, init=init,
                                         gamma=0.0, p=1, scale=1.0, mode="baseline", lr=a.lr, seed=seed))
                        for gamma in a.gammas:
                            for p in a.ps:
                                for scale in a.scales:
                                    for mode in a.modes:
                                        rows.append(dict(target=target, act="precap", c=c, depth=depth, init=init,
                                                         gamma=gamma, p=p, scale=scale, mode=mode, lr=a.lr, seed=seed))

    with open(a.out, "w", newline="") as fh:
        writer = None
        for i, cfg in enumerate(rows, 1):
            row = run(cfg, steps=a.steps, width=a.width, dev=dev)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            print(f"{i}/{len(rows)} {cfg['act']} depth={cfg['depth']} c={cfg['c']} gamma={cfg['gamma']} p={cfg['p']} scale={cfg['scale']} mode={cfg['mode']} test={row['test_loss']:.4g} bad={row['nan_or_inf']}")


if __name__ == "__main__":
    main()
