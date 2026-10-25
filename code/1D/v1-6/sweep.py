import argparse
import csv
import math
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def capexp(x, c=1.0):
    xc = torch.clamp(x, max=c)
    return torch.where(
        x <= c,
        torch.exp(xc) - 1,
        math.exp(c) - 1 + math.exp(c) * (x - c),
    )


class CapExp(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c

    def forward(self, x):
        return capexp(x, self.c)


class Blend(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
        self.b = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.b * capexp(x, self.c)


class WrappedAct(nn.Module):
    def __init__(self, factory, c=None):
        super().__init__()
        self.inner = factory() if c is None else factory(c=c)

    def forward(self, x):
        return self.inner(x)


ACTIVATIONS = [
    {"name": "relu", "factory": nn.ReLU, "uses_c": False},
    {"name": "gelu", "factory": nn.GELU, "uses_c": False},
    {"name": "silu", "factory": nn.SiLU, "uses_c": False},
    {"name": "capexp", "factory": CapExp, "uses_c": True},
    {"name": "blend", "factory": Blend, "uses_c": True},
]


def build_model(act_factory, depth, width=64, use_layernorm=True, c=None):
    layers = []
    in_features = 1
    for _ in range(depth):
        layers.append(nn.Linear(in_features, width))
        if use_layernorm:
            layers.append(nn.LayerNorm(width))
        layers.append(WrappedAct(act_factory, c=c))
        in_features = width
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers).to(DEV)


def make_data():
    f = lambda x: torch.sin(3 * x) + 0.3 * x ** 2
    x_train = torch.linspace(-3, 3, 512, device=DEV).unsqueeze(1)
    y_train = f(x_train)
    x_test = torch.linspace(-3.5, 3.5, 400, device=DEV).unsqueeze(1)
    y_test = f(x_test)
    return x_train, y_train, x_test, y_test


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.norm().item() ** 2)
    return total ** 0.5


def run_one(act_name, act_factory, seed, depth, width, steps, use_layernorm, lr, c=None):
    torch.manual_seed(seed)
    model = build_model(act_factory, depth, width=width, use_layernorm=use_layernorm, c=c)
    x_train, y_train, x_test, y_test = make_data()
    opt = torch.optim.Adam(model.parameters(), lr)

    g0 = None
    max_grad = 0.0
    nan_or_inf = False
    start = time.time()
    last_loss = float("nan")

    for step in range(steps):
        opt.zero_grad()
        pred = model(x_train)
        loss = ((pred - y_train) ** 2).mean()

        if not torch.isfinite(loss):
            nan_or_inf = True
            break

        loss.backward()
        current_grad = grad_norm(model)

        if not math.isfinite(current_grad):
            nan_or_inf = True
            break

        max_grad = max(max_grad, current_grad)
        if step == 0:
            g0 = current_grad

        opt.step()
        last_loss = loss.item()

    if DEV == "mps":
        torch.mps.synchronize()

    duration = time.time() - start

    if nan_or_inf:
        train_loss = float("nan")
        test_loss = float("nan")
    else:
        with torch.no_grad():
            test_loss = ((model(x_test) - y_test) ** 2).mean().item()
            train_loss = last_loss

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": DEV,
        "activation": act_name,
        "seed": seed,
        "depth": depth,
        "width": width,
        "steps": steps,
        "use_layernorm": use_layernorm,
        "c": "" if c is None else c,
        "lr": lr,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "grad0": g0,
        "max_grad": max_grad,
        "nan_or_inf": nan_or_inf,
        "duration_sec": duration,
    }


FIELDS = [
    "timestamp",
    "device",
    "activation",
    "seed",
    "depth",
    "width",
    "steps",
    "use_layernorm",
    "c",
    "lr",
    "train_loss",
    "test_loss",
    "grad0",
    "max_grad",
    "nan_or_inf",
    "duration_sec",
]


def ensure_csv_header(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()


def row_key(row):
    c_val = "" if row["c"] in (None, "", "nan") else str(row["c"])
    return (
        row["activation"],
        int(row["seed"]),
        int(row["depth"]),
        int(row["width"]),
        int(row["steps"]),
        str(row["use_layernorm"]),
        c_val,
        str(row["lr"]),
    )


def cfg_key(cfg):
    c_val = "" if cfg["c"] is None else str(cfg["c"])
    return (
        cfg["activation"],
        int(cfg["seed"]),
        int(cfg["depth"]),
        int(cfg["width"]),
        int(cfg["steps"]),
        str(cfg["use_layernorm"]),
        c_val,
        str(cfg["lr"]),
    )


def load_done(path):
    done = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "lr" not in row:
                continue
            done.add(row_key(row))
    return done


def append_result(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)
        f.flush()


def build_grid(args):
    grid = []
    for seed in range(args.seeds):
        for act in ACTIVATIONS:
            c_values = args.cs if act["uses_c"] else [None]
            for use_ln in [True, False]:
                for depth in args.depths:
                    for lr in args.lrs:
                        for c in c_values:
                            grid.append({
                                "activation": act["name"],
                                "factory": act["factory"],
                                "seed": seed,
                                "depth": depth,
                                "width": args.width,
                                "steps": args.steps,
                                "use_layernorm": use_ln,
                                "c": c,
                                "lr": lr,
                            })
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments_v2.csv")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 8, 16, 32])
    ap.add_argument("--cs", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--width", type=int, default=64)
    args = ap.parse_args()

    ensure_csv_header(args.out)
    done = load_done(args.out)
    grid = build_grid(args)
    todo = [cfg for cfg in grid if cfg_key(cfg) not in done]

    print(f"Device: {DEV}")
    print(f"CSV: {args.out}")
    print(f"{len(todo)} Laeufe offen von {len(grid)}")

    for i, cfg in enumerate(todo, 1):
        row = run_one(
            act_name=cfg["activation"],
            act_factory=cfg["factory"],
            seed=cfg["seed"],
            depth=cfg["depth"],
            width=cfg["width"],
            steps=cfg["steps"],
            use_layernorm=cfg["use_layernorm"],
            lr=cfg["lr"],
            c=cfg["c"],
        )
        append_result(args.out, row)

        c_text = "-" if cfg["c"] is None else str(cfg["c"])
        train_text = "nan" if math.isnan(row["train_loss"]) else f"{row['train_loss']:.6f}"
        test_text = "nan" if math.isnan(row["test_loss"]) else f"{row['test_loss']:.6f}"
        grad0_text = "nan" if row["grad0"] is None else f"{row['grad0']:.2f}"

        print(
            f"{i:4d}/{len(todo)} "
            f"{row['activation']:7s} "
            f"L={row['depth']:2d} "
            f"LN={str(row['use_layernorm']):5s} "
            f"c={c_text:>4s} "
            f"seed={row['seed']:2d} "
            f"lr={row['lr']:.0e} -> "
            f"train {train_text} | "
            f"test {test_text} | "
            f"grad0 {grad0_text} | "
            f"gmax {row['max_grad']:.2f} | "
            f"{row['duration_sec']:.1f}s",
            flush=True
        )


if __name__ == "__main__":
    main()
