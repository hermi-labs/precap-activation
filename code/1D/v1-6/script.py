import math, torch, torch.nn as nn
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

def capexp(x, c=1.0):
    xc = torch.clamp(x, max=c)
    return torch.where(x <= c, torch.exp(xc) - 1,
                       math.exp(c) - 1 + math.exp(c) * (x - c))

class CapExp(nn.Module):
    def __init__(s, c=1.0): super().__init__(); s.c = c
    def forward(s, x): return capexp(x, s.c)

class Blend(nn.Module):
    def __init__(s, c=1.0):
        super().__init__(); s.c = c; s.b = nn.Parameter(torch.tensor(0.1))
    def forward(s, x): return x + s.b * capexp(x, s.c)

def run(act, L, w=64, steps=800):
    layers, i = [], 1
    for _ in range(L):
        layers += [nn.Linear(i, w), nn.LayerNorm(w), act()]; i = w
    m = nn.Sequential(*layers, nn.Linear(w, 1)).to(dev)
    f = lambda x: torch.sin(3*x) + 0.3*x**2
    x = torch.linspace(-3, 3, 512, device=dev).unsqueeze(1)
    xt = torch.linspace(-3.5, 3.5, 400, device=dev).unsqueeze(1)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for s in range(steps):
        opt.zero_grad(); loss = ((m(x) - f(x))**2).mean(); loss.backward()
        if s == 0:
            g0 = sum(p.grad.norm()**2 for p in m.parameters()).sqrt().item()
        opt.step()
    with torch.no_grad():
        te = ((m(xt) - f(xt))**2).mean().item()
    return f"train {loss.item():.4f} | test(+Extrap.) {te:.4f} | grad0 {g0:.2f}"

for name, a in [("relu", nn.ReLU), ("gelu", nn.GELU),
                ("capexp", CapExp), ("blend", Blend)]:
    for L in (2, 8):
        print(f"{name:7s} L={L}: {run(a, L)}")
