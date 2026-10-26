"""PreCap/SPreCap Referenz (reines PyTorch, Autograd). Massgeblich fuer alle Kernel-Tests.

Allgemeine Form (s, c fest):
    capexp(w) = exp(w) - 1                      fuer w <= c
              = e^c - 1 + e^c * (w - c)         fuer w >  c
    y(x) = lam * (1 falls x > 0, sonst alpha) * capexp(s * tanh(b * x / s))
PreCap0:            lam=1, alpha=1, b=1, s=1.5, c=1
SPreCap (chi1=1.13): lam=0.80676869644, alpha=2.39695130922, b=1
SPreCap-cs:         lam=1.102702500,   alpha=2.053776267,   b=0.71188112
"""
import math, torch
PRESETS = {
    'precap0':     dict(lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0),
    'spreccap':    dict(lam=0.80676869644, alpha=2.39695130922, b=1.0, s=1.5, c=1.0),
    'spreccap_cs': dict(lam=1.102702500, alpha=2.053776267, b=0.71188112, s=1.5, c=1.0),
}
def precap_ref(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    w = s * torch.tanh(b * x / s)
    ec = math.exp(c)
    ce = torch.where(w <= c, torch.exp(torch.clamp(w, max=c)) - 1, ec - 1 + ec * (w - c))
    return lam * torch.where(x > 0, ce, alpha * ce)
def precap_ref_grad(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    """Analytische Ableitung dy/dx (zur Kontrolle)."""
    t = torch.tanh(b * x / s); w = s * t
    dce = torch.where(w <= c, torch.exp(torch.clamp(w, max=c)), torch.full_like(w, math.exp(c)))
    return lam * torch.where(x > 0, torch.ones_like(x), torch.full_like(x, alpha)) * dce * b * (1 - t * t)
