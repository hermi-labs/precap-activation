# Idee A (privat): Tiefen-PreCap. chi1 = 1 + k/L (L = Zahl der verdeckten Schichten), k in {1, 2}.
# Form wie SPreCap (s=1.5, c=1, tanh-Vorbegrenzer); (lam, alpha, b) exakt geloest: E[f]=0, E[f^2]=1, E[f'^2]=chi1 fuer z~N(0,1).
# Zusaetzlich: Steigung der Varianzabbildung dnu'/dnu am Fixpunkt (q=1), Wertebereich, vorhergesagtes Gradientengefaelle.
import math, json, numpy as np
from scipy.optimize import fsolve
X = np.linspace(-12, 12, 480001); W = np.exp(-X * X / 2); W = W / W.sum()
S, C = 1.5, 1.0
def f(x, lam, al, b):
    th = np.tanh(b * x / S); w = S * th; dw = b * (1 - th * th); ec = math.e ** C
    ce = np.where(w <= C, np.exp(np.minimum(w, C)) - 1, ec - 1 + ec * (w - C)); dce = np.where(w <= C, np.exp(np.minimum(w, C)), ec)
    k = np.where(x > 0, lam, lam * al); return k * ce, k * dce * dw
def mom(p, q=1.0):
    y, d = f(np.sqrt(q) * X, *p); return np.array([np.sum(W * y), np.sum(W * y * y), np.sum(W * d * d)])
def solve(chi, x0):
    g = lambda p: mom(p) - np.array([0, 1, chi]); p = fsolve(g, x0, xtol=1e-13); return p, float(np.abs(g(p)).max())
def slope(p, h=1e-4): return float((mom(p, 1 + h)[1] - mom(p, 1 - h)[1]) / (2 * h))
CS = (1.102702500, 2.053776267, 0.71188112); E03 = (1.985552225, 1.608136397, 0.4118896262)
if __name__ == '__main__':
    out = {}
    for name, p in [('spreccap_cs', CS), ('spreccap_e03', E03)]:
        m = mom(p); y, _ = f(np.array([-60.0, 60.0]), *p)
        out[name] = dict(lam=p[0], alpha=p[1], b=p[2], chi1=float(m[2]), mean=float(m[0]), var=float(m[1]), dnu=slope(p), lo=float(y[0]), hi=float(y[1]))
    for k in [1, 2]:
        x0 = CS
        for L in [50, 100, 200]:
            chi = 1 + k / L; p, r = solve(chi, x0); x0 = p; y, _ = f(np.array([-60.0, 60.0]), *p)
            out[f'kL{k}_d{L}'] = dict(lam=float(p[0]), alpha=float(p[1]), b=float(p[2]), chi1=chi, resid=r, dnu=slope(p), lo=float(y[0]), hi=float(y[1]),
                                     pred_log10_grad=(L - 1) / 2 * math.log10(chi))
    for L in [50, 100, 200]:
        for n in ['spreccap_cs', 'spreccap_e03']: out[n][f'pred_log10_grad_d{L}'] = (L - 1) / 2 * math.log10(out[n]['chi1'])
    for k, v in out.items(): print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items()})
    json.dump(out, open('a_constants.json', 'w'), indent=1)
