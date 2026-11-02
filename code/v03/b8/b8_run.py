# B8 (privat, praeregistriert, mit Claude): Haelt der B6-Befund (BPreCap im TAT-Rezept bei 4 Bit besser als TAT) bei Tiefe 200?
#   Code aus B6, einzige Aenderung der Frage: Tiefe 200 statt 100. Varianten kL2B_tr, LB_tr (LR neu getunt, Raster eine Stufe nach unten
#   verschoben: 5e-6 bis 2e-4, weil bei d200 kleinere LR gewaehlt werden) und tat (LR aus B3 d200, 2e-5). Neue Seeds 42144-42155.
#   Kontrolle: kL2B MNIST d100 Seed 42060 bitgleich zu B1.
# --- Kopf aus B6 ---
# B6 (privat, praeregistriert, mit Claude): Kommt der Vorsprung von TAT auf CIFAR-10 (B3) vom Rezept oder von der Form?
#   Einzige Aenderung gegenueber B3: neue Varianten kL2B_tr und LB_tr = dieselbe Form mit dem TAT-Rezept (SUO-orthogonale Init mal Gain,
#   Normierung je Beispiel, wie A2/Q3). Nur d100. LR fuer die _tr-Varianten neu getunt (Raster und Regel wie B3), Referenzen kL2B, LB, tat
#   mit der LR aus B3 (lr_from_b3.csv). Neue Seeds 42120-42131. Kontrolle: kL2B MNIST d100 Seed 42060 bitgleich zu B1.
# --- Kopf aus B3 ---
# B3 (privat, praeregistriert, mit Claude): BPreCap (kL2B) gegen LB, kL2 und TAT auf CIFAR-10 (schwerere Daten).
#   Frage: Traegt der PreCap-Kern auf schwereren Daten etwas bei (B1/B2 auf MNIST/Fashion: nein)? Code aus B2, Aenderungen:
#   Datensatz CIFAR-10 (Binaerversion, 3072 Eingaenge), Varianten kL2B, LB, kL2, TAT (Rezept wie A2/Q3: SUO-Init, Normierung je Beispiel),
#   Lernrate je Variante und Tiefe neu getunt (Raster und Regel wie 3a-3/A3), danach 12 neue Seeds. Quantisierung wie B2 (4 und 3 Bit).
#   Kontrolle: kL2B MNIST d100 Seed 42060 bitgleich zu B1 (Code-Pfad).
# --- Kopf aus B2 ---
# B2 (privat, praeregistriert, mit Claude): weiche Lagma-Huelle gegen harte Grenze bei gleicher Kritikalitaet. Einzige Aenderung
#   der Frage: Huelle weich (M*phi_rho(./M), wie B1) oder hart (clip(., -M, M)), Konstanten jeweils exakt fuer E[f]=0, E[f^2]=1,
#   chi1 = 1 + 2/L geloest (math/p7_hard_bound.py). Kerne: linear (LB/LBh) und PreCap (kL2B = BPreCap / kL2Bh = BPreCap hart).
#   Neu in der Messung (ohne Einfluss auf das Training): 3-Bit-Aktivierungen nach dem Training (w8a3_best, w8a3a_best).
#   Kontrolle: LB und kL2B MNIST d100 Seed 42060 bitgleich zu B1.
# --- Kopf aus B1 ---
# B1 (privat, praeregistriert, mit Claude): Tiefenkalibrierung und schmaler Bereich getrennt. Neue Form kL2B = M*phi_rho(PreCap(x)/M)
#   mit M = 3 und (lam, alpha, b) MIT der Grenze exakt geloest (E[f]=0, E[f^2]=1, E[f'^2]=1+2/L; math/p4_bounded_depth.py).
#   Referenzen: kL2 (ohne Grenze), LB (linearer Kern + dieselbe weiche Grenze, (beta, M) exakt fuer Varianz 1 und chi1 = 1 + 2/L,
#   math/p5_linear_bound.py; Gegenprobe nach der kritischen hard-tanh-Familie, Pennington et al. 2017: braucht es den PreCap-Kern?),
#   kL2_LD100 (Lagma-Standardform aus Lagma-5), spreccap_cs (schmaler Bereich, nicht tiefenkalibriert).
#   Neu gegenueber Lagma-5: asymmetrische 4-Bit-Quantisierung (aus Q3) als faire Zusatzmessung. Sonst Code wie Lagma-5.
# --- Kopf aus Lagma-5 ---
# Lagma-5: Bestaetigung des explorativen Lagma-4-Befunds LD100 > L1 bei 4 Bit. Code wie Lagma-4, geaendert nur ACTS (kL2, L1, LD100),
#   neue Seeds 42045-42059 und die Kontrolle (LD100 MNIST d100, Seeds 42030/42031, gegen Lagma-4).
# --- Kopf aus Lagma-4 ---
# Lagma-4 (privat): wie LA10 aus Lagma-3, einzige Aenderung: Deckel fuer c (obere Klemme kappa_up*c0 statt 4*c0).
#   LD100: c <= 1,00*c0, LD125: c <= 1,25*c0. Untere Klemme 0,25*c0, Anlauf, Ziel 1 %, eta wie LA10.
#   Neue Seeds 42030-42044; Basis kL2, L1 und LA10 auf diesen Seeds neu gerechnet. Kontrolle: LA10 MNIST d100 (42010/42011) gegen Lagma-3.
# --- Kopf aus Lagma-3 ---
# Lagma-3 (privat): Regler-Grenze wie Lagma-2, einzige Aenderung: der Zielverlauf r*(t, L) der Eingriffsrate.
# Regler (verallgemeinert): log c <- log c + eta*(r - a(t)*r_end)/r_end, a(t) = Anlauffaktor. Mit a = 1 ist das exakt die Regel aus Lagma-2.
#   LA10: r_end = 1 %, Anlauf a(t) = min(1, t/T), T = STEPS/4 (430 Schritte)
#   LT:   r_end = eps*chi1/(L*sup f'^2) mit eps = 0,5 (Schranke E3, lagma/L2_Mathematik.md), kein Anlauf
#   LTA:  r_end wie LT, Anlauf wie LA10
# Klemme, Start c0 = kappa*Q0, eta, Bremse, Quantisierung wie Lagma-2. Basis kL2 aus A, L1 aus Lagma-1, LR10/LR03 aus Lagma-2.
# Kontrolle: kL2_LR10 MNIST d100 (2 Seeds) neu gerechnet gegen Lagma-2.
import argparse, csv, gzip, json, math, os, struct, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', '/Users/aaronschmitt/PyCharmMiscProject/data'))
sys.path.insert(0, str(Path(os.environ.get('PRECAP_SRC', ROOT / 'src'))))

DATASETS = ['cifar10']; DEPTHS = [200]; WIDTH = 256; BATCH = 128; STEPS = 1720; NVAL = 5000
ACTS = ['kL2B_tr', 'LB_tr', 'tat']
REPRO = [('mnist', 100, 'kL2B', 42060)]   # Kontrolle gegen B1 (bitgleich, Code-Pfad)
# TAT-Parameter (eta = 0.9 wie dks) wie A2: d100/d200
TAT = {100: dict(neg=0.5704395323991776, scale=1.2284042441106242), 200: dict(neg=0.6795996427536011, scale=1.169668377971502)}
B_M = 3.0   # Grenze der neuen Form (in Einheiten der Fixpunkt-Standardabweichung 1)
LG_Q, LG_KAPPA, LG_RHO, LG_BETA, LG_L3S = 0.999, 1.5, 0.8, 0.99, 50.0   # wie Lagma-1
LG_ETA = 0.01; LG_TARGET = {'LR10': 0.01, 'LR03': 0.003, 'LA10': 0.01, 'LD100': 0.01, 'LD125': 0.01}   # Regler: Schrittweite in log c, feste Ziel-Eingriffsrate
LG_EPS = 0.5; LG_RAMP = 0.25                                   # LT/LTA: Anteil an der Schranke E3; Anlauf ueber die ersten 25 % der Schritte
REG = ('LR10', 'LR03', 'LA10', 'LT', 'LTA', 'LD100', 'LD125'); RAMPED = ('LA10', 'LTA', 'LD100', 'LD125')
LG_UP = {'LD100': 1.0, 'LD125': 1.25}                          # Deckel: obere Klemme als Vielfaches von c0 (sonst 4)
def isreg(g): return g in REG
# Verlauf der Eingriffsrate: nach STEPS/4, STEPS/2, 3*STEPS/4 (bei 1720: 430, 860, 1290) und am Ende
def base(n): return n.split('_L')[0].removesuffix('_tr')   # B6: _tr = Form mit TAT-Rezept (Init und Eingabe), Form selbst unveraendert
def recipe(n): return n.endswith('_tr')
def lag(n): return ('L' + n.split('_L')[1]) if '_L' in n else None
def tune_acts(depth): return ['kL2B_tr', 'LB_tr']   # B6: nur die neuen Varianten getunt, Referenzen mit der LR aus B3
LR_GRID = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4]               # B8: eine Stufe nach unten (d200)
TUNE_SEEDS = [42000, 42001, 42002, 42003]                     # wie 3a-3
S_, C_ = 1.5, 1.0
CONST = {('spreccap_cs', 0): dict(lam=1.1027025, alpha=2.053776267, b=0.71188112),       # wie Repo-Preset
         ('spreccap_e03', 0): dict(lam=1.985552225, alpha=1.608136397, b=0.4118896262),    # wie Repo-Preset
         ('kL1', 50): dict(lam=2.649586999, alpha=1.461343209, b=0.318766948),  ('kL1', 100): dict(lam=4.170485099, alpha=1.296823062, b=0.212387949),
         ('kL1', 200): dict(lam=6.296620086, alpha=1.197830367, b=0.145678357),
         ('kL2', 50): dict(lam=1.614100769, alpha=1.739743617, b=0.496159697),  ('kL2', 100): dict(lam=2.649586999, alpha=1.461343209, b=0.318766948),
         ('kL2', 200): dict(lam=4.170485099, alpha=1.296823062, b=0.212387949),
         ('kL2B', 100): dict(lam=2.9843721185770757, alpha=1.406734313682772, b=0.28967082304815317),   # math/p4_bounded_depth.py, M = 3
         ('kL2B', 200): dict(lam=4.909725609554288, alpha=1.2489087502974499, b=0.18464022949439854)}
CONST[('LB', 100)] = dict(beta=1.0353165703655918, M=2.2165089809922915)   # math/p5_linear_bound.py, chi1 = 1.02
CONST[('LB', 200)] = dict(beta=1.0187137878416879, M=2.454293815050829)   # chi1 = 1.01
CONST[('LBh', 100)] = dict(beta=1.0225784048613393, M=2.2992458453571873)   # math/p7_hard_bound.py, harte Grenze, chi1 = 1.02
CONST[('LBh', 200)] = dict(beta=1.0111280616465768, M=2.536872795621917)   # chi1 = 1.01
CONST[('kL2Bh', 100)] = dict(lam=3.21415945825123, alpha=1.379357352452743, b=0.27047331339461944)   # BPreCap hart, M = 3
CONST[('kL2Bh', 200)] = dict(lam=5.450386477477196, alpha=1.2253072717443303, b=0.16740005215585388)
def _chi(n, d):
    X = np.linspace(-12, 12, 240001); W = np.exp(-X * X / 2); W = W / W.sum(); h = 1e-6   # feines Gitter (Knick bei 0)
    y1 = np_act(n, X + h, d); y0 = np_act(n, X - h, d); return float(np.sum(W * ((y1 - y0) / (2 * h)) ** 2))
def par(n, depth): n = base(n); return CONST[(n, 0)] if (n, 0) in CONST else CONST[(n, depth)]
STAGE = 'confirm'; TUNEDIR = ROOT / 'tune'
SEEDS = list(range(42144, 42156))           # B8: neue Seeds (B7: 42132-42143), 12 je Zelle
CACHE_LIMIT = 512 * 2**20                   # B8: MLX-Puffercache je Worker begrenzen (B7: etwa 2,5 GB Cache je Worker, Swap). Nur Speicher, Rechnung bitgleich (Mechanik-Test)
NCAL = 1000                                  # Kalibrierung: erste 1000 Trainingsbilder (nicht Test, nicht Val)
SAVE_SEED = -1                               # Gewichte nur fuer diesen Seed speichern (Platz)
# Quantisierungsstufen: (Name, Gewichte, Aktivierungen); 'f8' = FP8 E4M3 simuliert
QUANT = [('w8a8', 'i8', 'i8', 'max'), ('w8a4', 'i8', 'i4', 'max'), ('w4a8', 'i4', 'i8', 'max'),
         ('w8a4_p999', 'i8', 'i4', 'p999'), ('w8a4_p9999', 'i8', 'i4', 'p9999'), ('w8a4_mse', 'i8', 'i4', 'mse4'), ('w8a8_mse', 'i8', 'i8', 'mse8')]
CALS4 = ['max', 'p999', 'p9999', 'mse4']   # Kandidaten fuer w8a4_best (Wahl je Lauf ueber den Fehler auf den Kalibrierbildern)
CALS3 = ['max', 'p999', 'p9999', 'mse3']   # dito fuer 3 Bit (B2, beschreibend/sekundaer)
BACKEND = 'ref'   # alle SPreCap-Varianten (auch cs) ueber dieselbe MLX-Funktion, damit nur der Begrenzer verschieden ist

# --- Aktivierungen ---------------------------------------------------------------
from precap._presets import PRESETS
def pc_mlx(x, lam, alpha, b):
    w = S_ * mx.tanh(b * x / S_)
    ec = math.exp(C_); ce = mx.where(w <= C_, mx.exp(mx.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_))
    return lam * mx.where(x > 0, ce, alpha * ce)
def make_act(n, depth):
    n = base(n)
    if n == 'tat': q = TAT[depth]; return lambda x: q['scale'] * mx.where(x > 0, x, q['neg'] * x)
    if n == 'selu': return nn.selu
    if n == 'elu': return nn.elu
    p = par(n, depth)
    if n == 'kL2B': Mb = mx.array(B_M); return lambda x: brake(pc_mlx(x, **p), Mb)
    if n == 'LB': Mb = mx.array(p['M']); return lambda x: brake(p['beta'] * x, Mb)
    if n == 'kL2Bh': return lambda x: mx.clip(pc_mlx(x, **p), -B_M, B_M)
    if n == 'LBh': return lambda x: mx.clip(p['beta'] * x, -p['M'], p['M'])
    return lambda x: pc_mlx(x, **p)

def np_act(n, x, depth=0):  # float64-Referenz fuer die Init-Gains
    n = base(n)
    if n == 'tat': q = TAT[depth]; return q['scale'] * np.where(x > 0, x, q['neg'] * x)
    if n == 'selu': l, a = 1.0507009873554805, 1.6732632423543772; return l * np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1))
    if n == 'elu': return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
    p = par(n, depth)
    if n == 'LBh': return np.clip(p['beta'] * x, -p['M'], p['M'])
    if n == 'LB':
        u = p['beta'] * x / p['M']; a = np.abs(u); return p['M'] * np.where(a <= LG_RHO, u, np.sign(u) * (LG_RHO + (1 - LG_RHO) * np.tanh((a - LG_RHO) / (1 - LG_RHO))))
    w = S_ * np.tanh(p['b'] * x / S_)
    ec = math.exp(C_); ce = np.where(w <= C_, np.exp(np.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_))
    y = p['lam'] * np.where(x > 0, ce, p['alpha'] * ce)
    if n == 'kL2Bh': return np.clip(y, -B_M, B_M)
    if n == 'kL2B':   # dieselbe Bremse wie brake(), in float64
        u = y / B_M; a = np.abs(u); return B_M * np.where(a <= LG_RHO, u, np.sign(u) * (LG_RHO + (1 - LG_RHO) * np.tanh((a - LG_RHO) / (1 - LG_RHO))))
    return y
def gain(n, depth):  # wie 3a-3: g = 1/sqrt(E[f(Z)^2]), Gauss-Hermite 150 Knoten
    X, W = np.polynomial.hermite_e.hermegauss(150); W = W / W.sum(); y = np_act(n, X, depth); return float(1 / np.sqrt(np.sum(W * y * y)))
ANAMES = ['kL2', 'kL2B', 'LB', 'kL2Bh', 'LBh', 'spreccap_cs', 'selu']
GAINS = {(n, d): gain(n, d) for n in ANAMES for d in [100, 200]}
CHI1 = {(n, d): _chi(n, d) for n in ANAMES for d in [100, 200]}
def _supf2(n, d):  # sup f'(x)^2, numerisch auf feinem Gitter
    X = np.linspace(-12, 12, 2400001); h = 1e-6; return float((((np_act(n, X + h, d) - np_act(n, X - h, d)) / (2 * h)) ** 2).max())
SUPF2 = {d: _supf2('kL2', d) for d in [100, 200]}
def r_end(g, depth):  # Endziel der Eingriffsrate; LT/LTA aus E3: eps*chi1/(L*sup f'^2) (d100 0,282 %, d200 0,127 %)
    if g in ('LT', 'LTA'): return LG_EPS * CHI1[('kL2', depth)] / (depth * SUPF2[depth])
    return LG_TARGET[g]

# --- Daten (identisch zu 3a-3) ------------------------------------------------------
def idx(p):
    with gzip.open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xff; sh = struct.unpack('>' + 'I' * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(sh)
_C = {}
def load(ds):
    if ds in _C: return _C[ds]
    if ds == 'cifar10':   # Binaerversion: je Datensatz 1 Byte Klasse + 3072 Bytes Bild (R, G, B je 32x32), keine Pickles
        def rd(fs):
            a = np.concatenate([np.fromfile(DATA / 'cifar10' / 'cifar-10-batches-bin' / f, dtype=np.uint8).reshape(-1, 3073) for f in fs])
            return a[:, 1:].astype(np.float32) / 255, a[:, 0].astype(np.int64)
        xtr, ytr = rd([f'data_batch_{i}.bin' for i in range(1, 6)]); xte, yte = rd(['test_batch.bin'])
        xv, yv = xtr[-NVAL:], ytr[-NVAL:]; xtr, ytr = xtr[:-NVAL], ytr[:-NVAL]
        m, s = xtr.mean(), xtr.std(); T = lambda a: ((a - m) / s).astype(np.float32)
        _C[ds] = (T(xtr), ytr, T(xv), yv, T(xte), yte); return _C[ds]
    if ds.startswith('synth'):  # nur fuer Tests ohne echte Daten
        r = np.random.default_rng(0); X = r.normal(size=(12000, 784)).astype(np.float32); Y = (X[:, :10].argmax(1)).astype(np.int64)
        xtr, ytr, xv, yv, xte, yte = X[:6000], Y[:6000], X[6000:9000], Y[6000:9000], X[9000:], Y[9000:]
    else:
        d = DATA / ds
        xtr = idx(d / 'train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255; ytr = idx(d / 'train-labels-idx1-ubyte.gz').astype(np.int64)
        xte = idx(d / 't10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255; yte = idx(d / 't10k-labels-idx1-ubyte.gz').astype(np.int64)
        xv, yv = xtr[-NVAL:], ytr[-NVAL:]; xtr, ytr = xtr[:-NVAL], ytr[:-NVAL]
    m, s = xtr.mean(), xtr.std(); T = lambda a: ((a - m) / s).astype(np.float32)
    _C[ds] = (T(xtr), ytr, T(xv), yv, T(xte), yte); return _C[ds]

def pex(a): return (a / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 1e-12)).astype(np.float32)   # TAT-Rezept wie A2/Q3
def data_for(ds, act):
    d = load(ds)
    if act != 'tat' and not recipe(act): return d   # B6: _tr-Varianten bekommen die Normierung je Beispiel wie TAT
    k = ds + '_pex'
    if k not in _C: xtr, ytr, xv, yv, xte, yte = d; _C[k] = (pex(xtr), ytr, pex(xv), yv, pex(xte), yte)
    return _C[k]

# --- Netz ----------------------------------------------------------------------------
def suo(r, b, a):  # skaliert-gleichverteilt-orthogonal wie dks (identisch zu A2/Q3)
    M = r.normal(size=(max(a, b), min(a, b))); Q, R_ = np.linalg.qr(M); Q = Q * np.sign(np.diag(R_))
    W = Q if b >= a else Q.T
    return (W * max(math.sqrt(b / a), 1.0)).astype(np.float32)
def brake(y, c):  # c * phi_rho(y / c): linear fuer |y| <= rho*c, darueber weich (C1) zur Grenze c
    u = y / c; a = mx.abs(u); k = mx.tanh((a - LG_RHO) / (1 - LG_RHO))
    return c * mx.where(a <= LG_RHO, u, mx.sign(u) * (LG_RHO + (1 - LG_RHO) * k))
def qabs(y, q):  # q-Quantil von |y| ueber den ganzen Tensor (Schicht), ueber die k groessten Werte
    f = mx.abs(y).reshape(-1); k = max(1, int(round((1 - q) * f.size))); return mx.min(mx.topk(f, k))
class Net(nn.Module):
    def __init__(self, depth, n, seed, din=784):
        super().__init__()
        self.lag = lag(n); tat = base(n) == 'tat'; tr = recipe(n)
        r = np.random.default_rng(seed); g = 1.0 if tat else GAINS[(base(n), depth)]; dims = [din] + [WIDTH] * depth
        self.hid = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]; self.out = nn.Linear(WIDTH, 10)
        for l, (a, b) in zip(self.hid, zip(dims[:-1], dims[1:])):
            l.weight = mx.array(suo(r, b, a)) if tat else (mx.array(suo(r, b, a) * np.float32(g)) if tr else mx.array(r.normal(0, g / math.sqrt(a), size=(b, a)).astype(np.float32))); l.bias = mx.zeros((b,))
        self.out.weight = mx.array(suo(r, 10, WIDTH)) if (tat or tr) else mx.array(r.normal(0, 1 / math.sqrt(WIDTH), size=(10, WIDTH)).astype(np.float32)); self.out.bias = mx.zeros((10,))
        self.act = make_act(n, depth); self.actname = base(n); self.depth = depth
        if self.lag:
            self.lg_q0 = [mx.array(1.0) for _ in self.hid]; self.lg_m = [mx.array(1.0) for _ in self.hid]
            if self.lag == 'L3': self.lg_theta = [mx.array(0.0) for _ in self.hid]
            if isreg(self.lag): self.lg_lc = [mx.array(0.0) for _ in self.hid]; self.lg_rt = mx.array(1.0); self.lg_re = r_end(self.lag, depth)
    def lagma_init(self, xc):  # q-Quantil von |y| je Schicht bei der Init (vor der Bremse), auf den Kalibrierbildern
        h = mx.array(xc); q0 = []
        for i, l in enumerate(self.hid):
            y = self.act(l(h)); q0.append(qabs(y, LG_Q)); h = brake(y, LG_KAPPA * q0[-1])
        self.lg_q0 = [mx.stop_gradient(v) for v in q0]; self.lg_m = [mx.stop_gradient(v) for v in q0]
        if isreg(self.lag): self.lg_lc = [mx.stop_gradient(mx.log(LG_KAPPA * v)) for v in q0]; self.lg_rt = mx.array(0.0 if self.lag in RAMPED else 1.0)
        self.freeze(keys=['lg_q0', 'lg_m', 'lg_lc', 'lg_rt'], strict=False)
    def cval(self, i):
        if self.lag == 'L3': return LG_KAPPA * self.lg_q0[i] * mx.exp(LG_L3S * self.lg_theta[i])
        if isreg(self.lag): return mx.exp(self.lg_lc[i])
        if self.lag == 'L2': return LG_KAPPA * mx.clip(self.lg_m[i], 0.25 * self.lg_q0[i], 4.0 * self.lg_q0[i])
        return LG_KAPPA * self.lg_q0[i]
    def post(self, i, z):
        y = self.act(z)
        if not self.lag: return y
        if self.lag == 'L2' and self.training:
            self.lg_m[i] = mx.stop_gradient(LG_BETA * self.lg_m[i] + (1 - LG_BETA) * qabs(y, LG_Q))
        if isreg(self.lag) and self.training:
            c = mx.exp(self.lg_lc[i]); r = mx.mean((mx.abs(y) > LG_RHO * c).astype(mx.float32)); l0 = mx.log(LG_KAPPA * self.lg_q0[i])
            out = brake(y, c)
            self.lg_lc[i] = mx.stop_gradient(mx.clip(self.lg_lc[i] + LG_ETA * (r - self.lg_rt * self.lg_re) / self.lg_re, l0 + math.log(0.25), l0 + math.log(LG_UP.get(self.lag, 4.0))))
            return out
        return brake(y, self.cval(i))
    def __call__(self, x):
        for i, l in enumerate(self.hid): x = self.post(i, l(x))
        return self.out(x)

def loss_fn(m, x, y): return nn.losses.cross_entropy(m(x), y, reduction='mean')

def evaluate(fwd, x, y):
    lo = 0.0; ac = 0
    for i in range(0, len(x), 2500):
        o = fwd(mx.array(x[i:i + 2500])); yy = mx.array(y[i:i + 2500])
        lo += float(nn.losses.cross_entropy(o, yy, reduction='sum')); ac += int((mx.argmax(o, 1) == yy).sum())
    return lo / len(x), ac / len(x)

# --- Quantisierung (simuliert, fp32-Rechnung mit gerundeten Werten) ------------------
def q_int(a, bits, scale):
    qm = 2 ** (bits - 1) - 1; return mx.clip(mx.round(a / scale), -qm, qm) * scale
def q_fp8(a, scale):  # E4M3: 3 Mantissenbits, kleinster Normal-Exponent -6, Max 448
    # Exponent und Stufe exakt ueber Bits (Metal-pow/log2 sind fuer Zweierpotenzen nicht exakt)
    v = a / scale; av = mx.maximum(mx.abs(v), 2.0 ** -9).astype(mx.float32)
    e = mx.maximum(((av.view(mx.int32) >> 23) & 255) - 127, -6)
    step = ((e - 3 + 127) << 23).view(mx.float32); q = mx.round(v / step) * step; return mx.clip(q, -448.0, 448.0) * scale
def qw(W, kind):  # Gewichte: pro Ausgangskanal symmetrisch
    am = mx.maximum(mx.max(mx.abs(W), axis=1, keepdims=True), 1e-12)
    if kind == 'i8': return q_int(W, 8, am / 127)
    if kind == 'i4': return q_int(W, 4, am / 7)
    if kind == 'f8': return q_fp8(W, am / 448)
    raise ValueError(kind)
def qa(a, kind, rng):  # Aktivierungen: pro Tensor symmetrisch, rng = kalibrierter Betrag
    if kind == 'i8': return q_int(a, 8, rng / 127)
    if kind == 'i4': return q_int(a, 4, rng / 7)
    if kind == 'i3': return q_int(a, 3, rng / 3)
    if kind == 'f8': return q_fp8(a, rng / 448)
    raise ValueError(kind)
def mse_clip(a, bits):  # Clip-Wert mit minimalem Quantisierungsfehler (MSE), Gitter 1..100 % des Maximums (vektorisiert, 4 Teilbloecke)
    qm = 2 ** (bits - 1) - 1; x = mx.array(a.astype(np.float32)); amax = float(mx.max(mx.abs(x)))
    if amax == 0: return 1e-12
    E = []
    for ks in (range(1, 26), range(26, 51), range(51, 76), range(76, 101)):
        sc = mx.array([amax * k / 100 / qm for k in ks], dtype=mx.float32)[:, None]
        E.append(mx.mean((mx.clip(mx.round(x[None, :] / sc), -qm, qm) * sc - x[None, :]) ** 2, axis=1))
    E = np.array(mx.concatenate(E)); k = int(np.argmin(E)) + 1; return amax * k / 100
def qa_asym(a, bits, lohi):  # Aktivierungen: pro Tensor asymmetrisch (affin, Nullpunkt), lohi = kalibriertes [lo, hi] (wie Q3)
    lo, hi = lohi; lo = min(lo, 0.0); hi = max(hi, 0.0); n = 2 ** bits - 1; sc = max(hi - lo, 1e-12) / n; zp = round(-lo / sc)
    return (mx.clip(mx.round(a / sc) + zp, 0, n) - zp) * sc
def mse_clip_asym(a, bits):  # [lo, hi] mit minimalem MSE, Gitter 1..100 % von [min, max] (beide Seiten gemeinsam skaliert, wie Q3)
    lo0, hi0 = float(min(a.min(), 0.0)), float(max(a.max(), 0.0)); x = mx.array(a.astype(np.float32)); n = 2 ** bits - 1
    if hi0 - lo0 == 0: return (0.0, 1e-12)
    E = []
    for ks in (range(1, 26), range(26, 51), range(51, 76), range(76, 101)):
        sc = np.array([(hi0 - lo0) * k / 100 / n for k in ks], dtype=np.float32); zp = np.round(-lo0 * np.array(list(ks)) / 100 / sc).astype(np.float32)
        S = mx.array(sc)[:, None]; Z = mx.array(zp)[:, None]
        E.append(mx.mean(((mx.clip(mx.round(x[None, :] / S) + Z, 0, n) - Z) * S - x[None, :]) ** 2, axis=1))
    E = np.array(mx.concatenate(E)); k = int(np.argmin(E)) + 1; return (lo0 * k / 100, hi0 * k / 100)
def calib_asym(m, xc):  # wie Q3, aber mit m.post (damit auch die Lagma-Bremse von LD100 mitlaeuft)
    R = {k: [] for k in ['max', 'p999', 'p9999', 'mse4', 'mse3']}; h = mx.array(xc)
    for i, l in enumerate(m.hid + [m.out]):
        a = np.array(h, dtype=np.float64).ravel()
        R['max'].append((float(a.min()), float(a.max()))); R['p999'].append((float(np.quantile(a, 0.0005)), float(np.quantile(a, 0.9995))))
        R['p9999'].append((float(np.quantile(a, 0.00005)), float(np.quantile(a, 0.99995)))); R['mse4'].append(mse_clip_asym(a, 4)); R['mse3'].append(mse_clip_asym(a, 3))
        if l is not m.out: h = m.post(i, l(h))
    return R
def qforward_asym(m, bits, rngs):
    Ws = [(qw(l.weight, 'i8'), l.bias) for l in m.hid + [m.out]]
    def f(x):
        h = x
        for i, (W, b) in enumerate(Ws):
            h = qa_asym(h, bits, rngs[i]) @ W.T + b
            if i < len(Ws) - 1: h = m.post(i, h)
        return h
    return f
def calib(m, xc):
    # Eingaenge jeder Linearschicht (fp32) auf den Kalibrierbildern: max, Perzentile, MSE-optimaler Clip fuer 4 und 8 Bit
    R = {k: [] for k in ['max', 'p999', 'p9999', 'mse4', 'mse8', 'mse3']}; h = mx.array(xc)
    for i, l in enumerate(m.hid + [m.out]):
        a = np.array(h, dtype=np.float64).ravel(); aa = np.abs(a)
        R['max'].append(float(aa.max())); R['p999'].append(float(np.quantile(aa, 0.999))); R['p9999'].append(float(np.quantile(aa, 0.9999)))
        R['mse4'].append(mse_clip(a, 4)); R['mse8'].append(mse_clip(a, 8)); R['mse3'].append(mse_clip(a, 3))
        if l is not m.out: h = m.post(i, l(h))
    return R
def qforward(m, wk, ak, rngs):
    Ws = [(qw(l.weight, wk), l.bias) for l in m.hid + [m.out]]
    def f(x):
        h = x
        for i, (W, b) in enumerate(Ws):
            h = qa(h, ak, rngs[i]) @ W.T + b
            if i < len(Ws) - 1: h = m.post(i, h)
        return h
    return f

# --- Ablauf --------------------------------------------------------------------------
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'train_loss_last50', 'val_err_fp32', 'test_err_fp32'] + \
       [f'test_err_{q[0]}' for q in QUANT] + ['test_err_w8a4_best', 'w8a4_best_cal', 'cal_err_w8a4_best'] + ['test_err_w8a4a_max', 'test_err_w8a4a_best', 'w8a4a_best_cal', 'act_lo_max_median', 'sat_rate'] + ['test_err_w8a3_best', 'w8a3_best_cal', 'test_err_w8a3a_best', 'w8a3a_best_cal'] + ['act_range_max_median', 'act_range_max_last', 'act_range_p9999_median', 'act_clip_mse4_median', 'chi1', 'grad_log10_first_last', 'lg_c_median', 'lg_c_ratio_median', 'lg_intervention', 'lg_ir_trace', 'lg_c_trace', 'sec_train', 'ms_per_step', 'backend', 'timestamp']
def rid(c): return f"{c['dataset']}_d{c['depth']}_{c['act']}_s{c['seed']}" + (f"_lr{c['lr']:g}" if STAGE == 'tune' else '')
def tune_configs():
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in TUNE_SEEDS:
                for act in tune_acts(depth):
                    for lr in LR_GRID: yield dict(dataset=ds, depth=depth, act=act, lr=lr, seed=seed)
def select_lr():
    # wie 3a-3: hoechste mediane val_acc (letzter Schritt) ueber die Tuning-Seeds; divergiert -> 0.1; Gleichstand -> kleinere LR
    rows = []
    for f in sorted((TUNEDIR / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    acc = {}
    for r in rows:
        a = (1 - float(r['val_err_fp32'])) if r['status'] == 'ok' else 0.1
        acc.setdefault((r['dataset'], int(r['depth']), r['act'], float(r['lr'])), []).append(a)
    out = []
    for depth in DEPTHS:
        for ds in DATASETS:
            for act in tune_acts(depth):
                cand = [(float(np.median(acc[(ds, depth, act, lr)])), -lr, lr) for lr in LR_GRID if (ds, depth, act, lr) in acc]
                if not cand: continue
                best = max(cand); out.append(dict(dataset=ds, depth=depth, act=act, lr_sel=best[2], med_val_acc=best[0],
                                                  at_edge=best[2] in (LR_GRID[0], LR_GRID[-1]), n_seeds=len(acc[(ds, depth, act, best[2])])))
    return out
def lrsel():
    sel = {}
    for r in csv.DictReader(open(ROOT / 'lr_from_p3a3.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])
    for r in select_lr(): sel[('mnist' if r['dataset'].startswith('synth') else r['dataset'], r['depth'], r['act'])] = r['lr_sel']
    return sel
def lrsel():   # B3: MNIST (nur Kontrolle) aus A wie B1, CIFAR-10 aus dem Tuning dieser Studie (select_lr)
    sel = {}
    for r in csv.DictReader(open(ROOT / 'lr_from_p3a3.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])
    for r in csv.DictReader(open(ROOT / 'lr_from_a.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])
    for r in csv.DictReader(open(ROOT / 'lr_from_b3.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])   # B6: Referenzen aus B3
    if (TUNEDIR / 'workers').exists():
        for r in select_lr(): sel[(r['dataset'], r['depth'], r['act'])] = r['lr_sel']
    return sel
LRNAME = {'kL2B': 'kL2', 'LB': 'kL2', 'kL2Bh': 'kL2', 'LBh': 'kL2'}   # alle mit der Lernrate von kL2 aus A (wie B1, nicht neu getunt)
def configs():
    sel = lrsel()
    def L(ds, d, a):   # CIFAR: eigene getunte LR je Variante; MNIST-Kontrolle: LR von kL2 aus A (wie B1)
        if ds == 'cifar10' or (ds.startswith('synth') and ('cifar10', d, a) in sel): return sel[('cifar10', d, a)]   # B6: volle Namen (kL2B_tr eigene LR)
        return sel[('mnist', d, LRNAME.get(base(a), base(a)))]
    for ds, d, a, s in REPRO: yield dict(dataset=ds, depth=d, act=a, lr=L(ds, d, a), seed=s)
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in SEEDS:
                for act in ACTS: yield dict(dataset=ds, depth=depth, act=act, lr=L(ds, depth, act), seed=seed)

def interv(m, xc):  # mittlere Eingriffsrate |y| > rho*c ueber die Schichten, Eval-Modus, ohne Zustandsaenderung
    h = mx.array(xc); ir = []
    for i, l in enumerate(m.hid):
        y = m.act(l(h)); ir.append(float(mx.mean(mx.abs(y) > LG_RHO * m.cval(i)))); h = brake(y, m.cval(i))
    return float(np.mean(ir))
def sat_rate(m, xc):  # mittlerer Anteil gesaettigter Werte ueber die Schichten: weich |Kern| > rho*M (wie B1), hart |Kern| > M
    p = par(m.actname, m.depth); h = mx.array(xc); sr = []; lin = m.actname in ('LB', 'LBh'); thr = 1.0 if m.actname in ('LBh', 'kL2Bh') else LG_RHO
    for i, l in enumerate(m.hid):
        z = l(h); core = p['beta'] * z if lin else pc_mlx(z, **p); M = p['M'] if lin else B_M
        sr.append(float(mx.mean(mx.abs(core) > thr * M))); h = m.post(i, z)
    return float(np.mean(sr))
def cratio(m):  # Median ueber die Schichten von c / c0 (beschreibend)
    return float(np.median([float(m.cval(i)) / (LG_KAPPA * float(m.lg_q0[i])) for i in range(len(m.hid))]))
def execute(c, wid):
    t0 = time.time(); xtr, ytr, xv, yv, xte, yte = data_for(c['dataset'], base(c['act']) + ('_tr' if recipe(c['act']) else ''))
    mx.random.seed(c['seed']); m = Net(c['depth'], c['act'], c['seed'], din=xtr.shape[1]); mx.eval(m.parameters())
    m.eval()
    if m.lag: m.lagma_init(xtr[:NCAL]); mx.eval(m.parameters())
    # Gradientengefaelle bei der Init (vor dem ersten Schritt): log10(|dL/dW_1| / |dL/dW_L|), erster Trainings-Batch dieses Laufs
    r0 = np.random.default_rng(c['seed'] + 777); b0 = r0.permutation(len(xtr))[:BATCH]
    _, g0 = nn.value_and_grad(m, loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    n1 = float(mx.sqrt(mx.sum(g0['hid'][0]['weight'] ** 2))); nL = float(mx.sqrt(mx.sum(g0['hid'][-1]['weight'] ** 2)))
    glog = math.log10(n1 / nL) if n1 > 0 and nL > 0 and math.isfinite(n1 / nL) else float('nan')
    chi1 = CHI1.get((base(c['act']), c['depth']), float('nan'))
    m.train()
    opt = optim.Adam(learning_rate=c['lr'], betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    vg = nn.value_and_grad(m, loss_fn); state = [m.state, opt.state]
    def step(x, y):
        loss, g = vg(m, x, y); opt.update(m, g); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(c['seed'] + 777); perm = r.permutation(len(xtr)); pos = 0; run = []; status = 'ok'; ts = None
    trace = []; ctrace = []
    for s in range(1, STEPS + 1):
        if pos + BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + BATCH]; pos += BATCH
        if m.lag in RAMPED: m.lg_rt = mx.array(min(1.0, (s - 1) / (LG_RAMP * STEPS)), dtype=mx.float32)   # Anlauf: Ziel in Schritt s
        loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if s == 20: ts = time.time()
        if m.lag and s in (STEPS // 4, STEPS // 2, 3 * STEPS // 4): m.eval(); trace.append(interv(m, xtr[:NCAL])); ctrace.append(cratio(m)); m.train()
        if s % 43 == 0 or s == STEPS:
            lv = float(loss); run.append(lv)
            if not math.isfinite(lv): status = 'nonfinite_loss'; break
    m.eval()
    sec = time.time() - t0; msps = (time.time() - ts) / max(STEPS - 20, 1) * 1000 if ts and status == 'ok' else float('nan')
    out = dict(run_id=rid(c), **c, status=status, train_loss_last50=float(np.mean(run[-2:])) if run else float('nan'), sec_train=round(sec, 1), ms_per_step=round(msps, 3), backend=BACKEND, chi1=chi1, grad_log10_first_last=glog)
    if status == 'ok' and STAGE == 'tune':
        _, va = evaluate(lambda x: m(x), xv, yv); out['val_err_fp32'] = 1 - va
    elif status == 'ok':
        fwd = lambda x: m(x)
        _, va = evaluate(fwd, xv, yv); _, ta = evaluate(fwd, xte, yte); out['val_err_fp32'] = 1 - va; out['test_err_fp32'] = 1 - ta
        if base(c['act']) in ('kL2B', 'LB', 'kL2Bh', 'LBh'): out['sat_rate'] = sat_rate(m, xtr[:NCAL])
        R = calib(m, xtr[:NCAL]); out['act_range_max_median'] = float(np.median(R['max'][1:])); out['act_range_max_last'] = R['max'][-1]
        out['act_range_p9999_median'] = float(np.median(R['p9999'][1:])); out['act_clip_mse4_median'] = float(np.median(R['mse4'][1:]))
        if m.lag:  # Grenze am Ende, Verhaeltnis zum Startwert, Eingriffsrate |y| > rho*c auf den Kalibrierbildern
            cs_ = [float(m.cval(i)) for i in range(len(m.hid))]; out['lg_c_median'] = float(np.median(cs_))
            out['lg_c_ratio_median'] = float(np.median([cs_[i] / (LG_KAPPA * float(m.lg_q0[i])) for i in range(len(m.hid))]))
            out['lg_intervention'] = interv(m, xtr[:NCAL]); out['lg_ir_trace'] = ';'.join(f'{v:.5f}' for v in trace + [out['lg_intervention']])
            out['lg_c_trace'] = ';'.join(f'{v:.4f}' for v in ctrace + [out['lg_c_ratio_median']])
        for name, wk, ak, cal in QUANT:
            f = qforward(m, wk, ak, R[cal]); _, a = evaluate(f, xte, yte); out[f'test_err_{name}'] = 1 - a
        # w8a4_best: je Lauf die Kalibrierung mit dem kleinsten Fehler auf den 1000 Kalibrierbildern (Trainingsdaten), Gleichstand -> Reihenfolge CALS4
        ce = {}
        for cal in CALS4:
            _, a = evaluate(qforward(m, 'i8', 'i4', R[cal]), xtr[:NCAL], ytr[:NCAL]); ce[cal] = 1 - a
        bc = min(CALS4, key=lambda k: (ce[k], CALS4.index(k))); out['w8a4_best_cal'] = bc; out['cal_err_w8a4_best'] = ce[bc]
        _, a = evaluate(qforward(m, 'i8', 'i4', R[bc]), xte, yte); out['test_err_w8a4_best'] = 1 - a
        # asymmetrisch (aus Q3): gleiche Kalibrierbilder, gleiche Wahlregel
        RA = calib_asym(m, xtr[:NCAL]); out['act_lo_max_median'] = float(np.median([lh[0] for lh in RA['max'][1:]]))
        _, a = evaluate(qforward_asym(m, 4, RA['max']), xte, yte); out['test_err_w8a4a_max'] = 1 - a
        ce = {}
        for cal in CALS4:
            _, a = evaluate(qforward_asym(m, 4, RA[cal]), xtr[:NCAL], ytr[:NCAL]); ce[cal] = 1 - a
        bc = min(CALS4, key=lambda k: (ce[k], CALS4.index(k))); out['w8a4a_best_cal'] = bc
        _, a = evaluate(qforward_asym(m, 4, RA[bc]), xte, yte); out['test_err_w8a4a_best'] = 1 - a
        # 3 Bit nach dem Training (B2): gleiche Wahlregel, symmetrisch und asymmetrisch
        ce = {}
        for cal in CALS3:
            _, a = evaluate(qforward(m, 'i8', 'i3', R[cal]), xtr[:NCAL], ytr[:NCAL]); ce[cal] = 1 - a
        bc = min(CALS3, key=lambda k: (ce[k], CALS3.index(k))); out['w8a3_best_cal'] = bc
        _, a = evaluate(qforward(m, 'i8', 'i3', R[bc]), xte, yte); out['test_err_w8a3_best'] = 1 - a
        ce = {}
        for cal in CALS3:
            _, a = evaluate(qforward_asym(m, 3, RA[cal]), xtr[:NCAL], ytr[:NCAL]); ce[cal] = 1 - a
        bc = min(CALS3, key=lambda k: (ce[k], CALS3.index(k))); out['w8a3a_best_cal'] = bc
        _, a = evaluate(qforward_asym(m, 3, RA[bc]), xte, yte); out['test_err_w8a3a_best'] = 1 - a
        if c['seed'] == SAVE_SEED:
            (OUTDIR / 'weights').mkdir(parents=True, exist_ok=True); m.save_weights(str(OUTDIR / 'weights' / f"{rid(c)}.safetensors"))
    out['timestamp'] = datetime.now().isoformat(timespec='seconds')
    p = OUTDIR / 'workers' / f'w{wid}.csv'; p.parent.mkdir(parents=True, exist_ok=True); new = not p.exists()
    with p.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(out)
    return out

def finished():
    d = set()
    for f in (OUTDIR / 'workers').glob('w*.csv'):
        for r in csv.DictReader(open(f)): d.add(r['run_id'])
    return d

def selfcheck():
    out = {'backend': BACKEND, 'device': str(mx.default_device()), 'n_confirm': len(REPRO) + len(DEPTHS) * len(DATASETS) * len(SEEDS) * len(ACTS),
           'n_tune': len(list(tune_configs()))}
    X0, W0 = np.polynomial.hermite_e.hermegauss(200); W0 = W0 / W0.sum()
    for d in (100, 200):   # TAT: Steigung 1 bei 0+ mal scale, E[f^2] = 1 (wie dks: scale = sqrt(2/(1+neg^2)))
        y = np_act('tat', X0, d); out[f'tat_d{d}'] = dict(Ef2=round(float(np.sum(W0 * y * y)), 6), mlx_vs_numpy=float(np.abs(np.array(make_act('tat', d)(mx.array(X0.astype(np.float32)))) - y).max()))
    y = mx.array(np.linspace(-10, 10, 2001).astype(np.float32)); bk = np.array(brake(y, mx.array(4.0))); yy = np.linspace(-10, 10, 2001)
    out['brake_core_identity'] = bool(np.allclose(bk[np.abs(yy) <= 3.2], yy[np.abs(yy) <= 3.2], atol=1e-6)); out['brake_bounded'] = float(np.abs(bk).max())
    out['brake_monotone'] = bool(np.all(np.diff(bk) >= -1e-6))   # fp32-Rundung an der Saettigung
    t = mx.array(np.random.default_rng(0).normal(size=(128, 256)).astype(np.float32)); out['qabs_vs_numpy'] = [float(qabs(t, LG_Q)), float(np.quantile(np.abs(np.array(t)), LG_Q))]
    z = mx.array(np.linspace(-8, 8, 20001).astype(np.float32)); X, W = np.polynomial.hermite_e.hermegauss(200); W = W / W.sum()
    for n in ANAMES:
        for d in [100, 200]:
            y = np_act(n, X, d); k = f'{n}_d{d}'
            out[k] = dict(gain=round(GAINS[(n, d)], 6), chi1=round(CHI1[(n, d)], 5), mean=round(float(np.sum(W * y)), 6), var=round(float(np.sum(W * y * y)), 6),
                          mlx_vs_numpy=float(np.abs(np.array(make_act(n, d)(z)) - np_act(n, np.array(z, dtype=np.float64), d)).max()),
                          range=[float(make_act(n, d)(mx.array([-50.0]))[0]), float(make_act(n, d)(mx.array([50.0]))[0])])
    from precap.mlx import precap_ref_mlx
    for n in []: out[n + '_vs_precap_ref'] = float(mx.max(mx.abs(make_act(n, 100)(z) - precap_ref_mlx(z, **PRESETS[n]))))
    a = mx.array(np.linspace(-3, 3, 13).astype(np.float32))
    out['q_int8_example'] = np.array(q_int(a, 8, 3 / 127)).round(4).tolist(); out['q_fp8_448'] = float(q_fp8(mx.array([448.0, 500.0, 0.0]), 1.0)[1])
    out['supf2'] = {str(d): round(SUPF2[d], 5) for d in SUPF2}; out['r_end'] = {f'{g}_d{d}': round(r_end(g, d), 6) for g in ('LA10', 'LT', 'LTA') for d in [100, 200]}
    out['q_fp8_grid_ok'] = bool(np.allclose(np.array(q_fp8(mx.array([1.0, 1.0625, 1.125, 0.015625 * 1.1]), 1.0)), [1.0, 1.0, 1.125, 0.017578125]))
    return out

def main():
    global STEPS, OUTDIR, SEEDS, DATASETS, DEPTHS, BACKEND, STAGE, TUNE_SEEDS, LR_GRID, TUNEDIR
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--probe', action='store_true')
    p.add_argument('--backend', default='ref', choices=['ref']); p.add_argument('--synth', action='store_true'); p.add_argument('--limit', type=int, default=0); a = p.parse_args()
    BACKEND = a.backend; mx.set_cache_limit(CACHE_LIMIT)
    if a.synth: DATASETS = ['synth']
    if a.smoke: STEPS = 40; OUTDIR = ROOT / 'smoke'
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    global REPRO
    if a.smoke: DEPTHS = [100]; TUNE_SEEDS = [42000]; LR_GRID = [2e-5, 5e-5]; SEEDS = [42144]
    base_ = OUTDIR
    # Stufe 1: Tuning (Raster und Regel wie 3a-3/A3), beide Worker, danach LR-Wahl
    STAGE = 'tune'; OUTDIR = base_ / 'tune'; TUNEDIR = OUTDIR; skip = finished(); T = list(tune_configs())
    for i, c in enumerate(T):
        if i % a.workers != a.worker or rid(c) in skip: continue
        o = execute(c, a.worker); print('tune', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)
    while len(finished()) < len(T): time.sleep(20)
    sel = select_lr()
    if a.worker == 0:
        with open(base_ / 'lr_tuned.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sel[0].keys())); w.writeheader(); w.writerows(sel)
    print('LR-Wahl', json.dumps(sel), flush=True)
    # Stufe 2: Bestaetigung auf neuen Seeds
    STAGE = 'confirm'; OUTDIR = base_; skip = finished()
    for i, c in enumerate(configs()):
        if i % a.workers != a.worker or rid(c) in skip: continue
        o = execute(c, a.worker); print('done', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
