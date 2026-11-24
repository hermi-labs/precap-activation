# V2m (privat, mit Claude, praeregistriert in V2m_Praeregistrierung.md): Eingriff an der Form. BPreCap mit frueher Saettigung (Knie der Huelle
# rho = 0,4 statt 0,8, Konstanten neu geloest: kL2B_d50_rho04.json), sonst alles wie V2a2/V2f: L = 50, AdamW wie V2a2, LR 3e-5, WD 0, 3440 Schritte,
# fruehes Stoppen, PTQ wie V2a2 (v2e_run.run_keep, eingefroren), Gewichte am fruehen Stopp, dann die V2k-Messung (v2k_run.measure, eingefroren)
# bei mse4 an 1000 Testbildern. Seeds 42353-42361 wie V2f/V2k (Paare mit BPreCap rho 0,8 und ctanh aus V2k).
# PreCap, BPreCap, Huelle, Kalibrierungsrezept: Ideen Aarons. Aufruf: --selfcheck | --smoke | --worker 0 --workers 1
import csv, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; S = ROOT.parent
sys.path.insert(0, str(S / 'v2e')); sys.path.insert(0, str(S / 'v2k'))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
import v2e_run as E   # eingefroren; importiert v2a2_run, v2d_run, v2c_run unveraendert
import v2k_run as K   # eingefroren; Messung wie V2k
R = E.R
SEEDS = tuple(range(42353, 42362)); ACT = 'bprecapr04'; LR, WD = 3e-5, 0.0; CAL = 'mse4'
KV = json.load(open(ROOT / 'kL2B_d50_rho04.json')); RHO_V = 0.4
def bprecap_rho(lam, al, b, rho):
    # Rechenweg Zeichen fuer Zeichen wie v2a2_run.make_act (bprecap), nur rho und (lam, alpha, b) als Parameter
    ec = math.exp(R.C_)
    def f(x):
        w = R.S_ * mx.tanh(b * x / R.S_); ce = mx.where(w <= R.C_, mx.exp(mx.minimum(w, R.C_)) - 1, ec - 1 + ec * (w - R.C_)); y = lam * mx.where(x > 0, ce, al * ce)
        u = y / R.M; a = mx.abs(u); return R.M * mx.where(a <= rho, u, mx.sign(u) * (rho + (1 - rho) * mx.tanh((a - rho) / (1 - rho))))
    return f
ACT_V = bprecap_rho(KV['lam'], KV['alpha'], KV['b'], RHO_V)
ACT_K = bprecap_rho(R.KC['lam'], R.KC['alpha'], R.KC['b'], R.RHO)   # nur fuer die Kontrolle: muss bitgleich zu R.make_act('bprecap') sein
_orig_act = R.make_act
def make_act(name): return ACT_V if name == ACT else (ACT_K if name == 'bprecapk08' else _orig_act(name))
R.make_act = make_act
def cfgs(): return [dict(stage='confirm', act=ACT, lr=LR, wd=WD, seed=s) for s in SEEDS]
COLS = ['run_id', 'act', 'seed', 'cal', 'status', 'es_step', 'val_err_es', 'es_test_err_fp32', 'es_test_err_w8a8', 'es_test_err_w8a4a_mse4', 'es_test_err_w8a4a_best',
        'es_w8a4a_best_cal'] + [c for c in K.COLS if c not in ('run_id', 'act', 'seed', 'cal', 'status', 'v2f_abw_dQ', 'v2f_abw_D', 'sec', 'timestamp')] + ['sec_train', 'sec_mess', 'timestamp']
def write(p, o):
    new = not p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(o)
def work(outdir, cl, ntest=K.NTEST, chunk=K.CHUNK):
    p = outdir / 'workers' / 'w0.csv'; done = {r['run_id'] for r in csv.DictReader(open(p))} if p.exists() else set()
    for c in cl:
        rid = R.rid(c)
        if rid in done: continue
        t0 = time.time(); row = dict(run_id=rid, act=c['act'], seed=c['seed'], cal=CAL)
        try:
            o, m = E.run_keep(c); t1 = time.time(); row['sec_train'] = round(t1 - t0, 1); row['status'] = o['status']
            for k in ('es_step', 'val_err_es', 'es_test_err_fp32', 'es_test_err_w8a8', 'es_test_err_w8a4a_mse4', 'es_test_err_w8a4a_best', 'es_w8a4a_best_cal'): row[k] = o.get(k, '')
            if m is not None:
                (outdir / 'gewichte').mkdir(parents=True, exist_ok=True); mx.savez(str(outdir / 'gewichte' / f'{rid}.npz'), **dict(tree_flatten(m.parameters())))
                M = K.measure(m, c['seed'], CAL, ntest, chunk); DEL = M.pop('_DEL'); (outdir / 'mess').mkdir(parents=True, exist_ok=True)
                json.dump(M, open(outdir / 'mess' / f'{rid}_{CAL}.json', 'w')); np.savez_compressed(outdir / 'mess' / f'{rid}_{CAL}_delta.npz', delta=DEL)
                row.update(K.summary(M)); row['sec_mess'] = round(time.time() - t1, 1)
                if not all(np.isfinite(float(row[k])) for k in ('D_allQ', 'D_allC', 'D_allR', 'kappa', 'nu', 'gbarQ', 'gbarC', 'gbarR')): row['status'] = 'nicht_endlich'
        except Exception as ex:
            row['status'] = 'fail'; print('FEHLER', rid, repr(ex), flush=True)
        row['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S'); write(p, row)
        print(time.strftime('%H:%M'), rid, row['status'], 'fp32 %s a4best %s | DC %.4f gC %.3f DQ %.4f' % (row.get('es_test_err_fp32'), row.get('es_test_err_w8a4a_best'),
              *[float(row.get(k, float('nan')) or float('nan')) for k in ('D_allC', 'gbarC', 'D_allQ')]), row.get('sec_train'), row.get('sec_mess'), flush=True)
        mx.clear_cache()
    print('Worker fertig', time.strftime('%H:%M'), flush=True)
def selfcheck():
    out = dict(n_runs=len(cfgs()), seeds=list(SEEDS), act=ACT, lr=LR, wd=WD, cal=CAL, ntest=K.NTEST, rho=RHO_V, konst=KV)
    out['seeds_wie_v2k'] = SEEDS == K.SEEDS
    X, Wq = np.polynomial.hermite_e.hermegauss(150); Wq = Wq / Wq.sum(); x = mx.array(X.astype(np.float32))
    for a in ('bprecap', ACT):
        y = np.array(R.make_act(a)(x), dtype=np.float64); dy = np.array(mx.vmap(mx.grad(lambda t: R.make_act(a)(t)))(x), dtype=np.float64)
        out[f'Ef_{a}'] = round(float(np.sum(Wq * y)), 4); out[f'Ef2_{a}'] = round(float(np.sum(Wq * y * y)), 4); out[f'chi1_{a}'] = round(float(np.sum(Wq * dy * dy)), 4)
    g = mx.array(np.linspace(-40, 40, 400001).astype(np.float32)); out['bereich_v'] = [float(mx.min(ACT_V(g))), float(mx.max(ACT_V(g)))]
    out['kontrolle_form_bitgleich'] = bool(mx.all(ACT_K(g) == _orig_act('bprecap')(g)))
    out['bprecap_unveraendert'] = R.make_act('bprecap') is not ACT_V and R.make_act('ctanh') is E.V2D.ctanh
    out['variante_anders'] = float(mx.max(mx.abs(ACT_V(g) - _orig_act('bprecap')(g))))
    out['anteil_jenseits_knie_init'] = round(float(np.sum(Wq * (np.abs(np.array(bprecap_rho(KV['lam'], KV['alpha'], KV['b'], 1e9)(x))) / R.M > RHO_V))), 4)
    # Kontrolle Training: 120 Schritte mit ACT_K und mit R.make_act('bprecap'), Seed 42353 -> alle Gewichte und die Val-Kurve bitgleich
    t0 = time.time(); ra = R.train('bprecap', LR, WD, 'cos', SEEDS[0], 120); rb = R.train('bprecapk08', LR, WD, 'cos', SEEDS[0], 120)
    pa, pb = dict(tree_flatten(ra[0].parameters())), dict(tree_flatten(rb[0].parameters()))
    out['kontrolle_training_bitgleich'] = bool(pa.keys() == pb.keys() and all(bool(mx.all(pa[k] == pb[k])) for k in pa) and ra[3] == rb[3] and ra[1] and rb[1])
    out['kontrolle_kurve'] = [list(map(float, t)) for t in ra[3]]; out['kontrolle_sec'] = round(time.time() - t0, 1)
    # Messmechanik an einem untrainierten Netz der Variante, 50 Bilder: endlich, Zerlegung, kappa
    m = R.CNN(ACT_V, 7); mx.eval(m.parameters()); M = K.measure(m, 7, CAL, ntest=50, chunk=50); M.pop('_DEL'); Sm = K.summary(M)
    out['mess_endlich'] = bool(all(np.isfinite(float(v)) for v in Sm.values())); out['mess_summary'] = {k: round(float(v), 5) for k, v in Sm.items()}
    out['alle_ok'] = bool(out['seeds_wie_v2k'] and out['kontrolle_form_bitgleich'] and out['kontrolle_training_bitgleich'] and out['bprecap_unveraendert'] and out['mess_endlich']
                          and abs(out[f'Ef_{ACT}']) < 2e-3 and abs(out[f'Ef2_{ACT}'] - 1) < 2e-3 and abs(out[f'chi1_{ACT}'] - 1.04) < 2e-3 and out['variante_anders'] > 1e-2)
    print(json.dumps(out, indent=1)); return out
def main():
    a = sys.argv[1:]; print('Geraet:', mx.default_device(), flush=True)
    if hasattr(mx, 'set_cache_limit'): mx.set_cache_limit(2 * 1024 ** 3)
    if '--selfcheck' in a: selfcheck(); return
    if '--smoke' in a: work(ROOT / 'smoke', [dict(c, steps=20) for c in cfgs()[:1]], ntest=50, chunk=50); return
    work(ROOT, cfgs())
if __name__ == '__main__': main()
