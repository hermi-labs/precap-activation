# V2e (privat, mit Claude, praeregistriert in V2e_Praeregistrierung.md): Woher kommt der kleinere 4-Bit-Preis von BPreCap gegenueber der gleich
# kalibrierten tanh (V2d)? Messstudie an nachtrainierten Netzen: je Form 3 Bestaetigungs-Seeds (42350-42352) mit dem eingefrorenen Code von
# V2a2/V2c/V2d bitgleich nachgerechnet (Kontrolle gegen die historische CSV-Zeile, alle Messspalten als Text), Gewichte am fruehen Stopp
# gespeichert, dann an 1000 Testbildern gemessen: (a) Rundungsrauschen je Schicht r_i, (b) Weitergabe d_i / r_i bis zu den Logits, wenn nur
# Eingang i auf 4 Bit gerundet wird, (c) dasselbe mit Gauss-Rauschen gleicher Norm, (d) alle Eingaenge gerundet (Verlauf ueber die Tiefe).
# Nur Aktivierungen (Gewichte fp32), Kalibrierung = die im Lauf gewaehlte w8a4a_best-Kalibrierung. PreCap, BPreCap, Huelle, Rezept: Ideen Aarons.
# Aufruf: --selfcheck | --smoke | --worker 0 --workers 1
import csv, json, math, sys, time, zlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent; S = ROOT.parent
for d in ('v2a2', 'v2c', 'v2d'): sys.path.insert(0, str(S / d))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
import v2a2_run as R
import v2d_run as V2D          # setzt R.make_act um 'ctanh' (eingefroren)
import v2c_run as V2C          # setzt R.make_act um 'tanh' (eingefroren), Kette: tanh -> ctanh -> Original
NTEST, CHUNK = 1000, 250
SEEDS = (42350, 42351, 42352)
CORE = ('bprecap', 'ctanh', 'selu'); EXTRA = ('tanh', 'tat'); FORMS = CORE + EXTRA
HIST = {'bprecap': 'v2a2', 'selu': 'v2a2', 'tat': 'v2a2', 'tanh': 'v2c', 'ctanh': 'v2d'}
SKIP = ('sec', 'timestamp')
def hist_rows():
    out = {}
    for a, d in HIST.items():
        for r in csv.DictReader(open(S / d / 'workers' / 'w0.csv')):
            if r['act'] == a and r['stage'] == 'confirm' and int(r['seed']) in SEEDS: out[(a, int(r['seed']))] = r
    return out
def cfgs():   # erst die drei Kernformen je Seed, dann tanh und TAT
    H = hist_rows(); out = []
    for forms in (CORE, EXTRA):
        for s in SEEDS:
            for a in forms:
                h = H[(a, s)]; out.append(dict(stage='confirm', act=a, lr=float(h['lr']), wd=float(h['wd']), seed=s, _hist=h['run_id']))
    return out
def run_keep(c):
    # wie R.run_one (Bestaetigung), gibt zusaetzlich das Netz am fruehen Stopp zurueck
    steps = c.get('steps', R.STEPS)
    m, ok, tl, kurve, best = R.train(c['act'], c['lr'], c['wd'], c.get('sched', 'cos'), c['seed'], steps)
    o = dict(run_id=R.rid(c), stage=c['stage'], act=c['act'], lr=c['lr'], wd=c['wd'], sched=c.get('sched', 'cos'), seed=c['seed'], steps=steps,
             status='ok' if ok else 'fail', train_loss_last50=tl, val_kurve=';'.join(f'{s}:{e:.4f}' for s, e in kurve))
    if not ok: return o, None
    o['val_err_end'] = kurve[-1][1]; o['val_err_es'] = best[1]; o['es_step'] = best[0]
    o.update(R.ptq(m, c['seed'], 'end_')); m.update(R.tree_unflatten(list(best[2].items()))); mx.eval(m.parameters()); o.update(R.ptq(m, c['seed'], 'es_'))
    if c['act'] == 'tat': mc = R.cle(m); o.update(R.ptq(mc, c['seed'], 'es_cle_')); o['es_cle_fp32_delta'] = o['es_cle_test_err_fp32'] - o['es_test_err_fp32']
    return o, m
def compare(o, ref):
    diff = [(k, str(o.get(k, '')), ref.get(k, '')) for k in R.COLS if k not in SKIP and str(o.get(k, '')) != ref.get(k, '')]
    return diff
def key(*parts): return zlib.crc32('|'.join(map(str, parts)).encode()) & 0x7fffffff   # eigener Generator je Messstelle
def gauss_like(shape, norm, k):
    g = mx.random.normal(shape, key=mx.random.key(k)); n = mx.sqrt(mx.sum(g * g)); return g * (norm / mx.maximum(n, 1e-30))
def measure(m, seed, cal, ntest=NTEST, chunk=CHUNK):
    # Alle Groessen ueber alle ntest Bilder aufsummiert (Frobenius), Eingang i = 0..L (0 = Bild, L = Linear-Eingang nach dem Mittel).
    # Sauberer Pfad, "alle gerundet" (Q) und "alle mit Gauss-Rauschen gleicher Norm" (G) laufen Schicht fuer Schicht nebeneinander;
    # die Gauss-Norm je Eingang ist die Norm des Rundungsfehlers im sauberen Pfad an derselben Stelle.
    xtr, ytr, _, _, xte, yte = R.load(); Rc = R.calib(m, xtr[:R.NCAL], seed); lohi = Rc[cal]; nl = len(m.convs) + 1
    Wc = [(c.weight, c.bias) for c in m.convs]; Wo, bo = m.out.weight, m.out.bias
    Q = lambda h, i: R.qa_asym(h, 4, lohi[i])
    def nrm(t): return float(mx.sqrt(mx.sum(t * t)))
    def fwd(x, inj):
        h = x
        for i, (W, b) in enumerate(Wc):
            h = m.act(mx.conv2d(inj(i, h), W, padding=1) + b); mx.eval(h)
        h = inj(len(Wc), mx.mean(h, axis=(1, 2))); return h @ Wo.T + bo
    A = {k: np.zeros(nl) for k in ('nq2', 'nh2', 'dQ2', 'dG2')}; chi = np.zeros(nl - 1); cnt = np.zeros(nl - 1)
    T = {k: np.zeros(nl - 1) for k in ('eQ2', 'eG2', 'h2')}; z2 = allQ2 = allG2 = 0.0; flipQ = flipG = eF = eQ = eW = 0; marg = []
    fw8 = R.qforward(m, f'w8a4a_{cal}', Rc); Bs = [0.0] * (nl - 1); Bn = np.zeros(nl - 1)   # Anteil des Rundungsfehlers, der je Kanal konstant ist
    for ci, j in enumerate(range(0, ntest, chunk)):
        x = mx.array(xte[j:j + chunk]); y = np.array(yte[j:j + chunk]); h = hq = hg = x; nq = []
        for i, (W, b) in enumerate(Wc):
            dq = Q(h, i) - h; nq.append(nrm(dq)); A['nq2'][i] += nq[-1] ** 2; A['nh2'][i] += float(mx.sum(h * h))
            Bs[i] = Bs[i] + np.array(mx.sum(dq, axis=(0, 1, 2)), dtype=np.float64); Bn[i] += dq.size // dq.shape[-1]
            u = mx.conv2d(h, W, padding=1) + b; gr = mx.grad(lambda v: mx.sum(m.act(v)))(u); chi[i] += float(mx.sum(gr * gr)); cnt[i] += gr.size
            hgi = hg + gauss_like(hg.shape, nq[i], key(seed, cal, 'all', i, ci))
            h = m.act(u); hq = m.act(mx.conv2d(Q(hq, i), W, padding=1) + b); hg = m.act(mx.conv2d(hgi, W, padding=1) + b); mx.eval(h, hq, hg)
            T['eQ2'][i] += float(mx.sum((hq - h) ** 2)); T['eG2'][i] += float(mx.sum((hg - h) ** 2)); T['h2'][i] += float(mx.sum(h * h))
        p, pq, pg = (mx.mean(t, axis=(1, 2)) for t in (h, hq, hg)); i = nl - 1
        nq.append(nrm(Q(p, i) - p)); A['nq2'][i] += nq[-1] ** 2; A['nh2'][i] += float(mx.sum(p * p))
        z = p @ Wo.T + bo; zq = Q(pq, i) @ Wo.T + bo; zg = (pg + gauss_like(pg.shape, nq[i], key(seed, cal, 'all', i, ci))) @ Wo.T + bo; mx.eval(z, zq, zg)
        zn, zqn, zgn = np.array(z), np.array(zq), np.array(zg); pr = zn.argmax(1)
        z2 += float((zn.astype(np.float64) ** 2).sum()); allQ2 += float(((zqn - zn).astype(np.float64) ** 2).sum()); allG2 += float(((zgn - zn).astype(np.float64) ** 2).sum())
        eF += int((pr != y).sum()); eQ += int((zqn.argmax(1) != y).sum()); flipQ += int((zqn.argmax(1) != pr).sum()); flipG += int((zgn.argmax(1) != pr).sum())
        zs = np.sort(zn, 1); marg += list((zs[:, -1] - zs[:, -2]) / np.maximum(np.linalg.norm(zn, axis=1), 1e-12))
        eW += int((np.array(fw8(x)).argmax(1) != y).sum())
        for i in range(nl):   # nur Eingang i gestoert
            for kind in ('Q', 'G'):
                if kind == 'Q': inj = lambda ii, t, i=i: Q(t, ii) if ii == i else t
                else: inj = lambda ii, t, i=i: t + gauss_like(t.shape, nq[i], key(seed, cal, 'one', i, ci)) if ii == i else t
                dz = np.array(fwd(x, inj)).astype(np.float64) - zn; A['d' + kind + '2'][i] += float((dz ** 2).sum())
        mx.clear_cache()
    zr = math.sqrt(z2); r = np.sqrt(A['nq2'] / A['nh2']); dQ = np.sqrt(A['dQ2']) / zr; dG = np.sqrt(A['dG2']) / zr
    dims = np.array([3] + [32] * (nl - 2)); bias = [float((Bs[i] ** 2).sum() / Bn[i] / A['nq2'][i]) for i in range(nl - 1)]
    return dict(cal=cal, ntest=ntest, r=r.tolist(), dQ=dQ.tolist(), dG=dG.tolist(), gQ=(dQ / r).tolist(), gG=(dG / r).tolist(),
                chi_emp=(chi / cnt).tolist(), bias_anteil=bias, rms_in=np.sqrt(A['nh2'][:nl - 1] / (ntest * 32 * 32 * dims)).tolist(),
                epsQ=np.sqrt(T['eQ2'] / T['h2']).tolist(), epsG=np.sqrt(T['eG2'] / T['h2']).tolist(),
                D_allQ=math.sqrt(allQ2) / zr, D_allG=math.sqrt(allG2) / zr, flipQ=flipQ / ntest, flipG=flipG / ntest,
                err_fp32=eF / ntest, err_a4=eQ / ntest, err_w8a4=eW / ntest, margin_median=float(np.median(marg)))
INNER = slice(1, -1)   # innere Faltungseingaenge i = 1..L-1, wie ptq
def gm(v): v = np.asarray(v, dtype=np.float64)[INNER]; return float(np.exp(np.mean(np.log(np.maximum(v, 1e-30)))))
def summary(M):
    lin = math.sqrt(sum(d * d for d in M['dQ'])); linG = math.sqrt(sum(d * d for d in M['dG']))
    return dict(rbar=gm(M['r']), gbarQ=gm(M['gQ']), gbarG=gm(M['gG']), D_allQ=M['D_allQ'], D_allG=M['D_allG'], superpos_Q=M['D_allQ'] / lin, superpos_G=M['D_allG'] / linG,
                flipQ=M['flipQ'], err_fp32_1000=M['err_fp32'], err_a4_1000=M['err_a4'], err_w8a4_1000=M['err_w8a4'], chi_emp_gm=float(np.exp(np.mean(np.log(np.asarray(M['chi_emp']))))),
                eps_end_Q=M['epsQ'][-1], margin_median=M['margin_median'], bias_anteil_median=float(np.median(M['bias_anteil'][1:])))
COLS = ['run_id', 'act', 'seed', 'hist_run_id', 'status', 'kontrolle_n_diff', 'kontrolle_diff', 'es_test_err_fp32', 'es_test_err_w8a4a_best', 'es_w8a4a_best_cal',
        'rbar', 'gbarQ', 'gbarG', 'D_allQ', 'D_allG', 'superpos_Q', 'superpos_G', 'flipQ', 'err_fp32_1000', 'err_a4_1000', 'err_w8a4_1000', 'chi_emp_gm', 'eps_end_Q',
        'margin_median', 'bias_anteil_median', 'sec_train', 'sec_mess', 'timestamp']
def write(p, o):
    new = not p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(o)
def work(outdir, cl, ntest=NTEST, check=True):
    wd_ = outdir / 'workers'; H = {r['run_id']: r for r in [h for h in hist_rows().values()]}
    done = {r['run_id'] for r in csv.DictReader(open(wd_ / 'w0.csv'))} if (wd_ / 'w0.csv').exists() else set()
    for c in cl:
        rid = R.rid(c)
        if rid in done: continue
        t0 = time.time(); o, m = run_keep(c); t1 = time.time()
        row = dict(run_id=rid, act=c['act'], seed=c['seed'], hist_run_id=c['_hist'], status=o['status'], timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'), sec_train=round(t1 - t0, 1))
        if check:
            diff = compare(o, H[c['_hist']]); row['kontrolle_n_diff'] = len(diff); row['kontrolle_diff'] = json.dumps(diff[:3])
        for k in ('es_test_err_fp32', 'es_test_err_w8a4a_best', 'es_w8a4a_best_cal'): row[k] = o.get(k, '')
        if m is not None:
            (outdir / 'gewichte').mkdir(parents=True, exist_ok=True); mx.savez(str(outdir / 'gewichte' / f'{rid}.npz'), **dict(tree_flatten(m.parameters())))
            M = measure(m, c['seed'], o['es_w8a4a_best_cal'], ntest); (outdir / 'mess').mkdir(parents=True, exist_ok=True); json.dump(M, open(outdir / 'mess' / f'{rid}.json', 'w'))
            row.update(summary(M)); row['sec_mess'] = round(time.time() - t1, 1)
        write(wd_ / 'w0.csv', row)
        print(time.strftime('%H:%M'), rid, row['status'], 'diff', row.get('kontrolle_n_diff'), 'rbar %.4f gQ %.3f gG %.3f D %.4f' % (row.get('rbar', -1), row.get('gbarQ', -1), row.get('gbarG', -1), row.get('D_allQ', -1)),
              row['sec_train'], row.get('sec_mess'), flush=True)
    print('Worker fertig', time.strftime('%H:%M'), flush=True)
def selfcheck():
    out = dict(n_runs=len(cfgs()), forms=list(FORMS), seeds=list(SEEDS), ntest=NTEST)
    H = hist_rows(); out['hist_vollstaendig'] = len(H) == 15 and all(r['status'] == 'ok' for r in H.values())
    out['act_ctanh_wie_v2d'] = R.make_act('ctanh') is V2D.ctanh; out['act_tanh_wie_v2c'] = R.make_act('tanh') is mx.tanh
    X, Wq = np.polynomial.hermite_e.hermegauss(150); Wq = Wq / Wq.sum(); x = mx.array(X.astype(np.float32))
    for a in FORMS:
        y = np.array(R.make_act(a)(x), dtype=np.float64); out[f'Ef2_{a}'] = round(float(np.sum(Wq * y * y)), 3)
    out['run_ids_wie_hist'] = all(R.rid(c) == c['_hist'] for c in cfgs())
    # Messmechanik an einem untrainierten Netz (BPreCap, L = 50), 50 Bilder
    m = R.CNN(R.make_act('bprecap'), 7); mx.eval(m.parameters()); M = measure(m, 7, 'max', ntest=50, chunk=50)
    out['mess_r_positiv'] = all(v > 0 for v in M['r']); out['mess_d_endlich'] = all(np.isfinite(M['dQ'] + M['dG']))
    # Kein Rauschen -> keine Abweichung; Gauss-Norm stimmt; doppelte Norm -> doppelte Wirkung bei kleinem Rauschen (linearer Bereich)
    g = gauss_like((4, 8, 8, 32), 3.0, key(1, 2)); out['gauss_norm_ok'] = abs(float(mx.sqrt(mx.sum(g * g))) - 3.0) < 1e-4
    out['gauss_reproduzierbar'] = bool(mx.all(g == gauss_like((4, 8, 8, 32), 3.0, key(1, 2))))
    xte = R.load()[4]; xb = mx.array(xte[:50])
    def f(eps):
        h = xb
        for i, c in enumerate(m.convs):
            if i == 10: h = h + gauss_like(h.shape, eps, key(9, 9))
            h = m.act(c(h))
        return mx.mean(h, axis=(1, 2)) @ m.out.weight.T + m.out.bias
    z0 = f(0.0); d1 = float(mx.sqrt(mx.sum((f(1e-2) - z0) ** 2))); d2 = float(mx.sqrt(mx.sum((f(2e-2) - z0) ** 2)))
    out['null_rauschen_null'] = float(mx.max(mx.abs(f(0.0) - z0))) == 0.0; out['linear_ratio'] = round(d2 / d1, 3); out['linear_ok'] = abs(d2 / d1 - 2) < 0.1
    print(json.dumps(out, indent=1)); return out
def main():
    a = sys.argv[1:]; print('Geraet:', mx.default_device(), flush=True)
    if hasattr(mx, 'set_cache_limit'): mx.set_cache_limit(2 * 1024 ** 3)
    if '--selfcheck' in a: selfcheck(); return
    if '--smoke' in a:
        cl = [dict(c, steps=20) for c in cfgs() if c['seed'] == SEEDS[0] and c['act'] in ('bprecap', 'ctanh')]
        work(ROOT / 'smoke', cl, ntest=250, check=False); return
    work(ROOT, cfgs())
if __name__ == '__main__': main()
