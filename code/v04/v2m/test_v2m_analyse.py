# V2m Auswertungstest an kuenstlichen Daten (mit Claude). Referenzen (V2k, V2a2) werden kuenstlich erzeugt; ein Fall nutzt die echten Referenzdateien.
# Faelle: Effekt nach oben -> E1, E2 bestaetigt; nach unten -> widerlegt; Null -> selten Fehlalarm; 1 Ausfall -> 8 Paare, S2/S3 mit 0,9;
# 3 Ausfaelle -> unklar (zu wenige Paare); fp32-Verschiebung -> S3; Lage t; echte Referenzdateien lesbar.
import csv, json, math, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; SEEDS = list(range(42353, 42362)); ok = True; REAL = H.parent
VC = ['run_id', 'act', 'seed', 'cal', 'status', 'es_step', 'val_err_es', 'es_test_err_fp32', 'es_test_err_w8a8', 'es_test_err_w8a4a_mse4', 'es_test_err_w8a4a_best', 'es_w8a4a_best_cal',
      'D_allQ', 'D_allC', 'D_allR', 'rbarQ', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR', 'n_ohne_clip', 'frac_clip_median', 'kappa', 'nu', 'superpos_Q', 'cos_nachbar_median',
      'cos_mittel_innen', 'qc_abw', 'sec_train', 'sec_mess', 'timestamp']
KC = ['run_id', 'act', 'seed', 'cal', 'status', 'D_allQ', 'D_allC', 'D_allR', 'rbarQ', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR', 'n_ohne_clip', 'frac_clip_median', 'kappa', 'nu',
      'superpos_Q', 'cos_nachbar_median', 'cos_mittel_innen', 'qc_abw', 'v2f_abw_dQ', 'v2f_abw_D', 'sec', 'timestamp']
AC = ['run_id', 'stage', 'act', 'seed', 'status', 'es_test_err_fp32', 'es_test_err_w8a4a_best', 'es_test_err_w8a4a_mse4']
def base(r, scale=1.0):
    e = lambda: math.exp(r.normal(0, 0.05))
    return dict(D_allQ=0.3 * e(), D_allC=0.2 * e() * scale, D_allR=0.2 * e(), rbarQ=0.08 * e(), rbarC=0.05 * e(), rbarR=0.06 * e(), gbarQ=0.2 * e(), gbarC=0.2 * e() * scale,
                gbarR=0.2 * e(), n_ohne_clip=0, frac_clip_median=0.01 * e(), kappa=1.0 + r.normal(0, 0.05), nu=1.0 + r.normal(0, 0.05), superpos_Q=1.05 * e(), cos_nachbar_median=0.3,
                cos_mittel_innen=0.05, qc_abw=0.0)
def make_refs(S, r):
    (S / 'v2k' / 'workers').mkdir(parents=True); (S / 'v2a2' / 'workers').mkdir(parents=True)
    with open(S / 'v2k' / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=KC); w.writeheader()
        for s in SEEDS:
            for a, sc in (('bprecap', 1.0), ('ctanh', 1.3)):
                for cal in ('mse4', 'max'): w.writerow(dict(base(r, sc), run_id=f'confirm_{a}_lr3e-05_wd0_s{s}', act=a, seed=s, cal=cal, status='ok', v2f_abw_dQ=0, v2f_abw_D=0, sec=1, timestamp='x'))
    with open(S / 'v2a2' / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=AC); w.writeheader()
        for s in SEEDS:
            fp = 0.30 + r.normal(0, 0.005); w.writerow(dict(run_id=f'confirm_bprecap_lr3e-05_wd0_s{s}', stage='confirm', act='bprecap', seed=s, status='ok', es_test_err_fp32=fp,
                                                         es_test_err_w8a4a_best=fp + 0.05 + r.normal(0, 0.005), es_test_err_w8a4a_mse4=fp + 0.06))
            w.writerow(dict(run_id=f'confirm_selu_lr3e-05_wd0_s{s}', stage='confirm', act='selu', seed=s, status='ok', es_test_err_fp32=0.5, es_test_err_w8a4a_best=0.6, es_test_err_w8a4a_mse4=0.6))
def make_v(d, r, scale=1.0, fail=(), dfp=0.0, real_refs=None):
    (d / 'workers').mkdir(parents=True)
    with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=VC, extrasaction='ignore'); w.writeheader()
        for s in SEEDS:
            row = dict(run_id=f'confirm_bprecapr04_lr3e-05_wd0_s{s}', act='bprecapr04', seed=s, cal='mse4')
            if s in fail: row['status'] = 'fail'
            else:
                fp = 0.30 + dfp + r.normal(0, 0.005); b = base(r, scale)
                if real_refs is not None:
                    b['gbarC'] = float(real_refs[s]['gbarC']) * scale * math.exp(r.normal(0, 0.02)); b['D_allC'] = float(real_refs[s]['D_allC']) * scale * math.exp(r.normal(0, 0.02))
                row.update(b, status='ok', es_test_err_fp32=fp, es_test_err_w8a4a_best=fp + 0.05 + r.normal(0, 0.005), es_test_err_w8a4a_mse4=fp + 0.06, es_w8a4a_best_cal='mse4')
            w.writerow(row)
def res(seed, real=False, **kw):
    with tempfile.TemporaryDirectory() as t:
        S = Path(t); r = np.random.default_rng(seed); d = S / 'v2m'
        if real:
            S = REAL; rr = {int(x['seed']): x for x in csv.DictReader(open(REAL / 'v2k' / 'workers' / 'w0.csv')) if x['act'] == 'bprecap' and x['cal'] == 'mse4'}
            make_v(d, r, real_refs=rr, **kw)
        else:
            make_refs(S, r); make_v(d, r, **kw)
        p = subprocess.run([sys.executable, '-I', str(H / 'v2m_analyze.py'), str(d), str(S)], capture_output=True, text=True)
        if p.returncode: print(p.stdout, p.stderr); raise SystemExit(1)
        return json.load(open(d / 'analysis' / 'v2m_result.json'))
def chk(n, c, i=''):
    global ok; print(('OK  ' if c else 'FEHLER ') + n, i); ok = ok and bool(c)
U = lambda r: (r['Primaer']['E1']['urteil'], r['Primaer']['E2']['urteil'])
r = res(1, scale=1.3); chk('Effekt nach oben -> E1, E2 bestaetigt', U(r) == ('bestaetigt', 'bestaetigt'), str(U(r))); chk('Deutung gesetzt', 'traegt' in r['Deutung'], r['Deutung'])
chk('Lage t bei Variante = ctanh-Niveau nahe 1', 0.6 < r['lage_t_median']['gbarC'] < 1.4, str(r['lage_t_median']['gbarC']))
r = res(2, scale=0.75); chk('Effekt nach unten -> widerlegt', U(r) == ('widerlegt', 'widerlegt'), str(U(r)))
n = 0; ns = 0
for i in range(20):
    rr = res(10 + i); n += any(v != 'unklar' for v in U(rr)); ns += rr['Sekundaer']['S3']['urteil'] != 'kein klarer Unterschied'
chk('Null: Fehlalarme E1/E2 <= 3 von 20', n <= 3, str(n)); chk('Null: Fehlalarme S3 <= 3 von 20', ns <= 3, str(ns))
r = res(3, scale=1.3, fail=(SEEDS[4],)); chk('1 Ausfall -> 8 Paare, E1 bestaetigt', len(r['gueltige_paare']) == 8 and r['Primaer']['E1']['urteil'] == 'bestaetigt', str(U(r)))
chk('1 Ausfall -> S2/S3 mit 9 Paaren (0,9)', r['Sekundaer']['S2']['n'] == 9 and max(r['Sekundaer']['S3']['werte']) > 0.5, str(r['Sekundaer']['S3']['werte'][4]))
r = res(4, scale=1.3, fail=tuple(SEEDS[:3])); chk('3 Ausfaelle -> unklar (zu wenige Paare)', U(r) == ('unklar (zu wenige Paare)',) * 2, str(U(r)))
r = res(5, dfp=0.03); chk('fp32 +3 pp -> S3 Unterschied (rho0,4 groesser)', r['Sekundaer']['S3']['urteil'] == 'Unterschied (rho0,4 groesser)', r['Sekundaer']['S3']['urteil'])
chk('fp32 +3 pp -> S1 Preis kein Unterschied', r['Sekundaer']['S1']['urteil'] == 'kein klarer Unterschied', r['Sekundaer']['S1']['urteil'])
r = res(6, real=True, scale=1.25); chk('echte Referenzen (V2k, V2a2) lesbar, Effekt erkannt', U(r) == ('bestaetigt', 'bestaetigt') and r['referenz_vollstaendig'], str(U(r)))
A2 = [float(x['es_test_err_fp32']) for x in csv.DictReader(open(REAL / 'v2a2' / 'workers' / 'w0.csv')) if x['act'] == 'bprecap' and x['stage'] == 'confirm' and int(x['seed']) in SEEDS]
chk('echte Referenzen: fp32-Median rho0,8 = direkt aus der V2a2-CSV (9 Seeds)', len(A2) == 9 and r['fehler_mediane_pp']['rho08']['es_test_err_fp32'] == round(100 * float(np.median(A2)), 2), str(r['fehler_mediane_pp']['rho08']))
print('AUSWERTUNG OK' if ok else 'AUSWERTUNG FEHLER'); sys.exit(0 if ok else 1)
