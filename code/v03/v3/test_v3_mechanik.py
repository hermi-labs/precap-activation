# Mechanik-Test V3 (vor dem Freeze, mit Claude). Ausgabe "MECHANIK OK" oder Fehler.
# (a) Basisnetz tat Seed 42168 (volle 6880 Schritte) bitgleich zur V1-Bestaetigung (val, test, w8a4a_best, Kalibrierung);
# (b) QNet am Start ergibt exakt den PTQ-Fehler w8a4a_best des Basisnetzes (Quantisierer starten auf dem PTQ-Gitter);
# (c) PACT: kein Gradient zu lo/hi aus Werten im Bereich, wohl aber aus geclippten; LSQ: Gradient zu s aus Werten im Bereich;
# (d) mit LR_Q = 0 bleiben die Quantisierer im Feintuning fest (kein WD auf ihnen), die Gewichte aendern sich; (e) Feintuning-Plan Kosinus auf 0.
import importlib.util, sys
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
H = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('v3', H / 'v3_qat.py'); Q = importlib.util.module_from_spec(sp); sys.modules['v3'] = Q; sp.loader.exec_module(Q); V = Q.V
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)
D = V.data_for('cifar10', 'tat'); xtr, ytr, xv, yv, xte, yte = D
m, st = Q.train_base('tat', 42168, D); row = Q.ref_row('tat', 42168, 'confirm')
_, va = V.evaluate(lambda x: m(x), xv, yv); _, ta = V.evaluate(lambda x: m(x), xte, yte); e4, bc, lohi = Q.ptq(m, D)
ok('a_basis_bitgleich_V1', st == 'ok' and 1 - va == float(row['val_err_fp32']) and 1 - ta == float(row['test_err_fp32']) and e4 == float(row['test_err_w8a4a_best']) and bc == row['w8a4a_best_cal'], (1 - va, row['val_err_fp32'], e4, row['test_err_w8a4a_best']))
numel = [l.weight.shape[1] for l in m.hid + [m.out]]
for mm in ('LSQ', 'PACT'):
    qn = Q.QNet(m, Q.Quant(mm, lohi, numel)); qn.eval(); _, a = V.evaluate(lambda x: qn(x), xte, yte)
    ok(f'b_{mm}_start_gleich_PTQ', 1 - a == e4, (1 - a, e4))
x = mx.array(np.linspace(-0.9, 2.9, 1000).astype(np.float32)); xo = mx.array(np.array([-3.0, 5.0], dtype=np.float32))
qp = Q.Quant('PACT', [(-1.0, 3.0)], [1]); f = lambda q_, v: mx.sum(q_(0, v))
g_in = nn.value_and_grad(qp, f)(qp, x)[1]; g_out = nn.value_and_grad(qp, f)(qp, xo)[1]
ok('c_PACT_kein_Gradient_im_Bereich', float(g_in['lo'][0]) == 0.0 and float(g_in['hi'][0]) == 0.0, (float(g_in['lo'][0]), float(g_in['hi'][0])))
ok('c_PACT_Gradient_geclippt', float(g_out['lo'][0]) == 1.0 and float(g_out['hi'][0]) == 1.0, (float(g_out['lo'][0]), float(g_out['hi'][0])))
ql = Q.Quant('LSQ', [(-1.0, 3.0)], [1]); g_l = nn.value_and_grad(ql, f)(ql, x)[1]
ok('c_LSQ_Gradient_s_im_Bereich', abs(float(g_l['s'][0])) > 0, float(g_l['s'][0]))
Q.FT_STEPS = 30; f0 = Q.copy_net(m, 'tat', 42168); net, st2, _ = Q.finetune(f0, 'tat', 42168, 'LSQ', 0.0, D, lohi)
q0 = Q.Quant('LSQ', lohi, numel)
ok('d_quantisierer_fest_bei_lrq0', st2 == 'ok' and all(float(a_) == float(b_) for a_, b_ in zip(net.q.s + net.q.z, q0.s + q0.z)))
ok('d_gewichte_aendern_sich', float(mx.max(mx.abs(net.m.hid[5].weight - m.hid[5].weight))) > 0)
f1 = Q.copy_net(m, 'tat', 42168); net1, _, _ = Q.finetune(f1, 'tat', 42168, 'PACT', 1e-2, D, lohi)
ok('d_PACT_grenzen_bewegen_sich_bei_lrq', any(abs(float(a_) - float(b_)) > 1e-4 for a_, b_ in zip(net1.q.hi, Q.Quant('PACT', lohi, numel).hi)))
sch = optim.cosine_decay(2e-5, 1376); ok('e_plan', abs(float(sch(mx.array(0))) - 2e-5) < 1e-12 and float(sch(mx.array(1376))) < 1e-12)
print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER'); sys.exit(0 if all(R.values()) else 1)
