"""Tests fuer v0.3 (CPU): weiche Huelle und BPreCap (Arbeitsname).
Referenzwerte (float64) stammen aus dem Studiencode (code/v03, Funktion np_act), damit Paket und Studien nachweislich dieselbe Funktion rechnen."""
import math, torch, pytest
import numpy as np
from precap import BPreCap, bprecap_ref, lb_ref, lagma_hull, BOUNDED, HULL_RHO

X = [-50.0, -4.0, -2.0, -0.7, -0.1, 0.0, 0.1, 0.7, 2.0, 4.0, 8.0, 50.0]
REF = {   # aus code/v03 (np_act) am 31.10.2026
    'bprecap_d100': [-2.935707301152874, -2.602508398885708, -1.7812725546974422, -0.766308940282297, -0.11985100103085572, 0.0, 0.08770175114609274,
                     0.6663785529641164, 2.199461596335903, 2.9997193984487636, 2.9999999938101776, 2.9999999998188316],
    'bprecap_d200': [-2.9995456738199695, -2.8722318482148825, -1.8623207868264382, -0.7417259666325404, -0.11217324574313947, 0.0, 0.09149070965950837,
                     0.6756254998925028, 2.141592722601559, 2.9996257185244453, 2.999999999994009, 3.0],
    'lb_d100': [-2.2165089809922915, -2.2164886652317115, -2.0328018347968286, -0.7247215992559142, -0.10353165703655919, 0.0, 0.10353165703655919,
                0.7247215992559142, 2.0328018347968286, 2.2164886652317115, 2.2165089809921357, 2.2165089809922915],
    'lb_d200': [-2.454293815050829, -2.4541136431337556, -2.036872183119809, -0.7130996514891813, -0.1018713787841688, 0.0, 0.1018713787841688,
                0.7130996514891813, 2.036872183119809, 2.4541136431337556, 2.4542938150397324, 2.454293815050829],
}
def f(name, x): return bprecap_ref(x, name) if name.startswith('bprecap') else lb_ref(x, name)

@pytest.mark.parametrize('name', list(REF))
def test_gleich_studiencode(name):
    y = f(name, torch.tensor(X, dtype=torch.float64)).tolist()
    assert max(abs(a - b) for a, b in zip(y, REF[name])) < 1e-12

@pytest.mark.parametrize('name,chi1', [('bprecap_d100', 1.02), ('bprecap_d200', 1.01), ('lb_d100', 1.02), ('lb_d200', 1.01)])
def test_momente_und_chi1(name, chi1):
    # z ~ N(0, 1): E[f^2] = 1, chi1 = E[f'^2] = 1 + 2/L (Toleranz wie im Studiencode, Huelle numerisch gerechnet)
    z, w = np.polynomial.hermite_e.hermegauss(200); w = w / w.sum()
    zt = torch.tensor(z, requires_grad=True); y = f(name, zt); g, = torch.autograd.grad(y.sum(), zt)
    y, g = y.detach().numpy(), g.numpy()
    assert abs((w * y * y).sum() - 1) < 1e-3 and abs((w * g * g).sum() - chi1) < 2e-3 and abs((w * y).sum()) < 2e-3

def test_huelle_identitaet_glatt_beschraenkt():
    M = 3.0; y = torch.linspace(-10, 10, 200001, dtype=torch.float64); h = lagma_hull(y, M)
    inner = y.abs() <= HULL_RHO * M
    assert (h[inner] - y[inner]).abs().max() < 1e-15                           # Identitaet bis rho*M (bis auf Rundung von M * (y / M))
    assert h.abs().max() < M and torch.all(torch.diff(h) >= 0)               # beschraenkt und monoton
    yt = torch.tensor([HULL_RHO * M - 1e-7, HULL_RHO * M + 1e-7], dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(lagma_hull(yt, M).sum(), yt, create_graph=False)
    assert torch.allclose(g, torch.ones(2, dtype=torch.float64), atol=1e-6)    # C1 an der Naht: Steigung 1 auf beiden Seiten

@pytest.mark.parametrize('depth', [100, 200])
def test_modul_beschraenkt_und_fehler(depth):
    m = BPreCap(depth); x = torch.tensor([-1e6, -50.0, 0.0, 50.0, 1e6], dtype=torch.float64); y = m(x)
    assert torch.isfinite(y).all() and y.abs().max() <= BOUNDED[f'bprecap_d{depth}']['M'] and y[2] == 0
    assert f'bprecap_d{depth}' in repr(m)
    with pytest.raises(ValueError): BPreCap(50)

def test_gradcheck():
    x = (torch.randn(64, dtype=torch.float64) * 4); x = (x + 0.05 * torch.sign(x)).requires_grad_()
    for name in REF:
        assert torch.autograd.gradcheck(lambda t: f(name, t), (x,), eps=1e-6, atol=1e-7)

def test_mlx_gleich_torch():   # nur wenn torch und mlx in derselben Umgebung sind; sonst tests/test_v03_mlx.py (ohne torch)
    mx = pytest.importorskip('mlx.core')
    from precap.mlx import BPreCap as BM, lagma_hull as hm
    x = np.linspace(-20, 20, 4001).astype(np.float32)
    for d in (100, 200):
        a = np.array(BM(d)(mx.array(x))); b = BPreCap(d)(torch.tensor(x)).numpy()
        assert np.abs(a - b).max() < 2e-6
    assert np.abs(np.array(hm(mx.array(x), 2.0)) - lagma_hull(torch.tensor(x), 2.0).numpy()).max() < 2e-6
