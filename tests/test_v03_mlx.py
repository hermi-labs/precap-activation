"""MLX-Teil der v0.3-Tests ohne torch: BPreCap und Huelle in MLX gegen die float64-Referenzwerte aus dem Studiencode.
Laeuft mit pytest oder direkt: python tests/test_v03_mlx.py"""
import ast, pathlib
import numpy as np
try:
    import mlx.core as mx
except ImportError:
    mx = None
_src = (pathlib.Path(__file__).parent / 'test_v03.py').read_text()
X = ast.literal_eval(_src.split('X = ', 1)[1].split('\n', 1)[0])
REF = ast.literal_eval(_src.split('REF = {', 1)[1].split('\n}', 1)[0].join(['{', '}']).replace('{   #', '{ #'))

def test_mlx_gleich_referenz():
    if mx is None:
        import pytest; pytest.skip('mlx fehlt')
    from precap.mlx import BPreCap, lagma_hull, bprecap_ref_mlx
    from precap import BOUNDED
    x = mx.array(np.array(X, dtype=np.float32))
    for name, ref in REF.items():
        if name.startswith('bprecap'):
            y = np.array(bprecap_ref_mlx(x, name))
        else:
            p = BOUNDED[name]; y = np.array(lagma_hull(p['beta'] * x, p['M']))
        assert np.abs(y - np.array(ref)).max() < 3e-6, name
    for d in (100, 200):
        assert np.array_equal(np.array(BPreCap(d, backend='ref')(x)), np.array(bprecap_ref_mlx(x, f'bprecap_d{d}')))

def test_mlx_kernel_gleich_referenz():
    # fusionierte Metal-Kernel (BPreCap, Lagma) gegen die MLX-Referenz: vorwaerts und Gradient, float32-Rundung
    if mx is None: return
    from precap.mlx import BPreCap, Lagma, bprecap_ref_mlx, lagma_hull
    x = mx.array(np.linspace(-40, 40, 400001).astype(np.float32))
    for d in (100, 200):
        k, r = BPreCap(d), (lambda t, d=d: bprecap_ref_mlx(t, f'bprecap_d{d}'))
        assert float(mx.max(mx.abs(k(x) - r(x)))) < 1e-6, d
        gk = mx.grad(lambda t: k(t).sum())(x); gr = mx.grad(lambda t: r(t).sum())(x)
        assert float(mx.max(mx.abs(gk - gr))) < 2e-6, d
        assert float(mx.max(mx.abs(k(x)))) <= 3.0
    for M in (2.2165089809922915, 3.0):
        L = Lagma(M); assert float(mx.max(mx.abs(L(x) - lagma_hull(x, M)))) < 1e-6
        gl = mx.grad(lambda t: L(t).sum())(x); gr = mx.grad(lambda t: lagma_hull(t, M).sum())(x); assert float(mx.max(mx.abs(gl - gr))) < 1e-6
    xh = mx.array(np.linspace(-8, 8, 1001).astype(np.float16))   # halbe Genauigkeit: Rechnung in float32, Rueckgabe im Eingangstyp
    assert BPreCap(100)(xh).dtype == mx.float16

if __name__ == '__main__':
    test_mlx_gleich_referenz(); test_mlx_kernel_gleich_referenz(); print('MLX-TEST OK')
