"""Paket-Tests (CPU). CUDA- und MLX-Kernel: siehe kernels/test_cuda.py und kernels/test_mlx.py."""
import math, sys, pathlib, torch, pytest
import precap
from precap import PreCap, precap as precap_fn, precap_ref, precap_ref_grad, PRESETS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'kernels'))
import reference as kref  # eingefrorene Kernel-Referenz

@pytest.mark.parametrize('name', ['precap0', 'spreccap', 'spreccap_cs'])
def test_gleich_kernel_referenz(name):
    assert PRESETS[name] == kref.PRESETS[name]
    x = torch.randn(10000, dtype=torch.float64) * 4
    assert torch.equal(precap_ref(x, **PRESETS[name]), kref.precap_ref(x, **PRESETS[name]))

@pytest.mark.parametrize('name', list(PRESETS))
def test_gradient_und_gradcheck(name):
    p = PRESETS[name]
    x = (torch.randn(64, dtype=torch.float64) * 3)
    x = (x + 0.05 * torch.sign(x)).requires_grad_()
    precap_ref(x, **p).sum().backward()
    assert torch.allclose(x.grad, precap_ref_grad(x.detach(), **p), atol=1e-12)
    assert torch.autograd.gradcheck(lambda t: precap_ref(t, **p), (x.detach().requires_grad_(),), eps=1e-6, atol=1e-8)

@pytest.mark.parametrize('name', ['spreccap', 'spreccap_cs', 'spreccap_e03'])
def test_selbstnormierung(name):
    # Gauss-Hermite: E[f]=0, E[f^2]=1 fuer z ~ N(0,1)
    import numpy as np
    z, w = np.polynomial.hermite_e.hermegauss(200); w = w / w.sum()
    zt = torch.tensor(z); f = precap_ref(zt, **PRESETS[name]).numpy()
    assert abs((w * f).sum()) < 2e-3 and abs((w * f * f).sum() - 1) < 2e-3

def test_beschraenkt():
    x = torch.tensor([-1e6, -50.0, 50.0, 1e6], dtype=torch.float64)
    y = precap_ref(x, **PRESETS['spreccap_e03'])
    assert torch.isfinite(y).all() and -2.49 < y.min() < -2.47 and 6.10 < y.max() < 6.12

def test_modul_und_fehler():
    m = PreCap('spreccap_e03'); x = torch.randn(5, 7)
    assert torch.equal(m(x), precap_fn(x, 'spreccap_e03'))
    assert 'spreccap_e03' in repr(m)
    with pytest.raises(ValueError): PreCap('gibtsnicht')
    with pytest.raises(ValueError): PreCap(backend='xyz')
    assert PreCap().preset == 'spreccap_cs' and precap.__version__ == '0.1.0'

def test_nan_inf():
    x = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float64)
    for p in PRESETS.values():
        y = precap_ref(x, **p)
        assert torch.isnan(y[0]) and torch.isfinite(y[1:]).all()

def test_fp16_bf16_cpu():
    x = torch.randn(1000) * 3
    for dt in (torch.float16, torch.bfloat16):
        y = PreCap()(x.to(dt)); assert y.dtype == dt
