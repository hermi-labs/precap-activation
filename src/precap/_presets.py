"""Feste Parameter der PreCap-Familie (s = 1.5, c = 1).

Herleitung der SPreCap-Konstanten: math/p1_spreccap_exact.py und math/p3a3_chi1_family.py.
chi1 = E[f'(z)^2] fuer z ~ N(0, 1); SELU-Fixpunkt Mittel 0 / Varianz 1.
"""
PRESETS = {
    # Grundform, nicht selbstnormierend
    'precap0':      dict(lam=1.0,           alpha=1.0,           b=1.0,          s=1.5, c=1.0),
    # SPreCap, chi1 = 1.1305 (Phase 3a)
    'spreccap':     dict(lam=0.80676869644, alpha=2.39695130922, b=1.0,          s=1.5, c=1.0),
    # SPreCap-cs, chi1 = 1.0716 (Phase 3a-3, Hauptvariante v0.1)
    'spreccap_cs':  dict(lam=1.102702500,   alpha=2.053776267,   b=0.71188112,   s=1.5, c=1.0),
    # SPreCap-e03, chi1 = 1.0300 (Phase 3a-3, beste Variante bei 100 Schichten)
    'spreccap_e03': dict(lam=1.985552225,   alpha=1.608136397,   b=0.4118896262, s=1.5, c=1.0),
    # Tiefenfamilie kL<k>, chi1 = 1 + k/L fuer L verdeckte Schichten (v0.2, Studie A; Werte wie in code/v02/a/a_run.py)
    'spreccap_kl2_d50':  dict(lam=1.614100769, alpha=1.739743617, b=0.496159697, s=1.5, c=1.0),  # chi1 = 1.040
    'spreccap_kl2_d100': dict(lam=2.649586999, alpha=1.461343209, b=0.318766948, s=1.5, c=1.0),  # chi1 = 1.020
    'spreccap_kl2_d200': dict(lam=4.170485099, alpha=1.296823062, b=0.212387949, s=1.5, c=1.0),  # chi1 = 1.010
    'spreccap_kl1_d50':  dict(lam=2.649586999, alpha=1.461343209, b=0.318766948, s=1.5, c=1.0),  # chi1 = 1.020, = kl2_d100
    'spreccap_kl1_d100': dict(lam=4.170485099, alpha=1.296823062, b=0.212387949, s=1.5, c=1.0),  # chi1 = 1.010, = kl2_d200
    'spreccap_kl1_d200': dict(lam=6.296620086, alpha=1.197830367, b=0.145678357, s=1.5, c=1.0),  # chi1 = 1.005
}
DEFAULT = 'spreccap_cs'
