# CapExp v11 — Abschluss

- v11 mini: 918/918 Runs, keine Ausfälle. `precap_wg` verworfen; hybrid_gate war in flachen Zellen knapp, verfehlte aber tiefe Toleranz.
- v11 confirm: 2400/2400 eindeutige Runs, 1 bad-Run (hybrid_gate; abs_wave; Tiefe 64; Seed 102). Hybrid_gate besteht keine vorab festgelegte Regel gegen PreCap: flach Median d=+0,0214, Win-Rate 46,1 %, nur 6/18 Zellen besser; tief schlechtester Zellmedian Faktor 2,28 > 1,5.
- Schlussfolgerung: PreCap bleibt die beste getestete Variante. Der Cap-Mechanismus ist gegenüber SiLU/GELU besonders in res_inv und tiefen Netzen stark; die Plain-Schwäche wird in v12 als Mechanismusfrage untersucht.

