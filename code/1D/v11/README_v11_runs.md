# v11 Runs

## Ziel
v11 testet einen Schutzmechanismus im Motorsteuergerät der Aktivierung: unten freier, oben geregelt. Die Referenz bleibt PreCap, weil PreCap in tiefen Netzen stark war, während flache Netze in v10 Teil A eher von SiLU oder GELU profitierten.

## Dateien
- `capexp_v11_proto.py` – Prototyp mit `precap`, `precap_wg`, `hybrid_gate`, `silu`, `gelu`, `relu`.
- `run_v11_mini.sh` – kleiner Vorabtest.
- `run_v11_full.sh` – großer Sweep.
- `README_v11_proto.md` – mathematische Beschreibung der Änderungen.

## Mini-Run
Der Mini-Run ist als schneller Sicherheitscheck gedacht, bevor der große Sweep startet.

### Mini-Run Inhalt
- Aufgaben: `sin3_quad`, `sin_mix`, `abs_wave`
- Tiefen: 8, 16, 32, 64
- Architekturen: `plain` und `res_inv` bei 8/16; nur `res_inv` bei 32/64
- Aktivierungen: `precap`, `precap_wg`, `hybrid_gate`, dazu `silu` und `gelu` als Baselines
- Seeds: 3 statt 10

### Start
`cd /Users/aaronschmitt/PyCharmMiscProject/v11 && caffeinate -i bash run_v11_mini.sh`

## Großer Sweep
Nur starten, wenn der Mini-Run in Tiefe 32/64 stabil bleibt und flach nicht klar schlechter ist.

### Start
`cd /Users/aaronschmitt/PyCharmMiscProject/v11 && caffeinate -i bash run_v11_full.sh`
