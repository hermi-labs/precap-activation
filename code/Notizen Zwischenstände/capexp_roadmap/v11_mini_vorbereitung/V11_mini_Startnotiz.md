# CapExp v11 mini — Startnotiz

## Ausgangslage

v10B Holdout ist abgeschlossen und Gate A wurde bestanden. Damit ist die starke Aussage aus v10B fixiert: In den drei getesteten Aufgaben und bei Tiefe 32/64 mit `res_inv` war PreCap auf frischen, gepaarten Seeds gegen SiLU und GELU durchgehend vorn, ohne numerische Ausfälle.

## Zweck von v11 mini

v11 mini ist keine Wiederholung des Holdouts. Ziel ist eine kleine, faire Mechanismenstudie:

- Kann `precap_wg` flache Netze verbessern, ohne die tiefe Stabilität/Leistung von PreCap zu verlieren?
- Kann `hybrid_gate` den offenen Bereich kleiner Signale besser erhalten und trotzdem im kritischen Bereich schützen?
- Bleibt der bestätigte tiefe Vorteil von PreCap als Referenz intakt?

## Mini-Plan für die Nacht

- Aufgaben: `sin3_quad`, `sin_mix`, `abs_wave`
- Tiefen: 8, 16, 32, 64 (bestehender v11-Stand); 32/64 sind die Schutzachsen aus v10B
- Architekturen: `plain` und `res_inv` bei 8/16; `res_inv` bei 32/64
- Aktivierungen: `precap`, `precap_wg`, `hybrid_gate`, `silu`, `gelu`
- Seeds: 3 als Mini-Stichprobe
- Ziel: technisch stabile, erste Richtungsentscheidung vor dem eigentlichen symmetrischen v11-mini-Hauptlauf

## Interpretation der Nacht-Ergebnisse

Ein positives Signal ist:

1. keine numerischen Ausfälle,
2. `precap_wg` oder `hybrid_gate` sind in flachen Settings nicht klar schlechter als PreCap,
3. PreCap bzw. die Varianten zerstören die bestätigten tiefen Regime nicht.

## Nächster Schritt nach dem Nachtlauf

Wenn der Nachtlauf stabil ist, wird daraus der saubere v11-mini-Hauptplan mit festem Manifest, Seeds 0–9 und kleinem symmetrischem Parameterbudget gebaut.
