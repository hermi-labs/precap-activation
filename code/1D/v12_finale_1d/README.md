# CapExp v12 finale — 1D-Abschluss

Dieser Ordner friert den Abschluss der 1D-Phase ein. Es werden keine neuen 1D-Varianten mehr gesucht.

## Kernergebnis

PreCap zeigt in dieser Studienreihe einen robusten Vorteil gegenüber SiLU und GELU **in der residualen inversen Architektur (`res_inv`)**. Der Vorteil ist nicht universell: In `plain`-Netzen ist PreCap meist schlechter und der Nachteil wächst mit der Tiefe. Gewichtete und hybride Reparaturen liefern keinen belastbaren Zusatznutzen gegenüber PreCap.

## Was eingefroren ist

- PreCap-Konfiguration in `meta/precap_frozen_reference.json`
- Abschlussauswertungen und Synthese in `docs/`
- Gültigkeitsbereich: synthetische 1D-Regression, getestete Breite/Architekturen/Optimierungsregime

## Nächste Phase

Kein weiterer 1D-Lauf. Als nächstes wird ein 2D-Pilot präregistriert: PreCap gegen SiLU und GELU, mit gleicher Parameterzahl, residualer Architektur, neuen Seeds und einem unabhängigen Holdout bei Pilot-Erfolg.
