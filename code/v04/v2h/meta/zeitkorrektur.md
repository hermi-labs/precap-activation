# Korrektur der Uhrzeiten in den eingefrorenen Dateien (mit Claude, 21.11.2026 18:42, nicht eingefroren)

In `V2h_Praeregistrierung.md` und im Kommentar von `v2h_run.py` stehen Uhrzeiten, die ich geschätzt statt mit `date` geprüft habe. Sie sind falsch, weil sie nach dem Freeze liegen. Die Dateien sind eingefroren und bleiben unverändert. Richtig ist:

| Angabe in der Datei | richtig (aus Datei-Zeitstempeln und `date`) |
|---|---|
| Präregistrierung „Stand: 21.11.2026, 18:50“ | zuletzt geändert 18:32:18, Freeze 18:32:29 (`frozen_at.txt`) |
| σ_b, σ_w², q* „gelöst … (21.11., 18:35)“ | vor 18:26:28 (dann war `v2h_run.py` geschrieben) |
| Varianzverlauf „am 21.11. um 18:40 gemessen“ | Selbsttest vor dem Freeze, etwa 18:26:40 bis 18:30:30 (Laufzeit 3:51) |

Die Reihenfolge stimmt trotzdem: Konstanten, Selbsttest, Präregistrierung, Freeze, Start (18:41:18). Inhaltlich ändert sich nichts.
