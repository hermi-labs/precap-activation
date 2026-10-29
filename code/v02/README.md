# Studien zu v0.2

Sieben präregistrierte Studien zum Preprint v0.2 (https://doi.org/10.5281/zenodo.23757287). Alle liefen mit Apple MLX auf einem MacBook Pro M1 Pro. Jede Studie ändert genau eine Sache gegenüber ihrer Vorgängerin.

| Ordner | Frage | Läufe | Bericht |
|---|---|---|---|
| `a/` | Tiefenfamilie kL1/kL2 (χ1 = 1 + k/L) gegen SELU, Tiefe 50/100/200 | 882 | `a/A_Auswertung.md` |
| `a2/` | kL2 gegen TAT (Leaky ReLU) und SELU, 1720 und 6880 Schritte | 280 | `a2/A2_Auswertung.md` |
| `a3/` | kL2 mit dem TAT-Rezept (SUO-Init, Normierung je Beispiel) gegen TAT | 198 | `a3/A3_Auswertung.md` |
| `q1/` | 4-Bit-Aktivierungen nach dem Training, Max-Kalibrierung: cs gegen SELU/ELU | 240 | `q1/Q1_Auswertung.md` |
| `q1b/` | wie Q1, mit fairer Kalibrierung (beste aus max, p99,9, p99,99, MSE) | 240 | `q1b/Q1b_Auswertung.md` |
| `q2/` | cs gegen TAT bei 4 Bit, symmetrisch pro Tensor | 158 | `q2/Q2_Auswertung.md` |
| `q3/` | cs gegen TAT bei 4 Bit, asymmetrisch pro Tensor | 180 | `q3/Q3_Auswertung.md` |

**Q1 und Q1b gehören immer zusammen.** Q1 allein überzeichnet den Effekt stark.

## Inhalt je Ordner

- `<NAME>_Praeregistrierung.md`: vor dem Start eingefroren, unverändert. Die Präregistrierungen erwähnen vereinzelt weitere, noch unveröffentlichte Studien. Sie wurden trotzdem nicht verändert, damit die Hashes stimmen.
- `meta/frozen_sha256.txt`, `meta/frozen_at.txt`: Hashes und Zeitpunkt des Einfrierens. Prüfen mit `(cd code/v02/<name> && shasum -a 256 -c meta/frozen_sha256.txt)`.
- `*_run.py`, `*_analyze.py`, `launch_*.sh`: eingefrorene Skripte. `src/` ist die eingefrorene Kopie des Pakets, mit der die Studie lief.
- `workers/`, `tune/workers/`: Rohergebnisse je Lauf (CSV). `analysis/`: Ausgabe der Auswertung. `logs/`: Protokolle der Worker.
- `<NAME>_Auswertung.md`: Bericht. In der veröffentlichten Fassung sind nur der Vermerk „privat“ und Querverweise auf unveröffentlichte Studien entfernt, die Ergebnisse sind unverändert.

**Lücke bei Q1:** `meta/` und `logs/` von Q1 wurden am 27.10.2026 durch einen Dateiabgleich auf dem Mac geleert, die Hash-Datei fehlt deshalb. Die Rohdaten in `q1/workers/` sind vollständig (SHA-256 im Bericht). Q1b hat alle 240 Läufe von Q1 neu trainiert und 960 von 960 Werten bitgleich reproduziert.

## Nachrechnen

```bash
export PRECAP_DATA=/pfad/zu/data          # Ordner mit mnist/ und fashion/ (IDX-Dateien)
python code/v02/a/a_run.py --selfcheck    # Selbsttest, MLX nötig
python code/v02/a2/a2_analyze.py code/v02/a2 code/v02/a    # Auswertung, scipy nötig
```

Die Auswertungsskripte erwarten die Ordner der Vorgängerstudien als weitere Argumente. Welche das sind, steht oben im jeweiligen Bericht (dort mit den alten Pfaden `studien/<name>`, im Repo `code/v02/<name>`). Die Abbildungen 7 bis 9 des Preprints entstehen mit `python paper/make_figures_v02.py` direkt aus diesen Rohdaten.

## Nachgerechnet vor der Veröffentlichung (29.10.2026)

Alle sieben Auswertungen wurden aus genau diesen Repo-Dateien neu berechnet (Python 3.9, scipy 1.13.1):

- **A2, A3, Q2, Q3:** Ausgaben bitgleich.
- **A, Q1, Q1b:** Alle Urteile und alle Signifikanzentscheidungen sind gleich, auch alle im Preprint genannten p-Werte. Einzelne p-Werte in Sekundärtests weichen um bis zu 0,11 ab. Grund: Diese drei Auswertungen liefen ursprünglich in einer anderen Umgebung mit einer anderen scipy-Version, die beim Wilcoxon-Test bei Bindungen bzw. Nulldifferenzen anders zwischen exakter Verteilung und Normalapproximation wählt. Im Repo liegen die ursprünglichen Ausgaben.
