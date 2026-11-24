# Studien zu v0.4

Zehn präregistrierte Studien zum Preprint v0.4 ([`paper/PreCap_paper_v0.4.pdf`](../../paper/PreCap_paper_v0.4.pdf)) an einfachen CNNs ohne Normierung und ohne Residuen auf CIFAR-10, dazu eine präregistrierte Wiederholung des Hauptvergleichs mit PyTorch/CUDA (C7). Dazu kommen eine beschreibende Nachmessung (V2e-kal) und eine Studie, die vor dem ersten Training abgebrochen wurde (V2g). Alle Studien außer C7 liefen mit Apple MLX auf einem MacBook Pro M1 Pro, C7 auf einer NVIDIA RTX 4070 (Windows, PyTorch 2.14.1, CUDA 12.6). Jede Studie ändert genau eine Sache gegenüber ihrer Vorgängerin.

| Ordner | Frage | Abschnitt im Preprint | Bericht |
|---|---|---|---|
| `v2a1/` | BPreCap gegen SELU, L = 32, einfaches Training (Adam, konstante LR) | 5.1 | `v2a1/V2a1_Auswertung.md` |
| `v2a2/` | BPreCap gegen SELU und TAT (mit und ohne CLE), L = 50, AdamW mit Kosinus, Tuning je Form | 5.2 | `v2a2/V2a2_Auswertung.md` |
| `v2c/` | gegen tanh ohne Tiefenkalibrierung (Begrenzung allein) | 5.3 | `v2c/V2c_Auswertung.md` |
| `v2d/` | gegen eine tanh nach BPreCaps Rezept (gleiche Kalibrierung, andere Form) | 5.4 | `v2d/V2d_Auswertung.md` |
| `v2e/` | Rauschmessung: Rundungsrauschen je Schicht und Weitergabe an die Logits (3 Seeds) | 5.5 | `v2e/V2e_Auswertung.md` |
| `v2e_kal/` | Nachmessung zu V2e bei fester Kalibrierung (beschreibend) | 5.5 | `v2e_kal/Notiz.md` |
| `v2f/` | Rauschmessung an 9 neuen Seeds je Form, feste Kalibrierungen | 5.5 | `v2f/V2f_Auswertung.md` |
| `v2k/` | exakte Zerlegung des 4-Bit-Fehlers in Abschneiden und Runden an den V2f-Netzen | 5.5 | `v2k/V2k_Auswertung.md` |
| `v2m/` | Eingriff an der Form: Knie der Hülle bei ρ = 0,4 statt 0,8 (frühere Sättigung), Messung wie V2k | 5.5 | `v2m/V2m_Auswertung.md` |
| `v2g/` | kritische tanh, vor dem ersten Training abgebrochen (Fehler beim Schreiben des Selbsttests) | 8 | `v2g/ABGEBROCHEN.md` |
| `v2gb/` | kritische tanh am fast linearen Punkt der kritischen Linie (Neustart von V2g mit unverändertem Design) | 5.6 | `v2gb/V2gb_Auswertung.md` |
| `v2h/` | kritische tanh am Punkt nach Hayou et al. (2019) | 5.6 | `v2h/V2h_Auswertung.md` |
| `c7/` | unabhängige Wiederholung von V2a2 (BPreCap, SELU, TAT) mit PyTorch/CUDA auf einer RTX 4070 | 4.6, 5.2 | `c7/C7_Auswertung.md` |

`v2a_entwurf/`, `v2c_entwurf/` und `v2d_entwurf/` enthalten nur die Pilot-Dateien, die in den Hash-Listen der Studien stehen.

## Inhalt je Ordner

- `<NAME>_Praeregistrierung.md`: Sie wurde vor dem Start eingefroren und ist unverändert. Einige Präregistrierungen und Skript-Kommentare erwähnen weitere, unveröffentlichte Arbeiten. Sie bleiben trotzdem unverändert, damit die Hashes stimmen.
- `meta/frozen_sha256.txt`, `meta/frozen_at.txt`, `meta/selfcheck.json`: Hashes, Zeitpunkt des Einfrierens und Selbsttest. Prüfen mit `(cd code/v04/<name> && shasum -a 256 -c meta/frozen_sha256.txt)`. Die Zeilen zu den CIFAR-Dateien zeigen auf den Datenordner des Mac.
- `*_run.py`, `*_analyze.py`, `launch_*.sh`, `test_*.py`: eingefrorene Skripte. Der Datenordner lässt sich mit der Umgebungsvariablen `PRECAP_DATA` setzen.
- `workers/`, `tune/workers/`: Rohergebnisse je Lauf (CSV). `analysis/`: Ausgabe der Auswertung. `logs/`: Protokolle der Worker. `mess/`: Messwerte der Rauschstudien (JSON).
- Die Berichte `*_Auswertung.md` sind auf Deutsch. Verweise auf unveröffentlichte Arbeiten sind darin entfernt, siehe `REDAKTION.md`.

## Hinweise zu C7 und V2m

- **C7, Hash-Datei:** `c7/meta/frozen_sha256.txt` wurde unter Windows geschrieben und hat Zeilenenden CRLF. Prüfen mit `(cd code/v04/c7 && tr -d '\r' < meta/frozen_sha256.txt | shasum -a 256 -c -)`. Die Zeilen `DATA/...` beziehen sich auf den Datenordner des PCs (CIFAR-10-Binärdateien, dieselben Hashes wie in den anderen Studien).
- **C7, Kopfzeile der Präregistrierung:** Sie sagt noch „Entwurf, noch nicht eingefroren“. Der Inhalt ist der eingefrorene Stand (Hash OK), nur die Kopfzeile wurde vor dem Einfrieren nicht angepasst. Die Datei bleibt unverändert, damit der Hash stimmt.
- **C7, Äquivalenztest:** `c7/test/` enthält den Test gegen die MLX-Referenz (`mlx_referenz_c7.py`, `test_c7_aequivalenz.py`), die Ergebnisse auf CPU und CUDA (`aequivalenz_ausgabe_cpu_*.txt`, `aequivalenz_ergebnis_cuda.json`) und den Test des Auswertungsskripts. Die Referenzwerte als Feld (`mlx_ref_c7.npz`, 11 MB) sind nicht enthalten; `mlx_referenz_c7.py` erzeugt sie neu aus dem eingefrorenen `v2a2/v2a2_run.py`; eine Neuerzeugung am 24.11.2026 war bitgleich (SHA-256 wie in der Hash-Liste).
- **C7, Abstände je Seed:** `c7/analysis/je_seed_cuda_minus_mlx.txt` wurde getrennt aus denselben CSV-Dateien berechnet (beschreibend, ohne Test), weil das eingefrorene Skript diese Sekundärgröße nicht ausgibt.
- **V2m, Konstanten:** `v2m/kL2B_d50_rho04.json` enthält die neu gelösten Konstanten für ρ = 0,4. Der Löser (`math/p4_bounded_depth.py`, von `v2m_konstanten.py` importiert) steht in der Hash-Liste, ist aber nicht enthalten. Ohne ihn meldet die Hash-Prüfung diese eine Zeile als fehlend. Für den Lauf selbst (`v2m_run.py`) wird er nicht gebraucht; die Konstanten prüft der Selbsttest (E f = 0, E f² = 1, χ1 = 1,04).
- **V2m, Messdaten:** `v2m/mess/*.json` sind enthalten. Die Gewichte der neun Netze und die Fehlerfelder (`mess/*_delta.npz`, zusammen etwa 32 MB) sind nicht enthalten.

## Nicht enthalten

- Die Gewichte der nachtrainierten Netze aus V2e und V2f (zusammen etwa 58 MB) und die Fehlerfelder aus V2k (`mess/*.npz`, etwa 62 MB). Sie sind für die Hash-Prüfung von V2k und für eine Neuberechnung der Zerlegung nötig. Ohne sie meldet die Hash-Prüfung von V2k die Zeilen zu `../v2f/gewichte/` als fehlend.
- Die Rauchtest-Ordner (`smoke/`).
