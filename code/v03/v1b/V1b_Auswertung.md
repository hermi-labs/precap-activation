# V1b – Auswertung: Wo liegt die LR-Grenze mit Anlauf? (beschreibend)

Stand: 03.11.2026, nachts. Auswertung strikt nach `V1b_Praeregistrierung.md` (eingefroren 03.11., 00:55). Lauf und Auswertung mit Claude, in Aarons Abwesenheit.

## Kurz

- **Grenze von BPreCap mit Anlauf: 3e-4.** Bei 3e-4 trainiert BPreCap ohne Einbruch, ist aber etwa 2,7 pp schlechter als bei 2e-4 (48,4–48,6 % gegen 45,6–45,9 %). Bei 5e-4 bricht es ein (81–84 %).
- **10 % Anlauf verschiebt die Grenze nicht.** Bei BPreCap ist die Grenze mit 5 % und mit 10 % Anlauf gleich. Bei 2e-4 sind beide Längen gleich gut (+0,12 pp).
- **5e-5 ist schlechter als 1e-4:**
  - BPreCap: +1,74 pp;
  - TAT: +2,15 pp.
  
  Die V1-Wahl am unteren Rand ist damit abgesichert.
- **TAT bei 5e-4:** Mit 5 % Anlauf liegt TAT bei 71,7 / 58,5 %. Das ist schlecht, aber nach dem Kriterium (über 80 %) kein Einbruch. Mit 10 % bricht es ein (83,8 / 87,7 %). Nach der Regel liegt TATs Grenze also bei 5e-4 (5 %) bzw. 3e-4 (10 %).
- **Kontrolle bitgleich** zum V1-Tuning (4 von 4), 32 von 32 Läufen ok, Hashes 15 von 15 OK.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 32 von 32 ok |
| Kontrolle (2e-4, Anlauf 344, beide Formen, beide Seeds) | **bitgleich** zum V1-Tuning in allen Kennzahlen |
| Hash-Prüfung | 15 von 15 OK |
| Vorab-Tests | Selbsttest (n_tune = 32), Mechanik, Auswertung an künstlichen Daten und Rauchtest ok |
| Laufzeit | 00:57–01:34, 2 Worker, M1 Pro am Netzteil |

Befehl: `.venv/bin/python -I studien/v1b/v1b_analyze.py studien/v1b studien/v1`. Die Ausgabe steht in `logs/analyse_ausgabe.txt`, das Ergebnis in `analysis/v1b_result.json`.

## Val-Fehler am Ende (fp32, Seeds 42000 / 42001)

| Form (WD) | LR | Anlauf 5 % (344) | Anlauf 10 % (688) |
|---|---|---|---|
| BPreCap (0,3) | 5e-5 | 46,3 / 46,6 | 45,8 / 46,2 |
| | 1e-4 (aus V1) | 44,8 / 44,6 | – |
| | 2e-4 | 45,9 / 45,6 | 45,9 / 45,9 |
| | 3e-4 | 48,6 / 48,4 | 47,4 / 48,0 |
| | 5e-4 | **80,9 / 81,7** | **81,4 / 83,6** |
| TAT (0) | 5e-5 | 49,7 / 49,7 | 50,6 / 50,8 |
| | 1e-4 (aus V1) | 47,1 / 48,0 | – |
| | 2e-4 | 44,9 / 45,7 | 46,1 / 44,7 |
| | 3e-4 | 47,5 / 47,6 | 47,0 / 47,8 |
| | 5e-4 | 71,7 / 58,5 | **83,8 / 87,7** |

Fett: Einbruch, also Val-Fehler über 80 %.

## Ergebnisse nach den Vorab-Kriterien

| Nr. | Frage | BPreCap | TAT |
|---|---|---|---|
| G1 | Grenze mit 5 % / 10 % Anlauf | **3e-4 / 3e-4** | 5e-4 / 3e-4 |
| G2 | 10 % verschiebt die Grenze | nein | nein |
| G3 | 5e-5 gegen 1e-4 (V1) | **schlechter**, +1,74 pp | schlechter, +2,15 pp |
| G4 | 10 % gegen 5 % bei 2e-4 | +0,12 pp | +0,12 pp |

## Lücke gegenüber der Präregistrierung

Die Präregistrierung nennt den 4-Bit-Fehler je Zelle als beschreibende Größe. In der Tuning-Stufe rechnet der Code aus B10/V1 aber keine Quantisierung, die Felder sind leer. Diese Größe fehlt deshalb. Auf die Fragen G1–G4 hat das keinen Einfluss.

## Deutung nach den Vorab-Regeln

- **G1, BPreCap ≥ 3e-4:** Die v0.4-Formulierung lautet „trainiert mit Anlauf bis 3·10⁻⁴“. Ehrlich dazu: Bei 3e-4 ist BPreCap etwa 2,7 pp schlechter als bei 2e-4. Es trainiert, aber nicht gut.
- **G2 nein:** Ein längerer Anlauf hilft hier nicht. Die Grenze bei etwa 3e-4 bis 5e-4 ist eher eine Eigenschaft von Form und Rezept.
- **G3 „schlechter“:** Die V1-Wahl 1e-4 ist am unteren Rand abgesichert. Die V1-Ergebnisse bleiben ohne Vorbehalt.
- **TAT verträgt 5e-4 mit 5 % Anlauf etwas besser als BPreCap,** ist dort aber ebenfalls weit von seinem besten Wert entfernt (58,5–71,7 % gegen 45 %). Für die Praxis liegen die brauchbaren LR beider Formen bei 1e-4 bis 3e-4.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| G1 BPreCap hält mit 5 % bis 3e-4 | 0,5 | ja |
| G1 TAT hält mit 5 % bis 3e-4 | 0,6 | ja (sogar bis 5e-4 nach dem Kriterium) |
| G2 10 % verschiebt die Grenze bei BPreCap | 0,4 | nein |
| G3 BPreCap „5e-5 besser“ | 0,15 | nein, schlechter |
| Kontrolle bitgleich | 0,95 | ja |

## Folge für v0.4 (Entwurf der LR-Aussage, ersetzt den Satz aus V1)

> „Ohne LR-Anlauf divergiert BPreCap bei 2·10⁻⁴. Mit einem linearen Anlauf über 5 % der Schritte trainiert es bis 3·10⁻⁴ und divergiert bei 5·10⁻⁴; ein Anlauf von 10 % ändert das nicht. Die beste LR bleibt 1·10⁻⁴. TAT verliert bei 5·10⁻⁴ ebenfalls stark.“

## Grenzen

- 2 Seeds, nur beschreibend.
- Keine Messpunkte zwischen 3e-4 und 5e-4.
- WD fest auf die V1-Wahl.
- Kein 4-Bit-Fehler in dieser Stufe.
- Nur CIFAR-10, d100, TAT-Rezept, MLPs.

**Vertrauen:**
- 0,8, dass die Grenze von BPreCap mit Anlauf zwischen 3e-4 und 5e-4 liegt;
- 0,75, dass ein längerer Anlauf sie nicht verschiebt (nur zwei Längen getestet).
