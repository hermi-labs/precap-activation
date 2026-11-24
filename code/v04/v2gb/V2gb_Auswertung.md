# V2gb – Auswertung

Stand: 21.11.2026, 18:10, mit Claude. Präregistrierung: `V2gb_Praeregistrierung.md` (Fassung von V2g mit drei Korrekturen, Design unverändert), Freeze 2026-11-21T09:39:09+0100. Hash-Prüfung vor und nach der Analyse: 21/21 OK. Analyse: `v2gb_analyze.py` unverändert, Ausgabe `analysis/v2gb_result.json`. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen.

## Urteil

- **P1 (Test-Fehler w8a4a_best am frühen Stopp): „BPreCap besser“.** Median BPreCap − xtanh = −18,51 pp, 12 von 12 Seeds, Holm-p = 0,001.
- **P2 (4-Bit-Preis): „xtanh besser“.** Median der Differenz +3,56 pp, 0 von 12, Holm-p = 0,001. Die kritische tanh verliert bei 4 Bit weniger. Sie startet aber von einem fp32-Fehler von 64,7 %.
- Deutung nach der vorab festgelegten Tabelle: **Auch gegen die kritische tanh nach Xiao et al. liegt BPreCap bei 4 Bit vorn. Die Grenze in Abschnitt 7 wird für diesen Punkt der kritischen Linie geschlossen.** Das Muster entspricht V2c (unkalibrierte tanh).

## Primärtests (Wilcoxon gepaart, zweiseitig, 12 Seeds 42350–42361, Holm über 2)

| Test | Größe | Median BPreCap | Median xtanh | Median Δ (pp) | BPreCap kleiner | p | p Holm | Urteil |
|---|---|---|---|---|---|---|---|---|
| P1 | es_test_err_w8a4a_best | 48,07 % | 65,74 % | −18,51 | 12/12 | 0,0005 | 0,001 | BPreCap besser |
| P2 | Preis = w8a4a_best − fp32 | 4,64 pp | 1,06 pp | +3,56 | 0/12 | 0,0005 | 0,001 | xtanh besser |

Δ je Seed:
- P1: −18,79; −19,50; −18,26; −18,77; −18,54; −18,71; −15,09; −16,97; −16,25; −18,84; −18,48; −17,29
- P2: 3,56; 3,57; 2,07; 3,73; 5,96; 4,05; 3,42; 5,21; 2,84; 3,30; 3,73; 2,51

## Sekundärtests (Holm über 4)

| Nr. | Vergleich | Größe | Median Δ (pp) | erstgenannter kleiner | p Holm | Urteil |
|---|---|---|---|---|---|---|
| S1 | BPreCap − xtanh | fp32 | −22,19 | 12/12 | 0,002 | besser |
| S2 | xtanh − tanh (V2c) | w8a4a_best | −1,73 | 12/12 | 0,002 | besser |
| S3 | xtanh − ctanh (V2d) | w8a4a_best | +13,88 | 0/12 | 0,002 | schlechter |
| S4 | xtanh − SELU | w8a4a_best | +15,48 | 0/12 | 0,002 | schlechter |

S2: Die kritische Initialisierung hilft der tanh messbar, aber wenig (1,7 pp, an allen 12 Seeds).

## Beschreibend

Mediane über 12 Seeds am frühen Stopp (alle Läufe stoppen bei Schritt 3440, also am Ende):

| Form | fp32 | w8a8 | w8a4a_max | w8a4a_best | Preis (pp) | Aktivierungsbereich (Median lo / hi) | Kalibrierwahl |
|---|---|---|---|---|---|---|---|
| BPreCap | 42,88 % | 43,71 % | 49,08 % | 48,07 % | 4,64 | −2,24 / 3,00 | mse4 9, p999 1, p9999 1, p99999 1 |
| SELU | 41,40 % | 43,02 % | 55,87 % | 50,33 % | 8,83 | −1,75 / 5,94 | mse4 7, p999 5 |
| ctanh | 46,25 % | 48,13 % | 53,73 % | 52,54 % | 6,50 | −2,27 / 2,27 | mse4 4, p999 4, p99999 3, max 1 |
| TAT | 40,27 % | 41,36 % | 89,89 % | 61,63 % | 21,52 | −3,70 / 9,10 | mse4 12 |
| **xtanh** | **64,72 %** | **65,16 %** | **66,10 %** | **65,74 %** | **1,06** | −0,94 / 0,94 | p999 4, p99999 4, p9999 3, max 1 |
| tanh (V2c) | 66,53 % | 67,01 % | 67,83 % | 68,11 % | 1,26 | −0,93 / 0,93 | p9999 5, mse4 3, p99999 2, p999 1, max 1 |

- Endpunkt: BPreCap − xtanh bei w8a4a_best −18,50 pp (12/0), bei fp32 −22,09 pp (12/0).
- w8a4a_max am frühen Stopp: −16,87 pp (12/0, p = 0,0005).
- Train-Loss (Mittel der letzten 50 Schritte): xtanh 1,72, tanh 1,76, BPreCap 1,18. Die xtanh lernt also kaum besser als die unkalibrierte tanh.
- Tuning: Gewählt wurde LR 3e-5, WD 0 (Median Val-Genauigkeit 35,7 %). **Die Wahl liegt nicht am Rand des Rasters.** 1e-4 war an beiden Tuning-Seeds schlechter (Val-Fehler 69,2 % gegen 65,5 % bzw. 63,1 %). Auch der Einwand, ein kleines q* vertrage größere Lernraten, trifft in diesem Raster nicht zu.

## Integrität

- Bestätigung 12/12 Zeilen, alle ok, run_ids eindeutig, 0 nicht-endliche Werte, keine fehlenden Seeds.
- Tuning 12/12 ok. Historische Arme 60/60 ok.
- Freeze-Hashes von V2a2, V2c und V2d sowie die CIFAR-Dateien sind bei beiden Prüfungen OK.
- Selbsttest (`meta/selfcheck.json`):
  - Fixpunkt ok, χ1(q*) = 1,0
  - Bias-Std 0,00444 (Soll √2·10⁻⁵ = 0,00447)
  - BPreCap-Parameter mit der Unterklasse bitgleich
  - xtanh-Gewichte genau σ_w mal Original
  - 20-Schritt-Probe bitgleich
- Lauf: 21.11. 09:47:59 bis 18:04, etwa 22 min je Bestätigungslauf, ohne Unterbrechung.
- Vorgeschichte: V2g brach am 21.11. um 09:32 im Selbsttest ab (numpy-bool in `json.dumps`), bevor ein Training lief (`../v2g/ABGEBROCHEN.md`). Die Ergebnisse stammen vollständig aus V2gb.
- Vorab-Wahrscheinlichkeiten gegen Ergebnis:
  - alle 12 ok 0,9 → ja
  - P1 BPreCap besser 0,85 → ja
  - P1 xtanh besser 0,03 → nein
  - P2 xtanh besser 0,6 → ja
  - S1 besser 0,85 → ja
  - S2 besser 0,25 → **ja**
  - S3 schlechter 0,75 → ja
  - Wahl am Rand 0,25 → nein

## Einordnung

- Die kritische tanh bei q* ≈ 0,026 verhält sich wie vorab erwartet ähnlich wie die unkalibrierte tanh aus V2c:
  - schlecht trainierbar (fp32 64,7 % gegen 66,5 %)
  - kleinster 4-Bit-Preis aller Formen (1,06 pp)
- Sie ist etwas besser als die V2c-tanh (S2, 1,7 pp). Das hatte ich mit 0,25 für unwahrscheinlich gehalten.
- Damit gilt die Rangfolge bei w8a4a_best jetzt für sechs Formen im selben Netz: **BPreCap 48,1 % < SELU 50,3 % < ctanh 52,5 % < TAT 61,6 % < xtanh 65,7 % < tanh 68,1 %**.
- Der kleine Preis der tanh-Varianten kommt nicht aus einer besseren 4-Bit-Eignung. Er kommt daher, dass das Netz fast linear arbeitet und wenig gelernt hat. Ein kleiner Preis auf hohem Fehler ist kein Vorteil.
- Prior art: Die Recherche vom 21.11. (`docs/literatur/reports/Quantisierungsfehler über die Tiefe.md`) fand keinen Vergleich begrenzter Aktivierungen bei 4-Bit-PTQ gegen eine kritisch initialisierte tanh in normierungsfreien CNNs. Neuheit behaupte ich damit nicht. Das Vertrauen der Recherche liegt bei 0,6–0,65.

## Grenzen

- **Nur ein Punkt der kritischen Linie**, nämlich der fast lineare (σ_b² = 2·10⁻⁵, q* = 0,026, Pennington 2017 Fall 1).
  - Hayou et al. 2019 empfehlen bei dieser Tiefe einen größeren Bias (σ_b ≈ 0,03–0,04, q* ≈ 0,10–0,13). Dieser Punkt ist nicht geprüft (Entwurf `../v2h_entwurf/V2h_Entwurf.md`).
  - Auch der Punkt q* = 1 mit Bias ist nicht geprüft. BPreCaps Rezept bei q = 1 ohne Bias entspricht ctanh (V2d).
- Regime: AdamW, 3440 Schritte, Kosinus auf 0. Xiao et al. trainieren deutlich länger mit SGD. Die kritische Initialisierung hilft vor allem bei sehr großer Tiefe, L = 50 ist dafür klein.
- LR-Raster ohne eigenen Pilot. Die Wahl lag in der Mitte.
- Die historischen Arme wurden nicht zeitgleich gerechnet. Die Prüfungen des Selbsttests ersetzen eine volle Kontrollzeile.
- Ein Datensatz (CIFAR-10), ein Netz, nur PTQ je Tensor, M1 Pro.

## Nächster Schritt

- v0.4-Entwurf, Abschnitt 7: Die Grenze „kritische tanh nicht verglichen“ durch das Ergebnis ersetzen. Die verbleibende Grenze offen nennen: Nur ein Punkt der kritischen Linie ist geprüft, der Hayou-Punkt nicht.
- Ob V2h (Hayou-Punkt, etwa 9 h GPU) vor v0.4 läuft, entscheidet Aaron. Meine Einschätzung: Bei q* ≈ 0,12 arbeitet die tanh weiter fast linear (tanh(0,35) ≈ 0,34). Ich erwarte ein ähnliches Bild, Vorab-Wahrscheinlichkeit für „BPreCap besser“ bei P1 etwa 0,85. Der Lauf würde eine Gutachter-Frage schließen, aber kaum etwas an der Aussage ändern.
