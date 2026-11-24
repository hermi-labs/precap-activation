# V2d – Auswertung (privat, mit Claude, 19.11.2026, 15:48)

Stand: 19.11.2026. Präregistrierung: `V2d_Praeregistrierung.md`, Freeze 2026-11-19T06:37:27+0100, Hash-Prüfung vor und nach der Analyse: 21/21 OK. BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen, die Auswertung habe ich mit Claude gemacht. Lauf: 19.11. 07:22 bis 15:47, ein Worker (`v2dw0`), am Netzteil, gestartet mit Aarons Freigabe („du hast den Mac bis 18:30 für dich allein“, 06:37).

Befehle (im Projektordner, 15:48):
```
(cd studien/v2d && shasum -a 256 -c meta/frozen_sha256.txt)      # 21/21 OK, vorher und nachher
python studien/v2d/v2d_analyze.py <absoluter Pfad>/studien/v2d
```
Ausgabe: `analysis/v2d_result.json`. ctanh kommt aus V2d, BPreCap, SELU und TAT aus der V2a2-CSV, tanh aus der V2c-CSV (historische Arme, abgesichert durch die bitgleiche Kontrolle in `meta/kontrolle.txt`, 54/54 Spalten).

## Urteil

- **P1 (Test-Fehler w8a4a_best): „BPreCap besser“.** BPreCap − ctanh: −5,22 pp, 12/0, Holm-p = 0,0010.
- **P2 (4-Bit-Preis, w8a4a_best − fp32): „BPreCap besser“.** BPreCap − ctanh: −2,04 pp, 11/1, Holm-p = 0,0034. Bei gleicher Kalibrierung und ähnlichem Bereich verliert BPreCap durch 4 Bit weniger als die tanh.
- **Deutung nach der Vorab-Tabelle (Zeile 1):** „Bei gleicher Kalibrierung trägt BPreCaps Form (Hülle, Asymmetrie) zum 4-Bit-Vorteil bei. Stärkste Aussage: nicht nur das Rezept, auch die Form zählt.“ P2 zeigt zusätzlich: Der Bereich allein erklärt den Preis nicht.

## Primärtests (Holm über 2)

| Test | Größe | Median BPreCap | Median ctanh | Δ BPreCap − ctanh (pp, Median je Seed) | Siege BPreCap | p (Holm) | Urteil |
|---|---|---|---|---|---|---|---|
| P1 | Test-Fehler w8a4a_best | 48,07 % | 52,54 % | −5,22 | 12/0 | 0,0010 | BPreCap besser |
| P2 | 4-Bit-Preis | 4,64 pp | 6,50 pp | −2,04 | 11/1 | 0,0034 | BPreCap besser |

## Sekundärtests (Holm über 4)

| Test | Größe | Δ (pp) | Siege | p (Holm) | Urteil | vorab (vor Pilot / nach Pilot) |
|---|---|---|---|---|---|---|
| S1 | fp32, BPreCap − ctanh | −3,08 | 12/0 | 0,0020 | besser | 0,35 / 0,7 |
| S2 | w8a4a_best, ctanh − tanh | −15,61 | 12/0 | 0,0020 | besser | 0,95 |
| S3 | w8a4a_best, ctanh − SELU | +2,41 | 1/11 | 0,016 | schlechter | „besser“ 0,45 / 0,15 |
| S4 | w8a4a_best, ctanh − TAT | −9,09 | 12/0 | 0,0020 | besser | 0,9 |

## Beschreibend

Mediane über 12 Seeds, Test-Fehler am frühen Stopp (für alle Formen im Median Schritt 3440, also Ende):

| Form | LR / WD | fp32 | w8a8 | w8a4a_max | w8a4a_best | 4-Bit-Preis | Aktivierung unten / oben / |max| | Trainingsverlust (letzte 50) |
|---|---|---|---|---|---|---|---|---|
| BPreCap | 3e-5 / 0 | 42,88 % | 43,71 % | 49,08 % | 48,07 % | 4,64 pp | −2,24 / 3,00 / 3,00 | 1,18 |
| **ctanh** | 3e-5 / 0 | 46,24 % | 48,13 % | 53,73 % | 52,54 % | 6,50 pp | −2,27 / 2,27 / 2,29 | 1,28 |
| SELU | 3e-5 / 0 | 41,40 % | 43,02 % | 55,87 % | 50,33 % | 8,83 pp | −1,75 / 5,94 / 29,1 | 1,11 |
| TAT | 1e-4 / 0 | 40,27 % | 41,36 % | 89,89 % | 61,63 % | 21,52 pp | −3,70 / 9,10 / 158,2 | 1,03 |
| tanh | 3e-5 / 0 | 66,53 % | 67,01 % | 67,83 % | 68,11 % | 1,26 pp | −0,93 / 0,93 / 1,00 | 1,76 |

- **Tuning ctanh:** 12/12 ok. Gewählt LR 3e-5, WD 0 (Median-Val-Genauigkeit 54,6 %), nicht am Rand (vorab 0,2 für „am Rand“). Val-Fehler am Ende: 1e-5 54,7/55,7 %, 3e-5 44,9/45,9 %, 1e-4 65,9/62,2 % (WD 0, Seeds 42000/42001). ctanh ist empfindlich gegen die Lernrate, WD 3,0 schadet bei 3e-5 auch hier.
- **Kalibrierung ctanh:** gemischt (mse4 4×, p999 4×, p99999 3×, max 1×); BPreCap fast immer mse4 (9×).
- **w8a4a_max (frühes Stoppen):** BPreCap − ctanh −4,13 pp, 12/0. **Am Ende statt am Stopp:** w8a4a_best −5,71 pp, fp32 −2,91 pp, je 12/0. Gleiche Richtung.
- Die Kalibrierung wirkt wie geplant: ctanh trainiert viel besser als tanh (fp32 46,2 % gegen 66,5 %), bleibt aber 3,1 pp hinter BPreCap.

## Explorativ (nicht präregistriert, nachträglich gerechnet)

1. **Relativer 4-Bit-Preis** wie in V2c, (w8a4a_best − fp32) / (90 % − fp32), Mediane: tanh 5,4 %, BPreCap 9,9 %, **ctanh 14,7 %**, SELU 18,2 %. Gepaart BPreCap − ctanh −5,17 Prozentpunkte, 11/1, Wilcoxon p = 0,0010. Der P2-Befund hängt also nicht an der unterschiedlichen fp32-Güte.
2. **Schon 8 Bit kosten ctanh mehr:** w8a8 − fp32 im Median ctanh 2,09 pp, SELU 1,34, BPreCap 0,65, tanh 0,42 pp. Die Empfindlichkeit der ctanh-Netze gegen Quantisierung ist also allgemein höher, nicht erst bei 4 Bit.

**Folgerung aus 1 und 2:** Die einfache Regel aus V2c („je enger der Bereich, desto kleiner der Preis“) gilt so nicht. ctanh hat einen engeren Bereich als BPreCap (±2,29 gegen −2,24…3,00, also kleinere 4-Bit-Stufen) und trotzdem einen größeren Preis. Der Bereich begrenzt Ausreißer, aber wie stark ein Netz auf Rundungsrauschen reagiert, hängt zusätzlich von der Form ab.

**Form gemessen** (Steigung f'(x) mit `make_act` aus dem eingefrorenen Code, MLX, 15:49):

| x | −3 | −2 | −1 | −0,5 | +0,5 | +1 | +1,5 | +2 | +2,5 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|
| BPreCap | 0,21 | 0,42 | 0,79 | 1,06 | 1,00 | 1,16 | 1,27 | 1,30 | 0,68 | 0,11 |
| ctanh | 0,18 | 0,46 | 0,93 | 1,14 | 1,14 | 0,93 | 0,68 | 0,46 | 0,30 | 0,18 |

BPreCap ist auf der positiven Seite bis etwa x = 2 fast linear (Steigung 1,0–1,3) und sättigt erst spät und steil an der Hülle. Links sättigt es weich wie ELU. ctanh krümmt sich dagegen schon ab |x| ≈ 0,5 gleichmäßig auf beiden Seiten.

**Vermutung (nicht geprüft):** Die frühe, gleichmäßige Krümmung von ctanh verändert die Verteilung der Aktivierungen in jeder Schicht stärker. Rundungsfehler wirken dann über 50 Schichten anders als bei der im Hauptbereich fast linearen BPreCap. Prüfbar wäre das mit einer Messung, wie sich Rauschen über die Schichten fortpflanzt. Das braucht gespeicherte Gewichte, die V2a2 nicht hat, also neue Läufe.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Zeilen | Bestätigung 12/12 ok, Tuning 12/12 ok, 0 Ausfälle, 0 fehlend; historisch 48/48 ok |
| run_id | eindeutig |
| nicht-endliche Werte | 0 |
| Tracebacks/Fehler | 0 in `w0.log` und `smoke.log` |
| Hashes | 21/21 OK, vor und nach der Analyse (einschließlich V2a2-Code, V2a2-CSV, V2c-CSV, Pilot, Daten) |
| Mechanik vorab | volle Kontrolle: V2a2-Zeile `confirm_bprecap_lr3e-05_wd0_s42350` mit dem V2d-Skript bitgleich nachgerechnet, 54/54 Spalten (06:38–07:21) |

## Vorab-Wahrscheinlichkeiten gegen Ausgang

| Aussage | vorab (02:55) | nach Pilot (03:43) | eingetreten |
|---|---|---|---|
| alle 12 ctanh-Läufe ok | 0,9 | – | ja |
| P1 „BPreCap besser“ | 0,35 | 0,6 | ja |
| P1 „ctanh besser“ | 0,25 | 0,1 | nein |
| P2 „BPreCap besser“ | 0,3 | – | ja |
| P2 „ctanh besser“ | 0,3 | 0,35 | nein |
| S1 „besser“ | 0,35 | 0,7 | ja |
| S2 „besser“ | 0,95 | – | ja |
| S3 „besser“ (ctanh vor SELU) | 0,45 | 0,15 | nein (schlechter) |
| S4 „besser“ | 0,9 | – | ja |
| Raster am Rand | 0,2 | – | nein |

Vor dem Pilot habe ich die kalibrierte tanh überschätzt. Bei P2 lag ich auch nach dem Pilot falsch: Ich hatte wegen des engeren Bereichs eher einen kleineren Preis für ctanh erwartet (0,35 gegen 0,3 für BPreCap).

## Einordnung

- **V2a2 und V2c:** V2a2 zeigte BPreCap bei 4 Bit vor SELU (−2,55 pp) und TAT (−13,73 pp). V2c zeigte, dass eine unkalibrierte tanh zwar den kleinsten Preis hat, aber schlecht trainiert. V2d schließt die Lücke: Mit demselben Rezept kalibriert, trainiert die tanh gut, BPreCap bleibt aber in fp32 (−3,08 pp), beim 4-Bit-Preis (−2,04 pp) und nach 4 Bit (−5,22 pp) vorn, jeweils 11/1 oder 12/0. **Der Vorteil ist also nicht nur das Rezept, sondern auch die Form.** Die kalibrierte tanh reicht nicht, um SELU bei 4 Bit zu schlagen (S3, +2,41 pp), schlägt aber TAT (S4).
- **Vorarbeit:** LeCun et al. 1998 („Efficient BackProp“) skalieren tanh als 1,7159·tanh(2x/3), damit die Varianz etwa erhalten bleibt. Nachgerechnet hat diese Form E f² = 0,75 und χ1 = 0,80, liegt also im schrumpfenden Bereich (`../v2d_entwurf/Notiz_LeCun_tanh.md`). Neu an ctanh ist nur die exakte Tiefenkalibrierung nach BPreCaps Rezept; als eigener Beitrag taugt ctanh nicht, es ist ein Vergleichsarm. Xiao et al. 2018 (https://arxiv.org/abs/1806.05393) kalibrieren tanh-CNNs über σ_w und σ_b auf χ = 1; diese Variante mit Zufallsbias ist weiter nicht geprüft.

## Grenzen

- **Eine Kalibrierung der tanh** (q = 1, χ1 = 1,04, ohne Bias). Die kritische tanh nach Xiao et al. mit σ_b > 0 und eine tanh mit anderem χ1 sind nicht geprüft.
- **Historische Arme**, nicht zeitgleich gerechnet; abgesichert durch die bitgleiche Kontrolle einer Zeile.
- **Tuning-Raster aus einem eigenen Pilot**; ctanh reagiert stark auf die LR (1e-4 deutlich schlechter). Eine LR zwischen 1e-5 und 3e-5 oder zwischen 3e-5 und 1e-4 könnte ctanh etwas verbessern. Bei 12/0 und −5,2 pp ist eine Umkehr von P1 dadurch unwahrscheinlich, eine Verkleinerung möglich.
- **Die Erklärung über die Form** (Steigung, Sättigung) ist eine Vermutung; gemessen ist nur, dass ctanh schon bei 8 Bit mehr verliert.
- **Alle Grenzen von V2a2:** ein Datensatz (CIFAR-10), ein schlichtes CNN (L = 50, Breite 32), nur PTQ je Tensor, M1 Pro.

## Nächster Schritt

- **v0.4:** Die Kernaussage wird wieder stärker: „Begrenzung verhindert Ausreißer, die Tiefenkalibrierung macht das Netz trainierbar, und BPreCaps Form senkt den 4-Bit-Preis zusätzlich gegenüber einer gleich kalibrierten tanh.“ Abschnitt 6.4 im Entwurf und die Grenze in Abschnitt 7 anpassen.
- **Möglicher Folgeversuch (Aaron entscheidet):** Rauschfortpflanzung über die Tiefe messen (BPreCap, ctanh, SELU), mit gespeicherten Gewichten, um die Form-Vermutung zu prüfen. Klein, etwa 3 Läufe je Form.
- **Sicherung:** `sicherung/v2d_2026-11-19/`.
