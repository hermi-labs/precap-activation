# B11 – Auswertung: Hält der Befund auf CIFAR-100? (BPreCap gegen TAT, praxisübliches Regime, d100)

Stand: 31.10.2026, 23:05. Die Auswertung folgt strikt `B11_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg.** Auf CIFAR-100 ist BPreCap bei fairen 4 Bit **1,92 pp besser als TAT** (11/1, p = 0,0010).
- **Anders als auf CIFAR-10 kommt der Vorteil hier vor allem aus fp32**, nicht aus einem kleineren 4-Bit-Preis.
  - BPreCap ist schon in fp32 1,98 pp besser (12/0, S1).
  - Der 4-Bit-Preis ist bei beiden fast gleich: BPreCap +9,8 pp, TAT +9,8 pp.
  - Die Vorab-Erwartung „TAT zahlt bei 4 Bit mehr“ (0,75) trifft **nicht** zu.
- **TATs Bereich bleibt hier klein** (8–9 statt 28 auf CIFAR-10), weil das Tuning für TAT LR 1e-4 wählt, nicht 2e-4. Das passt zu He et al. 2024 und B10: Der große Bereich entsteht durch hohe LR.
- **Die lineare Form LB_tr ist bei w4a besser als BPreCap** (S5 „schlechter“ für BPreCap, +1,03 pp, 1/11), obwohl sie in fp32 schlechter ist (S4).
- **Neu und für v0.3 wichtig: BPreCap verträgt nur kleine LR.** Im Tuning bricht BPreCap ab LR 2e-4 auf Zufallsniveau ein, TAT trainiert dort noch. Auf CIFAR-100 liegt BPreCaps Wahl am unteren Rand des Rasters (5e-5).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 72 von 72 ok, Bestätigung 38 von 38 ok, 0 Ausfälle |
| Hash-Prüfung | 18 von 18 OK (`shasum -a 256 -c meta/frozen_sha256.txt`) |
| Kontrolle | kL2B_tr und TAT auf CIFAR-10, Seed 42168, B10-Hyperparameter: alle Kennzahlen an allen Messpunkten bitgleich zu B10 |
| Schlaf | `pmset -g log` meldet zwischen 19:39 und 22:46 keinen Sleep- oder Wake-Eintrag. Am Netzteil |
| Laufzeit | 19:39–22:45 |
| Sicherung | `sicherung/b11_2026-10-31/`, 16 von 16 Dateien OK |

Befehl: `.venv/bin/python -I b11_analyze.py ~/…/studien/b11 ~/…/studien/b10`, Ausgabe in `logs/analyse_ausgabe.txt`, Ergebnis in `analysis/b11_result.json`.

**Abweichung, offen genannt:** S8 enthält eine Differenz von genau 0. Dann wechselt scipy trotz `method='exact'` still zur Normal-Näherung (Warnung in der Ausgabe). Der gespeicherte p-Wert für S8 ist 0,0058. Exakt ohne die Null wäre er 0,0029. Das Urteil „besser“ ändert sich nicht. Alle anderen Tests haben keine Nullen und sind exakt.

## Tuning (mediane Val-Genauigkeit in % am Ende, 2 Seeds)

| Form | LR-Spitze | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|---|
| kL2B_tr | 5e-5 | 22,9 | **24,7** | 17,6 |
| | 1e-4 | 22,0 | 22,9 | 17,3 |
| | 2e-4 | 1,9 | 1,7 | 11,2 |
| | 5e-4 | 1,7 | 1,6 | 3,7 |
| LB_tr | 5e-5 | 21,9 | **23,9** | 16,5 |
| | 1e-4 | 20,3 | 21,2 | 15,6 |
| | 2e-4 | 1,7 | 1,7 | 8,8 |
| | 5e-4 | 1,8 | 1,6 | 3,8 |
| TAT | 5e-5 | 22,0 | 21,3 | 14,8 |
| | 1e-4 | **23,6** | 22,7 | 11,3 |
| | 2e-4 | 22,4 | 16,7 | 5,1 |
| | 5e-4 | 1,4 | 0,7 | 0,7 |

- **Wahl:** kL2B_tr 5e-5 / WD 0,3 (LR am unteren Rand), LB_tr 5e-5 / WD 0,3 (unterer Rand), TAT 1e-4 / WD 0 (innen).
- Die Erweiterung um 5e-4 hat keiner Form geholfen.
- Weil BPreCap und LB am unteren Rand liegen, ist offen, ob eine noch kleinere LR ihnen hilft. Das könnte den Vorsprung nur vergrößern, nicht verkleinern.

## Primärtest

| Test | Median | Siege/Niederlagen | p | Urteil |
|---|---|---|---|---|
| kL2B_tr − TAT, w8a4a, frühes Stoppen | **−1,92 pp** | 11/1 | 0,0010 | **besser, Erfolg** |

Differenzen (pp, sortiert): −2,90, −2,82, −2,63, −2,42, −2,27, −1,98, −1,86, −1,60, −1,46, −0,91, −0,74, +0,28.

## Sekundärtests (Holm über 8)

| Nr. | Vergleich | Median | S/N | p (Holm) | Urteil |
|---|---|---|---|---|---|
| S1 | kL2B_tr − TAT, fp32, ES | −1,98 | 12/0 | 0,0039 | besser |
| S2 | kL2B_tr − TAT, w8a4, ES | −3,61 | 12/0 | 0,0039 | besser |
| S3 | LB_tr − TAT, w8a4a, ES | −2,84 | 12/0 | 0,0039 | besser |
| S4 | kL2B_tr − LB_tr, fp32, ES | −1,55 | 12/0 | 0,0039 | besser |
| S5 | kL2B_tr − LB_tr, w8a4a, ES | +1,03 | 1/11 | 0,0049 | **schlechter** |
| S6 | kL2B_tr − TAT, w8a4a, Ende | −2,37 | 12/0 | 0,0039 | besser |
| S7 | kL2B_tr − TAT, fp32, Ende | −2,16 | 12/0 | 0,0039 | besser |
| S8 | LB_tr − TAT, fp32, ES | −0,61 | 10/1 (1 Null) | 0,0058 | besser |

## Beschreibend (Mediane am Stopp-Punkt, Fehler in %)

| Form | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis (w4a − fp32) | Bereich |
|---|---|---|---|---|---|---|
| kL2B_tr | 74,1 | 86,3 | 83,9 | 97,8 | +9,8 | 3,0 |
| LB_tr | 75,4 | 83,0 | 82,9 | 96,2 | +7,5 | 2,22 |
| TAT | 76,1 | 89,6 | 85,9 | 98,5 | +9,8 | 8,2–8,6 |

- **Stopp-Punkte:** kL2B_tr 8 von 12 am Ende, LB_tr 8 von 12, TAT 6 von 12.
- **Verlauf w4a:** Bei allen Formen ist w4a um Schritt 3440–4300 am besten und wird danach wieder schlechter, obwohl fp32 weiter sinkt. Bei TAT am stärksten (84,4 → 86,0), bei BPreCap 82,5 → 83,7. Das frühe Stoppen nach fp32-Val trifft den besten 4-Bit-Punkt also nicht.
- **w8a3** liegt bei allen Formen nahe am Zufallsniveau (99 %) und ist auf CIFAR-100 nicht aussagekräftig.
- Das absolute Niveau ist wie erwartet schwach (MLP ohne Faltungen, etwa 25 % Genauigkeit). Es geht um den Vergleich.

## Deutung nach den Vorab-Regeln

- **Erfolg:** Der 4-Bit-Vorteil gegenüber TAT gilt auch auf einem zweiten, schwereren Datensatz.
- **Aber der Mechanismus ist ein anderer als auf CIFAR-10.** Auf CIFAR-10 (B10) waren beide in fp32 gleich, und TAT verlor bei 4 Bit durch seinen großen Bereich. Auf CIFAR-100 ist BPreCap schon in fp32 besser, und beide zahlen bei 4 Bit gleich viel. Die beiden Befunde werden gemeinsam berichtet.
- **BPreCap ist nicht die beste Form bei 4 Bit:** LB_tr (linearer Kern, gleiche Hülle) ist bei w4a besser. Das spricht dafür, dass die **Hülle** den 4-Bit-Vorteil trägt und der PreCap-Kern vor allem die fp32-Güte verbessert (S4). In B10 auf CIFAR-10 war es umgekehrt (S5 „besser“ für BPreCap, −1,36 pp). Die Rolle des Kerns bei 4 Bit hängt also am Datensatz.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,5 | ja |
| S1 „unklar“ (fp32 gleichauf) | 0,5 | nein, „besser“ |
| S2 „besser“ | 0,75 | ja |
| 4-Bit-Preis TAT > BPreCap | 0,75 | nein (beide +9,8) |
| Kontrolle bitgleich | 0,95 | ja |

## Kritische Prüfung

1. **Hängt das Ergebnis an TATs LR-Wahl?** Hier weniger als in B10. TAT liegt innen (1e-4), seine Nachbarn 5e-5 und 2e-4 sind im Tuning nur knapp schlechter. Der Vorsprung kommt vor allem aus fp32 und ist über alle Seeds stabil (S1 12/0).
2. **LR-Empfindlichkeit von BPreCap.** Bei 2e-4 bricht BPreCap ein, auf CIFAR-100 und schon in B10 auf CIFAR-10 (14,5 %). TAT nicht. Das ist eine echte Schwäche der Form im Rezept und gehört in v0.3. Eine mögliche Erklärung ist die Kritikalität χ1 = 1 + 2/L gegenüber TATs schwächerer Nichtlinearität, aber das ist nicht geprüft.
3. **Wahl am Rasterrand.** BPreCap und LB am unteren Rand. Konservativ für den Primärtest, aber für den Vergleich BPreCap gegen LB offen.
4. **Nur 2 Tuning-Seeds,** nur d100, MLP, Quantisierung nach dem Training.

## Folgen für v0.3

- Aussage: „Auf CIFAR-10 und CIFAR-100 bei 4 Bit besser als TAT im selben Rezept. Auf CIFAR-10 durch einen kleineren 4-Bit-Preis, auf CIFAR-100 durch bessere fp32-Güte.“
- Offen nennen: Die lineare Form mit gleicher Hülle ist auf CIFAR-100 bei 4 Bit besser. BPreCap verträgt nur kleine LR (ab 2e-4 instabil).
- Der Kern-Beitrag bei 4 Bit ist nicht einheitlich. Die Hülle ist der stabilere Teil der Aussage.

**Vertrauen:** 0,85 für den Primärbefund (Richtung und Größenordnung), 0,7 für die Deutung „Hülle trägt 4 Bit, Kern trägt fp32“.
