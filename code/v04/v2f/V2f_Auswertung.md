# V2f – Auswertung

Stand: 21.11.2026, 09:45, mit Claude. Präregistrierung: `V2f_Praeregistrierung.md`, Freeze 2026-11-20T06:56:12+0100. Hash-Prüfung vor und nach der Analyse: 41/41 OK. Analyse: `v2f_analyze.py` unverändert, Ausgabe `analysis/v2f_result.json`. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen.

## Urteil

- **H1 Kohärenz (sup bei mse4, BPreCap kleiner): nicht bestätigt.** Median q = 0,989, 6 von 9 Seeds in Richtung, p_Holm = 0,150.
- **H2 Weitergabe (ḡ^Q bei mse4, BPreCap kleiner): bestätigt.** Median q = 0,844, 9 von 9, p_Holm = 0,006.
- **H3 Rundung (r̄ bei mse4, BPreCap größer): bestätigt.** Median q = 1,096, 9 von 9, p_Holm = 0,006.
- Gesamtdeutung nach der vorab festgelegten Regel: **Der Kohärenz-Befund ist nicht bestätigt. Der Hinweis im v0.4-Entwurf wird entfernt.** Bestätigt ist dagegen: BPreCap rundet gröber, aber bei abschneidender Kalibrierung (mse4) erreicht ein Rundungsfehler gleicher Größe die Logits schwächer.

## Primärtest (Kalibrierung mse4, q = BPreCap / ctanh je Seed, exakter Wilcoxon auf log q, einseitig, Holm über 3)

| Test | Kennzahl | Richtung | Median q | in Richtung | p einseitig | p zweiseitig | p Holm | Urteil |
|---|---|---|---|---|---|---|---|---|
| H1 Kohärenz | sup | BPreCap kleiner | 0,989 | 6/9 | 0,150 | 0,301 | 0,150 | nicht bestätigt |
| H2 Weitergabe | ḡ^Q | BPreCap kleiner | 0,844 | 9/9 | 0,002 | 0,004 | 0,006 | bestätigt |
| H3 Rundung | r̄ | BPreCap größer | 1,096 | 9/9 | 0,002 | 0,004 | 0,006 | bestätigt |

q je Seed 42353–42361:
- sup: 0,989; 1,023; 0,906; 1,005; 1,109; 0,955; 0,934; 0,877; 0,991
- ḡ^Q: 0,888; 0,766; 0,811; 0,844; 0,900; 0,899; 0,795; 0,847; 0,829
- r̄: 1,097; 1,067; 1,097; 1,096; 1,107; 1,095; 1,082; 1,096; 1,089

Mediane der Kennzahlen bei mse4 (BPreCap / ctanh): r̄ 0,0819 / 0,0748; ḡ^Q 0,223 / 0,263; sup 1,072 / 1,097; D^Q 0,274 / 0,303; err_a4 (1000 Bilder) 46,9 % / 51,7 %; err_fp32 44,3 % / 47,8 %.

## Sekundärtests (Hinweise, eigene Holm-Korrektur je Block)

**S1, Kalibrierung max:**

| Test | Median q | in Richtung | p Holm | Urteil |
|---|---|---|---|---|
| H1 sup | 0,961 | 6/9 | 0,164 | nicht bestätigt |
| H2 ḡ^Q | 1,178 | 0/9 | 1,0 | nicht bestätigt (zweiseitig p = 0,004: bei max erreichen BPreCaps Einzelfehler die Logits **stärker**) |
| H3 r̄ | 1,124 | 9/9 | 0,006 | bestätigt |

**S2, Gesamtstörung und Fehler (Holm über 4):**

| Kennzahl | Kalibrierung | Median q | BPreCap kleiner | p Holm | Urteil |
|---|---|---|---|---|---|
| D^Q | mse4 | 0,846 | 7/9 | 0,027 | bestätigt |
| err_a4 | mse4 | 0,904 | 9/9 | 0,008 | bestätigt |
| D^Q | max | 1,007 | 4/9 | 0,545 | nicht bestätigt |
| err_a4 | max | 0,943 | 9/9 | 0,008 | bestätigt |

**S3, gepoolt mit den 3 Seeds der V2e-Nachmessung (12 Paare, beschreibend):**
- mse4: sup Median q 0,965 (9/12 kleiner), ḡ^Q 0,837 (12/12), r̄ 1,095 (0/12 kleiner, also 12/12 größer), D^Q 0,850 (10/12), err_a4 0,905 (12/12).
- max: sup 0,962 (9/12), ḡ^Q 1,176 (0/12), r̄ 1,122 (0/12), D^Q 1,000 (6/12), err_a4 0,945 (11/12).

**S4:** Spearman zwischen log q(sup) und log q(err_a4) bei mse4: ρ = 0,48, p = 0,19, n = 9.

**Zerlegung** (Mediane der log q): mse4: log q(D) = −0,168, davon log q(√Σd²) = −0,140 und log q(sup) = −0,011. max: +0,007, +0,006 und −0,040. Bei mse4 kommt die kleinere Gesamtstörung also fast ganz aus kleineren Einzelbeiträgen der Schichten, kaum aus geringerer Kohärenz.

## Explorativ (nicht präregistriert)

4-Bit-Preis auf den 1000 Messbildern (err_a4 − err_fp32), aus `workers/w0.csv` nachgerechnet:
- mse4: Median BPreCap 3,1 pp, ctanh 3,9 pp, Differenz-Median −1,4 pp, 6/9 kleiner, zweiseitig p ≈ 0,10 (Normal-Näherung wegen Bindungen).
- max: 5,4 pp gegen 6,0 pp, Differenz-Median 0,0 pp, 5/9 kleiner, p ≈ 0,48.

Daraus folgt: Der bestätigte err_a4-Vorteil bei max (S2) kommt fast ganz aus dem niedrigeren fp32-Fehler von BPreCap und nicht aus einem kleineren 4-Bit-Preis. Der Preisunterschied aus V2d (10 000 Testbilder, w8a4a_best) wird hier nicht neu getestet. 1000 Bilder sind dafür zu wenige.

## Integrität

- 36 von 36 Zeilen, alle ok, run_ids eindeutig.
- Alle 18 Läufe sind bitgleich zur historischen Zeile aus V2a2 bzw. V2d (`kontrolle_n_diff = 0`).
- 9 von 9 gültigen Paaren, keine nicht-endlichen Werte.
- Unterbrechung: Am 20.11. fiel um 09:12 das Netzteil aus, der Wächter beendete den Worker. Mit `resume_v2f.sh` wurde der Lauf um 21:41 nach einer Hash-Prüfung wieder aufgenommen. Der Worker ist wiederaufnehmbar, und die Bitgleichheit aller Läufe zeigt, dass die Unterbrechung nichts verändert hat.
- Vorab-Wahrscheinlichkeiten gegen Ergebnis:
  - bitgleich 0,9 → ja
  - H1 0,55 → nein
  - H2 0,75 → ja
  - H3 0,85 → ja
  - S1-H1 0,5 → nein
  - S2 D^Q bei mse4 0,75 → ja

## Einordnung

Begriff: ḡ^Q ist das geometrische Mittel über i = 1…49 von g_i = d_i / r_i. Dabei ist d_i die relative Abweichung der Logits, wenn nur der Eingang von Schicht i auf 4 Bit gerundet wird, und r_i ist der relative Rundungsfehler dort (`v2e_run.measure`). ḡ^Q misst also, wie stark ein Rundungsfehler gegebener Größe die Logits erreicht. Ein Verstärkungsfaktor von Schicht zu Schicht ist es nicht. „Weitergabe“ ist in diesem Sinn gemeint.

- V2e (präregistriert, 3 Seeds) fand keinen klaren Unterschied in der Weitergabe einzelner Störungen. Die explorative Nachmessung mit fester Kalibrierung (`../v2e_kal/Notiz.md`) zeigte drei Muster:
  - **gröbere Rundung**: an 9 neuen Seeds bestätigt
  - **schwächere Weitergabe bei mse4**: bestätigt
  - **geringere Kohärenz**: nicht bestätigt (die 9/9 aus 3 Seeds × 3 Kalibrierungen waren kleine, nicht unabhängige Effekte)
- Das Bild ist jetzt: **BPreCap rundet etwa 10 % gröber, verträgt das Abschneiden der Ränder aber besser.** Bei mse4 erreicht ein Rundungsfehler gleicher Größe die Logits etwa 16 % schwächer, daher die etwa 15 % kleinere Gesamtstörung. Bei der weiten max-Kalibrierung ist es umgekehrt, und die Gesamtstörung ist gleich.
- Der Vorteil hängt also an der Wechselwirkung von Form und abschneidender Kalibrierung, nicht an einer allgemein geringeren Fehlerkohärenz. Das passt dazu, dass bei diesen 9 Seeds in V2a2/V2d meist eine stark abschneidende Kalibrierung als w8a4a_best gewählt wurde (mse4 oder p999): BPreCap 7/9 (6× mse4), ctanh 7/9 (3× mse4). Das ist eine Deutung, kein kausaler Test.
- Prior art: Die Literaturprüfung zur Messung „Superposition gegen Einzelfehler“ läuft (Recherche 21.11., `docs/literatur/`). Bis dahin behaupte ich keine Neuheit.

## Grenzen

- 9 Seeds je Form, ein Netz (CNN, L = 50, ohne Normierung), ein Datensatz, nur Aktivierungen auf 4 Bit je Tensor, Gewichte fp32, 1000 Messbilder, M1 Pro.
- ḡ^Q ist das geometrische Mittel über die Schichten 1–49. Es sagt nicht, wo im Netz der Unterschied entsteht.
- Feste Kalibrierungen mse4 und max, nicht die im Lauf gewählte.
- Mechanismus-Beschreibung, kein Eingriff.

## Nächster Schritt

- v0.4-Entwurf 6.4 und 7: den Kohärenz-Hinweis entfernen. Stattdessen den bestätigten Befund berichten, mit Verweis darauf, dass V2f präregistriert ist und 9 neue Seeds hat: gröbere Rundung, schwächere Weitergabe bei abschneidender Kalibrierung.
- Mögliche Folgestudie (Idee, nicht geplant): Ein Eingriff, der nur die Ränder abschneidet (mse4 gegen max je Schicht), würde zeigen, ob die schwächere Weitergabe an der Sättigung der BPreCap-Hülle hängt.
