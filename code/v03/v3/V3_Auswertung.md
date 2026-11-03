# V3 – Auswertung: QAT als Obergrenze – BPreCap mit PTQ gegen LSQ- und PACT-QAT (CIFAR-10, d100, TAT-Rezept)

Stand: 03.11.2026, 13:15, mit Claude. Ausgewertet strikt nach `V3_Praeregistrierung.md` (Freeze 11:38). Läufe 11:45–13:11 auf 2 Workern (M1 Pro, Netzteil). PreCap und BPreCap sind Aarons Ideen. Den Anstoß gab ein Reddit-Kommentar zu v0.3.

## Kurzfassung

- **Primär: „schlechter“.** TAT mit LSQ-QAT ist bei 4 Bit besser als BPreCap mit einfacher PTQ: **+3,01 pp, 0/12, p = 0,0005**. Gegen TAT mit PACT-QAT ist es fast gleich (+3,00 pp, 0/12).
- **S2 unklar:** Wenn beide Formen mit LSQ-QAT trainieren, sind sie gleich (−0,05 pp, 7/5). Mit PACT ist BPreCap knapp besser (S3: −0,27 pp, 10/2).
- **Deutung nach der Vorab-Regel:** „QAT ist besser. Der Vorteil von BPreCap ist ein PTQ-Vorteil. S2 unklar: QAT schließt den Abstand.“
- **Einordnung:**
  - Mit PTQ allein ist BPreCap 11,85 pp besser als TAT (S6). QAT holt bei TAT 14,9 pp (S4), bei BPreCap nur 2,9 pp (S5).
  - BPreCap mit PTQ liegt 3 pp hinter dem QAT-Niveau, TAT mit PTQ fast 15 pp.
  - Die Hülle ersetzt QAT also nicht, macht es aber weitgehend entbehrlich, wenn ein Training mit Quantisierung nicht möglich ist.
- **Diesmal lernt PACT wirklich:** Die Grenzen bewegen sich um 32 % (BPreCap) bzw. 62 % (TAT) der Breite. S1 und S3 sind damit faire PACT-Tests.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Hashes (`shasum -a 256 -c meta/frozen_sha256.txt`) | 6 von 6 OK, darunter `../v1/v1_run.py` |
| Zeilen | Tuning 32/32 ok, Bestätigung 96/96 ok, 0 Ausfälle |
| Basis bitgleich zu V1 | Bestätigung 24/24, Tuning 4/4 |

Befehl: `.venv/bin/python -I studien/v3/v3_analyze.py studien/v3`. Ergebnis in `analysis/v3_result.json`.

## Tuning der Quantisierer-LR (Val-Fehler mit QAT-Inferenz %, Seeds 42000 / 42001)

| Form, Methode | LR_Q 1e-4 | 1e-3 | 1e-2 | Wahl |
|---|---|---|---|---|
| BPreCap LSQ | 47,66 / 47,42 | 47,66 / 46,32 | 46,36 / 46,88 | 1e-2 (Rand) |
| BPreCap PACT | 47,36 / 47,36 | 47,08 / 47,20 | 46,96 / 47,04 | 1e-2 (Rand) |
| TAT LSQ | 46,40 / 47,68 | 46,54 / 46,40 | 46,54 / 47,62 | 1e-3 |
| TAT PACT | 46,28 / 47,62 | 47,24 / 47,84 | 46,30 / 47,42 | 1e-2 (Rand) |

Die Unterschiede im Raster sind klein (meist unter 1 pp) und liegen im Bereich des Rauschens zwischen den beiden Seeds.

## Primärtest

| Vergleich | Median | W/L | p | Urteil |
|---|---|---|---|---|
| BPreCap-FT (PTQ) − TAT-LSQ (QAT) | **+3,01 pp** | 0/12 | 0,0005 | schlechter |

Einzeldifferenzen: 3,28 · 3,05 · 4,11 · 3,42 · 2,75 · 2,31 · 3,08 · 2,12 · 2,97 · 3,58 · 2,65 · 2,76.

## Sekundär (Holm über 6)

| Nr. | Vergleich | Median | W/L | Holm-p | Urteil |
|---|---|---|---|---|---|
| S1 | BPreCap-FT (PTQ) − TAT-PACT (QAT) | +3,00 | 0/12 | 0,003 | schlechter |
| S2 | BPreCap-LSQ − TAT-LSQ | −0,05 | 7/5 | 0,850 | unklar |
| S3 | BPreCap-PACT − TAT-PACT | −0,27 | 10/2 | 0,007 | besser |
| S4 | TAT-LSQ − TAT-FT (PTQ) | −14,87 | 12/0 | 0,003 | besser |
| S5 | BPreCap-LSQ − BPreCap-FT (PTQ) | −2,88 | 12/0 | 0,003 | besser |
| S6 | BPreCap-FT − TAT-FT (beide PTQ) | −11,85 | 12/0 | 0,003 | besser |

## Beschreibend (Mediane über 12 Seeds, Testfehler %)

| Form | Variante | 4 Bit | fp32 | 4-Bit-Preis | Bewegung der Grenzen | PTQ nach QAT |
|---|---|---|---|---|---|---|
| BPreCap | Basis (PTQ) | 49,95 | 45,71 | +4,24 | – | – |
| | FT (PTQ) | 50,85 | 45,79 | +5,07 | – | – |
| | LSQ (QAT) | 47,81 | 45,38 | +2,44 | 18 % | 48,60 |
| | PACT (QAT) | 47,70 | 45,50 | +2,20 | 32 % | 48,39 |
| TAT | Basis (PTQ) | 59,23 | 46,15 | +13,08 | – | – |
| | FT (PTQ) | 62,73 | 46,95 | +15,79 | – | – |
| | LSQ (QAT) | 47,85 | 45,44 | +2,41 | 12 % | 50,67 |
| | PACT (QAT) | 47,99 | 45,84 | +2,15 | 62 % | 52,86 |

- **Das Feintuning ohne Quantisierung verschlechtert PTQ bei beiden Formen:** BPreCap +0,9, TAT +3,5 pp gegen die Basis. Der Primärvergleich nutzt laut Präregistrierung FT, nicht die Basis. Gegen die Basis wäre der Abstand etwa 2,1 statt 3,0 pp. Das ist nur beschreibend, kein Test.
- **Nach QAT bleibt BPreCap robuster:** Mit der normalen PTQ-Kalibrierung statt der gelernten Quantisierer verliert BPreCap 0,8 pp, TAT 2,8 bis 4,9 pp.
- **fp32** ist nach dem Feintuning bei allen Varianten 45,4–47,0 %.

## Vorab-Erwartungen gegen Ergebnis

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Basis bitgleich (24 von 24) | 0,9 | ja |
| Primär „besser“ | 0,2 | nein |
| Primär „unklar“ | 0,35 | nein |
| Primär „schlechter“ | 0,45 | ja |
| S2 „besser“ (Hülle hilft auch unter QAT) | 0,5 | nein (unklar) |
| S4 „besser“ (QAT hilft TAT) | 0,85 | ja |
| S5 „besser“ (QAT hilft BPreCap) | 0,6 | ja |
| PACT-Grenzen bewegen sich um mehr als 1 % | 0,7 | ja (32 % / 62 %) |

## Deutung

1. **Die Obergrenze aus QAT liegt etwa 3 pp unter BPreCap mit PTQ.** Wer mit Quantisierung trainieren kann, braucht die feste Hülle für 4 Bit in diesem Rahmen nicht: TAT mit LSQ ist genauso gut wie BPreCap mit LSQ.
2. **Der Wert der Hülle ist PTQ.** Ohne Training mit Quantisierung ist BPreCap fast 12 pp besser als TAT und erreicht 4 Bit bis auf 3 pp am QAT-Niveau. Das ist ein klarer, aber engerer Anspruch als „ersetzt QAT“.
3. **Unter PACT hilft die Hülle ein wenig** (−0,27 pp, 10/2). Der Effekt ist klein, praktisch kaum relevant.
4. **Zum Reddit-Kommentar:** Die Antwort auf „QAT as an upper bound“ ist ehrlich: Die Obergrenze liegt 3 pp tiefer, und unter QAT verschwindet der Vorteil. Das gehört so in v0.4 bzw. in eine spätere Antwort, sobald es veröffentlicht ist.

## Grenzen

- QAT nur als Feintuning (20 % der Schritte) aus einem fp32-Netz, kein QAT von Anfang an.
- **Rand des Rasters:** Drei von vier gewählten LR_Q liegen am oberen Rand (1e-2). Die Unterschiede im Raster sind aber klein. Eine höhere LR_Q könnte QAT noch etwas besser machen, das würde den Abstand eher vergrößern.
- Die LR der Gewichte im Feintuning ist fest (0,1 × Spitze).
- **PACT:** asymmetrisch mit zwei Grenzen an den Eingängen der Linearschichten, nicht das Original hinter ReLU.
- **LSQ:** Nullpunkt kontinuierlich, nur STE.
- **Rahmen:** nur MLPs, CIFAR-10, d100, TAT-Rezept, simulierte 4-Bit-Aktivierungen mit int8-Gewichten. Gemessen wird am Ende des Feintunings.

Sicherung: `../../sicherung/v3_2026-11-03/` mit `SHA256.txt`.
