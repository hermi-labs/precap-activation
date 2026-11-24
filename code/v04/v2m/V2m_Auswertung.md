# V2m – Auswertung: Eingriff an der Form (frühe Sättigung der Hülle)

Stand: 23.11.2026, 14:16. Präregistrierung: `V2m_Praeregistrierung.md`, Freeze 2026-11-23T06:51:56+0100. Hash-Prüfung vor und nach der Analyse: 26/26 OK. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Ausgewertet mit Claude. Lauf: 07:03–14:15, ein Worker (`v2mw0`), am Netzteil, Wächter `v2mwache`.

Befehle:
```
(cd code/v04/v2m && shasum -a 256 -c meta/frozen_sha256.txt)      # 26/26 OK, vorher und nachher
python -I code/v04/v2m/v2m_analyze.py "$PWD/code/v04/v2m"
```
Ausgabe: `analysis/v2m_result.json` und `analysis_ausgabe.txt`.

## Urteil

- **E1 „bestätigt“:** Bei früher Sättigung (Knie ρ = 0,4) kommt ein Abschneidefehler gegebener Größe stärker an den Logits an als bei BPreCap (ρ = 0,8). Median-Quotient ḡ^C 1,13; 8 von 9 Seeds; Holm-p 0,016.
- **E2 „widerlegt“:** Das Abschneiden in allen Schichten zugleich stört die Logits bei früher Sättigung **schwächer**. Median-Quotient D^C 0,90; 8 von 9 Seeds kleiner; Holm-p 0,016.
- **Folge nach der Präregistrierung** (Zeile „einer oder beide widerlegt“): Die Lesart aus v0.4 („BPreCap sättigt spät und steil, deshalb schadet Abschneiden weniger“) ist für die Hülle so nicht haltbar. Sie wird in v0.4 umgeschrieben.
- Genauer: Die späte Sättigung dämpft die Weitergabe eines einzelnen Abschneidefehlers, wie vermutet. Sie macht aber das gesamte Abschneiden an den Logits nicht kleiner und ändert den 4-Bit-Fehler nicht (S1–S3).

## Primärtests (mse4, 9 Paare, exakter Wilcoxon zweiseitig, Holm über 2)

| Test | Größe | Median log q | q | + / − | Holm-p | Urteil |
|---|---|---|---|---|---|---|
| E1 | ḡ^C(ρ 0,4) / ḡ^C(ρ 0,8) | +0,121 | 1,13 | 8 / 1 | 0,016 | bestätigt |
| E2 | D^C(ρ 0,4) / D^C(ρ 0,8) | −0,107 | 0,90 | 1 / 8 | 0,016 | widerlegt |

## Sekundärtests (zweiseitig, Holm über 4)

| Test | Größe | Median | + / − | Holm-p | Urteil |
|---|---|---|---|---|---|
| S1 | 4-Bit-Preis, ρ 0,4 − ρ 0,8 | −0,39 pp | 4 / 5 | 1,0 | kein klarer Unterschied |
| S2 | Testfehler w8a4a_best, ρ 0,4 − ρ 0,8 | −0,68 pp | 3 / 6 | 1,0 | kein klarer Unterschied |
| S3 | Testfehler fp32, ρ 0,4 − ρ 0,8 | −0,05 pp | 4 / 5 | 1,0 | kein klarer Unterschied |
| S4 | ḡ^C(ρ 0,4) / ḡ^C(ctanh) | q 0,80 | 0 / 9 | 0,016 | Unterschied (ρ 0,4 kleiner) |

S3 zeigt: Die Variante lernt gleich gut. Die fehlende Neu-Abstimmung der Lernrate verzerrt den Vergleich also nicht sichtbar (Grenze aus der Präregistrierung).

## Beschreibend (präregistriert)

Mediane bei mse4 (9 Seeds):

| Größe | ρ 0,4 (V2m) | ρ 0,8 (BPreCap) | ctanh |
|---|---|---|---|
| ḡ^C (Weitergabe Abschneiden) | 0,560 | 0,495 | 0,648 |
| D^C (Abschneiden, alle Schichten) | 0,104 | 0,120 | 0,151 |
| r̄^C (Größe des Abschneidefehlers) | 0,0246 | 0,0224 | 0,0238 |
| D^Q (voller 4-Bit-Fehler) | 0,271 | 0,274 | 0,303 |
| ḡ^R (Weitergabe Runden) | 0,170 | 0,173 | 0,157 |
| κ (linearer Kreuzterm) | 0,96 | 1,16 | 0,77 |
| Superposition D^Q / √Σ d_i² | 0,98 | 1,07 | 1,10 |
| Anteil abgeschnittener Werte (Median über Schichten) | 4,2 % | 3,9 % | 3,3 % |
| Testfehler fp32 | 42,72 % | 42,72 % | – |
| Testfehler w8a4a_best | 47,62 % | 48,09 % | – |

- **Lage zwischen BPreCap und ctanh** (t = 0 wie BPreCap, t = 1 wie ctanh, Median über Seeds):
  - ḡ^C: 0,37. Die frühe Sättigung verschiebt die Weitergabe eines einzelnen Abschneidefehlers um etwa ein Drittel des Wegs zu ctanh.
  - D^C: −0,62. Die Variante liegt noch weiter von ctanh weg als BPreCap.
  - ḡ^Q: 0,56. D^Q: 0,34.
- **log-Quotienten gegen ρ 0,8:**
  - r̄^C +0,086 (die Abschneidefehler werden etwas größer);
  - ḡ^C +0,121;
  - D^C −0,107;
  - D^R −0,006 und ḡ^R −0,003 (am Runden ändert sich praktisch nichts).
- **Differenzen gegen ρ 0,8:** κ −0,32, ν −0,03, Superposition −0,06.
- Alle 9 Netze wählten mse4 als beste Kalibrierung.

Je Seed:

| Seed | ḡ^C ρ0,4 / ρ0,8 / ctanh | D^C ρ0,4 / ρ0,8 | κ ρ0,4 / ρ0,8 | w8a4a_best ρ0,4 / ρ0,8 |
|---|---|---|---|---|
| 42353 | 0,563 / 0,494 / 0,620 | 0,099 / 0,121 | 0,66 / 1,16 | 45,39 / 46,73 % |
| 42354 | 0,592 / 0,524 / 0,753 | 0,110 / 0,109 | 0,94 / 1,05 | 47,76 / 48,88 % |
| 42355 | 0,568 / 0,540 / 0,760 | 0,117 / 0,127 | 1,16 / 0,88 | 47,28 / 47,27 % |
| 42356 | 0,545 / 0,495 / 0,643 | 0,114 / 0,118 | 0,99 / 1,40 | 46,90 / 48,09 % |
| 42357 | 0,504 / 0,489 / 0,637 | 0,092 / 0,123 | 0,96 / 1,85 | 48,34 / 49,86 % |
| 42358 | 0,560 / 0,493 / 0,583 | 0,124 / 0,138 | 1,08 / 0,97 | 48,17 / 48,61 % |
| 42359 | 0,560 / 0,487 / 0,703 | 0,100 / 0,106 | 0,90 / 1,73 | 46,74 / 46,05 % |
| 42360 | 0,589 / 0,499 / 0,648 | 0,104 / 0,141 | 0,80 / 1,13 | 48,00 / 48,68 % |
| 42361 | 0,500 / 0,512 / 0,697 | 0,080 / 0,101 | 0,97 / 1,28 | 47,62 / 46,09 % |

## Einordnung

- **Warum E1 und E2 auseinanderlaufen.** ḡ^C misst, wie stark ein Abschneidefehler **einer** Schicht an den Logits ankommt (geometrisches Mittel über die Schichten). D^C misst das Abschneiden **aller** Schichten zugleich. Bei früher Sättigung wird der Einzelbeitrag größer: Die Fehler sind etwas größer und werden stärker weitergegeben. Die Beiträge zeigen aber weniger in dieselbe Richtung (κ 0,96 statt 1,16, Superposition 0,98 statt 1,07). Unterm Strich ist die Gesamtstörung kleiner.
- Das passt zu dem, was v0.4 in 5.5 schon festhält: Die Einzelbeiträge addieren sich nicht linear (ν ≈ 1). Aussagen über das Ganze müssen auf den Messungen aller Schichten beruhen.
- **Was das für die Lesart heißt.** Die späte, steile Sättigung der Hülle trägt einen Teil der schwächeren Weitergabe einzelner Abschneidefehler: etwa ein Drittel des Abstands zu ctanh (t = 0,37). Für das gesamte Abschneiden an den Logits und für den 4-Bit-Fehler ist sie aber nicht der Hebel. Eine früher sättigende Hülle ist bei 4 Bit gleich gut (S1–S3), und ihre gesamte Abschneidestörung ist sogar kleiner.
- Der größte Teil des Abstands zu ctanh bleibt bestehen (S4: ρ 0,4 immer noch 9/9 unter ctanh). Er sitzt also eher im Kern (tanh im Exponenten, Asymmetrie α) oder in der Verteilung nach dem Training. Beides ist nicht geprüft.
- **Vorab-Wahrscheinlichkeiten gegen Ausgang:**

  | Aussage | vorab | eingetreten |
  |---|---|---|
  | E1 bestätigt | 0,45 | ja |
  | E2 bestätigt | 0,40 | nein |
  | E2 widerlegt | 0,10 | ja |
  | S3 fp32 anders | 0,30 | nein |
  | alle 9 ok | 0,92 | ja |

  Das „widerlegt“ bei E2 war unerwartet.
- **Bezug zu früheren Studien:** V2k (dieselben Seeds, dieselbe Messung) und V2f.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Zeilen | 9/9 ok, run_id eindeutig, 0 Ausfälle |
| nicht-endliche Werte | 0 |
| Tracebacks / FEHLER im Log | 0 |
| Gewichte, Messungen | 9 `gewichte/*.npz`, 9 `mess/*.json` und 9 `*_delta.npz` |
| Referenzen | V2k (mse4, beide Formen) und V2a2 für alle 9 Seeds vollständig |
| Hashes | 26/26 OK vor und nach der Analyse; Vorgänger V2a2, V2c, V2d, V2e, V2f, V2k beim Start OK |
| Selbsttest | `alle_ok: true`, Form und 120 Trainingsschritte bitgleich mit den Original-Konstanten |
| Auswertungstest | AUSWERTUNG OK (13/13) |
| Netzteil | durchgehend, Wächter nicht ausgelöst |

## Grenzen

- Eine Variante (ρ = 0,4), keine spätere Sättigung (ρ = 0,95), kein Verlauf.
- Verglichen werden trainierte Netze. Einzelweitergabe und Kohärenz hängen auch an den gelernten Gewichten, nicht nur an der Form.
- Nur mse4, nur Aktivierungen, ein Netz, ein Datensatz, ein Rechner.
- Die Erklärung über κ und die Superposition ist beschreibend und nicht getestet.

## Nächster Schritt

- **v0.4, Abschnitt 5.5:** Der Satz „a plausible reading is that BPreCap saturates late and steeply …“ aus dem Entwurf wird durch das Ergebnis von V2m ersetzt.
- **Später (nicht für v0.4):** ein Eingriff am Kern, also α = 1 oder ohne tanh im Exponenten, mit derselben Messung.
