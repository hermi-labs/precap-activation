# V2a-1 – Auswertung (privat, mit Claude, 04.11.2026, 18:25)

Strikt nach `V2a1_Praeregistrierung.md`. PreCap, die Tiefenfamilie und die Hülle sind Aarons Ideen, die Auswertung habe ich mit Claude gemacht. Lauf: 13:45–18:22, 2 Worker (`v2a1w0`, `v2a1w1`), am Netzteil.

## Kurzfassung

- **Primär: Erfolg.** BPreCap − SELU, Test-Fehler w8a4a_max: **−7,91 pp, 12/0, p = 0,0005, „besser“.**
- **Alle drei Sekundärtests „besser“** (Holm):
  - S1 w8a4a_best: −6,17 pp, 12/0, Holm-p 0,0015;
  - S2 fp32: −2,68 pp, 11/1, Holm-p 0,0029;
  - S3 w8a8: −2,71 pp, 11/1, Holm-p 0,0029.
- **Deutung nach der Vorab-Regel:** „Erfolg“. Der 4-Bit-Vorteil gegen SELU trägt vom MLP ins CNN ohne Normierung (L = 32, einfaches Training). S2 ist nicht „schlechter“, also kostet der Vorteil keine fp32-Güte. BPreCap ist in diesem Aufbau auch in fp32 vorn.
- **Wichtige Einschränkung:** Ein Teil des 4-Bit-Abstands ist der fp32-Abstand. Der reine 4-Bit-Preis (w8a4a_max − fp32) ist bei BPreCap +1,9 pp, bei SELU +7,4 pp. Je Seed ist der Preis bei BPreCap um median 5,9 pp kleiner, 12/12 (beschreibend, kein präregistrierter Test).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Zeilen | 24/24 ok, 0 Ausfälle, 0 fehlend |
| Tracebacks | 0 in `w0.log`, `w1.log`, `smoke.log` |
| Hashes nach dem Lauf | `shasum -a 256 -c meta/frozen_sha256.txt`: 19/19 OK (Code, Pilot-Vergleichscode, Konstanten, CIFAR-Dateien) |
| Kontrolle vor dem Start | Mechanik bei 20 Schritten bitgleich zum Pilot-Code; volle Länge nicht geprüft (siehe Grenzen) |
| Sicherung | `sicherung/v2a1_2026-11-04/` mit SHA256.txt |

Befehle: `~/PyCharmMiscProject/.venv/bin/python -I v2a1_analyze.py` und `shasum -a 256 -c meta/frozen_sha256.txt` im Studienordner, 18:22.

## Tests

| Test | Größe | Median BPreCap − SELU | W/L | p (Holm) | Urteil |
|---|---|---|---|---|---|
| **Primär** | w8a4a_max | **−7,91 pp** | 12/0 | 0,0005 | **besser** |
| S1 | w8a4a_best | −6,17 | 12/0 | 0,0015 | besser |
| S2 | fp32 | −2,68 | 11/1 | 0,0029 | besser |
| S3 | w8a8 | −2,71 | 11/1 | 0,0029 | besser |

Primär, Differenzen je Seed (pp, 42330–42341): −8,09, −8,96, −7,36, −5,05, −8,85, −7,68, −5,26, −11,76, −10,85, −7,72, −12,11, −6,94. Die eine Niederlage in S2 und S3 ist Seed 42333 (+1,21 bzw. +1,29 pp).

## Beschreibend (Mediane über 12 Seeds, Test-Satz)

| Form | LR | Val fp32 | fp32 | w8a8 | w8a4a_max | w8a4a_p999 | w8a4a_p9999 | w8a4a_best | 4-Bit-Preis (max − fp32) | Bereich max (unten / oben) |
|---|---|---|---|---|---|---|---|---|---|---|
| BPreCap | 1e-4 | 44,1 % | 45,0 % | 45,0 % | 46,9 % | 47,0 % | 47,1 % | 47,2 % | +1,9 pp | −1,86 / 2,99 |
| SELU | 3e-5 | 46,5 % | 47,4 % | 47,5 % | 54,8 % | 52,2 % | 53,5 % | 52,5 % | +7,4 pp | −1,75 / 6,34 |

- **Spannweite über die Seeds:** BPreCap fp32 41,8–47,6 %, w8a4a_max 44,7–49,9 %; SELU fp32 46,4–49,2 %, w8a4a_max 54,2–58,7 %. Die Verteilungen bei 4 Bit überlappen nicht.
- **Bereich:** Die obere Grenze liegt bei BPreCap in allen Läufen bei 2,99 (Hülle), bei SELU zwischen 6,1 und 7,0. Das ist derselbe Mechanismus wie in den MLP-Studien.
- **Gewählte Kalibrierung (w8a4a_best):** BPreCap max 2, p999 8, p9999 2; SELU p999 8, p9999 4, nie max. Für SELU hilft das Abschneiden von Ausreißern (54,8 → 52,2 % mit p999), für BPreCap ändert es fast nichts, weil die Hülle den Bereich schon begrenzt.
- **Auffällig:** Die Wahl auf den Kalibrierbildern trifft nicht immer die beste Kalibrierung auf dem Test-Satz. Bei BPreCap ist w8a4a_best im Median (47,2 %) etwas schlechter als w8a4a_max (46,9 %). Die Unterschiede sind dort aber klein (≤ 0,3 pp).
- **Laufzeit je Lauf** (2 Worker): BPreCap 1495 s, SELU 1126 s.

## Kritische Prüfung

1. **fp32-Abstand.** BPreCap ist auch in fp32 um 2,7 pp besser. Das war vorab nur mit 0,25 erwartet. Mögliche Gründe:
   - Die LR von SELU (3e-5) stammt aus einem Pilot-Seed am Val-Satz. Bei konstanter LR und nur 1720 Schritten könnte SELU untertrainiert sein (Trainingsverlust im Median 1,29 gegen 1,23).
   - In Pilot 4 lag SELU bei L = 32 und LR 3e-5 ähnlich (fp32 45,8/47,7 %), bei 1e-4 schlechter. Es spricht also nichts für eine klar falsch gewählte LR, ein volles Tuning gab es aber nicht.
   - **Deshalb:** Der 4-Bit-Abstand von 7,9 pp ist nicht allein ein Quantisierungseffekt. Der Preisvergleich (5,9 pp kleiner, 12/12) zeigt aber, dass der größere Teil davon aus der Quantisierung kommt.
2. **Größe des Effekts:** im Pilot −8 bis −13 pp, hier −7,9 pp. Das passt.
3. **Gegen beste Kalibrierung:** Auch mit Ausreißer-Abschneiden (w8a4a_best) bleibt SELU um 6,2 pp hinten. Der Vorteil ist also nicht nur ein Artefakt der min/max-Kalibrierung.
4. **Was V2a-1 nicht zeigt:**
   - L = 50 und andere Tiefen;
   - Training mit AdamW, Kosinus und Tuning, das den SELU-Bereich bremsen könnte;
   - QAT (V3 zeigt, dass QAT den Hüllen-Vorteil im MLP aufhebt);
   - Augmentierung, Pooling, breitere Netze, andere Datensätze;
   - die Kontrolle über die volle Länge gegen die Pilot-Werte.

## Vorab-Erwartung gegen Ergebnis

| Größe | Erwartung | eingetroffen |
|---|---|---|
| Primär „besser“ | 0,75 | ja |
| S1 „besser“ | 0,65 | ja |
| S2 „unklar“ | 0,6 | nein („besser“, vorab 0,25) |
| S3 „unklar“ | 0,8 | nein („besser“) |

S3 kam anders als erwartet. w8a8 folgt fast genau fp32 (Preis ≤ 0,15 pp bei beiden), also ist S3 im Kern derselbe Befund wie S2.

## Grenzen

- Ein Netz: L = 32, 32 Kanäle, kein Pooling, keine Augmentierung, nur CIFAR-10, nur PTQ.
- Einfaches Regime: Adam mit konstanter LR, 1720 Schritte, kein WD, LR aus einem Pilot-Seed. TAT ist nicht dabei (Pilot 4: −44 pp).
- Die Gleichheit mit dem Pilot-Code ist nur über 20 Schritte geprüft, nicht über die volle Länge.
- Die Fehlerraten (45–55 %) sind für CIFAR-10 hoch. Das Netz ist ein Prüfstand für tiefe Netze ohne Normierung, kein Leistungsmodell.

## Folgen

- **Für v0.4 (Spur A, A1):** Ein erster präregistrierter CNN-Befund. Er beantwortet den Einwand „nur MLPs“ teilweise: bei L = 32 und einfachem Training ist BPreCap bei 4 Bit klar vor SELU und in fp32 nicht schlechter.
- **Nächster Schritt A2 (große V2a):** L = 32 und 50, gemeinsames LR-Raster, WD {0; 0,3}, Seeds ab 42350. Dort wird sich zeigen, ob der fp32-Abstand mit Tuning verschwindet und ob der 4-Bit-Abstand bleibt. Der Entwurf `V2a_Praeregistrierung_Entwurf.md` passt schon dazu. Die Vorab-Wahrscheinlichkeit für den Primärtest dort (0,55) würde ich nach diesem Ergebnis auf etwa 0,65 anheben, vor dem Freeze.
