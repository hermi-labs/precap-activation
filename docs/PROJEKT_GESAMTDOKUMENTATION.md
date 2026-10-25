# CapExp / PreCap – Gesamtdokumentation des Projekts

Von der ersten Idee (14.10.2026) bis V18 und dem Start von 3D-V1 (25.10.2026)

Autor der Idee und der Funktion: Aaron Schmitt. Durchführung der Läufe auf eigenem MacBook Pro (M1 Pro), Auswertung mit KI-Unterstützung.
Stand dieses Dokuments: 25.10.2026, ca. 21:00 Uhr.

---

## 1. Kurzfassung

- **Die Idee:** Eine Aktivierungsfunktion, die exponentiell wächst, solange das harmlos ist, und ab einer Schwelle \(c\) linear mit fester Steigung \(e^c\) weiterläuft. Ziel war, die Ausdrucksstärke exponentiellen Wachstums zu nutzen, ohne dass Netze explodieren.
- **Zwei Funktionen wurden untersucht:**
  - CapExp (die Urform),
  - PreCap (CapExp mit vorgeschalteter, weich begrenzender tanh-Stufe).
- **Was in 11 Tagen geschah:** rund 40 Versionen in 1D und 2D, mit Präregistrierung, eingefrorenen Hashes, getrennten Tuning- und Bestätigungs-Seeds, Holm-Korrektur und Integritätsprüfungen. Allein V10 bis V18 umfassen über 12.000 Trainingsläufe.
- **Ergebnis bis heute:**
  1. Bei gleicher, fester Lernrate (0,001) war PreCap in residualen Netzen über viele Aufgaben, Tiefen und Seeds klar besser als SiLU und GELU, oft um den Faktor 2 bis 30.
  2. Bekommt jede Aktivierung ihre eigene, optimal gewählte Lernrate (V18), verschwindet dieser Vorteil in 2D. Es gab 0 von 30 Tests mit PreCap besser und 2 mit PreCap schlechter.
  3. Der Hauptgrund: PreCap hat am Ursprung Steigung 1, SiLU und GELU nur 0,5. Bei gleicher Lernrate lernt PreCap dadurch effektiv schneller.
  4. Echte, belegte Eigenschaft: Die Begrenzung hält sehr tiefe, ungedämpfte Netze (128 Schichten) numerisch trainierbar, wo SiLU und GELU divergieren (V17).
- **Offen:** Bringt das reine CapExp mit dauerhafter Steigung mehr als PreCap? Und ändert LayerNorm das Bild? Das prüft 3D-V1, gestartet am 25.10.2026 um 20:38 Uhr.

---

## 2. Die Funktionen

### CapExp (Urform, 14.10.2026)

\[
\mathrm{CapExp}_c(x)=
\begin{cases}
e^x-1, & x\le c\\
(e^c-1)+e^c\,(x-c), & x>c
\end{cases}
\]

- Wert und erste Ableitung sind bei \(c\) stetig.
- Steigung bei 0: 1. Oberhalb von \(c\) bleibt sie dauerhaft \(e^c\) (bei c = 1 also 2,72).
- Negativ nähert sich die Funktion −1 an (wie ELU).

### PreCap (ab 16.10.2026)

\[
z=a\,x,\quad u=z+\gamma z^2,\quad v=s\tanh\!\left(\frac{u}{s}\right),\quad \mathrm{PreCap}(x)=\mathrm{CapExp}_c(v)
\]

- 2D-Referenz `precap_base`: c = 1, a = 1, γ = 0,01, s = 1,5. Die 1D-Referenz nutzte a = 1,25.
- Gedacht war die Kette so: γz² gibt großen Signalen mehr Schub, das tanh bremst weich ab, und CapExp liefert danach den exponentiell-linearen Mechanismus (aus `1D/v1-6/precap_notes.txt`).
- **Wichtige Eigenschaft (nachgerechnet am 25.10.):** Das tanh deckelt den Eingang von CapExp bei 1,5. Damit ist PreCap nach oben beschränkt, das Maximum liegt bei etwa 3,08.

| x | 0 | 0,5 | 1 | 1,5 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|---|
| Steigung PreCap | 1,00 | 1,47 | 1,61 | 1,15 | 0,66 | 0,18 | 0,01 |
| Steigung CapExp (c = 1) | 1,00 | 1,65 | 2,72 | 2,72 | 2,72 | 2,72 | 2,72 |

Die "dauerhafte Steigung" aus der Ursprungsidee steckt also nur in CapExp. PreCap sättigt.

- **Erratum (20.10.):** Ein früheres Spezifikationsdokument (`2D/v2_holdout/docs/PreCap_Urformel_Spezifikation.md`) beschrieb fälschlich eine Softplus-Form. Alle Runner implementieren die tanh+CapExp-Form oben. Das wurde in jedem Runner per Selbsttest geprüft, die Abweichung lag unter 1e-14. Siehe `2D/ERRATUM_Formel_Spezifikation.md`.

---

## 3. Chronologie

### Phase 0 – Die Idee (14.10.2026, abends)

- **Ausgangsgedanke:** Lineare Bausteine sind "schnell am Limit". Gesucht war eine Funktion zwischen der Stabilität von linear und der Stärke von exponentiell, die stabil bleibt.
- **Daraus entstand:** exponentiell bis zur Schwelle \(c\), danach tangential linear. Das ist CapExp.
- **Erster Test `script.py`:** 1D-Regression sin(3x)+0,3x², MLP mit LayerNorm, 800 Schritte.
  - Bei 8 Schichten hatte CapExp den kleinsten Testfehler: 0,043 gegenüber ReLU 0,123 und GELU 0,107.
  - Die getestete Mischvariante "Blend" war klar schlechter und wurde verworfen.

### Phase 1 – 1D-Erkundung (15.–17.10.2026, `1D/v1-6`, `v7`, `v8`)

- **Erste Sweeps (v2/v3), ohne LayerNorm:**
  - CapExp mit c = 1 war bei 8 und 16 Schichten deutlich besser als ReLU und GELU.
  - Bei 16 Schichten lag der mittlere Testverlust bei etwa 0,021 gegenüber 0,134 (ReLU) und 0,148 (GELU).
  - Ab c = 1,5 explodierten Verluste und Gradienten.
  - Mit LayerNorm war CapExp nur etwa gleichauf mit SiLU und GELU.
- **Sweep v4, Fairness-Gegentest:**
  - Mit Residualverbindungen holten SiLU und GELU auf oder gewannen.
  - Mit He-Initialisierung wurde CapExp instabil (NaN bei 32 Schichten).
  - Lehre: CapExp braucht kleine Gewichte und kleines c.
- **v5/v6:**
  - Residual-Varianten (`res_zero`, `res_inv` mit Skalierung 1/√Tiefe) und kleine Initialisierung.
  - Daraus entstand PreCap als "geregelte" Version (Notizen in `1D/v1-6/v6_notes.txt`).
- **v7/v8:** Feintuning der Parameter. Die Rohdaten liegen in den Ordnern.

### Phase 2 – 1D-Bestätigung (17.–20.10.2026, `1D/v9` bis `v12_finale_1d`)

- **v9, v9_depth, v9b:**
  - Instrumentierte Bestätigung mit Lernkurven und Schichtdiagnostik.
  - Tiefen-Stresstest bis 128 Schichten.
  - Fairness-Sweep mit 6480 Runs.
- **v10 (6360 Runs) und v10B (Holdout auf frischen Seeds):** Der Grundbefund repliziert.
- **v11:**
  - Die Varianten `precap_wg` und `hybrid_gate` brachten keinen robusten Mehrwert und wurden verworfen.
  - PreCap blieb die beste Variante.
- **v12 (1620 Runs, Regime: Basis, hohe LR, große Initialisierung):**
  - In `plain`-Netzen ist PreCap schlechter als SiLU, und der Nachteil wächst mit der Tiefe: Median d = +0,13 / +0,29 / +0,42 bei Tiefe 4 / 8 / 16.
  - In `res_inv` ist PreCap in 8 von 9 Zellen besser als SiLU und GELU. Auf Seed-Ebene gewinnt es 81 von 90 Einheiten gegen SiLU und 83 von 90 gegen GELU (p < 1e-15).
- **1D-Fazit:** PreCap ist kein allgemeiner Ersatz, aber in residualen Netzen klar stark. Ohne Residualpfad trägt der Vorteil nicht.

### Phase 3 – 2D (20.–25.10.2026, `2D/`)

**Gemeinsames Setup:**
- Synthetische 2D-Regression mit dem Residual-MLP `res_inv` (Breite 64, Skalierung 1/√Tiefe), Adam.
- **Feste Lernrate 0,001 bis einschließlich V17.**
- Metrik: \(d=\log_{10}(\mathrm{MSE}_{PreCap}/\mathrm{MSE}_{Ref})\). Negative Werte heißen: PreCap besser.

| Version | Frage | Umfang | Ergebnis |
|---|---|---|---|
| v1_pilot | erster Pilot | 135 Runs | ungültig: Implementierungsfehler in der Formel (Lehre: Selbsttest gegen unabhängige Referenz ist Pflicht) |
| v1b | Pilot mit korrigierter Formel | 135 Runs | PASS: 37/45 Siege gegen SiLU, 33/45 gegen GELU |
| v2 | Holdout, neue Seeds | 270 Runs | PASS: 83/90 gegen SiLU (p = 6,6e-18), 64/90 gegen GELU |
| v3 | große präregistrierte Replikation, 6 Aufgaben, Tiefe 4–64 | 900 Runs | Gate PASS (3/3 Kernaufgaben). Tiefe 4: d = −0,67 gegen SiLU; Tiefe 64: −0,04, also praktisch gleich |
| v4 | Mechanismus über die Zeit | Snapshots | Peaks mit PreCap meist kleiner; Gradienten uneinheitlich |
| v5 | Warum kippt Tiefe 64? | Diagnose | kein einfacher Gradienten-Crash; seed- und aufgabenabhängig |
| v6 | c-/s-Ablation | 105 Runs | kein belastbarer Treffer; Seed-Streuung dominiert |
| v7a / v7b | s = 2,25 gegen Basis, Tiefe × s | Seed-Erweiterung | s225 hilft auf radial_ripple, nicht allgemein |
| v8 | Aufgabenachse radial gegen ridge | – | s225-Vorteil eher bei radialen Aufgaben |
| v9 | Geometrie-Kontinuum | – | Zusammenhang nicht monoton; stärkster Effekt bei Mischgeometrie a = 0,25 |
| V10 | Mechanismus a = 0,25 gegen 0,75 | 20 Seeds | s225 hilft bei a = 0,25 (Verhältnis 0,52 / 0,27), nicht bei 0,75 |
| V11 | Bestätigung | 6 Zellen, 20 Seeds | teilweise bestätigt; Mechanismus (p99) nicht |
| V12 | Drehung oder Frequenz? | 480 Runs | weder noch belastbar (alle Holm n.s.) |
| V13 | Frequenztrend | 600 Runs | kein monotoner Trend; s225 in 9/10 Zellen besser als Basis |
| V14 | Übertragung auf 6 Aufgaben | 1440 Runs | s225 aufgabenspezifisch. Explorativ: PreCap schlägt SiLU auf 6/6 Aufgaben und GELU auf 4/6 (1200 Schritte) |
| V15 | Ausnahmezelle mit 100 Seeds | 200 Runs | Seed-Rauschen, kein echter Effekt |
| V16 | Tiefe 32/64/128, 3000 Schritte | 1440 Runs | Geomittel PreCap/SiLU: 0,43 (Tiefe 32), 0,89 (64), 1,03 (128). Vorteil schrumpft mit der Tiefe; Untertraining als Erklärung verworfen |
| V17 | Residualskalierung (inv_sqrt, inv_quart, constant) | 1440 Runs | Dämpfung hilft allen gleich. Konstante Skala, Tiefe 128: SiLU/GELU divergieren (MSE 4–28), PreCap bleibt trainierbar (0,003–0,017) |
| **V18** | **faire Lernrate je Aktivierung**, Tiefe 8/16/32 | 4050 Runs | **0/30 PreCap besser, 2/30 schlechter.** Bei fester LR 1e-3 kehrt der alte Vorteil zurück (Verhältnis 0,03–0,7) |

### Phase 4 – 3D (ab 25.10.2026, `3D/v1_transfer3d`)

- **Fragen:**
  - Hält ein Vorteil bei 3D-Eingaben?
  - Ist er mehr als Skalierung? Dafür gibt es die Kontrolle SiLU×2, die am Ursprung dieselbe Steigung 1 hat.
  - Ändert LayerNorm das Bild?
  - Hilft die dauerhafte Steigung des reinen CapExp?
- **Aufgaben:** smooth_bowl3d, gyroid, torus_sdf, box_sdf, spheres_union.
- **Aktivierungen:** precap_base, capexp_c1, silu, silu2x.
- **Varianten:** Tiefe 8 und 16, jeweils ohne und mit LayerNorm. Lernraten 1e-3, 3e-3 und 1e-2, jeweils pro Aktivierung gewählt.
- **Seeds:** 30000–30029 (10 Tuning, 20 Bestätigung).
- **Umfang:** 7200 Runs, 80 präregistrierte Tests mit Holm-Korrektur. Hashes eingefroren in `meta/frozen_sha256.txt`.
- **Schon vor dem Training gemessen:** Ohne LayerNorm liegen 3 % der Voraktivierungen über 0,5, mit LayerNorm 30 %. Mit LayerNorm arbeitet die Aktivierung also stärker in dem Bereich, in dem sie sich überhaupt von SiLU unterscheidet.

---

## 4. Was gesichert ist

1. **Bei gleicher Lernrate ist PreCap in residualen Netzen deutlich besser als SiLU und GELU.**
   - 1D: v10–v12.
   - 2D: v1b–v3 und V14–V16.
   - Viele Holdout-Replikationen, p-Werte bis unter 1e-15.
   - Diese Aussage ist korrekt, gilt aber nur unter dieser Bedingung.
2. **Mit fairem Lernraten-Tuning verschwindet dieser Vorteil in 2D (V18).** SiLU und GELU wählten dabei durchweg den oberen Rand des Lernraten-Rasters. Mit noch höheren Lernraten würden sie eher weiter aufholen.
3. **Die Erklärung ist die Skalierung.** Taylor-Entwicklung:
   - PreCap ≈ x + 0,51x²
   - SiLU ≈ x/2 + x²/4
   - GELU ≈ x/2 + 0,40x²

   Bei kleinen Werten ist PreCap fast ein mit 2 skaliertes SiLU, und genau dort liegen 70–97 % aller Voraktivierungen.
4. **Der Begrenzer wirkt als Stabilitätsgarant.** Ungedämpfte 128-Schichten-Netze bleiben mit PreCap trainierbar, SiLU und GELU divergieren dort (V17).
5. **Mechanismus-Hinweis (V18):** Wo mehr Voraktivierungen über 0,5 liegen, schneidet PreCap relativ besser ab (ρ = −0,52, p = 0,023). Das ist teilweise mit der Tiefe vermischt.

## 5. Was widerlegt oder verworfen wurde

- "PreCap ist allgemein besser": nein.
- Untertraining als Grund für das Schrumpfen des Vorteils in der Tiefe (V16): nein.
- "Die Residual-Dämpfung nimmt PreCap den Vorteil" (V17): nein, die Dämpfung wirkt auf alle gleich.
- Drehung oder Frequenz erklären die s-Wirkung (V12/V13): nein.
- Varianten Blend, `precap_wg` und `hybrid_gate`: verworfen.

## 6. Methodik (für den Nachweis)

- **Vor jedem Lauf schriftlich festgelegt:**
  - Präregistrierung mit Hypothesen, Tests und Entscheidungsregeln,
  - SHA-256-Hashes aller Skripte.
- **Fairness:**
  - getrennte Tuning- und Bestätigungs-Seeds,
  - Seed-Bereiche über alle Studien disjunkt,
  - Holdout-Replikationen.
- **Statistik:**
  - gepaarte Vergleiche auf denselben Seeds, Wilcoxon- bzw. Vorzeichentests,
  - Holm-Korrektur,
  - Bootstrap-Konfidenzintervalle.
- **Vor jedem Start geprüft:**
  - Formel-Selbsttest gegen eine unabhängige Referenz,
  - Smoke-Tests,
  - Trockenlauf der Analyse mit künstlich eingebautem Effekt.
- **Nach jedem Lauf:**
  - Integritätsprüfung (Zeilenzahl, Eindeutigkeit, nicht-endliche Werte, vollständige Paare).
- **Negative Ergebnisse und Pannen werden dokumentiert:** etwa der ungültige 2D-Pilot, das Formel-Erratum und der Doppelstart bei V18 mit 14.416 bitgenau identischen Duplikatzeilen. Diese wurden nach Sicherung entfernt, und die Analyse wurde neu ausgeführt.

## 7. Einschränkungen

- Nur synthetische Regressionsaufgaben in 1D, 2D und 3D. Keine realen Datensätze, keine CNNs oder Transformer.
- Kleine Netze (Breite 64), ein Optimierer (Adam).
- Bis V17 lief alles mit fester Lernrate. Alle früheren "PreCap besser"-Aussagen sind deshalb an diese Bedingung gebunden.

## 8. Was ein Einsatz in echten Netzen bräuchte

1. Ein fairer Vergleich mit Lernraten-Tuning pro Aktivierung (wie V18 und 3D-V1) auf einem realen Datensatz, zum Beispiel einer kleinen Bild- oder Tabellenaufgabe.
2. Immer die Kontrolle SiLU×2 bzw. eine gleich skalierte Referenz.
3. Mit und ohne Normalisierung testen.
4. Am ehesten nützlich dürfte die Stabilitätseigenschaft sein: tiefe Netze ohne Normalisierung oder mit aggressiver Lernrate.

## 9. Ordner und Belege

| Bereich | Pfad |
|---|---|
| Erstes Skript | `1D/v1-6/script.py` (14.10.2026) |
| PreCap-Idee | `1D/v1-6/precap_notes.txt`, `1D/v1-6/v6_notes.txt` |
| 1D-Gesamtbefund | `Notizen Zwischenstände/capexp_roadmap/CapExp_1D_Gesamterkenntnisse.md` |
| Forschungsagenda | `CAPEXP_FORSCHUNGSAGENDA.md` |
| 2D-Gesamtdokumentation | `2D/DOKUMENTATION_2D_Gesamt.md` |
| Formel-Erratum | `2D/ERRATUM_Formel_Spezifikation.md` |
| V17 ausführlich | `2D/v17_residual_scale/analysis/V17_Auswertung_ausfuehrlich.md` |
| V18 ausführlich | `2D/v18_lr_shallow/analysis/V18_Auswertung_ausfuehrlich.md` |
| 3D-V1 Präregistrierung | `3D/v1_transfer3d/docs/V1_3D_Praeregistrierung.md` |
| Hashes | `*/meta/frozen_sha256.txt` je Studie |

---

## 10. Persönliches Fazit

Die Ursprungsidee war, exponentielle Stärke zu nutzen und sie so zu begrenzen, dass nichts explodiert. Diese Idee wurde nicht widerlegt, sondern präzisiert. Die Begrenzung funktioniert und hält sehr tiefe Netze stabil.

Der große Leistungsvorteil, der anfangs sichtbar war, hing aber an der festen Lernrate. Das wurde erst durch einen bewusst harten, fairen Test (V18) sichtbar.

Dass dieser Test gemacht und das Ergebnis ehrlich festgehalten wurde, ist der eigentliche Wert dieser Dokumentation.

---

## 11. Einordnung in bestehende Arbeiten (Recherche 25.10.2026)

- **CapExp ist mathematisch exakt ein verschobenes und skaliertes ELU** (ELU mit α = 1, Clevert et al. 2015, [arXiv:1511.07289](https://arxiv.org/pdf/1511.07289v3)):

  \[
  \mathrm{CapExp}_c(x)=e^c\cdot\mathrm{ELU}(x-c)+(e^c-1)
  \]

  - Numerisch geprüft für c = 0,3 / 0,5 / 1,0 / 1,5 auf [−8, 8]. Die maximale Abweichung liegt unter 4e-15.
  - Folge: In einem Netz mit Bias-Termen und nachfolgender linearer Schicht lassen sich die Verschiebung und der Offset in die Biases verschieben, der Faktor \(e^c\) in die Gewichte. CapExp kann also dieselben Funktionen darstellen wie ELU. Unterschiede entstehen nur über Initialisierung und effektive Lernrate. Das passt zum V18-Befund.
- **PreCap ist durch die tanh-Vorstufe dagegen wirklich eine andere Funktion:** glatt, mit Steigung 1 am Ursprung und beidseitig beschränkt. Eine identische Funktion habe ich nicht gefunden. Es gibt verwandte Ideen:
  - TERELU, nach oben weich begrenzt ([arXiv:2006.02797](https://arxiv.org/pdf/2006.02797)),
  - TanhExp, Kombination aus tanh und exp, und weitere aus der Übersicht über 400 Aktivierungsfunktionen ([Kunc & Kléma 2024](https://arxiv.org/abs/2402.09092)),
  - Soft Exponential, Interpolation zwischen log, linear und exp ([Godfrey & Gashler](https://www.arxiv.org/abs/1602.01321)).
- **Konsequenz für die Weiterarbeit:** Die neue Aussage muss sich auf PreCap beziehen, also auf die beschränkte, glatte Form. ELU gehört künftig als Pflicht-Vergleich in jeden Test.
