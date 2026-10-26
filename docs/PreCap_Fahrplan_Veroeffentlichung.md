# Fahrplan: PreCap belastbar testen und veröffentlichen

Stand: 25.10.2026. Ziel ist eine eigenständige, nachprüfbare Veröffentlichung von PreCap. Sie besteht aus einem PDF im Stil eines Papers, Grafiken, Code und Daten unter einer DOI. Die Matrix-Idee wird erst danach als eigene Variante untersucht.

---

## Grundregeln

1. **Erst PreCap allein, dann Varianten.** Wird beides gleichzeitig getestet, ist später nicht klar, woher ein Effekt kommt.
2. **Jede Studie wird vorher festgelegt.** Präregistrierung, eingefrorene Hashes, Tuning-Seeds getrennt von Bestätigungs-Seeds. So wurde bisher gearbeitet, und so bleibt es.
3. **Jede Aktivierung bekommt ihre eigene beste Lernrate.** Das ist die zentrale Lehre aus V18.
4. **Die Erfolgskriterien stehen vorher fest.** Auch ein negatives Ergebnis wird veröffentlicht.

---

## Phase 0 – 3D-V1 abschließen (heute Nacht)

- Lauf fertigstellen, Integrität prüfen, präregistrierte Analyse durchführen.
- Zusätzlich explorativ: Robustheit gegenüber der Lernrate, also wie stark sich der Fehler über 1e-3, 3e-3 und 1e-2 ändert.
- **Entscheidung:**
  - Zeigt PreCap eine kleinere Spreizung über die Lernraten als SiLU und SiLU×2, wird die Lernraten-Robustheit Hauptthese der Veröffentlichung.
  - Sonst rückt die Stabilität in der Tiefe in den Vordergrund.
- **Erledigt am 26.10.2026:** Ohne Norm ist die Spreizung bei PreCap zwar kleiner (0,99 gegen 1,41 bei SiLU). Das liegt aber nur daran, dass PreCap bei zu kleiner Lernrate weniger verliert. Das Fenster der guten Lernraten ist kaum breiter (1,3 gegen 1,0 von 3), und der Bestwert ist schlechter. Mit LayerNorm ist SiLU am robustesten (Spreizung 0,08 gegen 0,18). **Entscheidung: Die Lernraten-Robustheit wird nicht Hauptthese.** In den Vordergrund rücken Stabilität in der Tiefe (V17) und das Verhalten mit LayerNorm. Phase 3 bekommt ein nach oben offenes Lernraten-Raster und einen Schwerpunkt auf Pre-LN-Netzen. Details: `analysis/V1_3D_Auswertung_ausfuehrlich.md`.

## Phase 1 – PreCap mathematisch vollständig beschreiben (1–2 Abende)

Das Ergebnis ist das Kapitel "Definition und Eigenschaften" des Papers. Ich rechne alles und prüfe es numerisch.

- Definition, erste und zweite Ableitung, Stetigkeit (Glattheitsklasse).
- Wertebereich und Schranken. Oben gilt etwa 3,08. Untere Schranke und Verhalten für große negative Werte. Achtung: Für \(x<-50\) steigt \(u\) durch den quadratischen Term wieder an, das wird untersucht und dokumentiert.
- Taylor-Entwicklung am Ursprung: PreCap ≈ \(x + 0{,}51x^2\).
- Bezug zu bekannten Funktionen:
  - CapExp ist ein verschobenes und skaliertes ELU,
  - nahe null verhält sich PreCap ähnlich wie SiLU×2.
- Rechenaufwand pro Aufruf im Vergleich zu ReLU, SiLU und ELU.
- **Neu: selbstnormierendes PreCap ("SPreCap").** Analog zu SELU werden Konstanten bestimmt, mit denen Mittelwert und Varianz über die Schichten stabil bleiben. Das ist ein möglicher eigener Beitrag.
- Möglichst wenige Parameter. Prüfen, ob c und s zu einem Parameter zusammengefasst werden können.
- **Erledigt am 26.10.2026** (`math/PreCap_Definition_und_Eigenschaften.md`):
  - PreCap ist \(C^1\) und nach oben durch 3,077 begrenzt.
  - Formelfehler: Durch \(\gamma z^2\) ist PreCap für \(x<-50\) nicht monoton und springt für \(x<-120\) auf den Höchstwert. Korrektur: \(\gamma=0\).
  - Nahe null ist PreCap bis zur dritten Ordnung gleich 2·SiLU, bei \(s=\sqrt2\) exakt.
  - SPreCap mit λ = 0,806769 und α = 2,396951 ist selbstnormierend und über 100 Schichten stabil wie SELU, der Gradient wächst aber stärker.
  - PreCap ist ohne `torch.compile` auf der GPU etwa 9-mal langsamer als SiLU.

## Phase 2 – Systematische Literaturrecherche (1 Abend)

- Die Übersicht über 400 Aktivierungsfunktionen (Kunc & Kléma 2024, arXiv:2402.09092) Eintrag für Eintrag nach tanh-, exp- und ELU-Kombinationen durchsuchen.
- Google Scholar, Semantic Scholar, arXiv und Google Patents mit festen Suchbegriffen durchsuchen. Begriffe und Datum werden protokolliert.
- Mathematisch prüfen, ob PreCap nur eine Umparametrisierung einer bekannten Funktion ist.
- Ergebnis: ein Abschnitt "Verwandte Arbeiten" mit ehrlicher Abgrenzung.

- **Erledigt am 26.10.2026 (erste Runde):** `literatur/Phase2_Literaturrecherche.md`.
  - Nicht neu sind CapExp (= ELU), Stabilität durch Beschränkung und das Rezept für selbstnormierende Konstanten.
  - Wahrscheinlich neu, aber klein sind die Verkettung CapExp(s·tanh(x/s)) und SPreCap.
  - Neu: SPreCap mit Eingangsskala b für \(\mathbb E f'^2=1+\varepsilon\). Der Preis sind weniger Gradientenverstärkung und weniger Nichtlinearität.
  - Offen sind die Patentrecherche und das vollständige Lesen der Survey-Abschnitte.

## Phase 3 – Tests an echten Netzen (mehrere Nächte auf dem M1)

- **Phase 3a erledigt am 26.10.2026:** Keine Hauptthese bestätigt. PreCap0 lernt bei 100 Schichten nicht. Beim Start verschwinden die Gradienten nicht (χ₁ ≈ 1,06, Gradient wächst zur Eingabe hin), der Ausfall entsteht erst im Training: Sättigung an den Schranken plus Mittelwert ≠ 0 (Hypothese, Konfidenz etwa 0,6, siehe `math/p3_precap_mechanism.py`). SPreCap ist bei Fashion 100 die beste Funktion, bei 50 Schichten schlechter als SELU. Größte Einschränkung: LR-Raster unten nicht offen (Kandidaten 6/6 am Rand 1e-4). Nächster Schritt: Nachlauf 3a-2 mit LR 1e-5 bis 2e-4. Details: `analysis/P3A_Auswertung.md`.
- **Phase 3a-2 erledigt am 26.10.2026:** Nachteil SPreCap gegen SELU bestätigt (4 von 4 Zellen schlechter, bei Tiefe 100 7–8 pp). PreCap0 lernt bei Tiefe 100 mit LR ≤ 2e-5 (18,2 % / 8,3 %). Das Gradientenprofil beim Start folgt \((L-1)/2\cdot\log_{10}\chi_1\), SPreCap ist mit χ₁ = 1,13 zu chaotisch. Beste Funktionen: ELU und SELU. Keine Divergenz bei kleinen LRs. Details: `analysis/P3A2_Auswertung.md`.
- **Phase 3a-3 (vorregistriert, gestartet am 26.10.2026):** χ₁-angepasstes SPreCap (`spreccap_cs`, χ₁ = SELU) gegen SELU, dazu χ₁ = 1,03 und lernbares s. Die Entscheidungsregel für v0.1 steht in `docs/P3A3_Praeregistrierung.md`.

### Vergleichsfunktionen

ReLU, GELU, SiLU, SiLU×2, ELU, SELU, Squared ReLU, PreCap und gegebenenfalls SPreCap.

### Benchmarks (klein genug für den Mac)

| Benchmark | Netz | Wozu |
|---|---|---|
| MNIST / Fashion-MNIST | tiefes MLP mit 8–64 Schichten, ohne Normalisierung | Stabilität in der Tiefe |
| CIFAR-10 | kleines ResNet und einfaches CNN ohne BatchNorm | echter Bildbenchmark |
| TinyShakespeare | Mini-Transformer, Zeichenebene | Sprachmodell mit Kurzbeleg |

### Protokoll

- Lernraten-Raster von 1e-4 bis 3e-2 in halben Zehnerpotenzen, für jede Aktivierung.
- 5 Tuning-Seeds und 10 Bestätigungs-Seeds.
- Gemessen werden:
  - Bestwert (Genauigkeit bzw. Verlust),
  - Breite des Lernratenfensters, in dem eine Funktion höchstens 1 % schlechter ist als ihr eigenes Optimum,
  - Divergenzrate,
  - Rechenzeit.

### Erfolgskriterien (werden vorher eingefroren)

- **Lernraten-Robustheit:** Das gute Lernratenfenster ist bei PreCap in mindestens 2 von 3 Benchmarks deutlich breiter als bei SiLU und GELU, und der Bestwert ist dabei nicht klar schlechter.
- **Tiefe:** PreCap divergiert ohne Normalisierung bei großer Tiefe seltener als ELU, SiLU und GELU. SELU ist dabei der Hauptgegner.

Ablauf: zuerst ein Pilot und ein Trockenlauf der Analyse, danach der große Lauf.

## Phase 4 – Auswertung und Grafiken

- Grafiken der Funktion und ihrer Ableitungen im Vergleich zu ELU, SiLU und GELU.
- Lernraten-Kurven: Fehler über der Lernrate je Aktivierung. Das ist die wichtigste Grafik.
- Stabilität in der Tiefe: Divergenzrate und Fehler über der Anzahl der Schichten.
- Tabellen mit Konfidenzintervallen und Holm-korrigierten Tests.

## Phase 4b – Effiziente Kernels (nur wenn Phase 3 positiv ist)

- Grund: Ohne Fusion ist PreCap auf MPS etwa 9,3-mal so teuer wie ReLU (Phase 1). Ohne schnellen Kernel testet es niemand ernsthaft.
- MLX: fusionierter Vorwärts- und Rückwärts-Kernel in Metal (`mx.fast.metal_kernel`), getestet auf dem M1 Pro.
- CUDA: zuerst Triton bzw. `torch.compile` als Referenz, dann eine eigene CUDA-Erweiterung (`torch.autograd.Function`), getestet auf dem Rechner mit RTX 4070.
- Pflichttests: numerischer Abgleich mit der Referenz (fp32, fp16, bf16), Gradcheck, Grenzfälle (sehr große und kleine x, NaN, Inf), Benchmarks gegen ReLU, SiLU und GELU.
- Auslieferung zusammen mit dem Release: pip-Paket mit PyTorch- und MLX-Variante.

## Entscheidungspunkt vor Phase 5 – Patent oder offene Veröffentlichung

- Die Entscheidung fällt erst, wenn Phase 3 komplett ausgewertet ist, also nicht schon nach 3a.
- Sie muss aber vor jeder öffentlichen Offenlegung fallen: öffentliches Repo, Zenodo, arXiv, Reddit, Mails. Im europäischen Patentrecht gibt es keine Neuheitsschonfrist; auch eine eigene Veröffentlichung zerstört die Neuheit. Das private Repo ist unschädlich.
- Entscheidung (26.10.2026): Ziel ist die Absicherung über eine offene, datierte Veröffentlichung mit DOI und Apache-2.0 (defensive Veröffentlichung), nicht ein Patent.
- Einschätzung: Ein Patent ist wenig aussichtsreich. Gründe: Eine Aktivierungsfunktion ist als mathematische Methode nur mit konkretem technischem Bezug schutzfähig, und alle Bausteine (ELU, tanh, DyT) sind bekannt, was die erfinderische Tätigkeit schwächt. Dazu kommen Kosten im vierstelligen Bereich. Empfehlung: offene Veröffentlichung mit DOI. Die macht PreCap zum datierten Stand der Technik, sodass niemand anderes es patentieren kann. Bei einem sehr starken Ergebnis vorher eine Erstberatung bei einem Patentanwalt einholen.
- Lizenz: Apache-2.0 statt MIT. Apache-2.0 enthält eine ausdrückliche Patentlizenz (Abschnitt 3) und ist deshalb für Nutzer sicherer. Sie verträgt sich aber nicht mit einem eigenen Patent, das man durchsetzen will.

## Zwischenziel v0.1 – Stand absichern (Entscheidung 26.10.2026)

- **Bedingung:** SPreCap bewährt sich in 3a-2 und 3a-3, mindestens "nicht schlechter als SELU" nach der vorregistrierten Regel, bei 0 % Divergenz. Fällt das Ergebnis negativ aus, wird trotzdem abgesichert, aber ohne Kernels und mit ehrlicher Einordnung.
- **Inhalt:** Repo öffentlich unter Apache-2.0 (LICENSE, NOTICE und CITATION.cff liegen bereit), GitHub-Release v0.1.0, DOI über Zenodo auf den echten vollen Namen (Profil vorhanden), GitHub-Konto `hermi-labs`.
- **Kernels:** Sie kommen mit in v0.1, wenn die Tests gut aussehen: MLX/Metal auf dem M1 Pro und CUDA auf der RTX 4070, jeweils mit Abgleich zur Referenz und Benchmarks nach Phase 4b.
- **Danach:** Test über MNIST hinaus (kleiner Transformer und/oder CIFAR-10), 3b (weniger LayerNorm), Paper und Posts. Entwürfe: `docs/P3A3_Entwurf.md`; Entwürfe für Posts liegen lokal (nicht im Repo).
- Reihenfolge bis dahin: 3a-2 auswerten → 3a-3 einfrieren und laufen lassen → auswerten → Kernels → v0.1.

## Phase 5 – Veröffentlichung

- **Paper** mit 6–10 Seiten als PDF im Stil des ELU-Papers:
  - Abstract, Motivation, Definition und Eigenschaften, verwandte Arbeiten, Methode,
  - Ergebnisse aus 1D, 2D und 3D sowie den echten Netzen,
  - Grenzen, Fazit.
  - Die Entstehungsgeschichte kommt in einen Anhang. Die Gesamtdoku liegt bei.
- **Code und Daten:** öffentliches GitHub-Repository unter Apache-2.0-Lizenz mit Skripten, Präregistrierungen, Hashes und Rohdaten oder Zusammenfassungen.
- **DOI:** GitHub-Release mit Zenodo verknüpfen. Das ist kostenlos, datiert, zitierbar und läuft auf deinen Namen. arXiv ist optional und braucht eine Empfehlung durch einen anderen Autor.
- **Bekanntmachen:** Beitrag auf r/MachineLearning und X, Mail an einzelne Forscher zu Aktivierungsfunktionen, optional an Anthropic und OpenAI.

## Phase 6 – Matrix-Idee als eigene Variante (erst nach der DOI)

- Handschriftliche Notiz vom 16.10.: \(y = \sigma\big((Wx+b)^p \cdot \lambda\big)\). Die Voraktivierung wird potenziert und durch einen Faktor gebremst, σ ist PreCap, SiLU oder CapExp.
- **Verwandte Arbeiten:**
  - Würfelfunktion \(x^3\) (Chen & Manning 2014),
  - Squared ReLU (Primer, So et al. 2021).
- **Test in derselben Benchmark-Suite:**
  - p = 2 und p = 3, λ fest oder lernbar,
  - jeweils mit SiLU und mit PreCap,
  - gegen Squared ReLU als Referenz.
- Wenn das trägt: eigene, zweite Veröffentlichung, die auf die PreCap-DOI verweist.

---

## Realistische Erwartung

- Eine saubere, nachprüfbare Veröffentlichung ist machbar, unabhängig vom Ergebnis.
- Ein belegbarer Vorteil bei der Lernraten-Robustheit oder bei der Stabilität in der Tiefe ist möglich, aber offen. Konfidenz etwa 0,4.
- Ein allgemeiner Leistungsvorteil gegenüber SiLU und GELU ist eher unwahrscheinlich. Konfidenz etwa 0,25.
