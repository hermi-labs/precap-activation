# CapExp 2D — Gesamtdokumentation (Stand: 20.10.2026, Abend)

## 1. Worum es geht
PreCap ist eine glatte Aktivierungsfunktion mit sanfter Begrenzung grosser positiver Voraktivierungen. Getestet wird sie in synthetischen 2D-Regressionsaufgaben (Residual-Netz "res_inv", Breite 64, Adam, LR 0.001) gegen SiLU und GELU.

Primaermetrik: d = log10(MSE_PreCap / MSE_Referenz). d < 0 heisst: PreCap besser. Klammern zeigen PreCap-Siege pro gepaarte Seeds.

## 2. Die Formel (verifiziert gegen alle Runner v1b bis v6)
z = a*x;  u = z + gamma*z^2;  v = s*tanh(u/s)
PreCap(x) = CapExp(v),  CapExp(v) = exp(v)-1 fuer v <= c,  sonst exp(c)-1 + exp(c)*(v-c)

Fixe Basis-Parameter: c = 1.0, a = 1.0, gamma = 0.01, s = 1.5.
Selbsttest in jedem Runner: formula_max_abs_err = 0 (v6 zusaetzlich cap_branch_jump ca. 5.4e-07).

WICHTIG (Erratum): v2_holdout/docs/PreCap_Urformel_Spezifikation.md beschreibt eine Softplus-Form (x - 0.01*softplus((x-2)/1.5)). Das stimmt NICHT mit dem Code ueberein. Massgeblich ist die tanh+CapExp-Form oben, siehe ERRATUM_Formel_Spezifikation.md. Die Ergebnisse sind davon nicht betroffen, weil sie vom Code erzeugt wurden.

## 3. Chronologie der Studien
| Studie | Zweck | Umfang | Ergebnis |
|---|---|---|---|
| v1_pilot | erster Pilot | 135 Runs | PASS = False. UNGUELTIG: Implementierungsfehler in der Formel, fliesst nicht ein |
| v1b_corrected | Pilot mit korrigierter Formel | 135 Runs, 3 Aufgaben, Tiefen 4/8/16, 5 Seeds | PASS. 37/45 Siege gegen SiLU, 33/45 gegen GELU; radial_ripple und ridge_mix bestehen, corner_kink gegen GELU nicht |
| v2_holdout | unabhaengiger Holdout, neue Seeds 400-409 | 270 Runs, 90 Paare, bad=0 | PASS. 83/90 gegen SiLU (p=6.6e-18), 64/90 gegen GELU (p=3.8e-05); Effekt nimmt mit Tiefe ab (Tiefe 16 gegen GELU nur -0.072) |
| v3_large_preregistered | grosse Replikation, Tiefen bis 64, 6 Aufgaben | 900 Runs, 300 Paare, bad=0 | Primaeres Gate PASS (3/3 Kernaufgaben). Tiefe 4..32 negativ gegen beide; Tiefe 64 kippt (SiLU -0.038, GELU +0.019) |
| v4_mechanism_2d | Mechanismus: Gradienten, Peaks, Cap-Anteil ueber die Zeit | Snapshots, 3000 Schritte | deskriptiv: Peak-Verhaeltnis fast immer unter 1, Gradientenverhaeltnis uneinheitlich; keine Kausalaussage |
| v5_depth64_diagnosis | warum kippt Tiefe 64? | Grenzfaelle radial_ripple 64, ridge_mix 64, Anker radial_ripple 16, Gegenbeispiel highfreq_grid 32 | kein einfacher Gradienten-Crash; Pfade sind seed-, aufgaben- und referenzabhaengig (z.B. highfreq_grid 32: Vorsprung spaet verloren 2/5, kein stabiler Vorsprung 2/5, gehalten 1/5) |
| v6_precap_ablation | c-/s-Ablation, FERTIG | 105 Runs, 630 Checkpoint-Zeilen | kein belastbarer c/s-Treffer; c125 und s225 nur in ridge_mix d64 formal 5/5 bzw. 4/5, Effekte klein; Varianz (Ausreißer-Seeds) dominiert Tiefe 64 |

## 4. Aktueller Wissensstand
Gesichert:
- Replizierbares 2D-Signal auf radial_ripple, ridge_mix (gegen SiLU und GELU) und corner_kink (gegen SiLU); in v3 3/3 Kern-PASS.
- Der Vorteil traegt bis Tiefe 32.
- Peak-Aktivierungen sind mit PreCap meist kleiner; der Cap-Anteil ist klein (Median 0.03 in v2).

Begrenzt / offen:
- Tiefe 64: Vorteil verschwindet im Aggregat, besonders gegen GELU.
- Corner_kink gegen GELU: kein Vorteil.
- highfreq_grid: praktisch kein Effekt (|d| unter 0.02).
- Nur synthetische 2D-Aufgaben, nur eine Architektur, keine realen Daten.
- Mechanismus nur deskriptiv, nicht kausal.

## 5. V6 — Design und Hypothesen
Frage: Liegt die Tiefe-64-Schwaeche an (a) zu starker Begrenzung, (b) zu schwacher Begrenzung, oder (c) an reiner Seed-/Aufgabenvariabilitaet?

Zellen: radial_ripple d64, ridge_mix d64 (Mechanismus), radial_ripple d16 (Anker).
Seeds 900-904. Schritte 1200, Checkpoints 0/100/300/500/800/1200, 4 Worker.
Varianten: precap_c075, precap_base (c=1.0), precap_c125 (Cap-Schwelle c variiert); precap_s075, precap_s225 (tanh-Breite s variiert); Referenzen silu, gelu.

Deutung vorab:
- Wenn c075/s075 (staerker begrenzt) besser als base: Hinweis auf Unterbegrenzung.
- Wenn c125/s225 (schwaecher begrenzt) besser: Hinweis auf Ueberbegrenzung.
- Wenn keine Variante konsistent besser und die Streuung zwischen Seeds gross: Variabilitaet dominiert, c/s sind nicht der Hebel.

Entscheidungsregel (vorab): Eine Variante zaehlt als bedeutsam, wenn sie bei d64 in einer Zielzelle niedrigere Test-MSE als precap_base hat und in mindestens 4 von 5 gepaarten Seeds besser ist. Mit nur 5 Seeds ist das ein Hinweis, kein Beweis; Folgestudie mit mehr Seeds waere noetig.

## 6. Methodik und Qualitaetssicherung
- Vorab festgelegte PASS-Regeln, eingefrorene Formel (SHA256-Hashes in v6/meta/frozen_sha256.json).
- Integritaetspruefung: Zeilenzahl, Eindeutigkeit, bad=0, vollstaendige gepaarte Zellen.
- Gepaarte Seeds, einseitiger Vorzeichentest.
- Lehre aus v1: Selbsttest gegen eine unabhaengige Referenz vor jedem Start ist Pflicht.

## 7. Ordnerstruktur
combined_visuals (Gesamtgrafiken, kompakte Folgerung), Wunschgrafik Liveverfolgung, v1_pilot, v1b_corrected, v2_holdout, v3_large_preregistered, v4_mechanism_2d, v5_depth64_diagnosis, v6_precap_ablation. Je Ordner: docs/, meta/manifest.json, results/ bzw. workers/, logs/, Runner-Skript.

## 8. Naechste Schritte
1. V6 abwarten (python status.py), dann run_v6_precap_ablation.py --integrity.
2. Auswertung: paarweise Differenz je Variante vs. base, Seed-Streuung, Checkpoint-Verlaeufe.
3. Entscheiden: c/s korrigieren (neues, eingefrorenes v7) oder Variabilitaet mit mehr Seeds absichern.
4. Erst danach neue Aufgaben, andere Architekturen oder reale Daten.

## 9. V6-Ergebnis und neuer Fahrplan
Siehe v6_precap_ablation/docs/V6_Auswertung.md und Fahrplan_nach_V6.md.

## 10. v7a-Ergebnis
v7a bestätigt keinen robusten globalen Vorteil von c125 oder s225 bei Tiefe 64. s225 verbessert radial_ripple d16 deutlich, ist bei Tiefe 64 aber nicht nach korrigiertem Kriterium bestätigt. Details: v7a_seed_extension/docs/V7a_Endauswertung.md und Fahrplan_nach_V7a.md.

## 11. v7b-Ergebnis
Tiefe x s formal nicht bestaetigt (nur d8 und d32 einzeln signifikant). s225 hilft aber konsistent auf radial_ripple (Median d -0.13 bis -0.31 in allen Tiefen; explorativ gepoolt p=3.7e-6), nicht auf ridge_mix. Aufgabenspezifischer, nicht tiefenspezifischer Effekt. Siehe v7b_depth_x_s/docs/V7b_Endauswertung.md und Fahrplan_nach_V7b.md.

## 12. V8-Ergebnis
V8 bestätigt den s225-Hinweis für die radiale Gruppe (6/6 negative Zellmediane, Gruppenmedian -0.121), aber nicht den strengsten radial-gegen-ridge-Kontrast, weil 4/6 ridge-Zellen ebenfalls leicht negativ sind. Explorativ ist der gepaarte Gruppenunterschied stark (p=3.15e-5). Ergebnis: aufgabenspezifischer radialer Vorteil, keine globale Umstellung. Siehe v8_task_axis/docs/V8_Endauswertung.md.

## 13. V9-Ergebnis
V9 bestaetigt keine einfache monotone Geometrie-Regel: Spearman rho=0.739, aber Endpunktkontrast a=0 gegen a=1 nicht signifikant (p=0.194). Der staerkste s225-Effekt liegt bei a=.25 statt am radialen Endpunkt. Damit ist der Effekt aufgabenspezifisch und nichtmonoton; base bleibt Standard. Siehe v9_geometry/docs/V9_Endauswertung.md und docs/Studiengeschichte_Plain_Language.md.

## 14. V10-Plan
Nach V9 folgt keine weitere blinde Parametersuche, sondern eine Mechanismusstudie: geo_a025 (starker s225-Fall) gegen geo_a075 (Gegenfall), Tiefe 16/32, 20 neue Seeds, mit Schichtdiagnostiken fuer Aktivierungen, Pre-Activations, Kappungsanteile und Gradienten. Siehe v10_mechanism/docs/V10_Praeregistrierung.md.

## 15. V10-Auswertung
V10 ist als Mechanismusstudie geplant. Die Auswertung trennt Datenintegritaet, die Leistungsreplikation und Zeitprofile interner Messgroessen. Ein behaupteter Mechanismus braucht ein konsistentes a025-vs-a075-Muster ueber Seeds und idealerweise beide Tiefen; sonst bleibt er explorativ. Siehe v10_mechanism/docs/V10_Auswertungsplan.md.

## 16. V10-Ergebnis
Replikation bestaetigt: s225 hilft bei geo_a025 (Median-MSE-Verhaeltnis 0.52 bei Tiefe 16, 0.27 bei Tiefe 32; 18/20 bzw. 17/20 Seeds), nicht bei geo_a075 (1.09/1.12; 8/20 bzw. 9/20). Mechanismus nur explorativ (Pre-Activation-p99, hmax_peak). Naechster Schritt: V11 Bestaetigungsstudie. Siehe v10_mechanism/analysis/V10_Auswertung.md.

## V12 – Diagnose der s-Wirkung (Drehung/Frequenz)

Ziel: Nach V11 nicht direkt neue PreCap-Varianten bauen, sondern die Bedingungen der s-Wirkung diagnostisch trennen. V12 testet, ob die Wirkung von `precap_s225` gegenüber `precap_base` vor allem durch Drehung oder durch Frequenz/Komplexität der Aufgabe bestimmt wird.

Design:
- Aktivierungen: `precap_base` (s=1,5), `precap_s225` (s=2,25), `precap_s300` (s=3,0 als Dosiskontrolle).
- Aufgaben: `rot_ang53_k10_a025` (Anker, identisch zu V11 `rot_a025`), `rot_ang00_k10_a025` (Drehung entfernt), `rot_ang53_k15_a025` (höhere Frequenz), `rot_ang53_k07_a025` (niedrigere Frequenz).
- Tiefen: 16 und 32.
- Seeds: 7000–7019 (20 neue, disjunkte Seeds).
- Umfang: 480 Runs, 3360 Checkpoint-Zeilen.

Konfirmatorische Regeln:
- Replikation des Ankers bei Tiefe 16.
- Sechs gepaarte Kontraste Aufgabe minus Anker (3 Nicht-Anker-Aufgaben × 2 Tiefen), Wilcoxon zweiseitig, Holm-korrigiert.

Status:
- Ordner `2D/v12_diagnosis/` erstellt.
- Präregistrierung, Runner, Analyse, Launcher und Statusskript vorhanden.
- Selbsttest, Syntaxprüfung, Anker-Äquivalenz zu V11, Seed-Disjunktheit und Smoke-Test bestanden.
- SHA-256-Freeze in `v12_diagnosis/meta/frozen_sha256.txt` geschrieben.

## V12 – Ergebnis
V12 vollstaendig (480/480 Runs, 3360/3360 Zeilen, Hashes unveraendert). Anker bei Tiefe 16 repliziert. Alle 6 konfirmatorischen Kontraste nach Holm nicht signifikant (kleinster Holm-Wert 0.68): weder Drehung noch Frequenz erklaeren die s225-Wirkung belastbar. Ohne Drehung hilft s225 gleich gut oder besser; bei hoher Frequenz (k=1.5) geht der Vorteil verloren, bei Tiefe 32 Verhaeltnis 1.50 (explorativ). Anker bei Tiefe 32 diesmal 0.70 (V11: 0.98), also Seed-Instabilitaet einzelner Zellen. s300 ohne konsistente Dosiswirkung. p99-Mechanismus ohne Beleg (rho=-0.04, p=0.61). base bleibt Standard. Siehe v12_diagnosis/analysis/V12_Auswertung.md.

## V13 – Frequenzanalyse (Plan)
Frage: Nimmt der s225-Vorteil monoton mit der Frequenz k ab? Design: base vs s225, k = 0.7/1.0/1.25/1.5/2.0 (Drehung 53 Grad, a=0.25), Tiefe 16/32, 30 neue Seeds 8000-8029, 600 Runs, ca. 3-3.5 h. Konfirmatorisch: pro Tiefe Steigung von d ueber log10(k) je Seed, Wilcoxon einseitig, Holm ueber 2 Tests. Explorativ: Zellen, Kontraste k=1.5/2.0 gegen k=1.0, Kreuzungsfrequenz. Hashes eingefroren in v13_frequency/meta/frozen_sha256.txt. Siehe v13_frequency/docs/V13_Praeregistrierung.md.

## V13 – Ergebnis
V13 vollstaendig (600/600 Runs, 4200/4200 Zeilen, keine NaN/Inf, Hashes unveraendert, Laufzeit 17:26-21:21). Konfirmatorisch: beide Steigungstests (d ueber log10 k) nach Holm nicht signifikant (tief 16: Steigung -0.36, Holm 0.994, Gegenrichtung; tief 32: +0.20, Holm 0.349). Hypothese 'Vorteil von s225 nimmt mit Frequenz ab' verworfen. Explorativ: s225 in 9 von 10 Zellen klar besser (MSE-Verhaeltnis ca. 0.47-0.78, p<=0.006); einzige Ausnahme k=1.5, Tiefe 32 (Verhaeltnis 1.00, 15/30 Seeds, p=0.36), obwohl k=2.0 bei Tiefe 32 wieder 0.61 zeigt. Der V12-Rueckschlag (1.50) bei k=1.5/Tiefe 32 ist damit kein Trend, sondern Zellschwankung oder Sonderfall. Paarkontraste k=1.5/2.0 gegen k=1.0 n.s. (kleinstes p=0.12). Kreuzungsfrequenz 1.5 (Tiefe 32) nur Artefakt. Einordnung: ueber V10-V13 ist s225 fuer diese Aufgabenfamilie (rot, 53 Grad, a=0.25, k 0.7-2.0) robust besser als base; Uebertragbarkeit auf andere Aufgabenfamilien ungetestet. Siehe v13_frequency/analysis/V13_Auswertung.md.

## V11 – Ergebnis
V11 (Bestaetigung, 6 Zellen: geo_a025 und rot_a025 als Erfolgs-, rot_a075 als Gegenzellen, Tiefe 16/32, 20 Seeds). H1 (Leistung in allen Erfolgszellen repliziert) nicht erfuellt: geo_a025 d16/d32 und rot_a025 d16 klar besser (Verhaeltnis 0.62/0.37/0.48, Holm <0.003), aber rot_a025 d32 nicht (0.98, p=0.25). H2 (kein Nutzen in Gegenzellen) nicht erfuellt: rot_a075 d16 Verhaeltnis 0.78 (p=0.008). H3 (Mechanismus pre-activation p99) nicht bestaetigt (zentriertes Spearman rho=-0.014 bzw. 0.015). Folge: V12/V13 zur Diagnose von Drehung und Frequenz. Siehe v11_confirmation/analysis/V11_Auswertung.md.


## Wissensstand-Update 22.10.2026 (nach V10-V13)
Gesichert (Aufgabenfamilie rot/geo, 53 Grad, a=0.25, k 0.7-2.0, Tiefe 16/32, 1200 Schritte, base vs s225): s225 ist in den meisten Zellen klar besser als base (V13: 9 von 10 Zellen, MSE-Verhaeltnis ca. 0.47-0.78). Weder Drehung (V12) noch Frequenz (V13) erklaeren die Wirkung belastbar; alle konfirmatorischen Kontraste n.s. Einzelne Zellen schwanken zwischen Seed-Sets stark (rot Anker d32: V11 0.98, V12 0.70; k=1.5/d32: V12 1.50, V13 1.00). Mechanismus (pre-activation p99, Peak) ohne Beleg.
Offen: Uebertragbarkeit auf andere Aufgabenfamilien (V14), Vergleich mit SiLU/GELU in V10-V13 nicht enthalten, Trainingsdauer 1200 statt 3000 Schritte (V3), Stabilitaet der Zelle k=1.5/Tiefe 32, feineres s-Raster. Nur 2D-Synthetik, eine Architektur (res_inv), keine realen Daten.
Formelpruefung 22.10.: Aktivierung (z=a*x, u=z+gamma*z^2, v=s*tanh(u/s), CapExp(c=1)), Netz (res_inv, Breite 64, INIT_K 0.5) und die sechs V3-Aufgaben sind im V14-Runner numerisch identisch zu v3_large_preregistered (max. Abweichung 0.0). Die Softplus-Spezifikation bleibt falsch (siehe ERRATUM); Neuschreiben als eigener Text steht aus.


## V14 – Uebertragbarkeit (Plan)
Frage: Ist s225 gegen base auch auf den sechs V3-Aufgaben (radial_ripple, ridge_mix, corner_kink, smooth_bowl, highfreq_grid, angled_kink) besser? Design: base, s225, silu (Referenz), Tiefe 16/32, 20 neue Seeds 9000-9019, 1200 Schritte, 720 Runs, ca. 4.5-5 h. Konfirmatorisch: je Aufgabe s225 vs base, Mittel ueber Tiefen je Seed, Wilcoxon zweiseitig, Holm ueber 6. Explorativ: Zellen und silu-Vergleiche. Hashes eingefroren in v14_transfer/meta/frozen_sha256.txt. Siehe v14_transfer/docs/V14_Praeregistrierung.md.

## V15 – Ausnahmezelle rot_k15/32 (Plan)
Frage: Ist die Zelle rot_k15 bei Tiefe 32 eine echte Ausnahme oder Seed-Instabilitaet? Design: nur rot_k15, nur Tiefe 32, nur precap_base vs precap_s225, 100 neue Seeds (9500-9599), 200 Runs. Metrik: d=log10(MSE_s225/MSE_base) bei Schritt 1200; primaer Wilcoxon zweiseitig, sekundaer gerichtet auf s225 besser. Auto-Start-Helfer liegt in v15_exception_cell/auto_start_after_v14.sh und wartet auf Abschluss von V14. Hashes eingefroren in v15_exception_cell/meta/frozen_sha256.txt.

## V14 – Ergebnis
Lauf 22./23.10.2026, 1440 Runs vollstaendig, Integritaet ok (10080/10080 Zeilen), Hashes unveraendert. Technischer Hinweis: Der Launcher hing nach Ende der Worker in 'wait' (eigener caffeinate-Hintergrundjob); Analyse und done.txt wurden manuell nachgeholt, Daten unberuehrt. Korrigierte Vorlage: v14_transfer/launch_v14_fixed.sh.
Konfirmatorisch (s225 vs base je Aufgabe, Mittel ueber Tiefen 8/16/32, Wilcoxon zweiseitig, Holm ueber 6): s225 besser bei smooth_bowl (Verhaeltnis 0.61) und radial_ripple (0.63); schlechter bei corner_kink (1.17) und highfreq_grid (1.01, Effekt winzig); kein gesicherter Unterschied bei ridge_mix und angled_kink (je 1.04). Vorab festgelegte Regel (>=4 von 6 besser, keine schlechter) nicht erfuellt: s225 ist aufgabenspezifisch, nicht allgemein besser als base.
Explorativ (Holm ueber 24 Tests, 1200 Schritte): base schlaegt SiLU in 6 von 6 Aufgaben, s225 in 5 von 6 (highfreq_grid n.s.). Gegen GELU: base besser in 4, n.s. in 2; s225 besser in 4, schlechter bei corner_kink (1.16), n.s. bei highfreq_grid. Mediane Fehlerverhaeltnisse PreCap/SiLU 0.03-1.0, PreCap/GELU 0.13-1.25. Mittlerer Rang ueber 18 Zellen: base 1.72, s225 1.78, GELU 2.56, SiLU 3.94. highfreq_grid: alle vier Aktivierungen praktisch gleich. Vorsicht: nur 1200 Schritte; SiLU kann bei dieser Initialisierung langsamer lernen. Dateien: v14_transfer/analysis/V14_Auswertung.md, v14_exploratory_vs_silu_gelu.csv.


## V15 – Ergebnis
200 Runs vollstaendig (Seeds 9500-9599, Aufgabe rot_ang53_f150_a025, Tiefe 32, base vs s225). Median-Verhaeltnis s225/base 1.19, 40 Seeds fuer s225 und 60 fuer base, Wilcoxon zweiseitig p=0.128, gerichtet (s225 besser) p=0.94. Streuung extrem: 26 von 100 Seeds mit |d|>0.5 (Faktor >3), Standardabweichung von d=0.48. Fazit: kein echter s225-Effekt in dieser Zelle; die frueheren Widersprueche (V12 1.50, V13 1.00) sind Seed-Rauschen einer instabilen Zelle. Die Zelle sollte nicht als Beleg fuer oder gegen s225 bei Tiefe 32 verwendet werden.


## Wissensstand-Update 23.10.2026 (nach V14/V15)
Gesichert (2D-Synthetik, res_inv, 1200 Schritte): PreCap (base und s225) schlaegt SiLU auf 5-6 von 6 Aufgabenfamilien und GELU auf 4 von 6, bei glatten Funktionen mit grossem Abstand. s225 gegen base ist aufgabenspezifisch (besser bei glatten Funktionen smooth_bowl/radial_ripple, schlechter bei Knicken corner_kink). Die Ausnahmezelle rot_k15/32 ist Seed-Rauschen. Offen: Kontrolle mit 3000 Schritten (SiLU-Lerngeschwindigkeit), Erklaerung des corner_kink-Nachteils, Mechanismus, reale Daten, weitere Architekturen.

## V16 – Tiefenstudie (Entwurf)
Ordner v16_depth_study (NICHT gestartet). Frage: warum ist PreCap bei Tiefe 64/128 meist gleichauf? Hypothesen H1 Untertraining (1200 vs 3000 Schritte), H2 Residual-Skalierung 1/sqrt(Tiefe) macht Schichten fast linear, H3 Seed-Rauschen. Design: 6 Aufgaben x Tiefe 32/64/128 x 4 Aktivierungen x 20 Seeds 9700-9719, 3000 Schritte, 1440 Runs, ca. 6-7 h. Konfirmatorisch: base vs SiLU/GELU bei Tiefe 64/128, 24 Tests Holm, plus Aequivalenzlesart (KI in 0.9-1.1). Details: v16_depth_study/docs/V16_Praeregistrierung.md. Start: ./launch_v16.sh.

## V16 – Ergebnis
1440 Runs vollstaendig (11520/11520 Zeilen, Hashes unveraendert), Launcher lief durch (done.txt automatisch). Konfirmatorisch (Schritt 3000, Tiefe 64/128, base vs SiLU/GELU, 24 Tests Holm): 1 PreCap besser (corner_kink d64 vs SiLU, Verhaeltnis 0.52), 1 schlechter (corner_kink d128 vs GELU, 1.94), 4 aequivalent (highfreq_grid, dort passt aber keine Aktivierung: MSE ca. 0.0085 bei allen), 18 unklar (weite KIs). Geo-Mittel Verhaeltnis base/SiLU nach Schritt 3000: Tiefe 32 0.43, 64 0.89, 128 1.03; base/GELU 0.64, 1.04, 1.11; s225/SiLU 0.59, 0.72, 0.96; s225/GELU 0.68, 0.96, 1.09. Lernkurve base/SiLU (Median) Schritt 200: 0.49/0.69/0.89 (Tiefe 32/64/128); Schritt 3000: 0.43/0.89/1.03. H1 (Untertraining) nicht gestuetzt: laengeres Training stellt den Vorsprung bei Tiefe 64/128 nicht wieder her, die Referenzen holen auf (Tiefe 64: 0.66 bei Schritt 1200, 0.89 bei 3000). H2 (fast linear durch 1/sqrt(Tiefe)) qualitativ vereinbar: Anteil |pre|>0.5 (base) 0.106/0.068/0.050, cap_fraction 0.67%/0.43%/0.25%, post_rms_scaled 0.057/0.032/0.019; aber Spearman ueber Zellen rho=-0.08 (p=0.80, n=12), also kein Beleg auf Zellebene. H3 (Seed-Rauschen) stark: Streuung log10 MSE (base) 0.34/0.37/0.42, 18 von 24 Tests unklar. Bei Tiefe 128 wechselt das Vorzeichen je Aufgabe (corner_kink/angled_kink PreCap eher besser, smooth_bowl/ridge_mix eher schlechter als SiLU/GELU). Wichtig fuer V14: der SiLU-Vorsprung bei 1200 Schritten war teils ein Lerngeschwindigkeitseffekt; radial_ripple d32 base/SiLU 0.34 (1200) vs 0.89 (3000), die anderen Aufgaben bei Tiefe 32 bleiben deutlich besser (angled 0.17, ridge 0.33, smooth 0.29, corner 0.39). s225 explorativ (ohne Tests): bei Tiefe 128 besser als base bei radial_ripple 0.60, smooth_bowl 0.62, ridge_mix 0.84, schlechter bei angled_kink 1.74. Offen: Lernrate/Residualskalierung pro Tiefe, Ursache des corner_kink-Nachteils bei 128, andere Architekturen, reale Daten.

## V17 – Residual-Skalierung (Entwurf)
Ordner v17_residual_scale angelegt und geprueft, aber noch nicht gestartet. Ziel: kausaler Test, ob 1/sqrt(Tiefe) den PreCap-Vorteil in tiefen Netzen wegskaliert. Design: 4 Aufgaben x Tiefen 64/128 x 3 Aktivierungen (base, SiLU, GELU) x 3 Residual-Skalen (inv_sqrt, inv_quart, constant) x 20 Seeds = 1440 Runs. Pilot mit 2 Seeds erfolgreich und ohne nonfinite Faelle. Formel/Tasks stimmen exakt mit V16; inv_sqrt-Netz ist bei gleicher Initialisierung identisch zu V16. Start erst nach manuellem Go.


## V17 – Ergebnis (ausgewertet 25.10.2026)

Vollständig: 1440/1440 Runs, 11520 Zeilen, alle `ok`. Details: `v17_residual_scale/analysis/V17_Auswertung_ausfuehrlich.md`.

- Konfirmatorisch (48 Tests, Holm): 11 PreCap besser, 8 schlechter, 29 unklar. Bei konstanter Skala 8 besser (alle Tiefe 128) und 8 schlechter (alle Tiefe 64); bei inv_sqrt 2 besser, bei inv_quart 1 besser, sonst unklar.
- Die 8 "PreCap besser"-Fälle bei Tiefe 128/constant kommen von divergierenden Referenzen: SiLU/GELU Median-MSE 4 bis 28 (95 bis 100 % der Seeds über 1), PreCap 0.003 bis 0.017. PreCap ist dort aber 6- bis 115-mal schlechter als mit Dämpfung.
- Bei Tiefe 64/constant verliert PreCap, weil 36 % der Voraktivierungen im Cap liegen und die Gradientennorm hoch ist.
- H2 (Dämpfung nimmt PreCap den Vorsprung) nicht gestützt: Weniger Dämpfung (D^-1/4) hilft SiLU, GELU und PreCap etwa gleich (Differenz PreCap gegen Referenz nicht signifikant, p >= 0.12). Regel "3 von 4 Aufgaben" wird nur bei constant/Tiefe 128 formal erfüllt, und dort durch instabile Referenzen ausgeschlossen.
- Belegte Eigenschaft: Der Cap hält ungedämpfte 128-Schichten-Netze numerisch trainierbar.
- Bei sinnvoller Skalierung kein belastbarer Unterschied; aufgabenabhängig PreCap vorn bei corner_kink/smooth_bowl, SiLU/GELU bei radial_ripple/ridge_mix (explorativ).


## V17 – Ergebnis (ausgewertet 25.10.2026)

Vollständig: 1440/1440 Runs, 11520 Zeilen, alle `ok`. Details: `v17_residual_scale/analysis/V17_Auswertung_ausfuehrlich.md`.

- Konfirmatorisch (48 Tests, Holm): 11 PreCap besser, 8 schlechter, 29 unklar. Bei konstanter Skala 8 besser (alle Tiefe 128) und 8 schlechter (alle Tiefe 64); bei inv_sqrt 2 besser, bei inv_quart 1 besser, sonst unklar.
- Die 8 "PreCap besser"-Fälle bei Tiefe 128/constant kommen von divergierenden Referenzen: SiLU/GELU Median-MSE 4 bis 28 (95 bis 100 % der Seeds über 1), PreCap 0.003 bis 0.017. PreCap ist dort aber 6- bis 115-mal schlechter als mit Dämpfung.
- Bei Tiefe 64/constant verliert PreCap, weil 36 % der Voraktivierungen im Cap liegen und die Gradientennorm hoch ist.
- H2 (Dämpfung nimmt PreCap den Vorsprung) nicht gestützt: Weniger Dämpfung (D^-1/4) hilft SiLU, GELU und PreCap etwa gleich (Differenz PreCap gegen Referenz nicht signifikant, p >= 0.12). Regel "3 von 4 Aufgaben" wird nur bei constant/Tiefe 128 formal erfüllt, und dort durch instabile Referenzen ausgeschlossen.
- Belegte Eigenschaft: Der Cap hält ungedämpfte 128-Schichten-Netze numerisch trainierbar.
- Bei sinnvoller Skalierung kein belastbarer Unterschied; aufgabenabhängig PreCap vorn bei corner_kink/smooth_bowl, SiLU/GELU bei radial_ripple/ridge_mix (explorativ).


## V18 – Faire Lernrate, Tiefe 8/16/32 (Entwurf, eingefroren 25.10.2026)

Ziel: Prüfen, ob der PreCap-Vorteil in flachen bis mitteltiefen Netzen echt ist oder an der festen Lernrate 0,001 hängt. Zusätzlich: Ist der Vorsprung dort größer, wo PreCap mehr große Voraktivierungen hat (Taylor-Argument: PreCap ≈ x+0,51x², SiLU ≈ x/2+x²/4, GELU ≈ x/2+0,40x²)?

Design: 5 Aufgaben (ohne highfreq_grid) × Tiefen 8/16/32 × precap_base/SiLU/GELU × LR 3e-4/1e-3/3e-3 × 30 Seeds (9900–9929; 10 Tuning, 20 Bestätigung) = 4050 Runs, 32400 Zeilen, 3000 Schritte, 1/sqrt(D). LR je Aktivierung auf Tuning-Seeds gewählt, Test auf Bestätigungs-Seeds, 30 Wilcoxon-Tests mit Holm. H_A je Referenz: mindestens 8/15 Zellen besser, höchstens 1 schlechter. Mechanismus: Spearman(log-ratio gegen SiLU, Anteil |pre|>0,5) < 0.

Prüfungen: Formel, Aufgaben, Netz und 60-Schritt-Training bei LR 1e-3 identisch zu V17 (Abweichung 0,0); LR 3e-3 ändert das Ergebnis; Smoke 27/27 Zeilen; Neustart überspringt fertige Runs ohne Duplikate; Analyse auf Testdaten erkennt eingebauten Effekt. Ordner: `v18_lr_shallow/`.

## V18 Ergebnis (25.10.2026, 20:25 Uhr)
Lauf vollständig (4050 Runs, 32400 Zeilen, 0 nicht-endlich). Ein doppelter Start erzeugte 14416 identische Duplikatzeilen; sie wurden nach Sicherung entfernt, und die Analyse wurde neu ausgeführt (gleiches Urteil).
- Mit LR-Tuning je Aktivierung: 0 von 30 Tests PreCap besser, 2 schlechter, 28 unklar. H_A nicht bestätigt.
- SiLU und GELU wählen überall LR 3e-3 (Rasterrand). Bei fester LR 1e-3 repliziert der alte Vorteil klar (Ratio 0,03–0,7).
- Deutung: Der flache Vorteil war überwiegend ein Lernraten-/Skalierungseffekt (PreCap hat Steigung 1 am Ursprung, SiLU/GELU 0,5). Konfidenz etwa 0,85.
- Mechanismus gestützt: rho = −0,52, p = 0,023 (teilweise mit der Tiefe vermischt).
- Details: v18_lr_shallow/analysis/V18_Auswertung_ausfuehrlich.md

## V18 Ergebnis (25.10.2026, 20:25 Uhr)
Lauf vollständig (4050 Runs, 32400 Zeilen, 0 nicht-endlich). Ein doppelter Start erzeugte 14416 identische Duplikatzeilen; sie wurden nach Sicherung entfernt, und die Analyse wurde neu ausgeführt (gleiches Urteil).
- Mit LR-Tuning je Aktivierung: 0 von 30 Tests PreCap besser, 2 schlechter, 28 unklar. H_A nicht bestätigt.
- SiLU und GELU wählen überall LR 3e-3 (Rasterrand). Bei fester LR 1e-3 repliziert der alte Vorteil klar (Ratio 0,03–0,7).
- Deutung: Der flache Vorteil war überwiegend ein Lernraten-/Skalierungseffekt (PreCap hat Steigung 1 am Ursprung, SiLU/GELU 0,5). Konfidenz etwa 0,85.
- Mechanismus gestützt: rho = −0,52, p = 0,023 (teilweise mit der Tiefe vermischt).
- Details: v18_lr_shallow/analysis/V18_Auswertung_ausfuehrlich.md
