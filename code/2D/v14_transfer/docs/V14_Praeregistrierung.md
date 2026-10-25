# V14 Praeregistrierung (eingefrorener Plan nach Erweiterung)

## Anlass
V10-V13 bestaetigten s225 gegen base nur fuer eine Aufgabenfamilie (rotierte Sinus-Kosinus-Funktion, 53 Grad, a=0.25, k 0.7-2.0). Offen: Uebertragbarkeit auf andere Aufgabenfamilien und Vergleich zu SiLU/GELU.

## Frage
Ist precap_s225 (s=2.25) gegenueber precap_base (s=1.5) auch auf den sechs V3-Aufgaben besser, oder nur auf der rot-Familie?

## Design
- Aufgaben (exakt aus v3_large_preregistered/run_large_2d_v3.py, numerisch identisch geprueft): radial_ripple, ridge_mix, corner_kink, smooth_bowl, highfreq_grid, angled_kink.
- Aktivierungen: precap_base, precap_s225, silu, gelu.
- Tiefen 8, 16 und 32. Breite 64, Batch 128, Adam lr 1e-3, INIT_K 0.5, Architektur res_inv, 1200 Schritte, Checkpoints 0/50/100/200/400/800/1200.
- Seeds 9000-9019 (20 neu; Test-Seeds +10000). Disjunkt zu V1b-V13.
- Umfang: 6 x 3 x 20 x 4 = 1440 Runs, 10080 Checkpoint-Zeilen. Geschaetzte Dauer ca. 8-9 h mit 6 Workern; Nachtlauf ok, Netzteil Pflicht.

## Metrik
d = log10(test_MSE_A / test_MSE_B) bei Schritt 1200, gepaart je Seed. d<0: A besser.

## Konfirmatorisch (6 Tests, Holm, alpha 0.05)
Je Aufgabe: s225 gegen base. Pro Seed Mittel von d ueber alle drei Tiefen, Wilcoxon zweiseitig. Verdict je Aufgabe: s225 besser / schlechter / kein gesicherter Unterschied.

## Explorativ
Zellen (Aufgabe x Tiefe): s225 vs base, s225 vs silu, s225 vs gelu, base vs silu, base vs gelu.

## Interpretation vorab
- >=4 von 6 Aufgaben s225 besser und keine schlechter: Uebertragbarkeit gestuetzt.
- Gemischtes Bild: s225 aufgabenspezifisch.
- Mehrheit n.s.: kein Beleg ausserhalb der rot-Familie.
- SiLU/GELU-Vergleiche nur explorativ.
