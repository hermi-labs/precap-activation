# V15 Praeregistrierung (Ausnahmezelle)

## Frage
Ist die Ausnahmezelle rot_k15 bei Tiefe 32 ein echter Effekt oder nur Seed-Instabilitaet?

## Anlass
In der Zelle k=1.5, Tiefe 32 wichen die Ergebnisse zwischen V12 und V13 stark ab (Verhaeltnis ca. 1.50 vs 1.00). Diese Instabilitaet soll mit deutlich mehr Seeds direkt geklaert werden.

## Design
- Aufgabe: rot_k15.
- Tiefe: 32.
- Aktivierungen: precap_base (s=1.5) und precap_s225 (s=2.25).
- Seeds: 9500-9599 (100 neue Seeds; Test-Seeds +10000), disjunkt zu V1b-V14.
- Sonst identisch zu V13: Breite 64, Batch 128, Adam lr 1e-3, INIT_K 0.5, 1200 Schritte, Checkpoints 0/50/100/200/400/800/1200.
- Umfang: 200 Runs, 1400 Checkpoint-Zeilen.

## Hauptmetrik
d = log10(test_MSE_s225 / test_MSE_base) bei Schritt 1200, gepaart je Seed. d<0 bedeutet Vorteil fuer s225.

## Auswertung
- Primar: Wilcoxon signed-rank, zweiseitig.
- Sekundaer: gerichteter Test fuer s225 besser.
- Bericht: Median d, Verhaeltnis 10^Median(d), Anzahl Seeds mit Vorteil fuer s225 vs base, Quantile, komplette Seed-Liste.

## Interpretation vorab
- Klar negatives d mit kleinem p: echte s225-Ueberlegenheit in dieser Zelle.
- d nahe 0 mit breiter Verteilung: die Zelle ist instabil und der fruehere Widerspruch war Seedrauschen.
- Positives d: spricht gegen die Generalitaet von s225 und fuer echte Ausnahmen.
