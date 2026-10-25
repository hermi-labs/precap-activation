# V12 Präregistrierung – Diagnose der s-Wirkung (Aufgabenfaktoren)

Stand: vor dem Start; Dateien per SHA-256 in meta/frozen_sha256.txt eingefroren.

## Ausgangslage
V10/V11: s225 (s=2,25) hilft bei geo_a025 (d16, d32) und rot_a025 (d16), nicht bei rot_a025 d32; H2/H3 aus V11 nicht bestätigt. Die Aufgaben unterscheiden sich in mehreren Faktoren gleichzeitig. V12 trennt Frequenz und Drehung. Es wird KEINE neue Aktivierung entwickelt; Ergebnis ist eine Landkarte der Wirkung von s.

## Design
- Aktivierungen: precap_base (s=1,5), precap_s225 (s=2,25), precap_s300 (s=3,0, nur Dosiskontrolle); c=1, a=1, gamma=0,01 unverändert.
- Aufgaben (Mischparameter a=0,25 fix): f = sin(k(2(1-a)r²+5a·u)) + 0,3·cos(k(3(1-a)·sqrt(r²+0,05)+2a·v)), u=c·x+s·y, v=s·x−c·y.
  - rot_ang53_k10_a025: Anker, identisch zu V11 rot_a025 (c=0,6; s=0,8; k=1,0)
  - rot_ang00_k10_a025: Drehung 0 (c=1; s=0), k=1,0
  - rot_ang53_k15_a025: Frequenzfaktor k=1,5
  - rot_ang53_k07_a025: Frequenzfaktor k=0,7
- Tiefen 16 und 32; Breite 64; 1200 Schritte; Batch 128; Adam lr 0,001; Init 0,5; 2048 Train / 512 Test.
- Seeds 7000–7019 (20), disjunkt zu allen früheren Versionen. Runs: 4 Aufgaben × 2 Tiefen × 3 Aktivierungen × 20 Seeds = 480; Checkpoints 0,50,100,200,400,800,1200 → 3360 Zeilen.
- Maß pro Seed: d = log10(MSE_test(s225)/MSE_test(base)) bei Schritt 1200; d<0 = Vorteil für s225.

## Hypothesen (konfirmatorisch)
- R (Replikation, 1 Test): Anker bei Tiefe 16: Median d<0 und einseitiger Wilcoxon p<0,05.
- K1–K6 (Kontraste, 6 Tests, gepaart über Seeds): Differenz d(Aufgabe) − d(Anker) für die drei Nicht-Anker-Aufgaben je Tiefe; zweiseitiger Wilcoxon, Holm-Korrektur über alle 6, alpha 0,05. Ein signifikanter Kontrast bedeutet: der entsprechende Faktor (Drehung bzw. Frequenz) verändert die s225-Wirkung.

## Explorativ (keine Hypothesentests)
Einzelzellen-Tests von s225 und s300 gegen base, Dosisvergleich s300 vs. s225, Pre-Activation p99 (Spearman, zentriert pro Zelle). Aussagen darüber werden als explorativ gekennzeichnet.

## Entscheidungsregeln
- Wird R nicht erfüllt, gilt der V11-Anker als nicht robust; Kontraste werden nur vorsichtig gelesen.
- Kein Kontrast signifikant: die Faktoren Drehung/Frequenz erklären die Unterschiede nicht; weitere Faktoren (z. B. a, Komplexität) wären der nächste Schritt.
- Die Auswertung wird nach Datenansicht nicht geändert. Abweichungen werden separat als Post-hoc gekennzeichnet.

## Rechenzeit
V11 (240 Runs) brauchte ca. 55 Minuten mit 6 Workern; V12 hat 480 Runs, geschätzt 1,5–2 Stunden.

## Start
cd 2D/v12_diagnosis && nohup ./launch_v12.sh 6 > logs/launcher.log 2>&1 &
