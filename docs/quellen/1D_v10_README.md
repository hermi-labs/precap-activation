# v10 - Flacher Fairness-Check + PreCap-Verfeinerung

## Zweck
Zwei offene Fragen aus v9b:
1. Wie verhaelt sich PreCap in normalen, flachen Netzen (Tiefe 4 und 8) gegen SiLU/GELU/ReLU? (v9b hatte nur Tiefe 16/32/64; die alten Tiefe-8-Daten in v1-6 stammen von frueheren PreCap-Versionen.)
2. Welche PreCap-Parameter sind im Bereich um die v9b-Referenz (c=1.0, a=1.25, gamma=0.01, s=1.5) am besten, wenn Tiefe 32/64 mit res_inv laeuft?

## Teil A - flach (Tiefe 4, 8), 2880 Runs
- plain und res_inv, 3 Aufgaben, 10 Seeds
- silu, gelu, relu (init_k 0.5/1.0, lr 0.001/0.003)
- precap Referenz und precap(c=0.5,a=1.5,gamma=0) (init_k 0.5/1.0, lr 0.001/0.003)
- capexp c=1.0 und c=1.5 (init_k 0.25/0.5, lr 0.001)

## Teil B - PreCap-Verfeinerung (Tiefe 32, 64, res_inv), 3480 Runs
- c in 0.75/1.0/1.25, a in 1.0/1.25/1.5, gamma in 0/0.01/0.02, s in 1.5/2.0 (54 Configs), init_k=0.5, lr=0.001
- Baselines silu, gelu mit res_inv (init_k 0.5/1.0, lr 0.001)

Gesamt 6360 Runs, grob 9-11 Stunden auf 7 Shards.

## Start
cd /Users/aaronschmitt/PyCharmMiscProject/v10 && caffeinate -i bash run_v10.sh

Fortschritt: wc -l v10_results_shard*.csv (Resume ist eingebaut: einfach neu starten).

## Entscheidungsregeln
- Teil A: PreCap gilt als gleichauf, wenn der Median-Testloss der besten Config in der Naehe von SiLU/GELU liegt (Faktor < 2) bei null Ausfaellen.
- Teil B: Gewinner nach mittlerem Rang ueber 3 Aufgaben, nicht nach Einzelbestwert.

