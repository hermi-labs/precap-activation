# V2g – abgebrochen vor dem Training (21.11.2026, 09:32, mit Claude)

`start_nach_v2f.sh` hat V2g nach V2f (36/36) um 09:28 gestartet. Der Freeze steht in `meta/frozen_sha256.txt`. Der Selbsttest ist danach abgebrochen:

    TypeError: Object of type bool is not JSON serializable when serializing dict item 'bias_ok'

`bias_ok` war ein numpy-bool. Trainiert wurde nichts, Ergebnisse gibt es keine. Weil die Dateien eingefroren sind, läuft die Studie unverändert als Fassung b in `../v2gb/` weiter. Dort sind nur dieser Fehler behoben, die Namen angepasst und die Quellenangabe zu σ_b² präzisiert. Die Einzelheiten stehen im Versionshinweis in `../v2gb/V2gb_Praeregistrierung.md`.
