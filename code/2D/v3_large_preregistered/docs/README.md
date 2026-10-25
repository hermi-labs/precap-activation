# CapExp v3 — große, präregristrierte 2D-Studie

**Noch nicht gestartet.** Dieser Ordner ist von v2 getrennt und überschreibt keine bisherigen Ergebnisse.

- **900 Runs:** 6 Aufgaben × 5 Tiefen × 3 Aktivierungen × 10 neue Seeds.
- **Formel:** unveränderte, validierte PreCap-v9-Urformel; Details in `../v2_holdout/docs/PreCap_Urformel_Spezifikation.md`.
- **Kernentscheidung:** ausschließlich die drei bisherigen Kernaufgaben; neue Aufgaben werden separat berichtet.
- **Tiefe:** 4, 8, 16, 32, 64. Damit wird direkt geprüft, ob sich das beobachtete Abschwächen, Stabilisieren oder Umkehren des Effekts mit Tiefe fortsetzt.

## Start

```zsh
cd /Users/aaronschmitt/PyCharmMiscProject/2D/v3_large_preregistered
./launch.sh
```

## Status

```zsh
/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python status.py
tail -20 logs/progress.log
```

Nach `Done True` ausführen:

```zsh
/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python analyze_large_v3.py
```
