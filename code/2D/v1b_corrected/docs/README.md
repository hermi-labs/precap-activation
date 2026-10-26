# 2D v1b — korrigierter Pilot

Der frühere Ordner `2D/v1_pilot` ist **ungültig für PreCap**: Dort wurde versehentlich eine andere, SiLU-basierte Cap-Funktion implementiert. Seine Dateien bleiben zur Nachvollziehbarkeit erhalten.

`v1b_corrected` übernimmt die eingefrorene 1D-v9-Formel wörtlich:

$$
z=ax,\quad u=z+\gamma z^2,\quad v=s\tanh(u/s),\quad \mathrm{PreCap}(x)=\mathrm{CapExp}_{c,1}(v)
$$

mit c=1, a=1, gamma=0,01, s=1,5. Die Residualstruktur übernimmt ebenfalls die 1D-Definition einschließlich der Skalierung $1/\sqrt{L}$. Vor dem Start prüft `selfcheck` die Funktionsgleichheit auf 10.001 Eingabewerten und die Form des Netzes für alle Tiefen.
