# Kernel: eigenständige Test-Fassung

Dieser Ordner ist die Fassung, mit der die Kernel auf echter Hardware geprüft wurden. Die Messergebnisse liegen als JSON daneben. Für den normalen Einsatz bitte das Paket nutzen (`from precap import PreCap`, `from precap.mlx import PreCap`, siehe Haupt-README). Es enthält denselben Kernel-Code und zusätzlich das Preset `spreccap_e03`.

## Dateien

| Datei | Inhalt |
|---|---|
| `reference.py` | Referenz in reinem PyTorch (Autograd) und analytische Ableitung. Maßstab für alle Tests |
| `precap_cuda.py` | fusionierter CUDA-Kernel über `torch.cuda.jiterator` (NVRTC zur Laufzeit, kein CUDA-Toolkit, kein Visual Studio nötig) |
| `precap_mlx.py` | fusionierter Metal-Kernel über `mx.fast.metal_kernel` mit eigenem Rückwärtsschritt (`mx.custom_function`) |
| `test_cuda.py`, `test_mlx.py` | Tests: Werte, Gradienten, gradcheck (fp64, nur CUDA), Randwerte bis ±1e4, kleines 20-Schichten-MLP Kernel gegen Referenz, Tempo |
| `test_cuda_ergebnis_rtx4070.json` | 17/17 OK, RTX 4070, torch 2.11.0+cu128, Python 3.12.10 |
| `test_mlx_ergebnis_m1pro.json` | 12/12 OK, M1 Pro, mlx 0.32.4, Python 3.15 |

## Ausführen

Die Tests laufen direkt aus diesem Ordner, ohne Pfade anzupassen.

```bash
# Windows / Linux mit NVIDIA-GPU
python -m venv .venv && .venv/Scripts/activate     # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
cd kernels && python test_cuda.py

# macOS mit Apple Silicon
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy
cd kernels && python test_mlx.py
```

## Ergebnisse (16M Elemente, vorwärts + rückwärts, float32)

| Gerät | Kernel | Referenz | natives SELU | Faktor gegen Referenz |
|---|---|---|---|---|
| RTX 4070 | 0,909 ms | 10,463 ms | 0,89 ms | 11,5× |
| M1 Pro | 4,15 ms | 34,1 ms | 11,47 ms | 8,2× |

Größte relative Abweichung zur Referenz in fp32: höchstens 1,1e-6 (CUDA).

## Hinweise

- **Datentypen CUDA:** float32 und float64 nativ. float16/bfloat16 werden über float32 gerechnet und wieder zurückgewandelt.
- **Datentypen MLX:** Der Kernel rechnet und schreibt immer float32. Diese Test-Fassung gibt float32 zurück. Das Paket (`precap.mlx`) wandelt in den Eingangstyp zurück, auf dem M1 Pro geprüft für float16.
- Der Knick bei 0 (Faktor α) und der Übergang bei w = c sind Stellen ohne stetige zweite Ableitung. Für gradcheck werden Punkte dort gemieden.
- Die MLX-Messung lief parallel zu einem Trainingslauf und ist daher eher zu langsam.
