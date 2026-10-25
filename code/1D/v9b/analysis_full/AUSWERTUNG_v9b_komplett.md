# v9b komplette Auswertung

Runs: 12600 (3 Aufgaben x 3 Tiefen x 140 Konfigs x 10 Seeds). Ausfall = NaN/Inf, wird als Verlust unendlich gewertet.

## Ausfallraten

```
   act    arch       16       32       64
capexp   plain 0.443000 0.500000 0.342000
capexp res_inv 0.294000 0.391000 0.543000
  gelu   plain 0.000000 0.000000 0.000000
  gelu res_inv 0.000000 0.000000 0.000000
precap   plain 0.000000 0.000000 0.000000
precap res_inv 0.000000 0.000000 0.000000
  relu   plain 0.000000 0.000000 0.000000
  relu res_inv 0.000000 0.000000 0.000000
  silu   plain 0.000000 0.000000 0.000000
  silu res_inv 0.000000 0.000000 0.000000
```

## Sieger je Aufgabe und Tiefe

```
   target  depth    act        c        a    gamma   soft_s    arch   init_k       lr  fails      med
 abs_wave     16 precap 1.000000 1.250000 0.010000 1.500000 res_inv 1.000000 0.001000      0 0.000045
 abs_wave     32 precap 1.000000 1.250000 0.010000 1.500000 res_inv 1.000000 0.001000      0 0.000054
 abs_wave     64 precap 1.000000 1.250000 0.010000 1.500000 res_inv 0.500000 0.001000      0 0.000045
sin3_quad     16 capexp 1.500000 1.000000 0.000000 1.000000 res_inv 0.500000 0.001000      0 0.000014
sin3_quad     32 capexp 1.500000 1.000000 0.000000 1.000000 res_inv 0.250000 0.001000      0 0.000017
sin3_quad     64 capexp 1.500000 1.000000 0.000000 1.000000 res_inv 0.250000 0.000300      0 0.000014
  sin_mix     16   silu 0.000000 1.000000 0.000000 1.000000   plain 1.000000 0.000300      0 0.000144
  sin_mix     32 capexp 1.500000 1.000000 0.000000 1.000000 res_inv 0.500000 0.001000      2 0.000389
  sin_mix     64 precap 1.000000 1.250000 0.010000 1.500000 res_inv 0.500000 0.003000      0 0.000421
```

## Robustheit Tiefe 16 (mittlerer Rang ueber 3 Aufgaben, Top 8)

```
   act        c        a    gamma   soft_s    arch   init_k       lr     worst      mean  fails
  silu 0.000000 1.000000 0.000000 1.000000   plain 1.000000 0.000300 23.000000 10.000000      0
  silu 0.000000 1.000000 0.000000 1.000000   plain 1.000000 0.001000 15.000000 10.000000      0
  gelu 0.000000 1.000000 0.000000 1.000000   plain 1.000000 0.001000 19.000000 10.333333      0
  gelu 0.000000 1.000000 0.000000 1.000000   plain 1.000000 0.000300 19.000000 12.333333      0
precap 1.000000 1.000000 0.010000 1.500000 res_inv 1.000000 0.001000 25.000000 13.333333      0
capexp 1.500000 1.000000 0.000000 1.000000 res_inv 0.500000 0.001000 34.000000 17.000000      0
capexp 1.500000 1.000000 0.000000 1.000000 res_inv 0.250000 0.001000 42.000000 17.333333      0
precap 1.000000 1.250000 0.010000 1.500000 res_inv 1.000000 0.000300 32.000000 21.333333      0
```

## Robustheit Tiefe 32 (mittlerer Rang ueber 3 Aufgaben, Top 8)

```
   act        c        a    gamma   soft_s    arch   init_k       lr     worst      mean  fails
precap 1.000000 1.250000 0.010000 1.500000 res_inv 1.000000 0.001000 11.000000  6.333333      0
precap 1.000000 1.000000 0.010000 1.500000 res_inv 1.000000 0.001000 17.000000  8.666667      0
precap 1.000000 1.250000 0.010000 1.500000 res_inv 0.500000 0.003000 21.000000 11.000000      0
precap 1.000000 1.000000 0.010000 1.500000 res_inv 0.500000 0.003000 15.000000 11.333333      0
capexp 1.250000 1.000000 0.000000 1.000000 res_inv 0.500000 0.001000 20.000000 13.333333      4
precap 1.000000 1.250000 0.010000 1.500000 res_inv 1.000000 0.000300 21.000000 13.333333      0
precap 1.000000 1.000000 0.010000 1.500000 res_inv 1.000000 0.003000 29.000000 14.666667      0
precap 0.500000 1.500000 0.000000 1.500000 res_inv 1.000000 0.001000 32.000000 15.000000      0
```

## Robustheit Tiefe 64 (mittlerer Rang ueber 3 Aufgaben, Top 8)

```
   act        c        a    gamma   soft_s    arch   init_k       lr     worst      mean  fails
precap 1.000000 1.250000 0.010000 1.500000 res_inv 0.500000 0.001000 11.000000  6.333333      0
precap 1.000000 1.250000 0.010000 1.500000 res_inv 0.500000 0.003000 20.000000  7.666667      0
precap 1.000000 1.000000 0.010000 1.500000 res_inv 0.500000 0.001000 20.000000  8.333333      0
precap 1.000000 1.000000 0.010000 1.500000 res_inv 1.000000 0.001000 13.000000  9.666667      0
capexp 1.500000 1.000000 0.000000 1.000000 res_inv 0.250000 0.000300 21.000000 11.000000      1
precap 1.000000 1.250000 0.010000 1.500000 res_inv 1.000000 0.000300 16.000000 11.333333      0
precap 0.500000 1.500000 0.000000 1.500000 res_inv 0.500000 0.003000 26.000000 11.666667      0
precap 1.000000 1.000000 0.010000 1.500000 res_inv 1.000000 0.003000 17.000000 12.333333      0
```

## Bester Median je Aktivierung/Architektur

```
   target  depth    act    arch      med  fails        c        a    gamma   soft_s   init_k       lr
 abs_wave     16 precap res_inv 0.000045      0 1.000000 1.250000 0.010000 1.500000 1.000000 0.001000
 abs_wave     16 capexp res_inv 0.000067      0 1.500000 1.000000 0.000000 1.000000 0.250000 0.001000
 abs_wave     16   gelu   plain 0.000082      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.001000
 abs_wave     16   silu   plain 0.000122      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.001000
 abs_wave     16 capexp   plain 0.000155      0 1.000000 1.000000 0.000000 1.000000 0.500000 0.001000
 abs_wave     16 precap   plain 0.000199      0 1.000000 1.000000 0.010000 1.500000 1.000000 0.000300
 abs_wave     16   silu res_inv 0.000275      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.003000
 abs_wave     16   relu   plain 0.000648      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
 abs_wave     16   gelu res_inv 0.000813      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
 abs_wave     16   relu res_inv 0.001630      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
 abs_wave     32 precap res_inv 0.000054      0 1.000000 1.250000 0.010000 1.500000 1.000000 0.001000
 abs_wave     32   gelu   plain 0.000072      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
 abs_wave     32   silu   plain 0.000092      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
 abs_wave     32 capexp res_inv 0.000141      3 1.250000 1.000000 0.000000 1.000000 0.500000 0.001000
 abs_wave     32   silu res_inv 0.000177      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.003000
 abs_wave     32 precap   plain 0.000335      0 1.000000 1.000000 0.010000 1.500000 0.500000 0.000300
 abs_wave     32   gelu res_inv 0.000393      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.001000
 abs_wave     32   relu   plain 0.000675      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
 abs_wave     32 capexp   plain 0.001457      0 0.750000 1.000000 0.000000 1.000000 0.500000 0.000100
 abs_wave     32   relu res_inv 0.002055      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.001000
 abs_wave     64 precap res_inv 0.000045      0 1.000000 1.250000 0.010000 1.500000 0.500000 0.001000
 abs_wave     64   silu res_inv 0.000107      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.001000
 abs_wave     64 capexp res_inv 0.000108      0 1.500000 1.000000 0.000000 1.000000 0.250000 0.000300
 abs_wave     64   gelu res_inv 0.000250      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
 abs_wave     64 precap   plain 0.001551      0 1.000000 1.000000 0.010000 1.500000 1.000000 0.000300
 abs_wave     64   relu res_inv 0.001771      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
 abs_wave     64   relu   plain 0.001801      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
 abs_wave     64 capexp   plain 0.155321      0 0.750000 1.000000 0.000000 1.000000 0.250000 0.000100
 abs_wave     64   gelu   plain 0.155321      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
 abs_wave     64   silu   plain 0.155321      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
sin3_quad     16 capexp res_inv 0.000014      0 1.500000 1.000000 0.000000 1.000000 0.500000 0.001000
sin3_quad     16 precap res_inv 0.000029      0 1.000000 1.000000 0.010000 1.500000 0.500000 0.000300
sin3_quad     16   silu   plain 0.000029      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
sin3_quad     16   gelu   plain 0.000032      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
sin3_quad     16 precap   plain 0.000051      0 1.000000 1.000000 0.010000 1.500000 0.500000 0.000300
sin3_quad     16 capexp   plain 0.000076      0 0.750000 1.000000 0.000000 1.000000 0.500000 0.000300
sin3_quad     16   silu res_inv 0.000140      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.001000
sin3_quad     16   gelu res_inv 0.000461      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.001000
sin3_quad     16   relu   plain 0.000646      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
sin3_quad     16   relu res_inv 0.000955      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
sin3_quad     32 capexp res_inv 0.000017      0 1.500000 1.000000 0.000000 1.000000 0.250000 0.001000
sin3_quad     32 precap res_inv 0.000017      0 1.000000 1.000000 0.010000 1.500000 1.000000 0.001000
sin3_quad     32   silu   plain 0.000043      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
sin3_quad     32   silu res_inv 0.000058      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.003000
sin3_quad     32   gelu   plain 0.000065      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
sin3_quad     32 capexp   plain 0.000102      0 0.750000 1.000000 0.000000 1.000000 0.500000 0.000300
sin3_quad     32   gelu res_inv 0.000162      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
sin3_quad     32 precap   plain 0.000235      0 1.000000 1.250000 0.010000 1.500000 0.500000 0.000300
sin3_quad     32   relu   plain 0.001142      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
sin3_quad     32   relu res_inv 0.001448      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
sin3_quad     64 capexp res_inv 0.000014      0 1.500000 1.000000 0.000000 1.000000 0.250000 0.000300
sin3_quad     64 precap res_inv 0.000017      0 1.000000 1.000000 0.010000 1.500000 0.500000 0.001000
sin3_quad     64   silu res_inv 0.000040      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.001000
sin3_quad     64   gelu res_inv 0.000133      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
sin3_quad     64   relu   plain 0.001877      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
sin3_quad     64   relu res_inv 0.001964      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
sin3_quad     64 precap   plain 0.004501      0 1.000000 1.000000 0.010000 1.500000 1.000000 0.000300
sin3_quad     64 capexp   plain 0.655996      0 0.750000 1.000000 0.000000 1.000000 0.250000 0.000100
sin3_quad     64   gelu   plain 0.655996      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
sin3_quad     64   silu   plain 0.655996      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.001000
  sin_mix     16   silu   plain 0.000144      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
  sin_mix     16   gelu   plain 0.000151      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.001000
  sin_mix     16 capexp   plain 0.000206      0 1.250000 1.000000 0.000000 1.000000 0.500000 0.000300
  sin_mix     16 precap   plain 0.000268      0 1.000000 1.250000 0.010000 1.500000 1.000000 0.000300
  sin_mix     16 precap res_inv 0.000461      0 1.000000 1.250000 0.010000 1.500000 1.000000 0.003000
  sin_mix     16   relu   plain 0.000610      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.001000
  sin_mix     16 capexp res_inv 0.000666      0 1.500000 1.000000 0.000000 1.000000 0.500000 0.000300
  sin_mix     16   gelu res_inv 0.001097      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
  sin_mix     16   silu res_inv 0.001263      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
  sin_mix     16   relu res_inv 0.001942      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
  sin_mix     32 capexp res_inv 0.000389      2 1.500000 1.000000 0.000000 1.000000 0.500000 0.001000
  sin_mix     32 precap   plain 0.000396      0 1.000000 1.000000 0.010000 1.500000 1.000000 0.000300
  sin_mix     32 precap res_inv 0.000467      0 0.500000 1.500000 0.000000 1.500000 1.000000 0.001000
  sin_mix     32   relu   plain 0.000588      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
  sin_mix     32   gelu res_inv 0.001065      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.003000
  sin_mix     32   gelu   plain 0.001157      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.000300
  sin_mix     32   relu res_inv 0.001161      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
  sin_mix     32   silu res_inv 0.001200      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
  sin_mix     32   silu   plain 0.001515      0 0.000000 1.000000 0.000000 1.000000 1.000000 0.003000
  sin_mix     32 capexp   plain 0.027425      2 0.750000 1.000000 0.000000 1.000000 0.500000 0.001000
  sin_mix     64 precap res_inv 0.000421      0 1.000000 1.250000 0.010000 1.500000 0.500000 0.003000
  sin_mix     64 precap   plain 0.000516      0 1.000000 1.000000 0.010000 1.500000 1.000000 0.000300
  sin_mix     64 capexp res_inv 0.000600      3 1.000000 1.000000 0.000000 1.000000 0.500000 0.001000
  sin_mix     64   silu res_inv 0.000770      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
  sin_mix     64   gelu res_inv 0.001152      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.001000
  sin_mix     64   relu res_inv 0.001873      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.003000
  sin_mix     64 capexp   plain 0.481859      0 0.750000 1.000000 0.000000 1.000000 0.250000 0.000100
  sin_mix     64   gelu   plain 0.481859      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
  sin_mix     64   relu   plain 0.481859      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
  sin_mix     64   silu   plain 0.481859      0 0.000000 1.000000 0.000000 1.000000 0.500000 0.000300
```

## Typische Konfiguration (Median der Konfig-Mediane)

```
   target  depth    act    arch      med
 abs_wave     16 precap res_inv 0.000081
 abs_wave     16   silu   plain 0.000324
 abs_wave     16   silu res_inv 0.000376
 abs_wave     16   gelu   plain 0.000423
 abs_wave     16 precap   plain 0.001129
 abs_wave     16   relu   plain 0.001272
 abs_wave     16 capexp res_inv 0.002462
 abs_wave     16   gelu res_inv 0.002572
 abs_wave     16   relu res_inv 0.026093
 abs_wave     16 capexp   plain 0.038193
 abs_wave     32 precap res_inv 0.000104
 abs_wave     32   silu res_inv 0.000354
 abs_wave     32   gelu res_inv 0.001243
 abs_wave     32   relu   plain 0.001848
 abs_wave     32 capexp res_inv 0.003833
 abs_wave     32 precap   plain 0.025848
 abs_wave     32   relu res_inv 0.028385
 abs_wave     32   silu   plain 0.078014
 abs_wave     32   gelu   plain 0.078242
 abs_wave     32 capexp   plain 0.155321
 abs_wave     64 precap res_inv 0.000090
 abs_wave     64   silu res_inv 0.000211
 abs_wave     64   gelu res_inv 0.000665
 abs_wave     64   relu res_inv 0.020065
 abs_wave     64 capexp res_inv 0.026644
 abs_wave     64 precap   plain 0.155321
 abs_wave     64 capexp   plain 0.155321
 abs_wave     64   gelu   plain 0.155321
 abs_wave     64   relu   plain 0.155321
 abs_wave     64   silu   plain 0.155321
sin3_quad     16   gelu   plain 0.000068
sin3_quad     16 capexp res_inv 0.000100
sin3_quad     16 precap res_inv 0.000108
sin3_quad     16 capexp   plain 0.000146
sin3_quad     16   silu   plain 0.000166
sin3_quad     16   silu res_inv 0.000202
sin3_quad     16   gelu res_inv 0.000702
sin3_quad     16 precap   plain 0.000725
sin3_quad     16   relu   plain 0.001404
sin3_quad     16   relu res_inv 0.002571
sin3_quad     32 precap res_inv 0.000052
sin3_quad     32 capexp res_inv 0.000072
sin3_quad     32   silu res_inv 0.000122
sin3_quad     32   gelu res_inv 0.000514
sin3_quad     32   relu   plain 0.002226
sin3_quad     32 precap   plain 0.002276
sin3_quad     32   relu res_inv 0.003001
sin3_quad     32   gelu   plain 0.328347
sin3_quad     32   silu   plain 0.328479
sin3_quad     32 capexp   plain 0.655996
sin3_quad     64 precap res_inv 0.000041
sin3_quad     64 capexp res_inv 0.000046
sin3_quad     64   silu res_inv 0.000082
sin3_quad     64   gelu res_inv 0.000315
sin3_quad     64   relu res_inv 0.003237
sin3_quad     64 capexp   plain 0.655996
sin3_quad     64 precap   plain 0.655996
sin3_quad     64   relu   plain 0.655996
sin3_quad     64   silu   plain 0.655996
sin3_quad     64   gelu   plain 0.655996
  sin_mix     16   gelu   plain 0.000536
  sin_mix     16   silu   plain 0.000579
  sin_mix     16 precap res_inv 0.000608
  sin_mix     16   relu   plain 0.000710
  sin_mix     16 precap   plain 0.000937
  sin_mix     16 capexp res_inv 0.001296
  sin_mix     16   gelu res_inv 0.001512
  sin_mix     16   silu res_inv 0.001764
  sin_mix     16   relu res_inv 0.006108
  sin_mix     16 capexp   plain 0.035460
  sin_mix     32 precap res_inv 0.000694
  sin_mix     32 capexp res_inv 0.001119
  sin_mix     32   gelu res_inv 0.001367
  sin_mix     32   silu res_inv 0.001616
  sin_mix     32   relu res_inv 0.008197
  sin_mix     32   relu   plain 0.241763
  sin_mix     32   gelu   plain 0.241799
  sin_mix     32   silu   plain 0.241903
  sin_mix     32 precap   plain 0.331951
  sin_mix     32 capexp   plain 0.481859
  sin_mix     64 precap res_inv 0.000639
  sin_mix     64 capexp res_inv 0.001157
  sin_mix     64   gelu res_inv 0.001287
  sin_mix     64   silu res_inv 0.001400
  sin_mix     64   relu res_inv 0.008911
  sin_mix     64 capexp   plain 0.481859
  sin_mix     64   gelu   plain 0.481859
  sin_mix     64 precap   plain 0.481859
  sin_mix     64   relu   plain 0.481859
  sin_mix     64   silu   plain 0.481859
```

Hinweis: Beste Einzelwerte sind durch die Auswahl aus 140 Konfigs optimistisch; Rangwerte und typische Werte sind robuster.
