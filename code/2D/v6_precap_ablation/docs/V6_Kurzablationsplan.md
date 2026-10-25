# V6 — Kurzablationsplan: Cap-Stärke und Weichheit

V5 zeigte keinen einfachen Gradienten-Crash, sondern aufgaben-, referenz- und seedabhängige späte Trainingspfade. V6 testet deshalb gezielt zwei mögliche, voneinander trennbare Beiträge: zu frühes/zu spätes Greifen des Caps und zu enge/zu weite weiche Eingangsbegrenzung.

Basisparameter: c=1.00, a=1.00, gamma=0.01, s=1.50.

Design: radial_ripple d64, ridge_mix d64, radial_ripple d16; Seeds 900–904; 1.200 Schritte; vier Worker.
