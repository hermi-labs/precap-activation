# V16 Auswertung (preregistered)

## Konfirmatorisch (Schritt 3000, Tiefe 64/128, base vs SiLU/GELU, Holm ueber 24)

step,task,depth,cmp,n,ratio,ci90_lo,ci90_hi,seeds_first_better,p,equivalent,holm,verdict
3000,angled_kink,64,precap_base_vs_silu,20,0.5336,0.3867,1.2188,12,0.1327,False,1.0,unklar
3000,angled_kink,64,precap_base_vs_gelu,20,0.8524,0.4545,1.9077,11,0.5217,False,1.0,unklar
3000,angled_kink,128,precap_base_vs_silu,20,0.6195,0.4453,1.1545,13,0.2611,False,1.0,unklar
3000,angled_kink,128,precap_base_vs_gelu,20,0.5967,0.5089,0.7841,15,0.0583,False,1.0,unklar
3000,corner_kink,64,precap_base_vs_silu,20,0.5167,0.4528,0.7416,17,0.0001,False,0.0019,PreCap besser
3000,corner_kink,64,precap_base_vs_gelu,20,1.1528,1.0364,1.5438,6,0.0583,False,1.0,unklar
3000,corner_kink,128,precap_base_vs_silu,20,0.7049,0.6405,0.9013,15,0.0107,False,0.2245,unklar
3000,corner_kink,128,precap_base_vs_gelu,20,1.9379,1.6958,2.1571,0,0.0,False,0.0,PreCap schlechter
3000,highfreq_grid,64,precap_base_vs_silu,20,1.0111,0.9558,1.0547,9,0.8408,True,1.0,aequivalent (90%-KI in +-10%)
3000,highfreq_grid,64,precap_base_vs_gelu,20,1.0274,1.0023,1.0521,6,0.1054,True,1.0,aequivalent (90%-KI in +-10%)
3000,highfreq_grid,128,precap_base_vs_silu,20,1.0436,1.0297,1.0663,4,0.0192,True,0.3847,aequivalent (90%-KI in +-10%)
3000,highfreq_grid,128,precap_base_vs_gelu,20,1.0534,1.0078,1.0756,4,0.0037,True,0.0804,aequivalent (90%-KI in +-10%)
3000,radial_ripple,64,precap_base_vs_silu,20,0.801,0.2873,3.1775,10,0.7285,False,1.0,unklar
3000,radial_ripple,64,precap_base_vs_gelu,20,1.1767,0.8253,2.1489,8,0.5706,False,1.0,unklar
3000,radial_ripple,128,precap_base_vs_silu,20,0.8677,0.4912,1.2653,10,0.4749,False,1.0,unklar
3000,radial_ripple,128,precap_base_vs_gelu,20,1.0734,0.7229,1.5035,9,0.7285,False,1.0,unklar
3000,ridge_mix,64,precap_base_vs_silu,20,0.5201,0.2808,1.0078,13,0.1231,False,1.0,unklar
3000,ridge_mix,64,precap_base_vs_gelu,20,0.6458,0.463,1.4301,12,0.33,False,1.0,unklar
3000,ridge_mix,128,precap_base_vs_silu,20,1.2257,0.4183,1.8161,9,0.7841,False,1.0,unklar
3000,ridge_mix,128,precap_base_vs_gelu,20,1.7829,1.0437,2.1801,6,0.2774,False,1.0,unklar
3000,smooth_bowl,64,precap_base_vs_silu,20,1.0627,0.3274,2.7952,9,0.5958,False,1.0,unklar
3000,smooth_bowl,64,precap_base_vs_gelu,20,1.0344,0.5483,1.2699,9,0.7285,False,1.0,unklar
3000,smooth_bowl,128,precap_base_vs_silu,20,1.4825,0.9348,5.6009,7,0.0532,False,1.0,unklar
3000,smooth_bowl,128,precap_base_vs_gelu,20,1.2416,0.4152,2.066,9,0.8124,False,1.0,unklar


## Verdict

{
  "final_step": 3000,
  "n_confirmatory": 24,
  "counts": {
    "unklar": 18,
    "aequivalent (90%-KI in +-10%)": 4,
    "PreCap besser": 1,
    "PreCap schlechter": 1
  },
  "by_depth": {
    "64": {
      "unklar": 9,
      "aequivalent (90%-KI in +-10%)": 2,
      "PreCap besser": 1
    },
    "128": {
      "unklar": 9,
      "aequivalent (90%-KI in +-10%)": 2,
      "PreCap schlechter": 1
    }
  },
  "spearman_fracpre_vs_absd": {
    "rho": -0.08391608391608392,
    "p": 0.7954145018538977,
    "n": 12
  },
  "learning_curve": [
    {
      "depth": 32,
      "step": 0,
      "geo_ratio_base_over_silu": 1.0953
    },
    {
      "depth": 32,
      "step": 100,
      "geo_ratio_base_over_silu": 0.8849
    },
    {
      "depth": 32,
      "step": 200,
      "geo_ratio_base_over_silu": 0.4934
    },
    {
      "depth": 32,
      "step": 400,
      "geo_ratio_base_over_silu": 0.5832
    },
    {
      "depth": 32,
      "step": 800,
      "geo_ratio_base_over_silu": 0.4539
    },
    {
      "depth": 32,
      "step": 1200,
      "geo_ratio_base_over_silu": 0.4977
    },
    {
      "depth": 32,
      "step": 2000,
      "geo_ratio_base_over_silu": 0.5594
    },
    {
      "depth": 32,
      "step": 3000,
      "geo_ratio_base_over_silu": 0.4303
    },
    {
      "depth": 64,
      "step": 0,
      "geo_ratio_base_over_silu": 1.1583
    },
    {
      "depth": 64,
      "step": 100,
      "geo_ratio_base_over_silu": 0.9371
    },
    {
      "depth": 64,
      "step": 200,
      "geo_ratio_base_over_silu": 0.6906
    },
    {
      "depth": 64,
      "step": 400,
      "geo_ratio_base_over_silu": 0.6437
    },
    {
      "depth": 64,
      "step": 800,
      "geo_ratio_base_over_silu": 0.6546
    },
    {
      "depth": 64,
      "step": 1200,
      "geo_ratio_base_over_silu": 0.6562
    },
    {
      "depth": 64,
      "step": 2000,
      "geo_ratio_base_over_silu": 0.856
    },
    {
      "depth": 64,
      "step": 3000,
      "geo_ratio_base_over_silu": 0.8875
    },
    {
      "depth": 128,
      "step": 0,
      "geo_ratio_base_over_silu": 1.4561
    },
    {
      "depth": 128,
      "step": 100,
      "geo_ratio_base_over_silu": 0.9778
    },
    {
      "depth": 128,
      "step": 200,
      "geo_ratio_base_over_silu": 0.8903
    },
    {
      "depth": 128,
      "step": 400,
      "geo_ratio_base_over_silu": 0.7156
    },
    {
      "depth": 128,
      "step": 800,
      "geo_ratio_base_over_silu": 0.7367
    },
    {
      "depth": 128,
      "step": 1200,
      "geo_ratio_base_over_silu": 0.7875
    },
    {
      "depth": 128,
      "step": 2000,
      "geo_ratio_base_over_silu": 0.9245
    },
    {
      "depth": 128,
      "step": 3000,
      "geo_ratio_base_over_silu": 1.0253
    }
  ]
}

Weitere Tabellen: v16_all_comparisons.csv, v16_mechanism_medians.csv, v16_learning_curve_ratio.csv
