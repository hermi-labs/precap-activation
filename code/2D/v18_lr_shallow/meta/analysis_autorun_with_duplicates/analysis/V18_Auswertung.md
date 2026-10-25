# V18 Auswertung (automatisch)

## Konfirmatorisch

task,depth,ref,n,lr_precap,lr_ref,ratio,ci90_lo,ci90_hi,wins,p,holm,verdict
angled_kink,8,silu,20,0.001,0.003,1.0014,0.593,1.4006,10,0.7562,1.0,unklar
angled_kink,8,gelu,20,0.001,0.003,0.9785,0.43,1.1887,10,0.3488,1.0,unklar
angled_kink,16,silu,20,0.001,0.003,0.6013,0.409,1.0045,13,0.0532,1.0,unklar
angled_kink,16,gelu,20,0.001,0.003,0.9469,0.4384,1.6871,10,0.6742,1.0,unklar
angled_kink,32,silu,20,0.001,0.003,0.5854,0.3247,1.1044,13,0.0637,1.0,unklar
angled_kink,32,gelu,20,0.001,0.003,0.7238,0.3567,0.9455,14,0.0973,1.0,unklar
corner_kink,8,silu,20,0.003,0.003,0.4948,0.374,0.9293,14,0.0064,0.1725,unklar
corner_kink,8,gelu,20,0.003,0.003,0.8345,0.5249,1.5748,11,0.8695,1.0,unklar
corner_kink,16,silu,20,0.003,0.003,0.7267,0.3081,0.902,15,0.024,0.6227,unklar
corner_kink,16,gelu,20,0.003,0.003,0.9869,0.6394,1.691,10,0.7562,1.0,unklar
corner_kink,32,silu,20,0.003,0.003,1.3664,0.9991,1.8365,7,0.0826,1.0,unklar
corner_kink,32,gelu,20,0.003,0.003,2.5454,1.4362,3.3853,3,0.0003,0.0076,PreCap schlechter
radial_ripple,8,silu,20,0.003,0.003,0.532,0.2543,1.23,13,0.04,0.9997,unklar
radial_ripple,8,gelu,20,0.003,0.003,0.5582,0.2529,0.7609,16,0.0023,0.0651,unklar
radial_ripple,16,silu,20,0.003,0.003,1.3953,0.454,2.3114,8,0.8124,1.0,unklar
radial_ripple,16,gelu,20,0.003,0.003,1.3058,0.8868,2.0648,7,0.3488,1.0,unklar
radial_ripple,32,silu,20,0.001,0.003,1.6092,0.7209,2.398,8,0.4749,1.0,unklar
radial_ripple,32,gelu,20,0.001,0.003,1.2476,0.7607,1.8228,8,0.3118,1.0,unklar
ridge_mix,8,silu,20,0.003,0.003,0.5943,0.3202,1.719,13,0.2774,1.0,unklar
ridge_mix,8,gelu,20,0.003,0.003,0.9473,0.5993,1.2659,10,0.6742,1.0,unklar
ridge_mix,16,silu,20,0.003,0.003,0.8923,0.6651,2.3353,11,0.8408,1.0,unklar
ridge_mix,16,gelu,20,0.003,0.003,1.1106,0.7456,2.0531,9,0.2455,1.0,unklar
ridge_mix,32,silu,20,0.001,0.003,0.6538,0.3482,1.2362,12,0.498,1.0,unklar
ridge_mix,32,gelu,20,0.001,0.003,0.6315,0.3659,0.7712,16,0.2024,1.0,unklar
smooth_bowl,8,silu,20,0.003,0.003,0.9714,0.5949,1.4914,10,0.8124,1.0,unklar
smooth_bowl,8,gelu,20,0.003,0.003,0.6987,0.4045,1.4478,11,0.6215,1.0,unklar
smooth_bowl,16,silu,20,0.003,0.003,2.7388,1.7385,4.111,3,0.0001,0.0019,PreCap schlechter
smooth_bowl,16,gelu,20,0.003,0.003,1.6472,1.0304,2.5607,6,0.114,1.0,unklar
smooth_bowl,32,silu,20,0.001,0.003,1.8163,0.9017,3.3087,7,0.3683,1.0,unklar
smooth_bowl,32,gelu,20,0.001,0.003,2.1011,0.9031,2.919,8,0.0826,1.0,unklar


## Verdikt

{
  "n_rows": 32400,
  "runs": 4050,
  "nonfinite_runs": 0,
  "final_step": 3000,
  "lr_selection_counts": [
    {
      "act": "gelu",
      "lr_sel": 0.003,
      "n": 15
    },
    {
      "act": "precap_base",
      "lr_sel": 0.001,
      "n": 6
    },
    {
      "act": "precap_base",
      "lr_sel": 0.003,
      "n": 9
    },
    {
      "act": "silu",
      "lr_sel": 0.003,
      "n": 15
    }
  ],
  "confirmatory_counts": {
    "unklar": 28,
    "PreCap schlechter": 2
  },
  "criterion": {
    "gelu": {
      "besser": 0,
      "schlechter": 1,
      "unklar": 14,
      "H_A_bestaetigt": false,
      "je_tiefe": {
        "8": {
          "unklar": 5
        },
        "16": {
          "unklar": 5
        },
        "32": {
          "unklar": 4,
          "PreCap schlechter": 1
        }
      }
    },
    "silu": {
      "besser": 0,
      "schlechter": 1,
      "unklar": 14,
      "H_A_bestaetigt": false,
      "je_tiefe": {
        "8": {
          "unklar": 5
        },
        "16": {
          "unklar": 4,
          "PreCap schlechter": 1
        },
        "32": {
          "unklar": 5
        }
      }
    }
  },
  "mechanism_spearman_frac_gt_half_vs_log_ratio_silu": {
    "rho": -0.5214285714285714,
    "p_one_sided_rho_lt_0": 0.02310961723928651,
    "n": 15,
    "gestuetzt": true
  }
}

## LR-Empfindlichkeit (Median log10 Test-MSE, alle Seeds)

depth,act,0.0003,0.001,0.003
8,gelu,-2.932,-3.424,-4.006
8,precap_base,-3.157,-3.77,-4.167
8,silu,-2.748,-3.133,-3.878
16,gelu,-2.915,-3.459,-3.937
16,precap_base,-3.111,-3.656,-4.031
16,silu,-2.728,-3.233,-3.831
32,gelu,-2.872,-3.528,-3.879
32,precap_base,-3.117,-3.688,-3.648
32,silu,-2.729,-3.362,-3.741

