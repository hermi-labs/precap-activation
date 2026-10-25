# V17 Auswertung

step,task,depth,scale,ref,n,ratio,ci90_lo,ci90_hi,wins,p,holm,verdict
3000,corner_kink,64,constant,silu,20,7.4754,4.5761,21.3017,1,0.0,0.0001,PreCap schlechter
3000,corner_kink,64,constant,gelu,20,13.0547,3.4924,18.3501,3,0.0014,0.043,PreCap schlechter
3000,corner_kink,64,inv_quart,silu,20,0.5341,0.2335,0.8182,15,0.0637,1.0,unklar
3000,corner_kink,64,inv_quart,gelu,20,0.5486,0.2853,1.2961,12,0.0973,1.0,unklar
3000,corner_kink,64,inv_sqrt,silu,20,0.4755,0.4146,0.5314,16,0.0002,0.0057,PreCap besser
3000,corner_kink,64,inv_sqrt,gelu,20,1.3523,1.1934,1.6012,3,0.0121,0.2899,unklar
3000,corner_kink,128,constant,silu,20,0.0006,0.0003,0.0017,20,0.0,0.0001,PreCap besser
3000,corner_kink,128,constant,gelu,20,0.0002,0.0001,0.0006,20,0.0,0.0001,PreCap besser
3000,corner_kink,128,inv_quart,silu,20,0.225,0.1682,0.3615,20,0.0,0.0001,PreCap besser
3000,corner_kink,128,inv_quart,gelu,20,0.4562,0.2022,0.6074,15,0.0042,0.1182,unklar
3000,corner_kink,128,inv_sqrt,silu,20,0.6735,0.5868,0.8292,16,0.0002,0.0057,PreCap besser
3000,corner_kink,128,inv_sqrt,gelu,20,1.628,1.2203,1.9306,4,0.0073,0.1824,unklar
3000,radial_ripple,64,constant,silu,20,94.8584,15.3298,157.8415,1,0.0,0.0001,PreCap schlechter
3000,radial_ripple,64,constant,gelu,20,50.438,9.6113,159.2451,3,0.0002,0.0067,PreCap schlechter
3000,radial_ripple,64,inv_quart,silu,20,1.7451,0.6703,3.4333,8,0.1429,1.0,unklar
3000,radial_ripple,64,inv_quart,gelu,20,2.006,1.0014,2.8423,6,0.4524,1.0,unklar
3000,radial_ripple,64,inv_sqrt,silu,20,0.9671,0.4062,2.0082,11,0.9273,1.0,unklar
3000,radial_ripple,64,inv_sqrt,gelu,20,0.8391,0.401,1.3084,11,0.8408,1.0,unklar
3000,radial_ripple,128,constant,silu,20,0.0032,0.0017,0.0062,20,0.0,0.0001,PreCap besser
3000,radial_ripple,128,constant,gelu,20,0.0006,0.0001,0.0011,20,0.0,0.0001,PreCap besser
3000,radial_ripple,128,inv_quart,silu,20,4.7325,1.3684,13.1206,4,0.0064,0.1661,unklar
3000,radial_ripple,128,inv_quart,gelu,20,3.5148,1.5907,8.8202,5,0.0532,1.0,unklar
3000,radial_ripple,128,inv_sqrt,silu,20,1.4232,0.5198,2.9609,8,0.8408,1.0,unklar
3000,radial_ripple,128,inv_sqrt,gelu,20,1.5994,0.8837,2.9161,7,0.1231,1.0,unklar
3000,ridge_mix,64,constant,silu,20,52.2258,26.5967,81.5182,0,0.0,0.0001,PreCap schlechter
3000,ridge_mix,64,constant,gelu,20,15.377,4.5382,72.0479,3,0.0003,0.0081,PreCap schlechter
3000,ridge_mix,64,inv_quart,silu,20,1.0506,0.4885,2.0898,10,0.8695,1.0,unklar
3000,ridge_mix,64,inv_quart,gelu,20,0.7216,0.36,2.6554,11,1.0,1.0,unklar
3000,ridge_mix,64,inv_sqrt,silu,20,0.3861,0.1759,0.7457,16,0.024,0.5269,unklar
3000,ridge_mix,64,inv_sqrt,gelu,20,0.5381,0.3923,1.2171,12,0.2611,1.0,unklar
3000,ridge_mix,128,constant,silu,20,0.0017,0.0004,0.0034,20,0.0,0.0001,PreCap besser
3000,ridge_mix,128,constant,gelu,20,0.001,0.0002,0.0032,20,0.0,0.0001,PreCap besser
3000,ridge_mix,128,inv_quart,silu,20,2.1396,0.8588,5.2309,7,0.1231,1.0,unklar
3000,ridge_mix,128,inv_quart,gelu,20,6.0096,1.9378,12.0683,4,0.0023,0.0674,unklar
3000,ridge_mix,128,inv_sqrt,silu,20,0.6786,0.4661,1.8225,11,0.5459,1.0,unklar
3000,ridge_mix,128,inv_sqrt,gelu,20,1.0772,0.6563,2.707,9,0.6742,1.0,unklar
3000,smooth_bowl,64,constant,silu,20,101.514,42.4312,332.2051,0,0.0,0.0001,PreCap schlechter
3000,smooth_bowl,64,constant,gelu,20,50.8212,5.3329,151.1234,3,0.0001,0.0047,PreCap schlechter
3000,smooth_bowl,64,inv_quart,silu,20,0.4027,0.2841,0.9788,15,0.0153,0.3522,unklar
3000,smooth_bowl,64,inv_quart,gelu,20,0.6024,0.4236,0.9469,14,0.114,1.0,unklar
3000,smooth_bowl,64,inv_sqrt,silu,20,0.8243,0.441,0.9819,14,0.024,0.5269,unklar
3000,smooth_bowl,64,inv_sqrt,gelu,20,0.5603,0.4175,0.7262,15,0.0049,0.1312,unklar
3000,smooth_bowl,128,constant,silu,20,0.0006,0.0003,0.0011,20,0.0,0.0001,PreCap besser
3000,smooth_bowl,128,constant,gelu,20,0.0002,0.0001,0.0005,20,0.0,0.0001,PreCap besser
3000,smooth_bowl,128,inv_quart,silu,20,0.3888,0.2483,0.7558,15,0.0441,0.8811,unklar
3000,smooth_bowl,128,inv_quart,gelu,20,0.4457,0.3509,0.6806,15,0.2611,1.0,unklar
3000,smooth_bowl,128,inv_sqrt,silu,20,1.5035,0.9426,2.7381,7,0.1769,1.0,unklar
3000,smooth_bowl,128,inv_sqrt,gelu,20,1.4431,0.5086,2.5081,9,0.498,1.0,unklar


{
  "n_rows": 11520,
  "complete_ok_runs": 1440,
  "final_counts": {
    "unklar": 29,
    "PreCap besser": 11,
    "PreCap schlechter": 8
  },
  "by_scale": {
    "constant": {
      "PreCap schlechter": 8,
      "PreCap besser": 8
    },
    "inv_quart": {
      "unklar": 15,
      "PreCap besser": 1
    },
    "inv_sqrt": {
      "unklar": 14,
      "PreCap besser": 2
    }
  },
  "post_rms_vs_test_mse": {
    "rho": 0.7069565217391304,
    "p": 0.00011232149069224439,
    "n": 24
  }
}