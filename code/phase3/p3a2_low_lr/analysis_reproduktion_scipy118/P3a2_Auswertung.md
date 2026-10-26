# Phase 3a-2 - Auswertung (automatisch)

## Bestaetigende Tests

dataset,depth,cand,ref,primary,n,med_err_cand,med_err_ref,med_diff_pp,wins,losses,p,holm_all,holm_primary,verdict_primary,verdict_all
fashion,50,spreccap,selu,True,15,0.1599,0.1476,1.13,3,12,0.00201,0.01813,0.00201,Kandidat schlechter,Kandidat schlechter
fashion,50,spreccap,relu,False,15,0.1599,0.1555,0.21,7,8,0.3028,0.90839,,,unklar
fashion,50,spreccap,elu,False,15,0.1599,0.1412,1.37,0,15,6e-05,0.00244,,,Kandidat schlechter
fashion,50,spreccap,silu_he,False,15,0.1599,0.1726,-1.93,13,2,0.00043,0.01025,,,Kandidat besser
fashion,50,spreccap,gelu_he,False,15,0.1599,0.172,-0.72,10,5,0.10699,0.64197,,,unklar
fashion,50,precap0,selu,False,15,0.1489,0.1476,0.33,5,10,0.14745,0.64197,,,unklar
fashion,50,precap0,relu,False,15,0.1489,0.1555,-0.64,12,3,0.01807,0.12646,,,unklar
fashion,50,precap0,elu,False,15,0.1489,0.1412,0.78,2,13,0.00043,0.01025,,,Kandidat schlechter
fashion,50,precap0,silu_he,False,15,0.1489,0.1726,-2.33,15,0,6e-05,0.00244,,,Kandidat besser
fashion,50,precap0,gelu_he,False,15,0.1489,0.172,-2.3,14,1,0.00116,0.01307,,,Kandidat besser
fashion,100,spreccap,selu,True,15,0.2415,0.1721,7.1,0,15,6e-05,0.00244,0.00024,Kandidat schlechter,Kandidat schlechter
fashion,100,spreccap,relu,False,15,0.2415,0.2166,2.97,1,14,0.00012,0.00305,,,Kandidat schlechter
fashion,100,spreccap,elu,False,15,0.2415,0.1648,7.49,0,15,6e-05,0.00244,,,Kandidat schlechter
fashion,100,spreccap,silu_he,False,15,0.2415,0.9,-65.85,15,0,0.00065,0.01307,,,Kandidat besser
fashion,100,spreccap,gelu_he,False,15,0.2415,0.9,-65.85,15,0,0.00065,0.01307,,,Kandidat besser
fashion,100,precap0,selu,False,15,0.1825,0.1721,0.68,0,15,0.00065,0.01307,,,Kandidat schlechter
fashion,100,precap0,relu,False,15,0.1825,0.2166,-3.01,15,0,6e-05,0.00244,,,Kandidat besser
fashion,100,precap0,elu,False,15,0.1825,0.1648,1.46,0,15,6e-05,0.00244,,,Kandidat schlechter
fashion,100,precap0,silu_he,False,15,0.1825,0.9,-71.75,15,0,0.00065,0.01307,,,Kandidat besser
fashion,100,precap0,gelu_he,False,15,0.1825,0.9,-71.75,15,0,0.00065,0.01307,,,Kandidat besser
mnist,50,spreccap,selu,True,15,0.0512,0.0448,0.64,1,14,0.0008,0.01307,0.00161,Kandidat schlechter,Kandidat schlechter
mnist,50,spreccap,relu,False,15,0.0512,0.0506,0.05,6,9,0.55085,1.0,,,unklar
mnist,50,spreccap,elu,False,15,0.0512,0.0389,1.27,0,15,0.00065,0.01307,,,Kandidat schlechter
mnist,50,spreccap,silu_he,False,15,0.0512,0.0684,-1.64,15,0,0.00065,0.01307,,,Kandidat besser
mnist,50,spreccap,gelu_he,False,15,0.0512,0.0579,-0.77,15,0,6e-05,0.00244,,,Kandidat besser
mnist,50,precap0,selu,False,15,0.049,0.0448,0.51,2,13,0.00201,0.01813,,,Kandidat schlechter
mnist,50,precap0,relu,False,15,0.049,0.0506,-0.08,8,7,0.93408,1.0,,,unklar
mnist,50,precap0,elu,False,15,0.049,0.0389,0.97,1,14,0.00061,0.01282,,,Kandidat schlechter
mnist,50,precap0,silu_he,False,15,0.049,0.0684,-2.24,15,0,6e-05,0.00244,,,Kandidat besser
mnist,50,precap0,gelu_he,False,15,0.049,0.0579,-1.09,12,3,0.00153,0.01526,,,Kandidat besser
mnist,100,spreccap,selu,True,15,0.1387,0.0607,8.03,0,15,6e-05,0.00244,0.00024,Kandidat schlechter,Kandidat schlechter
mnist,100,spreccap,relu,False,15,0.1387,0.1152,0.77,5,10,0.12054,0.64197,,,unklar
mnist,100,spreccap,elu,False,15,0.1387,0.0637,7.77,0,15,6e-05,0.00244,,,Kandidat schlechter
mnist,100,spreccap,silu_he,False,15,0.1387,0.8865,-74.78,15,0,6e-05,0.00244,,,Kandidat besser
mnist,100,spreccap,gelu_he,False,15,0.1387,0.8865,-72.71,14,1,0.00043,0.01025,,,Kandidat besser
mnist,100,precap0,selu,False,15,0.0834,0.0607,2.37,0,15,6e-05,0.00244,,,Kandidat schlechter
mnist,100,precap0,relu,False,15,0.0834,0.1152,-2.76,15,0,6e-05,0.00244,,,Kandidat besser
mnist,100,precap0,elu,False,15,0.0834,0.0637,2.13,0,15,0.00065,0.01307,,,Kandidat schlechter
mnist,100,precap0,silu_he,False,15,0.0834,0.8865,-80.31,15,0,6e-05,0.00244,,,Kandidat besser
mnist,100,precap0,gelu_he,False,15,0.0834,0.8865,-80.16,15,0,6e-05,0.00244,,,Kandidat besser


## Sekundaer

dataset,depth,act,median_test_err_conf,lr_window_0.5pp,divergence_rate_tune,divergence_rate_conf
fashion,50,elu,0.1412,2,0.0,0.0
fashion,50,gelu_he,0.172,2,0.0,0.0
fashion,50,precap0,0.1489,1,0.0,0.0
fashion,50,relu,0.1555,1,0.0,0.0
fashion,50,selu,0.1476,3,0.0,0.0
fashion,50,silu_he,0.1726,2,0.0,0.0
fashion,50,spreccap,0.1599,2,0.0,0.0
fashion,100,elu,0.1648,2,0.0,0.0
fashion,100,gelu_he,0.9,2,0.0,0.0
fashion,100,precap0,0.1825,1,0.0,0.0
fashion,100,relu,0.2166,1,0.0,0.0
fashion,100,selu,0.1721,1,0.0,0.0
fashion,100,silu_he,0.9,1,0.0,0.0
fashion,100,spreccap,0.2415,1,0.0,0.0
mnist,50,elu,0.0389,2,0.0,0.0
mnist,50,gelu_he,0.0579,2,0.0,0.0
mnist,50,precap0,0.049,2,0.0,0.0
mnist,50,relu,0.0506,3,0.0,0.0
mnist,50,selu,0.0448,2,0.0,0.0
mnist,50,silu_he,0.0684,1,0.0,0.0
mnist,50,spreccap,0.0512,1,0.0,0.0
mnist,100,elu,0.0637,2,0.0,0.0
mnist,100,gelu_he,0.8865,6,0.0,0.0
mnist,100,precap0,0.0834,1,0.0,0.0
mnist,100,relu,0.1152,1,0.0,0.0
mnist,100,selu,0.0607,1,0.0,0.0
mnist,100,silu_he,0.8865,6,0.0,0.0
mnist,100,spreccap,0.1387,1,0.0,0.0
