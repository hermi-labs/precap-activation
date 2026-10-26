# Phase 3a-3 - Auswertung (automatisch)

## Bestaetigende Tests

dataset,depth,cand,ref,primary,n,med_err_cand,med_err_ref,med_diff_pp,wins,losses,p,p_scipy_info,holm_all,holm_primary,verdict_primary,verdict_all
fashion,50,spreccap_cs,selu,True,15,0.1492,0.1469,0.2,4,11,0.08325,0.08325,,0.33301,unklar,unklar
fashion,50,spreccap_cs,spreccap,False,15,0.1492,0.1615,-1.04,15,0,6e-05,6e-05,0.00146,,,Kandidat besser
fashion,50,spreccap_cs_ls,spreccap_cs,False,15,0.1483,0.1492,-0.02,8,7,0.49744,0.47764,1.0,,,unklar
fashion,50,spreccap_cs,elu,False,15,0.1492,0.1467,0.2,4,11,0.05536,0.05536,0.36035,,,unklar
fashion,50,spreccap_e03,selu,False,15,0.1457,0.1469,0.16,7,8,0.89038,0.89038,1.0,,,unklar
fashion,50,spreccap_e03,spreccap_cs,False,15,0.1457,0.1492,-0.28,10,5,0.08325,0.08325,0.41626,,,unklar
fashion,50,spreccap,selu,False,15,0.1615,0.1469,1.21,0,15,6e-05,6e-05,0.00146,,,Kandidat schlechter
fashion,100,spreccap_cs,selu,True,15,0.1698,0.1729,-0.12,9,6,0.59949,0.59949,,1.0,unklar,unklar
fashion,100,spreccap_cs,spreccap,False,15,0.1698,0.2421,-7.23,15,0,6e-05,0.00065,0.00146,,,Kandidat besser
fashion,100,spreccap_cs_ls,spreccap_cs,False,15,0.1702,0.1698,-0.06,9,6,0.33734,0.32016,1.0,,,unklar
fashion,100,spreccap_cs,elu,False,15,0.1698,0.1662,0.36,4,11,0.03534,0.03534,0.35339,,,unklar
fashion,100,spreccap_e03,selu,False,15,0.1661,0.1729,-0.74,12,3,0.01025,0.01025,0.12305,,,unklar
fashion,100,spreccap_e03,spreccap_cs,False,15,0.1661,0.1698,-1.01,10,5,0.04126,0.04126,0.36035,,,unklar
fashion,100,spreccap,selu,False,15,0.2421,0.1729,7.13,0,15,6e-05,6e-05,0.00146,,,Kandidat schlechter
mnist,50,spreccap_cs,selu,True,15,0.0442,0.0431,0.12,7,8,0.76154,0.76154,,1.0,unklar,unklar
mnist,50,spreccap_cs,spreccap,False,15,0.0442,0.0513,-0.64,14,1,0.00018,0.00018,0.00311,,,Kandidat besser
mnist,50,spreccap_cs_ls,spreccap_cs,False,15,0.0401,0.0442,-0.45,13,2,0.00085,0.00215,0.01196,,,Kandidat besser
mnist,50,spreccap_cs,elu,False,15,0.0442,0.0375,0.58,2,12,0.00171,0.00344,0.02222,,,Kandidat schlechter
mnist,50,spreccap_e03,selu,False,15,0.0405,0.0431,-0.37,11,4,0.04004,0.04085,0.36035,,,unklar
mnist,50,spreccap_e03,spreccap_cs,False,15,0.0405,0.0442,-0.44,11,4,0.02155,0.02155,0.237,,,unklar
mnist,50,spreccap,selu,False,15,0.0513,0.0431,1.03,1,14,0.00012,0.00012,0.0022,,,Kandidat schlechter
mnist,100,spreccap_cs,selu,True,15,0.0627,0.0636,0.02,7,8,0.76154,0.76154,,1.0,unklar,unklar
mnist,100,spreccap_cs,spreccap,False,15,0.0627,0.1356,-6.74,15,0,6e-05,6e-05,0.00146,,,Kandidat besser
mnist,100,spreccap_cs_ls,spreccap_cs,False,15,0.0592,0.0627,-0.42,12,3,0.04126,0.04126,0.36035,,,unklar
mnist,100,spreccap_cs,elu,False,15,0.0627,0.0578,0.53,3,12,0.0946,0.0946,0.41626,,,unklar
mnist,100,spreccap_e03,selu,False,15,0.0528,0.0636,-1.33,13,2,0.00043,0.00043,0.00641,,,Kandidat besser
mnist,100,spreccap_e03,spreccap_cs,False,15,0.0528,0.0627,-0.7,14,1,0.00018,0.00089,0.00311,,,Kandidat besser
mnist,100,spreccap,selu,False,15,0.1356,0.0636,7.04,0,15,6e-05,6e-05,0.00146,,,Kandidat schlechter


## Sekundaer

dataset,depth,act,median_test_err_conf,lr_window_0.5pp,divergence_rate_tune,divergence_rate_conf
fashion,50,elu,0.1467,2,0.0,0.0
fashion,50,selu,0.1469,2,0.0,0.0
fashion,50,spreccap,0.1615,1,0.0,0.0
fashion,50,spreccap_cs,0.1492,2,0.0,0.0
fashion,50,spreccap_cs_ls,0.1483,2,0.0,0.0
fashion,50,spreccap_e03,0.1457,2,0.0,0.0
fashion,100,elu,0.1662,1,0.0,0.0
fashion,100,selu,0.1729,1,0.0,0.0
fashion,100,spreccap,0.2421,1,0.0,0.0
fashion,100,spreccap_cs,0.1698,1,0.0,0.0
fashion,100,spreccap_cs_ls,0.1702,1,0.0,0.0
fashion,100,spreccap_e03,0.1661,3,0.0,0.0
mnist,50,elu,0.0375,2,0.0,0.0
mnist,50,selu,0.0431,1,0.0,0.0
mnist,50,spreccap,0.0513,2,0.0,0.0
mnist,50,spreccap_cs,0.0442,2,0.0,0.0
mnist,50,spreccap_cs_ls,0.0401,2,0.0,0.0
mnist,50,spreccap_e03,0.0405,3,0.0,0.0
mnist,100,elu,0.0578,2,0.0,0.0
mnist,100,selu,0.0636,1,0.0,0.0
mnist,100,spreccap,0.1356,1,0.0,0.0
mnist,100,spreccap_cs,0.0627,1,0.0,0.0
mnist,100,spreccap_cs_ls,0.0592,1,0.0,0.0
mnist,100,spreccap_e03,0.0528,2,0.0,0.0
