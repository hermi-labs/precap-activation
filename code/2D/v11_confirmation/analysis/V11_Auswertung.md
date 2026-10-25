# V11 Auswertung (preregistered)

task,depth,role,n,median_d,ratio,seeds_better,p_perf_less,median_d_peak,p_peak_less,p_peak_greater,median_d_prep99,p_prep99_less,median_d_cap,median_d_grad,holm_perf,holm_peak,holm_prep99
geo_a025,16,success,20,-0.21,0.6165,17,0.0002,-0.0606,0.1153,0.8919,-0.0615,0.0319,0.0021,0.0329,0.0007,0.2305,0.0956
geo_a025,32,success,20,-0.4278,0.3734,16,0.0002,-0.0597,0.0181,0.9836,-0.1896,0.0001,-0.0016,-0.1364,0.0006,0.0725,0.0003
rot_a025,16,success,20,-0.3181,0.4807,16,0.0012,-0.0659,0.0291,0.9734,0.0157,0.4927,0.0028,-0.0034,0.0023,0.0874,0.9854
rot_a025,32,success,20,-0.0099,0.9776,10,0.249,0.1073,0.9984,0.0018,0.0317,0.9968,0.0025,-0.0088,0.249,0.9984,0.9968
rot_a075,16,failure,20,-0.1073,0.7811,14,0.0077,-0.0023,0.835,0.1744,0.0318,0.9923,0.0077,-0.0414,,,
rot_a075,32,failure,20,-0.0341,0.9244,12,0.1559,-0.018,0.7021,0.3108,0.0592,0.9964,0.007,0.0679,,,


{
  "H1_performance_replicated_in_all_success_cells": false,
  "H2_no_meaningful_benefit_in_failure_cells": false,
  "H3_mechanism_prep99_confirmed_primary": false,
  "centered_spearman": {
    "d_hmax_peak": {
      "rho": 0.01544551704979512,
      "p": 0.8670268697698237
    },
    "d_layer_pre_p99": {
      "rho": -0.014438502673796788,
      "p": 0.8756243289457546
    }
  },
  "secondary_output_peak_success_cells_significant_holm": 0,
  "secondary_output_peak_failure_cells_positive": false,
  "n_success_cells": 4,
  "n_failure_cells": 2
}
