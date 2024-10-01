import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# sns.set_palette("deep")
# plt.style.use("seaborn")
sns.set_style("whitegrid")

# ***********
# ANT RESULTS
# ***********
ANT_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra/sd000_s_56955647.0.1718292963_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra/sd001_s_56955648.0.1718292963_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_with_goal_metrics/sd000_s_21488199.0.1721961088_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_with_goal_metrics/sd001_s_21488200.0.1721961088_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_with_goal_metrics/sd002_s_21488201.0.1721961088_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd000_s_21497671.0.1722112978_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd001_s_21497672.0.1722112978_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_NO_DONE = [
    ''
]

ANT_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_FP_NO_DONE = [
    ''
]

ANT_METRA_SF_TD = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td/sd000_s_56969167.0.1718294685_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td/sd001_s_56969165.0.1718294682_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_L2_GAUSSIAN_Z_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_with_goal_metrics_l2_penalty_with_fixed_lambda_0.5_gaussian_z_no_done/sd000_s_21412764.0.1721597423_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_with_goal_metrics_l2_penalty_with_fixed_lambda_0.5_gaussian_z_no_done/sd001_s_21412765.0.1721597423_ant_metra/progress_eval.csv'
]

ANT_METRA_L2_GAUSSIAN_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_with_goal_metrics_l2_penalty_with_fixed_lambda_0.5_gaussian_z/sd000_s_21412762.0.1721597423_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_with_goal_metrics_l2_penalty_with_fixed_lambda_0.5_gaussian_z/sd001_s_21412763.0.1721597423_ant_metra/progress_eval.csv'
]

ANT_METRA_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_no_done/sd000_s_21377697.0.1721417591_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_no_done/sd001_s_21377698.0.1721417591_ant_metra/progress_eval.csv',
]

ANT_METRA_INFONCE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_with_goal_metrics/sd000_s_21430564.0.1721679815_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_with_goal_metrics/sd001_s_21430565.0.1721679815_ant_metra/progress_eval.csv',
]

ANT_METRA_INFONCE_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_no_done_with_goal_metrics/sd000_s_21430566.0.1721679912_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_no_done_with_goal_metrics/sd001_s_21430567.0.1721679913_ant_metra/progress_eval.csv',
]

ANT_METRA_INFONCE_1024 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_with_goal_metrics/sd000_s_21472625.0.1721860103_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_with_goal_metrics/sd001_s_21472626.0.1721860103_ant_metra/progress_eval.csv',
]

ANT_METRA_BESSEL_PENALTY = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_bessel_penalty_lambda_1_with_goal_metrics/sd000_s_21436497.0.1721741610_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_bessel_penalty_lambda_1_with_goal_metrics/sd001_s_21436498.0.1721741607_ant_metra/progress_eval.csv',
]

ANT_METRA_BESSEL_PENALTY_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_bessel_penalty_lambda_1_no_done_with_goal_metrics/sd000_s_21436499.0.1721741611_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_bessel_penalty_lambda_1_no_done_with_goal_metrics/sd001_s_21436500.0.1721741611_ant_metra/progress_eval.csv',
]

ANT_METRA_SF_TD_RELABEL_ACTOR_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td_with_goal_metrics_relabel_actor_z/sd000_s_21432003.0.1721696650_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td_with_goal_metrics_relabel_actor_z/sd001_s_21432004.0.1721696650_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SF_TD_RELABEL_CRITIC_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td_with_goal_metrics_relabel_critic_z/sd000_s_21431984.0.1721696514_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td_with_goal_metrics_relabel_critic_z/sd001_s_21431985.0.1721696513_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SF_TD_RELABEL_ACTOR_AND_CRITIC_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td_with_goal_metrics_relabel_actor_and_critic_z/sd000_s_21432012.0.1721696754_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sf_td_with_goal_metrics_relabel_actor_and_critic_z/sd001_s_21432013.0.1721696754_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_INFO_NCE_256_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd000_s_21472591.0.1721859211_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd001_s_21472592.0.1721859212_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_2560_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd000_s_21472593.0.1721859212_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd001_s_21472594.0.1721859212_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_25600_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd000_s_21472595.0.1721859212_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd001_s_21472596.0.1721859211_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_256000_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd000_s_21472597.0.1721859212_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics/sd001_s_21472598.0.1721859211_ant_metra/progress_eval.csv',
]

ANT_METRA_LAYERNORM = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_layernorm_with_goal_metrics/sd000_s_21476215.0.1721914848_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_layernorm_with_goal_metrics/sd001_s_21476216.0.1721914848_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_256000_Z_LAYERNORM = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_layernorm_with_goal_metrics/sd000_s_21476219.0.1721914970_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_layernorm_with_goal_metrics/sd001_s_21476220.0.1721915047_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_256000_Z_TE_LR_2e4 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_te_lr_sweep/sd000_s_21476558.0.1721918779_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_te_lr_sweep/sd001_s_21476559.0.1721918779_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_256000_Z_TE_LR_5e5 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_te_lr_sweep/sd000_s_21476556.0.1721918779_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_te_lr_sweep/sd001_s_21476557.0.1721918779_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_25600_Z_OPTION_DIM_4 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_option_dim_sweep/sd000_s_21476716.0.1721919206_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_option_dim_sweep/sd001_s_21476717.0.1721919205_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_25600_Z_OPTION_DIM_8 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_option_dim_sweep/sd000_s_21476718.0.1721919205_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_option_dim_sweep/sd001_s_21476719.0.1721919205_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_25600_Z_OPTION_DIM_16 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_option_dim_sweep/sd000_s_21476720.0.1721919205_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_option_dim_sweep/sd001_s_21476721.0.1721919206_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_25600_Z_LAYERS_3 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_num_layer_sweep/sd000_s_21477076.0.1721920373_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_num_layer_sweep/sd001_s_21477077.0.1721920373_ant_metra/progress_eval.csv',
]

ANT_METRA_INFO_NCE_25600_Z_LAYERS_4 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_num_layer_sweep/sd000_s_21477078.0.1721920373_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_infonce_sample_new_z_with_goal_metrics_num_layer_sweep/sd001_s_21477079.0.1721920373_ant_metra/progress_eval.csv',
]

ANT_METRA_GAUSSIAN_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_gaussian_z_with_goal_metrics/sd000_s_21476234.0.1721916074_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_gaussian_z_with_goal_metrics/sd001_s_21476235.0.1721916074_ant_metra/progress_eval.csv',
]

ANT_anonymous_INFONCE = [
    'anonymous-il-scale/metra-with-avalon/exp/anonymous_infonce/progress_eval_0.csv',
    'anonymous-il-scale/metra-with-avalon/exp/anonymous_infonce/progress_eval_1.csv'
]

ANT_METRA_anonymous = [
    'anonymous-il-scale/metra-with-avalon/exp/metra_anonymous/progress_eval_0.csv',
    'anonymous-il-scale/metra-with-avalon/exp/metra_anonymous/progress_eval_1.csv'
]

ANT_METRA_SUM_INFONCE_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_turn_off_dones_with_goal_metrics/sd000_s_21511519.0.1722272095_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_turn_off_dones_with_goal_metrics/sd001_s_21511520.0.1722272095_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_FP_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_fp_turn_off_dones_with_goal_metrics/sd000_s_21511512.0.1722271935_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_fp_turn_off_dones_with_goal_metrics/sd001_s_21511513.0.1722271935_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_RELABEL_ACTOR_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_actor_relabeling_turn_off_dones_with_goal_metrics/sd000_s_21511523.0.1722272224_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_actor_relabeling_turn_off_dones_with_goal_metrics/sd001_s_21511524.0.1722272227_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_no_done_with_goal_metrics/sd000_s_21511507.0.1722271692_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_no_done_with_goal_metrics/sd001_s_21511508.0.1722271692_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_NO_ENT_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_no_ent_no_done_with_goal_metrics/sd000_s_21511505.0.1722271613_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_no_ent_no_done_with_goal_metrics/sd001_s_21511506.0.1722271613_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548291.0.1722480041_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548292.0.1722480041_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548293.0.1722480041_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548294.0.1722480040_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_DIM_16_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548295.0.1722480042_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548296.0.1722480043_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548297.0.1722480042_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548298.0.1722480042_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_DIM_64_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548299.0.1722480042_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_dim_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548300.0.1722480042_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_LAM_0p05_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548256.0.1722479862_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548257.0.1722479861_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_LAM_0p1_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548258.0.1722479861_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548259.0.1722479860_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_LAM_0p5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548260.0.1722479860_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548261.0.1722479860_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_LAM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548260.0.1722479860_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548263.0.1722479861_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_LAM_5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548264.0.1722479861_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548265.0.1722479861_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_LAM_10_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548266.0.1722479859_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548267.0.1722479860_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_INFONCE_LAM_20_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21548268.0.1722479860_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_infonce_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21548269.0.1722479860_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_0p1_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555511.0.1722567081_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555512.0.1722567082_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_0p5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555513.0.1722567081_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555514.0.1722567081_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555515.0.1722567082_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555516.0.1722567082_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555517.0.1722567082_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555518.0.1722567081_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_10_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555519.0.1722567081_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555520.0.1722567084_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_20_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555521.0.1722567085_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555522.0.1722567084_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_50_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555523.0.1722567084_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555524.0.1722567084_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_100_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd000_s_21555525.0.1722567084_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_lam_sweep_turn_off_dones_with_goal_metrics/sd001_s_21555526.0.1722567084_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_0p5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567105.0.1722889982_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567106.0.1722950262_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_1_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd000_s_21565968.0.1722869139_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd001_s_21565969.0.1722869139_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567109.0.1722950262_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567110.0.1722950266_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567111.0.1722950266_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567112.0.1722950266_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_10_NO_DONE = [
   'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567113.0.1722950266_ant_metra_sf/progress_eval.csv',
   'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567114.0.1722950266_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_20_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567115.0.1722950564_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567116.0.1722950564_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_50_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567117.0.1722950564_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567118.0.1722950564_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd000_s_21566003.0.1722869808_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd001_s_21566004.0.1722869808_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd000_s_21566005.0.1722869807_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd001_s_21566006.0.1722869808_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd000_s_21566007.0.1722869807_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd001_s_21567098.0.1722889982_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_16_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd000_s_21567099.0.1722889982_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd001_s_21567100.0.1722889982_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd000_s_21567101.0.1722889983_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd001_s_21567102.0.1722889982_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_64_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd000_s_21567103.0.1722889983_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_option_dim_sweep/sd001_s_21567104.0.1722889983_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p02_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595313.0.1723048777_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595314.0.1723048778_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p05_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595315.0.1723048779_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595316.0.1723048777_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p1_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595317.0.1723048775_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595318.0.1723048776_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595319.0.1723048776_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595320.0.1723048777_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595321.0.1723048776_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595322.0.1723048776_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_1_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595323.0.1723048776_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595324.0.1723048775_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595325.0.1723048775_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595326.0.1723048779_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595327.0.1723048778_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595328.0.1723048779_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_10_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595329.0.1723048777_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595330.0.1723048779_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_20_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595331.0.1723048778_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595332.0.1723048778_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_TEMP_50_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd000_s_21595333.0.1723048777_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_rep_and_actor_temp_sweep/sd001_s_21595334.0.1723048778_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567111.0.1722950266_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567112.0.1722950266_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645863.0.1723232688_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645864.0.1723232688_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645865.0.1723232688_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645866.0.1723232687_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645867.0.1723232687_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645868.0.1723232688_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645869.0.1723232689_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645870.0.1723232687_ant_metra_sf/progress_eval.csv',
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645871.0.1723232688_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645872.0.1723232687_ant_metra_sf/progress_eval.csv',
]

ANT_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd000_s_21715859.0.1723560411_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd001_s_21715860.0.1723560411_ant_metra/progress_eval.csv',
]

# ***************
# CHEETAH RESULTS
# ***************

CHEETAH_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra/sd000_s_21288510.0.1721008040_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra/sd001_s_21288511.0.1721008040_half_cheetah_metra/progress_eval.csv',
]

CHEETAH_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd000_s_21625261.0.1723157129_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd001_s_21625262.0.1723157129_half_cheetah_metra/progress_eval.csv',
]

CHEETAH_METRA_SF_TD = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sf_td/sd000_s_21300740.0.1721056164_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sf_td/sd001_s_21300741.0.1721056285_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd000_s_21565970.0.1722869156_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd001_s_21565971.0.1722869156_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_0p5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625249.0.1723156994_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625250.0.1723156995_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_1_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd000_s_21565970.0.1722869156_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd001_s_21565971.0.1722869156_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625251.0.1723156994_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625252.0.1723156994_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625253.0.1723156995_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625254.0.1723156993_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_10_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625255.0.1723156994_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625256.0.1723156995_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_20_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625257.0.1723156994_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625258.0.1723156993_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_50_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625259.0.1723156995_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625260.0.1723156993_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625253.0.1723156995_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625254.0.1723156993_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645873.0.1723232865_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645874.0.1723232865_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645875.0.1723232865_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645876.0.1723232866_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645877.0.1723232864_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645878.0.1723232865_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645879.0.1723232864_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645880.0.1723232865_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645881.0.1723232865_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645882.0.1723232864_half_cheetah_metra_sf/progress_eval.csv',
]

CHEETAH_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd000_s_21715902.0.1723560640_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd001_s_21715903.0.1723560640_half_cheetah_metra/progress_eval.csv',
]

# *****************
# QUADRUPED RESULTS
# *****************

QUADRUPED_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra/sd000_s_56953767.0.1718228487_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra/sd001_s_56953768.0.1718228486_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_SF_TD = [
    # 'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sf_td/sd000_s_21303238.0.1721081537_dmc_quadruped_metra_sf/progress_eval.csv',
    # 'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sf_td/sd001_s_21303239.0.1721081538_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sf_td_with_goal_metrics/sd000_s_21311343.0.1721099400_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sf_td_with_goal_metrics/sd001_s_21311344.0.1721099400_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_L2_GAUSSIAN_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_with_goal_metrics_metra_l2_penalty_fixed_lambda_0.5_gaussian_z/sd000_s_21404276.0.1721575803_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_with_goal_metrics_metra_l2_penalty_fixed_lambda_0.5_gaussian_z/sd001_s_21404277.0.1721575803_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_L2_GAUSSIAN_Z_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_with_goal_metrics_metra_l2_penalty_fixed_lambda_0.5_gaussian_z_no_done/sd000_s_21404278.0.1721575976_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_with_goal_metrics_metra_l2_penalty_fixed_lambda_0.5_gaussian_z_no_done/sd001_s_21404279.0.1721575976_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_with_goal_metrics_no_done/sd000_s_21411422.0.1721593872_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_with_goal_metrics_no_done/sd001_s_21411423.0.1721593872_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_INFONCE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_infonce_with_goal_metrics/sd000_s_21430569.0.1721680031_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_infonce_with_goal_metrics/sd001_s_21430570.0.1721680031_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_INFONCE_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_infonce_no_done_with_goal_metrics/sd000_s_21430571.0.1721680083_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_infonce_no_done_with_goal_metrics/sd001_s_21430572.0.1721680083_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_BESSEL_PENALTY = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_bessel_penalty_lambda_1_with_goal_metrics/sd000_s_21436495.0.1721741567_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_bessel_penalty_lambda_1_with_goal_metrics/sd001_s_21436496.0.1721741566_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_BESSEL_PENALTY_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_bessel_penalty_lambda_1_no_done_with_goal_metrics/sd000_s_21436493.0.1721741564_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_bessel_penalty_lambda_1_no_done_with_goal_metrics/sd001_s_21436494.0.1721741564_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd000_s_21497823.0.1722113111_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd001_s_21497824.0.1722113111_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_infonce_relabel_actor_no_done_with_goal_metrics/sd000_s_21502692.0.1722173653_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_infonce_relabel_actor_no_done_with_goal_metrics/sd001_s_21502693.0.1722173653_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_FP_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_infonce_relabel_actor_fp_no_done_with_goal_metrics/sd000_s_21502694.0.1722173655_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_infonce_relabel_actor_fp_no_done_with_goal_metrics/sd001_s_21502695.0.1722173655_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_INFONCE_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_infonce_with_goal_metrics_no_done/sd000_s_21511550.0.1722273005_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_infonce_with_goal_metrics_no_done/sd001_s_21511551.0.1722273005_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_SUM_FP_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_fp_with_goal_metrics_no_done/sd000_s_21511542.0.1722272771_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_fp_with_goal_metrics_no_done/sd001_s_21511543.0.1722272771_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_SUM_RELABEL_ACTOR_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_actor_relabeling_with_goal_metrics_no_done/sd000_s_21511558.0.1722273266_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_actor_relabeling_with_goal_metrics_no_done/sd001_s_21511559.0.1722273266_dmc_quadruped_metra/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_with_goal_metrics_no_done/sd000_s_21511534.0.1722272542_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_with_goal_metrics_no_done/sd001_s_21511535.0.1722272542_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_NO_ENT_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_no_ent_with_goal_metrics_no_done/sd000_s_21511538.0.1722272644_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_no_ent_with_goal_metrics_no_done/sd001_s_21511539.0.1722272644_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done/sd000_s_21565972.0.1722869183_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done/sd001_s_21565973.0.1722869183_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p2 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595382.0.1723052307_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p5 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595380.0.1723052189_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595381.0.1723052248_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_1 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done/sd000_s_21565972.0.1722869183_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done/sd001_s_21565973.0.1722869183_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_2 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595370.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595371.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_5 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595372.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595373.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_10 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595374.0.1723049187_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595375.0.1723049187_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_20 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595376.0.1723049187_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595377.0.1723049306_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_50 = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595378.0.1723052189_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595379.0.1723052189_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd000_s_21646107.0.1723244437_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd001_s_21646108.0.1723244437_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595372.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595373.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd000_s_21646109.0.1723249430_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd001_s_21646110.0.1723249430_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd000_s_21646111.0.1723249430_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd001_s_21646112.0.1723262336_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd000_s_21646113.0.1723275182_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd001_s_21646114.0.1723301012_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd000_s_21646115.0.1723320790_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_sweep/sd001_s_21646116.0.1723330447_dmc_quadruped_metra_sf/progress_eval.csv',
]

QUADRUPED_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd000_s_21736571.0.1723669025_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd001_s_21736572.0.1723669025_dmc_quadruped_metra/progress_eval.csv',
]

# ****************
# HUMANOID RESULTS
# ****************

HUMANOID_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_with_goal_metrics/sd000_s_21319897.0.1721185680_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_with_goal_metrics/sd001_s_21319898.0.1721185680_dmc_humanoid_metra/progress_eval.csv',
]

HUMANOID_METRA_SF_TD = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sf_td_with_goal_metrics/sd000_s_21319899.0.1721185681_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sf_td_with_goal_metrics/sd001_s_21319900.0.1721185680_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd000_s_21500598.0.1722126378_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd001_s_21500599.0.1722126378_dmc_humanoid_metra/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_infonce_relabel_actor_no_done_with_goal_metrics/sd000_s_21502698.0.1722173855_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_infonce_relabel_actor_no_done_with_goal_metrics/sd001_s_21502699.0.1722173855_dmc_humanoid_metra_sf/progress_eval.csv',    
]

HUMANOID_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_FP_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_infonce_relabel_actor_fp_no_done_with_goal_metrics/sd000_s_21502696.0.1722173662_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_infonce_relabel_actor_fp_no_done_with_goal_metrics/sd001_s_21502697.0.1722173662_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_INFONCE_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_infonce_no_done_with_goal_metrics/sd000_s_21511655.0.1722277956_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_infonce_no_done_with_goal_metrics/sd001_s_21511656.0.1722277956_dmc_humanoid_metra/progress_eval.csv',
]

HUMANOID_METRA_SUM_FP_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_fp_no_done_with_goal_metrics/sd000_s_21511651.0.1722277017_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_fp_no_done_with_goal_metrics/sd001_s_21511652.0.1722277017_dmc_humanoid_metra/progress_eval.csv',
]

HUMANOID_METRA_SUM_RELABEL_ACTOR_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_relabel_actor_no_done_with_goal_metrics/sd000_s_21511653.0.1722277106_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_relabel_actor_no_done_with_goal_metrics/sd001_s_21511654.0.1722277105_dmc_humanoid_metra/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_no_done_with_goal_metrics/sd000_s_21511637.0.1722276809_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_no_done_with_goal_metrics/sd001_s_21511638.0.1722276809_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_NO_ENT_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_no_ent_no_done_with_goal_metrics/sd000_s_21511647.0.1722276874_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_no_ent_no_done_with_goal_metrics/sd001_s_21511648.0.1722276874_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd000_s_21565974.0.1722869204_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd001_s_21565975.0.1722869204_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p5 = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625283.0.1723157421_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625284.0.1723157421_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_1 = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd000_s_21565974.0.1722869204_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd001_s_21565975.0.1722869204_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_2 = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625285.0.1723157421_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625286.0.1723157422_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_5 = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625287.0.1723157422_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625288.0.1723157421_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_10 = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625289.0.1723157421_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625290.0.1723157422_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_20 = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625291.0.1723157578_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625292.0.1723157578_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_50 = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625293.0.1723157578_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625294.0.1723157578_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646157.0.1723330684_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646158.0.1723330683_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625287.0.1723157422_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625288.0.1723157421_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646159.0.1723330684_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646160.0.1723330684_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646161.0.1723330683_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646162.0.1723330684_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646163.0.1723330683_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646164.0.1723330870_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646165.0.1723330870_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646166.0.1723330871_dmc_humanoid_metra_sf/progress_eval.csv',
]

HUMANOID_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd000_s_21736569.0.1723669009_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd001_s_21736570.0.1723669009_dmc_humanoid_metra/progress_eval.csv',
]

# ***************
# KITCHEN RESULTS
# ***************

KITCHEN_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra/sd000_s_21301874.0.1721058984_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra/sd001_s_21301875.0.1721058984_kitchen_metra/progress_eval.csv',
]

KITCHEN_METRA_SF_TD = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sf_td/sd000_s_21301961.0.1721059466_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sf_td/sd001_s_21301962.0.1721059465_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd000_s_21645821.0.1723234382_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd001_s_21645822.0.1723243983_kitchen_metra/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd000_s_21565976.0.1722950262_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd001_s_21565977.0.1722950262_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p5 = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625341.0.1723157957_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625342.0.1723157957_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_1 = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd000_s_21565976.0.1722950262_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics/sd001_s_21565977.0.1722950262_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_2 = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625343.0.1723157956_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625344.0.1723157957_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_5 = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625345.0.1723157957_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625346.0.1723157956_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_10 = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625347.0.1723162982_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625348.0.1723162982_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_20 = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625349.0.1723163016_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625350.0.1723175888_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_50 = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625351.0.1723188682_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625352.0.1723214590_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646097.0.1723243984_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646098.0.1723243983_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646099.0.1723243984_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646100.0.1723243983_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625345.0.1723157957_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625346.0.1723157956_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646101.0.1723243984_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646102.0.1723243983_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646103.0.1723244005_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646104.0.1723244437_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646105.0.1723244437_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646106.0.1723244437_kitchen_metra_sf/progress_eval.csv',
]

KITCHEN_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd000_s_21716367.0.1723562043_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd001_s_21716368.0.1723562043_kitchen_metra/progress_eval.csv',
]

YLABELS = [
    'EvalOp/MjNumUniqueCoords'
]

def compute_mean_and_std(filepaths, y_label):
    all_values = []
    min_length = float('inf')

    for csv in filepaths:
        df = pd.read_csv(csv)
        all_values.append(df[y_label].values)
        if len(df[y_label]) < min_length:
            min_length = len(df[y_label])
    
    # Truncate each array to the minimum length found
    truncated_values = [values[:min_length] for values in all_values]
    truncated_values = np.array(truncated_values)
    
    mean_values = np.mean(truncated_values, axis=0)
    std_dev = np.std(truncated_values, axis=0)
    
    total_env_steps = df['TotalEnvSteps'].values[:min_length]
    return mean_values, std_dev, total_env_steps

def plot_with_confidence_bands(total_env_steps, mean_values, std_dev, label):
    plt.plot(total_env_steps, mean_values, label=label) #marker='o', markersize=3, markeredgewidth=0.5, markeredgecolor="#F7F7FF", linewidth=1)
    plt.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2)

# Define the experiment settings
ant_experiments = {
    # 'Metra': ANT_METRA,
    # 'Metra anonymous InfoNCE': ANT_anonymous_INFONCE,
    # 'Metra anonymous': ANT_METRA_anonymous,
    # 'Metra (no done)': ANT_METRA_NO_DONE,
    # 'Metra sum': ANT_METRA_SUM,
    'Metra sum (no done)': ANT_METRA_SUM_NO_DONE,
    # 'Metra sum SF TD INFONCE RELABEL ACTOR (no done)': ANT_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_NO_DONE,
    # 'Metra sum SF TD INFONCE RELABEL ACTOR FP (no done)':
    # ANT_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_FP_NO_DONE,
    # 'Metra Gaussian Z': ANT_METRA_GAUSSIAN_Z,
    # 'Metra +layernorm': ANT_METRA_LAYERNORM,
    # 'Metra InfoNCE (256000) +layernorm': ANT_METRA_INFO_NCE_256000_Z_LAYERNORM,
    # 'Metra InfoNCE (256000) TE LR 2e-4': ANT_METRA_INFO_NCE_256000_Z_TE_LR_2e4,
    # 'Metra InfoNCE (256000) TE LR 5e-5': ANT_METRA_INFO_NCE_256000_Z_TE_LR_5e5,
    # 'Metra InfoNCE (1024)': ANT_METRA_INFONCE_1024,
    # 'Metra InfoNCE Z (256)': ANT_METRA_INFO_NCE_256_Z,
    # 'Metra Energy LAM 0.1 (no done)': ANT_METRA_SUM_ENERGY_0p1_NO_DONE,
    # 'Metra Energy LAM 0.5 (no done)': ANT_METRA_SUM_ENERGY_0p5_NO_DONE,
    # 'Metra Energy LAM 2 (no done)': ANT_METRA_SUM_ENERGY_2_NO_DONE,
    # 'Metra Energy LAM 5 (no done)': ANT_METRA_SUM_ENERGY_5_NO_DONE,
    # 'Metra Energy LAM 10 (no done)': ANT_METRA_SUM_ENERGY_10_NO_DONE,
    # 'Metra Energy LAM 20 (no done)': ANT_METRA_SUM_ENERGY_20_NO_DONE,
    # 'Metra Energy LAM 50 (no done)': ANT_METRA_SUM_ENERGY_50_NO_DONE,
    # 'Metra Energy LAM 100 (no done)': ANT_METRA_SUM_ENERGY_100_NO_DONE,
    # 'Metra InfoNCE Z (2560)': ANT_METRA_INFO_NCE_2560_Z,
    # 'Metra InfoNCE Z (25600)': ANT_METRA_INFO_NCE_25600_Z,
    # 'Metra InfoNCE Z (256000)': ANT_METRA_INFO_NCE_256000_Z,
    # 'Metra InfoNCE Z (25600) DIM 4': ANT_METRA_INFO_NCE_25600_Z_OPTION_DIM_4,
    # 'Metra InfoNCE Z (25600) DIM 8': ANT_METRA_INFO_NCE_25600_Z_OPTION_DIM_8,
    # 'Metra InfoNCE Z (25600) DIM 16': ANT_METRA_INFO_NCE_25600_Z_OPTION_DIM_16,
    # 'Metra InfoNCE Z (25600) LAYERS 3': ANT_METRA_INFO_NCE_25600_Z_LAYERS_3,
    # 'Metra InfoNCE Z (25600) LAYERS 4': ANT_METRA_INFO_NCE_25600_Z_LAYERS_4,
    # 'Metra SF TD': ANT_METRA_SF_TD,
    # 'Metra L2 Gaussian Z': ANT_METRA_L2_GAUSSIAN_Z,
    # 'Metra L2 Gaussian Z (no done)': ANT_METRA_L2_GAUSSIAN_Z_NO_DONE,
    # 'Metra InfoNCE': ANT_METRA_INFONCE,
    # 'Metra InfoNCE (no done)': ANT_METRA_INFONCE_NO_DONE,
    # 'Metra sum InfoNCE LAM 1.0 (no done)': ANT_METRA_SUM_INFONCE_NO_DONE,
    # 'Metra sum InfoNCE DIM 4 (no done)': ANT_METRA_SUM_INFONCE_DIM_4_NO_DONE,
    # 'Metra sum InfoNCE DIM 8 (no done)': ANT_METRA_SUM_INFONCE_DIM_8_NO_DONE,
    # 'Metra sum InfoNCE DIM 16 (no done)': ANT_METRA_SUM_INFONCE_DIM_16_NO_DONE,
    # 'Metra sum InfoNCE DIM 32 (no done)': ANT_METRA_SUM_INFONCE_DIM_32_NO_DONE,
    # 'Metra sum InfoNCE DIM 64 (no done)': ANT_METRA_SUM_INFONCE_DIM_64_NO_DONE,
    # 'Metra sum InfoNCE LAM 0.05 (no done)': ANT_METRA_SUM_INFONCE_LAM_0p05_NO_DONE,
    # 'Metra sum InfoNCE LAM 0.1 (no done)': ANT_METRA_SUM_INFONCE_LAM_0p1_NO_DONE,
    # 'Metra sum InfoNCE LAM 0.5 (no done)': ANT_METRA_SUM_INFONCE_LAM_0p5_NO_DONE,
    # 'Metra sum InfoNCE LAM 2 (no done)': ANT_METRA_SUM_INFONCE_LAM_2_NO_DONE,
    # 'Metra sum InfoNCE LAM 5 (no done)': ANT_METRA_SUM_INFONCE_LAM_5_NO_DONE,
    # 'Metra sum InfoNCE LAM 10 (no done)': ANT_METRA_SUM_INFONCE_LAM_10_NO_DONE,
    # 'Metra sum InfoNCE LAM 20 (no done)': ANT_METRA_SUM_INFONCE_LAM_20_NO_DONE,
    # 'Metra Bessel Penalty': ANT_METRA_BESSEL_PENALTY,
    # 'Metra Bessel Penalty (no done)': ANT_METRA_BESSEL_PENALTY_NO_DONE,
    # 'Metra SF TD Relabel Actor Z': ANT_METRA_SF_TD_RELABEL_ACTOR_Z,
    # 'Metra SF TD Relabel Critic Z': ANT_METRA_SF_TD_RELABEL_CRITIC_Z,
    # 'Metra SF TD Relabel Actor and Critic Z': ANT_METRA_SF_TD_RELABEL_ACTOR_AND_CRITIC_Z,
    # 'Metra sum FP (no done)': ANT_METRA_SUM_FP_NO_DONE,
    # 'Metra sum Relabel Actor (no done)': ANT_METRA_SUM_RELABEL_ACTOR_NO_DONE,
    # 'Metra sum SF TD (no done)': ANT_METRA_SUM_SF_TD_NO_DONE,
    # 'Metra sum SF TD (no ent no done)': ANT_METRA_SUM_SF_TD_NO_ENT_NO_DONE,
    # 'Metra sum SF TD Energy LAM 0.5 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_0p5_NO_DONE,
    # 'Metra sum SF TD Energy LAM 1 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_1_NO_DONE,
    # 'Metra sum SF TD Energy LAM 2 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_2_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_NO_DONE,
    # 'Metra sum SF TD Energy LAM 10 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_10_NO_DONE,
    # 'Metra sum SF TD Energy LAM 20 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_20_NO_DONE,
    # 'Metra sum SF TD Energy LAM 50 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_50_NO_DONE,
    # 'Metra sum SF TD Energy OPTION DIM 2 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_2_NO_DONE,
    # 'Metra sum SF TD Energy OPTION DIM 4 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_4_NO_DONE,
    # 'Metra sum SF TD Energy OPTION DIM 8 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_8_NO_DONE,
    # 'Metra sum SF TD Energy OPTION DIM 16 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_16_NO_DONE,
    # 'Metra sum SF TD Energy OPTION DIM 32 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_32_NO_DONE,
    # 'Metra sum SF TD Energy OPTION DIM 64 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_OPTION_DIM_64_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 0.02 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p02_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 0.05 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p05_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 0.1 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p1_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 0.2 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p2_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 0.5 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_0p5_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 1 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_1_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 2 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_2_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 5 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_5_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 10 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_10_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 20 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_20_NO_DONE,
    # 'Metra sum SF TD Energy TEMP 50 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_TEMP_50_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 2 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 4 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 8 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 16 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 32 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 64 (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE,
    'Metra sum SF TD Energy BEST LAM OPTION DIM (no done)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    'DIAYN (no done)': ANT_DIAYN_NO_DONE,
}

cheetah_experiments = {
    # 'Metra': CHEETAH_METRA,
    'Metra sum (no done)': CHEETAH_METRA_SUM_NO_DONE,
    # 'Metra SF TD': CHEETAH_METRA_SF_TD,
    # 'Metra sum SF TD Energy LAM 0.5 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_0p5_NO_DONE,
    # 'Metra sum SF TD Energy LAM 1 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_1_NO_DONE,
    # 'Metra sum SF TD Energy LAM 2 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_2_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_NO_DONE,
    # 'Metra sum SF TD Energy LAM 10 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_10_NO_DONE,
    # 'Metra sum SF TD Energy LAM 20 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_20_NO_DONE,
    # 'Metra sum SF TD Energy LAM 50 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_50_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 2 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 4 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 8 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 16 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 32 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 64 (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE,
    'Metra sum SF TD Energy BEST LAM OPTION DIM (no done)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    'DIAYN (no done)': CHEETAH_DIAYN_NO_DONE,
}

quadruped_experiments = {
    # 'Metra': QUADRUPED_METRA,
    # 'Metra (no done)': QUADRUPED_METRA_NO_DONE,
    # 'Metra SF TD': QUADRUPED_METRA_SF_TD,
    # 'Metra L2 Gaussian Z': QUADRUPED_METRA_L2_GAUSSIAN_Z,
    # 'Metra L2 Gaussian Z (no done)': QUADRUPED_METRA_L2_GAUSSIAN_Z_NO_DONE,
    # 'Metra InfoNCE': QUADRUPED_METRA_INFONCE,
    # 'Metra InfoNCE (no done)': QUADRUPED_METRA_INFONCE_NO_DONE,
    # 'Metra Bessel Penalty': QUADRUPED_METRA_BESSEL_PENALTY,
    # 'Metra Bessel Penalty (no done)': QUADRUPED_METRA_BESSEL_PENALTY_NO_DONE,
    'Metra sum (no done)': QUADRUPED_METRA_SUM_NO_DONE,
    # 'Metra sum SF TD INFONCE RELABEL ACTOR (no done)': QUADRUPED_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_NO_DONE,
    # 'Metra sum SF TD INFONCE RELABEL ACTOR FP (no done)':
    # QUADRUPED_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_FP_NO_DONE,
    # 'Metra sum InfoNCE (no done)': QUADRUPED_METRA_SUM_INFONCE_NO_DONE,
    # 'Metra sum FP (no done)': QUADRUPED_METRA_SUM_FP_NO_DONE,
    # 'Metra sum Relabel Actor (no done)': QUADRUPED_METRA_SUM_RELABEL_ACTOR_NO_DONE,
    # 'Metra sum SF TD (no done)': QUADRUPED_METRA_SUM_SF_TD_NO_DONE,
    # 'Metra sum SF TD (no ent no done)': QUADRUPED_METRA_SUM_SF_TD_NO_ENT_NO_DONE,
    # 'Metra sum SF TD Energy LAM 0.2 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p2,
    # 'Metra sum SF TD Energy LAM 0.5 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p5,
    # 'Metra sum SF TD Energy LAM 1 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE,
    # 'Metra sum SF TD Energy LAM 2 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_2,
    # 'Metra sum SF TD Energy LAM 5 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_5,
    # 'Metra sum SF TD Energy LAM 10 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_10,
    # 'Metra sum SF TD Energy LAM 20 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_20,
    # 'Metra sum SF TD Energy LAM 50 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_50,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 2 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 4 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 8 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 16 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 32 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 64 (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE,
    'Metra sum SF TD Energy BEST LAM OPTION DIM (no done)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    'DIAYN (no done)': QUADRUPED_DIAYN_NO_DONE,
}

humanoid_experiments = {
    # 'Metra': HUMANOID_METRA,
    # 'Metra SF TD': HUMANOID_METRA_SF_TD,
    'Metra sum (no done)': HUMANOID_METRA_SUM_NO_DONE,
    # 'Metra sum SF TD INFONCE RELABEL ACTOR (no done)': HUMANOID_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_NO_DONE,
    # 'Metra sum SF TD INFONCE RELABEL ACTOR FP (no done)':
    # HUMANOID_METRA_SUM_SF_TD_INFONCE_RELABEL_ACTOR_FP_NO_DONE,
    # 'Metra sum InfoNCE (no done)': HUMANOID_METRA_SUM_INFONCE_NO_DONE,
    # 'Metra sum FP (no done)': HUMANOID_METRA_SUM_FP_NO_DONE,
    # 'Metra sum Relabel Actor (no done)': HUMANOID_METRA_SUM_RELABEL_ACTOR_NO_DONE,
    # 'Metra sum SF TD (no done)': HUMANOID_METRA_SUM_SF_TD_NO_DONE,
    # 'Metra sum SF TD (no ent no done)': HUMANOID_METRA_SUM_SF_TD_NO_ENT_NO_DONE,
    # 'Metra sum SF TD Energy LAM 0.5 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p5,
    # 'Metra sum SF TD Energy LAM 1 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_1,
    # 'Metra sum SF TD Energy LAM 2 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_2,
    # 'Metra sum SF TD Energy LAM 5 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_5,
    # 'Metra sum SF TD Energy LAM 10 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_10,
    # 'Metra sum SF TD Energy LAM 20 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_20,
    # 'Metra sum SF TD Energy LAM 50 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_50,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 2 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 4 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 8 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 16 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 32 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 64 (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE,
    'Metra sum SF TD Energy BEST LAM OPTION DIM (no done)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    'DIAYN (no done)': HUMANOID_DIAYN_NO_DONE,
}

kitchen_experiments = {
    # 'Metra': KITCHEN_METRA,
    # 'Metra SF TD': KITCHEN_METRA_SF_TD,
    'Metra sum (no done)': KITCHEN_METRA_SUM_NO_DONE,
    # 'Metra sum SF TD Energy LAM 0.5 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_0p5,
    # 'Metra sum SF TD Energy LAM 1 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_1,
    # 'Metra sum SF TD Energy LAM 2 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_2,
    # 'Metra sum SF TD Energy LAM 5 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_5,
    # 'Metra sum SF TD Energy LAM 10 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_10,
    # 'Metra sum SF TD Energy LAM 20 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_20,
    # 'Metra sum SF TD Energy LAM 50 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_NO_DONE_LAM_50,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 2 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 4 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 8 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 16 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_16_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 32 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE,
    # 'Metra sum SF TD Energy LAM 5 OPTION DIM 64 (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_64_NO_DONE,
    'Metra sum SF TD Energy BEST LAM OPTION DIM (no done)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    'DIAYN (no done)': KITCHEN_DIAYN_NO_DONE,
}

YLABEL = 'EvalOp/MjNumUniqueCoords'
all_experiments = [ant_experiments, cheetah_experiments, quadruped_experiments, humanoid_experiments, kitchen_experiments]
titles = ['Ant (States)', 'HalfCheetah (States)', 'Quadruped (Pixels)', 'Humanoid (Pixels)', 'Kitchen (Pixels)']

# Create a figure with subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 3))

for ax, experiment, title in zip(axes, all_experiments, titles):
    for label, filepaths in experiment.items():
        mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, YLABEL if not 'Kitchen' in title else 'EvalOp/KitchenOverall')
        ax.plot(total_env_steps, mean_values, label=label)
        ax.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2)
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('State Coverage')
    ax.set_title(title)
    # Position the legend outside the plot
    # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust the plot area to make space for the legend
plt.savefig('figures/paper/state_space_coverage.pdf', bbox_inches='tight')
plt.show()
