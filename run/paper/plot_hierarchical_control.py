import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# sns.set_palette("deep")
# plt.style.use("seaborn")
sns.set_style("whitegrid")
SNS_PALETTE = "colorblind"

# ***********
# ANT RESULTS
# ***********
ANT_METRA_SUM_NO_DONE_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd000_s_21903013.0.1724190098_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd001_s_21903014.0.1724190098_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd002_s_21996881.0.1724940241_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd003_s_21996882.0.1724940241_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd004_s_21996883.0.1724940241_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd005_s_22248512.0.1727366727_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd006_s_22248513.0.1727366727_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd007_s_22248514.0.1727366726_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd008_s_22248515.0.1727366727_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sum_no_done_chkpt_40k/sd009_s_22248516.0.1727366726_ant_nav_prime_sac/progress_eval.csv',
]

ANT_METRA_SF_TD_ENERGY_LAM_5_DIM_2_NO_DONE_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd000_s_21902835.0.1724189675_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd001_s_21902836.0.1724189800_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd002_s_22003049.0.1725025947_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd003_s_22003050.0.1725025947_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd004_s_22003051.0.1725025947_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd005_s_22248465.0.1727366612_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd006_s_22248466.0.1727366612_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd007_s_22248467.0.1727366615_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd008_s_22248468.0.1727366615_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_energy_lam_5_dim_2_chkpt_40k/sd009_s_22248469.0.1727366616_ant_nav_prime_sac/progress_eval.csv',
]

ANT_DIAYN_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd000_s_21968217.0.1724694406_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd001_s_21968218.0.1724694407_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd002_s_21996928.0.1724940598_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd003_s_21996929.0.1724940597_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd004_s_21996930.0.1724940597_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd005_s_22248420.0.1727366514_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd006_s_22248421.0.1727366514_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd007_s_22248422.0.1727366514_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd008_s_22248423.0.1727366514_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_diayn_chkpt_40k/sd009_s_22248424.0.1727366514_ant_nav_prime_sac/progress_eval.csv',
]

ANT_CIC_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd000_s_21965105.0.1724686156_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd001_s_21965106.0.1724686156_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd002_s_21996971.0.1724944747_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd003_s_21996972.0.1724944748_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd004_s_21996973.0.1724944747_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd005_s_22248265.0.1727366341_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd006_s_22248266.0.1727366340_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd007_s_22248267.0.1727366340_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd008_s_22248268.0.1727366341_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_cic_chkpt_40k/sd009_s_22248269.0.1727366345_ant_nav_prime_sac/progress_eval.csv',
]

ANT_DADS_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd000_s_21965064.0.1724686110_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd001_s_21965065.0.1724686110_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd002_s_21996931.0.1724944748_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd003_s_21996932.0.1724944747_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd004_s_21996933.0.1724944747_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd005_s_22248350.0.1727366422_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd006_s_22248351.0.1727366423_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd007_s_22248352.0.1727366422_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd008_s_22248353.0.1727366422_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_dads_chkpt_40k/sd009_s_22248354.0.1727366422_ant_nav_prime_sac/progress_eval.csv',
]

ANT_VISR_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd000_s_22093422.0.1725996285_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd001_s_22093423.0.1725996285_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd002_s_22093424.0.1725996285_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd003_s_22093425.0.1725996285_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd004_s_22093426.0.1725996285_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd005_s_22248539.0.1727366801_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd006_s_22248540.0.1727366800_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd007_s_22248541.0.1727366800_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd008_s_22248542.0.1727366800_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_visr_chkpt_40k/sd009_s_22248543.0.1727366799_ant_nav_prime_sac/progress_eval.csv',
]

# ********************
# CHEETAH GOAL RESULTS
# ********************

CHEETAH_GOAL_METRA_SUM_NO_DONE_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd000_s_21903516.0.1724191297_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd001_s_21903517.0.1724191295_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd002_s_21983726.0.1724802299_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd003_s_21983727.0.1724802298_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd004_s_21983728.0.1724802298_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd005_s_22251329.0.1727399926_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd006_s_22251330.0.1727405900_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd007_s_22251331.0.1727409136_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd008_s_22251332.0.1727415551_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sum_no_done_chkpt_40k/sd009_s_22251333.0.1727448811_half_cheetah_goal_ppo/progress_eval.csv',
]

CHEETAH_GOAL_METRA_SF_TD_ENERGY_LAM_5_DIM_2_NO_DONE_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd000_s_21917005.0.1724254652_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd001_s_21917006.0.1724254652_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd002_s_21999667.0.1724982455_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd003_s_21999668.0.1724982455_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd004_s_21999669.0.1724982454_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd005_s_22251298.0.1727373034_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd006_s_22251299.0.1727396864_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd007_s_22251300.0.1727398395_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd008_s_22251301.0.1727399024_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd009_s_22251302.0.1727399177_half_cheetah_goal_ppo/progress_eval.csv',
]

CHEETAH_GOAL_DIAYN_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd000_s_21967613.0.1724690660_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd001_s_21967614.0.1724690660_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd002_s_21997008.0.1724944748_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd003_s_21997009.0.1724944747_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd004_s_21997010.0.1724944747_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd005_s_22251250.0.1727372942_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd006_s_22251251.0.1727372943_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd007_s_22251252.0.1727372943_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd008_s_22251253.0.1727372942_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_diayn_chkpt_40k/sd009_s_22251254.0.1727372942_half_cheetah_goal_ppo/progress_eval.csv',
]

CHEETAH_GOAL_CIC_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd000_s_21967573.0.1724690593_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd001_s_21967574.0.1724690592_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd002_s_21997024.0.1724944801_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd003_s_21997025.0.1724944801_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd004_s_21997026.0.1724944804_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd005_s_22251189.0.1727372800_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd006_s_22251190.0.1727372800_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd007_s_22251191.0.1727372800_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd008_s_22251192.0.1727372800_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_cic_chkpt_40k/sd009_s_22251193.0.1727372800_half_cheetah_goal_ppo/progress_eval.csv',
]

CHEETAH_GOAL_DADS_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd000_s_21967657.0.1724690793_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd001_s_21967658.0.1724690793_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd002_s_21997011.0.1724944747_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd003_s_21997012.0.1724944748_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd004_s_21997013.0.1724944747_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd005_s_22251222.0.1727372871_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd006_s_22251223.0.1727372877_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd007_s_22251224.0.1727372877_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd008_s_22251225.0.1727372877_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_dads_chkpt_40k/sd009_s_22251226.0.1727372877_half_cheetah_goal_ppo/progress_eval.csv',
]

CHEETAH_GOAL_VISR_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd000_s_22093427.0.1725996361_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd001_s_22093428.0.1725996361_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd002_s_22093429.0.1725996361_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd003_s_22093430.0.1725996371_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd004_s_22093431.0.1725996371_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd005_s_22251355.0.1727448810_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd006_s_22251356.0.1727448810_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd007_s_22251357.0.1727448811_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd008_s_22251358.0.1727448810_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_visr_chkpt_40k/sd009_s_22251359.0.1727449194_half_cheetah_goal_ppo/progress_eval.csv',
]


# **********************
# CHEETAH HURDLE RESULTS
# **********************

CHEETAH_HURDLE_METRA_SUM_NO_DONE_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd000_s_21904335.0.1724192978_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd001_s_21904336.0.1724192978_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd002_s_21983738.0.1724802298_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd003_s_21983739.0.1724802298_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd004_s_21983740.0.1724802298_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd005_s_22251470.0.1727452863_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd006_s_22251471.0.1727453065_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd007_s_22251472.0.1727453065_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd008_s_22251473.0.1727453068_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sum_no_done_chkpt_40k/sd009_s_22251474.0.1727453068_half_cheetah_hurdle_ppo/progress_eval.csv',
]

CHEETAH_HURDLE_METRA_SF_TD_ENERGY_LAM_5_DIM_2_NO_DONE_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd000_s_21917014.0.1724254652_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd001_s_21917015.0.1724254652_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd002_s_21999683.0.1724982580_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd003_s_21999684.0.1724982580_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd004_s_21999685.0.1724982579_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd005_s_22251445.0.1727452786_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd006_s_22251446.0.1727452840_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd007_s_22251447.0.1727452841_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd008_s_22251448.0.1727452858_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td_energy_lam_5_dim_2_no_done_chkpt_40k/sd009_s_22251449.0.1727452863_half_cheetah_hurdle_ppo/progress_eval.csv',
]

CHEETAH_HURDLE_DIAYN_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd000_s_21967695.0.1724691189_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd001_s_21967696.0.1724691188_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd002_s_21997064.0.1724945403_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd003_s_21997065.0.1724946024_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd004_s_21997066.0.1724945582_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd005_s_22251437.0.1727450159_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd006_s_22251438.0.1727450160_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd007_s_22251439.0.1727450159_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd008_s_22251440.0.1727450160_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_diayn_chkpt_40k/sd009_s_22251441.0.1727452783_half_cheetah_hurdle_ppo/progress_eval.csv',
]

CHEETAH_HURDLE_CIC_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd000_s_21967771.0.1724691364_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd001_s_21967772.0.1724691364_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd002_s_21997070.0.1724946062_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd003_s_21997071.0.1724946123_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd004_s_21997072.0.1724946181_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd005_s_22251382.0.1727449195_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd006_s_22251383.0.1727449194_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd007_s_22251384.0.1727449198_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd008_s_22251385.0.1727449197_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_cic_chkpt_40k/sd009_s_22251386.0.1727449900_half_cheetah_hurdle_ppo/progress_eval.csv',
]

CHEETAH_HURDLE_DADS_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd000_s_21967737.0.1724691291_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd001_s_21967738.0.1724691291_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd002_s_21997067.0.1724946024_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd003_s_21997068.0.1724946024_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd004_s_21997069.0.1724945884_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd005_s_22251408.0.1727449900_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd006_s_22251409.0.1727449902_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd007_s_22251410.0.1727449903_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd008_s_22251411.0.1727449903_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_dads_chkpt_40k/sd009_s_22251412.0.1727450159_half_cheetah_hurdle_ppo/progress_eval.csv',
]

CHEETAH_HURDLE_VISR_CHKPT_40k = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd000_s_22093433.0.1725996468_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd001_s_22093434.0.1725996468_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd002_s_22093435.0.1725996477_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd003_s_22093436.0.1725996477_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd004_s_22093437.0.1725996477_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd005_s_22251476.0.1727453068_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd006_s_22251477.0.1727453159_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd007_s_22251478.0.1727453159_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd008_s_22251479.0.1727453159_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_visr_chkpt_40k/sd009_s_22251480.0.1727453159_half_cheetah_hurdle_ppo/progress_eval.csv',
]


# *****************
# QUADRUPED RESULTS
# *****************

QUADRUPED_METRA_SUM_NO_DONE_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd000_s_21904102.0.1724192437_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd001_s_21904103.0.1724192441_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd002_s_21983756.0.1724802298_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd003_s_21983757.0.1724802298_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd004_s_21983758.0.1724802340_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd005_s_22251847.0.1727536338_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd006_s_22251848.0.1727536337_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd007_s_22251849.0.1727536340_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd008_s_22251850.0.1727536340_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sum_no_done_chkpt_3k/sd009_s_22251851.0.1727536340_dmc_quadruped_goal_sac/progress_eval.csv',
]

QUADRUPED_METRA_SF_TD_ENERGY_LAM_5_DIM_4_NO_DONE_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd000_s_21904280.0.1724192865_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd001_s_21904281.0.1724192865_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd002_s_21983790.0.1724802338_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd003_s_21983791.0.1724802338_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd004_s_21983792.0.1724802460_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd005_s_22251818.0.1727535625_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd006_s_22251819.0.1727535626_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd007_s_22251820.0.1727535625_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd008_s_22251821.0.1727535630_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td_energy_lam_5_dim_4_no_done_chkpt_3k/sd009_s_22251822.0.1727535630_dmc_quadruped_goal_sac/progress_eval.csv',
]

QUADRUPED_DIAYN_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd000_s_21968261.0.1724694645_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd001_s_21968262.0.1724694645_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd002_s_21997260.0.1724947686_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd003_s_21997261.0.1724948350_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd004_s_21997262.0.1724948587_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd005_s_22251790.0.1727535226_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd006_s_22251791.0.1727535225_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd007_s_22251792.0.1727535226_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd008_s_22251793.0.1727535226_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_diayn_chkpt_3/sd009_s_22251794.0.1727535226_dmc_quadruped_goal_sac/progress_eval.csv',
]

QUADRUPED_CIC_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd000_s_21967841.0.1724692009_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd001_s_21967842.0.1724692009_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd002_s_21997263.0.1724949626_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd003_s_21997264.0.1724949626_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd004_s_21997265.0.1724949967_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd005_s_22251764.0.1727485626_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd006_s_22251765.0.1727486348_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd007_s_22251766.0.1727492330_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd008_s_22251767.0.1727495552_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_cic_chkpt_3/sd009_s_22251768.0.1727502030_dmc_quadruped_goal_sac/progress_eval.csv',
]

QUADRUPED_VISR_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd000_s_22093443.0.1725996846_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd001_s_22093444.0.1725996845_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd002_s_22093445.0.1725996845_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd003_s_22093446.0.1725996845_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd004_s_22093447.0.1725996850_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd005_s_22251894.0.1727536586_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd006_s_22251895.0.1727536586_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd007_s_22251896.0.1727536586_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd008_s_22251897.0.1727536586_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_visr_chkpt_3k/sd009_s_22251898.0.1727536587_dmc_quadruped_goal_sac/progress_eval.csv',
]

# ****************
# HUMANOID RESULTS
# ****************

HUMANOID_METRA_SUM_NO_DONE_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd000_s_21904797.0.1724194063_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd001_s_21904798.0.1724194063_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd002_s_21983826.0.1724802459_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd003_s_21983827.0.1724802765_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd004_s_21983828.0.1724802820_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd005_s_22251670.0.1727459307_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd006_s_22251671.0.1727459385_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd007_s_22251672.0.1727459385_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd008_s_22251673.0.1727459385_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sum_no_done_chkpt_3k/sd009_s_22251674.0.1727459384_dmc_humanoid_goal_sac/progress_eval.csv',
]

HUMANOID_METRA_SF_TD_ENERGY_LAM_5_DIM_8_NO_DONE_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd000_s_21904761.0.1724193937_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd001_s_21904762.0.1724193937_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd002_s_21983829.0.1724803000_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd003_s_21983830.0.1724803116_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd004_s_21983831.0.1724803116_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd005_s_22251643.0.1727459246_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd006_s_22251644.0.1727459299_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd007_s_22251645.0.1727459306_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd008_s_22251646.0.1727459306_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_metra_sf_td_energy_lam_5_dim_8_no_done_chkpt_3k/sd009_s_22251647.0.1727459307_dmc_humanoid_goal_sac/progress_eval.csv',
]

HUMANOID_DIAYN_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd000_s_21968259.0.1724694430_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd001_s_21968260.0.1724694429_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd002_s_21997296.0.1724952076_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd003_s_21997297.0.1724959100_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd004_s_21997298.0.1724959100_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd005_s_22251597.0.1727453243_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd006_s_22251598.0.1727459247_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd007_s_22251599.0.1727459246_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd008_s_22251600.0.1727459246_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_diayn_chkpt_3k/sd009_s_22251601.0.1727459246_dmc_humanoid_goal_sac/progress_eval.csv',
]

HUMANOID_CIC_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd000_s_21967919.0.1724692411_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd001_s_21967920.0.1724692411_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd002_s_21997299.0.1724959096_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd003_s_21997300.0.1724959096_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd004_s_21997301.0.1724962646_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd005_s_22251571.0.1727453160_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd006_s_22251572.0.1727453242_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd007_s_22251573.0.1727453243_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd008_s_22251574.0.1727453243_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_cic_chkpt_3k/sd009_s_22251575.0.1727453243_dmc_humanoid_goal_sac/progress_eval.csv',
]

HUMANOID_VISR_CHKPT_3k = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd000_s_22093438.0.1725996683_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd001_s_22093439.0.1725996689_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd002_s_22093440.0.1725996689_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd003_s_22093441.0.1725996689_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd004_s_22093442.0.1725996689_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd005_s_22251716.0.1727459385_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd006_s_22251717.0.1727459457_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd007_s_22251718.0.1727483301_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd008_s_22251719.0.1727484856_dmc_humanoid_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_goal_visr_chkpt_3k/sd009_s_22251720.0.1727485461_dmc_humanoid_goal_sac/progress_eval.csv',
]

def compute_mean_and_std(filepaths, y_label, smooth: bool = False):
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

    if smooth:
        truncated_values_smoothed = np.zeros_like(truncated_values)
        for b in range(len(truncated_values)):
            for t in range(len(truncated_values[b])):
                truncated_values_smoothed[b, t] = np.mean(truncated_values[b, max(0, t-10):t + 1])
        
        truncated_values = truncated_values_smoothed
    
    mean_values = np.mean(truncated_values, axis=0)
    std_dev = np.std(truncated_values, axis=0, ddof=1)
    
    total_env_steps = df['TotalEnvSteps'].values[:min_length]
    return mean_values, std_dev, total_env_steps

def plot_with_confidence_bands(total_env_steps, mean_values, std_dev, label):
    plt.plot(total_env_steps, mean_values, label=label) #marker='o', markersize=3, markeredgewidth=0.5, markeredgecolor="#F7F7FF", linewidth=1)
    plt.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2)

# Define the experiment settings
ant_experiments = {
    'CSF (ours)': ANT_METRA_SF_TD_ENERGY_LAM_5_DIM_2_NO_DONE_CHKPT_40k,
    'METRA': ANT_METRA_SUM_NO_DONE_CHKPT_40k,
    'DIAYN': ANT_DIAYN_CHKPT_40k,
    'CIC': ANT_CIC_CHKPT_40k,
    'DADS': ANT_DADS_CHKPT_40k,
    'VISR': ANT_VISR_CHKPT_40k,
}

cheetah_goal_experiments = {
    'CSF (ours)': CHEETAH_GOAL_METRA_SF_TD_ENERGY_LAM_5_DIM_2_NO_DONE_CHKPT_40k,
    'METRA': CHEETAH_GOAL_METRA_SUM_NO_DONE_CHKPT_40k,
    'DIAYN': CHEETAH_GOAL_DIAYN_CHKPT_40k,
    'CIC': CHEETAH_GOAL_CIC_CHKPT_40k,
    'DADS': CHEETAH_GOAL_DADS_CHKPT_40k,
    'VISR': CHEETAH_GOAL_VISR_CHKPT_40k,
}

cheetah_hurdle_experiments = {
    'CSF (ours)': CHEETAH_HURDLE_METRA_SF_TD_ENERGY_LAM_5_DIM_2_NO_DONE_CHKPT_40k,
    'METRA': CHEETAH_HURDLE_METRA_SUM_NO_DONE_CHKPT_40k,
    'DIAYN': CHEETAH_HURDLE_DIAYN_CHKPT_40k,
    'CIC': CHEETAH_HURDLE_CIC_CHKPT_40k,
    'DADS': CHEETAH_HURDLE_DADS_CHKPT_40k,
    'VISR': CHEETAH_HURDLE_VISR_CHKPT_40k,
}

quadruped_experiments = {
    'CSF (ours)': QUADRUPED_METRA_SF_TD_ENERGY_LAM_5_DIM_4_NO_DONE_CHKPT_3k,
    'METRA': QUADRUPED_METRA_SUM_NO_DONE_CHKPT_3k,
    'DIAYN': QUADRUPED_DIAYN_CHKPT_3k,
    'CIC': QUADRUPED_CIC_CHKPT_3k,
    'VISR': QUADRUPED_VISR_CHKPT_3k,
}

humanoid_experiments = {
    'CSF (ours)': HUMANOID_METRA_SF_TD_ENERGY_LAM_5_DIM_8_NO_DONE_CHKPT_3k,
    'METRA': HUMANOID_METRA_SUM_NO_DONE_CHKPT_3k,
    'DIAYN': HUMANOID_DIAYN_CHKPT_3k,
    'CIC': HUMANOID_CIC_CHKPT_3k,
    'VISR': HUMANOID_VISR_CHKPT_3k,
}

COLOR_MAP = {
    'CSF (ours)': sns.color_palette(SNS_PALETTE)[9],
    'METRA': sns.color_palette(SNS_PALETTE)[1],
    'DIAYN': sns.color_palette(SNS_PALETTE)[2],
    'DADS': sns.color_palette(SNS_PALETTE)[4],
    'CIC': sns.color_palette(SNS_PALETTE)[7],
    'VISR': sns.color_palette(SNS_PALETTE)[0],
}

MARKER_MAP = {
    'CSF (ours)': 'o',
    'METRA': 's',
    'DIAYN': 'v',
    'DADS': 'D',
    'CIC': 'X',
    'VISR': 'P',
}

MARKEVERY_MAP = {
    'AntMultiGoal': 15,
    'HalfCheetahGoal': 15,
    'HalfCheetahHurdle': 15,
    'QuadrupedGoal': 15,
    'HumanoidGoal': 15,
}

YLABEL = 'EvalOp/AverageReturn'
all_experiments = [ant_experiments, cheetah_goal_experiments, cheetah_hurdle_experiments, quadruped_experiments, humanoid_experiments]
titles = ['AntMultiGoal', 'HalfCheetahGoal', 'HalfCheetahHurdle', 'QuadrupedGoal', 'HumanoidGoal']

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 6.5))

legend_labels = []
legend_handles = []
ax_idx = 0
for ax, experiment, title in zip(axes.flatten(), all_experiments, titles):
    xmax = int(1e9)
    ymax = 0
    for label, filepaths in experiment.items():
        mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, YLABEL if not 'Kitchen' in title else 'EvalOp/KitchenOverall', smooth=True)
        handle, = ax.plot(total_env_steps, mean_values, label=label, linewidth=3, color=COLOR_MAP[label], marker=MARKER_MAP[label], markevery=MARKEVERY_MAP[title], markersize=12)
        ax.tick_params(axis='x', labelsize="16")
        ax.tick_params(axis='y', labelsize="16")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2, color=COLOR_MAP[label])

        if label not in legend_labels:
            legend_labels.append(label)
            legend_handles.append(handle)

        xmax = min(xmax, total_env_steps[-1])
        ymax = max(ymax, np.max(mean_values + std_dev))

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_xlabel('Env Steps', fontsize="18")
    if ax_idx % 3 == 0:
        ax.set_ylabel('Return (past 10)', fontsize="18")
    ax.set_title(title, fontsize="22", fontweight="bold")
    ax.get_xaxis().get_offset_text().set_fontsize(14)
    # Position the legend outside the plot
    # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_idx += 1

axes[1][-1].axis('off')
# fig.legend(legend_handles, legend_labels, frameon=True, bbox_to_anchor=(1.08, 0.6), fontsize="20")
plt.legend(handles=legend_handles, labels=legend_labels, loc='center', fontsize="20", frameon=True, bbox_to_anchor=(0.5, 0.5))
fig.tight_layout()  # Adjust the plot area to make space for the legend
plt.savefig('figures/paper/hierarchical_control.pdf', bbox_inches='tight')
plt.show()
