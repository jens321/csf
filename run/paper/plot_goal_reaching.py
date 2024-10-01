import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# sns.set_palette("deep")
# plt.style.use("seaborn")
sns.set_style("whitegrid")
SNS_PALETTE = "colorblind"

# ***********
# ANT RESULTS
# ***********
ANT_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd000_s_21497671.0.1722112978_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd001_s_21497672.0.1722112978_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd002_s_21737075.0.1723674377_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd003_s_21737076.0.1723674376_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd004_s_21737077.0.1723674376_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd005_s_22093963.0.1726004892_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd006_s_22093964.0.1726004893_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd007_s_22093965.0.1726004892_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd008_s_22093966.0.1726004893_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd009_s_22093967.0.1726004892_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567111.0.1722950266_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567112.0.1722950266_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd002_s_21989134.0.1724882684_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd003_s_21989135.0.1724882683_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd004_s_21989136.0.1724882683_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd005_s_22094014.0.1726004952_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd006_s_22094015.0.1726082544_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd007_s_22094017.0.1726082543_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd008_s_22094018.0.1726082544_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd009_s_22094019.0.1726082544_ant_metra_sf/progress_eval.csv'
]

ANT_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd000_s_21744886.0.1723744180_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd001_s_21744887.0.1723744179_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd002_s_21983393.0.1724796218_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd003_s_21983394.0.1724796218_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd004_s_21983395.0.1724796218_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd005_s_22093874.0.1726004791_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd006_s_22093875.0.1726004792_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd007_s_22093876.0.1726004791_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd008_s_22093877.0.1726004791_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd009_s_22093878.0.1726004792_ant_metra/progress_eval.csv'
]

ANT_DADS_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd000_s_21742705.0.1723741056_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd001_s_21742706.0.1723741056_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd002_s_21983406.0.1724796269_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd003_s_21983407.0.1724796269_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd004_s_21983408.0.1724796270_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd005_s_22093869.0.1726004780_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd006_s_22093870.0.1726004781_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd007_s_22093871.0.1726004781_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd008_s_22093872.0.1726004780_ant_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_dads_no_done/sd009_s_22093873.0.1726004780_ant_dads/progress_eval.csv'
]

ANT_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd000_s_21988746.0.1724872711_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd001_s_21988747.0.1724872711_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd002_s_21989998.0.1724882737_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd003_s_21989999.0.1724882737_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd004_s_21990000.0.1724882737_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd005_s_22094066.0.1726082544_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd006_s_22094067.0.1726082779_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd007_s_22094068.0.1726082778_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd008_s_22094069.0.1726082778_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd009_s_22094070.0.1726082783_ant_metra_sf/progress_eval.csv'
]

# ***************
# CHEETAH RESULTS
# ***************

CHEETAH_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd000_s_21625261.0.1723157129_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd001_s_21625262.0.1723157129_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd002_s_21737085.0.1723674577_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd003_s_21737086.0.1723674591_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd004_s_21737087.0.1723674591_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd005_s_22094134.0.1726091225_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd006_s_22094135.0.1726091277_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd007_s_22094136.0.1726091277_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd008_s_22094137.0.1726091285_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd009_s_22094138.0.1726091285_half_cheetah_metra/progress_eval.csv'
]

CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625253.0.1723156995_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625254.0.1723156993_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd002_s_21989200.0.1724882737_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd003_s_21989201.0.1724882738_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd004_s_21989202.0.1724882737_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd005_s_22094111.0.1726082902_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd006_s_22094112.0.1726083111_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd007_s_22094113.0.1726083111_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd008_s_22094114.0.1726083111_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd009_s_22094115.0.1726083110_half_cheetah_metra_sf/progress_eval.csv'
]

CHEETAH_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd000_s_21744888.0.1723744179_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd001_s_21744889.0.1723744179_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd002_s_21983412.0.1724796671_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd003_s_21983413.0.1724796671_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd004_s_21983414.0.1724796671_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd005_s_22094121.0.1726083262_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd006_s_22094122.0.1726091225_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd007_s_22094123.0.1726091225_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd008_s_22094124.0.1726091226_half_cheetah_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd009_s_22094125.0.1726091225_half_cheetah_metra/progress_eval.csv'
]

CHEETAH_DADS_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd000_s_21742707.0.1723741056_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd001_s_21742708.0.1723741057_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd002_s_21983415.0.1724796671_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd003_s_21983416.0.1724796672_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd004_s_21983417.0.1724796671_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd005_s_22094116.0.1726083110_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd006_s_22094117.0.1726083265_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd007_s_22094118.0.1726083265_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd008_s_22094119.0.1726083265_half_cheetah_dads/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_dads_no_done/sd009_s_22094120.0.1726083265_half_cheetah_dads/progress_eval.csv'
]

CHEETAH_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd000_s_21988748.0.1724872821_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd001_s_21988749.0.1724872821_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd002_s_21990001.0.1724882864_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd003_s_21990002.0.1724882863_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd004_s_21990003.0.1724882863_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd005_s_22094128.0.1726091226_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd006_s_22094129.0.1726091226_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd007_s_22094130.0.1726091226_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd008_s_22094131.0.1726091225_half_cheetah_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd009_s_22094132.0.1726091226_half_cheetah_metra_sf/progress_eval.csv'
]

# *****************
# QUADRUPED RESULTS
# *****************

QUADRUPED_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd000_s_21497823.0.1722113111_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd001_s_21497824.0.1722113111_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd002_s_21737101.0.1723675020_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd003_s_21737102.0.1723675020_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd004_s_21737103.0.1723675020_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd005_s_22114610.0.1726523194_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd006_s_22114611.0.1726523194_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd007_s_22114612.0.1726523194_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd008_s_22114613.0.1726523194_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd009_s_22114614.0.1726523195_dmc_quadruped_metra/progress_eval.csv'
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595372.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595373.0.1723049191_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd002_s_21737108.0.1723675162_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd003_s_21737109.0.1723675162_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd004_s_21737110.0.1723675162_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd005_s_22114615.0.1726523313_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd006_s_22114616.0.1726523313_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd007_s_22114617.0.1726523313_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd008_s_22114618.0.1726523313_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd009_s_22114619.0.1726523313_dmc_quadruped_metra_sf/progress_eval.csv'
]

QUADRUPED_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd000_s_21744892.0.1723762165_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd001_s_21744893.0.1723762164_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd002_s_21983451.0.1724796576_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd003_s_21983452.0.1724796576_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd004_s_21983453.0.1724796576_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd005_s_22114620.0.1726523319_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd006_s_22114621.0.1726523319_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd007_s_22114622.0.1726523319_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd008_s_22114623.0.1726523319_dmc_quadruped_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd009_s_22114624.0.1726523319_dmc_quadruped_metra/progress_eval.csv'
]

QUADRUPED_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd000_s_21988761.0.1724883020_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd001_s_21988762.0.1724883020_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd002_s_21990024.0.1724883102_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd003_s_21990025.0.1724883102_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd004_s_21990026.0.1724884360_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd005_s_22114630.0.1726609624_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd006_s_22114631.0.1726609624_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd007_s_22114632.0.1726609624_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd008_s_22114633.0.1726609624_dmc_quadruped_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd009_s_22114634.0.1726609624_dmc_quadruped_metra_sf/progress_eval.csv'
]

# ****************
# HUMANOID RESULTS
# ****************

HUMANOID_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd000_s_21500598.0.1722126378_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd001_s_21500599.0.1722126378_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd002_s_21737187.0.1723677005_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd003_s_21737188.0.1723762164_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd004_s_21737189.0.1723762164_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd005_s_22114672.0.1726696073_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd006_s_22114673.0.1726696073_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd007_s_22114674.0.1726696073_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd008_s_22114675.0.1726696073_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd009_s_22114676.0.1726696073_dmc_humanoid_metra/progress_eval.csv'

]

HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646159.0.1723330684_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646160.0.1723330684_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd002_s_21737156.0.1723675597_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd003_s_21737157.0.1723675605_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd004_s_21737158.0.1723675605_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd005_s_22114667.0.1726609793_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd006_s_22114668.0.1726609794_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd007_s_22114669.0.1726609793_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd008_s_22114670.0.1726609794_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd009_s_22114671.0.1726609794_dmc_humanoid_metra_sf/progress_eval.csv'
]

HUMANOID_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd000_s_21744890.0.1723762165_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd001_s_21744891.0.1723762164_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd002_s_21983457.0.1724796628_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd003_s_21983458.0.1724796635_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd004_s_21983459.0.1724796635_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd005_s_22114658.0.1726609750_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd006_s_22114659.0.1726609750_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd007_s_22114660.0.1726609750_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd008_s_22114661.0.1726609750_dmc_humanoid_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd009_s_22114662.0.1726609750_dmc_humanoid_metra/progress_eval.csv'
]

HUMANOID_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd000_s_21988763.0.1724883019_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd001_s_21988764.0.1724883019_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd002_s_21990038.0.1724884360_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd003_s_21990039.0.1724884360_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd004_s_21990040.0.1724884360_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd005_s_22114677.0.1726696177_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd006_s_22114678.0.1726696176_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd007_s_22114679.0.1726696176_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd008_s_22114680.0.1726696177_dmc_humanoid_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd009_s_22114681.0.1726696176_dmc_humanoid_metra_sf/progress_eval.csv'
]

# ***************
# KITCHEN RESULTS
# ***************

KITCHEN_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd000_s_21983615.0.1724883020_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd001_s_21983616.0.1724883020_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd002_s_21948564.0.1724528083_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd003_s_21948565.0.1724528083_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd004_s_21948566.0.1724528083_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd005_s_22114697.0.1726782634_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd006_s_22114698.0.1726782633_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd007_s_22114699.0.1726782633_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd008_s_22114700.0.1726782634_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd009_s_22114701.0.1726782633_kitchen_metra/progress_eval.csv'
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd000_s_21983604.0.1724883020_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd001_s_21983605.0.1724883019_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd002_s_21948561.0.1724528044_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd003_s_21948562.0.1724528044_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd004_s_21948563.0.1724528044_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd005_s_22114687.0.1726696234_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd006_s_22114688.0.1726696234_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd007_s_22114689.0.1726696234_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd008_s_22114690.0.1726696234_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd009_s_22114691.0.1726696234_kitchen_metra_sf/progress_eval.csv'
]

KITCHEN_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd000_s_21948559.0.1724528022_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd001_s_21948560.0.1724528022_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd002_s_21983601.0.1724797950_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd003_s_21983602.0.1724797950_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd004_s_21983603.0.1724797950_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd005_s_22114692.0.1726782515_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd006_s_22114693.0.1726782515_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd007_s_22114694.0.1726782515_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd008_s_22114695.0.1726782515_kitchen_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd009_s_22114696.0.1726782515_kitchen_metra/progress_eval.csv'
]

KITCHEN_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd000_s_21988765.0.1724883020_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd001_s_21988766.0.1724883072_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd002_s_21990042.0.1724884360_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd003_s_21990043.0.1724884418_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd004_s_21990044.0.1724884418_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd005_s_22114718.0.1726782638_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd006_s_22114719.0.1726782637_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd007_s_22114720.0.1726782637_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd008_s_22114721.0.1726782638_kitchen_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd009_s_22114722.0.1726782638_kitchen_metra_sf/progress_eval.csv'
]

# ***************
# ROBOBIN RESULTS
# ***************

# ROBOBIN_METRA_anonymous = [
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_anonymous/progress_eval_0.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_anonymous/progress_eval_1.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_anonymous/progress_eval_2.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_anonymous/progress_eval_3.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_anonymous/progress_eval_4.csv'
# ]

# ROBOBIN_CSF_anonymous = [
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_csf_anonymous/progress_eval_0.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_csf_anonymous/progress_eval_1.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_csf_anonymous/progress_eval_600.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/robobin_csf_anonymous/progress_eval_700.csv'
# ]

ROBOBIN_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd000_s_21956211.0.1724615917_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd001_s_21956212.0.1724615917_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd002_s_21991775.0.1724897402_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd003_s_21991776.0.1724969434_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd004_s_21991777.0.1724969434_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd005_s_22114728.0.1726868940_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd006_s_22114729.0.1726868940_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd007_s_22114730.0.1726868940_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd008_s_22114731.0.1726869053_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd009_s_22114732.0.1726869053_robobin_image_metra/progress_eval.csv'
]

ROBOBIN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_9_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd000_s_21956213.0.1724615934_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd001_s_21956214.0.1724615934_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd002_s_21991778.0.1724969434_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd003_s_21991779.0.1724969434_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd004_s_21991780.0.1724969434_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd005_s_22114733.0.1726869053_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd006_s_22114734.0.1726869053_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd007_s_22114735.0.1726869057_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd008_s_22114736.0.1726869057_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd009_s_22114737.0.1726869057_robobin_image_metra_sf/progress_eval.csv'
]

ROBOBIN_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd000_s_21956203.0.1724615792_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd001_s_21956204.0.1724615792_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd002_s_21991781.0.1724969435_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd003_s_21991782.0.1724969434_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd004_s_21991783.0.1724969540_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd005_s_22114738.0.1726869057_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd006_s_22114739.0.1726869115_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd007_s_22114740.0.1726869114_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd008_s_22114741.0.1726869115_robobin_image_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd009_s_22114742.0.1726869114_robobin_image_metra/progress_eval.csv'
]

ROBOBIN_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd000_s_21996808.0.1724970801_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd001_s_21996809.0.1724970801_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd002_s_21996810.0.1724970800_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd003_s_21996811.0.1724970826_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd004_s_21996812.0.1724983825_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd005_s_22114723.0.1726782693_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd006_s_22114724.0.1726782693_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd007_s_22114725.0.1726782693_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd008_s_22114726.0.1726782693_robobin_image_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd009_s_22114727.0.1726868940_robobin_image_metra_sf/progress_eval.csv'
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
    # NOTE: We are plotting the negative of the goal distance
    # if not 'Kitchen' in y_label:
    #     truncated_values = -truncated_values

    if 'kitchen' in filepaths[0]:
        truncated_values /= 50
    elif 'robobin' in filepaths[0]:
        truncated_values /= 200

    mean_values = np.mean(truncated_values, axis=0)
    std_dev = np.std(truncated_values, axis=0, ddof=1)
    # t_ci = stats.sem(truncated_values, axis=0) * stats.t.isf(0.05 / 2, len(truncated_values) - 1)
    # std_dev = t_ci
    
    total_env_steps = df['TotalEnvSteps'].values[:min_length]
    return mean_values, std_dev, total_env_steps

def plot_with_confidence_bands(total_env_steps, mean_values, std_dev, label):
    plt.plot(total_env_steps, mean_values, label=label) #marker='o', markersize=3, markeredgewidth=0.5, markeredgecolor="#F7F7FF", linewidth=1)
    plt.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2)

# Define the experiment settings
ant_experiments = {
    'CSF (ours)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    'METRA': ANT_METRA_SUM_NO_DONE,
    'DIAYN': ANT_DIAYN_NO_DONE,
    'VISR': ANT_VISR,
}

cheetah_experiments = {
    'CSF (ours)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    'METRA': CHEETAH_METRA_SUM_NO_DONE,
    'DIAYN': CHEETAH_DIAYN_NO_DONE,
    'VISR': CHEETAH_VISR,
}

quadruped_experiments = {
    'CSF (ours)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    'METRA': QUADRUPED_METRA_SUM_NO_DONE,
    'DIAYN': QUADRUPED_DIAYN_NO_DONE,
    'VISR': QUADRUPED_VISR,
}

humanoid_experiments = {
    'CSF (ours)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    'METRA': HUMANOID_METRA_SUM_NO_DONE,
    'DIAYN': HUMANOID_DIAYN_NO_DONE,
    'VISR': HUMANOID_VISR,
}

kitchen_experiments = {
    'CSF (ours)': KITCHEN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_4_NO_DONE,
    'METRA': KITCHEN_METRA_SUM_NO_DONE,
    'DIAYN': KITCHEN_DIAYN_NO_DONE,
    'VISR': KITCHEN_VISR,
}

robobin_experiments = {
    'CSF (ours)': ROBOBIN_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_9_NO_DONE, # ROBOBIN_CSF_anonymous,
    'METRA': ROBOBIN_METRA_SUM_NO_DONE, # ROBOBIN_METRA_anonymous,
    'DIAYN': ROBOBIN_DIAYN_NO_DONE,
    'VISR': ROBOBIN_VISR,
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
    'Ant (States)': 7,
    'HalfCheetah (States)': 5,
    'Quadruped (Pixels)': 3,
    'Humanoid (Pixels)': 3,
    'Kitchen (Pixels)': 3,
    'Robobin (Pixels)': 3,
}


# ylabels = ['EvalOp/GoalDistance', 'EvalOp/GoalDistance', 'EvalOp/GoalDistance', 'EvalOp/KitchenGoalOverall', 'EvalOp/GoalDistance'] # 'EvalOp/GoalAdaptiveDistance', for cheetah
all_experiments = [ant_experiments, cheetah_experiments, quadruped_experiments, humanoid_experiments, kitchen_experiments, robobin_experiments]
titles = ['Ant (States)', 'HalfCheetah (States)', 'Quadruped (Pixels)', 'Humanoid (Pixels)', 'Kitchen (Pixels)', 'Robobin (Pixels)']

def plot(experiments, titles, yv_labels, ylabel: str = 'Success Rate'):
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))

    legend_labels = []
    legend_handles = []
    ax_idx = 0
    for ax, experiment, title, yv_label in zip(axes.flatten(), experiments, titles, yv_labels):
        xmax = int(1e9)
        ymax = 0
        for label, filepaths in experiment.items():
            if 'Kitchen' in title and label == 'METRA':
                mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, f'EvalOp/KitchenSingleGoalStayingTimeOverall')
            else:
                suffix = ''
                if 'Cheetah' in title and label == 'METRA':
                    suffix = 'Adaptive'
                mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, f'{yv_label}{suffix}')
            handle, = ax.plot(total_env_steps, mean_values, label=label, linewidth=3, color=COLOR_MAP[label], marker=MARKER_MAP[label], markevery=MARKEVERY_MAP[title], markersize=10)
            ax.tick_params(axis='x', labelsize="16")
            ax.tick_params(axis='y', labelsize="16")
            ax.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2, color=COLOR_MAP[label])

            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(handle)

            xmax = min(xmax, total_env_steps[-1])
            ymax = max(ymax, np.max(mean_values + std_dev))

        ax.set_xlim(left=0, right=xmax)
        ax.set_ylim(bottom=0, top=ymax)

        ax.set_xlabel('Env Steps', fontsize="18")
        # if not 'Kitchen' in yv_label:
        #     ax.set_ylabel(ylabel, fontsize="18")
        # else:
        #     # ax.set_ylabel('# Achieved Tasks', fontsize="17")
        #     ax.set_ylabel('Staying Time', fontsize="18")
        if ax_idx % 3 == 0:
            ax.set_ylabel('Staying Time Fraction', fontsize="18")
        ax.set_title(title, fontsize="22", fontweight="bold")
        ax.get_xaxis().get_offset_text().set_fontsize(14)
        # Position the legend outside the plot
        # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax_idx += 1

    fig.legend(legend_handles, legend_labels, frameon=True, bbox_to_anchor=(0.5, 0.0), loc='upper center', fontsize="20", ncols=4)
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'figures/paper/goal_reaching_{yv_labels[0].replace("/", "_")}.pdf', bbox_inches='tight')
    plt.show()

# plot(all_experiments, titles, ['EvalOp/HitSuccess3', 'EvalOp/HitSuccess3', 'EvalOp/HitSuccess3', 'EvalOp/HitSuccess3', 'EvalOp/KitchenGoalStayingTimeOverall', 'EvalOp/RobobinGoalStayingTimeOverall'], ylabel='Success Rate')
plot(all_experiments, titles, ['EvalOp/AtSuccess3', 'EvalOp/AtSuccess3', 'EvalOp/AtSuccess3', 'EvalOp/AtSuccess3', 'EvalOp/KitchenGoalStayingTimeOverall', 'EvalOp/RobobinGoalStayingTimeOverall'], ylabel='Staying Time')
# plot(all_experiments, titles, ['EvalOp/EndSuccess3', 'EvalOp/EndSuccess3', 'EvalOp/EndSuccess3', 'EvalOp/EndSuccess3', 'EvalOp/KitchenGoalStayingTimeOverall', 'EvalOp/RobobinGoalStayingTimeOverall'], ylabel='End Success Rate')

# plot(all_experiments, titles, ['EvalOp/HitSuccess1', 'EvalOp/HitSuccess1', 'EvalOp/HitSuccess1', 'EvalOp/HitSuccess1', 'EvalOp/KitchenGoalStayingTimeOverall', 'EvalOp/RobobinGoalStayingTimeOverall'], ylabel='Success Rate')
# plot(all_experiments, titles, ['EvalOp/AtSuccess1', 'EvalOp/AtSuccess1', 'EvalOp/AtSuccess1', 'EvalOp/AtSuccess1', 'EvalOp/KitchenGoalStayingTimeOverall', 'EvalOp/RobobinGoalStayingTimeOverall'], ylabel='Staying Time')
# plot(all_experiments, titles, ['EvalOp/EndSuccess1', 'EvalOp/EndSuccess1', 'EvalOp/EndSuccess1', 'EvalOp/EndSuccess1', 'EvalOp/KitchenGoalStayingTimeOverall', 'EvalOp/RobobinGoalStayingTimeOverall'], ylabel='End Success Rate')

# EvalOp/KitchenAdaptiveGoalStayingTimeOverall