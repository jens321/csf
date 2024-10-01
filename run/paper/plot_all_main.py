import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# sns.set_palette("deep")
# plt.style.use("seaborn")
sns.set_style("whitegrid")
SNS_PALETTE = "colorblind"

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

CHEETAH_CIC_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd000_s_21897379.0.1724172310_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd001_s_21897380.0.1724172310_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd002_s_21983431.0.1724796672_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd003_s_21983432.0.1724796671_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd004_s_21983433.0.1724796671_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd005_s_22094106.0.1726082783_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd006_s_22094107.0.1726082897_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd007_s_22094108.0.1726082897_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd008_s_22094109.0.1726082901_half_cheetah_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cic_no_done/sd009_s_22094110.0.1726082902_half_cheetah_cic/progress_eval.csv'
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

HUMANOID_CIC_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd000_s_21897573.0.1724172888_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd001_s_21897574.0.1724172888_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd002_s_21983454.0.1724796576_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd003_s_21983455.0.1724796575_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd004_s_21983456.0.1724796576_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd005_s_22114653.0.1726609737_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd006_s_22114654.0.1726609737_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd007_s_22114655.0.1726609736_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd008_s_22114656.0.1726609736_dmc_humanoid_cic/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_cic_no_done/sd009_s_22114657.0.1726609737_dmc_humanoid_cic/progress_eval.csv'
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

# *********************
# HUMANOID GOAL RESULTS
# *********************

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

def plot_state_coverage(axes):
    cheetah_experiments = {
        'CSF (ours)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
        'METRA': CHEETAH_METRA_SUM_NO_DONE,
        'DIAYN': CHEETAH_DIAYN_NO_DONE,
        'DADS': CHEETAH_DADS_NO_DONE,
        'CIC': CHEETAH_CIC_NO_DONE,
        'VISR': CHEETAH_VISR,
    }

    humanoid_experiments = {
        'CSF (ours)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
        'METRA': HUMANOID_METRA_SUM_NO_DONE,
        'DIAYN': HUMANOID_DIAYN_NO_DONE,
        'CIC': HUMANOID_CIC_NO_DONE,
        'VISR': HUMANOID_VISR,
    }

    MARKEVERY_MAP = {
        'Ant (States)': 7,
        'HalfCheetah (State Coverage)': 5,
        'Quadruped (Pixels)': 3,
        'Humanoid (State Coverage)': 3,
        'Kitchen (Pixels)': 3,
        'Robobin (Pixels)': 3,
    }

    YLABEL = 'EvalOp/MjNumUniqueCoords'
    all_experiments = [cheetah_experiments, humanoid_experiments]
    titles = ['HalfCheetah (State Coverage)', 'Humanoid (State Coverage)']

    legend_labels = []
    legend_handles = []
    for ax, experiment, title in zip([axes[0, 0], axes[1, 0]], all_experiments, titles):
        xmax = int(1e9)
        ymax = 0
        for label, filepaths in experiment.items():
            mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, YLABEL if not 'Kitchen' in title else 'EvalOp/KitchenOverall')
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
        # if ax_idx % 3 == 0:
        #     ax.set_ylabel('State Coverage', fontsize="18")
        ax.set_ylabel('State Coverage', fontsize="18")
        ax.set_title(title, fontsize="22", fontweight="bold")
        ax.get_xaxis().get_offset_text().set_fontsize(14)

    return legend_handles, legend_labels

def plot_goal_reaching(axes):
    cheetah_experiments = {
        'CSF (ours)': CHEETAH_CONT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
        'METRA': CHEETAH_METRA_SUM_NO_DONE,
        'DIAYN': CHEETAH_DIAYN_NO_DONE,
        'VISR': CHEETAH_VISR,
    }

    humanoid_experiments = {
        'CSF (ours)': HUMANOID_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
        'METRA': HUMANOID_METRA_SUM_NO_DONE,
        'DIAYN': HUMANOID_DIAYN_NO_DONE,
        'VISR': HUMANOID_VISR,
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
        'HalfCheetah (Zero-shot Goal)': 5,
        'Quadruped (Pixels)': 3,
        'Humanoid (Zero-shot Goal)': 3,
        'Kitchen (Pixels)': 3,
        'Robobin (Pixels)': 3,
    }

    experiments = [cheetah_experiments, humanoid_experiments]
    titles = ['HalfCheetah (Zero-shot Goal)', 'Humanoid (Zero-shot Goal)', ]
    yv_labels = ['EvalOp/AtSuccess3', 'EvalOp/AtSuccess3']

    legend_labels = []
    legend_handles = []
    for ax, experiment, title, yv_label in zip([axes[0, 1], axes[1, 1]], experiments, titles, yv_labels):
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
        ax.set_ylabel('Staying Time', fontsize="18")
        ax.set_title(title, fontsize="22", fontweight="bold")
        ax.get_xaxis().get_offset_text().set_fontsize(14)
        # Position the legend outside the plot
        # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_hierarchical_control(axes):
    cheetah_hurdle_experiments = {
        'CSF (ours)': CHEETAH_HURDLE_METRA_SF_TD_ENERGY_LAM_5_DIM_2_NO_DONE_CHKPT_40k,
        'METRA': CHEETAH_HURDLE_METRA_SUM_NO_DONE_CHKPT_40k,
        'DIAYN': CHEETAH_HURDLE_DIAYN_CHKPT_40k,
        'CIC': CHEETAH_HURDLE_CIC_CHKPT_40k,
        'DADS': CHEETAH_HURDLE_DADS_CHKPT_40k,
        'VISR': CHEETAH_HURDLE_VISR_CHKPT_40k,
    }

    humanoid_experiments = {
        'CSF (ours)': HUMANOID_METRA_SF_TD_ENERGY_LAM_5_DIM_8_NO_DONE_CHKPT_3k,
        'METRA': HUMANOID_METRA_SUM_NO_DONE_CHKPT_3k,
        'DIAYN': HUMANOID_DIAYN_CHKPT_3k,
        'CIC': HUMANOID_CIC_CHKPT_3k,
        'VISR': HUMANOID_VISR_CHKPT_3k,
    }

    MARKEVERY_MAP = {
        'AntMultiGoal': 15,
        'HalfCheetahGoal': 15,
        'HalfCheetahHurdle (Hierarchical)': 15,
        'QuadrupedGoal': 15,
        'HumanoidGoal (Hierarchical)': 15,
    }

    YLABEL = 'EvalOp/AverageReturn'
    all_experiments = [cheetah_hurdle_experiments, humanoid_experiments]
    titles = ['HalfCheetahHurdle (Hierarchical)', 'HumanoidGoal (Hierarchical)']

    legend_labels = []
    legend_handles = []
    ax_idx = 0
    for ax, experiment, title in zip([axes[0, 2], axes[1, 2]], all_experiments, titles):
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
        ax.set_ylabel('Return (past 10)', fontsize="18")
        ax.set_title(title, fontsize="22", fontweight="bold")
        ax.get_xaxis().get_offset_text().set_fontsize(14)
        # Position the legend outside the plot
        # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')


def main():
    fig, axes = plt.subplots(2, 3, figsize=(20, 6.5))

    legend_handles, legend_labels = plot_state_coverage(axes)
    plot_goal_reaching(axes)
    plot_hierarchical_control(axes)

    fig.legend(legend_handles, legend_labels, frameon=True, bbox_to_anchor=(0.5, 0.0), loc='upper center', fontsize="20", ncols=6)
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('figures/paper/all_main.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()