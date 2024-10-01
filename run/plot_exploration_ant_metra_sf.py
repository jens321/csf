import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# sns.set_palette("deep")
# plt.style.use("seaborn")
sns.set_style("whitegrid")

METRA = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_utd50/sd000_s_56955647.0.1718292963_ant_metra/progress_eval.csv',
    # seed 1
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_utd50/sd001_s_56955648.0.1718292963_ant_metra/progress_eval.csv'
]

METRA_NORM_BONUS = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_norm_bonus_turn_off_dones/sd000_s_21254688.0.1720721973_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_norm_bonus_turn_off_dones/sd001_s_21254689.0.1720721973_ant_metra/progress_eval.csv'
]

METRA_ANGLE_AND_DIFF_NORM_BONUS = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_diff_norm_bonus_add_angle_bonus/sd000_s_21219392.0.1720536762_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_diff_norm_bonus_add_angle_bonus/sd001_s_21219393.0.1720536762_ant_metra/progress_eval.csv'
]

METRA_ANGLE_BONUS = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_angle_bonus/sd000_s_21219631.0.1720536762_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_angle_bonus/sd001_s_21219632.0.1720536762_ant_metra/progress_eval.csv'
]

METRA_DIFF_NORM_BONUS = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_diff_norm_bonus/sd000_s_21219509.0.1720536762_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_add_diff_norm_bonus/sd001_s_21219510.0.1720536762_ant_metra/progress_eval.csv'
]

METRA_L2_01_01 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.1/sd000_s_21217361.0.1720469781_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.1/sd001_s_21217362.0.1720469781_ant_metra/progress_eval.csv'
]

METRA_L2_02_02 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.2/sd000_s_21217354.0.1720469733_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.2/sd001_s_21217355.0.1720469733_ant_metra/progress_eval.csv'
]

METRA_L2_05_05 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5/sd001_s_21217381.0.1720470023_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5/sd001_s_21217382.0.1720470022_ant_metra/progress_eval.csv'
]

METRA_L2_10_10 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_1.0/sd000_s_21217365.0.1720469853_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_1.0/sd001_s_21217366.0.1720469853_ant_metra/progress_eval.csv'
]

METRA_L2_05_06 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_reward_lam_0.1/sd001_s_21217646.0.1720472063_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_reward_lam_0.1/sd001_s_21217647.0.1720472063_ant_metra/progress_eval.csv'
]

METRA_L2_05_07 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_reward_lam_0.2/sd001_s_21217648.0.1720472063_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_reward_lam_0.2/sd001_s_21217649.0.1720472063_ant_metra/progress_eval.csv'
]

METRA_L2_05_10 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_reward_lam_0.5/sd001_s_21217650.0.1720487682_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_l2_penalty_with_fixed_lambda_0.5_add_penalty_in_rewards_reward_lam_0.5/sd001_s_21217651.0.1720487682_ant_metra/progress_eval.csv'
]

METRA_DIM16 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_dim16/sd000_s_57053790.0.1718635746_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_dim16/sd001_s_57053794.0.1718635746_ant_metra/progress_eval.csv'
]

METRA_DIM64 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_dim64/sd000_s_57053819.0.1718635930_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_dim64/sd001_s_57053820.0.1718635929_ant_metra/progress_eval.csv'    
]

METRA_DIM512 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_dim512/sd000_s_57066107.0.1718666064_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_dim512/sd001_s_57066108.0.1718666064_ant_metra/progress_eval.csv'
]

METRA_UTD150 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_utd150/sd000_s_56953741.0.1718228485_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_utd150/sd001_s_56953740.0.1718228485_ant_metra/progress_eval.csv'
]

METRA_UTD300 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_utd300/sd000_s_56953749.0.1718228485_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_utd300/sd001_s_56953747.0.1718228485_ant_metra/progress_eval.csv'
]

METRA_LOG_SUM_EXP = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55189848.0.1711468761_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55540891.0.1712681572_ant_metra/progress_eval.csv'
]

METRA_SF_TD = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td/sd000_s_56969167.0.1718294685_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td/sd001_s_56969165.0.1718294682_ant_metra_sf/progress_eval.csv'
]

METRA_SF_TD_TURN_OFF_DONES = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_turn_off_dones/sd000_s_56968999.0.1718294159_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_turn_off_dones/sd001_s_56969000.0.1718294159_ant_metra_sf/progress_eval.csv'
]

METRA_SF_TD_UTD150 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_utd150/sd000_s_57003279.0.1718394214_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_utd150/sd001_s_57003280.0.1718394214_ant_metra_sf/progress_eval.csv'
]

METRA_SF_TD_UTD300 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_utd300/sd000_s_57003310.0.1718394220_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_utd300/sd001_s_57003315.0.1718394219_ant_metra_sf/progress_eval.csv'
]

METRA_SF_TD_anonymous = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/progress_eval_0.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/progress_eval_1.csv',
    # seed 2
    '/anonymous/anonymous/metra-with-avalon/progress_eval_2.csv'
]

METRA_SF_MC = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc/sd000_s_56807991.0.1717626135_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc/sd001_s_56807992.0.1717626135_ant_metra_sf/progress_eval.csv'
]

METRA_SF_MC_UTD150 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_utd150/sd000_s_56954001.0.1718229262_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_utd150/sd001_s_56954002.0.1718229262_ant_metra_sf/progress_eval.csv'
]

METRA_SF_MC_INFO_NCE = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_infonce_repr/sd001_s_56807587.0.1717624366_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_infonce_repr/sd002_s_56808002.0.1717626163_ant_metra_sf/progress_eval.csv'
]

METRA_SF_MC_L2 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_l2_penalty_fix_lam_0.5/sd000_s_56808016.0.1717626261_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_l2_penalty_fix_lam_0.5/sd001_s_56808017.0.1717626261_ant_metra_sf/progress_eval.csv'
]

METRA_SF_MC_GEO_ENT = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_geometric_entropy/sd000_s_56820113.0.1717675607_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_geometric_entropy/sd001_s_56820112.0.1717675607_ant_metra_sf/progress_eval.csv'
]

METRA_SF_MC_GEO_ENT_LAST = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_geometric_entropy_mass_on_last_state/sd000_s_56820573.0.1717679415_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_mc_geometric_entropy_mass_on_last_state/sd001_s_56820575.0.1717679415_ant_metra_sf/progress_eval.csv'
]

METRA_SF_TD_GEO_ENT = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_geometric_entropy/sd000_s_56820115.0.1717675684_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_geometric_entropy/sd001_s_56820116.0.1717675684_ant_metra_sf/progress_eval.csv'
]

METRA_SF_TD_GEO_ENT_LAST = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_geometric_entropy_mass_on_last_state/sd000_s_56820581.0.1717679454_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_geometric_entropy_mass_on_last_state/sd001_s_56820583.0.1717679484_ant_metra_sf/progress_eval.csv'
]

METRA_SF_TD_INFO_NCE_B512 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_infonce_repr_B512/sd000_s_56832185.0.1717711735_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_infonce_repr_B512/sd001_s_56832184.0.1717710540_ant_metra_sf/progress_eval.csv'
]

METRA_SF_CONTRASTIVE_vL2 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5/sd000_s_56947752.0.1718210220_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5/sd001_s_56947753.0.1718210220_ant_metra_sf/progress_eval.csv'
]

METRA_SF_CONTRASTIVE_vL2_bonus = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_bonus/sd000_s_56953264.0.1718226105_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_bonus/sd001_s_56953265.0.1718226637_ant_metra_sf/progress_eval.csv'
]

METRA_SF_CONTRASTIVE_vL2_FREEZE_TRAJ = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_freeze_traj/sd000_s_56948883.0.1718211994_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_freeze_traj/sd001_s_56948884.0.1718211994_ant_metra_sf/progress_eval.csv'
]

METRA_SF_CONTRASTIVE_vInner = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner/sd000_s_56948973.0.1718212653_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner/sd001_s_56948974.0.1718212653_ant_metra_sf/progress_eval.csv'
]

METRA_SF_CONTRASTIVE_vInner_bonus = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner_bonus/sd000_s_56953247.0.1718226105_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner_bonus/sd001_s_56953248.0.1718226106_ant_metra_sf/progress_eval.csv'
]

METRA_SF_CONTRASTIVE_vInner_dim16 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner_dim16/sd000_s_56953288.0.1718226637_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner_dim16/sd001_s_56953289.0.1718226636_ant_metra_sf/progress_eval.csv'
]

METRA_SF_CONTRASTIVE_vInner_dim64 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner_dim64/sd000_s_56953302.0.1718226638_ant_metra_sf/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_contrastive_l2_penalty_fix_lam_0.5_inner_dim64/sd001_s_56953301.0.1718226635_ant_metra_sf/progress_eval.csv'
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
experiments = {
    # 'Metra (angle + diff norm bonus)': METRA_ANGLE_AND_DIFF_NORM_BONUS,
    # 'Metra (norm bonus)': METRA_NORM_BONUS,
    # 'Metra (angle bonus)': METRA_ANGLE_BONUS,
    # 'Metra (diff norm bonus)': METRA_DIFF_NORM_BONUS,
    # 'Metra (dim 2)': METRA,
    # 'Metra L2 (0.1)': METRA_L2_01_01,
    # 'Metra L2 (0.2)': METRA_L2_02_02,
    # 'Metra L2 (0.5)': METRA_L2_05_05,
    # 'Metra L2 (1.0)': METRA_L2_10_10,
    # 'Metra L2 (0.5) + Reward (0.1)': METRA_L2_05_06,
    # 'Metra L2 (0.5) + Reward (0.2)': METRA_L2_05_07,
    # 'Metra L2 (0.5) + Reward (0.5)': METRA_L2_05_10,
    # 'Metra (dim 16)': METRA_DIM16,
    # 'Metra (dim 64)': METRA_DIM64,
    # 'Metra (dim 512)': METRA_DIM512,
    # 'Metra UTD150': METRA_UTD150,
    # 'Metra UTD300': METRA_UTD300,
    # 'Metra Log Sum Exp': METRA_LOG_SUM_EXP,
    # 'Metra SF TD': METRA_SF_TD,
    # 'Metra SF TD UTD150': METRA_SF_TD_UTD150,
    # 'Metra SF TD UTD300': METRA_SF_TD_UTD300,
    # 'Metra SF TD (no done)': METRA_SF_TD_TURN_OFF_DONES,
    # 'Metra SF TD InfoNCE': METRA_SF_TD_INFO_NCE,
    # 'Metra SF TD InfoNCE 16': METRA_SF_TD_INFO_NCE_16,
    # 'Metra SF TD L2': METRA_SF_TD_L2,
    # 'Metra SF TD anonymous': METRA_SF_TD_anonymous,
    # 'Metra SF MC': METRA_SF_MC,
    # 'Metra SF MC UTD150': METRA_SF_MC_UTD150,
    # 'Metra SF MC InfoNCE': METRA_SF_MC_INFO_NCE,
    # 'Metra SF MC L2': METRA_SF_MC_L2,
    # 'Metra SF MC Geo Ent': METRA_SF_MC_GEO_ENT,
    # 'Metra SF MC Geo Ent Last': METRA_SF_MC_GEO_ENT_LAST,
    # 'Metra SF TD Geo Ent': METRA_SF_TD_GEO_ENT,
    # 'Metra SF TD Geo Ent Last': METRA_SF_TD_GEO_ENT_LAST,
    # 'Metra SF TD InfoNCE B512': METRA_SF_TD_INFO_NCE_B512,
    # 'Metra SF Contrastive (L2)': METRA_SF_CONTRASTIVE_vL2,
    # 'Metra SF Contrastive (L2) Bonus': METRA_SF_CONTRASTIVE_vL2_bonus,
    # 'Metra SF Contrastive (L2) Freeze': METRA_SF_CONTRASTIVE_vL2_FREEZE_TRAJ,
    # 'Metra SF Contrastive (Inner)': METRA_SF_CONTRASTIVE_vInner,
    # 'Metra SF Contrastive (Inner) Bonus': METRA_SF_CONTRASTIVE_vInner_bonus,
    # 'Metra SF Contrastive (Inner) Dim16': METRA_SF_CONTRASTIVE_vInner_dim16,
    # 'Metra SF Contrastive (Inner) Dim64': METRA_SF_CONTRASTIVE_vInner_dim64
}

YLABELS = ['EvalOp/MjNumUniqueCoords']

# Iterate over each ylabel to make individual plots
for y_label in YLABELS:
    plt.figure(figsize=(16, 6))  # Increase the figure size for a better layout

    for label, filepaths in experiments.items():
        mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, y_label)
        plot_with_confidence_bands(total_env_steps, mean_values, std_dev, label)

    plt.xlabel('Environment Steps')
    plt.ylabel('Unique Coordinates Visited')
    
    # Position the legend outside the plot
    plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the plot area to make space for the legend

    plt.savefig(f'figures/ant/metra_sf_{y_label.split("/")[-1]}.pdf', bbox_inches='tight')
    plt.clf()
