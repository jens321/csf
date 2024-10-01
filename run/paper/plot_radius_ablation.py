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

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_0p5 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227788.0.1727318257_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_2 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227789.0.1727318254_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_5 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227790.0.1727318256_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_10 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227791.0.1727318254_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_20 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227792.0.1727318257_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_50 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227793.0.1727318254_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_100 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227794.0.1727318257_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_200 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227795.0.1727318256_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_500 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227796.0.1727318256_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_1000 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_1_option_dim_2_add_log_sum_exp_to_reward_sweep_vmf_radius/sd000_s_22227797.0.1727318254_ant_metra/progress_eval.csv'
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
    'METRA': ANT_METRA_SUM_NO_DONE,
    'Radius 0.5': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_0p5,
    'Radius 2': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_2,
    'Radius 5': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_5,
    'Radius 10': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_10,
    'Radius 20': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_20,
    'Radius 50': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_50,
    'Radius 100': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_100,
    'Radius 200': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_200,
    'Radius 500': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_500,
    'Radius 1000': ANT_METRA_SUM_ENERGY_LAM_1_OPTION_DIM_2_NO_DONE_ADD_LOG_SUM_EXP_RADIUS_1000,
}

YLABEL = 'EvalOp/MjNumUniqueCoords'
all_experiments = [ant_experiments]
titles = ['Ant (States)']

# Create a figure with subplots
fig, axes = plt.subplots(1, 1, figsize=(10, 5))

legend_labels = []
legend_handles = []
for ax, experiment, title in zip([axes], all_experiments, titles):
    xmax = int(1e9)
    ymax = 0
    for label, filepaths in experiment.items():
        mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, YLABEL if not 'Kitchen' in title else 'EvalOp/KitchenOverall')
        handle, = ax.plot(total_env_steps, mean_values, label=label, linewidth=3, markersize=10)

        ax.tick_params(axis='x', labelsize="16")
        ax.tick_params(axis='y', labelsize="16")

        ax.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2)

        if label not in legend_labels:
            legend_labels.append(label)
            legend_handles.append(handle)
        
        xmax = min(xmax, total_env_steps[-1])
        ymax = max(ymax, np.max(mean_values + std_dev))

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_xlabel('Env Steps', fontsize="18")
    ax.set_ylabel('State Coverage', fontsize="18")
    ax.set_title(title, fontsize="22", fontweight="bold")
    # Position the legend outside the plot
    # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.get_xaxis().get_offset_text().set_fontsize(14)

plt.legend(legend_handles, legend_labels, frameon=True, fontsize="20", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('figures/paper/radius_ablation.pdf', bbox_inches='tight')
plt.show()
