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
ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567111.0.1722950266_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567112.0.1722950266_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd002_s_21989134.0.1724882684_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd003_s_21989135.0.1724882683_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd004_s_21989136.0.1724882683_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645865.0.1723232688_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645866.0.1723232687_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd002_s_22003227.0.1725033389_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd003_s_22003228.0.1725033389_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd004_s_22003229.0.1725033063_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21645869.0.1723232689_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21645870.0.1723232687_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd002_s_22003230.0.1725033243_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd003_s_22003231.0.1725033243_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd004_s_22003232.0.1725033389_ant_metra_sf/progress_eval.csv'
]

ANT_METRA_SUM_NO_DONE_DIM_2 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd000_s_21497671.0.1722112978_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd001_s_21497672.0.1722112978_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd002_s_21737075.0.1723674377_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd003_s_21737076.0.1723674376_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd004_s_21737077.0.1723674376_ant_metra/progress_eval.csv',
]

ANT_METRA_SUM_NO_DONE_DIM_8 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd000_s_21925070.0.1724279787_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd001_s_21925071.0.1724279787_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd002_s_22003242.0.1725033154_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd003_s_22003243.0.1725033228_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd004_s_22003244.0.1725033228_ant_metra/progress_eval.csv'
]

ANT_METRA_SUM_NO_DONE_DIM_32 = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd000_s_21925074.0.1724279785_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd001_s_21925075.0.1724279785_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd002_s_22003245.0.1725033265_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd003_s_22003246.0.1725033324_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics_option_dim_sweep/sd004_s_22003247.0.1725033324_ant_metra/progress_eval.csv'
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
    'CSF (2)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    'CSF (8)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_8_NO_DONE,
    'CSF (32)': ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_32_NO_DONE,
    'METRA (2)': ANT_METRA_SUM_NO_DONE_DIM_2,
    'METRA (8)': ANT_METRA_SUM_NO_DONE_DIM_8,
    'METRA (32)': ANT_METRA_SUM_NO_DONE_DIM_32,
}

COLOR_MAP = {
    'CSF (2)': sns.color_palette(SNS_PALETTE)[9],
    'CSF (8)': tuple(c * 0.8 for c in sns.color_palette(SNS_PALETTE)[9]),
    'CSF (32)': tuple(c * 0.6 for c in sns.color_palette(SNS_PALETTE)[9]),
    'METRA (2)': sns.color_palette(SNS_PALETTE)[1],
    'METRA (8)': tuple(c * 0.8 for c in sns.color_palette(SNS_PALETTE)[1]),
    'METRA (32)': tuple(c * 0.6 for c in sns.color_palette(SNS_PALETTE)[1]),
}

MARKER_MAP = {
    'CSF (2)': 'o',
    'CSF (8)': 'v',
    'CSF (32)': 'p',
    'METRA (2)': 's',
    'METRA (8)': 'D',
    'METRA (32)': 'X',
}

MARKEVERY_MAP = {
    'Ant (States)': 7,
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
    ax.set_ylabel('State Coverage', fontsize="18")
    ax.set_title(title, fontsize="22", fontweight="bold")
    # Position the legend outside the plot
    # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.get_xaxis().get_offset_text().set_fontsize(14)

plt.legend(legend_handles, legend_labels, frameon=True, fontsize="20", loc="lower center", ncol=3)
plt.savefig('figures/paper/skill_dimension_ablation.pdf', bbox_inches='tight')
plt.show()
