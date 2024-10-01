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

ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE_MLP_REP = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2_mlp_rep/sd000_s_21930981.0.1724337141_ant_metra_sf/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2_mlp_rep/sd001_s_21930982.0.1724337141_ant_metra_sf/progress_eval.csv',
]

# ANT_METRA_SUM_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE = [
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd000_s_21924927.0.1724278445_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd001_s_21924928.0.1724278444_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd002_s_22003128.0.1725026725_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd003_s_22003129.0.1725026726_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd004_s_22003130.0.1725026725_ant_metra/progress_eval.csv'
# ]

# ANT_METRA_SUM_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE_MLP_REP = [
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2_mlp_rep/sd000_s_21933731.0.1724364466_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2_mlp_rep/sd001_s_21933732.0.1724364465_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2_mlp_rep/sd002_s_22003163.0.1725032873_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2_mlp_rep/sd003_s_22003164.0.1725032873_ant_metra/progress_eval.csv',
#     'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_energy_no_done_with_goal_metrics_lam_5_option_dim_2_mlp_rep/sd004_s_22003165.0.1725032873_ant_metra/progress_eval.csv'
# ]


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
    # "METRA": ANT_METRA_SUM_NO_DONE,
    # "METRA-C": ANT_METRA_SUM_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    # "METRA-C $f(s, s')^Tz$": ANT_METRA_SUM_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE_MLP_REP,
    "CSF": ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE,
    "CSF $f(s, s')^Tz$": ANT_METRA_SUM_SF_TD_ENERGY_LAM_5_OPTION_DIM_2_NO_DONE_MLP_REP
}

scale_factor = 0.7
COLOR_MAP = {
    # "METRA": sns.color_palette(SNS_PALETTE)[1],
    "CSF": sns.color_palette(SNS_PALETTE)[9],
    "CSF $f(s, s')^Tz$": tuple(c * scale_factor for c in sns.color_palette(SNS_PALETTE)[9]),
    # "METRA-C": tuple(c * 0.8 for c in sns.color_palette(SNS_PALETTE)[1]),
    # "METRA-C $f(s, s')^Tz$": tuple(c * 0.6 for c in sns.color_palette(SNS_PALETTE)[1]),

}

MARKER_MAP = {
    'CSF': 'o',
    "CSF $f(s, s')^Tz$": 'v',
    # 'METRA': 's',
    # 'METRA - IB': 'D',
    # 'METRA-C': 'D',
    # "METRA-C $f(s, s')^Tz$": 'X',
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

plt.legend(legend_handles, legend_labels, frameon=True, fontsize="20")
plt.savefig('figures/paper/parametrization_ablation.pdf', bbox_inches='tight')
plt.show()
