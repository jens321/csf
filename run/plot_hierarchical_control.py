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
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_chkpt_30k/sd000_s_21377600.0.1721411268_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_chkpt_30k/sd001_s_21377601.0.1721411268_ant_nav_prime_sac/progress_eval.csv',
]

ANT_METRA_SF_TD = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_chkpt_30k/sd000_s_21377629.0.1721411543_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_sf_td_chkpt_30k/sd001_s_21377630.0.1721411543_ant_nav_prime_sac/progress_eval.csv',
]

ANT_METRA_L2_FIXED_LAMBDA_05_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_l2_penalty_with_fixed_lambda_0.5_gaussian_z_no_done_chkpt_30k/sd000_s_21404272.0.1721574705_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_l2_penalty_with_fixed_lambda_0.5_gaussian_z_no_done_chkpt_30k/sd001_s_21404273.0.1721574705_ant_nav_prime_sac/progress_eval.csv',
]

# ********************
# CHEETAH GOAL RESULTS
# ********************

CHEETAH_GOAL_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra/sd000_s_21317544.0.1721129744_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra/sd001_s_21317545.0.1721129744_half_cheetah_goal_ppo/progress_eval.csv',
]

CHEETAH_GOAL_METRA_SF_TD = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td/sd000_s_21317541.0.1721129633_half_cheetah_goal_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_goal_metra_sf_td/sd001_s_21317542.0.1721129633_half_cheetah_goal_ppo/progress_eval.csv',
]

# **********************
# CHEETAH HURDLE RESULTS
# **********************

CHEETAH_HURDLE_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra/sd000_s_21317548.0.1721129987_half_cheetah_hurdle_ppo/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra/sd001_s_21317549.0.1721129988_half_cheetah_hurdle_ppo/progress_eval.csv',
]

CHEETAH_HURDLE_METRA_SF_TD = [
   'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td/sd000_s_21317552.0.1721130115_half_cheetah_hurdle_ppo/progress_eval.csv',
   'anonymous-il-scale/metra-with-avalon/exp/cheetah_hurdle_metra_sf_td/sd001_s_21317553.0.1721130253_half_cheetah_hurdle_ppo/progress_eval.csv',
]

# *****************
# QUADRUPED RESULTS
# *****************

QUADRUPED_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra/sd000_s_21317574.0.1721130992_dmc_quadruped_goal_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra/sd001_s_21317575.0.1721130992_dmc_quadruped_goal_sac/progress_eval.csv',
]

QUADRUPED_METRA_SF_TD = [
   'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td/sd000_s_21317571.0.1721130869_dmc_quadruped_goal_sac/progress_eval.csv',
   'anonymous-il-scale/metra-with-avalon/exp/quadruped_goal_metra_sf_td/sd001_s_21317572.0.1721130870_dmc_quadruped_goal_sac/progress_eval.csv',
]

# ****************
# HUMANOID RESULTS
# ****************

HUMANOID_METRA = [
  
]

HUMANOID_METRA_SF_TD = [
   
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
    'Metra': ANT_METRA,
    'Metra SF TD': ANT_METRA_SF_TD,
    'Metra L2 Penalty (no done)': ANT_METRA_L2_FIXED_LAMBDA_05_NO_DONE,
}

cheetah_goal_experiments = {
    'Metra': CHEETAH_GOAL_METRA,
    'Metra SF TD': CHEETAH_GOAL_METRA_SF_TD,
}

cheetah_hurdle_experiments = {
    'Metra': CHEETAH_HURDLE_METRA,
    'Metra SF TD': CHEETAH_HURDLE_METRA_SF_TD,
}

quadruped_experiments = {
    'Metra': QUADRUPED_METRA,
    'Metra SF TD': QUADRUPED_METRA_SF_TD,
}

humanoid_experiments = {
    'Metra': HUMANOID_METRA,
    'Metra SF TD': HUMANOID_METRA_SF_TD,
}


YLABEL = 'EvalOp/AverageReturn'
all_experiments = [ant_experiments, cheetah_goal_experiments, cheetah_hurdle_experiments, quadruped_experiments, humanoid_experiments]
titles = ['AntMultiGoal', 'HalfCheetahGoal', 'HalfCheetahHurdle', 'QuadrupedGoal']#, 'Quadruped (Pixels)', 'Humanoid (Pixels)', 'Kitchen (Pixels)']

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 3))

for ax, experiment, title in zip(axes, all_experiments, titles):
    for label, filepaths in experiment.items():
        mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, YLABEL if not 'Kitchen' in title else 'EvalOp/KitchenOverall')
        ax.plot(total_env_steps, mean_values, label=label)
        ax.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2)
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Return')
    ax.set_title(title)
    # Position the legend outside the plot
    ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust the plot area to make space for the legend
plt.savefig('figures/paper/hierarchical_control.pdf', bbox_inches='tight')
plt.show()
