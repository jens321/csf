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
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics/sd000_s_21308664.0.1721099386_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics/sd001_s_21308665.0.1721099386_ant_metra/progress_eval.csv',
]

ANT_METRA_RELABEL_ACTOR_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_relabel_actor_z/sd000_s_21344984.0.1721256908_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_relabel_actor_z/sd001_s_21344985.0.1721256908_ant_metra/progress_eval.csv',
]

ANT_METRA_RELABEL_CRITIC_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_relabel_critic_z/sd000_s_21344986.0.1721256908_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_relabel_critic_z/sd001_s_21344987.0.1721256908_ant_metra/progress_eval.csv',
]

ANT_METRA_RELABEL_ACTOR_AND_CRITIC_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_relabel_actor_and_critic_z/sd000_s_21344988.0.1721256908_ant_metra/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_with_goal_metrics_relabel_actor_and_critic_z/sd001_s_21344989.0.1721256908_ant_metra/progress_eval.csv',
]

ANT_MULTI_GOALS_METRA = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_chkpt_30k/sd000_s_21355208.0.1721310176_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_chkpt_30k/sd001_s_21355209.0.1721310176_ant_nav_prime_sac/progress_eval.csv',
]

ANT_MULTI_GOALS_METRA_RELABEL_ACTOR_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_relabel_actor_z_chkpt_30k/sd000_s_21355210.0.1721310178_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_relabel_actor_z_chkpt_30k/sd001_s_21355211.0.1721310178_ant_nav_prime_sac/progress_eval.csv',
]

ANT_MULTI_GOALS_METRA_RELABEL_CRITIC_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_relabel_critic_z_chkpt_30k/sd000_s_21355212.0.1721310208_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_relabel_critic_z_chkpt_30k/sd001_s_21355213.0.1721310208_ant_nav_prime_sac/progress_eval.csv',
]

ANT_MULTI_GOALS_METRA_RELABEL_ACTOR_AND_CRITIC_Z = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_relabel_actor_and_critic_z_chkpt_30k/sd000_s_21355214.0.1721310208_ant_nav_prime_sac/progress_eval.csv',
    'anonymous-il-scale/metra-with-avalon/exp/ant_multi_goals_metra_relabel_actor_and_critic_z_chkpt_30k/sd001_s_21355215.0.1721310208_ant_nav_prime_sac/progress_eval.csv',
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
    if 'Goal' in y_label:
        truncated_values = -truncated_values
    
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
    'Metra Relabel Actor Z': ANT_METRA_RELABEL_ACTOR_Z,
    'Metra Relabel Critic Z': ANT_METRA_RELABEL_CRITIC_Z,
    'Metra Relabel Actor and Critic Z': ANT_METRA_RELABEL_ACTOR_AND_CRITIC_Z,
}

ant_hierarchical_experiments = {
    'Metra': ANT_MULTI_GOALS_METRA,
    'Metra Relabel Actor Z': ANT_MULTI_GOALS_METRA_RELABEL_ACTOR_Z,
    'Metra Relabel Critic Z': ANT_MULTI_GOALS_METRA_RELABEL_CRITIC_Z,
    'Metra Relabel Actor and Critic Z': ANT_MULTI_GOALS_METRA_RELABEL_ACTOR_AND_CRITIC_Z,
}

ylabels = ['EvalOp/MjNumUniqueCoords', 'EvalOp/GoalDistance', 'EvalOp/AverageReturn']
all_experiments = [ant_experiments, ant_experiments, ant_hierarchical_experiments]
titles = ['Ant (States)', 'Ant (States)', 'Ant (States)']

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 3))

for ax, experiment, title, ylabel in zip(axes, all_experiments, titles, ylabels):
    for label, filepaths in experiment.items():
        mean_values, std_dev, total_env_steps = compute_mean_and_std(filepaths, ylabel)
        ax.plot(total_env_steps, mean_values, label=label)
        ax.fill_between(total_env_steps, mean_values - std_dev, mean_values + std_dev, alpha=0.2)
    ax.set_xlabel('Environment Steps')
    ax_ylabel = None
    if ylabel == 'EvalOp/MjNumUniqueCoords':
        ax_ylabel = 'State Coverage'
    elif ylabel == 'EvalOp/GoalDistance':
        ax_ylabel = 'Negative Goal Distance'
    elif ylabel == 'EvalOp/AverageReturn':
        ax_ylabel = 'Return'
    ax.set_ylabel(ax_ylabel)
    ax.set_title(title)
    # Position the legend outside the plot
    # ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust the plot area to make space for the legend
plt.savefig('figures/paper/relabeling_ablations.pdf', bbox_inches='tight')
plt.show()
