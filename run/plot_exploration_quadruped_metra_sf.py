import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# sns.set_palette("deep")
# plt.style.use("seaborn")
sns.set_style("whitegrid")

METRA = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped/sd000_s_56953767.0.1718228487_dmc_quadruped_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped/sd001_s_56953768.0.1718228486_dmc_quadruped_metra/progress_eval.csv'
]

METRA_SF_TD = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_metra_sf_td/sd000_s_57003462.0.1718394220_dmc_quadruped_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_metra_sf_td/sd001_s_57003463.0.1718394253_dmc_quadruped_metra/progress_eval.csv'
]

METRA_UTD400 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_utd400/sd000_s_56953869.0.1718229262_dmc_quadruped_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_utd400/sd001_s_56953870.0.1718229263_dmc_quadruped_metra/progress_eval.csv'
]

METRA_SF_TD_UTD400 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_metra_sf_td_utd400/sd000_s_57003478.0.1718394253_dmc_quadruped_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_metra_sf_td_utd400/sd001_s_57003479.0.1718394266_dmc_quadruped_metra/progress_eval.csv'
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
    'Metra': METRA,
    'Metra SF TD': METRA_SF_TD,
    'Metra UTD400': METRA_UTD400,
    'Metra SF TD UTD400': METRA_SF_TD_UTD400
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

    plt.savefig(f'figures/quadruped/metra_sf_{y_label.split("/")[-1]}.pdf', bbox_inches='tight')
    plt.clf()