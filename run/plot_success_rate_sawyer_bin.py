import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette("deep")
# plt.style.use("seaborn")
# sns.set_style("whitegrid")

UTD_600 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_high_UTD/sd000_s_55973260.0.1713984561_sawyer_bin_crl/progress_eval.csv',
]

UTD_300 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_high_UTD/sd000_s_55973267.0.1713984561_sawyer_bin_crl/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_high_UTD/sd001_s_55976127.0.1713994349_sawyer_bin_crl/progress_eval.csv',
    # seed 2
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_high_UTD/sd002_s_55976149.0.1713994384_sawyer_bin_crl/progress_eval.csv'
]

UTD_150 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_high_UTD/sd000_s_55973276.0.1713984610_sawyer_bin_crl/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_high_UTD/sd001_s_55976098.0.1713994285_sawyer_bin_crl/progress_eval.csv',
    # seed 2
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_high_UTD/sd002_s_55976114.0.1713994315_sawyer_bin_crl/progress_eval.csv'
]

UTD_150_HALF = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_half_random_goals/sd000_s_55974608.0.1713989190_sawyer_bin_crl/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_half_random_goals/sd001_s_55976026.0.1713994055_sawyer_bin_crl/progress_eval.csv',
    # seed 2
    '/anonymous/anonymous/metra-with-avalon/exp/sawyer_bin_crl_info_nce_symmetrized_half_random_goals/sd002_s_55976032.0.1713994081_sawyer_bin_crl/progress_eval.csv'
]


YLABELS = [
    'EvalOp/success_rate'
]

# write me some python that reads the CSV files above and plots the columns

for y_label in YLABELS:
    utd_150_vals = None
    for csv in UTD_150:
        df = pd.read_csv(csv)
        if utd_150_vals is None:
            utd_150_vals = df[y_label]
        else:
            utd_150_vals += df[y_label]
    utd_150_vals /= len(UTD_150)

    utd_300_vals = None
    for csv in UTD_300:
        df = pd.read_csv(csv)
        if utd_300_vals is None:
            utd_300_vals = df[y_label]
        else:
            utd_300_vals += df[y_label]
    utd_300_vals /= len(UTD_300)

    utd_600_vals = None
    for csv in UTD_600:
        df = pd.read_csv(csv)
        if utd_600_vals is None:
            utd_600_vals = df[y_label]
        else:
            utd_600_vals += df[y_label]
    utd_600_vals /= len(UTD_600)

    utd_150_half_vals = None
    for csv in UTD_150_HALF:
        df = pd.read_csv(csv)
        if utd_150_half_vals is None:
            utd_150_half_vals = df[y_label]
        else:
            utd_150_half_vals += df[y_label]
    utd_150_half_vals /= len(UTD_150_HALF)

    
    min_idx = min(len(utd_150_vals), len(utd_300_vals), len(utd_600_vals), len(df['TotalEnvSteps']))
    plt.plot(df['TotalEnvSteps'][:min_idx], utd_150_vals[:min_idx], label='UTD_150')
    plt.plot(df['TotalEnvSteps'][:min_idx], utd_300_vals[:min_idx], label='UTD_300')
    plt.plot(df['TotalEnvSteps'][:min_idx], utd_150_half_vals[:min_idx], label='UTD_150_HALF')
    # plt.plot(df['TotalEnvSteps'][:min_idx], utd_600_vals[:min_idx], label='UTD_600')

    plt.xlabel('TotalEnvSteps')
    plt.ylabel(y_label)
    plt.legend(frameon=True)
    plt.savefig(f'figures/sawyer_bin/UTD_{y_label.split("/")[-1]}.pdf')
    plt.clf()
