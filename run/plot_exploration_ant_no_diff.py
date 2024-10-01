import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette("deep")
# plt.style.use("seaborn")
# sns.set_style("whitegrid")

METRA = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55023395.0.1710858248_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55540821.0.1712681476_ant_metra/progress_eval.csv'
]

NO_DIFF_IN_PENALTY = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_penalty/sd000_s_56121194.0.1714674429_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_penalty/sd001_s_56121198.0.1714674467_ant_metra/progress_eval.csv'
]

NO_DIFF_IN_PENALTY_FIXED_LAMBDA_05 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_penalty_fixed_lambda_0.5/sd000_s_56121268.0.1714674731_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_penalty_fixed_lambda_0.5/sd001_s_56121260.0.1714674701_ant_metra/progress_eval.csv'
]

NO_DIFF_IN_PENALTY_FIXED_LAMBDA_05_USE_L2 = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_penalty_fixed_lambda_0.5_use_l2/sd000_s_56121299.0.1714674910_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_penalty_fixed_lambda_0.5_use_l2/sd001_s_56121302.0.1714674941_ant_metra/progress_eval.csv'
]

NO_DIFF_IN_REP = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_rep/sd000_s_56121418.0.1714675988_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_rep/sd001_s_56121419.0.1714676020_ant_metra/progress_eval.csv'
]

NO_DIFF_IN_REP_ONE_MINUS_GAMMA = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_rep_with_one_minus_gamma/sd000_s_56121439.0.1714676263_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_no_diff_in_rep_with_one_minus_gamma/sd001_s_56121446.0.1714676283_ant_metra/progress_eval.csv'
]

YLABELS = [
    'EvalOp/MjNumUniqueCoords'
]

# write me some python that reads the CSV files above and plots the columns

for y_label in YLABELS:
    metra_vals = None
    for csv in METRA:
        df = pd.read_csv(csv)
        if metra_vals is None:
            metra_vals = df[y_label]
        else:
            metra_vals += df[y_label]
    metra_vals /= len(METRA)

    no_diff_in_penalty_vals = None
    for csv in NO_DIFF_IN_PENALTY:
        df = pd.read_csv(csv)
        if no_diff_in_penalty_vals is None:
            no_diff_in_penalty_vals = df[y_label]
        else:
            no_diff_in_penalty_vals += df[y_label]
    no_diff_in_penalty_vals /= len(NO_DIFF_IN_PENALTY)

    no_diff_in_penalty_fixed_lambda_05_vals = None
    for csv in NO_DIFF_IN_PENALTY_FIXED_LAMBDA_05:
        df = pd.read_csv(csv)
        if no_diff_in_penalty_fixed_lambda_05_vals is None:
            no_diff_in_penalty_fixed_lambda_05_vals = df[y_label]
        else:
            no_diff_in_penalty_fixed_lambda_05_vals += df[y_label]
    no_diff_in_penalty_fixed_lambda_05_vals /= len(NO_DIFF_IN_PENALTY_FIXED_LAMBDA_05)

    no_diff_in_penalty_fixed_lambda_05_use_l2_vals = None
    for csv in NO_DIFF_IN_PENALTY_FIXED_LAMBDA_05_USE_L2:
        df = pd.read_csv(csv)
        if no_diff_in_penalty_fixed_lambda_05_use_l2_vals is None:
            no_diff_in_penalty_fixed_lambda_05_use_l2_vals = df[y_label]
        else:
            no_diff_in_penalty_fixed_lambda_05_use_l2_vals += df[y_label]
    no_diff_in_penalty_fixed_lambda_05_use_l2_vals /= len(NO_DIFF_IN_PENALTY_FIXED_LAMBDA_05_USE_L2)

    no_diff_in_rep_vals = None
    for csv in NO_DIFF_IN_REP:
        df = pd.read_csv(csv)
        if no_diff_in_rep_vals is None:
            no_diff_in_rep_vals = df[y_label]
        else:
            no_diff_in_rep_vals += df[y_label]
    no_diff_in_rep_vals /= len(NO_DIFF_IN_REP)

    no_diff_in_rep_one_minus_gamma_vals = None
    for csv in NO_DIFF_IN_REP_ONE_MINUS_GAMMA:
        df = pd.read_csv(csv)
        if no_diff_in_rep_one_minus_gamma_vals is None:
            no_diff_in_rep_one_minus_gamma_vals = df[y_label]
        else:
            no_diff_in_rep_one_minus_gamma_vals += df[y_label]
    no_diff_in_rep_one_minus_gamma_vals /= len(NO_DIFF_IN_REP_ONE_MINUS_GAMMA)

    min_idx = min(len(metra_vals), len(no_diff_in_penalty_vals), len(no_diff_in_penalty_fixed_lambda_05_vals), len(no_diff_in_penalty_fixed_lambda_05_use_l2_vals), len(no_diff_in_rep_vals), len(no_diff_in_rep_one_minus_gamma_vals), len(df['TotalEnvSteps']))
    plt.plot(df['TotalEnvSteps'][:min_idx], metra_vals[:min_idx], label='Metra')
    plt.plot(df['TotalEnvSteps'][:min_idx], no_diff_in_penalty_vals[:min_idx], label='(adaptive) 1 - phi(s)')
    plt.plot(df['TotalEnvSteps'][:min_idx], no_diff_in_penalty_fixed_lambda_05_vals[:min_idx], label='(lam = 0.5) 1 - phi(s)')
    plt.plot(df['TotalEnvSteps'][:min_idx], no_diff_in_penalty_fixed_lambda_05_use_l2_vals[:min_idx], label='(lam = 0.5) phi(s)')
    plt.plot(df['TotalEnvSteps'][:min_idx], no_diff_in_rep_vals[:min_idx], label='phi(s)^T z in rep loss')
    plt.plot(df['TotalEnvSteps'][:min_idx], no_diff_in_rep_one_minus_gamma_vals[:min_idx], label='(1 - gamma) phi(s)^T z in rep loss')

    plt.xlabel('TotalEnvSteps')
    plt.ylabel(y_label)
    plt.legend(frameon=True)
    plt.savefig(f'figures/ant/no_diff_{y_label.split("/")[-1]}.pdf')
    plt.clf()
