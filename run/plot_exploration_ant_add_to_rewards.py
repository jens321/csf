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

ADD_ORIGINAL_PENALTY_TO_REWARDS = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_penalty_to_rewards/sd000_s_56120955.0.1714672558_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_penalty_to_rewards/sd001_s_56121095.0.1714673459_ant_metra/progress_eval.csv'
]

ADD_L2_TO_REWARDS = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_l2_to_rewards/sd000_s_56121098.0.1714673561_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_l2_to_rewards/sd001_s_56121101.0.1714673594_ant_metra/progress_eval.csv'
]

ADD_LOG_SUM_EXP_TO_REWARDS = [
    # seed 0
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_log_sum_exp_to_rewards/sd000_s_55158804.0.1711388332_ant_metra/progress_eval.csv',
    # seed 1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_log_sum_exp_to_rewards/sd000_s_55158813.0.1711388369_ant_metra/progress_eval.csv'
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

    add_original_penalty_to_rewards_vals = None
    for csv in ADD_ORIGINAL_PENALTY_TO_REWARDS:
        df = pd.read_csv(csv)
        if add_original_penalty_to_rewards_vals is None:
            add_original_penalty_to_rewards_vals = df[y_label]
        else:
            add_original_penalty_to_rewards_vals += df[y_label]
    add_original_penalty_to_rewards_vals /= len(ADD_ORIGINAL_PENALTY_TO_REWARDS)

    add_l2_to_rewards_vals = None
    for csv in ADD_L2_TO_REWARDS:
        df = pd.read_csv(csv)
        if add_l2_to_rewards_vals is None:
            add_l2_to_rewards_vals = df[y_label]
        else:
            add_l2_to_rewards_vals += df[y_label]
    add_l2_to_rewards_vals /= len(ADD_L2_TO_REWARDS)

    add_log_sum_exp_to_rewards_vals = None
    for csv in ADD_LOG_SUM_EXP_TO_REWARDS:
        df = pd.read_csv(csv)
        if add_log_sum_exp_to_rewards_vals is None:
            add_log_sum_exp_to_rewards_vals = df[y_label]
        else:
            add_log_sum_exp_to_rewards_vals += df[y_label]
    add_log_sum_exp_to_rewards_vals /= len(ADD_LOG_SUM_EXP_TO_REWARDS)

    min_idx = min(len(metra_vals), len(add_original_penalty_to_rewards_vals), len(add_l2_to_rewards_vals), len(add_log_sum_exp_to_rewards_vals), len(df['TotalEnvSteps']))
    plt.plot(df['TotalEnvSteps'][:min_idx], metra_vals[:min_idx], label='Metra')
    plt.plot(df['TotalEnvSteps'][:min_idx], add_original_penalty_to_rewards_vals[:min_idx], label='add original penalty to rewards')
    plt.plot(df['TotalEnvSteps'][:min_idx], add_l2_to_rewards_vals[:min_idx], label='add l2 penalty to rewards')
    plt.plot(df['TotalEnvSteps'][:min_idx], add_log_sum_exp_to_rewards_vals[:min_idx], label='add log sum exp to rewards')

    plt.xlabel('TotalEnvSteps')
    plt.ylabel(y_label)
    plt.legend(frameon=True)
    plt.savefig(f'figures/ant/add_to_rewards_{y_label.split("/")[-1]}.pdf')
    plt.clf()
