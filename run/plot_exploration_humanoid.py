import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette("deep")
# plt.style.use("seaborn")
# sns.set_style("whitegrid")

CSV_LIST = [
    # metra
    '/anonymous/anonymous/metra-with-avalon/exp/humanoid/progress_eval_humanoid_continuous_og.csv',
    # manual normalize
    '/anonymous/anonymous/metra-with-avalon/exp/humanoid_self_normalizing/sd000_s_55035673.0.1710884996_dmc_humanoid_metra/progress_eval.csv',
    # l2 penalty
    '/anonymous/anonymous/metra-with-avalon/exp/humanoid_l2_penalty/progress_eval_humanoid_l2_penalty.csv',
    # log sum exp
    '/anonymous/anonymous/metra-with-avalon/exp/humanoid_log_sum_exp/sd000_s_55034651.0.1710880692_dmc_humanoid_metra/progress_eval.csv',
    # symmetrize log sum exp
    '/anonymous/anonymous/metra-with-avalon/exp/humanoid_symmetrize_log_sum_exp/sd000_s_55048985.0.1710949185_dmc_humanoid_metra/progress_eval.csv',
    # fixed lambda 0.1
    '/anonymous/anonymous/metra-with-avalon/exp/humanoid_fixed_lambda_0.1/progress_eval_humanoid_lambda01.csv',
    # fixed lambda 50
    '/anonymous/anonymous/metra-with-avalon/exp/humanoid_fixed_lambda_50/progress_eval_humanoid_lambda50.csv'
]

LABELS = [
    'metra',
    'manual normalize',
    'l2 penalty',
    'log sum exp',
    'symmetrize log sum exp',
    'fixed lambda 0.1',
    'fixed lambda 50'
]

YLABELS = [
    'EvalOp/MjNumUniqueCoords'
]

# write me some python that reads the CSV files above and plots the columns

for y_label in YLABELS:
    for csv, label in zip(CSV_LIST, LABELS):
        df = pd.read_csv(csv)
        plt.plot(df['TotalEnvSteps'], df[y_label], label=label)

    plt.xlabel('TotalEnvSteps')
    plt.ylabel(y_label)
    plt.legend(frameon=True)
    plt.savefig(f'figures/humanoid/exploration_{y_label.split("/")[-1]}.pdf')
    plt.clf()
