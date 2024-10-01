import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette("deep")
# plt.style.use("seaborn")
# sns.set_style("whitegrid")

CSV_LIST = [
    # metra
    '/anonymous/anonymous/metra-with-avalon/exp/cheetah/progress_eval_half_cheetah_og.csv',
    # manual normalize
    # '/anonymous/anonymous/metra-with-avalon/exp/cheetah_self_normalizing/sd000_s_55035577.0.1710884391_half_cheetah_metra/progress_eval.csv',
    # l2 penalty
    # '/anonymous/anonymous/metra-with-avalon/exp/cheetah_l2_penalty/progress_eval_half_cheetah_no_eps.csv',
    # log sum exp discrete
    # '/anonymous/anonymous/metra-with-avalon/exp/cheetah_log_sum_exp/sd000_s_55034554.0.1710880329_half_cheetah_metra/progress_eval.csv',
    # symmetrize log sum exp
    # '/anonymous/anonymous/metra-with-avalon/exp/cheetah_symmetrize_log_sum_exp/sd000_s_55048893.0.1710948886_half_cheetah_metra/progress_eval.csv',
    # fixed lambda 0.1
    # '/anonymous/anonymous/metra-with-avalon/exp/cheetah_fixed_lambda_0.1/progress_eval_half_cheetah_lambda01.csv',
    # fixed lambda 50
    # '/anonymous/anonymous/metra-with-avalon/exp/cheetah_fixed_lambda_50/progress_eval_half_cheetah_lambda50.csv',
    # log sum exp continuous
    # '/anonymous/anonymous/metra-with-avalon/exp/cheetah_log_sum_exp_continuous/sd000_s_55540936.0.1712681637_half_cheetah_metra/progress_eval.csv'
]

LABELS = [
    'metra',
    # 'manual normalize',
    # 'l2 penalty',
    'log sum exp discrete',
    # 'symmetrize log sum exp',
    # 'fixed lambda 0.1',
    # 'fixed lambda 50',
    'log sum exp continuous'
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
    plt.savefig(f'figures/cheetah/exploration_{y_label.split("/")[-1]}.pdf')
    plt.clf()
