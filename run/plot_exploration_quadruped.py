import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette("deep")
# plt.style.use("seaborn")
# sns.set_style("whitegrid")

CSV_LIST = [
    # metra
    # '/anonymous/anonymous/metra-with-avalon/exp/quadruped/progress_eval_quadruped_og.csv',
    # manual normalize
    # '/anonymous/anonymous/metra-with-avalon/exp/quadruped_self_normalizing/sd000_s_55035602.0.1710884615_dmc_quadruped_metra/progress_eval.csv',
    # l2 penalty
    # '/anonymous/anonymous/metra-with-avalon/exp/quadruped_l2_penalty/progress_eval_quadruped_l2_penalty.csv',
    # log sum exp
    # '/anonymous/anonymous/metra-with-avalon/exp/quadruped_log_sum_exp/sd000_s_55034558.0.1710880381_dmc_quadruped_metra/progress_eval.csv',
    # symmetrize log sum exp
    # '/anonymous/anonymous/metra-with-avalon/exp/quadruped_symmetrize_log_sum_exp/sd000_s_55048952.0.1710949121_dmc_quadruped_metra/progress_eval.csv',
    # fixed lambda 0.1
    # '/anonymous/anonymous/metra-with-avalon/exp/quadruped_fixed_lambda_0.1/progress_eval_quadruped_lambda01.csv',
    # fixed lambda 50
    # '/anonymous/anonymous/metra-with-avalon/exp/quadruped_fixed_lambda_50/progress_eval_quadruped_lambda50.csv',
    # log sum exp batch 256
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_log_sum_exp/sd000_s_55034558.0.1710880381_dmc_quadruped_metra/progress_eval.csv',
    # log sum exp batch 512
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_log_sum_exp/sd000_s_55540906.0.1712681593_dmc_quadruped_metra/progress_eval.csv',
    # log sum exp batch 1024
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_log_sum_exp/sd000_s_55540907.0.1712681593_dmc_quadruped_metra/progress_eval.csv',
    # log sum exp batch 2048
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_log_sum_exp/sd000_s_55540908.0.1712681637_dmc_quadruped_metra/progress_eval.csv',
    # log sum exp batch 4096
    '/anonymous/anonymous/metra-with-avalon/exp/quadruped_log_sum_exp/sd000_s_55540909.0.1712681638_dmc_quadruped_metra/progress_eval.csv'
]

LABELS = [
    # 'metra',
    # 'manual normalize',
    # 'l2 penalty',
    # 'log sum exp',
    # 'symmetrize log sum exp',
    # 'fixed lambda 0.1',
    # 'fixed lambda 50',
    'log sum exp batch 256',
    'log sum exp batch 512',
    'log sum exp batch 1024',
    'log sum exp batch 2048',
    'log sum exp batch 4096'
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
    plt.savefig(f'figures/quadruped/exploration_{y_label.split("/")[-1]}.pdf')
    plt.clf()
