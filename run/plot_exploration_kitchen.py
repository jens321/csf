import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette("deep")
# plt.style.use("seaborn")
# sns.set_style("whitegrid")

CSV_LIST = [
    # metra
    '/anonymous/anonymous/metra-with-avalon/exp/kitchen/sd000_s_55034755.0.1710881351_kitchen_metra/progress_eval.csv',
    # manual normalize
    '/anonymous/anonymous/metra-with-avalon/exp/kitchen_self_normalizing/sd000_s_55035624.0.1710884793_kitchen_metra/progress_eval.csv',
    # l2 penalty
    '/anonymous/anonymous/metra-with-avalon/exp/kitchen_with_l2_penalty/sd000_s_55026827.0.1710860774_kitchen_metra/progress_eval.csv',
    # log sum exp
    '/anonymous/anonymous/metra-with-avalon/exp/kitchen_log_sum_exp/sd000_s_55034733.0.1710881047_kitchen_metra/progress_eval.csv',
    # symmetrize log sum exp
    '/anonymous/anonymous/metra-with-avalon/exp/kitchen_symmetrize_log_sum_exp/sd000_s_55048916.0.1710949008_kitchen_metra/progress_eval.csv',
    # fixed lambda 0.1
    '/anonymous/anonymous/metra-with-avalon/exp/kitchen_fixed_lambda_0.1/progress_eval_kitchen_lambda01.csv',
    # fixed lambda 50
    '/anonymous/anonymous/metra-with-avalon/exp/kitchen_fixed_lambda_50/progress_eval_kitchen_lambda50.csv'
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
    'EvalOp/KitchenOverall'
]

# write me some python that reads the CSV files above and plots the columns

for y_label in YLABELS:
    for csv, label in zip(CSV_LIST, LABELS):
        df = pd.read_csv(csv)
        plt.plot(df['TotalEnvSteps'], df[y_label], label=label)

    plt.xlabel('TotalEnvSteps')
    plt.ylabel(y_label)
    plt.legend(frameon=True)
    plt.savefig(f'figures/kitchen/exploration_{y_label.split("/")[-1]}.pdf')
    plt.clf()
