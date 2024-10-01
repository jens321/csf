import pandas as pd
import matplotlib.pyplot as plt

CSV_LIST = [
    # add_log_sum_exp_to_rewards
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_log_sum_exp_to_rewards/sd000_s_55158804.0.1711388332_ant_metra/progress.csv',
    # add_log_sum_exp_to_rewards * 0.1
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_log_sum_exp_to_rewards/sd000_s_55175131.0.1711422141_ant_metra/progress.csv',
    # add_log_sum_exp_to_rewards * 0.01
    '/anonymous/anonymous/metra-with-avalon/exp/ant_add_log_sum_exp_to_rewards/sd000_s_55175110.0.1711422083_ant_metra/progress.csv',
    # metra
    '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55023395.0.1710858248_ant_metra/progress.csv',
    # log_sum_exp
    '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55159459.0.1711390708_ant_metra/progress.csv'
]

LABELS = [
    'add_log_sum_exp_to_rewards',
    'add_log_sum_exp_to_rewards * 0.1',
    'add_log_sum_exp_to_rewards * 0.01',
    'metra',
    'log_sum_exp'
]

YLABELS = [
    'TrainSp/METRA/LossQf1',
    'TrainSp/METRA/SacpNewActionLogProbMean',
    'TrainSp/METRA/LossQf1',
    'TrainSp/METRA/LossQf2',
    'TrainSp/METRA/LossTe',
    'TrainSp/METRA/PureRewardMean',
    'TrainSp/METRA/QTargetsMean',
    'TrainSp/METRA/QTdErrsMean',
    'TrainSp/METRA/TeObjMean',
    'TrainSp/METRA/TotalGradNormAll',
    'TrainSp/METRA/TotalGradNormTrajEncoder',
    'TrainSp/METRA/TotalGradNormQf1',
    'TrainSp/METRA/TotalGradNormQf2'
]

# write me some python that reads the CSV files above and plots the columns

for y_label in YLABELS:
    for csv, label in zip(CSV_LIST, LABELS):
        df = pd.read_csv(csv)
        plt.plot(df['TotalEnvSteps'], df[y_label], label=label)

    plt.xlabel('TotalEnvSteps')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f'figures/ant/add_log_sum_exp_to_rewards_{y_label.split("/")[-1]}.pdf')
    plt.clf()
