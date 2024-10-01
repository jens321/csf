import pandas as pd
import matplotlib.pyplot as plt

CSV_LIST = [
    # metra
    '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55023395.0.1710858248_ant_metra/progress.csv',
    # L2
    '/anonymous/anonymous/metra-with-avalon/exp/ant_l2_penalty/sd000_s_55034747.0.1710881238_ant_metra/progress.csv',
    # 
    '/anonymous/anonymous/metra-with-avalon/exp/ant_self_normalizing/sd000_s_55026775.0.1710860584_ant_metra/progress.csv'
]

LABELS = [
    'metra',
    'l2',
    'self_normalizing'
]

YLABELS = [
    'TrainSp/METRA/LossSacp',
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
    plt.savefig(f'figures/ant/l2_self_normalizing_{y_label.split("/")[-1]}.pdf')
    plt.clf()
