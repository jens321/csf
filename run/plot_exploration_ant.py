import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_palette("deep")
# plt.style.use("seaborn")
# sns.set_style("whitegrid")

CSV_LIST = [
    # metra
    '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55023395.0.1710858248_ant_metra/progress_eval.csv',
    # metra sf TD
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td/sd001_s_56805547.0.1717616393_ant_metra_sf/progress_eval.csv',
    # metra sf TD InfoNCE
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_infonce_repr/sd001_s_56806915.0.1717619808_ant_metra_sf/progress_eval.csv',
    # metra sf TD l2 penalty
    '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_sf_td_l2_penalty_fix_lam_0.5/sd001_s_56806369.0.1717618663_ant_metra_sf/progress_eval.csv',
    # anonymous metra sf TD
    '/anonymous/anonymous/metra-with-avalon/progress_eval_0.csv',
    # manual normalize
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_self_normalizing/sd000_s_55189865.0.1711468816_ant_metra/progress_eval.csv',
    # # l2 penalty
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_l2_penalty/sd000_s_55189870.0.1711468816_ant_metra/progress_eval.csv',
    # # log sum exp
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55189848.0.1711468761_ant_metra/progress_eval.csv',
    # # symmetrize log sum exp
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_symmetrize_log_sum_exp/sd000_s_55048871.0.1710948798_ant_metra/progress_eval.csv',
    # # actor contrastive
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_contrastive/vanilla/progress_eval.csv',
    # # actor contrastive + log sum exp
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_contrastive/+log_sum_exp/progress_eval.csv',
    # # actor contrastive + symmetrize log sum exp
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_contrastive/+symmetrize_log_sum_exp/progress_eval.csv',
    # # fixed lambda 0.1
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_fixed_lambda_0.1/progress_eval_ant_lambda_01.csv',
    # # fixed lambda 50
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_fixed_lambda_50/progress_eval_ant_lambda50.csv',
    # actor contrastive every s
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_contrastive_every/obs/progress_eval.csv',
    # actor contrastive every s'
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_contrastive_every/next_obs/progress_eval.csv',
    # actor contrastive every s (1-gamma)
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_contrastive_every/obs_with_(1-gamma)/progress_eval.csv',
    # crl raw goals
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl/sd001_s_55391731.0.1712183757_ant_crl/progress_eval.csv',
    # crl rep goals dim 32
    # '/anonymous/anonymous/metra-with-avalon/exp/CRL-debug-ant/sd000_s_55408588.0.1712241247_ant_crl/progress_eval.csv',
    # metra with lambda = 0
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_lambda_0/sd000_s_55455986.0.1712346884_ant_metra/progress_eval.csv',
    # crl rep goals MSE
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl_use_mse/dim_option_2/progress_eval.csv',
    # crl rep goals dim 2
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl/sd001_s_55409382.0.1712244352_ant_crl/progress_eval.csv',
    # crl rep goals dim 16
    # '/anonymous/anonymous/metra-with-avalon/exp/CRL-debug-ant/sd000_s_55407068.0.1712239544_ant_crl/progress_eval.csv',
    # crl rep goals dim 32
    # '/anonymous/anonymous/metra-with-avalon/exp/CRL-debug-ant/sd000_s_55408588.0.1712241247_ant_crl/progress_eval.csv'
    # crl rep goals batch 256
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl/sd001_s_55409382.0.1712244352_ant_crl/progress_eval.csv',
    # crl rep goals batch 512
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl/sd001_s_55540857.0.1712681507_ant_crl/progress_eval.csv',
    # crl rep goals batch 1024
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl/sd001_s_55540858.0.1712681539_ant_crl/progress_eval.csv',
    # crl rep goals batch 2048
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl/sd001_s_55540859.0.1712681539_ant_crl/progress_eval.csv',
    # crl rep goals batch 4096
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl/sd001_s_55540860.0.1712681538_ant_crl/progress_eval.csv',
    # log sum exp batch 256
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55189848.0.1711468761_ant_metra/progress_eval.csv',
    # log sum exp batch 512
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55540891.0.1712681572_ant_metra/progress_eval.csv',
    # log sum exp batch 1024
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55540892.0.1712681572_ant_metra/progress_eval.csv',
    # log sum exp batch 2048
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55540893.0.1712681572_ant_metra/progress_eval.csv',
    # log sum exp batch 4096
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp/sd000_s_55540894.0.1712681593_ant_metra/progress_eval.csv',
    # metra batch 256
    # '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55023395.0.1710858248_ant_metra/progress_eval.csv',
    # metra batch 512
    # '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55540821.0.1712681476_ant_metra/progress_eval.csv',
    # metra batch 1024
    # '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55540823.0.1712681475_ant_metra/progress_eval.csv',
    # metra batch 2048
    # '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55540830.0.1712681496_ant_metra/progress_eval.csv',
    # metra batch 4096
    # '/anonymous/anonymous/metra-with-avalon/exp/ant/sd000_s_55540831.0.1712681507_ant_metra/progress_eval.csv',
    # crl metra rep 
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl_metra_rep/sd001_s_55539068.0.1712681429_ant_crl/progress_eval.csv',
    # metra with actions
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_metra_include_actions/sd000_s_55540181.0.1712681462_ant_metra/progress_eval.csv',
    # crl layernorm
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl_layernorm/sd001_s_55540745.0.1712681462_ant_crl/progress_eval.csv',
    # metra layernorm
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_layernorm/sd000_s_55540715.0.1712681463_ant_metra/progress_eval.csv',
    # log sum exp layernorm
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_log_sum_exp_layernorm/sd000_s_55540756.0.1712681476_ant_metra/progress_eval.csv',
    # crl standardize output
    # '/anonymous/anonymous/metra-with-avalon/exp/ant_crl_standardize_output/sd001_s_55540478.0.1712681462_ant_crl/progress_eval.csv'
]

LABELS = [
    'Metra',
    'Metra SF TD',
    'Metra SF TD InfoNCE',
    'Metra SF TD L2 Penalty',
    'anonymous Metra SF TD',
    # 'manual normalize',
    # 'l2 penalty',
    # 'log sum exp',
    # 'symmetrize log sum exp',
    # 'actor contrastive',
    # 'actor contrastive + log sum exp',
    # 'actor contrastive + symmetrize log sum exp',
    # 'fixed lambda 0.1',
    # 'fixed lambda 50',
    # 'actor contrastive every s',
    # "actor contrastive every s'",
    # "actor contrastive every s (1-gamma)",
    # 'crl raw goals',
    # 'crl rep goals dim 32',
    # 'metra with lambda = 0',
    # 'crl rep goals MSE',
    # 'crl rep goals dim 2',
    # 'crl rep goals dim 16',
    # 'crl rep goals dim 32',
    # 'crl rep goals batch 256',
    # 'crl rep goals batch 512',
    # 'crl rep goals batch 1024',
    # 'crl rep goals batch 2048',
    # 'crl rep goals batch 4096',
    # 'log sum exp batch 256',
    # 'log sum exp batch 512',
    # 'log sum exp batch 1024',
    # 'log sum exp batch 2048',
    # 'log sum exp batch 4096',
    # 'metra batch 256',
    # 'metra batch 512',
    # 'metra batch 1024',
    # 'metra batch 2048',
    # 'metra batch 4096',
    # 'crl metra rep',
    # 'metra with actions',
    # 'crl layernorm',
    # 'metra layernorm'
    # 'log sum exp layernorm'
    # 'crl standardize output'
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
    plt.savefig(f'figures/ant/exploration_{y_label.split("/")[-1]}.pdf')
    plt.clf()
