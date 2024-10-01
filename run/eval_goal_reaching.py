import os
import pickle
from collections import defaultdict
import platform
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

sns.set_style("whitegrid")

SNS_PALETTE = "colorblind"

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

ANT_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd000_s_21497671.0.1722112978_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd001_s_21497672.0.1722112978_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd002_s_21737075.0.1723674377_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd003_s_21737076.0.1723674376_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd004_s_21737077.0.1723674376_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd005_s_22093963.0.1726004892_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd006_s_22093964.0.1726004893_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd007_s_22093965.0.1726004892_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd008_s_22093966.0.1726004893_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_turn_off_dones_with_goal_metrics/sd009_s_22093967.0.1726004892_ant_metra'
]

ANT_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21567111.0.1722950266_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21567112.0.1722950266_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd002_s_21989134.0.1724882684_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd003_s_21989135.0.1724882683_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd004_s_21989136.0.1724882683_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd005_s_22094014.0.1726004952_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd006_s_22094015.0.1726082544_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd007_s_22094017.0.1726082543_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd008_s_22094018.0.1726082544_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd009_s_22094019.0.1726082544_ant_metra_sf'
]

ANT_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd000_s_21744886.0.1723744180_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd001_s_21744887.0.1723744179_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd002_s_21983393.0.1724796218_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd003_s_21983394.0.1724796218_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd004_s_21983395.0.1724796218_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd005_s_22093874.0.1726004791_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd006_s_22093875.0.1726004792_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd007_s_22093876.0.1726004791_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd008_s_22093877.0.1726004791_ant_metra',
    'anonymous-il-scale/metra-with-avalon/exp/ant_diayn_no_done/sd009_s_22093878.0.1726004792_ant_metra'
]

ANT_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd000_s_21988746.0.1724872711_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd001_s_21988747.0.1724872711_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd002_s_21989998.0.1724882737_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd003_s_21989999.0.1724882737_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd004_s_21990000.0.1724882737_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd005_s_22094066.0.1726082544_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd006_s_22094067.0.1726082779_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd007_s_22094068.0.1726082778_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd008_s_22094069.0.1726082778_ant_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/ant_visr/sd009_s_22094070.0.1726082783_ant_metra_sf'
]

CHEETAH_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd000_s_21625261.0.1723157129_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd001_s_21625262.0.1723157129_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd002_s_21737085.0.1723674577_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd003_s_21737086.0.1723674591_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd004_s_21737087.0.1723674591_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd005_s_22094134.0.1726091225_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd006_s_22094135.0.1726091277_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd007_s_22094136.0.1726091277_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd008_s_22094137.0.1726091285_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_metra_sum_no_done_with_goal_metrics/sd009_s_22094138.0.1726091285_half_cheetah_metra'
]

CHEETAH_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd000_s_21625253.0.1723156995_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_sweep/sd001_s_21625254.0.1723156993_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd002_s_21989200.0.1724882737_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd003_s_21989201.0.1724882738_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd004_s_21989202.0.1724882737_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd005_s_22094111.0.1726082902_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd006_s_22094112.0.1726083111_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd007_s_22094113.0.1726083111_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd008_s_22094114.0.1726083111_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_2/sd009_s_22094115.0.1726083110_half_cheetah_metra_sf'
]

CHEETAH_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd000_s_21744888.0.1723744179_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd001_s_21744889.0.1723744179_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd002_s_21983412.0.1724796671_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd003_s_21983413.0.1724796671_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd004_s_21983414.0.1724796671_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd005_s_22094121.0.1726083262_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd006_s_22094122.0.1726091225_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd007_s_22094123.0.1726091225_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd008_s_22094124.0.1726091226_half_cheetah_metra',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_diayn_no_done/sd009_s_22094125.0.1726091225_half_cheetah_metra'
]

CHEETAH_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd000_s_21988748.0.1724872821_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd001_s_21988749.0.1724872821_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd002_s_21990001.0.1724882864_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd003_s_21990002.0.1724882863_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd004_s_21990003.0.1724882863_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd005_s_22094128.0.1726091226_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd006_s_22094129.0.1726091226_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd007_s_22094130.0.1726091226_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd008_s_22094131.0.1726091225_half_cheetah_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/cheetah_visr/sd009_s_22094132.0.1726091226_half_cheetah_metra_sf'
]

QUADRUPED_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd000_s_21497823.0.1722113111_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd001_s_21497824.0.1722113111_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd002_s_21737101.0.1723675020_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd003_s_21737102.0.1723675020_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd004_s_21737103.0.1723675020_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd005_s_22114610.0.1726523194_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd006_s_22114611.0.1726523194_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd007_s_22114612.0.1726523194_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd008_s_22114613.0.1726523194_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_with_goal_metrics_no_done/sd009_s_22114614.0.1726523195_dmc_quadruped_metra'
]

QUADRUPED_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd000_s_21595372.0.1723049191_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_sweep/sd001_s_21595373.0.1723049191_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd002_s_21737108.0.1723675162_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd003_s_21737109.0.1723675162_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd004_s_21737110.0.1723675162_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd005_s_22114615.0.1726523313_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd006_s_22114616.0.1726523313_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd007_s_22114617.0.1726523313_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd008_s_22114618.0.1726523313_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_metra_sum_sf_td_energy_with_goal_metrics_no_done_lam_5_option_dim_4/sd009_s_22114619.0.1726523313_dmc_quadruped_metra_sf'
]

QUADRUPED_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd000_s_21744892.0.1723762165_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd001_s_21744893.0.1723762164_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd002_s_21983451.0.1724796576_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd003_s_21983452.0.1724796576_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd004_s_21983453.0.1724796576_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd005_s_22114620.0.1726523319_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd006_s_22114621.0.1726523319_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd007_s_22114622.0.1726523319_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd008_s_22114623.0.1726523319_dmc_quadruped_metra',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_diayn_no_done/sd009_s_22114624.0.1726523319_dmc_quadruped_metra'
]

QUADRUPED_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd000_s_21988761.0.1724883020_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd001_s_21988762.0.1724883020_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd002_s_21990024.0.1724883102_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd003_s_21990025.0.1724883102_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd004_s_21990026.0.1724884360_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd005_s_22114630.0.1726609624_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd006_s_22114631.0.1726609624_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd007_s_22114632.0.1726609624_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd008_s_22114633.0.1726609624_dmc_quadruped_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/quadruped_visr/sd009_s_22114634.0.1726609624_dmc_quadruped_metra_sf'
]

HUMANOID_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd000_s_21500598.0.1722126378_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd001_s_21500599.0.1722126378_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd002_s_21737187.0.1723677005_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd003_s_21737188.0.1723762164_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd004_s_21737189.0.1723762164_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd005_s_22114672.0.1726696073_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd006_s_22114673.0.1726696073_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd007_s_22114674.0.1726696073_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd008_s_22114675.0.1726696073_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_no_done_with_goal_metrics/sd009_s_22114676.0.1726696073_dmc_humanoid_metra'
]

HUMANOID_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd000_s_21646159.0.1723330684_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_sweep/sd001_s_21646160.0.1723330684_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd002_s_21737156.0.1723675597_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd003_s_21737157.0.1723675605_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd004_s_21737158.0.1723675605_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd005_s_22114667.0.1726609793_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd006_s_22114668.0.1726609794_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd007_s_22114669.0.1726609793_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd008_s_22114670.0.1726609794_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_8/sd009_s_22114671.0.1726609794_dmc_humanoid_metra_sf'
]

HUMANOID_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd000_s_21744890.0.1723762165_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd001_s_21744891.0.1723762164_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd002_s_21983457.0.1724796628_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd003_s_21983458.0.1724796635_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd004_s_21983459.0.1724796635_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd005_s_22114658.0.1726609750_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd006_s_22114659.0.1726609750_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd007_s_22114660.0.1726609750_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd008_s_22114661.0.1726609750_dmc_humanoid_metra',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_diayn_no_done/sd009_s_22114662.0.1726609750_dmc_humanoid_metra'
]

HUMANOID_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd000_s_21988763.0.1724883019_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd001_s_21988764.0.1724883019_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd002_s_21990038.0.1724884360_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd003_s_21990039.0.1724884360_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd004_s_21990040.0.1724884360_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd005_s_22114677.0.1726696177_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd006_s_22114678.0.1726696176_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd007_s_22114679.0.1726696176_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd008_s_22114680.0.1726696177_dmc_humanoid_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/humanoid_visr/sd009_s_22114681.0.1726696176_dmc_humanoid_metra_sf'
]

KITCHEN_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd000_s_21983615.0.1724883020_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd001_s_21983616.0.1724883020_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd002_s_21948564.0.1724528083_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd003_s_21948565.0.1724528083_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd004_s_21948566.0.1724528083_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd005_s_22114697.0.1726782634_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd006_s_22114698.0.1726782633_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd007_s_22114699.0.1726782633_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd008_s_22114700.0.1726782634_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_metra_sum_no_done_with_goal_metrics/sd009_s_22114701.0.1726782633_kitchen_metra'
]

KITCHEN_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd000_s_21983604.0.1724883020_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd001_s_21983605.0.1724883019_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd002_s_21948561.0.1724528044_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd003_s_21948562.0.1724528044_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd004_s_21948563.0.1724528044_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd005_s_22114687.0.1726696234_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd006_s_22114688.0.1726696234_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd007_s_22114689.0.1726696234_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd008_s_22114690.0.1726696234_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_cont_metra_sum_sf_td_energy_no_done_with_goal_metrics_lam_5_option_dim_4/sd009_s_22114691.0.1726696234_kitchen_metra_sf'
]

KITCHEN_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd000_s_21948559.0.1724528022_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd001_s_21948560.0.1724528022_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd002_s_21983601.0.1724797950_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd003_s_21983602.0.1724797950_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd004_s_21983603.0.1724797950_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd005_s_22114692.0.1726782515_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd006_s_22114693.0.1726782515_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd007_s_22114694.0.1726782515_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd008_s_22114695.0.1726782515_kitchen_metra',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_diayn_no_done/sd009_s_22114696.0.1726782515_kitchen_metra'
]

KITCHEN_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd000_s_21988765.0.1724883020_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd001_s_21988766.0.1724883072_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd002_s_21990042.0.1724884360_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd003_s_21990043.0.1724884418_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd004_s_21990044.0.1724884418_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd005_s_22114718.0.1726782638_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd006_s_22114719.0.1726782637_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd007_s_22114720.0.1726782637_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd008_s_22114721.0.1726782638_kitchen_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/kitchen_visr/sd009_s_22114722.0.1726782638_kitchen_metra_sf'
]

ROBOBIN_METRA_SUM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd000_s_21956211.0.1724615917_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd001_s_21956212.0.1724615917_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd002_s_21991775.0.1724897402_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd003_s_21991776.0.1724969434_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd004_s_21991777.0.1724969434_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd005_s_22114728.0.1726868940_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd006_s_22114729.0.1726868940_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd007_s_22114730.0.1726868940_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd008_s_22114731.0.1726869053_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_with_goal_metrics_no_done/sd009_s_22114732.0.1726869053_robobin_image_metra'
]

ROBOBIN_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd000_s_21956213.0.1724615934_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd001_s_21956214.0.1724615934_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd002_s_21991778.0.1724969434_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd003_s_21991779.0.1724969434_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd004_s_21991780.0.1724969434_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd005_s_22114733.0.1726869053_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd006_s_22114734.0.1726869053_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd007_s_22114735.0.1726869057_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd008_s_22114736.0.1726869057_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_metra_sum_sf_td_energy_lam_5_dim_9_with_goal_metrics_no_done/sd009_s_22114737.0.1726869057_robobin_image_metra_sf'
]

ROBOBIN_DIAYN_NO_DONE = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd000_s_21956203.0.1724615792_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd001_s_21956204.0.1724615792_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd002_s_21991781.0.1724969435_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd003_s_21991782.0.1724969434_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd004_s_21991783.0.1724969540_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd005_s_22114738.0.1726869057_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd006_s_22114739.0.1726869115_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd007_s_22114740.0.1726869114_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd008_s_22114741.0.1726869115_robobin_image_metra',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_diayn/sd009_s_22114742.0.1726869114_robobin_image_metra'
]

ROBOBIN_VISR = [
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd000_s_21996808.0.1724970801_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd001_s_21996809.0.1724970801_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd002_s_21996810.0.1724970800_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd003_s_21996811.0.1724970826_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd004_s_21996812.0.1724983825_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd005_s_22114723.0.1726782693_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd006_s_22114724.0.1726782693_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd007_s_22114725.0.1726782693_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd008_s_22114726.0.1726782693_robobin_image_metra_sf',
    'anonymous-il-scale/metra-with-avalon/exp/robobin_visr/sd009_s_22114727.0.1726868940_robobin_image_metra_sf'
]

ANT_FOLDER_MAP = {
    # ant
    'CSF (ours)': ANT_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE,
    'METRA': ANT_METRA_SUM_NO_DONE,
    'DIAYN': ANT_DIAYN_NO_DONE,
    'VISR': ANT_VISR,
}

HALF_CHEETAH_FOLDER_MAP = {
    # cheetah
    'CSF (ours)': CHEETAH_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE,
    'METRA': CHEETAH_METRA_SUM_NO_DONE,
    'DIAYN': CHEETAH_DIAYN_NO_DONE,
    'VISR': CHEETAH_VISR,
}

QUADRUPED_FOLDER_MAP = {
    # quadruped
    'CSF (ours)': QUADRUPED_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE,
    'METRA': QUADRUPED_METRA_SUM_NO_DONE,
    'DIAYN': QUADRUPED_DIAYN_NO_DONE,
    'VISR': QUADRUPED_VISR,
}

HUMANOID_FOLDER_MAP = {
    # humanoid
    'CSF (ours)': HUMANOID_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE,
    'METRA': HUMANOID_METRA_SUM_NO_DONE,
    'DIAYN': HUMANOID_DIAYN_NO_DONE,
    'VISR': HUMANOID_VISR,
}

KITCHEN_FOLDER_MAP = {
    # kitchen
    'CSF (ours)': KITCHEN_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE,
    'METRA': KITCHEN_METRA_SUM_NO_DONE,
    'DIAYN': KITCHEN_DIAYN_NO_DONE,
    'VISR': KITCHEN_VISR,
}

ROBOBIN_FOLDER_MAP = {
    # robobin
    'CSF (ours)': ROBOBIN_METRA_SUM_NO_DONE,
    'METRA': ROBOBIN_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE,
    'DIAYN': ROBOBIN_DIAYN_NO_DONE,
    'VISR': ROBOBIN_VISR,
}

ENV_MAP = {
    'Ant (States)': ANT_FOLDER_MAP, 
    'HalfCheetah (States)': HALF_CHEETAH_FOLDER_MAP,
    'Quadruped (Pixels)': QUADRUPED_FOLDER_MAP, 
    'Humanoid (Pixels)': HUMANOID_FOLDER_MAP,
    'Kitchen (Pixels)': KITCHEN_FOLDER_MAP,
    'Robobin (Pixels)': ROBOBIN_FOLDER_MAP,
}

COLOR_MAP = {
    'CSF (ours)': sns.color_palette(SNS_PALETTE)[9],
    'METRA': sns.color_palette(SNS_PALETTE)[1],
    'DIAYN': sns.color_palette(SNS_PALETTE)[2],
    'DADS': sns.color_palette(SNS_PALETTE)[4],
    'CIC': sns.color_palette(SNS_PALETTE)[7],
    'VISR': sns.color_palette(SNS_PALETTE)[0],
}

HATCH_MAP = {
    'CSF (ours)': '*',
    'METRA': '/',
    'DIAYN': '.',
    'VISR': '\\',
}

CHKPT_TO_ENV_STEPS = {
    1000: 1.6e+06,
    2000: 3.2e+06,
    3000: 4.8e+06,
    5000: 8e+06,
    10_000: 1.6e+07,
    20_000: 3.2e+07,
    40_000: 6.4e+07,
}

def make_env(cfg, seed: int):

    if cfg.env_name == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)

    elif cfg.env_name == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100)

    elif cfg.env_name.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        if 'dmc_cheetah' in cfg.env_name:
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif 'dmc_quadruped' in cfg.env_name:
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif 'dmc_humanoid' in cfg.env_name:
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError

    elif cfg.env_name == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        env = MyKitchenEnv(log_per_goal=True)
    
    elif cfg.env_name.startswith('robobin'):
        sys.path.append('lexa')
        from envs.lexa.robobin import MyRoboBinEnv
        if cfg.env_name == 'robobin':
            env = MyRoboBinEnv(log_per_goal=True)
        elif cfg.env_name == 'robobin_image':
            env = MyRoboBinEnv(obs_type='image', log_per_goal=True)

    else:
        raise NotImplementedError

    if cfg.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, cfg.frame_stack)

    normalizer_type = 'preset' if cfg.env_name == 'ant' or cfg.env_name == 'half_cheetah' else 'off'
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)

    elif normalizer_type == 'preset':
        normalizer_name = cfg.env_name
        additional_dim = 0
        if cfg.env_name in ['ant_nav_prime']:
            normalizer_name = 'ant'
            additional_dim = cp_num_truncate_obs
        elif cfg.env_name in ['half_cheetah_goal', 'half_cheetah_hurdle']:
            normalizer_name = 'half_cheetah'
            additional_dim = cp_num_truncate_obs
        else:
            normalizer_name = cfg.env_name
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        if additional_dim > 0:
            normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
            normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    elif normalizer_type == 'craftax':
        env = consistent_normalize(env, normalize_obs=False, flatten_obs=False, **normalizer_kwargs)

    return env

def load_traj_encoder_and_policy(load_dir: str, chkpt: int):
    traj_encoder = load_model(os.path.join(load_dir, f'traj_encoder{chkpt}.pt'))
    option_policy = load_model(ANT_METRA_SUM_SF_TD_ENERGY_BEST_LAM_OPTION_DIM_NO_DONE[0], f'option_policy{chkpt}.pt')

    return traj_encoder, option_policy

def get_chkpts(env_name: str):
    if env_name == 'ant':
        return [40_000]
    elif env_name == 'half_cheetah':
        return [40_000]
    elif env_name == 'dmc_quadruped':
        return [3000]
    elif env_name == 'dmc_humanoid':
        return [3000]
    elif env_name == 'kitchen':
        return [3000]
    elif env_name == 'robobin_image':
        return [3000]
    else:
        raise NotImplementedError

def collect_goals(cfg, env, num_goals: int = 40):
    goals = []  # list of (goal_obs, goal_info)

    if cfg.env_name == 'kitchen':
        goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']
        for i in range(num_goals):
            goal_idx = np.random.randint(len(goal_names))
            goal_name = goal_names[goal_idx]
            goal_obs = env.render_goal(goal_idx=goal_idx).copy().astype(np.float32)
            goal_obs = np.tile(goal_obs, 3).flatten() # self.frame_stack = 3
            goals.append((goal_obs, {'goal_idx': goal_idx, 'goal_name': goal_name}))

    elif cfg.env_name == 'robobin_image':
        goal_names = ['ReachLeft', 'ReachRight', 'PushFront', 'PushBack']
        for i in range(num_goals):
            goal_idx = np.random.randint(len(goal_names))
            goal_name = goal_names[goal_idx]
            goal_obs = env.render_goal(goal_idx=goal_idx).copy().astype(np.float32)
            goal_obs = np.tile(goal_obs, 3).flatten() # self.frame_stack = 3
            goals.append((goal_obs, {'goal_idx': goal_idx, 'goal_name': goal_name}))

    elif cfg.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid']:
        for i in range(num_goals):
            env.reset()
            state = env.physics.get_state().copy()
            if cfg.env_name == 'dmc_cheetah':
                goal_loc = (np.random.rand(1) * 2 - 1) * cfg.goal_range
                state[:1] = goal_loc
            else:
                goal_loc = (np.random.rand(2) * 2 - 1) * cfg.goal_range
                state[:2] = goal_loc
            env.physics.set_state(state)
            if cfg.env_name == 'dmc_humanoid':
                for _ in range(50):
                    env.step(np.zeros_like(env.action_space.sample()))
            else:
                env.step(np.zeros_like(env.action_space.sample()))
            goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
            goal_obs = np.tile(goal_obs, cfg.frame_stack or 1).flatten()
            goals.append((goal_obs, {'goal_loc': goal_loc}))

    elif cfg.env_name in ['ant', 'ant_pixel', 'half_cheetah']:
        for i in range(num_goals):
            env.reset()
            state = env.unwrapped._get_obs().copy()
            if cfg.env_name in ['half_cheetah']:
                goal_loc = (np.random.rand(1) * 2 - 1) * cfg.goal_range
                state[:1] = goal_loc
                env.set_state(state[:9], state[9:])
            else:
                goal_loc = (np.random.rand(2) * 2 - 1) * cfg.goal_range
                state[:2] = goal_loc
                env.set_state(state[:15], state[15:])
            for _ in range(5):
                env.step(np.zeros_like(env.action_space.sample()))
            if cfg.env_name == 'ant_pixel':
                goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
                goal_obs = np.tile(goal_obs, cfg.frame_stack or 1).flatten()
            else:
                goal_obs = env._apply_normalize_obs(state).astype(np.float32)
            goals.append((goal_obs, {'goal_loc': goal_loc}))

    return goals

def eval_goals(goals, env, traj_encoder, option_policy, cfg, max_path_length: int):
    goal_metrics = defaultdict(list)

    for method in ['Single', 'Adaptive'] if (cfg.discrete and cfg.inner) else ['']:
        for goal_obs, goal_info in goals:
            obs = env.reset()
            step = 0
            done = False
            success = 0
            staying_time = 0

            hit_success_3 = 0
            end_success_3 = 0
            at_success_3 = 0

            hit_success_1 = 0
            end_success_1 = 0
            at_success_1 = 0

            option = None
            while step < max_path_length and not done:
                if cfg.inner:
                    if cfg.no_diff_in_rep: 
                        te_input = torch.from_numpy(goal_obs[None, ...]).to(cfg.device)
                        phi = traj_encoder(te_input).mean[0]

                        if cfg.self_normalizing:
                            phi = phi / phi.norm(dim=-1, keepdim=True)

                        phi = phi.detach().cpu().numpy()
                        if cfg.discrete:
                            option = np.eye(cfg.dim_option)[phi.argmax()]
                        else:
                            option = phi
                    else:
                        te_input = torch.from_numpy(np.stack([obs, goal_obs])).to(cfg.device)
                        phi_s, phi_g = traj_encoder(te_input).mean
                        phi_s, phi_g = phi_s.detach().cpu().numpy(), phi_g.detach().cpu().numpy()
                        if cfg.discrete:
                            if method == 'Adaptive':
                                option = np.eye(cfg.dim_option)[(phi_g - phi_s).argmax()]
                            else:
                                if option is None:
                                    option = np.eye(cfg.dim_option)[(phi_g - phi_s).argmax()]
                        else:
                            option = (phi_g - phi_s) / np.linalg.norm(phi_g - phi_s)
                else:
                    te_input = torch.from_numpy(goal_obs[None, ...]).to(cfg.device)
                    phi = traj_encoder(te_input).mean[0]
                    phi = phi.detach().cpu().numpy()
                    if cfg.discrete:
                        option = np.eye(cfg.dim_option)[phi.argmax()]
                    else:
                        option = phi
                action, agent_info = option_policy.get_action(np.concatenate([obs, option]))
                next_obs, _, done, info = env.step(action)
                obs = next_obs

                if cfg.env_name == 'kitchen':
                    _success = env.compute_success(goal_info['goal_idx'])[0]
                    success = max(success, _success)
                    staying_time += _success

                if cfg.env_name == 'robobin_image':
                    success = max(success, info['success'])
                    staying_time += info['success']

                if cfg.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
                    if cfg.env_name in ['dmc_cheetah']:
                        cur_loc = env.physics.get_state()[:1]
                    elif cfg.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                        cur_loc = env.physics.get_state()[:2]
                    elif cfg.env_name in ['half_cheetah']:
                        cur_loc = env.unwrapped._get_obs()[:1]
                    else:
                        cur_loc = env.unwrapped._get_obs()[:2] 

                    if np.linalg.norm(cur_loc - goal_info['goal_loc']) < 3:
                        hit_success_3 = 1.
                        at_success_3 += 1.

                    if np.linalg.norm(cur_loc - goal_info['goal_loc']) < 1:
                        hit_success_1 = 1.
                        at_success_1 += 1.

                step += 1

            if cfg.env_name == 'kitchen':
                goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']
                goal_metrics[f'Kitchen{method}Goal{goal_info["goal_name"]}'].append(success)
                goal_metrics[f'Kitchen{method}GoalOverall'].append(success * len(goal_names))
                goal_metrics[f'Kitchen{method}GoalStayingTime{goal_info["goal_name"]}'].append(staying_time)
                goal_metrics[f'Kitchen{method}GoalStayingTimeOverall'].append(staying_time)

            if cfg.env_name == 'robobin_image':
                goal_names = ['ReachLeft', 'ReachRight', 'PushFront', 'PushBack']
                goal_metrics[f'Robobin{method}Goal{goal_info["goal_name"]}'].append(success)
                goal_metrics[f'Robobin{method}GoalOverall'].append(success * len(goal_names))
                goal_metrics[f'Robobin{method}GoalStayingTime{goal_info["goal_name"]}'].append(staying_time)
                goal_metrics[f'Robobin{method}GoalStayingTimeOverall'].append(staying_time)

            elif cfg.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
                if cfg.env_name in ['dmc_cheetah']:
                    cur_loc = env.physics.get_state()[:1]
                elif cfg.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                    cur_loc = env.physics.get_state()[:2]
                elif cfg.env_name in ['half_cheetah']:
                    cur_loc = env.unwrapped._get_obs()[:1]
                else:
                    cur_loc = env.unwrapped._get_obs()[:2]
                distance = np.linalg.norm(cur_loc - goal_info['goal_loc'])
                squared_distance = distance ** 2
                if distance < 3:
                    end_success_3 = 1.
                if distance < 1:
                    end_success_1 = 1.
            
                goal_metrics[f'HitSuccess3{method}'].append(hit_success_3)
                goal_metrics[f'EndSuccess3{method}'].append(end_success_3)
                goal_metrics[f'AtSuccess3{method}'].append(at_success_3)

                goal_metrics[f'HitSuccess1{method}'].append(hit_success_1)
                goal_metrics[f'EndSuccess1{method}'].append(end_success_1)
                goal_metrics[f'AtSuccess1{method}'].append(at_success_1)

                goal_metrics[f'Goal{method}Distance'].append(distance)
                goal_metrics[f'Goal{method}SquaredDistance'].append(squared_distance)

    goal_metrics = {key: np.mean(value) for key, value in goal_metrics.items()}

    return goal_metrics

def main():
    print('ENV MAP KEYS', ENV_MAP.keys())
    NUM_GOALS = 200
    MAX_PATH_LENGTH = None
    EVAL_METRIC = 'AtSuccess3'
    PLOT_ONLY = True
    BAR_WIDTH = 0.2

    if not PLOT_ONLY:
        for ENV_FOLDER_MAPS in ENV_MAP.values():
            for method_folders in ENV_FOLDER_MAPS.values():
                for seed_folder in method_folders:
                    with open(os.path.join(seed_folder, 'params.pkl'), 'rb') as f:
                        data = pickle.load(f)

                    cfg = data['algo']
                    seed = data['setup_args'].seed

                    # check if at least some goal metrics have already been computed
                    mpl = MAX_PATH_LENGTH or cfg.max_path_length
                    goal_metrics_file = os.path.join(seed_folder, f'goal_metrics_num_goals_{NUM_GOALS}_mpl_{mpl}.pkl')
                    if os.path.exists(goal_metrics_file):
                        print('Found goal metrics file: ', goal_metrics_file)
                        with open(goal_metrics_file, 'rb') as f:
                            chkpt_to_goal_metrics = pickle.load(f)
                    else:
                        chkpt_to_goal_metrics = {}

                    env = make_env(cfg, seed=seed)
                    chkpts = get_chkpts(cfg.env_name)

                    for chkpt in chkpts:
                        # check if this checkpoint has already been evaluated
                        if chkpt in chkpt_to_goal_metrics:
                            continue

                        # load trajectory encoder
                        traj_data = torch.load(os.path.join(seed_folder, f'traj_encoder{chkpt}.pt'))
                        traj_encoder = traj_data['traj_encoder']
                        traj_encoder.eval()

                        # load option policy
                        option_data = torch.load(os.path.join(seed_folder, f'option_policy{chkpt}.pt'))
                        option_policy = option_data['policy']
                        option_policy.eval()

                        # collect goals
                        goals = collect_goals(cfg, env, num_goals=NUM_GOALS)
                        eval_goal_metrics = eval_goals(goals, env, traj_encoder, option_policy, cfg, mpl)
                        
                        chkpt_to_goal_metrics[chkpt] = eval_goal_metrics

                        # save
                        print(f'Saving goal metrics: {goal_metrics_file}')
                        with open(goal_metrics_file, 'wb') as f:
                            pickle.dump(chkpt_to_goal_metrics, f)

    # Create a figure with subplots
    fig, ax = plt.subplots(figsize=(20, 4))
    visited_labels = set()

    for env_idx, (title, ENV_FOLDER_MAPS) in enumerate(ENV_MAP.items()):
        bar_xs = []
        bar_ys = []
        bar_stds = []

        for method_idx, (label, method_folders) in enumerate(ENV_FOLDER_MAPS.items()):

            all_seed_ys = []
            min_length = float('inf')

            for seed_folder in method_folders:
                # with open(os.path.join(seed_folder, 'params.pkl'), 'rb') as f:
                #     data = pickle.load(f)

                # cfg = data['algo']
                # mpl = MAX_PATH_LENGTH or cfg.max_path_length

                if MAX_PATH_LENGTH is None:
                    if 'Kitchen' in title:
                        mpl = 50
                    else:
                        mpl = 200
                else:
                    with open(os.path.join(seed_folder, 'params.pkl'), 'rb') as f:
                        data = pickle.load(f)

                    cfg = data['algo']
                    mpl = MAX_PATH_LENGTH or cfg.max_path_length

                # load results
                goal_metrics_file = os.path.join(seed_folder, f'goal_metrics_num_goals_{NUM_GOALS}_mpl_{mpl}.pkl')
                if os.path.exists(goal_metrics_file):
                    with open(goal_metrics_file, 'rb') as f:
                        chkpt_to_goal_metrics = pickle.load(f)

                # plot
                xs = list(chkpt_to_goal_metrics.keys())
                if 'Kitchen' in list(chkpt_to_goal_metrics[xs[0]].keys())[0]:
                    if label == 'METRA':
                        ys = [chkpt_to_goal_metrics[x]['KitchenSingleGoalStayingTimeOverall'] for x in xs]
                    else:
                        ys = [chkpt_to_goal_metrics[x]['KitchenGoalStayingTimeOverall'] for x in xs]
                elif 'Robobin' in list(chkpt_to_goal_metrics[xs[0]].keys())[0]:
                    ys = [chkpt_to_goal_metrics[x]['RobobinGoalStayingTimeOverall'] for x in xs]
                elif 'cheetah' in goal_metrics_file and label == 'METRA':
                    ys = [chkpt_to_goal_metrics[x][f'{EVAL_METRIC}Adaptive'] for x in xs]
                else:
                    ys = [chkpt_to_goal_metrics[x][f'{EVAL_METRIC}'] for x in xs]
                all_seed_ys.append(ys)

                if len(xs) < min_length:
                    min_length = len(xs)

            truncated_values = [values[:min_length] for values in all_seed_ys]
            truncated_values = np.array(truncated_values)

            # normalize
            truncated_values /= 50 if 'Kitchen' in title else 200

            mean_values = np.mean(truncated_values, axis=0)
            std_dev = np.std(truncated_values, axis=0, ddof=1)
            t_ci = stats.sem(truncated_values, axis=0) * stats.t.isf(0.05 / 2, len(truncated_values) - 1)
            b_ci = stats.bootstrap((truncated_values.flatten(),), np.mean, confidence_level=0.95).confidence_interval.high - mean_values
            std_dev = t_ci

            xs = [CHKPT_TO_ENV_STEPS[x] for x in xs]

            if len(xs) > 1:
                ax.plot(xs, mean_values, label=label)
                ax.fill_between(xs, mean_values - std_dev, mean_values + std_dev, alpha=0.2)
                ax.set_xlabel('Environment Steps')
                ax.set_ylabel(EVAL_METRIC)
                ax.set_title(title)
            else:
                if label not in visited_labels:
                    bar = plt.bar(env_idx + method_idx * BAR_WIDTH, mean_values, BAR_WIDTH, color=COLOR_MAP[label], label=label, yerr=std_dev, capsize=5)[0]
                    visited_labels.add(label)
                else:
                    bar = plt.bar(env_idx + method_idx * BAR_WIDTH, mean_values, BAR_WIDTH, color=COLOR_MAP[label], yerr=std_dev, capsize=5)[0]
                bar.set_hatch(HATCH_MAP[label])
                # bar_xs.append(label)
                # bar_ys.append(mean_values[0])
                # bar_stds.append(std_dev[0])

        # if len(xs) == 1:
        #     ax.bar(bar_xs, bar_ys, color=[COLOR_MAP[_label] for _label in bar_xs], yerr=bar_stds, capsize=5)
        #     ax.set_xlabel('Method')
        #     ax.set_ylabel('Staying Time')
        #     ax.set_title(title, fontsize="20", fontweight="bold")

    if not len(xs) == 1:
        plt.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust the plot area to make space for the legend
    else:
        plt.ylim(bottom=0, top=0.35)
        plt.ylabel('Staying Time Fraction', fontsize="18")
        plt.xticks(np.arange(len(ENV_MAP.keys())) + BAR_WIDTH, list(ENV_MAP.keys()), fontsize="16", fontweight="bold")
        plt.yticks(fontsize="16")
        plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', fontsize="20", ncols=4)
        plt.tight_layout(rect=[0, 0, 1.0, 1])
    plt.savefig(f'figures/paper/post_goal_reaching_{mpl}_{EVAL_METRIC}.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()