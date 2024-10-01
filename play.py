import functools

import torch
import pickle as pkl
import numpy as np
import torch.multiprocessing as mp

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

from iod.utils import FigManager

from envs.mujoco.ant_env import AntEnv
from envs.mujoco.ant_with_goals_env import AntWithGoalsEnv

# load trajectory encoder
traj_data = torch.load('/anonymous/anonymous/metra-with-avalon/exp/ant_with_goals_crl_info_nce_symmetrized_mse/sd001_s_55764633.0.1713397338_ant_with_goals_crl/traj_encoder9000.pt')
traj_encoder = traj_data['traj_encoder']
traj_encoder.to('cpu')

# load option policy
option_data = torch.load('/anonymous/anonymous/metra-with-avalon/exp/ant_with_goals_crl_info_nce_symmetrized_mse/sd001_s_55764633.0.1713397338_ant_with_goals_crl/option_policy9000.pt')
option_policy = option_data['policy']
option_policy.to('cpu')

# load algo 
with open('/anonymous/anonymous/metra-with-avalon/exp/ant_with_goals_crl_info_nce_symmetrized_mse/sd001_s_55764633.0.1713397338_ant_with_goals_crl/itr_9000.pkl', 'rb') as f:
    algo = pkl.load(f)['algo']

# load phi encoder
phi_encoder = torch.load('/anonymous/anonymous/metra-with-avalon/exp/ant_with_goals_crl_info_nce_symmetrized_mse/sd001_s_55764633.0.1713397338_ant_with_goals_crl/phi_encoder9000.pt')
phi_encoder = phi_encoder['phi_encoder']
phi_encoder.to('cpu')

algo.device = 'cpu'

phi_encoder.eval()
traj_encoder.eval()
option_policy.eval()

env = AntEnv(render_hw=100)
goal_obs = env.reset()
def make_env():
    from iod.utils import get_normalizer_preset
    from garagei.envs.consistent_normalized_env import consistent_normalize
    # from envs.mujoco.ant_env import AntEnv
    env = AntWithGoalsEnv(render_hw=100, goal_obs=goal_obs, goal_limit=5, fixed_goal=np.array([0., 0.]))

    normalizer_name = 'ant_with_goals'
    normalizer_type = 'preset'
    normalizer_kwargs = {}
    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    else:
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    return env

def _get_trajectories(runner,
                        sampler_key,
                        batch_size=None,
                        extras=None,
                        update_stats=False,
                        worker_update=None,
                        env_update=None):
    if batch_size is None:
        batch_size = len(extras)
    policy_sampler_key = sampler_key[6:] if sampler_key.startswith('local_') else sampler_key
    time_get_trajectories = [0.0]

    trajectories, infos = runner.obtain_exact_trajectories(
        runner.step_itr,
        sampler_key=sampler_key,
        batch_size=batch_size,
        agent_update=_get_policy_param_values({'option_policy':option_policy}, policy_sampler_key),
        env_update=env_update,
        worker_update=worker_update,
        extras=extras,
        update_stats=update_stats,
    )
    print(f'_get_trajectories({sampler_key}) {time_get_trajectories[0]}s')

    for traj in trajectories:
        for key in ['ori_obs', 'next_ori_obs', 'coordinates', 'next_coordinates']:
            if key not in traj['env_infos']:
                continue

    return trajectories

def _get_policy_param_values(policy, key):
    param_dict = policy[key].get_param_values()
    for k in param_dict.keys():
        param_dict[k] = param_dict[k].detach().cpu()
    return param_dict

def _generate_option_extras(options):
    return [{'option': option} for option in options]

def main():
    from iod.utils import get_option_colors
    import dowel_wrapper
    assert dowel_wrapper is not None
    import dowel
    from garagei.experiment.option_local_runner import OptionLocalRunner
    from garaged.src.garage.experiment.experiment import ExperimentContext
    from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler

    NUM_RANDOM_TRAJECTORIES = 10

    random_options = np.random.randn(NUM_RANDOM_TRAJECTORIES, 2)
    # random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
    random_option_colors = get_option_colors(random_options * 4)

    runner = OptionLocalRunner(ExperimentContext(
        snapshot_dir='.',
        snapshot_mode='last',
        snapshot_gap=1,
    ))

    env = make_env()
    contextualized_make_env = make_env

    runner.setup(
        algo=algo,
        env=env,
        make_env=contextualized_make_env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=1),
        n_workers=1,
    )

    random_trajectories = _get_trajectories(
        runner,
        sampler_key='option_policy',
        # extras=_generate_option_extras(random_options),
        extras=None,
        batch_size=NUM_RANDOM_TRAJECTORIES,
        worker_update=dict(
            _render=False,
            _deterministic_policy=True,
        ),
        env_update=dict(_action_noise_std=None),
    )

    # last_obs = random_trajectories[0]['next_observations'][-1]
    # last_obs_rep = traj_encoder(torch.from_numpy(last_obs)).mean.detach().cpu().numpy()
    # random_trajectories = random_trajectories[:1]

    with FigManager(runner, 'random_trajectory') as fm:
        runner._env.render_trajectories(
            random_trajectories, random_option_colors, [-10, 10, -10, 10], fm.ax
        )
    
    # with FigManager(runner, 'random_trajectory') as fm:
    #     runner._env.render_trajectories(
    #         random_trajectories, random_option_colors, [-3, 3, -3, 3], fm.ax
    #     )

    # goal_reaching_trajectories = _get_trajectories(
    #     runner,
    #     sampler_key='option_policy',
    #     extras=_generate_option_extras([last_obs_rep] * 3),
    #     worker_update=dict(
    #         _render=False,
    #         _deterministic_policy=True,
    #     ),
    #     env_update=dict(_action_noise_std=None),
    # )

    # with FigManager(runner, 'all_trajectory') as fm:
    #     runner._env.render_trajectories(
    #         goal_reaching_trajectories, random_option_colors, [-10, 10, -10, 10], fm.ax
    #     )

    # actions = torch.from_numpy(random_trajectories[0]['actions'])
    # obs = torch.from_numpy(random_trajectories[0]['observations'])
    # s_a_rep = phi_encoder(obs, actions)
    # values = torch.einsum('ij,ij->i', s_a_rep, torch.from_numpy(last_obs_rep).unsqueeze(0).repeat((200, 1)))
    # print('USING OWN FINAL OBS')
    # print(f'Return: {(values[:-1] - values[1:]).sum()}')
    # print()

    # r_t_pairs = []
    # for i in range(100):
    #     # last_obs = goal_reaching_trajectories[i]['next_observations'][-1]
    #     # last_obs_rep = traj_encoder(torch.from_numpy(last_obs)).mean.detach().cpu().numpy()
    #     actions = torch.from_numpy(goal_reaching_trajectories[i]['actions'])
    #     obs = torch.from_numpy(goal_reaching_trajectories[i]['observations'])
    #     s_a_rep = phi_encoder(obs, actions)
    #     values = torch.einsum('ij,ij->i', s_a_rep, torch.from_numpy(last_obs_rep).unsqueeze(0).repeat((200, 1)))
    #     print(f'Return: {(values[:-1] - values[1:]).sum()}')
    #     r_t_pairs.append(((values[:-1] - values[1:]).sum().item(), goal_reaching_trajectories[i]))

    # r_t_pairs = list(sorted(r_t_pairs, key=lambda x: x[0], reverse=True))
    # top_trajectories = [x[1] for x in r_t_pairs[:3]]

    # with FigManager(runner, 'top_trajectory') as fm:
    #     runner._env.render_trajectories(
    #         top_trajectories, random_option_colors[:3], [-10, 10, -10, 10], fm.ax
    #     )

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()