"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
import os
import jax
import jax.numpy as jnp
import pickle

import numpy as np
import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax
import flashbax as fbx
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.renderer import render_craftax_pixels
from moviepy import editor as mpy
from wandb_osh.hooks import TriggerWandbSyncHook


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x
    
class PhiNetwork(nn.Module):
    dim_option: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim_option)(x)
        return x


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    option: chex.Array


class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape

    if n_cols is None:
        if v.shape[0] <= 3:
            n_cols = v.shape[0]
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate(
            (v, np.zeros(shape=(len_addition, t, c, h, w), dtype=np.uint8)), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    # basic_env, env_params = gymnax.make(config["ENV_NAME"])
    basic_env = make_craftax_env_from_name(
        "Craftax-Classic-Symbolic-v1", auto_reset=True
    )
    env_params = basic_env.default_params

    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):
        trigger_sync = TriggerWandbSyncHook()  # <--- New!

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        rng, _rng = jax.random.split(rng)
        init_options = jax.random.multivariate_normal(_rng, jnp.zeros(config["DIM_OPTION"]), jnp.eye(config["DIM_OPTION"]), shape=(config["NUM_ENVS"],))

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _, _reward, _done, _info = env.step(rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done, option=jnp.zeros(config["DIM_OPTION"]))
        buffer_state = buffer.init(_timestep)

        _init_pixels = render_craftax_pixels(_env_state.env_state, block_pixel_size=16)

        init_info = _info
        keys = list(_info.keys())
        for k in keys:
            if k.startswith('Achievements'):
                init_info[f'returned_{k}'] = jnp.zeros(config['NUM_ENVS'])
                init_info[k] = _info[k].repeat(config['NUM_ENVS'])
            else:
                init_info[k] = _info[k].repeat(config['NUM_ENVS'])

        # INIT Q-NETWORK AND OPTIMIZER
        q_network = QNetwork(action_dim=env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        init_z = jnp.zeros(config["DIM_OPTION"])
        init_x_with_z = jnp.concatenate([init_x, init_z], axis=-1)
        q_network_params = q_network.init(_rng, init_x_with_z)

        q_tx = optax.adam(learning_rate=config["LR"])

        q_train_state = CustomTrainState.create(
            apply_fn=q_network.apply,
            params=q_network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), q_network_params),
            tx=q_tx,
            timesteps=0,
            n_updates=0,
        )

        # INIT PHI-NETWORK AND OPTIMIZER
        phi_network = PhiNetwork(dim_option=config["DIM_OPTION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        phi_network_params = phi_network.init(_rng, init_x)

        phi_tx = optax.adam(learning_rate=config["LR"])

        phi_train_state = CustomTrainState.create(
            apply_fn=phi_network.apply,
            params=phi_network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), phi_network_params),
            tx=phi_tx,
            timesteps=0,
            n_updates=0,
        )

        # INIT DUAL LAM
        dual_lam = jnp.log(config["DUAL_LAM"])
        dual_lam_train_state = CustomTrainState.create(
            apply_fn=lambda x: x,
            params={"params": dual_lam},
            target_network_params={"params": jnp.copy(dual_lam)},
            tx=optax.adam(learning_rate=config["LR"]),
            timesteps=0,
            n_updates=0,
        )

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )
            return chosed_actions

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            q_train_state, phi_train_state, dual_lam_train_state, buffer_state, env_state, last_obs, last_options, last_info, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            last_obs_with_option = jnp.concatenate([last_obs, last_options], axis=-1)
            q_vals = q_network.apply(q_train_state.params, last_obs_with_option)
            action = eps_greedy_exploration(
                rng_a, q_vals, q_train_state.timesteps
            )  # explore with epsilon greedy_exploration
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )

            # update info
            keys = list(info.keys())
            for k in keys:
                if k.startswith('Achievements'):
                    info[f'returned_{k}'] = last_info[f'returned_{k}'] * (1 - done) + info[k] * done

            # sample new options for any env that is done
            rng, _rng = jax.random.split(rng)
            new_options = jnp.where(
                done[:, None], 
                jax.random.multivariate_normal(_rng, jnp.zeros((config["DIM_OPTION"],)), jnp.eye(config["DIM_OPTION"]), shape=(obs.shape[0],)),
                last_options,
            )

            # update timesteps count
            q_train_state = q_train_state.replace(
                timesteps=q_train_state.timesteps + config["NUM_ENVS"]
            ) 

            phi_train_state = phi_train_state.replace(
                timesteps=phi_train_state.timesteps + config["NUM_ENVS"]
            )

            dual_lam_train_state = dual_lam_train_state.replace(
                timesteps=dual_lam_train_state.timesteps + config["NUM_ENVS"]
            )

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done, option=last_options)
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(q_train_state, phi_train_state, dual_lam_train_state, rng):
                learn_batch = buffer.sample(buffer_state, rng).experience

                # recompute rewards from Metra loss
                cur_z = phi_network.apply(
                    phi_train_state.params, learn_batch.first.obs
                )
                next_z = phi_network.apply(
                    phi_train_state.params, learn_batch.second.obs
                )
                target_z = next_z - cur_z
                metra_rewards = jnp.sum(target_z * learn_batch.first.option, axis=-1)

                next_obs_with_option = jnp.concatenate([learn_batch.second.obs, learn_batch.first.option], axis=-1)
                q_next_target = q_network.apply(
                    q_train_state.target_network_params, next_obs_with_option
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                target = (
                    metra_rewards
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )

                def _q_loss_fn(params):
                    obs_with_options = jnp.concatenate([learn_batch.first.obs, learn_batch.first.option], axis=-1)
                    q_vals = q_network.apply(
                        params, obs_with_options
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2)
                
                def _phi_loss_fn(params, dual_lam):
                    cur_z = phi_network.apply(
                        params, learn_batch.first.obs
                    )
                    next_z = phi_network.apply(
                        params, learn_batch.second.obs
                    )
                    target_z = next_z - cur_z

                    rewards = jnp.sum(target_z * learn_batch.first.option, axis=-1)

                    if config["CRITIC_TYPE"] == 'l2':
                        l2 = jnp.sum((next_z - cur_z) ** 2, axis=-1)
                        loss = -1 * (rewards - 0.5 * l2).mean()

                        return loss, jnp.mean(l2)
                    
                    elif config["CRITIC_TYPE"] == "metra":
                        cst_dist = jnp.ones_like(cur_z[:, 0])
                        l2 = jnp.square(next_z - cur_z).mean(axis=-1)
                        cst_penalty = cst_dist - jnp.square(next_z - cur_z).mean(axis=-1)
                        cst_penalty = jnp.clip(cst_penalty, a_max=config["DUAL_SLACK"])
                        loss = -1 * (rewards + jnp.exp(dual_lam) * cst_penalty).mean()

                        return loss, cst_penalty.mean()

                def _dual_lam_loss_fn(params, cst_penalty):
                    log_dual_lam = params["params"]
                    loss = log_dual_lam * cst_penalty

                    return loss

                (phi_loss, phi_l2), phi_grads = jax.value_and_grad(_phi_loss_fn, has_aux=True)(phi_train_state.params, dual_lam_train_state.params["params"])
                phi_train_state = phi_train_state.apply_gradients(grads=phi_grads)
                phi_train_state = phi_train_state.replace(n_updates=phi_train_state.n_updates + 1)

                dual_lam_loss, dual_lam_grads = jax.value_and_grad(_dual_lam_loss_fn)(dual_lam_train_state.params, phi_l2)
                dual_lam_train_state = dual_lam_train_state.apply_gradients(grads=dual_lam_grads)
                dual_lam_train_state = dual_lam_train_state.replace(n_updates=dual_lam_train_state.n_updates + 1)

                q_loss, q_grads = jax.value_and_grad(_q_loss_fn)(q_train_state.params)
                q_train_state = q_train_state.apply_gradients(grads=q_grads)
                q_train_state = q_train_state.replace(n_updates=q_train_state.n_updates + 1)

                return q_train_state, q_loss, phi_train_state, phi_loss, phi_l2, dual_lam_loss, dual_lam_train_state

            def _eval_phase(q_train_state, rng):
                jax.debug.print('in eval phase!')
                def eval_episode(rng_env):
                    rng, rng_r, rng_o = jax.random.split(rng_env, num=3)
                    obs, state = env.reset(rng_r, env_params)
                    total_reward = 0.0
                    total_length = 0

                    init_pixels = render_craftax_pixels(state.env_state, block_pixel_size=16).astype(jnp.uint8)

                    option = jax.random.multivariate_normal(rng_o, jnp.zeros(config["DIM_OPTION"]), jnp.eye(config["DIM_OPTION"]))

                    def cond_fn(loop_state):
                        done, *_ = loop_state
                        return jnp.logical_not(done)

                    def body_fn(loop_state):
                        done, obs, option, state, total_reward, total_length, frames, rng = loop_state
                        rng, rng_e, rng_a, rng_s = jax.random.split(rng, num=4)
                        obs_with_option = jnp.concatenate([obs, option], axis=-1)
                        q_vals = q_network.apply(q_train_state.params, obs_with_option)
                        greedy_action = jnp.argmax(q_vals, axis=-1)
                        chosen_action = jnp.where(
                            jax.random.uniform(rng_e, greedy_action.shape) < config['EVAL_EPS'],
                            jax.random.randint(rng_a, shape=greedy_action.shape, minval=0, maxval=q_vals.shape[-1]),
                            greedy_action,
                        )
                        new_obs, new_state, reward, done, _ = env.step(rng_s, state, chosen_action, env_params)
                        pixels = render_craftax_pixels(new_state.env_state, block_pixel_size=16).astype(jnp.uint8)
                        total_reward += reward
                        total_length += 1
                        frames = frames.at[total_length].set(pixels)

                        return done, new_obs, option, new_state, total_reward, total_length, frames, rng

                    # Initialize the loop state
                    init_frames = jnp.zeros((500, *init_pixels.shape), dtype=jnp.uint8)  # Adjust frame_shape as necessary
                    init_frames = init_frames.at[0].set(init_pixels)
                    initial_loop_state = (False, obs, option, state, total_reward, total_length, init_frames, rng)

                    # Run the loop
                    final_loop_state = jax.lax.while_loop(cond_fn, body_fn, initial_loop_state)

                    # Extract the final total_reward from the loop state
                    _, _, _, _, total_reward, total_length, frames, _, = final_loop_state

                    return total_reward, total_length, frames

                # Run evaluation over several episodes
                rng_episodes = jax.random.split(rng, config["NUM_EVAL_EPISODES"])
                total_rewards, total_lengths, videos = jax.vmap(eval_episode)(rng_episodes)
                # avg_reward = jnp.mean(total_rewards)
                # avg_length = jnp.mean(total_lengths)
                # jax.debug.print('Avg reward: {avg_reward}', avg_reward=avg_reward)
                # jax.debug.print('Avg length: {avg_length}', avg_length=avg_length)

                return videos

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    q_train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    q_train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )

            q_train_state, q_loss, phi_train_state, phi_loss, phi_l2, dual_lam_loss, dual_lam_train_state = jax.lax.cond(
                is_learn_time,
                lambda q_train_state, phi_train_state, dual_lam_train_state, rng: _learn_phase(q_train_state, phi_train_state, dual_lam_train_state, rng),
                lambda q_train_state, phi_train_state, dual_lam_train_state, rng: (q_train_state, jnp.array(0.0), phi_train_state, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), dual_lam_train_state),  # do nothing
                q_train_state, phi_train_state, dual_lam_train_state,
                _rng,
            )

            rng, _rng = jax.random.split(rng)
            is_eval_time = (q_train_state.timesteps % config["EVAL_INTERVAL"] == 0)
            # videos = jax.lax.cond(
            #     is_eval_time,
            #     lambda q_train_state, rng: _eval_phase(q_train_state, rng),
            #     lambda q_train_state, rng: jnp.zeros((config["NUM_EVAL_EPISODES"], 500, *_init_pixels.shape), dtype=jnp.uint8),  # do nothing
            #     q_train_state, _rng,
            # )
            videos = None

            # update target network
            q_train_state = jax.lax.cond(
                q_train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda q_train_state: q_train_state.replace(
                    target_network_params=optax.incremental_update(
                        q_train_state.params,
                        q_train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda q_train_state: q_train_state,
                operand=q_train_state,
            )

            metrics = {
                "timesteps": q_train_state.timesteps,
                "updates": q_train_state.n_updates,
                "q_loss": q_loss.mean(),
                "phi_loss": phi_loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
                "phi_l2": phi_l2.mean(),
                "dual_lam_loss": dual_lam_loss.mean(),
                "dual_lam": jnp.exp(dual_lam_train_state.params["params"]),
            }

            for k in info:
                if k.startswith('Achievements'):
                    metrics[f'returned_{k}'] = info[f'returned_{k}'].mean()

            # report on wandb if required
            def callback(metrics, videos):
                if metrics["timesteps"] % config['WANDB_LOG_INTERVAL'] == 0:
                    print('\nMETRICS:')
                    for k, v in metrics.items():
                        print(f"{k}: {v}")
                    wandb.log(metrics)
                    trigger_sync()

                # if metrics["timesteps"] % config['EVAL_INTERVAL'] == 0:
                #     videos = videos.transpose(0, 1, 4, 2, 3)
                #     videos = prepare_video(videos)
                #     # Encode sequence of images into gif string
                #     clip = mpy.ImageSequenceClip(list(videos), fps=15)
                #     plot_path = os.path.join('logs', f'metra-jax-craftax-{config["DIM_OPTION"]}', f'video-{metrics["timesteps"] // config["EVAL_INTERVAL"]}.mp4')
                #     clip.write_videofile(str(plot_path), audio=False, verbose=False, logger=None)
                    
            jax.debug.callback(callback, metrics, videos)

            runner_state = (q_train_state, phi_train_state, dual_lam_train_state, buffer_state, env_state, obs, new_options, info, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (q_train_state, phi_train_state, dual_lam_train_state, buffer_state, env_state, init_obs, init_options, init_info, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main():
    config = {
        "NUM_ENVS": 100,
        "BUFFER_SIZE": 1000_000,
        "BUFFER_BATCH_SIZE": 256,
        "TOTAL_TIMESTEPS": 100_000_000,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 1e-4,
        "LEARNING_STARTS": 10_000,
        "TRAINING_INTERVAL": 10,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "CartPole-v1",
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "disabled",  # set to online to activate wandb
        "ENTITY": "anonymous",
        "PROJECT": "scaling-unsup-rl",
        # METRA STUFF
        "DIM_OPTION": 512,
        "EVAL_INTERVAL": 10_000_000,
        "NUM_EVAL_EPISODES": 9,
        "EVAL_EPS": 0.01,
        "WANDB_LOG_INTERVAL": 100,
        "CRITIC_TYPE": 'metra',
        "DUAL_LAM": 30.0,
        "DUAL_SLACK": 1e-3
    }

    wandb_name = f'metra-jax-craftax-dim-{config["DIM_OPTION"]}-critic-{config["CRITIC_TYPE"]}-anneal-{config["EPSILON_ANNEAL_TIME"]}'
    LOG_FOLDER = os.path.join('logs', wandb_name)
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        name=wandb_name,
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

if __name__ == "__main__":
    main()
