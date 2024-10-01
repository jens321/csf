import time

import numpy as np
import jax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.renderer import render_craftax_pixels


rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
rngs = jax.random.split(_rng, 3)

# Create environment
env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=True)
env_params = env.default_params

print('made the env. Resetting ...')
# Get an initial state and observation
obs, state = env.reset(rngs[0], env_params)
print('done resetting')

# # pixels = render_craftax_pixels(state, block_pixel_size=64)
# # pixels = np.array(pixels)
# # # convert to int
# # pixels = np.array(pixels, dtype=np.uint8)

step_fn = jax.jit(env.step)

done = False
i = 0
g = 0
while not done:

    # Pick random action
    action = env.action_space(env_params).sample(rngs[1])

    # Step environment
    # start = time.time()
    obs, state, reward, done, info = step_fn(rngs[2], state, action, env_params)
    print(reward)
    g += reward
    # end = time.time()
    # print(f'stepped in {end - start} seconds')

    i += 1
    
    # if i % 100 == 0:
    #     print(f'Stepped {i} times')
    #     end = time.time()
    #     print(f'fps: {100 / (end - start)}')
    #     start = time.time()

breakpoint()


# print('done with stepping!')
