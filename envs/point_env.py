"""Utility for loading the 2D navigation environments."""
from typing import Optional

import gym
import numpy as np
import scipy

WALLS = {
    'Small':  # max_goal_dist = 3
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':  # max_goal_dist = 9
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'FourRooms':  # max_goal_dist = 14
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'U':  # max_goal_dist = 14
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Spiral11x11':  # max_goal_dist = 45
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Maze11x11':  # max_goal_dist = 49
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls

class PointEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               walls = 'Spiral11x11', resize_factor = 1):
    """Initialize the point environment.

    Args:
      walls: (str or array) binary, H x W array indicating locations of walls.
        Can also be the name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = 0.01
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.array([0, 0, 0, 0]),
        high=np.array([height, width, height, width]),
        dtype=np.float32)
    self.reset()

  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=float)
    state += np.random.uniform(size=2)
    assert not self._is_blocked(state)
    return state

  def _get_obs(self):
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self):
    self.goal = self._sample_empty_state()
    self.state = self._sample_empty_state()
    return self._get_obs()

  def _discretize_state(self, state, resolution=1.0):
    ij = np.floor(resolution * state).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    assert len(state) == 2
    if (np.any(state < self.observation_space.low[:2])
        or np.any(state > self.observation_space.high[:2])):
      return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def step(self, action):
    action = action.copy()
    if not self.action_space.contains(action):
      print('WARNING: clipping invalid action:', action)
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise, (2,))
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state
    
    info = {}
    done = False
    obs = self._get_obs()
    dist = np.linalg.norm(self.goal - self.state)
    rew = float(dist < 2.0)
    return obs, rew, done, info
  
  def calc_eval_metrics(self, trajectories, is_option_trajectories: bool):
    successes = []
    for t in trajectories:
      successes.append(np.sum(t['rewards']) >= 1.0)

    return {
        'success_rate': np.mean(successes),
    }   
  
  @property
  def walls(self):
    return self._walls