"""Finds a heuristic policy for the pendulum swingup task.

On every iteration, improve policy_v1 over the policy_vX methods from previous iterations.
Make only small changes. Try to make the code short. 
"""

import numpy as np

import funsearch

from dm_control import suite


@funsearch.run
def solve(num_runs) -> float:
  """Returns the reward for a heuristic.
  """
  env = suite.load(domain_name="pendulum", task_name="swingup")
  avg_reward = 0.0
  for _ in range(num_runs):
    time_step = env.reset()
    total_reward = 0.0
    sum_diff = 0.0
    for _ in range(1000):
      ref = heuristic(env._step_count, time_step.observation, env.action_spec().shape)
      np.clip(ref, -np.pi, np.pi, out=ref)
      cos_ref = np.cos(ref)
      sin_ref = np.sin(ref)
      cos_obs = time_step.observation['orientation'][0]
      sin_obs = -time_step.observation['orientation'][1]
      cos_diff = cos_ref*cos_obs + sin_ref*sin_obs
      sin_diff = sin_ref*cos_obs - cos_ref*sin_obs
      diff = np.arctan2(sin_diff, cos_diff)
      action = 3/np.pi * diff - 2*time_step.observation['velocity'] + 0.1*sum_diff
      np.clip(action, env.action_spec().minimum, env.action_spec().maximum, out=action)
      time_step = env.step(action)
      total_reward += time_step.reward
    avg_reward += total_reward
  return avg_reward / num_runs
  # return total_reward

@funsearch.evolve
def heuristic(t: int, obs: np.ndarray, output_shape: tuple) -> float:
  """Returns an action reference between -pi and pi of shape output_shape.
  t is a time counter. obs is the observation.
  """
  ref = np.random.uniform(-1, 1, output_shape)
  return ref