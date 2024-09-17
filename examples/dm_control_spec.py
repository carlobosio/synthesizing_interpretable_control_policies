"""Finds a heuristic policy for the pendulum swingup task.

On every iteration, improve policy_v1 over the policy_vX methods from previous iterations.
Make only small changes. Try to make the code short. 
"""

import numpy as np

import funsearch

from dm_control import suite


@funsearch.run
def solve(param) -> float:
  """Returns the reward for a heuristic.
  """
  env = suite.load(domain_name="pendulum", task_name="swingup")
  time_step = env.reset()
  total_reward = 0.0
  for _ in range(1000):
    action = heuristic(time_step.step_type.value, time_step.observation, env.action_spec().shape)
    np.clip(action, env.action_spec().minimum, env.action_spec().maximum, out=action)
    time_step = env.step(action)
    total_reward += time_step.reward
  return total_reward

@funsearch.evolve
def heuristic(t: int, obs: np.ndarray, output_shape: tuple) -> float:
  """Returns an action of shape output_shape.
  t is a time counter.
  """

  action = np.random.uniform(-1, 1, output_shape)
  return action