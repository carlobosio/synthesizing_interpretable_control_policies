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
  obs_spec = env.observation_spec()
  action_spec = env.action_spec()
  avg_reward = 0.0
  for _ in range(num_runs):
    time_step = env.reset()
    initialize_env(env)
    total_reward = 0.0
    obs = concatenate_obs(time_step, obs_spec)
    # sum_diff = 0.0
    for _ in range(1000):
      cos_theta = time_step.observation['orientation'][0]
      sin_theta = -time_step.observation['orientation'][1]
      theta = np.arctan2(sin_theta, cos_theta)
      action = heuristic(env._step_count, obs)
      action = np.clip(action, -1, 1)
      time_step = env.step(action)
      # total_reward += time_step.reward
      total_reward += 1.0 - np.abs(theta)/np.pi - 0.1*np.abs(action) - 0.1*np.abs(obs[2])
      if np.abs(theta) < 0.5:
        total_reward += 1.0
      obs = concatenate_obs(time_step, obs_spec)
    avg_reward += total_reward
  return avg_reward / num_runs

def concatenate_obs(time_step, obs_spec):
  return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

def initialize_env(env):
  env.physics.named.data.qpos['hinge'][0] = np.pi
  env.physics.named.data.qvel['hinge'][0] = 0.0
  env.physics.named.data.qacc['hinge'][0] = 0.0
  env.physics.named.data.qacc_smooth['hinge'][0] = 0.0
  env.physics.named.data.qacc_warmstart['hinge'][0] = 0.0
  env.physics.named.data.actuator_moment['torque'][0] = 0.0
  env.physics.named.data.qfrc_bias['hinge'][0] = 0.0

@funsearch.evolve
def heuristic(t: int, obs: np.ndarray) -> float:
  """Returns an action between -1 and 1.
  t is a time counter. obs size is 3.
  """

  x1 = np.arctan2(-obs[1], obs[0])
  x2 = obs[2]
  if t < 20:
    action = 1.0
  # elif t < ...
  else: # at the end
    action = 5*x1 - 0.9*x2
  return action