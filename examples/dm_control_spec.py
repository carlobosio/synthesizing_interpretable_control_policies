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
    initialize_env(env)
    total_reward = 0.0
    obs = concatenate_obs(time_step)
    sum_diff = 0.0
    for _ in range(1000):
      # ref = heuristic(env._step_count, time_step.observation)
      # ref = np.clip(ref, -np.pi, np.pi)
      # cos_ref = np.cos(ref)
      # sin_ref = np.sin(ref)
      cos_theta = time_step.observation['orientation'][0]
      sin_theta = -time_step.observation['orientation'][1]
      theta = np.arctan2(sin_theta, cos_theta)
      # cos_diff = cos_ref*cos_theta + sin_ref*sin_theta
      # sin_diff = sin_ref*cos_theta - cos_ref*sin_theta
      # diff = np.arctan2(sin_diff, cos_diff)
      # action = 3/np.pi * diff + 0.1*sum_diff - 0.1*time_step.observation['velocity']
      action = heuristic(env._step_count, obs)
      action = np.clip(action, env.action_spec().minimum, env.action_spec().maximum)
      # np.clip(action, env.action_spec().minimum, env.action_spec().maximum, out=action)
      time_step = env.step(action)
      # total_reward += time_step.reward
      total_reward += 1.0 - np.abs(theta)/np.pi
      if np.abs(theta) < 0.5:
        total_reward += 1.0
      obs = concatenate_obs(time_step)
    avg_reward += total_reward
  return avg_reward / num_runs
  # return total_reward

def concatenate_obs(time_step):
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
  """Returns an action between -1 and 1 of shape output_shape.
  t is a time counter. obs is the observation [zz, xz, vel].
  """
  t_oscillation = 35  # Period of local oscillations
  if t < t_oscillation:
    action = np.array([1])
  # if
  # elif
  # if
  # elif
  # ...
  else: # end part of the heuristic
    action = -5*np.arctan2(obs[1], obs[0]) - 0.9*obs[2]
  return action