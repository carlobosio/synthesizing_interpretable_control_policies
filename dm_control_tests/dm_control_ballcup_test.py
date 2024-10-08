"""Test for ball in cup
"""

import numpy as np
import matplotlib.pyplot as plt
import dm_control
from dm_control import suite
from dm_control import viewer

def initialize_to_zero(env):
  env.physics.named.data.qpos['ball_x'][0] = 0.0
  env.physics.named.data.qpos['ball_z'][0] = 0.0

def heuristic_1(obs: np.ndarray, output_shape: tuple) -> np.ndarray: # best one so far (upper right corner)
  """Returns two actions between -1 and 1.
  obs size is 8.
  """
  p1 = obs[0:2]
  p2 = obs[2:4]
  v1 = obs[4:6]
  v2 = obs[6:8]

  action = np.zeros((2,))
  action[1] = -0.1
  if obs[0] < 0.2:
    action[0] = 1

  if obs[3] < -0.2:
    action[1] = 1
  elif obs[4] > 0.2:
    action[1] = -1
  elif obs[5] < -0.2:
    action[1] = -1

  if obs[6] > 0.5:
    action[1] = 1
  elif obs[7] < -0.5:
    action[1] = 1

  if obs[1] < 0.2:
    action[0] = 1

  # if obs[3] - obs[1] > 0.1:
  #   action[1] = action[1] - 0.1

  return action

def heuristic_2(obs: np.ndarray, output_shape: tuple) -> np.ndarray:
  """Returns a control input. state is a 4D array.
  The function is going to return a float value.
  """
  """Improved version of `heuristic_v0`."""

  action  = np.zeros((2,))

  ball_pos   = obs[:2]
  cup_pos    = obs[2:4]

  if abs(cup_pos[0] - ball_pos[0]) < 1e-2:
    action[1] = 1.0
  elif cup_pos[0] < ball_pos[0]:
    action[0] = 1.0
  else:
    action[0] = -1.0

  return action

def heuristic_3(obs: np.ndarray, output_shape: tuple) -> np.ndarray: 

  action = np.zeros(output_shape)

  if obs[0] > 0:
    action[1] = 1
  elif obs[0] < 0:
    action[1] = -1
  elif obs[3] < 0:
    action[0] = -1
  elif obs[3] > 0:
    action[0] = 1

  if obs[1] < obs[4]:
    action[0] = 1
  elif obs[1] > obs[4]:
    action[0] = -1

  if (obs[2] > obs[5]) and (action[1] < 0):
    action[1] = -0.5
  elif (obs[2] < obs[5]) and (action[1] > 0):
    action[1] = 0.5

  return action

def heuristic_4(obs: np.ndarray, output_shape: tuple) -> np.ndarray: # this is the one that moves around
  p1 = obs[0:2]
  p2 = obs[2:4]
  v1 = obs[4:6]
  v2 = obs[6:8]

  action = np.zeros((2,))
  if p1[0] < 0:
    action[0] = -1
  if p2[0] > 0:
    action[0] = 1
  if p1[1] < 0:
    action[1] = -1
  if p2[1] > 0:
    action[1] = 1

  if np.sign(v2[0]) == -1:
    action[0] = -1
  if np.sign(v2[0]) == 1:
    action[0] = 1
  if np.sign(v2[1]) == -1:
    action[1] = -1
  if np.sign(v2[1]) == 1:
    action[1] = 1
  if any([action[0] == -1, action[1] == -1]):
    action += np.array([-0.5, 0])
  if any([action[0] == 1, action[1] == 1]):
    action += np.array([0.5, 0])

  return action 

if __name__ == "__main__":
  env = suite.load(domain_name="ball_in_cup", task_name="catch")  
  obs_spec = env.observation_spec()
  action_spec = env.action_spec()
  
  action_size = action_spec.shape[0]
  steps_offset = 0
  action_seq = np.zeros((action_size, 1000-steps_offset))

  def concatenate_obs(time_step, obs_spec):
    return np.concatenate([time_step.observation[k].ravel() for k in obs_spec])

  def custom_reward(obs: np.ndarray) -> float:
    x_cup = obs[0]
    z_cup = obs[1]
    x_ball = obs[2]
    z_ball = obs[3]
    angle = np.arctan2(x_ball - x_cup, z_ball - z_cup)
    vx_ball = obs[6]
    vz_ball = obs[7]
    v_ball = np.sqrt(vx_ball**2 + vz_ball**2)
    reward = 1 - np.abs(angle)/np.pi
    if v_ball > 4.0:
      reward -= 0.1*v_ball
    return reward
  
  # time.sleep(.002)
  time_step = env.reset()
  initialize_to_zero(env)
  # for _ in range(steps_offset):
  #   env.step(0.0)

  # time_step = env.step(0.0)
  total_reward = 0.0
  obs = concatenate_obs(time_step, obs_spec)
  obs[3] -= 0.3
  obs_size = obs.shape[0]
  obs_seq = np.zeros((obs_size, 1001-steps_offset))
  obs_seq[:, 0] = obs
  time = range(1000-steps_offset)
  # sum_diff = 0.0
  def heuristic_timestep(time_step):
    obs = concatenate_obs(time_step, obs_spec)
    obs[3] -= 0.3
    action = heuristic_1(obs, action_spec.shape)
    # print("Cup Pos: ", obs, " Action: ", action)
    return action
  viewer.launch(env, policy=heuristic_timestep)

  for i in time:
    action = heuristic_2(obs, action_spec.shape)
    action = np.array(action, dtype=np.float64)
    action = np.clip(action, action_spec.minimum[0], action_spec.maximum[0])
    action_seq[:, i] = action
    time_step = env.step(action)
    obs = concatenate_obs(time_step, obs_spec)
    obs[3] -= 0.3
    obs_seq[:, i+1] = obs
    # obs_seq[3, i+1] -= 0.3
    total_reward += time_step.reward
    total_reward += custom_reward(obs)

  # time = np.array(time).squeeze()
  print(f"Score: {total_reward}")
  LW = 3
  start = 1
  plt.rcParams.update({'font.size': 22})

  # Create figure and axes for subplots
  fig = plt.figure(figsize=(10, 8))
  axs1 = plt.subplot(2, 2, 1)

  # First subplot (Action)
  axs1.plot(time[start:], action_seq[0,start:], linewidth=LW, label="action x")
  axs1.plot(time[start:], action_seq[1,start:], linewidth=LW, linestyle='--', label="action y")
  axs1.set_ylabel("actions")

  axs2 = plt.subplot(2, 2, 3)
  vel_ball = np.sqrt(obs_seq[6, start:-1]**2 + obs_seq[7, start:-1]**2)
  axs2.plot(time[start:], vel_ball, linewidth=LW, color="blue")
  axs2.set_ylabel("ball velocity")
  axs2.set_xlabel("time")

  axs3 = plt.subplot(1, 2, 2)
  # Plot positions
  axs3.plot(obs_seq[0, start:], obs_seq[1, start:], label="cup", ls="-", linewidth=LW, color="green")
  axs3.plot(obs_seq[2, start:], obs_seq[3, start:], label="ball", ls="--", linewidth=LW, color="red")
  axs3.set_ylabel("positions y")
  axs3.set_xlabel("positions x")
  axs3.legend()

  # Third subplot (Theta_dot)
  # axs[2].plot(time[1:], obs_seq[2, 1:-1], linewidth=LW, color="red")
  # axs[2].set_xlabel("Time")
  # axs[2].set_ylabel("theta_dot")

  # Display the plot
  # plt.tight_layout()
  plt.show()