"""Test for ball and beam
"""

import numpy as np
import matplotlib.pyplot as plt

def test(init_angle):
  """Returns the negative log rmse score for a policy."""
  rmse_value, time, input, state = solve(init_angle)
  # print(f"[run] output rmse: {rmse_value}")
  if np.isfinite(rmse_value):
    return float(-np.log(rmse_value)), time, input, state
  else:
    print(f"[run] output rmse is not finite: {rmse_value}")
    return -100.0


def solve(init_pos) -> float:
  """Returns the RMSE value for a run of the inverted pendulum."""
  initial_state = np.array([init_pos, 0.0, 0.0, 0.0], dtype=np.float32)
  horizon_length = 1000
  sampling_time = 0.1
  state = initial_state.copy()
  rmse_sum = 0.0

  control_input_seq = np.zeros(horizon_length)
  state_seq = np.zeros((horizon_length, 4))
  state_seq[0] = state.copy()
  time_horizon = sampling_time*np.arange(horizon_length)
  for t in range(horizon_length):
    control_input = heuristic(state)
    control_input_seq[t] = control_input
    state = simulate(state, control_input, sampling_time)
    state_seq[t] = state.copy()
    rmse_sum += np.linalg.norm(state)

  return rmse_sum / horizon_length, time_horizon, control_input_seq, state_seq

def simulate(state: np.ndarray, control_input: float, sampling_time: float) -> np.ndarray:
  """Simulates a step.
  """
  next_state = state.copy()
  next_state[0] += state[1] * sampling_time
  next_state[1] += (state[0]*state[3]**2 - 9.81*np.sin(state[2])) * sampling_time
  next_state[2] += state[3] * sampling_time
  next_state[3] += control_input * sampling_time

  return next_state


def heuristic(state: np.ndarray) -> float:
  """Returns a control input. state is a 4D array.
  The function is going to return a float value.
  """
  """Improved version of `heuristic_v0`."""
  # K = np.array([-1.0, -1.5, 10.7,  4.6])
  r, r_dot, theta, theta_dot = state
  K = np.array([-1.0, -1, 10,  5])
  return r + 2*r_dot - 10*theta - 10*theta_dot


if __name__ == "__main__":
  score, time, input, state = test(0.2)
  print(f"Score: {score}")
  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(time, input)
  plt.xlabel("Time")
  plt.ylabel("Control Input")
  plt.subplot(2, 1, 2)
  plt.plot(time, state[:, 0], label="Pos")
  plt.plot(time, state[:, 1], label="Vel")
  plt.xlabel("Time")
  plt.legend()
  plt.show()