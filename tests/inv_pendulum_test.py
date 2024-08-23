"""Finds a control policy to stabilize a two dimensional nonlinear system.

On every iteration, improve policy_v1 over the policy_vX methods from previous iterations.
Make only small changes.
Try to make the code short and be creative with the method you use.
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


def solve(init_angle) -> float:
  """Returns the RMSE value for a run of the inverted pendulum."""
  initial_state = np.array([init_angle, 0.0], dtype=np.float32)
  horizon_length = 100
  sampling_time = 0.1
  state = initial_state.copy()
  rmse_sum = 0.0

  control_input_seq = np.zeros(horizon_length)
  state_seq = np.zeros((horizon_length, 2))
  state_seq[0] = state.copy()
  time_horizon = sampling_time*np.arange(horizon_length)
  for t in range(horizon_length):
    control_input = policy(state)
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
  # print("ctrl type: ", type(control_input))
  next_state[1] += (np.sin(state[0]) - state[1] + control_input) * sampling_time

  return next_state


def policy(state: np.ndarray) -> float:
  """Returns a control input. state is a 2D array contaning x and x_dot.
  The function is going to return a float input value.
  """
  x = state
  x1, x2 = x[0], x[1]
  return x1**2 + x2**2 - 2*x1 - 2*x2

if __name__ == "__main__":
  score, time, input, state = test(-1.0)
  print(f"Score: {score}")
  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(time, input)
  plt.xlabel("Time")
  plt.ylabel("Control Input")
  plt.subplot(2, 1, 2)
  plt.plot(time, state[:, 0], label="Ang")
  plt.plot(time, state[:, 1], label="Ang Vel")
  plt.xlabel("Time")
  plt.legend()
  plt.show()