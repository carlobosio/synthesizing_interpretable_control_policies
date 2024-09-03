"""Finds a control heuristic for a four dimensional non feedback-linearizable system.

On every iteration, improve heuristic_v1 over the heuristic_vX methods from previous iterations.
Make only small changes. Try to make the code short.
"""

import numpy as np

import funsearch


@funsearch.run
def evaluate(init_pos) -> float:
  """Returns the negative rmse score for a heuristic."""
  rmse_value = solve(init_pos)
  # print(f"[run] output rmse: {rmse_value}")
  if np.isfinite(rmse_value):
    return float(-np.log(rmse_value))
  else:
    # print(f"[run] output rmse is not finite: {rmse_value}")
    return -100.0


def solve(init_pos) -> float:
  """Returns the RMSE value for a run of the ball and beam.
  state = [r, r_dot, theta, theta_dot]
  """

  initial_state = np.array([init_pos, 0.0, 0.0, 0.0], dtype=np.float32)
  horizon_length = 100
  sampling_time = 0.1
  state = initial_state.copy()
  rmse_sum = 0.0

  for _ in range(horizon_length):
    control_input = heuristic(state)
    state = simulate(state, control_input, sampling_time)
    rmse_sum += np.linalg.norm(state)
  
  return rmse_sum / horizon_length

def simulate(state: np.ndarray, control_input: float, sampling_time: float) -> np.ndarray:
  """Simulates a step.
  """
  next_state = state.copy()
  next_state[0] += state[1] * sampling_time
  next_state[1] += (state[0]*state[3]**2 - 9.81*np.sin(state[2])) * sampling_time
  next_state[2] += state[3] * sampling_time
  next_state[3] += control_input * sampling_time

  return next_state


@funsearch.evolve
def heuristic(state: np.ndarray) -> float:
  """Returns a control input. state is a 4D array.
  The function is going to return a float value.
  """
  return 0.0