"""Finds a heuristics function for the task to solve.

On every iteration, improve function_v1 over the function_vX methods from previous iterations.
Make only small changes.
"""

import numpy as np

import funsearch


@funsearch.run
def evaluate(init_angle) -> float:
  """Returns the negative rmse score for a policy."""
  rmse_value = solve(init_angle)
  # print(f"[run] output rmse: {rmse_value}")
  if np.isfinite(rmse_value):
    return float(-np.log(rmse_value))
  else:
    # print(f"[run] output rmse is not finite: {rmse_value}")
    return -100.0


def solve(init_angle) -> float:
  """Returns the RMSE value for a run."""
  initial_state = np.array([init_angle, 0.0], dtype=np.float32)
  horizon_length = 100
  sampling_time = 0.1
  state = initial_state.copy()
  rmse_sum = 0.0

  for _ in range(horizon_length):
    control_input = function(state)
    state = simulate(state, control_input, sampling_time)
    rmse_sum += np.linalg.norm(state)
  
  return rmse_sum / horizon_length

def simulate(state: np.ndarray, control_input: float, sampling_time: float) -> np.ndarray:
  """Simulates a step.
  """
  next_state = state.copy()
  next_state[0] += state[1] * sampling_time
  # print("ctrl type: ", type(control_input))
  next_state[1] += (np.sin(state[0]) - state[1] + control_input) * sampling_time

  return next_state


@funsearch.evolve
def function(x: np.ndarray) -> float:
  """x is a 2D array contaning the floats x1 and x2.
  The function is going to return a float scalar value.
  """
  return 0.0