"""Finds a controller to stabilize a two dimensional nonlinear system.

On every iteration, improve controller_v1 over the controller_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""

import numpy as np

import funsearch


@funsearch.run
def evaluate(init_angle) -> float:
  """Returns the negative rmse score for a controller."""
  rmse_value = solve(init_angle)
  return -rmse_value


def solve(init_angle) -> float:
  """Returns the RMSE value for a run of the inverted pendulum."""
  initial_state = np.array([init_angle, 0.0], dtype=np.float32)
  horizon_length = 100
  sampling_time = 0.1
  state = initial_state.copy()
  rmse_sum = 0.0

  for _ in range(horizon_length):
    control_input = controller(state)
    state = simulate(state, control_input, sampling_time)
    rmse_sum += np.linalg.norm(state)

  return rmse_sum / horizon_length

def simulate(state: np.ndarray, control_input: float, sampling_time: float) -> np.ndarray:
  """Simulates a step.
  """
  next_state = state.copy()
  next_state[0] += state[1] * sampling_time
  next_state[1] += (np.sin(state[0]) - state[1] + control_input) * sampling_time

  return next_state


@funsearch.evolve
def controller(state: np.ndarray) -> float:
  """Returns a control input. state is a 2D array contaning x and x_dot.
  The function is going to return a float input value.
  """
  return 0.0