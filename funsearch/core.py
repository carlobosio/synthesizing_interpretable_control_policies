# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""
import logging
import multiprocessing

from funsearch import code_manipulation


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]


def run(samplers, database, iterations: int = -1):
  """Launches a FunSearch experiment."""

  try:
    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    while iterations != 0:
      for s in samplers:
        s.sample()
      if iterations > 0:
        iterations -= 1
  except KeyboardInterrupt:
    logging.info("Keyboard interrupt. Stopping.")
  database.backup()

def sample_sampler(sampler):
    """Function to run a single sampler."""
    sampler.sample()

def run_parallel(samplers, database, iterations: int = -1):
    """Launches a FunSearch experiment in parallel."""
    try:
        processes = []
        for s in samplers:
            p = multiprocessing.Process(target=sample_sampler, args=(s,))
            p.start()
            processes.append(p)

        for _ in range(iterations if iterations > 0 else float('inf')):
            # Wait for all processes to finish one iteration
            for p in processes:
                p.join()

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt. Stopping.")
        for p in processes:
            p.terminate()  # Kill processes safely

    database.backup()