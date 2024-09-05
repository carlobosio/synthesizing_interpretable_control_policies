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

"""Class for evaluating programs proposed by the Sampler."""
import ast
import re
from collections.abc import Sequence
import copy
from typing import Any, Tuple

from funsearch import code_manipulation
from funsearch import programs_database
from funsearch import sandbox

"""
  Regex to find all methods named 'priority_vX'.
  With each match, start from the 'def priority_vX(' and continue until there's a new line with any of
  - a new 'def'
  - ` or ' or # without indentation
"""
# METHOD_MATCHER = re.compile(r"def priority_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
# METHOD_NAME_MATCHER = re.compile(r"priority_v\d+")
# method_str = "def priority_v"

# [carlo] trying to change this to solve inverted pendulum
METHOD_MATCHER = re.compile(r"def policy_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
METHOD_NAME_MATCHER = re.compile(r"policy_v\d+")
method_str = "def policy_v"

# METHOD_MATCHER = re.compile(r"def heuristic_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
# METHOD_NAME_MATCHER = re.compile(r"heuristic_v\d+")
# method_str = "def heuristic_v"

# [carlo] solve inverted pendulum with reduced context 
# METHOD_MATCHER = re.compile(r"def function_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
# METHOD_NAME_MATCHER = re.compile(r"function_v\d+")
# method_str = "def function_v"

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _find_method_implementation(generated_code: str) -> Tuple[str, str]:
  """Find the last method specified in METHOD_MATCHER within generated code.

  Return the code and the name of the method.
  """
  matches = METHOD_MATCHER.findall(generated_code)
  if not matches:
    return "", ""
  last_match = matches[-1]
  name = METHOD_NAME_MATCHER.search(last_match).group()
  return last_match, name


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  if not type(generated_code) is str:
    generated_code = str(generated_code)

  method_name = "fake_function_header"
  # Check is the response only a continuation for our prompt or full method implementation with header
  if method_str in generated_code:
    code, method_name = _find_method_implementation(generated_code)
  else:
    code = f'def {method_name}():\n{generated_code}'

  # Finally parse the code to make sure it's valid Python
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor(method_name)
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  body = _trim_function_body(generated_code)
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  return evolved_function, str(program)



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      sbox: sandbox.DummySandbox,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      timeout_seconds: int = 30,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = sbox

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)

    scores_per_test = {}
    for current_input in self._inputs:            # runs the function on all inputs provided in the launch command
      test_output, runs_ok = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        scores_per_test[current_input] = test_output
    if scores_per_test:
      self._database.register_program(new_function, island_id, scores_per_test)
