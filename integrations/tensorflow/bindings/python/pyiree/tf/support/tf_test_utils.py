# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test utilities interop with TensorFlow."""

# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=unsupported-assignment-operation

# This file uses the following abbreviations:
#   ref: reference – for the reference CompiledModule
#   tar: target - for one of the target CompiledModules

import collections
import copy
import os
import tempfile
from typing import Any, Callable, Sequence, Set, Tuple, Type

from absl import flags
from absl import logging
from pyiree.tf.support import module_utils
from pyiree.tf.support import tf_utils
from pyiree.tf.support import trace_utils
import tensorflow.compat.v2 as tf

flags.DEFINE_string("reference_backend", "tf",
                    "The backend to treat as a source of truth.")
flags.DEFINE_list("target_backends", None,
                  "Explicit comma-delimited list of target backends.")
flags.DEFINE_string(
    "artifacts_dir", None,
    "Specifies a directory to dump compilation artifacts and traces to. "
    "Defaults to the OS's tempdir.")
flags.DEFINE_bool(
    "summarize", True,
    "Summarize the inputs and outputs of each module trace logged to disk.")
flags.DEFINE_bool("log_all_traces", False,
                  "Log all traces to logging.info, even if comparison passes.")
flags.DEFINE_bool(
    "get_saved_model", False,
    "Creates and stores a SavedModel for the tf.Module class to be tested.")
FLAGS = flags.FLAGS


def _setup_artifacts_dir(module_name: str) -> str:
  parent_dirs = [
      FLAGS.artifacts_dir,
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'),
      os.environ.get('TEST_TMPDIR'),
      os.path.join(tempfile.gettempdir(), "iree", "modules"),
  ]
  # Use the most preferred path in parent_dirs that isn't None.
  parent_dir = next(parent for parent in parent_dirs if parent is not None)

  artifacts_dir = os.path.join(parent_dir, module_name)
  logging.info("Saving compilation artifacts and traces to '%s'", artifacts_dir)
  os.makedirs(artifacts_dir, exist_ok=True)
  return artifacts_dir


def _parse_target_backends() -> Tuple[Sequence[str], Sequence[str]]:
  """Decodes --target_backends and creates unique ids for them."""
  backend_names = FLAGS.target_backends
  backend_to_index = {k: 0 for k in backend_names if backend_names.count(k) > 1}
  backend_ids = []

  # If there are multiple copies of the same backend_name, index them. e.g.
  # backend_names = ["tf", "iree_vmla", "tf"]
  # --> backend_ids = ["tf_0", "iree_vmla", "tf_1"]
  for backend_name in backend_names:
    if backend_name in backend_to_index:
      backend_ids.append(f"{backend_name}_{backend_to_index[backend_name]}")
      backend_to_index[backend_name] += 1
    else:
      backend_ids.append(backend_name)

  return backend_names, backend_ids


def get_target_backends() -> Sequence[module_utils.BackendInfo]:
  """Gets the BackendInfo instances to compare with the reference backend.

  By default all backends in BackendInfo will be used. Specific backends to
  run on can be specified using the `--target_backends` flag.

  Returns:
    Sequence of BackendInfo that should be used.
  """
  if FLAGS.target_backends is not None:
    logging.info("Using backends from command line: %s", FLAGS.target_backends)
    backend_names, backend_ids = _parse_target_backends()
    backends = [
        module_utils.BackendInfo(backend_name, backend_id)
        for backend_name, backend_id in zip(backend_names, backend_ids)
    ]
  else:
    # If no backends are specified, use them all.
    backends = module_utils.BackendInfo.get_all_backends()
  return backends


Modules = collections.namedtuple("Modules",
                                 ["ref_module", "tar_modules", "artifacts_dir"])

# We have to use a global variable to store the compiled modules so that we can
# avoid recompilation. This is because the TestCase class resets it's entire
# state and calls __init__ before each unittest. It also calls __init__ one
# additional time before that for good measure, which means without storing the
# modules somewhere else we would have to compile each of them at least twice.
# We can't store the modules on the class itself via setUpClass because of #2900
global _global_modules
_global_modules = None


def compile_tf_module(
    module_class: Type[tf.Module],
    exported_names: Sequence[str] = ()) -> Modules:
  """Compiles module_class to each backend that we test.

  Args:
    module_class: the tf.Module subclass to compile.
    exported_names: optional iterable of strings representing which of
      module_class's functions to compile. If exported_names is empty all
      functions will be compiled.

  Returns:
    A 'Modules' namedtuple containing the reference module, target modules and
    artifacts directory.
  """
  global _global_modules
  if _global_modules is not None:
    return _global_modules

  # Setup the directory for saving compilation artifacts and traces.
  artifacts_dir = _setup_artifacts_dir(module_class.__name__)

  # Get the backend information for this test.
  ref_backend_info = module_utils.BackendInfo(FLAGS.reference_backend,
                                              f"{FLAGS.reference_backend}_ref")
  tar_backend_infos = get_target_backends()

  compile_backend = lambda backend_info: backend_info.compile_from_class(
      module_class, exported_names, artifacts_dir)

  ref_module = compile_backend(ref_backend_info)
  tar_modules = [
      compile_backend(backend_info) for backend_info in tar_backend_infos
  ]
  _global_modules = Modules(ref_module, tar_modules, artifacts_dir)
  return _global_modules


def compile_tf_signature_def_saved_model(
    saved_model_dir: str, saved_model_tags: Set[str], module_name: str,
    exported_name: str, input_names: Sequence[str],
    output_names: Sequence[str]) -> Modules:
  """Compiles a SignatureDef SavedModel to each backend that we test.

  Args:
    saved_model_dir: Directory of the saved model.
    saved_model_tags: Optional set of tags to use when loading the model.
    module_name: A name for this compiled module.
    backend_info: BackendInfo with the details for compiling the saved model.
    exported_name: A str representing the signature on the saved model to
      compile.
    input_names: A sequence of kwargs to feed to the saved model.
    output_names: A sequence of named outputs to extract from the saved model.

  Returns:
    A 'Modules' namedtuple containing the reference module, target modules and
    artifacts directory.
  """
  global _global_modules
  if _global_modules is not None:
    return _global_modules

  # Setup the directory for saving compilation artifacts and traces.
  artifacts_dir = _setup_artifacts_dir(module_name)

  # Get the backend information for this test.
  ref_backend_info = module_utils.BackendInfo(FLAGS.reference_backend,
                                              f"{FLAGS.reference_backend}_ref")
  tar_backend_infos = get_target_backends()

  compile_backend = (
      lambda backend_info: backend_info.compile_signature_def_saved_model(
          saved_model_dir, saved_model_tags, module_name, exported_name,
          input_names, output_names, artifacts_dir))

  ref_module = compile_backend(ref_backend_info)
  tar_modules = [
      compile_backend(backend_info) for backend_info in tar_backend_infos
  ]
  _global_modules = Modules(ref_module, tar_modules, artifacts_dir)
  return _global_modules


# We use global variables to store the configuration information for
# tf_function_unittests because tensorflow.python.eager.def_function.Function
# is not an API that we can subclass, and storing the information directly
# that class results in it being deleted at tf.Module initialization.
global _global_unittest_configs
_global_unittest_configs = dict()


def tf_function_unittest(input_generator: tf_utils.InputGeneratorType = None,
                         input_args: Sequence[Any] = None,
                         atol: float = None,
                         rtol: float = None,
                         name: str = None,
                         **tf_function_kwargs):
  """Creates a tf.function that can be used to generate unittests.

  If 'input_generator' and 'input_args' are unspecified then the function will
  be tested using random uniform data.

  Args:
    input_generator:
      an optional callable taking a shape and dtype that returns input data for
      the unittest.
    input_args:
      an optional sequence of values to pass as positional args to the function.
    atol:
      optional, the absolute tolerance to use when comparing the decorated
      function's output.
    rtol:
      optional, the relative tolerance to use when comparing the decorated
      function's output.
    name:
      optional, the name to reference this function with. Must be used if
      decorating a lambda.

  Raises:
    ValueError: if 'input_generator' and 'input_args' are both specified.

  Returns:
    A tf.function with the additional attributes 'input_generator' (from above)
    'trace_kwargs' (from 'atol' and 'rtol' above), and with an updated
    __name__ attribute if 'name' was specified.
  """

  def _store_unittest_info(function):
    # Validate arguments.
    if input_generator is not None and input_args is not None:
      raise ValueError(
          "'input_generator' and 'input_args' cannot both be specified.")

    function = tf.function(**tf_function_kwargs)(function)

    # Set function.__name__
    if name is not None:
      function.__name__ = name
    elif function.__name__ == "<lambda>":
      raise ValueError("The 'name' kwarg must be provided when decorating a "
                       "lambda function.")

    global _global_unittest_configs
    if function.__name__ not in _global_unittest_configs:

      if input_generator is not None:
        # Use the user-specificed input_generator.
        get_trace_args = lambda: tf_utils.generate_inputs(
            function.input_signature, input_generator)
      elif input_args is not None:
        # Use the user-specified input_args.
        get_trace_args = lambda: copy.deepcopy(input_args)
      else:
        # No user data specification – default to using random uniform data.
        get_trace_args = lambda: tf_utils.generate_inputs(
            function.input_signature, tf_utils.uniform)

      _global_unittest_configs[function.__name__] = dict(
          get_trace_args=get_trace_args,
          trace_kwargs=dict(atol=atol, rtol=rtol))

    return function

  return _store_unittest_info


class TestModule(tf.Module):
  """Thin wrapper of tf.Module with helper methods for tf_function_unittests."""

  @classmethod
  def get_tf_function_unittests(cls):
    """Get all tf_function_unittest-created tf.functions on the class."""
    # Initialize the module to ensure that _global_unittest_configs has the
    # info for all of the unittests. (Only doing this if
    # _global_unittest_configs is empty wouldn't address the case where some
    # unittests are defined on the class and some are generated by __init__).
    cls()

    tf_function_unittests = list(_global_unittest_configs.keys())
    if not len(tf_function_unittests):
      raise ValueError(
          "'get_tf_function_unittests' was called but no unittests were found.")
    return tf_function_unittests


class TracedModuleTestCase(tf.test.TestCase):
  """Compiles a tf.Module to multiple backends to test their correctness."""

  def setUp(self) -> None:
    # Runs before each unit test.
    super().setUp()
    self._modules.ref_module.reinitialize()
    for module in self._modules.tar_modules:
      module.reinitialize()

  @classmethod
  def generate_unittests(cls, module_class: Type[TestModule]):
    """Generates unittests for each 'tf_function_unittest' on 'module_class'."""
    for function_name in module_class.get_tf_function_unittests():
      # We have to pass the closure arguments 'function_name', 'get_args' and
      # 'kwargs' to 'trace' via a kwarg instead of using it directly in the body
      # because 'function_name' and 'unittest_config' are overwritten in each
      # iteration of this loop, and python will only use the most recent version
      # of each. If we didn't do this, then we would only test the last function
      # in this loop. The same is true for passing 'trace' to 'unittest'.
      unittest_config = _global_unittest_configs[function_name]

      # Runs the inputs through a (traced) module.
      def trace(module,
                function_name=function_name,
                get_args=unittest_config["get_trace_args"],
                kwargs=unittest_config["trace_kwargs"]):
        getattr(module, function_name)(*get_args(), **kwargs)

      # Give the trace the name of the tf.function that it is testing.
      trace.__name__ = function_name

      # Runs 'trace' on modules compiled to each backend and compares them.
      def unittest(self, trace=trace):
        self.compare_backends(trace, self._modules)

      # Make 'unittest' a function on the TracedModuleTestCase, which tells
      # the test runner to run it.
      unittest.__name__ = f"test_{function_name}"
      if hasattr(cls, unittest.__name__):
        raise ValueError("Tried to generate multiple instances of the unittest "
                         f"'{unittest.__name__}'.")
      setattr(cls, unittest.__name__, unittest)

  def compare_backends(self,
                       trace_function: Callable[[trace_utils.TracedModule],
                                                None],
                       modules: Modules) -> None:
    """Run the reference and target backends on trace_function and compare them.

    Random seeds for tensorflow, numpy and python are set before each invocation
    of trace_function.

    Args:
      trace_function: a function accepting a TracedModule as its argument.
    """
    # Create Traces for each backend.
    ref_trace = trace_utils.Trace(modules.ref_module, trace_function)
    tar_traces = [
        trace_utils.Trace(module, trace_function)
        for module in modules.tar_modules
    ]

    # Run the traces through trace_function with their associated modules.
    tf_utils.set_random_seed()
    trace_function(trace_utils.TracedModule(modules.ref_module, ref_trace))
    if FLAGS.log_all_traces:
      logging.info(ref_trace)
    for module, trace in zip(modules.tar_modules, tar_traces):
      tf_utils.set_random_seed()
      trace_function(trace_utils.TracedModule(module, trace))
      if FLAGS.log_all_traces:
        logging.info(trace)

    # Compare each target trace of trace_function with the reference trace.
    failed_backend_indices = []
    error_messages = []
    for i, tar_trace in enumerate(tar_traces):
      logging.info("Comparing the reference backend '%s' with '%s'",
                   ref_trace.backend_id, tar_trace.backend_id)
      traces_match, errors = trace_utils.compare_traces(ref_trace, tar_trace)
      if not traces_match:
        failed_backend_indices.append(i)
        error_messages.extend(errors)

    # Save the results to disk before validating.
    ref_trace_dir = trace_utils.get_trace_dir(modules.artifacts_dir, ref_trace)
    ref_trace.save_plaintext(ref_trace_dir, FLAGS.summarize)
    ref_trace.serialize(ref_trace_dir)
    for tar_trace in tar_traces:
      tar_trace_dir = trace_utils.get_trace_dir(modules.artifacts_dir,
                                                tar_trace)
      tar_trace.save_plaintext(tar_trace_dir, FLAGS.summarize)
      tar_trace.serialize(tar_trace_dir)

    # Validate results.
    if failed_backend_indices:
      # Extract info for logging.
      failed_backends = [
          tar_traces[i].backend_id for i in failed_backend_indices
      ]
      error_list = ''.join([f'\n  - {message}' for message in error_messages])
      self.fail(
          "Comparison between the reference backend and the following targets "
          f"failed: {failed_backends}. Errors: {error_list}\n"
          "See the logs above for more details about the non-matching calls.")

  @classmethod
  def tearDownClass(cls) -> None:
    # Runs after all unit tests are completed.
    super().tearDownClass()
