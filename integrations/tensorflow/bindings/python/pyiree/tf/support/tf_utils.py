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
"""Utilities interop with TensorFlow."""

import os
import random
from typing import Any, Callable, Sequence, Set, Tuple, Union

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

InputGeneratorType = Callable[[Sequence[int], Union[tf.DType, np.dtype]],
                              np.ndarray]


def set_random_seed(seed: int = 0) -> None:
  """Set random seed for tf, np and random."""
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def uniform(shape: Sequence[int],
            dtype: Union[tf.DType, np.dtype] = np.float32,
            low: float = -1.0,
            high: float = 1.0) -> np.ndarray:
  """np.random.uniform with simplified API and dtype and bool support."""
  dtype = dtype.as_numpy_dtype if isinstance(dtype, tf.DType) else dtype
  if dtype == np.bool:
    return np.random.choice(2, shape).astype(np.bool)
  else:
    values = np.random.uniform(size=shape, low=low, high=high)
    if np.issubdtype(dtype, np.integer):
      values = np.round(values)
    return values.astype(dtype)


def ndarange(shape: Sequence[int],
             dtype: Union[tf.DType, np.dtype] = np.float32) -> np.ndarray:
  """np.ndarange for arbitrary input shapes."""
  dtype = dtype.as_numpy_dtype if isinstance(dtype, tf.DType) else dtype
  return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


def random_permutation(
    shape: Sequence[int],
    dtype: Union[tf.DType, np.dtype] = np.float32) -> np.ndarray:
  """Returns a random permutation of 'np.prod(shape)' numbers."""
  values = ndarange(shape, dtype)
  np.random.shuffle(values)
  return values


def apply_function(values, function):
  """Applies 'function' recursively to the inputted values."""
  if isinstance(values, list):
    return [apply_function(v, function) for v in values]
  elif isinstance(values, tuple):
    return tuple([apply_function(v, function) for v in values])
  elif isinstance(values, dict):
    return {k: apply_function(v, function) for k, v in values.items()}
  else:
    return function(values)


def generate_inputs(
    spec,  # Union[Sequence[tf.TensorSpec], tf.TensorSpec]
    input_generator: InputGeneratorType,
) -> Sequence[np.ndarray]:
  """Generates inputs for a given input signature using 'input_generator'."""
  make_static = lambda shape: [dim if dim is not None else 2 for dim in shape]
  generate = lambda spec: input_generator(make_static(spec.shape), spec.dtype)
  return apply_function(spec, generate)


def normalize_numpy(result: np.ndarray):
  """Normalizes TF and TFLite's outputs to match IREE's"""
  if np.isscalar(result):
    result = np.array(result)
  if result.dtype == np.bool:
    # IREE interprets bools as int8s, so we modify this for comparison.
    result = result.astype(dtype=np.int8)
  return result


def convert_to_numpy(values: Any) -> Any:
  """Converts any tf.Tensor in values to numpy."""

  def _convert_to_numpy(tensor: Any) -> Any:
    if not isinstance(tensor, tf.Tensor):
      return tensor
    return normalize_numpy(tensor.numpy())

  return apply_function(values, _convert_to_numpy)


def to_mlir_type(dtype: np.dtype) -> str:
  """Returns a string that denotes the type 'dtype' in MLIR style."""
  if not isinstance(dtype, np.dtype):
    # Handle np.int8 _not_ being a dtype.
    dtype = np.dtype(dtype)
  bits = dtype.itemsize * 8
  if np.issubdtype(dtype, np.integer):
    return f"i{bits}"
  elif np.issubdtype(dtype, np.floating):
    return f"f{bits}"
  else:
    raise TypeError(f"Expected integer or floating type, but got {dtype}")


def get_shape_and_dtype(array: np.ndarray,
                        allow_non_mlir_dtype: bool = False) -> str:
  shape_dtype = [str(dim) for dim in list(array.shape)]
  if np.issubdtype(array.dtype, np.number):
    shape_dtype.append(to_mlir_type(array.dtype))
  elif allow_non_mlir_dtype:
    shape_dtype.append(f"<dtype '{array.dtype}'>")
  else:
    raise TypeError(f"Expected integer or floating type, but got {array.dtype}")
  return "x".join(shape_dtype)


def save_input_values(inputs: Sequence[np.ndarray],
                      artifacts_dir: str = None) -> str:
  """Saves input values with IREE tools format if 'artifacts_dir' is set."""
  result = []
  for array in inputs:
    shape_dtype = get_shape_and_dtype(array)
    values = " ".join([str(x) for x in array.flatten()])
    result.append(f"{shape_dtype}={values}")
  result = "\n".join(result)
  if artifacts_dir is not None:
    inputs_path = os.path.join(artifacts_dir, "inputs.txt")
    logging.info("Saving IREE input values to: %s", inputs_path)
    with open(inputs_path, "w") as f:
      f.write(result)
      f.write("\n")
  return result


def check_same(ref: Any, tar: Any, rtol: float,
               atol: float) -> Tuple[bool, Union[str, None]]:
  """Checks that ref and tar have identical datastructures and values."""
  # Check for matching types.
  if not isinstance(tar, type(ref)):
    error = ("Expected ref and tar to have the same type but got "
             f"'{type(ref)}' and '{type(tar)}'")
    logging.error(error)
    return False, error

  if ref is None:
    # Nothing to compare (e.g. the called method had no outputs).
    return True, None

  # Recursive check for dicts.
  if isinstance(ref, dict):
    if ref.keys() != tar.keys():
      error = ("Expected ref and tar to have the same keys, but got "
               f"'{ref.keys()}' and '{tar.keys()}'")
      logging.error(error)
      return False, error
    # Check that all of the dictionaries' values are the same.
    for key in ref:
      same, error = check_same(ref[key], tar[key], rtol, atol)
      if not same:
        return same, error

  # Recursive check for iterables.
  elif isinstance(ref, list) or isinstance(ref, tuple):
    if len(ref) != len(tar):
      error = ("Expected ref and tar to have the same length, but got "
               f"{len(ref)} and {len(tar)}")
      logging.error(error)
      return False, error
    # Check that all of the iterables' values are the same.
    for i in range(len(ref)):
      same, error = check_same(ref[i], tar[i], rtol, atol)
      if not same:
        return same, error

  # Base check for numpy arrays.
  elif isinstance(ref, np.ndarray):
    if ref.dtype != tar.dtype:
      error = ("Expected ref and tar to have the same dtype, but got "
               f"'{ref.dtype}' and '{tar.dtype}'")
      logging.error(error)
      return False, error
    if ref.size == tar.size == 0:
      return True, None

    if np.issubdtype(ref.dtype, np.floating):
      same = np.allclose(ref, tar, rtol=rtol, atol=atol, equal_nan=True)
      abs_diff = np.max(np.abs(ref - tar))
      rel_diff = np.max(np.abs(ref - tar) / np.max(np.abs(tar)))
      diff_string = (f"Max abs diff: {abs_diff:.2e}, atol: {atol:.2e}, "
                     f"max relative diff: {rel_diff:.2e}, rtol: {rtol:.2e}")
      if not same:
        error = ("Floating point difference between ref and tar was too "
                 f"large. {diff_string}")
        logging.error(error)
      else:
        error = None
        logging.info(
            "Floating point difference between ref and tar was within "
            "tolerance. %s", diff_string)
      return same, error
    elif np.issubdtype(ref.dtype, np.integer):
      same = np.array_equal(ref, tar)
      if not same:
        abs_diff = np.max(np.abs(ref - tar))
        error = ("Expected array equality between ref and tar, but got "
                 f"a max elementwise difference of {abs_diff}")
        logging.error(error)
      else:
        error = None
      return same, error
    else:
      return np.array_equal(ref, tar), None

  # Base check for native number types.
  elif isinstance(ref, (int, float)):
    return ref == tar, None

  # If outputs end up here then an extra branch for that type should be added.
  else:
    raise TypeError(f"Encountered results with unexpected type {type(ref)}")
  return True, None
