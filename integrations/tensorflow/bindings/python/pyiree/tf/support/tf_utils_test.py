# Lint as: python3
# Copyright 2020 Google LLC
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
"""Tests for pyiree.tf.support.tf_utils."""

from absl.testing import parameterized
import numpy as np
from pyiree.tf.support import tf_utils
import tensorflow as tf


class UtilsTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([('int8_to_i8', np.int8, 'i8'),
                                   ('int32_to_i32', np.int32, 'i32'),
                                   ('float32_to_f32', np.float32, 'f32'),
                                   ('float64_to_f64', np.float64, 'f64')])
  def test_to_mlir_type(self, numpy_type, mlir_type):
    self.assertEqual(tf_utils.to_mlir_type(numpy_type), mlir_type)

  @parameterized.named_parameters([
      ('single_i32', [np.array([1, 2], dtype=np.int32)], '2xi32=1 2'),
      ('single_f32', [np.array([1, 2], dtype=np.float32)], '2xf32=1.0 2.0'),
  ])
  def test_save_input_values(self, inputs, inputs_str):
    self.assertEqual(tf_utils.save_input_values(inputs), inputs_str)

  def test_apply_function(self):
    inputs = [1, [2, 3], (4, 5), {'6': 6, '78': [7, 8]}]
    expected = [0, [1, 2], (3, 4), {'6': 5, '78': [6, 7]}]
    result = tf_utils.apply_function(inputs, lambda x: x - 1)
    self.assertEqual(result, expected)
    self.assertNotEqual(inputs, expected)

  @parameterized.named_parameters([
      {
          'testcase_name': 'all the same',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tar_same': True,
      },
      {
          'testcase_name': 'wrong int',
          'array_c': np.array([1, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tar_same': False,
      },
      {
          'testcase_name': 'wrong string',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['a', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tar_same': False,
      },
      {
          'testcase_name': 'wrong float',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([1.0, 0.1, 0.2]),
          'tar_same': False,
      },
  ])
  def test_recursive_check_same(self, array_c, array_d, array_e, tar_same):

    # yapf: disable
    ref = {
        'a': 1,
        'b': [
            {'c': np.array([0, 1, 2])},
            {'d': np.array(['0', '1', '2'])},
            {'e': np.array([0.0, 0.1, 0.2])}
        ],
    }
    tar = {
        'a': 1,
        'b': [
            {'c': array_c},
            {'d': array_d},
            {'e': array_e}
        ],
    }
    # yapf: enable
    same, _ = tf_utils.check_same(ref, tar, rtol=1e-6, atol=1e-6)
    self.assertEqual(tar_same, same)


if __name__ == '__main__':
  tf.test.main()
