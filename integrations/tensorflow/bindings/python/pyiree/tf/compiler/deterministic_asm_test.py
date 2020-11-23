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

import random

from pyiree.tf import compiler
import numpy as np
import tensorflow as tf


def set_random_seed(seed: int = 0) -> None:
  """Set random seed for tf, np and random."""
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


class StatelessModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def add(self, a, b):
    return tf.tanh(a + b)


class DeterministicTest(tf.test.TestCase):

  def test_tf_asm_same_module(self):
    module = StatelessModule()
    module = compiler.tf_module_to_compiler_module(module, pass_pipeline=())
    self.assertEqual(compiler_module.to_asm(), compiler_module.to_asm())

  def test_tf_asm_with_seed(self):
    set_random_seed()
    module_1 = StatelessModule()
    set_random_seed()
    module_2 = StatelessModule()

    module_1 = compiler.tf_module_to_compiler_module(module_1, pass_pipeline=())
    module_2 = compiler.tf_module_to_compiler_module(module_2, pass_pipeline=())
    self.assertEqual(module_1.to_asm(), module_2.to_asm())

  def test_tf_asm_without_seed(self):
    module_1 = StatelessModule()
    module_2 = StatelessModule()
    module_1 = compiler.tf_module_to_compiler_module(module_1, pass_pipeline=())
    module_2 = compiler.tf_module_to_compiler_module(module_2, pass_pipeline=())
    self.assertEqual(module_1.to_asm(), module_2.to_asm())


if __name__ == "__main__":
  tf.test.main()
