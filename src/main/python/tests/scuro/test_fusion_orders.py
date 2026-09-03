# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

import unittest
import numpy as np

from systemds.scuro import Concatenation, RowMax, Hadamard
from systemds.scuro.representations.average import Average
from tests.scuro.data_generator import ModalityRandomDataGenerator
from systemds.scuro.modality.type import ModalityType


class TestFusionOrders(unittest.TestCase):
    """
    The interesting content is the table below rather than the call sequence:
    which operator is commutative, whose result depends on the order of a
    pairwise chain, and where a pairwise chain equals the n-ary form. Written as
    a table those differences are visible at a glance and a new operator is one
    line.
    """

    # (operator, chain_order_independent, chain_equals_nary)
    # Commutativity is not listed: every Fusion operator declares a
    # "commutative" attribute, so the test compares the measured behaviour
    # against that declaration instead of against a second copy of it. A new
    # operator whose declaration contradicts its implementation fails here
    # without anyone having to remember to extend this table.
    # Combining a pair is never the same as combining all three, so that case
    # is asserted for every operator instead of being listed here.
    FUSION_PROPERTIES = [
        (Average, True, False),
        (Concatenation, False, True),
        (RowMax, True, True),
        (Hadamard, True, True),
    ]

    @classmethod
    def setUpClass(cls):
        # The properties under test hold for any input shape.
        cls.num_instances = 4
        cls.num_features = 8
        cls.data_generator = ModalityRandomDataGenerator()

    def setUp(self):
        self.r_1 = self.data_generator.create1DModality(
            self.num_instances, self.num_features, ModalityType.AUDIO
        )
        self.r_2 = self.data_generator.create1DModality(
            self.num_instances, self.num_features, ModalityType.TEXT
        )
        self.r_3 = self.data_generator.create1DModality(
            self.num_instances, self.num_features, ModalityType.TEXT
        )

    @staticmethod
    def _equal(left, right):
        return np.array_equal(np.asarray(left.data), np.asarray(right.data))

    def test_fusion_order_properties(self):
        for (
            fusion_operator,
            chain_order_independent,
            chain_equals_nary,
        ) in self.FUSION_PROPERTIES:
            with self.subTest(fusion=fusion_operator.__name__):
                r_1_r_2 = self.r_1.combine(self.r_2, fusion_operator())
                r_2_r_1 = self.r_2.combine(self.r_1, fusion_operator())
                r_1_r_2_r_3 = r_1_r_2.combine(self.r_3, fusion_operator())
                r_2_r_1_r_3 = r_2_r_1.combine(self.r_3, fusion_operator())
                r1_r2_r3 = self.r_1.combine([self.r_2, self.r_3], fusion_operator())

                self.assertEqual(
                    self._equal(r_1_r_2, r_2_r_1), fusion_operator().commutative
                )
                self.assertEqual(
                    self._equal(r_1_r_2_r_3, r_2_r_1_r_3), chain_order_independent
                )
                self.assertEqual(self._equal(r_1_r_2_r_3, r1_r2_r3), chain_equals_nary)
                self.assertFalse(self._equal(r_1_r_2, r1_r2_r3))
