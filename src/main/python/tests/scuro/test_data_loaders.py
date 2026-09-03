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

import shutil
import unittest
import numpy as np

from tests.scuro.data_generator import setup_data
from systemds.scuro.dataloader.audio_loader import AudioLoader, AudioStats
from systemds.scuro.dataloader.image_loader import ImageLoader, ImageStats
from systemds.scuro.dataloader.text_loader import TextLoader, TextStats
from systemds.scuro.dataloader.video_loader import VideoLoader, VideoStats
from systemds.scuro.modality.type import ModalityType


class TestDataLoadersLoadFromFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file_path = "test_data"
        cls.num_instances = 2
        cls.mods = [
            ModalityType.AUDIO,
            ModalityType.VIDEO,
            ModalityType.TEXT,
            ModalityType.IMAGE,
            ModalityType.TIMESERIES,
        ]
        cls.data_generator = setup_data(cls.mods, cls.num_instances, cls.test_file_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_file_path)

    # Loading is the same contract for every loader -- one array plus one
    # metadata entry per instance, at the dimensionality that modality has --
    # so the three cases only differed in the loader class and the expected
    # ndim. The stats tests below stay separate: each stats class exposes
    # different fields, so there is no shared assertion to parameterise.
    LOADERS_AND_DIMENSIONS = [
        (AudioLoader, ModalityType.AUDIO, 1),
        (VideoLoader, ModalityType.VIDEO, 4),
        (ImageLoader, ModalityType.IMAGE, 3),
    ]

    def test_loaders_load_all_instances(self):
        for loader_class, modality_type, expected_ndim in self.LOADERS_AND_DIMENSIONS:
            with self.subTest(loader=loader_class.__name__):
                loader = loader_class(
                    self.data_generator.get_modality_path(modality_type),
                    self.data_generator.indices,
                )
                data, metadata = loader.load()

                self.assertEqual(len(data), self.num_instances)
                self.assertEqual(len(metadata), self.num_instances)

                for arr in data:
                    self.assertIsInstance(arr, np.ndarray)
                    self.assertEqual(arr.ndim, expected_ndim)

    def test_audio_loader_stats(self):
        loader = AudioLoader(
            self.data_generator.get_modality_path(ModalityType.AUDIO),
            self.data_generator.indices,
        )
        stats = loader.stats

        self.assertIsInstance(stats, AudioStats)
        self.assertEqual(stats.num_instances, 2)
        self.assertEqual(stats.sampling_rate, 22050)
        self.assertEqual(stats.max_length, 44100)
        self.assertAlmostEqual(stats.avg_length, (44100 * 2) / 2.0)

    def test_video_loader_stats(self):
        loader = VideoLoader(
            self.data_generator.get_modality_path(ModalityType.VIDEO),
            self.data_generator.indices,
        )
        stats = loader.stats

        self.assertIsInstance(stats, VideoStats)
        self.assertEqual(stats.num_instances, 2)
        self.assertEqual(stats.max_length, 60)
        self.assertEqual(stats.max_width, 160)
        self.assertEqual(stats.max_height, 120)
        self.assertEqual(stats.max_channels, 3)

    def test_text_loader_loads_all_instances(self):
        loader = TextLoader(
            self.data_generator.get_modality_path(ModalityType.TEXT),
            self.data_generator.indices,
        )
        data, metadata = loader.load()

        self.assertEqual(len(data), self.num_instances)
        self.assertEqual(len(metadata), self.num_instances)

        for arr in data:
            self.assertIsInstance(arr, str)

    def test_text_loader_stats(self):
        loader = TextLoader(
            self.data_generator.get_modality_path(ModalityType.TEXT),
            self.data_generator.indices,
        )
        stats = loader.stats

        self.assertIsInstance(stats, TextStats)
        self.assertEqual(stats.num_instances, 2)
        self.assertEqual(stats.max_length, 7)
        self.assertAlmostEqual(stats.avg_length, (7 + 7) / 2.0)

    def test_image_loader_stats(self):
        loader = ImageLoader(
            self.data_generator.get_modality_path(ModalityType.IMAGE),
            self.data_generator.indices,
        )
        stats = loader.stats

        self.assertIsInstance(stats, ImageStats)
        self.assertEqual(stats.num_instances, 2)
        self.assertEqual(stats.max_width, 160)
        self.assertEqual(stats.max_height, 120)
        self.assertEqual(stats.max_channels, 3)

    # def test_timeseries_loader_loads_all_instances(self):
    #     loader = TimeseriesLoader(
    #         self.data_generator.get_modality_path(ModalityType.TIMESERIES),
    #         self.data_generator.indices,
    #     )
    #     data, metadata = loader.load()

    #     self.assertEqual(len(data), self.num_instances)
    #     self.assertEqual(len(metadata), self.num_instances)

    #     for arr in data:
    #         self.assertIsInstance(arr, np.ndarray)
    #         self.assertEqual(arr.ndim, 2)

    # def test_timeseries_loader_stats(self):
    #     loader = TimeseriesLoader(
    #         self.data_generator.get_modality_path(ModalityType.TIMESERIES),
    #         self.data_generator.indices,
    #     )
    #     stats = loader.stats

    #     self.assertIsInstance(stats, TimeseriesStats)
    #     self.assertEqual(stats.num_instances, 2)
    #     self.assertEqual(stats.max_length, 10)
    #     self.assertEqual(stats.num_signals, 2)


if __name__ == "__main__":
    unittest.main()
