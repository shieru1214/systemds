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

# TODO: Test edge cases: unequal number of audio-video timestamps (should still work and add the average over all audio/video samples)

import copy
import unittest

import numpy as np

from systemds.scuro.modality.joined import JoinCondition, JoinedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from tests.scuro.data_generator import ModalityRandomDataGenerator, TestDataLoader


class SpyRepresentation(UnimodalRepresentation):
    """
    Cheap stand-in for a real representation.

    Whether a representation is a neural network or a summary statistic is
    irrelevant to the join: it only has to turn every element into one row. A
    deterministic stand-in keeps the join behaviour comparable across
    configurations and keeps these tests off the torch path.
    """

    def __init__(self):
        super().__init__("Spy", ModalityType.EMBEDDING)
        self.call_count = 0
        self.seen_element_counts = []

    def transform(self, modality, aggregation=None):
        self.call_count += 1
        self.seen_element_counts.append([len(instance) for instance in modality.data])
        transformed = TransformedModality(modality, self, self.output_modality_type)
        transformed.data = [
            np.stack([self._embed(element) for element in instance])
            for instance in modality.data
        ]
        return transformed

    @staticmethod
    def _embed(element):
        values = np.asarray(element, dtype=np.float32).reshape(-1)
        return np.array(
            [values.mean(), values.min(), values.max(), float(values.size)],
            dtype=np.float32,
        )


def _modality_with_timestamps(modality_type, instances, timestamps, metadata_args):
    """
    Build a modality whose timestamps are written by hand, one entry per list.

    Modality.data is a property: assigning it recomputes the timestamps from the
    frequency stored in the metadata, so the timestamps have to be overwritten
    afterwards. The assertion at the end guards that order - swap the two and
    the generated timestamps would silently win.
    """
    modality = Modality(
        modality_type,
        0,
        [modality_type.create_metadata(*args) for args in metadata_args],
        np.float32,
    )
    modality.data = list(instances)
    for metadata, stamps in zip(modality.metadata, timestamps):
        metadata["timestamp"] = np.asarray(stamps)

    for metadata, stamps in zip(modality.metadata, timestamps):
        assert np.array_equal(metadata["timestamp"], np.asarray(stamps)), (
            "timestamps were recomputed - Modality.data has to be assigned "
            "before the timestamps are written, not after"
        )
    return modality


def _video_modality(frame_timestamps):
    """frame_timestamps: one list of frame timestamps per instance."""
    return _modality_with_timestamps(
        ModalityType.VIDEO,
        [np.zeros((len(stamps), 2), dtype=np.float32) for stamps in frame_timestamps],
        frame_timestamps,
        [(30, len(stamps), 2, 2, 3) for stamps in frame_timestamps],
    )


def _audio_modality(rows, row_timestamps):
    """rows / row_timestamps: one entry per instance."""
    return _modality_with_timestamps(
        ModalityType.AUDIO,
        rows,
        row_timestamps,
        [(1, np.zeros(len(stamps), dtype=np.float32)) for stamps in row_timestamps],
    )


class TestJoinMapping(unittest.TestCase):
    """
    Unit tests for the join mapping itself: fixed timestamps, no representation.
    These are the tests that pin down which right hand rows end up under which
    left hand frame.
    """

    # Two instances so that a mix-up between them shows: instance 0 carries the
    # row values 0..5, instance 1 carries 100..102.
    LEFT_TIMESTAMPS = [[0, 10, 20], [0, 5]]
    RIGHT_TIMESTAMPS = [[0, 1, 5, 11, 12, 25], [0, 3, 7]]
    RIGHT_VALUES = [[0, 1, 2, 3, 4, 5], [100, 101, 102]]

    def setUp(self):
        self.right_rows = [
            np.array([[value, value] for value in values], dtype=np.float32)
            for values in self.RIGHT_VALUES
        ]
        joined = JoinedModality(
            ModalityType.VIDEO,
            _video_modality(self.LEFT_TIMESTAMPS),
            _audio_modality(self.right_rows, self.RIGHT_TIMESTAMPS),
            JoinCondition("timestamp", "timestamp", "<"),
        )
        joined.execute()
        self.blocks = [
            [np.asarray(block) for block in instance]
            for instance in joined.joined_right.data
        ]

    def _rows(self, instance, *indices):
        return self.right_rows[instance][list(indices)]

    def test_join_assigns_every_left_frame_a_block(self):
        self.assertEqual(len(self.blocks), len(self.LEFT_TIMESTAMPS))
        for instance, frame_timestamps in enumerate(self.LEFT_TIMESTAMPS):
            self.assertEqual(len(self.blocks[instance]), len(frame_timestamps))

    def test_join_maps_right_samples_before_the_next_left_frame(self):
        # instance 0: frames at t = 0, 10, 20 over rows at t = 0, 1, 5, 11, 12, 25
        np.testing.assert_array_equal(self.blocks[0][0], self._rows(0, 0, 1, 2))
        np.testing.assert_array_equal(self.blocks[0][1], self._rows(0, 3, 4))
        # the last frame covers everything from 20 on -> the row at t = 25.
        # Regression: the final right row has to be reachable, the last frame
        # must not fall back to a copy of the previous one.
        np.testing.assert_array_equal(self.blocks[0][2], self._rows(0, 5))

    def test_join_keeps_instances_apart(self):
        # instance 1: frames at t = 0, 5 over rows at t = 0, 3, 7. The values are
        # from the 100 range, so borrowing a row from instance 0 would show.
        np.testing.assert_array_equal(self.blocks[1][0], self._rows(1, 0, 1))
        np.testing.assert_array_equal(self.blocks[1][1], self._rows(1, 2))

    def test_join_without_matching_samples_falls_back_to_the_instance_average(self):
        # A left frame that no right sample falls into gets the average over all
        # right rows of that instance, as described by the TODO at the top of
        # this file. Regression: an empty match must not produce NaN.
        # every right row lies before the first left frame, so frame 1 matches
        # nothing at all
        right_rows = np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32)
        joined = JoinedModality(
            ModalityType.VIDEO,
            _video_modality([[0, 100]]),
            _audio_modality([right_rows], [[0, 1]]),
            JoinCondition("timestamp", "timestamp", "<"),
        )
        joined.execute()
        blocks = [np.asarray(block) for block in joined.joined_right.data[0]]

        np.testing.assert_array_equal(blocks[0], right_rows)
        np.testing.assert_array_equal(blocks[1], np.array([[3.0, 3.0]]))
        self.assertFalse(np.isnan(blocks[1]).any())

    def test_chunked_execution_offsets_into_the_right_modality(self):
        # execute(starting_idx) is the index arithmetic chunked runs depend on:
        # the left modality holds one chunk while the right one holds every
        # instance, so the chunk has to be paired with the right instances that
        # start at starting_idx. Off by one here pairs instance A's video with
        # instance B's audio, silently and with the expected shapes.
        right_rows = [
            np.array([[10 * instance, 10 * instance]], dtype=np.float32)
            for instance in range(4)
        ]
        joined = JoinedModality(
            ModalityType.VIDEO,
            _video_modality([[0, 10], [0, 10]]),
            _audio_modality(right_rows, [[0]] * 4),
            JoinCondition("timestamp", "timestamp", "<"),
        )
        joined.chunked_execution = True
        joined.chunk_left = True

        joined.execute(starting_idx=2)

        # the chunk is instances 0 and 1 of the left modality, so it must pick up
        # right instances 2 and 3, i.e. the rows carrying 20 and 30
        for chunk_position, right_instance in enumerate([2, 3]):
            with self.subTest(chunk_position=chunk_position):
                for block in joined.joined_right.data[chunk_position]:
                    np.testing.assert_array_equal(
                        np.asarray(block), right_rows[right_instance]
                    )

    def test_equality_join_maps_rows_with_matching_timestamps(self):
        # Covers the branch taken for join types other than "<", which had no
        # test at all and could not run: it called .append() on a numpy array.
        right_timestamps = [0, 10, 10, 20, 30, 40]
        right_rows = np.array(
            [[value, value] for value in range(len(right_timestamps))],
            dtype=np.float32,
        )
        joined = JoinedModality(
            ModalityType.VIDEO,
            _video_modality([[0, 10, 20]]),
            _audio_modality([right_rows], [right_timestamps]),
            JoinCondition("timestamp", "timestamp", "=="),
        )
        joined.execute()
        blocks = [np.asarray(block) for block in joined.joined_right.data[0]]

        # every left frame collects the right rows carrying the same timestamp
        np.testing.assert_array_equal(blocks[0], right_rows[[0]])
        np.testing.assert_array_equal(blocks[1], right_rows[[1, 2]])
        np.testing.assert_array_equal(blocks[2], right_rows[[3]])


class TestMultimodalJoin(unittest.TestCase):
    """
    End to end joins over generated data. These check that the pipeline holds
    together and that chunking does not change the result; the mapping itself
    is covered by TestJoinMapping above.
    """

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 4
        cls.indices = np.array(range(cls.num_instances))
        cls.audio_data, cls.audio_md = ModalityRandomDataGenerator().create_audio_data(
            cls.num_instances, 500
        )
        cls.video_data, cls.video_md = (
            ModalityRandomDataGenerator().create_visual_modality(cls.num_instances, 60)
        )

    def _prepare_data(self, l_chunk_size=None, r_chunk_size=None):
        audio = UnimodalModality(
            TestDataLoader(
                self.indices,
                r_chunk_size,
                ModalityType.AUDIO,
                copy.deepcopy(self.audio_data),
                np.float32,
                copy.deepcopy(self.audio_md),
            )
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices,
                l_chunk_size,
                ModalityType.VIDEO,
                copy.deepcopy(self.video_data),
                np.float32,
                copy.deepcopy(self.video_md),
            )
        )
        return video, audio.apply_representation(MelSpectrogram())

    def _join(self, left_modality, right_modality, representation, window_size=2):
        return (
            left_modality.join(
                right_modality, JoinCondition("timestamp", "timestamp", "<")
            )
            .apply_representation(representation)
            .window_aggregation(window_size, "mean")
            .combine("concat")
        )

    def test_video_audio_join(self):
        video, mel_audio = self._prepare_data()
        joined = self._join(video, mel_audio, SpyRepresentation())

        self.assertEqual(len(joined.left_modality.data), self.num_instances)
        self.assertEqual(len(joined.right_modality.data), self.num_instances)
        self.assertEqual(len(joined.data), self.num_instances)

    # TODO
    # def test_chunked_audio_video_join(self):
    #     self._execute_av_join(2)

    # TODO
    # def test_chunked_audio_chunked_video_join(self):
    #     self._execute_av_join(2, 2)

    def test_audio_video_join(self):
        # Audio has a much higher frequency than video, hence we would need to
        # duplicate or interpolate frames to match them to the audio frequency
        video, mel_audio = self._prepare_data()
        joined = self._join(mel_audio, video, SpyRepresentation())

        self.assertEqual(len(joined.left_modality.data), self.num_instances)
        self.assertEqual(len(joined.data), self.num_instances)

    def test_chunked_and_unchunked_joins_agree(self):
        # Chunking is a memory setting, so it must not change the result. Every
        # combination is compared against the unchunked run.
        chunk_configurations = [(None, None), (2, None), (None, 2), (2, 2)]
        results = {}

        for l_chunk_size, r_chunk_size in chunk_configurations:
            video, mel_audio = self._prepare_data(l_chunk_size, r_chunk_size)
            joined = self._join(video, mel_audio, SpyRepresentation())
            results[(l_chunk_size, r_chunk_size)] = [
                np.asarray(instance) for instance in joined.data
            ]

        expected = results[(None, None)]
        for configuration, actual in results.items():
            with self.subTest(chunk_sizes=configuration):
                self.assertEqual(len(actual), len(expected))
                for instance, expected_instance in zip(actual, expected):
                    self.assertEqual(instance.shape, expected_instance.shape)
                    # Regression: an empty match must never produce NaN, so no
                    # equal_nan here.
                    self.assertFalse(np.isnan(instance).any())
                    self.assertTrue(np.allclose(instance, expected_instance))

    def test_join_applies_the_representation_to_both_sides(self):
        video, mel_audio = self._prepare_data()
        spy = SpyRepresentation()

        video.join(
            mel_audio, JoinCondition("timestamp", "timestamp", "<")
        ).apply_representation(spy)

        self.assertEqual(spy.call_count, 2)
        # Both sides have to arrive with one entry per left hand frame: the
        # video frames themselves, and the block of right hand rows the join
        # assigned to each of those frames.
        left_counts, right_counts = spy.seen_element_counts
        self.assertEqual(left_counts, right_counts)
        self.assertEqual(len(left_counts), self.num_instances)

    def test_video_audio_join_with_resnet(self):
        # The one test that runs a real representation end to end, so that a
        # change breaking the torch path is still caught.
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            2, 12
        )
        audio_data, audio_md = ModalityRandomDataGenerator().create_audio_data(2, 500)
        indices = np.array(range(2))

        audio = UnimodalModality(
            TestDataLoader(
                indices, None, ModalityType.AUDIO, audio_data, np.float32, audio_md
            )
        )
        video = UnimodalModality(
            TestDataLoader(
                indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )

        joined = self._join(
            video, audio.apply_representation(MelSpectrogram()), ResNet()
        )

        self.assertEqual(len(joined.data), 2)


if __name__ == "__main__":
    unittest.main()
