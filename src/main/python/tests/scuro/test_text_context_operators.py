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
from systemds.scuro.representations.text_context import (
    SentenceBoundarySplit,
    OverlappingSplit,
)
from systemds.scuro.representations.text_context_with_indices import (
    SentenceBoundarySplitIndices,
    OverlappingSplitIndices,
)
from tests.scuro.data_generator import TestDataLoader
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.modality.type import ModalityType


class TestTextContextOperator(unittest.TestCase):
    """
    The input is fixed so that the exact chunk boundaries and character spans
    can be written down. With randomly generated sentences the only assertable
    properties are invariants ("a chunk has at most max_words", "consecutive
    chunks share their first/last words"), and those pass for a large family of
    wrong implementations.
    """

    # 3 sentences, 5 words each, 77 characters
    THREE_SENTENCES = (
        "The cat reads the document. A dog writes the code. The bird studies the data."
    )
    # 1 sentence, 5 words, 27 characters - stays below max_words for every case
    ONE_SENTENCE = "The cat reads the document."

    SENTENCE_MAX_WORDS = 10
    SENTENCE_MIN_WORDS = 4
    # sentence 1 + 2 fill the 10 word budget, sentence 3 starts a new chunk
    EXPECTED_SENTENCE_CHUNKS = [
        [
            "The cat reads the document. A dog writes the code.",
            "The bird studies the data.",
        ],
        [ONE_SENTENCE],
    ]

    OVERLAP_MAX_WORDS = 6
    OVERLAP = 0.5  # -> stride of 3 words, i.e. 3 words shared per chunk pair
    EXPECTED_OVERLAPPING_CHUNKS = [
        [
            "The cat reads the document. A",
            "the document. A dog writes the",
            "dog writes the code. The bird",
            "code. The bird studies the data.",
        ],
        [ONE_SENTENCE],
    ]

    def setUp(self):
        # Rebuilt for every test: the *Indices operators write "text_spans"
        # into the modality metadata, so a class level modality would leak the
        # spans of one test into the next one (test order is alphabetical).
        self.texts = [self.THREE_SENTENCES, self.ONE_SENTENCE]
        metadata = [
            ModalityType.TEXT.create_metadata(len(text), text) for text in self.texts
        ]
        self.text_modality = UnimodalModality(
            TestDataLoader(
                list(range(len(self.texts))),
                None,
                ModalityType.TEXT,
                list(self.texts),
                str,
                metadata,
            )
        )
        self.text_modality.extract_raw_data()

    def _spans(self):
        return [metadata["text_spans"] for metadata in self.text_modality.metadata]

    def _sliced_by_spans(self):
        return [
            [text[start:end] for start, end in spans]
            for text, spans in zip(self.text_modality.data, self._spans())
        ]

    def test_sentence_boundary_split(self):
        chunks = SentenceBoundarySplit(
            self.SENTENCE_MAX_WORDS, min_words=self.SENTENCE_MIN_WORDS
        ).execute(self.text_modality)

        self.assertEqual(chunks, self.EXPECTED_SENTENCE_CHUNKS)

    def test_overlapping_split(self):
        chunks = OverlappingSplit(self.OVERLAP_MAX_WORDS, self.OVERLAP).execute(
            self.text_modality
        )

        self.assertEqual(chunks, self.EXPECTED_OVERLAPPING_CHUNKS)

    def test_sentence_boundary_split_indices(self):
        SentenceBoundarySplitIndices(
            self.SENTENCE_MAX_WORDS, min_words=self.SENTENCE_MIN_WORDS
        ).execute(self.text_modality)

        self.assertEqual(self._spans(), [[(0, 50), (51, 77)], [(0, 27)]])
        # the spans have to cut the original text into the same chunks the
        # string returning variant produces
        self.assertEqual(self._sliced_by_spans(), self.EXPECTED_SENTENCE_CHUNKS)

    def test_overlapping_split_indices(self):
        OverlappingSplitIndices(self.OVERLAP_MAX_WORDS, self.OVERLAP).execute(
            self.text_modality
        )

        self.assertEqual(
            self._spans(), [[(0, 29), (14, 44), (30, 59), (45, 77)], [(0, 27)]]
        )
        self.assertEqual(self._sliced_by_spans(), self.EXPECTED_OVERLAPPING_CHUNKS)


if __name__ == "__main__":
    unittest.main()
