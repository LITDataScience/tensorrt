# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Unit tests for path_utils (no TensorFlow dependency)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import sys
import tempfile
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from path_utils import safe_join_under, sha256_file, verify_saved_model_sha256


class SafeJoinUnderTest(unittest.TestCase):

    def test_joins_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "images")
            os.makedirs(nested)
            target = os.path.join(nested, "0001.jpg")
            with open(target, "wb") as handle:
                handle.write(b"x")
            self.assertEqual(
                safe_join_under(nested, "0001.jpg"),
                os.path.realpath(target),
            )

    def test_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "images")
            os.makedirs(nested)
            with self.assertRaises(ValueError):
                safe_join_under(nested, "../secret.txt")

    def test_rejects_absolute_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "images")
            os.makedirs(nested)
            with self.assertRaises(ValueError):
                safe_join_under(nested, "/etc/passwd")


class SavedModelSha256Test(unittest.TestCase):

    def test_verify_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            pb_path = os.path.join(tmp, "saved_model.pb")
            payload = b"fake-saved-model-bytes"
            with open(pb_path, "wb") as handle:
                handle.write(payload)
            digest = hashlib.sha256(payload).hexdigest()
            verified_path, actual = verify_saved_model_sha256(tmp, digest)
            self.assertEqual(verified_path, pb_path)
            self.assertEqual(actual, digest)
            self.assertEqual(sha256_file(pb_path), digest)

            with self.assertRaises(ValueError):
                verify_saved_model_sha256(tmp, "0" * 64)


if __name__ == "__main__":
    unittest.main()
