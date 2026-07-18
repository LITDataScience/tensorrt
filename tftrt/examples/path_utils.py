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
"""Filesystem helpers used by benchmark scripts for path safety."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os


def safe_join_under(base_dir, *paths):
    """Join paths under ``base_dir`` and reject escapes (path traversal).

    Resolves symlinks and ``..`` components. Raises ``ValueError`` if the
    resulting path would fall outside ``base_dir``.
    """
    if not base_dir:
        raise ValueError("base_dir must be a non-empty path")

    base = os.path.realpath(base_dir)
    if not os.path.isdir(base):
        raise ValueError("base_dir does not exist or is not a directory: {!r}"
                         .format(base_dir))

    candidate = os.path.realpath(os.path.join(base, *paths))
    # Exact match (base itself) or a proper subdirectory.
    if candidate != base and not candidate.startswith(base + os.sep):
        raise ValueError(
            "Path {!r} escapes base directory {!r}".format(
                os.path.join(*paths) if paths else "", base_dir
            )
        )
    return candidate


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_saved_model_sha256(saved_model_dir, expected_sha256):
    """Verify integrity of a SavedModel directory against an expected digest.

    Hashes ``saved_model.pb`` when present, otherwise ``saved_model.pbtxt``.
    Raises ``ValueError`` / ``FileNotFoundError`` on mismatch or missing files.
    """
    if not expected_sha256:
        raise ValueError("expected_sha256 must be a non-empty hex digest")

    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(
            "expected_sha256 must be a 64-character hex SHA-256 digest, "
            "got {!r}".format(expected_sha256)
        )

    pb_path = os.path.join(saved_model_dir, "saved_model.pb")
    pbtxt_path = os.path.join(saved_model_dir, "saved_model.pbtxt")
    if os.path.isfile(pb_path):
        target = pb_path
    elif os.path.isfile(pbtxt_path):
        target = pbtxt_path
    else:
        raise FileNotFoundError(
            "No saved_model.pb or saved_model.pbtxt under {!r}"
            .format(saved_model_dir)
        )

    actual = sha256_file(target)
    if actual != expected:
        raise ValueError(
            "SavedModel integrity check failed for {!r}: "
            "expected sha256={}, got {}".format(target, expected, actual)
        )
    return target, actual
