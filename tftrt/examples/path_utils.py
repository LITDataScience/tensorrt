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
import hmac
import os


def safe_join_under(base_dir, *paths):
    """Join paths under ``base_dir`` and reject escapes (path traversal).

    Resolves symlinks and ``..`` components. Raises ``ValueError`` if the
    resulting path would fall outside ``base_dir``.
    """
    if not base_dir:
        raise ValueError("base_dir must be a non-empty path")

    normalized = []
    for part in paths:
        if part is None:
            raise ValueError("path components must not be None")
        if isinstance(part, bytes):
            part = part.decode("utf-8")
        # Explicit reject: os.path.join discards base when given an abs segment.
        if part == "" or "\x00" in part:
            raise ValueError("invalid path component: {!r}".format(part))
        if os.path.isabs(part) or part.startswith("~"):
            raise ValueError(
                "absolute or home-relative path component rejected: {!r}"
                .format(part)
            )
        normalized.append(part)

    base = os.path.realpath(base_dir)
    if not os.path.isdir(base):
        raise ValueError("base_dir does not exist or is not a directory: {!r}"
                         .format(base_dir))

    joined = os.path.join(base, *normalized) if normalized else base
    candidate = os.path.realpath(joined)
    try:
        common = os.path.commonpath([base, candidate])
    except ValueError:
        # Different drives / invalid mix — treat as escape.
        raise ValueError(
            "Path {!r} escapes base directory {!r}".format(
                os.path.join(*normalized) if normalized else "", base_dir
            )
        )

    if os.path.realpath(common) != base:
        raise ValueError(
            "Path {!r} escapes base directory {!r}".format(
                os.path.join(*normalized) if normalized else "", base_dir
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


def sha256_directory(root_dir):
    """Return a deterministic SHA-256 over a directory tree.

    Hashes relative POSIX paths + per-file content digests in sorted walk
    order. Symlinks are rejected so the digest cannot be redirected.
    """
    root = os.path.realpath(root_dir)
    if not os.path.isdir(root):
        raise ValueError("root_dir does not exist or is not a directory: {!r}"
                         .format(root_dir))

    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == os.curdir:
            rel_dir = ""
        for name in filenames:
            rel = name if not rel_dir else os.path.join(rel_dir, name)
            rel_posix = rel.replace(os.sep, "/")
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                raise ValueError(
                    "symlinks are not allowed in SavedModel integrity hash: "
                    "{!r}".format(rel_posix)
                )
            if not os.path.isfile(path):
                raise ValueError(
                    "non-regular file in SavedModel integrity hash: {!r}"
                    .format(rel_posix)
                )
            digest.update(rel_posix.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(path).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _normalize_sha256_hex(expected_sha256):
    if not expected_sha256:
        raise ValueError("expected_sha256 must be a non-empty hex digest")
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(
            "expected_sha256 must be a 64-character hex SHA-256 digest, "
            "got {!r}".format(expected_sha256)
        )
    return expected


def verify_saved_model_sha256(saved_model_dir, expected_sha256):
    """Verify integrity of an entire SavedModel directory.

    Computes :func:`sha256_directory` over ``saved_model_dir`` (graph proto,
    ``variables/``, ``assets/``, etc.) and compares with
    :func:`hmac.compare_digest`.

    Raises ``ValueError`` / ``FileNotFoundError`` on mismatch or missing files.
    """
    expected = _normalize_sha256_hex(expected_sha256)

    if not os.path.isdir(saved_model_dir):
        raise FileNotFoundError(
            "SavedModel directory does not exist: {!r}".format(saved_model_dir)
        )

    pb_path = os.path.join(saved_model_dir, "saved_model.pb")
    pbtxt_path = os.path.join(saved_model_dir, "saved_model.pbtxt")
    if not os.path.isfile(pb_path) and not os.path.isfile(pbtxt_path):
        raise FileNotFoundError(
            "No saved_model.pb or saved_model.pbtxt under {!r}"
            .format(saved_model_dir)
        )

    actual = sha256_directory(saved_model_dir)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            "SavedModel integrity check failed for {!r}: "
            "expected sha256={}, got {}".format(
                saved_model_dir, expected, actual
            )
        )
    return saved_model_dir, actual
