#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""CLI: compute a deterministic SHA-256 over a SavedModel directory tree.

Example:
  python hash_saved_model.py /path/to/saved_model
  python image_classification.py ... --model_sha256=$(python hash_saved_model.py /path/to/saved_model)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

# Allow import of sibling modules when executed as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from path_utils import sha256_directory  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hash an entire TensorFlow SavedModel directory "
                    "(graph + variables/ + assets/)."
    )
    parser.add_argument(
        "saved_model_dir",
        help="Path to a SavedModel directory containing saved_model.pb "
             "or saved_model.pbtxt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the hex digest (useful for scripting).",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.saved_model_dir):
        parser.error("not a directory: {!r}".format(args.saved_model_dir))

    pb = os.path.join(args.saved_model_dir, "saved_model.pb")
    pbtxt = os.path.join(args.saved_model_dir, "saved_model.pbtxt")
    if not os.path.isfile(pb) and not os.path.isfile(pbtxt):
        parser.error(
            "No saved_model.pb or saved_model.pbtxt under {!r}"
            .format(args.saved_model_dir)
        )

    digest = sha256_directory(args.saved_model_dir)
    if args.quiet:
        print(digest)
    else:
        print("SavedModel: {}".format(os.path.realpath(args.saved_model_dir)))
        print("sha256:     {}".format(digest))
        print()
        print("Pass to benchmarks:")
        print("  --model_sha256={}".format(digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
