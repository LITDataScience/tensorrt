#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
COCO_API_DIR="${SCRIPT_DIR}/../third_party/cocoapi"
PYCOCO_DIR="${COCO_API_DIR}/PythonAPI"

if [[ ! -d "${PYCOCO_DIR}" ]]; then
    echo "ERROR: COCO API sources missing at \`${PYCOCO_DIR}\`."
    echo "Initialize the git submodule first:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo "Install PyBind11 and UJson (pinned for supply-chain hygiene)"
# Pins are minimum compatible floors used by this example suite; bump deliberately.
pip install "pybind11>=2.10,<3" "ujson>=5.7,<6"

echo "Install Cython..."
pip install "Cython>=0.29,<4"

echo "Install cocodataset/cocoapi/PythonAPI..."
pushd "${PYCOCO_DIR}"
python setup.py build_ext --inplace
make
python setup.py install
popd
