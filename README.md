# TF-TRT Examples — Hardened TensorFlow ↔ TensorRT Suite

<p align="center">
  <a href="https://github.com/LITDataScience/tensorrt"><img alt="GitHub" src="https://img.shields.io/badge/repo-LITDataScience%2Ftensorrt-181717?logo=github" /></a>
  <a href="https://github.com/LITDataScience/tensorrt/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/LITDataScience/tensorrt?style=social" /></a>
  <a href="CHANGELOG.md"><img alt="Changelog" src="https://img.shields.io/badge/changelog-0.1.0-76b900" /></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>
  <a href="VULNERABILITY.md"><img alt="Security" src="https://img.shields.io/badge/security-hardened-success" /></a>
</p>

**Benchmark and verify TensorRT-accelerated TensorFlow inference** for ImageNet classification, COCO detection, and Hugging Face transformers — with launchers that don’t `eval` your CLI, path-safe data loading, and optional full-tree SavedModel integrity checks.

> **Start here → [`TENSORRT_DOC.md`](TENSORRT_DOC.md)** (interactive user & creator guide)

If this repo helps you ship faster inference, **[⭐ star it](https://github.com/LITDataScience/tensorrt)** so other engineers find the hardened fork.

---

## Why this repo

| | Classic TF-TRT demos | **This suite** |
|--|---------------------|----------------|
| Shell wrappers | Often `eval` user strings | Argv arrays, `set -euo pipefail` |
| Model trust | Load anything | Optional `--model_sha256` (entire tree) |
| Docs | Sparse READMEs | Interactive [`TENSORRT_DOC.md`](TENSORRT_DOC.md) |
| Security | Undocumented | [`VULNERABILITY.md`](VULNERABILITY.md) + tests |
| Roadmap | — | [`IMPROVEMENTS.md`](IMPROVEMENTS.md) |

Upstream TF-TRT concepts: [NVIDIA TF-TRT User Guide](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html).

---

## Quickstart

```bash
git clone https://github.com/LITDataScience/tensorrt.git
cd tensorrt
git submodule update --init --recursive
pip install -e .

# No-GPU sanity
python -m unittest discover -s tftrt/examples/tests -v

# ImageNet + TF-TRT FP16 (needs GPU, data, SavedModel layout)
cd tftrt/examples/image_classification
./scripts/resnet_v1_50.sh \
  --data_dir=/data/imagenet/train-val-tfrecord \
  --input_saved_model_dir=/models \
  --use_tftrt --precision=FP16
```

Hash a SavedModel before load:

```bash
python tftrt/examples/hash_saved_model.py /path/to/SavedModel
# then pass --model_sha256=<digest> to any benchmark
```

---

## Examples

| Task | Path | Guide |
|------|------|-------|
| Image classification | [`tftrt/examples/image_classification`](tftrt/examples/image_classification) | [DOC §4.1](TENSORRT_DOC.md#41-image-classification-imagenet) |
| Object detection | [`tftrt/examples/object_detection`](tftrt/examples/object_detection) | [DOC §4.2](TENSORRT_DOC.md#42-object-detection-coco) |
| Transformers | [`tftrt/examples/transformers`](tftrt/examples/transformers) | [DOC §4.3](TENSORRT_DOC.md#43-transformers-bert--bart) |
| GTC notebooks | [`tftrt/examples/presentations`](tftrt/examples/presentations) | Dynamic-shape demos |

---

## Docs index

| Document | Audience |
|----------|----------|
| **[`TENSORRT_DOC.md`](TENSORRT_DOC.md)** | Everyone — interactive paths, precision matrix, CLI, creators |
| [`CHANGELOG.md`](CHANGELOG.md) | What shipped in 0.1.0 |
| [`VULNERABILITY.md`](VULNERABILITY.md) | Security findings & mitigations |
| [`CRITIQUE.md`](CRITIQUE.md) | Self-review of the hardening PR |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | SOTA roadmap (MLPerf, FP8, TensorRT-LLM, …) |

---

## Install notes

- **TF-TRT** ships with current TensorFlow GPU / NGC containers — you typically do **not** install a separate `tftrt` runtime wheel beyond this examples package.
- **TensorRT** must be present (NGC image or [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)).
- Jetson: use NVIDIA’s Jetson TensorFlow packages — see [NVIDIA frameworks for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html).

```bash
pip install -e .   # installs local helpers (version 0.1.0)
```

---

## Contributing / starring

- New models: follow the [Creator guide](TENSORRT_DOC.md#9-creator-guide-extend-the-suite).
- Security: never reintroduce `eval` in launchers; add tests under `tftrt/examples/tests/`.
- Love the hardening work? **[Star the repo](https://github.com/LITDataScience/tensorrt/stargazers)** and open issues with your GPU + TF/TRT versions.

---

## License

[Apache License 2.0](LICENSE)
