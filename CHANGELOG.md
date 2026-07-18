# Changelog

All notable changes to this repository are documented here.

Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/) for the `tftrt` package metadata.

Repository: [LITDataScience/tensorrt](https://github.com/LITDataScience/tensorrt)

---

## [0.1.0] — 2026-07-18

### Added

- **`TENSORRT_DOC.md`** — interactive user & creator guide (persona paths, Mermaid flows, precision matrix, security playbook).
- **`CHANGELOG.md`** — this file.
- **`CRITIQUE.md`**, **`VULNERABILITY.md`**, **`IMPROVEMENTS.md`** — security audit, self-review, and research roadmap.
- **`tftrt/examples/path_utils.py`** — `safe_join_under`, `sha256_directory`, SavedModel integrity verify (`hmac.compare_digest`).
- **`tftrt/examples/hash_saved_model.py`** — CLI to compute full-tree SavedModel digests for `--model_sha256`.
- **`--model_sha256`** flag on the shared benchmark CLI (`benchmark_args.py` / `benchmark_runner.py`).
- Unit tests: `tftrt/examples/tests/test_path_utils.py` (path traversal + variable-tamper detection).
- Transformers example README (`tftrt/examples/transformers/README.md`).
- Package metadata bump to **0.1.0** in `setup.py` (was `0.0`).

### Security

- Removed shell **`eval`** from all `*/scripts/base_script.sh` launchers; execute via argv arrays + env exports under `set -euo pipefail`.
- Replaced fixed `/tmp/tmp_detection_results` and `/tmp/$RANDOM` with `tempfile.mkdtemp` / `mktemp -d`.
- COCO annotation `file_name` path traversal blocked via `safe_join_under`.
- Notebook model download switched to **HTTPS**.
- Dependency installer fails closed if cocoapi submodule is missing; pins version floors.
- OD launcher: `set -e`-safe pycocotools probe; validate `--model_name` / `--batch_size` before path join.

### Fixed

- `object_detection/__init__.py` no longer imports deleted symbols.
- Image classification / notebook bare `except:` narrowed to `tf.errors.InvalidArgumentError`.
- Object detection & image classification READMEs: correct `--input_saved_model_dir` flag names.

### Changed

- Root `README.md` rewritten as a modern front door into `TENSORRT_DOC.md`.
- Security warning printed only when `--model_sha256` is omitted (not after a successful verify).

---

## [Unreleased]

Items tracked in [`IMPROVEMENTS.md`](IMPROVEMENTS.md) (MLPerf timing, ONNX/TensorRT dual path, FP8, CI, SLSA).
