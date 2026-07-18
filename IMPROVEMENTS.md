# World-Class Improvements for the TF-TRT Examples Suite

Research- and industry-backed roadmap to modernize this repository beyond the security fixes in `VULNERABILITY.md`. Prioritized for an NVIDIA-class TF-TRT / TensorRT inference benchmarking codebase.

Sources lean on arXiv, Papers with Code leaderboards, MLPerf, and current NVIDIA software stacks (TensorRT 10.x, CUDA Graphs, TensorRT-LLM, Dynamo).

---

## 1. Inference stack modernization

### 1.1 Dual-path: TF-TRT **and** ONNX → TensorRT
TF-TRT remains valuable for SavedModel workflows, but production SOTA increasingly uses **ONNX export → TensorRT builder** (or TensorRT-LLM for generative models). Adopt a dual harness:
- Keep current `TrtGraphConverterV2` path for regression parity.
- Add ONNX Runtime / Polygraphy / `trtexec` path for apples-to-apples engine builds.

**Why:** Broader op coverage, stronger dynamic-shape tooling, and alignment with NVIDIA deployment docs.

### 1.2 Migrate transformers benches to TensorRT-LLM / NVIDIA Dynamo
BERT/BART synthetic benches under `tftrt/examples/transformers` are useful historically, but LLM/encoder inference SOTA moved to:
- **TensorRT-LLM** (in-flight batching, paged KV cache, FP8)
- **NVIDIA TensorRT Model Optimizer** (PTQ / QAT)
- Speculative decoding research (e.g. Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, arXiv:2211.17192)

**Papers / trackers:**
- Speculative decoding — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192), Papers with Code: *Speculative Decoding*
- SmoothQuant — Xiao et al., arXiv:2211.10438
- GPTQ — Frantar et al., arXiv:2210.17323
- AWQ — Lin et al., arXiv:2306.00978

### 1.3 CUDA Graphs + CUDA MPS for steady-state latency
Capture inference graphs after warmup to cut CPU launch overhead. Pair with careful stream management and (where multi-tenant) MPS.

**Refs:** NVIDIA CUDA Graphs best practices; MLPerf Inference submissions routinely report graph-capture wins for CNN/RNN offline/server scenarios.

### 1.4 FP8 / NVFP4 and calibrated INT8
Extend precision matrix beyond FP32/FP16/INT8:
- Hopper/Blackwell **FP8** (E4M3/E5M2) via TensorRT
- Quantization-aware training (QAT) where PTQ accuracy cliffs appear
- Activation-aware / SmoothQuant-style calibration for transformers

**Papers:** SmoothQuant (arXiv:2211.10438); NVIDIA Transformer Engine docs; MLPerf Inference v4+/v5 quantization notes.

---

## 2. Benchmark science (stop measuring noise)

### 2.1 Align with MLPerf Inference methodology
Current timing mixes warmup, host sync, and percentile stats, but lacks MLPerf-grade:
- LoadGen-style closed/open loop
- Latency constraints (99th %ile server scenario)
- Power / efficiency optional metrics
- Frozen accuracy targets per model

**Ref:** [MLPerf Inference](https://mlcommons.org/benchmarks/inference/) rules + reference implementations.

### 2.2 Proper GPU synchronization & clock policy
`_force_gpu_resync` via a tiny tensor is a start; upgrade to:
- Explicit `tf.experimental.async_scope` / CUDA event timing where available
- Locked GPU/mem clocks for publishable numbers
- Separate **host→device H2D**, **compute**, **D2H** phases (Nsight Systems / NVTX ranges)

### 2.3 Statistical rigor
Report mean ± CI, bootstrap 99th %ile, and reject runs with thermal throttling. Store raw latency vectors (Parquet) for re-analysis.

### 2.4 Accuracy–latency Pareto, not single points
Sweep batch size, sequence length, precision, and `minimum_segment_size`. Emit Pareto fronts (Papers with Code style leaderboard tables).

---

## 3. Data pipeline & systems performance

### 3.1 Modern `tf.data` service / tf.data options
Replace deprecated `tf.data.experimental.map_and_batch` with `map` + `batch`, enable:
- `tf.data.Options().experimental_optimization` / deterministic toggles
- `prefetch(AUTOTUNE)`, `cache()` for calibration subsets
- Optional NVIDIA DALI for GPU decode/resize (common in high-throughput ImageNet pipelines)

**Refs:** Google tf.data papers; DALI user guide; *tf.data: A Machine Learning Data Processing Framework* (arXiv:2101.12127).

### 3.2 I/O locality
Use local NVMe for TFRecords/COCO; document remote FS penalties. Add synthetic *and* real-data modes with identical preprocess graphs so TRT engines stay valid.

---

## 4. Model / graph optimization research to productize

| Technique | Apply to | Key refs |
|-----------|----------|----------|
| Structured / N:M sparsity | ResNet / MobileNet benches | NVIDIA ASP; Hubara et al.; Mishra et al. *Accelerating Sparse DNN* |
| Knowledge distillation | Smaller deploy graphs | Hinton et al.; Park et al. |
| Operator fusion audits | TF-TRT segmenter | TensorRT layer fusion docs |
| Dynamic shape profiles | Detection / NLP | TensorRT optimization profiles |
| Graph rewriting / Grappler passes | Pre-TRT | TF Grappler; XLA |
| FlashAttention-style attention | Transformer path | Dao et al., arXiv:2205.14135; Dao 2023 FlashAttention-2 arXiv:2307.08691 |

Integrate optional **XLA** vs **TF-TRT** vs **XLA+TRT** A/B (already partially flagged via `--use_xla` / `--use_xla_auto_jit`) into a single report matrix.

---

## 5. Software engineering excellence

### 5.1 Packaging & dependency hygiene
- Replace minimal `setup.py` (`version='0.0'`, only `tqdm`) with `pyproject.toml` (PEP 517/621), extras: `[od]`, `[transformers]`, `[dev]`.
- Lock files (`requirements.txt` hashes or `uv.lock` / `poetry.lock`).
- Pin TensorFlow ↔ TensorRT ↔ CUDA compatibility matrix in a table (NGC container tags as source of truth).

### 5.2 Typing, lint, format
- `mypy` / `pyright` on public APIs
- `ruff` + `black` (or `ruff format`)
- Pre-commit hooks

### 5.3 Testing pyramid
Already started: `tftrt/examples/tests/test_path_utils.py`. Expand to:
- CLI validation unit tests (no GPU)
- Golden accuracy smoke on tiny fixtures (CPU/GPU gated markers)
- Bash script tests (`bats` / `shellcheck`)
- Notebooks via `nbconvert --execute` in CI with mocks where GPUs absent

### 5.4 CI/CD
GitHub Actions (or GitLab) matrix:
- `shellcheck`, `bash -n`
- `python -m unittest`
- Optional self-hosted GPU runner for nightly TRT convert smoke
- Dependabot / Renovate for pins
- SARIF upload from `pip-audit` / Trivy

### 5.5 Observability
- Structured JSON logs (latency, precision, GPU UUID, TRT version, git SHA)
- Optional OpenTelemetry traces around convert / calibrate / infer
- Export MLFlow or W&B runs for experiment tracking

---

## 6. Security & supply chain (beyond current mitigations)

See `VULNERABILITY.md` for completed fixes. Next bar:

1. **SLSA / provenance** for published benchmark containers (in-toto attestations).
2. **Sigstore cosign** for SavedModel tarballs + notebook assets.
3. **Model cards + SBOM** (CycloneDX) listing TF/TRT/CUDA hashes.
4. **gVisor / Kata / dedicated user** for untrusted model evaluation.
5. Official `SECURITY.md` with disclosure policy.
6. Drop remaining cleartext image URLs in notebook markdown (`http://developer.download.nvidia.com/...` logos → HTTPS).

**ML security research context:**
- Model poisoning / trojans — Gu et al. BadNets; Papers with Code *Backdoor Attack*
- Adversarial examples under deployment — Goodfellow et al.; Madry et al. arXiv:1706.06083
- Treat SavedModels like binaries (this repo now warns + optional SHA-256)

---

## 7. API / UX redesign

### 7.1 Single entrypoint
```text
tftrt-bench image-classification|object-detection|transformers [options]
```
Replace N thin `.sh` wrappers with one Click/Typer CLI sharing pydantic-validated configs (YAML/JSON).

### 7.2 Config-as-code
Hydra or OmegaConf experiment configs (model, precision, batch, data, TRT params) for reproducible sweeps.

### 7.3 Results schema
Versioned JSON Schema for results; auto-generate Markdown/HTML comparison tables and Plotly latency CDFs.

### 7.4 Notebooks → Quarto / Myst
Executable docs with pinned NGC container tags; separate “tutorial” vs “publishable benchmark” modes.

---

## 8. Hardware & platform coverage

| Target | Improvement |
|--------|-------------|
| Data Center (H100/B200) | FP8, larger workspaces, multi-instance GPU (MIG) benches |
| Jetson / edge | TRT lean engines, DLA offload where applicable |
| Multi-GPU | `tf.distribute` or explicit multi-engine; OR-Tools-free simple data parallel throughput mode |
| CPU fallback CI | Tiny graphs for convert-unit tests without GPU |

Document PCIe vs NVLink effects on host-staged pipelines.

---

## 9. Accuracy & robustness evaluation

1. Full COCO metrics suite (AP@[.5:.95], AR) — already partial via `COCOeval`; persist full `stats[]`.
2. ImageNet top-1 **and** top-5; per-class failure dumps.
3. **Perturbation robustness** sampling (ImageNet-C style, Hendrycks & Dietterich arXiv:1903.12261) to ensure TRT precision modes don’t silently crater under shift.
4. Deterministic seeds for synthetic transformers inputs; save golden logits for bitwise/near-bitwise TRT diffs (`np.allclose` tolerances by precision).

---

## 10. Suggested implementation phases

### Phase A — Foundations (low risk, high leverage)
1. `pyproject.toml` + pinned deps + `shellcheck` CI  
2. Expand unit tests; gate GPU tests with markers  
3. Replace deprecated `tf.data` APIs  
4. Results JSON schema + git metadata capture  

### Phase B — Measurement quality
1. MLPerf-inspired timing modes  
2. NVTX ranges + Nsight recipe docs  
3. Precision × batch Pareto runner  

### Phase C — Stack evolution
1. ONNX/TensorRT parallel path  
2. FP8 + Model Optimizer integration  
3. TensorRT-LLM track for encoder/decoder models  
4. DALI preprocess option  

### Phase D — Hardening & publishing
1. Signed models + SBOM  
2. Public leaderboard generator  
3. NGC-published containers with reproducible digests  

---

## 11. Quick wins already landed in this PR

- Hardened Bash launchers (no `eval`)
- Safe path joins + temp dirs
- Optional SavedModel SHA-256 verification
- HTTPS model download in the SavedModel notebook
- Dependency installer fails closed without cocoapi submodule
- Initial `unittest` suite for filesystem integrity helpers

---

## Key references (start here)

1. NVIDIA TF-TRT User Guide — https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/  
2. NVIDIA TensorRT Developer Guide — https://docs.nvidia.com/deeplearning/tensorrt/  
3. MLPerf Inference — https://mlcommons.org/benchmarks/inference/  
4. Dao et al., FlashAttention — https://arxiv.org/abs/2205.14135  
5. Xiao et al., SmoothQuant — https://arxiv.org/abs/2211.10438  
6. Frantar et al., GPTQ — https://arxiv.org/abs/2210.17323  
7. Leviathan et al., Speculative Decoding — https://arxiv.org/abs/2211.17192  
8. Murray et al., tf.data — https://arxiv.org/abs/2101.12127  
9. Hendrycks & Dietterich, ImageNet-C — https://arxiv.org/abs/1903.12261  
10. Papers with Code leaderboards (Image Classification, Object Detection, Efficient Inference) — https://paperswithcode.com/  

---

*This document is intentionally ambitious. Treat Phase A as the engineering baseline; Phases B–D as the research-to-product bridge that keeps this suite relevant against TensorRT 10 / Blackwell-era stacks.*
