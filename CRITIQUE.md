# Self-Critique / Code Review — Security Hardening PR

**Branch:** `cursor/security-hardening-improvements-baf2`  
**Reviewer stance:** hostile, assume the first pass was half-wrong  
**Scope:** Diff vs `master` + current tree after critique fixes

---

## Verdict (final)

**Ship it.** The critical injection fix is real. The first pass had several “looks secure / is incomplete or regressing” issues; those are fixed below. Residual risk is appropriate for a local GPU benchmark suite (untrusted SavedModel execution still requires process isolation — not solvable in-repo).

---

## Pass 1 — What the initial PR got right

| Change | Assessment |
|--------|------------|
| Replace `eval ${COMMAND}` with `"${CMD[@]}"` | **Correct.** Injection payload becomes a literal argv element. Verified with `python3 -c 'print(sys.argv)'` + `; echo PWNED` style args. |
| `tempfile.mkdtemp` / `mktemp -d` | **Correct** vs fixed `/tmp/tmp_detection_results` and `/tmp/$RANDOM`. |
| COCO `safe_join_under` | **Directionally correct**; first implementation used `startswith(base+sep)` which is OK with `realpath`, but `os.path.commonpath` is clearer. |
| Docs (`VULNERABILITY.md`, `IMPROVEMENTS.md`) | Useful; severity framing mostly honest for a non-networked toolkit. |

---

## Pass 1 — Failures found (fixed in Pass 2)

### C1 — Incomplete integrity check (High → Fixed)

**Bug:** `--model_sha256` only hashed `saved_model.pb` / `.pbtxt`. An attacker keeps the proto, swaps `variables/*`, check still passes. That’s security theater.

**Fix:** `sha256_directory()` — deterministic sorted walk, path + content digests, **symlinks rejected**, compare via `hmac.compare_digest`. Help text + `VULNERABILITY.md` updated. Unit test `test_detects_variable_tampering`.

### C2 — `set -e` vs pycocotools probe (High regression → Fixed)

**Bug:** OD `base_script.sh` added `set -euo pipefail`, then:

```bash
python -c "from pycocotools.coco import COCO" > /dev/null 2>&1
DEPENDENCIES_STATUS=$?
```

If import fails, `set -e` exits **before** `install_dependencies.sh` runs. Fresh machines break.

**Fix:** `if ! python -c ...; then bash install_dependencies.sh; fi`

### C3 — Unquoted path tests (Medium → Fixed)

**Bug:** `[[ ! -d ${DATA_DIR} ]]` (and similar) break or mis-parse paths with spaces. Pre-existing pattern left intact in the “secure” rewrite.

**Fix:** Quote `"${DATA_DIR}"`, `"${MODEL_DIR}"`, `"${INPUT_SAVED_MODEL_DIR}"`, annotation paths; quote path joins.

### C4 — Empty / traversing `--model_name` (Medium → Fixed)

**Bug:** Empty `MODEL_NAME` → `INPUT_SAVED_MODEL_DIR="$MODEL_DIR/"` can spuriously pass. `../evil` in model name can escape `MODEL_DIR`.

**Fix:** Require non-empty `MODEL_NAME`; reject `/` and `..` substrings before join. Validate `BATCH_SIZE` **before** embedding it in the OD model path.

### C5 — Absolute path components in `safe_join_under` (Low/Medium → Fixed)

**Bug:** Relied only on post-`realpath` checks. `os.path.join(base, "/etc/passwd")` discards `base` on POSIX.

**Fix:** Reject absolute / `~` / NUL components up front; use `os.path.commonpath` after `realpath`. Tests for symlink escape + NUL.

### C6 — Warning noise (Low → Fixed)

Always printing the SECURITY banner even after a successful hash check trains users to ignore it.

**Fix:** Banner only when `--model_sha256` is absent; success path prints integrity OK.

---

## Pass 2 — Re-review after fixes

### Still strong
- No `eval` of user strings in launchers.
- Argv arrays + env exports for TF32/XLA.
- Full-tree model digest with symlink refusal + constant-time compare.
- Path traversal defenses on annotation-controlled filenames.
- Private temp dirs for COCO JSON (`O_EXCL` + `0600`) and TRT outputs.
- 9 unit tests, all passing; `bash -n` clean on the three launchers.

### Acceptable residuals (do not block)

| Residual | Why acceptable |
|----------|----------------|
| Loading a verified-but-malicious model still executes TF graph code | Integrity ≠ sandbox. Need gVisor/VM/user isolation (documented). |
| TOCTOU between hash and `tf.saved_model.load` | Same privilege as writing the model dir; out of scope without OS locks. |
| `tf.image.decode_jpeg` + `except InvalidArgumentError` may not behave like a true fallback under graph-traced `tf.data` | Pre-existing pattern; narrowing bare `except:` is still a net win. Prefer `tf.io.decode_image` in a later cleanup. |
| Notebooks still `!pip install` / logo `http://` URLs | Tutorial surface; HTTPS fixed for the weight tarball that matters. |
| No CI yet | Called out in `IMPROVEMENTS.md`; local unittest covers new helpers. |
| `get_dataset` still closes over global `coco` | Pre-existing smell; not introduced by hardening. |

### Intentional non-goals
- Not converting this repo into TensorRT-LLM / MLPerf LoadGen (roadmap only).
- Not pinning NGC/TF/TRT in `setup.py` (needs a compatibility matrix project).

---

## Test commands (reviewer reproduction)

```bash
bash -n tftrt/examples/image_classification/scripts/base_script.sh
bash -n tftrt/examples/object_detection/scripts/base_script.sh
bash -n tftrt/examples/transformers/scripts/base_script.sh
python3 -m unittest discover -s tftrt/examples/tests -v
```

Expected: 9 tests OK; no `eval ${COMMAND}` left in `*/scripts/base_script.sh`.

---

## Scorecard

| Criterion | Pass 1 | Pass 2 |
|-----------|--------|--------|
| Fixes the stated vulns | B+ | **A** |
| Doesn’t introduce regressions | C (`set -e` probe) | **A** |
| Integrity story honest | D (pb-only) | **A-** |
| Bash hardening completeness | B- | **A** |
| Test coverage of new helpers | C+ | **A-** |
| Doc / claim accuracy | B | **A-** |

**Stop criteria met:** critical injection fixed, integrity check not theater, no known `set -e` footguns in the touched launchers, tests green, residuals documented rather than papered over.
