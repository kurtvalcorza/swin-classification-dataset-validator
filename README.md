# Swin Classification Dataset Validator

NATIVE `ml-worker` validator for `core.task.vision.image-classification` using the candidate `core.dataset.vision.image-folder` representation profile.

## Current Builder scope

This repository implements the validator half of the Swin classification worker family. It intentionally precedes the finetuner: the finetuner must consume the immutable `DataPlan` and `ValidatedDatasetManifest` emitted here rather than re-deriving splits or class semantics.

Key behavior:

- `train/` and `val/` are canonical; `valid/` is an alias for validation.
- `test/` remains test and is never silently repurposed as validation.
- `val/` + `valid/` together is a typed fatal ambiguity.
- class membership is directory-derived; names that collide after NFC normalization and case folding (`Cat` vs `cat`) are a typed fatal collision, not two classes.
- stable sample IDs are POSIX relative paths.
- exact image bytes are SHA-256 identified.
- image decode failures are FATAL; no synthetic pixels are substituted.
- symlinks/path escape are rejected.
- byte-identical images present in more than one split are reported as a `VISION_DUPLICATE_CONTENT_ACROSS_SPLITS` WARNING (L3) naming every affected sample id; nothing is dropped or re-split (leakage detection, DIMER Pipeline Spec SPL9).
- successful validation emits `LogicalDatasetManifest`, `SemanticDatasetSchema`, `DataPlan`, L0-L4 `ValidationEvidence`, `ValidatedDatasetManifest`, and a validated-identity sidecar.
- `result.json` and all other output artifacts use staged atomic writes.
- runtime authority digests are caller-supplied and fail-closed; the worker does not invent release/security/admission identity.

## CLI

```bash
swin-classification-validate DATASET_ROOT OUTPUT_DIR \
  --job-id job-1 \
  --attempt-id attempt-1 \
  --worker-release-digest sha256:<64hex> \
  --effective-job-spec-digest sha256:<64hex> \
  --admission-record-digest sha256:<64hex> \
  --security-grant-digest sha256:<64hex>
```

See `CONTRACT_SNAPSHOT.md` for the pinned Contract revision and identity algorithms.

## Status

Builder scaffold. Blackwell/GPU execution is not relevant to the validator and no release-readiness claim is made from this repository alone.
