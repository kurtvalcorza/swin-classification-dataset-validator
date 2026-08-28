# Contract snapshot

This worker is implemented against `kurtvalcorza/ml-worker@0f0c221222402721ee7716edf01378604cbd6ef3`.

Normative surfaces used by this scaffold:

- task: `core.task.vision.image-classification`
- representation: `core.dataset.vision.image-folder` (candidate)
- `LogicalDatasetManifest`
- `SemanticDatasetSchema`
- `DataPlan`
- `ValidationFinding` / `ValidationEvidence`
- `ValidatedDatasetManifest`
- `ExecutionPlan`
- `RunManifest`
- `WorkerResult`
- `WorkerManifest`

The worker does not copy or modify legacy `NAIRA-SEU/*` implementations or audit records.

## Identity algorithms

`org.valcorza.swin-classification-validator.v1.source-snapshot` identifies the exact directory representation as the sorted set of POSIX relative file paths plus SHA-256 of each file's exact bytes.

`org.valcorza.swin-classification-validator.v1.logical-dataset` hashes canonical logical records containing stable sample ID, logical split, class label, and exact content digest. JSON canonicalization is UTF-8, sorted object keys, no insignificant whitespace, and SHA-256.

Runtime authority values (`workerReleaseDigest`, effective JobSpec, admission record, security grant, resource bindings) are supplied by the caller and validated fail-closed. The worker never fabricates those identities.
