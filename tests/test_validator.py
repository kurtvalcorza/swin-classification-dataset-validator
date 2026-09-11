from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from swin_classification_validator.validator import RuntimeBinding, ValidationFailure, validate_dataset

D = "sha256:" + "1" * 64


def binding() -> RuntimeBinding:
    return RuntimeBinding(
        job_id="job-test",
        attempt_id="attempt-1",
        worker_release_digest=D,
        effective_job_spec_digest=D,
        admission_record_digest=D,
        security_grant_digest=D,
    )


def image(path: Path, value: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def valid_dataset(root: Path) -> None:
    image(root / "train" / "cat" / "a.png", 10)
    image(root / "train" / "dog" / "b.png", 20)
    image(root / "val" / "cat" / "c.png", 30)
    image(root / "val" / "dog" / "d.png", 40)
    image(root / "test" / "cat" / "e.png", 50)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_emits_immutable_handoff(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    valid_dataset(root)

    summary = validate_dataset(root, out, binding())

    assert summary["validatedDatasetManifestDigest"].startswith("sha256:")
    plan = load(out / "data-plan.json")
    assignments = {x["sampleId"]: x["split"] for x in plan["assignments"]}
    assert assignments["test/cat/e.png"] == "test"
    assert assignments["val/cat/c.png"] == "validation"
    assert all("test/" not in sid or split == "test" for sid, split in assignments.items())

    logical = load(out / "logical-dataset-manifest.json")
    assert [x["sampleId"] for x in logical["samples"]] == sorted(x["sampleId"] for x in logical["samples"])
    assert load(out / "result.json")["state"] == "SUCCEEDED"
    assert load(out / "run-manifest.json")["reproducibility"] == "REEXECUTABLE"


def test_valid_alias_maps_to_validation(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    image(root / "train" / "cat" / "a.png")
    image(root / "valid" / "cat" / "b.png")
    validate_dataset(root, out, binding())
    plan = load(out / "data-plan.json")
    assert {x["split"] for x in plan["assignments"]} == {"train", "validation"}


def test_test_is_never_reinterpreted_as_validation(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    image(root / "train" / "cat" / "a.png")
    image(root / "test" / "cat" / "b.png")
    with pytest.raises(ValidationFailure) as exc:
        validate_dataset(root, out, binding())
    assert exc.value.finding["code"] == "VISION_MISSING_VALIDATION_SPLIT"
    assert load(out / "result.json")["failure"]["code"] == "VISION_MISSING_VALIDATION_SPLIT"


def test_val_and_valid_is_fatal_ambiguity(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    image(root / "train" / "cat" / "a.png")
    image(root / "val" / "cat" / "b.png")
    image(root / "valid" / "cat" / "c.png")
    with pytest.raises(ValidationFailure) as exc:
        validate_dataset(root, out, binding())
    assert exc.value.finding["code"] == "VISION_AMBIGUOUS_VALIDATION_SPLIT"


def test_corrupt_image_is_fatal_not_skipped(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    image(root / "train" / "cat" / "a.png")
    bad = root / "val" / "cat" / "bad.png"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not an image")
    with pytest.raises(ValidationFailure) as exc:
        validate_dataset(root, out, binding())
    assert exc.value.finding["code"] == "VISION_IMAGE_DECODE_FAILED"


def test_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    valid_dataset(root)
    target = root / "train" / "cat" / "a.png"
    link = root / "train" / "cat" / "link.png"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValidationFailure) as exc:
        validate_dataset(root, out, binding())
    assert exc.value.finding["code"] == "VISION_SYMLINK_REJECTED"


def test_runtime_authority_digest_is_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    valid_dataset(root)
    bad = RuntimeBinding("j", "a", "latest", D, D, D)
    with pytest.raises(ValueError, match="worker_release_digest"):
        validate_dataset(root, out, bad)


def test_duplicate_content_across_splits_is_warned_not_dropped(tmp_path: Path) -> None:
    """SPL9: the same bytes in train/ and val/ are reported with both sample ids; nothing is removed or re-split."""
    root = tmp_path / "data"
    out = tmp_path / "out"
    valid_dataset(root)
    image(root / "train" / "cat" / "leak.png", 30)  # byte-identical to val/cat/c.png
    image(root / "train" / "dog" / "twin-a.png", 77)  # within-split duplicate: not leakage
    image(root / "train" / "dog" / "twin-b.png", 77)

    validate_dataset(root, out, binding())

    l3 = load(out / "evidence" / "l3.json")
    codes = [f["code"] for f in l3["findings"]]
    assert codes.count("VISION_DUPLICATE_CONTENT_ACROSS_SPLITS") == 1
    warning = next(f for f in l3["findings"] if f["code"] == "VISION_DUPLICATE_CONTENT_ACROSS_SPLITS")
    assert warning["severity"] == "WARNING" and l3["outcome"] == "PASS"
    assert warning["observed"]["duplicateGroups"] == 1
    group = warning["observed"]["groups"][0]
    assert group["sampleIds"] == ["train/cat/leak.png", "val/cat/c.png"]
    assert group["splits"] == ["train", "validation"]
    # all eight samples are still assigned; the validator does not drop or move the duplicates
    plan = load(out / "data-plan.json")
    assert len(plan["assignments"]) == 8
    assert load(out / "result.json")["state"] == "SUCCEEDED"


def test_no_duplicate_warning_when_splits_are_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "data"
    out = tmp_path / "out"
    valid_dataset(root)

    validate_dataset(root, out, binding())

    codes = [f["code"] for f in load(out / "evidence" / "l3.json")["findings"]]
    assert "VISION_DUPLICATE_CONTENT_ACROSS_SPLITS" not in codes
