"""A frozen comparison must reject code drift and incomplete snapshots."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_frozen_harness", Path(__file__).resolve().parents[1] / "scripts/check_frozen_harness.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_manifest(root):
    source = root / "runtime.py"
    source.write_text("BASELINE = 1\n", encoding="utf-8")
    return {"schema_version": "shopping-harness-eval-freeze-v1", "freeze_id": "test",
            "protected_globs": ["*.py"],
            "files": {"runtime.py": {"sha256": module.sha256_file(source)}}}


def test_matching_snapshot_passes(tmp_path):
    assert module.verify_freeze(tmp_path, fixture_manifest(tmp_path))["status"] == "passed"


def test_edited_or_missing_runtime_fails(tmp_path):
    manifest = fixture_manifest(tmp_path)
    source = tmp_path / "runtime.py"
    source.write_text("BASELINE = 2\n", encoding="utf-8")
    assert module.verify_freeze(tmp_path, manifest)["problems"][0]["problem"] == "sha256_mismatch"
    source.unlink()
    assert module.verify_freeze(tmp_path, manifest)["problems"][0]["problem"] == "missing"


def test_new_runtime_file_fails(tmp_path):
    manifest = fixture_manifest(tmp_path)
    (tmp_path / "new_policy.py").write_text("", encoding="utf-8")
    assert module.verify_freeze(tmp_path, manifest)["problems"] == [
        {"path": "new_policy.py", "problem": "unfrozen_addition"}]


@pytest.mark.parametrize("name", ["../other.py", "/other.py", "C:/other.py"])
def test_manifest_cannot_escape_project(tmp_path, name):
    manifest = fixture_manifest(tmp_path)
    manifest["files"][name] = {"sha256": "0" * 64}
    with pytest.raises(ValueError, match="Unsafe manifest path"):
        module.verify_freeze(tmp_path, manifest)
