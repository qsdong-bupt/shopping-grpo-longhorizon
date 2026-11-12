"""Read-only verification of a frozen harness/evaluation manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze(root, manifest):
    root = Path(root).resolve()
    if manifest.get("schema_version") != "shopping-harness-eval-freeze-v1":
        raise ValueError("Unsupported freeze manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Freeze manifest contains no files")
    problems = []
    for name, expected in sorted(files.items()):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe manifest path: {name}")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Manifest path escapes root: {name}")
        if not path.is_file():
            problems.append({"path": name, "problem": "missing"})
        elif sha256_file(path) != expected["sha256"]:
            problems.append({"path": name, "problem": "sha256_mismatch"})
    # Additions can change imports or runtime behavior even if old files match.
    for pattern in manifest.get("protected_globs", []):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                name = path.relative_to(root).as_posix()
                if name not in files:
                    problems.append({"path": name, "problem": "unfrozen_addition"})
    return {"status": "passed" if not problems else "failed",
            "freeze_id": manifest["freeze_id"], "checked_files": len(files),
            "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    result = verify_freeze(args.root, json.loads(manifest_path.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
