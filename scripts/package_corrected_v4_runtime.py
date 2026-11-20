from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_TIMESTAMP = (2024, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def add_file(archive: zipfile.ZipFile, path: Path) -> None:
    info = zipfile.ZipInfo(str(path.relative_to(PROJECT_ROOT)), ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Package corrected V4 train/eval runtime.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(
        path for path in (PROJECT_ROOT / "recommender").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    sources.extend(
        [
            PROJECT_ROOT / "scripts/evaluate_adaptive_protocol_v2.py",
            PROJECT_ROOT / "scripts/train_corrected_adaptive_v4.py",
            PROJECT_ROOT / "scripts/compare_corrected_adaptive_v4.py",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as archive:
        for path in sources:
            add_file(archive, path)
        manifest = {
            "runtime": "corrected_adaptive_v4_v1",
            "train_entrypoint": "scripts/train_corrected_adaptive_v4.py",
            "compare_entrypoint": "scripts/compare_corrected_adaptive_v4.py",
            "files": [str(path.relative_to(PROJECT_ROOT)) for path in sources],
        }
        info = zipfile.ZipInfo("runtime_manifest.json", ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True).encode())
    print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
