from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_TIMESTAMP = (2024, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest used by the Colab runtime preflight."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def add_file(archive: zipfile.ZipFile, path: Path) -> None:
    """Write one source file with deterministic archive metadata."""
    info = zipfile.ZipInfo(str(path.relative_to(PROJECT_ROOT)), ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the adaptive test rebuild runtime.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(
        path
        for path in (PROJECT_ROOT / "recommender").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    sources.extend(
        [
            PROJECT_ROOT / "scripts/build_adaptive_test_artifacts.py",
            PROJECT_ROOT / "scripts/build_adaptive_eval_protocol_v2.py",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as archive:
        for path in sources:
            add_file(archive, path)
        manifest = {
            "runtime": "adaptive_test_rebuild_v1",
            "build_entrypoint": "scripts/build_adaptive_test_artifacts.py",
            "protocol_entrypoint": "scripts/build_adaptive_eval_protocol_v2.py",
            "files": [str(path.relative_to(PROJECT_ROOT)) for path in sources],
        }
        info = zipfile.ZipInfo("runtime_manifest.json", ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True).encode())
    print(
        json.dumps(
            {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
