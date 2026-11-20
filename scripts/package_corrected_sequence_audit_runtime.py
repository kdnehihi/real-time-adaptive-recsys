from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_TIMESTAMP = (2024, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum for notebook integrity checks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package corrected-sequence audit runtime.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    """Add one deterministic compressed member to the runtime archive."""
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        path
        for path in (PROJECT_ROOT / "recommender").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    sources.append(PROJECT_ROOT / "scripts/audit_corrected_adaptive_sequences.py")
    manifest = {
        "runtime": "corrected_adaptive_sequence_audit_v1",
        "entrypoint": "scripts/audit_corrected_adaptive_sequences.py",
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in sources],
    }
    with zipfile.ZipFile(args.output, "w") as archive:
        for path in sources:
            add_bytes(archive, str(path.relative_to(PROJECT_ROOT)), path.read_bytes())
        add_bytes(
            archive,
            "runtime_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode(),
        )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": sha256(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
