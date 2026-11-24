from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAMP = (2024, 1, 1, 0, 0, 0)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(path for path in (ROOT / "recommender").rglob("*.py") if "__pycache__" not in path.parts)
    sources.append(ROOT / "scripts/export_corrected_v4_serving_preprocessing.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as archive:
        for path in sources:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("runtime_manifest.json", STAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps({"runtime": "corrected_v4_preprocessing_v1", "entrypoint": "scripts/export_corrected_v4_serving_preprocessing.py"}, indent=2).encode())
    print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
