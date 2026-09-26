from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.models.two_tower.training import load_config, train_two_tower


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic Two-Tower retrieval baseline.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_two_tower(load_config(args.config))
    print("Training finished.")
    print("Final validation:", metrics["final_validation"])
    print("Output metrics:", Path(load_config(args.config).output_dir) / "metrics.json")


if __name__ == "__main__":
    main()
