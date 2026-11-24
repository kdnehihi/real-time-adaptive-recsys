from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.checkpoint import (
    ITEM_CATEGORICAL,
    ITEM_ID_FEATURES,
    ITEM_NUMERIC,
    NumericalPreprocessor,
    USER_NUMERIC,
    V4TransformerRetrieval,
    Vocabulary,
    encode_frame,
    move,
)


class ServingUserEncoder:
    """Load train-fitted mappings and encode point-in-time user state without Gold data."""

    def __init__(self, bundle_dir: Path, checkpoint_path: Path, device: torch.device) -> None:
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        vocabs: dict[str, Vocabulary] = {}
        for feature in ("user_id", "video_id"):
            values = np.load(bundle_dir / f"vocab_{feature}.npy", allow_pickle=False)
            vocabs[feature] = Vocabulary(feature, values.tolist())
        for feature in ("user_active_degree", "video_type", "upload_type", "event_type"):
            entries = json.loads((bundle_dir / f"vocab_{feature}.json").read_text())
            vocabs[feature] = Vocabulary(feature, [], {row["value"]: int(row["index"]) for row in entries})
        expected = {key: int(value) for key, value in manifest["vocab_sizes"].items()}
        actual = {key: vocab.size for key, vocab in vocabs.items()}
        if actual != expected:
            raise ValueError(f"Serving vocabulary sizes do not match manifest: {actual} != {expected}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self.model = V4TransformerRetrieval(expected, self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device).eval()
        self.vocabs = vocabs
        self.numeric = NumericalPreprocessor(bundle_dir / "numeric_stats.json")
        self.device = device

    @torch.inference_mode()
    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode request-time user/state rows into normalized retrieval vectors."""
        prepared = frame.copy()
        defaults = {"video_id": 0, "video_type": None, "upload_type": None}
        defaults.update({feature: 0.0 for feature in ITEM_NUMERIC})
        defaults.update({feature: 0.0 for feature in USER_NUMERIC})
        defaults.update(
            {
                "user_active_degree": None,
                "recent_video_ids": [],
                "recent_target_classes": [],
                "recent_engagement_strengths": [],
                "recent_long_views": [],
                "recent_likes": [],
                "recent_hates": [],
                "recent_clicks": [],
                "recent_time_ms": [],
                "current_time_ms": 0,
            }
        )
        for feature, default in defaults.items():
            if feature not in prepared:
                prepared[feature] = [default for _ in range(len(prepared))]
        batch = encode_frame(
            prepared, self.vocabs, self.numeric, int(self.config["sequence_max_len"])
        )
        return self.model.encode_user(move(batch, self.device)).cpu().numpy().astype("float32")
