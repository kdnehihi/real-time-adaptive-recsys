from __future__ import annotations

import json
import math
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

USER_CATEGORICAL = ("user_active_degree",)
ITEM_CATEGORICAL = ("video_type", "upload_type")
USER_ID_FEATURES = ("user_id",)
ITEM_ID_FEATURES = ("video_id",)

USER_NUMERIC = (
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    *[f"onehot_feat{i}" for i in range(18)],
    "user_hist_events",
    "user_hist_long_view_rate",
    "user_hist_like_rate",
    "user_hist_comment_rate",
    "user_hist_forward_rate",
    "user_hist_follow_rate",
    "user_hist_hate_rate",
    "user_hist_avg_watch_ratio",
    "user_hist_avg_play_time_sec",
    "session_event_index",
    "session_elapsed_sec",
    "session_prior_long_view_rate",
    "session_prior_like_rate",
    "session_prior_hate_rate",
    "session_prior_avg_watch_ratio",
    "session_vs_user_long_view_delta",
    "session_vs_user_watch_ratio_delta",
    "event_hour",
    "event_dayofweek",
    "tab",
    "is_rand",
)

ITEM_NUMERIC = (
    "video_duration_sec",
    "aspect_ratio",
    "visible_status",
    "music_type",
    "upload_age_days_at_event",
    "item_hist_events",
    "item_hist_long_view_rate",
    "item_hist_like_rate",
    "item_hist_hate_rate",
    "item_hist_avg_watch_ratio",
)

SEQUENCE_COLUMNS = (
    "recent_video_ids",
    "recent_target_classes",
    "recent_engagement_strengths",
    "recent_long_views",
    "recent_likes",
    "recent_hates",
    "recent_clicks",
    "recent_time_ms",
    "recent_event_count",
    "seconds_since_last_event",
)

TARGET_COLUMNS = (
    "target_class",
    "is_positive",
    "is_observed_negative",
    "is_ambiguous",
    "long_view",
    "is_like",
    "is_follow",
    "is_hate",
    "engagement_strength",
)

BASE_COLUMNS = ("example_id", "user_id", "video_id", "as_of_time", "current_time_ms")
INPUT_COLUMNS = list(
    dict.fromkeys(
        [
            *BASE_COLUMNS,
            *TARGET_COLUMNS,
            *USER_ID_FEATURES,
            *USER_CATEGORICAL,
            *USER_NUMERIC,
            *ITEM_ID_FEATURES,
            *ITEM_CATEGORICAL,
            *ITEM_NUMERIC,
            *SEQUENCE_COLUMNS,
        ]
    )
)


def parquet_files(path: Path) -> list[Path]:
    """Return real Parquet data files and ignore Spark metadata sidecars."""
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    files = [p for p in files if p.is_file() and not p.name.startswith(".")]
    if not files:
        visible = sorted(x.name for x in path.iterdir())[:30] if path.is_dir() else []
        raise FileNotFoundError(f"No parquet files found under {path}. Visible entries: {visible}")
    return files


def existing_cols(path: Path) -> list[str]:
    """Inspect a Parquet directory/file without materializing rows."""
    return pq.ParquetFile(parquet_files(path)[0]).schema_arrow.names


def read_parquet(
    path: Path,
    columns: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
    filter_expr: ds.Expression | None = None,
) -> pd.DataFrame:
    """Read a bounded Parquet slice into pandas for checkpoint evaluation."""
    files = [str(p) for p in parquet_files(path)]
    dataset = ds.dataset(files, format="parquet")
    cols = list(columns) if columns is not None else None
    if limit is not None:
        table = dataset.head(limit, columns=cols, filter=filter_expr)
    else:
        table = dataset.to_table(columns=cols, filter=filter_expr)
    frame = table.to_pandas()
    return frame


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON artifacts with numpy-safe conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class Vocabulary:
    """Train-fitted ID/OOV mapping with 0 reserved for unseen values."""
    def __init__(self, name: str, values: list[Any] | pd.Series, index: dict[Any, int] | None = None):
        self.name = name
        if index is None:
            clean = pd.Series(list(values)).dropna().drop_duplicates().tolist()
            self.index = {value: i + 1 for i, value in enumerate(sorted(clean))}
        else:
            self.index = dict(index)
        self.size = max(self.index.values(), default=0) + 1

    def encode_many(self, values: list[Any]) -> np.ndarray:
        return np.asarray([self.index.get(value, 0) for value in values], dtype=np.int64)


def load_parquet_vocabulary(path: Path, feature: str) -> Vocabulary:
    """Load categorical vocabularies created by the Gold data pipeline."""
    frame = read_parquet(path)
    return Vocabulary(feature, [], dict(zip(frame[feature].tolist(), frame[f"{feature}_idx"].astype(int).tolist())))


class NumericalPreprocessor:
    """Apply train-fitted impute, clip, and z-score transforms."""
    def __init__(self, path: Path):
        payload = json.loads(path.read_text())
        self.stats = payload.get("features", payload)

    def transform(self, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for feature in features:
            if feature in frame.columns:
                values = pd.to_numeric(frame[feature], errors="coerce").astype("float32")
            else:
                values = pd.Series(np.zeros(len(frame), dtype=np.float32), dtype="float32")
            stats = self.stats.get(feature)
            if stats:
                values = values.fillna(float(stats.get("mean", 0.0) or 0.0))
                lower = stats.get("p01", stats.get("min"))
                upper = stats.get("p99", stats.get("max"))
                if lower is not None and upper is not None:
                    values = values.clip(float(lower), float(upper))
                std = float(stats.get("stddev", 0.0) or 0.0)
                mean = float(stats.get("mean", 0.0) or 0.0)
                values = (values - mean) / std if std > 1e-6 else values * 0.0
            outputs.append(np.nan_to_num(values.to_numpy("float32"), nan=0.0, posinf=0.0, neginf=0.0))
        if not outputs:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return np.stack(outputs, axis=1).astype("float32")


def as_sequence(value: Any) -> list[Any]:
    """Normalize Arrow/list/numpy sequence cells to a plain Python list."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def as_of_time_ms(frame: pd.DataFrame) -> np.ndarray:
    return (pd.to_datetime(frame["as_of_time"]).astype("int64") // 1_000_000).to_numpy(dtype="int64")


def sanitize_point_in_time_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop sequence events whose timestamp is not strictly before the query."""
    frame = frame.copy()
    boundaries = as_of_time_ms(frame)
    sequence_cols = [
        "recent_video_ids",
        "recent_target_classes",
        "recent_engagement_strengths",
        "recent_long_views",
        "recent_likes",
        "recent_hates",
        "recent_clicks",
        "recent_time_ms",
    ]
    cleaned = {col: [] for col in sequence_cols if col in frame.columns}
    cleaned_counts: list[int] = []
    dropped_rows = 0
    dropped_events = 0
    for (_, row), boundary in zip(frame.iterrows(), boundaries):
        times = as_sequence(row.get("recent_time_ms"))
        keep = [i for i, ts in enumerate(times) if ts is not None and not pd.isna(ts) and float(ts) < float(boundary)]
        if len(keep) != len(times):
            dropped_rows += 1
            dropped_events += len(times) - len(keep)
        for col in cleaned:
            values = as_sequence(row.get(col))
            cleaned[col].append([values[i] for i in keep] if len(values) == len(times) else (values[-len(keep) :] if keep else []))
        cleaned_counts.append(len(keep))
    for col, values in cleaned.items():
        frame[col] = values
    frame["recent_event_count"] = cleaned_counts
    frame["current_time_ms"] = boundaries
    print(f"point-in-time sanitize: rows_with_drops={dropped_rows:,} dropped_events={dropped_events:,}")
    return frame


def pad_num(values: Any, max_len: int, dtype: str = "float32") -> np.ndarray:
    values = as_sequence(values)[-max_len:]
    output = np.zeros(max_len, dtype=dtype)
    if values:
        arr = np.asarray(values, dtype=dtype)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        output[-len(arr) :] = arr
    return output


def pad_enc(values: Any, vocab: Vocabulary, max_len: int) -> np.ndarray:
    values = as_sequence(values)[-max_len:]
    output = np.zeros(max_len, dtype=np.int64)
    if values:
        output[-len(values) :] = vocab.encode_many(values)[-max_len:]
    return output


def build_vocabs(base_gold_root: Path, train_frame: pd.DataFrame, item_pool: pd.DataFrame) -> dict[str, Vocabulary]:
    """Rebuild the same vocab sizes used by the checkpointed V4/V5 models."""
    return {
        "user_active_degree": load_parquet_vocabulary(base_gold_root / "vocabularies/user_active_degree", "user_active_degree"),
        "video_type": load_parquet_vocabulary(base_gold_root / "vocabularies/video_type", "video_type"),
        "upload_type": load_parquet_vocabulary(base_gold_root / "vocabularies/upload_type", "upload_type"),
        "user_id": Vocabulary("user_id", train_frame["user_id"].tolist()),
        "video_id": Vocabulary("video_id", item_pool["video_id"].tolist()),
        "event_type": Vocabulary("event_type", ["AMBIGUOUS_WEAK", "OBSERVED_NEGATIVE", "STRONG_POSITIVE"]),
    }


def encode_frame(frame: pd.DataFrame, vocabs: dict[str, Vocabulary], numeric: NumericalPreprocessor, max_len: int) -> dict[str, torch.Tensor]:
    """Convert Gold/adaptive rows into tensor inputs for V4 encoders."""
    batch: dict[str, torch.Tensor] = {}
    for feature in USER_ID_FEATURES + USER_CATEGORICAL + ITEM_ID_FEATURES + ITEM_CATEGORICAL:
        batch[feature] = torch.as_tensor(vocabs[feature].encode_many(frame[feature].tolist()), dtype=torch.long)
    batch["user_numeric"] = torch.as_tensor(numeric.transform(frame, USER_NUMERIC), dtype=torch.float32)
    batch["item_numeric"] = torch.as_tensor(numeric.transform(frame, ITEM_NUMERIC), dtype=torch.float32)
    batch["seq_video_id"] = torch.as_tensor(np.stack([pad_enc(v, vocabs["video_id"], max_len) for v in frame["recent_video_ids"]]), dtype=torch.long)
    batch["seq_event_type"] = torch.as_tensor(
        np.stack([pad_enc(v, vocabs["event_type"], max_len) for v in frame["recent_target_classes"]]), dtype=torch.long
    )
    signals = np.stack(
        [np.stack([pad_num(v, max_len) for v in frame[col]]) for col in ["recent_engagement_strengths", "recent_long_views", "recent_likes", "recent_hates", "recent_clicks"]],
        axis=-1,
    )
    batch["seq_signals"] = torch.as_tensor(signals, dtype=torch.float32)
    current = pd.to_numeric(frame["current_time_ms"], errors="coerce").fillna(0).to_numpy("float64")
    recent = np.stack([pad_num(v, max_len, "float64") for v in frame["recent_time_ms"]])
    age = np.maximum((current[:, None] - recent) / 1000.0, 0.0)
    age[recent <= 0] = 0.0
    batch["seq_time_bucket"] = torch.as_tensor(np.clip(np.floor(np.log1p(age) / np.log(2.0)), 0, 31).astype("int64"))
    batch["seq_padding_mask"] = batch["seq_video_id"].eq(0)
    return batch


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def minimal_user_stub(source_row: dict[str, Any], n: int) -> dict[str, Any]:
    """Fill user/sequence columns while encoding item-only candidate rows."""
    row = source_row
    data: dict[str, Any] = {col: row.get(col, 0) for col in USER_NUMERIC}
    data.update(
        {
            "user_id": row["user_id"],
            "user_active_degree": row.get("user_active_degree"),
            "recent_video_ids": [row.get("recent_video_ids", []) for _ in range(n)],
            "recent_target_classes": [row.get("recent_target_classes", []) for _ in range(n)],
            "recent_engagement_strengths": [row.get("recent_engagement_strengths", []) for _ in range(n)],
            "recent_long_views": [row.get("recent_long_views", []) for _ in range(n)],
            "recent_likes": [row.get("recent_likes", []) for _ in range(n)],
            "recent_hates": [row.get("recent_hates", []) for _ in range(n)],
            "recent_clicks": [row.get("recent_clicks", []) for _ in range(n)],
            "recent_time_ms": [row.get("recent_time_ms", []) for _ in range(n)],
            "current_time_ms": row.get("current_time_ms", 0),
        }
    )
    return data


def mlp(input_dim: int, hidden_dims: tuple[int, ...] | list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(previous, int(hidden)), nn.ReLU(), nn.Dropout(dropout)]
        previous = int(hidden)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class V4TransformerRetrieval(nn.Module):
    """Transformer sequence-aware Two-Tower retrieval model used by V4."""
    def __init__(self, vocab_sizes: dict[str, int], cfg: dict[str, Any]):
        super().__init__()
        retrieval_dim = int(cfg["retrieval_dim"])
        transformer_dim = int(cfg["transformer_dim"])
        self.user_id_emb = nn.Embedding(vocab_sizes["user_id"], 8, padding_idx=0)
        self.user_active_emb = nn.Embedding(vocab_sizes["user_active_degree"], 4, padding_idx=0)
        self.video_id_emb = nn.Embedding(vocab_sizes["video_id"], 16, padding_idx=0)
        self.video_type_emb = nn.Embedding(vocab_sizes["video_type"], 3, padding_idx=0)
        self.upload_type_emb = nn.Embedding(vocab_sizes["upload_type"], 8, padding_idx=0)
        self.seq_video_emb = nn.Embedding(vocab_sizes["video_id"], transformer_dim, padding_idx=0)
        self.seq_event_emb = nn.Embedding(vocab_sizes["event_type"], 32, padding_idx=0)
        self.seq_event_proj = nn.Linear(32, transformer_dim)
        self.seq_signal_proj = nn.Linear(5, transformer_dim)
        self.seq_time_emb = nn.Embedding(32, transformer_dim)
        layer = nn.TransformerEncoderLayer(
            transformer_dim,
            int(cfg["transformer_heads"]),
            transformer_dim * 4,
            float(cfg["transformer_dropout"]),
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, int(cfg["transformer_layers"]))
        self.short_proj = nn.Sequential(nn.LayerNorm(transformer_dim), nn.Linear(transformer_dim, retrieval_dim), nn.ReLU())
        self.long_user = mlp(8 + 4 + len(USER_NUMERIC), tuple(cfg["user_hidden_dims"]), retrieval_dim, float(cfg["dropout"]))
        self.item_tower = mlp(16 + 3 + 8 + len(ITEM_NUMERIC), tuple(cfg["item_hidden_dims"]), retrieval_dim, float(cfg["dropout"]))
        self.gate = nn.Sequential(nn.Linear(retrieval_dim * 2, retrieval_dim), nn.Sigmoid())

    def encode_sequence(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token = self.seq_video_emb(batch["seq_video_id"])
        token = token + self.seq_event_proj(self.seq_event_emb(batch["seq_event_type"]))
        token = token + self.seq_signal_proj(batch["seq_signals"])
        token = token + self.seq_time_emb(batch["seq_time_bucket"].clamp(0, 31))
        mask = batch["seq_padding_mask"]
        encoded = self.transformer(token, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return self.short_proj(pooled)

    def encode_user(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        long_features = torch.cat([self.user_id_emb(batch["user_id"]), self.user_active_emb(batch["user_active_degree"]), batch["user_numeric"]], dim=1)
        long_repr = self.long_user(long_features)
        short_repr = self.encode_sequence(batch)
        gate = self.gate(torch.cat([long_repr, short_repr], dim=1))
        return F.normalize(gate * short_repr + (1.0 - gate) * long_repr, dim=1)

    def encode_item(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item_features = torch.cat(
            [self.video_id_emb(batch["video_id"]), self.video_type_emb(batch["video_type"]), self.upload_type_emb(batch["upload_type"]), batch["item_numeric"]],
            dim=1,
        )
        return F.normalize(self.item_tower(item_features), dim=1)


class V5Ranker(nn.Module):
    """Second-stage ranker over V4 candidate user/item embeddings."""
    def __init__(self, retrieval_dim: int, hidden_dims: tuple[int, ...] | list[int], dropout: float):
        super().__init__()
        self.net = mlp(retrieval_dim * 4 + 1, tuple(hidden_dims), 1, dropout)

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, v4_score: torch.Tensor) -> torch.Tensor:
        user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)
        item_emb = torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0)
        v4_score = torch.nan_to_num(v4_score, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        features = torch.cat([user_emb, item_emb, user_emb * item_emb, torch.abs(user_emb - item_emb), v4_score[:, None]], dim=1)
        return self.net(torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)).squeeze(1)


class V5ResidualRanker(nn.Module):
    """Ranking-aware V5 that starts as a monotonic copy of the V4 ordering."""

    def __init__(self, retrieval_dim: int, hidden_dims: tuple[int, ...] | list[int], dropout: float):
        super().__init__()
        self.delta = mlp(retrieval_dim * 4 + 1, tuple(hidden_dims), 1, dropout)
        final_layer = self.delta[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        self.raw_base_weight = nn.Parameter(torch.tensor(0.54132485))

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, v4_score: torch.Tensor) -> torch.Tensor:
        user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)
        item_emb = torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0)
        v4_score = torch.nan_to_num(v4_score, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
        features = torch.cat(
            [user_emb, item_emb, user_emb * item_emb, torch.abs(user_emb - item_emb), v4_score[:, None]],
            dim=1,
        )
        delta = self.delta(torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)).squeeze(1)
        base_weight = F.softplus(self.raw_base_weight)
        return base_weight * v4_score + delta


@dataclass
class LoadedAdaptiveArtifacts:
    v4: V4TransformerRetrieval
    v5: V5Ranker | None
    v4_cfg: dict[str, Any]
    v5_cfg: dict[str, Any] | None
    vocabs: dict[str, Vocabulary]
    numeric: NumericalPreprocessor
    device: torch.device


def select_device(raw: str = "auto") -> torch.device:
    """Choose CUDA when available, otherwise CPU for predictable checkpoint loading."""
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_train_vocab_frame(adaptive_root: Path, limit: int | None = None) -> pd.DataFrame:
    """Read only train user IDs needed to reconstruct entity vocab shape."""
    return read_parquet(
        adaptive_root / "sequence_examples/train",
        ["user_id"],
        limit,
        ds.field("target_class") == "STRONG_POSITIVE",
    ).reset_index(drop=True)


def prepare_item_pool(adaptive_root: Path, split: str, limit: int | None = None) -> pd.DataFrame:
    """Build a unique item candidate table from adaptive sequence examples."""
    columns = list(dict.fromkeys([*ITEM_ID_FEATURES, *ITEM_CATEGORICAL, *ITEM_NUMERIC]))
    return read_parquet(adaptive_root / "sequence_examples" / split, columns, limit).drop_duplicates("video_id", keep="last").reset_index(drop=True)


def prepare_train_item_vocab_frame(adaptive_root: Path, limit: int | None = None) -> pd.DataFrame:
    """Read only train item IDs needed to reconstruct the checkpoint vocabulary."""
    return read_parquet(
        adaptive_root / "sequence_examples/train",
        ["video_id"],
        limit,
    ).drop_duplicates("video_id").reset_index(drop=True)


def load_artifacts(
    adaptive_root: Path,
    base_gold_root: Path,
    v4_dir: Path,
    v5_dir: Path | None = None,
    device: torch.device | None = None,
    train_vocab_limit: int | None = None,
    item_pool_limit: int | None = None,
) -> LoadedAdaptiveArtifacts:
    """Load V4/V5 checkpoints plus the train-fitted preprocessing objects."""
    device = device or select_device()
    v4_ckpt = torch.load(v4_dir / "best_model.pt", map_location="cpu")
    v4_cfg = v4_ckpt["config"]
    train_for_vocab = prepare_train_vocab_frame(adaptive_root, train_vocab_limit or v4_cfg.get("max_train_examples"))
    item_pool = prepare_train_item_vocab_frame(adaptive_root, item_pool_limit or v4_cfg.get("max_item_pool_examples"))
    vocabs = build_vocabs(base_gold_root, train_for_vocab, item_pool)
    vocab_sizes = v4_ckpt.get("vocab_sizes", {key: vocab.size for key, vocab in vocabs.items()})
    numeric = NumericalPreprocessor(base_gold_root / "transforms/numeric_stats.json")
    v4 = V4TransformerRetrieval(vocab_sizes, v4_cfg)
    v4.load_state_dict(v4_ckpt["model_state_dict"])
    del v4_ckpt
    gc.collect()
    v4 = v4.to(device)
    v4.eval()

    v5 = None
    v5_cfg = None
    if v5_dir is not None and (v5_dir / "best_model.pt").exists():
        v5_ckpt = torch.load(v5_dir / "best_model.pt", map_location="cpu")
        v5_cfg = v5_ckpt["config"]
        ranker_class = V5ResidualRanker if v5_cfg.get("model_type") == "residual_listwise_v2" else V5Ranker
        v5 = ranker_class(int(v4_cfg["retrieval_dim"]), tuple(v5_cfg["hidden_dims"]), float(v5_cfg["dropout"]))
        v5.load_state_dict(v5_ckpt["model_state_dict"])
        del v5_ckpt
        gc.collect()
        v5 = v5.to(device)
        v5.eval()
    return LoadedAdaptiveArtifacts(v4=v4, v5=v5, v4_cfg=v4_cfg, v5_cfg=v5_cfg, vocabs=vocabs, numeric=numeric, device=device)


@torch.no_grad()
def encode_items(
    model: V4TransformerRetrieval,
    frame: pd.DataFrame,
    source_row: dict[str, Any],
    artifacts: LoadedAdaptiveArtifacts,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode candidate items once so retrieval can use matrix scoring."""
    embeddings: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start : start + batch_size].reset_index(drop=True)
        batch = encode_frame(part.assign(**minimal_user_stub(source_row, len(part))), artifacts.vocabs, artifacts.numeric, int(artifacts.v4_cfg["sequence_max_len"]))
        emb = model.encode_item(move(batch, artifacts.device)).cpu().numpy()
        embeddings.append(np.nan_to_num(emb.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0))
        ids.append(part["video_id"].to_numpy("int64"))
    return np.concatenate(ids), np.vstack(embeddings).astype("float32")


@torch.no_grad()
def encode_users(model: V4TransformerRetrieval, frame: pd.DataFrame, artifacts: LoadedAdaptiveArtifacts, batch_size: int = 512) -> np.ndarray:
    """Encode point-in-time query rows into adaptive user embeddings."""
    embeddings: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start : start + batch_size].reset_index(drop=True)
        batch = encode_frame(part, artifacts.vocabs, artifacts.numeric, int(artifacts.v4_cfg["sequence_max_len"]))
        emb = model.encode_user(move(batch, artifacts.device)).cpu().numpy()
        embeddings.append(np.nan_to_num(emb.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0))
    return np.vstack(embeddings).astype("float32")


def top_indices(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted top-k column indices and scores for each row."""
    k = min(top_k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


@torch.no_grad()
def retrieve_for_queries(
    artifacts: LoadedAdaptiveArtifacts,
    query_frame: pd.DataFrame,
    item_ids: np.ndarray,
    item_emb: np.ndarray,
    retrieval_candidates: int,
    batch_size: int,
    item_score_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run exact matrix retrieval for query users against candidate items."""
    user_emb = encode_users(artifacts.v4, query_frame, artifacts, batch_size=batch_size)
    idx_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for start in range(0, len(user_emb), batch_size):
        scores = np.nan_to_num(user_emb[start : start + batch_size] @ item_emb.T, nan=-1e4, posinf=1e4, neginf=-1e4)
        if item_score_bias is not None:
            scores = scores + item_score_bias[None, :]
        idx, vals = top_indices(scores, retrieval_candidates)
        idx_parts.append(idx)
        score_parts.append(vals.astype("float32"))
    return user_emb, np.vstack(idx_parts), np.vstack(score_parts)


@torch.no_grad()
def rerank_with_v5(
    artifacts: LoadedAdaptiveArtifacts,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    candidate_idx: np.ndarray,
    v4_scores: np.ndarray,
    batch_queries: int,
) -> np.ndarray:
    """Score the same V4 candidates with the V5 ranker."""
    if artifacts.v5 is None:
        raise ValueError("V5 checkpoint was not loaded.")
    ranker_scores = v4_scores
    if artifacts.v5_cfg and artifacts.v5_cfg.get("model_type") == "residual_listwise_v2":
        score_mean = v4_scores.mean(axis=1, keepdims=True)
        score_std = v4_scores.std(axis=1, keepdims=True)
        ranker_scores = np.nan_to_num(
            (v4_scores - score_mean) / (score_std + 1e-6),
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        ).astype("float32")
    outputs: list[np.ndarray] = []
    for start in range(0, len(user_emb), batch_queries):
        end = min(start + batch_queries, len(user_emb))
        flat_user: list[np.ndarray] = []
        flat_item: list[np.ndarray] = []
        flat_score: list[float] = []
        for row in range(start, end):
            for item_pos, score in zip(candidate_idx[row], ranker_scores[row]):
                flat_user.append(user_emb[row])
                flat_item.append(item_emb[item_pos])
                flat_score.append(float(score))
        fu = torch.as_tensor(np.nan_to_num(np.vstack(flat_user).astype("float32")), device=artifacts.device)
        fi = torch.as_tensor(np.nan_to_num(np.vstack(flat_item).astype("float32")), device=artifacts.device)
        fs = torch.as_tensor(np.clip(np.nan_to_num(np.asarray(flat_score, dtype="float32")), -1.0, 1.0), device=artifacts.device)
        scores = artifacts.v5(fu, fi, fs).detach().cpu().numpy()
        outputs.append(scores.reshape(end - start, candidate_idx.shape[1]))
    return np.vstack(outputs).astype("float32")


def exact_next_metrics(candidate_ids: np.ndarray, ranked_idx: np.ndarray, true_items: np.ndarray, top_ks: tuple[int, ...]) -> dict[str, float]:
    """Measure whether the exact next positive item appears in Top-K."""
    output: dict[str, float] = {}
    for k in top_ks:
        recs = candidate_ids[ranked_idx[:, :k]]
        hit = (recs == true_items[:, None]).any(axis=1)
        output[f"hitrate@{k}"] = float(hit.mean()) if len(hit) else 0.0
        ndcg_values = []
        for row, true_item in zip(recs, true_items):
            pos = np.where(row == true_item)[0]
            ndcg_values.append(0.0 if len(pos) == 0 else 1.0 / math.log2(int(pos[0]) + 2))
        output[f"ndcg@{k}"] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    output["evaluated_queries"] = int(len(true_items))
    output["candidate_items"] = int(len(candidate_ids))
    return output


def multi_positive_metrics(
    candidate_ids: np.ndarray,
    ranked_idx: np.ndarray,
    query_user_ids: np.ndarray,
    validation_positive_frame: pd.DataFrame,
    top_ks: tuple[int, ...],
) -> dict[str, float]:
    """Measure Top-K quality against all positives for the same user in split."""
    relevance = validation_positive_frame.groupby("user_id")["video_id"].apply(lambda s: set(int(x) for x in s.dropna().tolist())).to_dict()
    rows = []
    evaluated_user_ids: set[int] = set()
    for row_idx, user_id in enumerate(query_user_ids):
        relevant = relevance.get(int(user_id), set())
        if relevant:
            rows.append((candidate_ids[ranked_idx[row_idx, : max(top_ks)]], relevant))
            evaluated_user_ids.add(int(user_id))
    output: dict[str, float] = {}
    for k in top_ks:
        hits: list[bool] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        for ranked, relevant in rows:
            top = [int(x) for x in ranked[:k]]
            rel_flags = [1 if item in relevant else 0 for item in top]
            hit_count = sum(rel_flags)
            hits.append(hit_count > 0)
            recalls.append(hit_count / max(len(relevant), 1))
            dcg = sum(flag / math.log2(pos + 2) for pos, flag in enumerate(rel_flags))
            idcg = sum(1.0 / math.log2(pos + 2) for pos in range(min(len(relevant), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        output[f"hitrate@{k}"] = float(np.mean(hits)) if hits else 0.0
        output[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        output[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    output["evaluated_queries"] = int(len(rows))
    output["evaluated_users"] = int(len(evaluated_user_ids))
    output["candidate_items"] = int(len(candidate_ids))
    return output
