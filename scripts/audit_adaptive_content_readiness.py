from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from recommender.models.adaptive.checkpoint import parquet_files, save_json
from scripts.evaluate_adaptive_protocol_v2 import load_event_history, read_shard


STATIC_COLUMNS = (
    "video_id",
    "author_id",
    "video_type",
    "upload_type",
    "upload_date",
    "video_duration_sec",
    "aspect_ratio",
    "visible_status",
    "music_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether current item metadata can support cold retrieval.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation"])
    return parser.parse_args()


def load_distinct_ids(path: Path, column: str = "video_id") -> set[int]:
    """Read one ID column shard-by-shard so Google Drive handles stay bounded."""
    values: set[int] = set()
    files = parquet_files(path)
    for index, file_path in enumerate(files, start=1):
        part = read_shard(file_path, [column])
        clean = pd.to_numeric(part[column], errors="coerce").dropna().astype("int64")
        values.update(int(value) for value in clean.unique())
        if index % 16 == 0 or index == len(files):
            print(f"IDs {path.name}: {index}/{len(files)} shards; {len(values):,} distinct", flush=True)
    return values


def load_static_metadata(
    static_path: Path,
    candidate_ids: set[int],
    train_ids: set[int],
) -> tuple[pd.DataFrame, dict[int, int], set[int]]:
    """Load candidate metadata plus the full video-to-author map in one scan."""
    candidate_parts: list[pd.DataFrame] = []
    video_to_author: dict[int, int] = {}
    warm_authors: set[int] = set()
    files = parquet_files(static_path)
    for index, file_path in enumerate(files, start=1):
        part = read_shard(file_path, list(STATIC_COLUMNS))
        part["video_id"] = pd.to_numeric(part["video_id"], errors="coerce")
        part["author_id"] = pd.to_numeric(part["author_id"], errors="coerce")
        clean = part.dropna(subset=["video_id"]).copy()
        clean["video_id"] = clean["video_id"].astype("int64")
        author_rows = clean.dropna(subset=["author_id"])[["video_id", "author_id"]]
        author_rows["author_id"] = author_rows["author_id"].astype("int64")
        video_to_author.update(zip(author_rows["video_id"], author_rows["author_id"]))
        warm_authors.update(
            int(value)
            for value in author_rows.loc[author_rows["video_id"].isin(train_ids), "author_id"].unique()
        )
        selected = clean[clean["video_id"].isin(candidate_ids)]
        if not selected.empty:
            candidate_parts.append(selected)
        if index % 8 == 0 or index == len(files):
            print(
                f"static metadata: {index}/{len(files)} shards; map={len(video_to_author):,}; "
                f"candidate rows={sum(len(frame) for frame in candidate_parts):,}",
                flush=True,
            )
    candidates = pd.concat(candidate_parts, ignore_index=True).drop_duplicates("video_id", keep="last")
    return candidates.reset_index(drop=True), video_to_author, warm_authors


def add_content_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    """Create coarse deterministic signatures to quantify metadata collisions."""
    output = frame.copy()
    duration = pd.to_numeric(output["video_duration_sec"], errors="coerce")
    aspect = pd.to_numeric(output["aspect_ratio"], errors="coerce")
    output["duration_bucket"] = pd.cut(
        duration,
        bins=[-np.inf, 5, 10, 20, 30, 60, 120, np.inf],
        labels=False,
    ).fillna(-1).astype("int16")
    output["aspect_bucket"] = pd.cut(
        aspect,
        bins=[-np.inf, 0.75, 1.0, 1.33, 1.78, 2.0, np.inf],
        labels=False,
    ).fillna(-1).astype("int16")
    base_columns = [
        "video_type",
        "upload_type",
        "duration_bucket",
        "aspect_bucket",
        "visible_status",
        "music_type",
    ]
    for column in base_columns:
        output[column] = output[column].astype("string").fillna("<MISSING>")
    output["metadata_signature"] = pd.util.hash_pandas_object(output[base_columns], index=False).astype("uint64")
    author = pd.to_numeric(output["author_id"], errors="coerce").fillna(-1).astype("int64")
    output["author_content_signature"] = pd.util.hash_pandas_object(
        pd.concat([author.rename("author_id"), output[base_columns]], axis=1),
        index=False,
    ).astype("uint64")
    return output


def prior_author_features(
    protocol: pd.DataFrame,
    history: dict[int, tuple[np.ndarray, np.ndarray]],
    video_to_author: dict[int, int],
) -> pd.DataFrame:
    """Measure target-author exposure using only events strictly before each query."""
    rows: list[dict[str, object]] = []
    for user_id, query_group in protocol.groupby("user_id", sort=False):
        times, videos = history.get(
            int(user_id),
            (np.asarray([], dtype="int64"), np.asarray([], dtype="int64")),
        )
        order = np.argsort(times, kind="stable")
        times, videos = times[order], videos[order]
        author_counts: dict[int, int] = {}
        cursor = 0
        for query in query_group.sort_values(["query_time_ms", "example_id"]).itertuples(index=False):
            while cursor < len(times) and int(times[cursor]) < int(query.query_time_ms):
                author = video_to_author.get(int(videos[cursor]))
                if author is not None:
                    author_counts[author] = author_counts.get(author, 0) + 1
                cursor += 1
            target_author = video_to_author.get(int(query.video_id))
            count = 0 if target_author is None else author_counts.get(target_author, 0)
            rows.append(
                {
                    "example_id": query.example_id,
                    "target_author_id": target_author,
                    "target_author_seen_before": count > 0,
                    "target_author_events_before": count,
                }
            )
    return pd.DataFrame(rows)


def rate(value: pd.Series) -> float:
    return float(value.mean()) if len(value) else float("nan")


def quantiles(value: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(value, errors="coerce").dropna()
    return {
        name: float(clean.quantile(q)) if len(clean) else float("nan")
        for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
    }


def build_feature_quality(candidates: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate and target fields after the target ID rename."""
    rows: list[dict[str, object]] = []
    for column in STATIC_COLUMNS:
        target_column = "target_video_id" if column == "video_id" else column
        rows.append(
            {
                "feature": column,
                "candidate_missing_rate": float(candidates[column].isna().mean()),
                "candidate_cardinality": int(candidates[column].nunique(dropna=True)),
                "target_missing_rate": float(audit[target_column].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = pd.read_parquet(args.protocol_dir / "queries.parquet")
    protocol = protocol.sort_values(["user_id", "query_time_ms", "example_id"]).reset_index(drop=True)
    candidate_ids = load_distinct_ids(args.adaptive_root / "sequence_examples" / args.split)
    train_ids = load_distinct_ids(args.adaptive_root / "sequence_examples/train")
    print("candidate/train IDs:", len(candidate_ids), len(train_ids), flush=True)

    static_path = args.base_gold_root / "item_features/static"
    candidates, video_to_author, warm_authors = load_static_metadata(
        static_path, candidate_ids, train_ids
    )
    candidates = add_content_signatures(candidates)
    metadata_counts = candidates["metadata_signature"].value_counts()
    author_counts = candidates["author_content_signature"].value_counts()
    candidates["metadata_collision_size"] = candidates["metadata_signature"].map(metadata_counts)
    candidates["author_content_collision_size"] = candidates["author_content_signature"].map(author_counts)
    candidates["author_warm"] = pd.to_numeric(candidates["author_id"], errors="coerce").isin(warm_authors)

    query_users = set(protocol["user_id"].astype("int64"))
    history = load_event_history(
        args.adaptive_root,
        args.split,
        query_users,
        int(protocol["query_time_ms"].max()) + 1,
    )
    author_history = prior_author_features(protocol, history, video_to_author)
    target_metadata = candidates.rename(columns={"video_id": "target_video_id"})
    audit = protocol.rename(columns={"video_id": "target_video_id"}).merge(
        target_metadata,
        on="target_video_id",
        how="left",
        validate="many_to_one",
    ).merge(author_history, on="example_id", how="left", validate="one_to_one")
    if audit["author_content_signature"].isna().any():
        raise RuntimeError("Some protocol targets are missing static item metadata")
    audit["target_author_warm"] = audit["target_author_id"].isin(warm_authors)
    audit.to_parquet(args.output_dir / "per_query_content_readiness.parquet", index=False)

    build_feature_quality(candidates, audit).to_csv(args.output_dir / "feature_quality.csv", index=False)

    cold = audit[~audit["target_warm"]]
    warm = audit[audit["target_warm"]]
    report = {
        "protocol": "adaptive_eval_v2_a3_content_readiness",
        "queries": int(len(audit)),
        "candidate_items": int(len(candidates)),
        "static_catalog_items": int(len(video_to_author)),
        "warm_authors": int(len(warm_authors)),
        "metadata_only_unique_signature_rate": rate(candidates["metadata_collision_size"].eq(1)),
        "author_content_unique_signature_rate": rate(candidates["author_content_collision_size"].eq(1)),
        "candidate_metadata_collision_size": quantiles(candidates["metadata_collision_size"]),
        "candidate_author_content_collision_size": quantiles(candidates["author_content_collision_size"]),
        "all_targets": {
            "author_available_rate": rate(audit["target_author_id"].notna()),
            "target_author_warm_rate": rate(audit["target_author_warm"]),
            "target_author_seen_before_rate": rate(audit["target_author_seen_before"]),
            "metadata_collision_size": quantiles(audit["metadata_collision_size"]),
            "author_content_collision_size": quantiles(audit["author_content_collision_size"]),
        },
        "warm_targets": {
            "queries": int(len(warm)),
            "target_author_warm_rate": rate(warm["target_author_warm"]),
            "target_author_seen_before_rate": rate(warm["target_author_seen_before"]),
        },
        "cold_targets": {
            "queries": int(len(cold)),
            "target_author_warm_rate": rate(cold["target_author_warm"]),
            "target_author_seen_before_rate": rate(cold["target_author_seen_before"]),
            "metadata_unique_rate": rate(cold["metadata_collision_size"].eq(1)),
            "author_content_unique_rate": rate(cold["author_content_collision_size"].eq(1)),
            "metadata_collision_size": quantiles(cold["metadata_collision_size"]),
            "author_content_collision_size": quantiles(cold["author_content_collision_size"]),
        },
        "available_content_fields": list(STATIC_COLUMNS),
        "missing_high_semantic_features": ["tag", "text_embedding", "audio_embedding", "visual_embedding"],
        "decision_rules": {
            "author_branch_promising": "cold target_author_seen_before_rate or target_author_warm_rate is materially above zero",
            "metadata_branch_limit": "large collision groups imply basic metadata cannot identify exact cold videos",
            "next_model": "retain adaptive user encoder; add author/content item branch before another ranker",
        },
    }
    save_json(args.output_dir / "content_readiness.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("wrote:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
