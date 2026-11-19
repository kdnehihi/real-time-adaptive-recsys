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
import torch

from recommender.models.adaptive.author_branch import AuthorResidualRetrieval, AuthorVocabulary, encode_author_inputs
from recommender.models.adaptive.checkpoint import encode_items, encode_users, load_artifacts, save_json, select_device
from scripts.evaluate_adaptive_protocol_v2 import (
    build_safe_candidate_pool,
    clustered_bootstrap,
    load_event_history,
    load_protocol_queries,
    masked_exact_retrieval,
    metric_columns,
    per_query_metrics,
    segment_summary,
    user_macro_summary,
)
from scripts.train_adaptive_author_branch import load_video_author_map
from scripts.audit_adaptive_content_readiness import load_distinct_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4.1 author residual on protocol v2.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--author-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = tuple(int(value) for value in args.top_ks.split(","))
    device = select_device(args.device)
    query_features, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, "validation", None
    )
    candidate_pool = build_safe_candidate_pool(args.adaptive_root, "validation")
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, device=device)
    item_ids, v4_items = encode_items(
        artifacts.v4, candidate_pool, query_features.iloc[0].to_dict(), artifacts, args.item_batch_size
    )
    v4_users = encode_users(artifacts.v4, query_features, artifacts, batch_size=512)

    checkpoint = torch.load(args.author_dir / "best_model.pt", map_location="cpu")
    author_vocab = AuthorVocabulary(np.asarray(checkpoint["author_values"], dtype=np.int64))
    branch = AuthorResidualRetrieval(
        author_vocab.size,
        int(checkpoint["config"]["embedding_dim"]),
        float(checkpoint["config"]["dropout"]),
    )
    branch.load_state_dict(checkpoint["model_state_dict"])
    branch = branch.to(device).eval()
    train_ids = load_distinct_ids(args.adaptive_root / "sequence_examples/train")
    video_to_author, _ = load_video_author_map(args.base_gold_root / "item_features/static", train_ids)
    recent_author_ids, _ = encode_author_inputs(
        query_features, video_to_author, author_vocab, int(artifacts.v4_cfg["sequence_max_len"])
    )
    item_frame = pd.DataFrame({"video_id": item_ids, "recent_video_ids": [[] for _ in item_ids]})
    _, item_author_ids = encode_author_inputs(item_frame, video_to_author, author_vocab, 1)
    with torch.no_grad():
        v4_user_tensor = torch.as_tensor(v4_users, device=device)
        v4_item_tensor = torch.as_tensor(v4_items, device=device)
        combined_user, combined_item = branch.concatenate_embeddings(
            v4_user_tensor,
            v4_item_tensor,
            recent_author_ids.to(device),
            item_author_ids.to(device),
        )
        zero_author = torch.zeros(
            (len(v4_users), combined_user.shape[1] - v4_user_tensor.shape[1]), device=device
        )
        baseline_user = torch.cat([v4_user_tensor, zero_author], dim=1)
        combined_user = combined_user.cpu().numpy().astype("float32")
        combined_item = combined_item.cpu().numpy().astype("float32")
        baseline_user = baseline_user.cpu().numpy().astype("float32")

    popularity = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0))
    popularity = np.nan_to_num((popularity - popularity.mean()) / (popularity.std() + 1e-6)).astype("float32")
    history = load_event_history(
        args.adaptive_root,
        "validation",
        set(protocol["user_id"].astype("int64")),
        int(protocol["query_time_ms"].max()) + 1,
    )
    ranked, diagnostics, repeat_flags = masked_exact_retrieval(
        combined_user,
        baseline_user,
        combined_item,
        item_ids,
        candidate_pool["eligible_from_ms"].to_numpy("int64"),
        popularity,
        protocol,
        history,
        max(top_ks),
        args.query_batch_size,
        0.02,
        device,
    )
    protocol = protocol.copy()
    protocol["target_repeated_full_history"] = repeat_flags
    names = {
        "popularity": "popularity",
        "v4_static_ablation": "v4_adaptive_baseline",
        "v4_adaptive": "v4_1_author",
        "v4_adaptive_popularity": "v4_1_author_popularity",
    }
    metrics = metric_columns(top_ks)
    summaries, segment_frames, confidence = [], [], {}
    for internal, ranked_ids in ranked.items():
        name = names[internal]
        frame = per_query_metrics(ranked_ids, protocol, top_ks)
        frame.insert(0, "variant", name)
        frame.to_parquet(args.output_dir / f"per_query_{name}.parquet", index=False)
        summaries.append({"variant": name, **user_macro_summary(frame, metrics)})
        segment = segment_summary(frame, metrics)
        segment.insert(0, "variant", name)
        segment_frames.append(segment)
        confidence[name] = clustered_bootstrap(frame, metrics, args.bootstrap_samples, 44)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(segment_frames, ignore_index=True).to_csv(args.output_dir / "segments.csv", index=False)
    report = {
        "protocol": "adaptive_eval_v2_v4_1_author",
        "author_checkpoint_epoch": checkpoint.get("epoch"),
        "author_scale": float(branch.scale.detach().cpu()),
        "diagnostics": diagnostics,
        "summary": {row["variant"]: {k: v for k, v in row.items() if k != "variant"} for row in summaries},
        "clustered_user_bootstrap_ci95": confidence,
    }
    save_json(args.output_dir / "v4_1_author_evaluation.json", report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"author_scale": report["author_scale"], "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
