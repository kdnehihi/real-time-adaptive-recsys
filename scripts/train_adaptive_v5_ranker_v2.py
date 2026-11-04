from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch import nn

from recommender.models.adaptive.checkpoint import (
    INPUT_COLUMNS,
    V5ResidualRanker,
    encode_items,
    load_artifacts,
    multi_positive_metrics,
    prepare_item_pool,
    read_parquet,
    retrieve_for_queries,
    sanitize_point_in_time_sequences,
    save_json,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V5 v2 with cached V4 candidates and listwise ranking loss.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-queries", type=int, default=80000)
    parser.add_argument("--max-validation-users", type=int, default=0)
    parser.add_argument("--train-candidates", type=int, default=128)
    parser.add_argument("--validation-candidates", type=int, default=1000)
    parser.add_argument("--popularity-blend-weight", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-query-size", type=int, default=64)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--retrieval-query-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--margin-weight", type=float, default=0.15)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hidden-dims", default="512,256,128")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_dims(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_balanced_ids(index_frame: pd.DataFrame, limit: int | None, latest_only: bool) -> list[str]:
    frame = index_frame.sort_values(["user_id", "as_of_time", "example_id"]).reset_index(drop=True)
    if latest_only:
        frame = frame.drop_duplicates("user_id", keep="last")
    if limit is None or len(frame) <= limit:
        return frame["example_id"].astype(str).tolist()

    user_ids = np.sort(frame["user_id"].dropna().unique())
    base_quota, remainder = divmod(limit, len(user_ids))
    selected: list[pd.DataFrame] = []
    for position, user_id in enumerate(user_ids):
        group = frame[frame["user_id"] == user_id]
        quota = min(max(base_quota + (1 if position < remainder else 0), 1), len(group))
        indices = np.linspace(0, len(group) - 1, num=quota, dtype=int)
        selected.append(group.iloc[np.unique(indices)])
    output = pd.concat(selected, ignore_index=True)
    return output.sort_values(["as_of_time", "example_id"]).head(limit)["example_id"].astype(str).tolist()


def load_queries(
    adaptive_root: Path,
    split: str,
    limit: int | None,
    latest_only: bool,
) -> pd.DataFrame:
    path = adaptive_root / "sequence_examples" / split
    index_frame = read_parquet(
        path,
        ["example_id", "user_id", "as_of_time"],
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    )
    selected_ids = select_balanced_ids(index_frame, limit, latest_only)
    if not selected_ids:
        raise ValueError(f"No strong-positive queries found in {path}")
    frame = read_parquet(
        path,
        INPUT_COLUMNS,
        None,
        (ds.field("target_class") == "STRONG_POSITIVE") & ds.field("example_id").isin(selected_ids),
    ).reset_index(drop=True)
    frame = sanitize_point_in_time_sequences(frame)
    return frame.sort_values(["as_of_time", "example_id"]).reset_index(drop=True)


def load_relevance(adaptive_root: Path, split: str) -> pd.DataFrame:
    return read_parquet(
        adaptive_root / "sequence_examples" / split,
        ["user_id", "video_id"],
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    ).drop_duplicates(["user_id", "video_id"]).reset_index(drop=True)


def find_item_positions(item_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    order = np.argsort(item_ids)
    sorted_ids = item_ids[order]
    positions = np.searchsorted(sorted_ids, target_ids)
    valid = positions < len(sorted_ids)
    output = np.full(len(target_ids), -1, dtype=np.int64)
    valid_rows = np.where(valid)[0]
    exact = sorted_ids[positions[valid_rows]] == target_ids[valid_rows]
    exact_rows = valid_rows[exact]
    output[exact_rows] = order[positions[exact_rows]]
    return output


def force_training_targets(
    candidate_idx: np.ndarray,
    candidate_scores: np.ndarray,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: np.ndarray,
    true_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    target_item_positions = find_item_positions(item_ids, true_ids)
    keep = target_item_positions >= 0
    if not keep.all():
        candidate_idx = candidate_idx[keep]
        candidate_scores = candidate_scores[keep]
        user_embeddings = user_embeddings[keep]
        target_item_positions = target_item_positions[keep]
    positive_positions = np.full(len(candidate_idx), candidate_idx.shape[1] - 1, dtype=np.int64)
    natural_hits = 0
    for row, item_position in enumerate(target_item_positions):
        matches = np.where(candidate_idx[row] == item_position)[0]
        if len(matches):
            positive_positions[row] = int(matches[0])
            natural_hits += 1
        else:
            candidate_idx[row, -1] = item_position
            candidate_scores[row, -1] = float(user_embeddings[row] @ item_embeddings[item_position])
    recall = natural_hits / max(len(candidate_idx), 1)
    return candidate_idx, candidate_scores, positive_positions, recall


def normalize_group_scores(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return np.nan_to_num((scores - mean) / (std + 1e-6), nan=0.0, posinf=5.0, neginf=-5.0).astype("float32")


@torch.no_grad()
def evaluate_ranker(
    ranker: V5ResidualRanker,
    user_embeddings: np.ndarray,
    candidate_idx: np.ndarray,
    candidate_scores: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: np.ndarray,
    query_frame: pd.DataFrame,
    relevance_frame: pd.DataFrame,
    device: torch.device,
    batch_queries: int,
) -> dict[str, dict[str, float]]:
    ranker.eval()
    normalized = normalize_group_scores(candidate_scores)
    reranked_parts: list[np.ndarray] = []
    for start in range(0, len(query_frame), batch_queries):
        end = min(start + batch_queries, len(query_frame))
        indices = candidate_idx[start:end]
        query_users = user_embeddings[start:end]
        flat_users = np.repeat(query_users[:, None, :], indices.shape[1], axis=1).reshape(-1, query_users.shape[1])
        flat_items = item_embeddings[indices].reshape(-1, item_embeddings.shape[1])
        flat_scores = normalized[start:end].reshape(-1)
        logits = ranker(
            torch.as_tensor(flat_users, device=device),
            torch.as_tensor(flat_items, device=device),
            torch.as_tensor(flat_scores, device=device),
        ).reshape(end - start, indices.shape[1])
        order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        reranked_parts.append(np.take_along_axis(indices, order, axis=1))
    reranked_idx = np.vstack(reranked_parts)
    top_ks = (10, 50, 100)
    query_users = query_frame["user_id"].to_numpy("int64")
    return {
        "v4": multi_positive_metrics(item_ids, candidate_idx, query_users, relevance_frame, top_ks),
        "v5": multi_positive_metrics(item_ids, reranked_idx, query_users, relevance_frame, top_ks),
    }


def save_checkpoint(
    path: Path,
    ranker: V5ResidualRanker,
    config: dict,
    v4_config: dict,
    epoch: int,
    metric: float,
) -> None:
    torch.save(
        {
            "model_state_dict": ranker.state_dict(),
            "config": config,
            "v4_config": v4_config,
            "best_epoch": epoch,
            "best_validation_ndcg@10": metric,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("V5 v2 full candidate generation requires a CUDA runtime.")
    hidden_dims = parse_dims(args.hidden_dims)
    config = {
        **vars(args),
        "adaptive_root": str(args.adaptive_root),
        "base_gold_root": str(args.base_gold_root),
        "v4_dir": str(args.v4_dir),
        "output_dir": str(args.output_dir),
        "hidden_dims": list(hidden_dims),
        "model_type": "residual_listwise_v2",
        "selection_metric": "validation_multi_positive_ndcg@10",
    }
    save_json(args.output_dir / "config.json", config)

    print("loading frozen V4 and train-fitted preprocessing...", flush=True)
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, None, device=device)
    print("loading balanced train queries...", flush=True)
    train_queries = load_queries(args.adaptive_root, "train", args.max_train_queries, latest_only=False)
    validation_limit = None if args.max_validation_users == 0 else args.max_validation_users
    validation_queries = load_queries(args.adaptive_root, "validation", validation_limit, latest_only=True)
    validation_relevance = load_relevance(args.adaptive_root, "validation")
    print("queries:", len(train_queries), len(validation_queries), flush=True)

    print("encoding the train item catalog once...", flush=True)
    item_pool = prepare_item_pool(args.adaptive_root, "train", None)
    item_ids, item_embeddings = encode_items(
        artifacts.v4,
        item_pool,
        train_queries.iloc[0].to_dict(),
        artifacts,
        batch_size=args.item_batch_size,
    )
    print("item embeddings:", item_embeddings.shape, flush=True)
    popularity = pd.to_numeric(
        item_pool.get("item_hist_events", pd.Series(np.zeros(len(item_pool)))),
        errors="coerce",
    ).fillna(0).to_numpy("float32")
    popularity = np.log1p(np.maximum(popularity, 0.0))
    popularity = (popularity - popularity.mean()) / (popularity.std() + 1e-6)
    item_score_bias = float(args.popularity_blend_weight) * np.nan_to_num(popularity, nan=0.0, posinf=0.0, neginf=0.0)

    print("caching V4 train candidates once...", flush=True)
    train_user_embeddings, train_candidate_idx, train_candidate_scores = retrieve_for_queries(
        artifacts,
        train_queries,
        item_ids,
        item_embeddings,
        args.train_candidates,
        args.retrieval_query_batch_size,
        item_score_bias=item_score_bias,
    )
    train_candidate_idx, train_candidate_scores, positive_positions, natural_train_recall = force_training_targets(
        train_candidate_idx,
        train_candidate_scores,
        train_user_embeddings,
        item_embeddings,
        item_ids,
        train_queries["video_id"].to_numpy("int64"),
    )
    print(f"natural train target recall@{args.train_candidates}: {natural_train_recall:.6f}", flush=True)

    print("caching V4 validation candidates once...", flush=True)
    validation_user_embeddings, validation_candidate_idx, validation_candidate_scores = retrieve_for_queries(
        artifacts,
        validation_queries,
        item_ids,
        item_embeddings,
        args.validation_candidates,
        args.retrieval_query_batch_size,
        item_score_bias=item_score_bias,
    )

    ranker = V5ResidualRanker(int(artifacts.v4_cfg["retrieval_dim"]), hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    cross_entropy = nn.CrossEntropyLoss()
    normalized_train_scores = normalize_group_scores(train_candidate_scores)

    baseline = evaluate_ranker(
        ranker,
        validation_user_embeddings,
        validation_candidate_idx,
        validation_candidate_scores,
        item_embeddings,
        item_ids,
        validation_queries,
        validation_relevance,
        device,
        args.batch_query_size,
    )
    best_metric = float(baseline["v5"]["ndcg@10"])
    best_epoch = 0
    save_checkpoint(args.output_dir / "best_model.pt", ranker, config, artifacts.v4_cfg, best_epoch, best_metric)
    history: list[dict] = [{"epoch": 0, "train_loss": None, "validation": baseline}]
    print("identity baseline:", json.dumps(baseline, indent=2), flush=True)

    bad_epochs = 0
    order_rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        ranker.train()
        order = order_rng.permutation(len(train_user_embeddings))
        loss_sum = 0.0
        seen = 0
        for step, start in enumerate(range(0, len(order), args.batch_query_size), start=1):
            rows = order[start : start + args.batch_query_size]
            indices = train_candidate_idx[rows]
            users = train_user_embeddings[rows]
            flat_users = np.repeat(users[:, None, :], indices.shape[1], axis=1).reshape(-1, users.shape[1])
            flat_items = item_embeddings[indices].reshape(-1, item_embeddings.shape[1])
            flat_scores = normalized_train_scores[rows].reshape(-1)
            logits = ranker(
                torch.as_tensor(flat_users, device=device),
                torch.as_tensor(flat_items, device=device),
                torch.as_tensor(flat_scores, device=device),
            ).reshape(len(rows), indices.shape[1])
            targets = torch.as_tensor(positive_positions[rows], device=device, dtype=torch.long)
            listwise_loss = cross_entropy(logits, targets)
            positive_logits = logits.gather(1, targets[:, None]).squeeze(1)
            negative_logits = logits.masked_fill(
                torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).bool(),
                float("-inf"),
            )
            hard_negative_logits = negative_logits.max(dim=1).values
            margin_loss = torch.nn.functional.softplus(hard_negative_logits - positive_logits).mean()
            base_scores = torch.as_tensor(normalized_train_scores[rows], device=device)
            delta_penalty = (logits - torch.nn.functional.softplus(ranker.raw_base_weight) * base_scores).square().mean()
            loss = listwise_loss + args.margin_weight * margin_loss + args.delta_l2_weight * delta_penalty
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite V5 v2 loss detected.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(rows)
            seen += len(rows)
            if step % 100 == 0:
                print(f"epoch={epoch} step={step} train_loss={loss_sum / seen:.5f}", flush=True)

        validation = evaluate_ranker(
            ranker,
            validation_user_embeddings,
            validation_candidate_idx,
            validation_candidate_scores,
            item_embeddings,
            item_ids,
            validation_queries,
            validation_relevance,
            device,
            args.batch_query_size,
        )
        train_loss = loss_sum / max(seen, 1)
        metric = float(validation["v5"]["ndcg@10"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "elapsed_min": (time.time() - started) / 60.0,
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if metric > best_metric + 1e-8:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(args.output_dir / "best_model.pt", ranker, config, artifacts.v4_cfg, best_epoch, best_metric)
            print("saved improved checkpoint", best_epoch, best_metric, flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stopping_patience:
                print("early stopping", flush=True)
                break

    summary = {
        "model_type": "residual_listwise_v2",
        "best_epoch": best_epoch,
        "best_validation_ndcg@10": best_metric,
        "natural_train_target_recall": natural_train_recall,
        "identity_baseline": baseline,
        "history": history,
        "accepted_over_v4": best_metric > float(baseline["v4"]["ndcg@10"]) + 1e-8,
    }
    save_json(args.output_dir / "metrics.json", summary)
    pd.DataFrame(
        {
            "epoch": [row["epoch"] for row in history],
            "train_loss": [row["train_loss"] for row in history],
            "validation_v4_ndcg@10": [row["validation"]["v4"]["ndcg@10"] for row in history],
            "validation_v5_ndcg@10": [row["validation"]["v5"]["ndcg@10"] for row in history],
        }
    ).to_csv(args.output_dir / "training_history.csv", index=False)
    print(json.dumps(summary, indent=2)[:8000], flush=True)
    print("artifacts:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
