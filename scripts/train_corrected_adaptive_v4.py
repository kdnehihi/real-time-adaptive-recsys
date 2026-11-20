from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from recommender.models.adaptive.checkpoint import (
    INPUT_COLUMNS,
    ITEM_CATEGORICAL,
    ITEM_ID_FEATURES,
    ITEM_NUMERIC,
    SEQUENCE_COLUMNS,
    NumericalPreprocessor,
    V4TransformerRetrieval,
    build_vocabs,
    encode_frame,
    minimal_user_stub,
    move,
    save_json,
)


CORRECTED_COLUMNS = (
    "current_time_ms",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4 from scratch on corrected strict-prior sequences.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--prior-best-checkpoint", type=Path)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--max-item-pool-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--negative-sample-count", type=int, default=8192)
    parser.add_argument("--popular-negative-fraction", type=float, default=0.7)
    parser.add_argument("--item-popularity-alpha", type=float, default=0.75)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--hf-path-prefix", default="v4_corrected/v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite-prepared", action="store_true")
    return parser.parse_args()


def quote(path: Path) -> str:
    return str(path).replace("'", "''")


def parquet_glob(path: Path) -> str:
    return quote(path / "**/*.parquet")


def model_config(args: argparse.Namespace) -> dict[str, Any]:
    """Keep the selected V4 architecture and negative strategy unchanged."""
    return {
        "adaptive_root": str(args.adaptive_root),
        "base_gold_root": str(args.base_gold_root),
        "corrected_root": str(args.corrected_root),
        "seed": args.seed,
        "max_train_examples": None if not args.max_train_examples else args.max_train_examples,
        "max_validation_examples": None if not args.max_validation_examples else args.max_validation_examples,
        "max_item_pool_examples": None if not args.max_item_pool_examples else args.max_item_pool_examples,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "retrieval_dim": 128,
        "user_hidden_dims": (768, 384),
        "item_hidden_dims": (768, 384),
        "transformer_dim": 256,
        "transformer_heads": 4,
        "transformer_layers": 2,
        "transformer_dropout": 0.1,
        "dropout": 0.12,
        "temperature": 0.05,
        "sequence_max_len": 30,
        "negative_sample_count": args.negative_sample_count,
        "popular_negative_fraction": args.popular_negative_fraction,
        "item_popularity_column": "item_hist_events",
        "item_popularity_alpha": args.item_popularity_alpha,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_metric": "ndcg@50",
        "data_contract": "corrected_adaptive_sequences_v2",
    }


def prepare_positive_split(args: argparse.Namespace, split: str, limit: int) -> Path:
    """Join immutable Gold features to corrected arrays out of core by example_id."""
    output = args.work_dir / f"corrected_positive_{split}.parquet"
    if output.exists() and not args.overwrite_prepared:
        print("using prepared split:", output, flush=True)
        return output
    output.unlink(missing_ok=True)
    base = args.adaptive_root / "sequence_examples" / split
    corrected = args.corrected_root / "sequences" / f"split={split}"
    if not base.exists() or not corrected.exists():
        raise FileNotFoundError(f"Missing split inputs: {base} | {corrected}")
    base_columns = [name for name in INPUT_COLUMNS if name not in set(CORRECTED_COLUMNS)]
    select_base = ",\n".join(f'b."{name}"' for name in base_columns)
    select_corrected = ",\n".join(f'd."{name}"' for name in CORRECTED_COLUMNS)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
        COPY (
            SELECT {select_base}, {select_corrected}
            FROM read_parquet('{parquet_glob(base)}', union_by_name=true) AS b
            INNER JOIN read_parquet('{parquet_glob(corrected)}', union_by_name=true) AS d
            USING (example_id)
            WHERE b.target_class = 'STRONG_POSITIVE'
            {limit_sql}
        ) TO '{quote(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    print(f"materializing corrected {split} positives...", flush=True)
    duckdb.connect().execute(sql).close()
    return output


def prepare_item_pool(args: argparse.Namespace) -> Path:
    """Materialize one latest train snapshot per item for vocab and explicit negatives."""
    output = args.work_dir / "train_item_pool.parquet"
    if output.exists() and not args.overwrite_prepared:
        print("using prepared item pool:", output, flush=True)
        return output
    output.unlink(missing_ok=True)
    columns = [*ITEM_ID_FEATURES, *ITEM_CATEGORICAL, *ITEM_NUMERIC]
    selected = ", ".join(f'"{name}"' for name in columns)
    limit_sql = f"LIMIT {int(args.max_item_pool_examples)}" if args.max_item_pool_examples else ""
    source = args.adaptive_root / "sequence_examples/train"
    sql = f"""
        COPY (
            SELECT {selected} FROM (
                SELECT {selected}, row_number() OVER (
                    PARTITION BY video_id ORDER BY current_time_ms DESC, example_id DESC
                ) AS row_number
                FROM read_parquet('{parquet_glob(source)}', union_by_name=true)
            ) WHERE row_number = 1
            {limit_sql}
        ) TO '{quote(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    print("materializing train item pool...", flush=True)
    duckdb.connect().execute(sql).close()
    return output


class FrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> int:
        return index


class FrameCollator:
    def __init__(self, frame: pd.DataFrame, vocabs, numeric, max_len: int):
        self.frame = frame
        self.vocabs = vocabs
        self.numeric = numeric
        self.max_len = max_len

    def __call__(self, indices: list[int]):
        part = self.frame.iloc[indices].reset_index(drop=True)
        return encode_frame(part, self.vocabs, self.numeric, self.max_len)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_payload(model, config, vocab_sizes, epoch, history, best_metric, best_epoch):
    """Store resumable FP16 weights without duplicating the optimizer's multi-GB state."""
    state = {
        name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    return {
        "model_state_dict": state,
        "config": config,
        "vocab_sizes": vocab_sizes,
        "epoch": epoch,
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "resume_contract": "weights_and_history_only_optimizer_resets",
    }


def upload_checkpoint(path: Path, args: argparse.Namespace, remote_name: str) -> None:
    if not args.hf_repo_id:
        return
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise RuntimeError("HF_WRITE_TOKEN is required for checkpoint upload")
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        repo_id=args.hf_repo_id,
        repo_type="model",
        path_or_fileobj=str(path),
        path_in_repo=f"{args.hf_path_prefix}/{remote_name}",
        commit_message=f"Update corrected V4 {remote_name}",
    )
    print("uploaded checkpoint:", remote_name, flush=True)


def evaluate_in_batch(model, loader, loss_fn, device, temperature, maximum_batches):
    model.eval()
    totals = {"loss": 0.0, "rows": 0}
    hits = {10: 0, 50: 0, 100: 0}
    ndcg = {10: 0.0, 50: 0.0, 100: 0.0}
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if index >= maximum_batches:
                break
            batch = move(raw, device)
            user = model.encode_user(batch)
            item = model.encode_item(batch)
            logits = torch.nan_to_num(user @ item.T / temperature, nan=-1e4, posinf=1e4, neginf=-1e4)
            labels = torch.arange(len(user), device=device)
            loss = loss_fn(logits, labels)
            order = torch.argsort(logits, dim=1, descending=True)
            for k in hits:
                match = order[:, :k].eq(labels[:, None])
                found = match.any(dim=1)
                position = match.float().argmax(dim=1) + 1
                hits[k] += int(found.sum())
                ndcg[k] += float((found.float() / torch.log2(position.float() + 1.0)).sum())
            totals["loss"] += float(loss) * len(user)
            totals["rows"] += len(user)
    rows = max(totals["rows"], 1)
    result = {"loss": totals["loss"] / rows, "examples": totals["rows"]}
    for k in hits:
        result[f"hitrate@{k}"] = hits[k] / rows
        result[f"ndcg@{k}"] = ndcg[k] / rows
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type != "cuda" and not args.max_train_examples:
        raise RuntimeError("Full corrected V4 training requires a CUDA GPU")
    print("device:", device, flush=True)

    train_path = prepare_positive_split(args, "train", args.max_train_examples)
    validation_path = prepare_positive_split(args, "validation", args.max_validation_examples)
    item_pool_path = prepare_item_pool(args)
    print("loading prepared frames...", flush=True)
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    item_pool = pd.read_parquet(item_pool_path)
    if train.empty or validation.empty or item_pool.empty:
        raise ValueError("Prepared training inputs must be non-empty")
    print("train/validation/items:", len(train), len(validation), len(item_pool), flush=True)
    strict_violations = sum(
        any(float(ts) >= float(current) for ts in (times if isinstance(times, (list, np.ndarray)) else []))
        for times, current in zip(train["recent_time_ms"], train["current_time_ms"])
    )
    if strict_violations:
        raise AssertionError(f"Corrected train sequences contain {strict_violations} strict-prior violations")

    config = model_config(args)
    numeric = NumericalPreprocessor(args.base_gold_root / "transforms/numeric_stats.json")
    vocabs = build_vocabs(args.base_gold_root, train[["user_id"]], item_pool)
    vocab_sizes = {name: vocab.size for name, vocab in vocabs.items()}
    model = V4TransformerRetrieval(vocab_sizes, config).to(device)
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = 0
    if args.resume_checkpoint and args.resume_checkpoint.exists():
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        shutil.copy2(args.resume_checkpoint, args.output_dir / "latest_resume_fp16.pt")
        start_epoch = int(resume.get("epoch", 0)) + 1
        history = list(resume.get("history", []))
        if args.prior_best_checkpoint and args.prior_best_checkpoint.exists():
            shutil.copy2(args.prior_best_checkpoint, args.output_dir / "best_model.pt")
            best_metric = float(resume.get("best_metric", -math.inf))
            best_epoch = int(resume.get("best_epoch", 0))
        else:
            best_metric, best_epoch = -math.inf, 0
            print("prior best checkpoint unavailable; next completed epoch becomes the new best", flush=True)
        print(f"resumed weights at epoch {start_epoch - 1}; optimizer intentionally reset", flush=True)

    save_json(args.output_dir / "config.json", config)
    save_json(
        args.output_dir / "metrics.json",
        {"best_epoch": best_epoch, "best_metric": best_metric, "history": history},
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)

    train_loader = DataLoader(
        FrameDataset(train), batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=FrameCollator(train, vocabs, numeric, 30), pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        FrameDataset(validation), batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
        collate_fn=FrameCollator(validation, vocabs, numeric, 30), pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)
    popularity = pd.to_numeric(item_pool.get("item_hist_events", 0), errors="coerce").fillna(0).to_numpy("float64")
    weights = np.power(np.maximum(popularity, 0) + 1, args.item_popularity_alpha)
    popularity_probabilities = weights / weights.sum()
    source_row = train.iloc[0].to_dict()

    def negative_batch(count: int):
        popular_count = int(round(count * args.popular_negative_fraction))
        indices = np.concatenate(
            [
                rng.choice(len(item_pool), popular_count, replace=True, p=popularity_probabilities),
                rng.integers(0, len(item_pool), count - popular_count),
            ]
        )
        frame = item_pool.iloc[indices].reset_index(drop=True)
        frame = frame.assign(**minimal_user_stub(source_row, len(frame)))
        return move(encode_frame(frame, vocabs, numeric, 30), device)

    bad_epochs = 0
    temperature = float(config["temperature"])
    amp_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        started = time.perf_counter()
        for step, raw in enumerate(train_loader, start=1):
            batch = move(raw, device)
            negatives = negative_batch(args.negative_sample_count)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                user = model.encode_user(batch)
                positive = model.encode_item(batch)
                explicit = model.encode_item(negatives)
                logits = torch.nan_to_num(user @ torch.cat([positive, explicit]).T / temperature, nan=-1e4, posinf=1e4, neginf=-1e4)
                labels = torch.arange(len(user), device=device)
                loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(user)
            seen += len(user)
            if step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum/max(seen, 1):.5f}", flush=True)

        validation_metrics = evaluate_in_batch(
            model, validation_loader, loss_fn, device, temperature, args.validation_batches
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "elapsed_minutes": (time.perf_counter() - started) / 60,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        metric = float(validation_metrics["ndcg@50"])
        improved = metric > best_metric
        if improved:
            best_metric, best_epoch, bad_epochs = metric, epoch, 0
            payload = checkpoint_payload(model, config, vocab_sizes, epoch, history, best_metric, best_epoch)
            torch.save(payload, args.output_dir / "best_model.pt")
            del payload
            gc.collect()
        else:
            bad_epochs += 1

        latest = checkpoint_payload(model, config, vocab_sizes, epoch, history, best_metric, best_epoch)
        torch.save(latest, args.output_dir / "latest_resume_fp16.pt")
        del latest
        gc.collect()
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        save_json(args.output_dir / "config.json", config)
        save_json(
            args.output_dir / "metrics.json",
            {"best_epoch": best_epoch, "best_metric": best_metric, "history": history},
        )
        if epoch % args.checkpoint_every == 0:
            upload_checkpoint(args.output_dir / "latest_resume_fp16.pt", args, "latest_resume_fp16.pt")
        if bad_epochs >= args.early_stopping_patience:
            print("early stopping", flush=True)
            break

    upload_checkpoint(args.output_dir / "best_model.pt", args, "best_model.pt")
    upload_checkpoint(args.output_dir / "latest_resume_fp16.pt", args, "latest_resume_fp16.pt")
    print(f"training complete: best_epoch={best_epoch} best_ndcg@50={best_metric}", flush=True)


if __name__ == "__main__":
    main()
