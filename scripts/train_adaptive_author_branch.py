from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from recommender.models.adaptive.author_branch import AuthorResidualRetrieval, AuthorVocabulary, encode_author_inputs
from recommender.models.adaptive.checkpoint import INPUT_COLUMNS, encode_frame, load_artifacts, move, parquet_files, sanitize_point_in_time_sequences, save_json, select_device
from scripts.audit_adaptive_content_readiness import STATIC_COLUMNS, load_distinct_ids
from scripts.evaluate_adaptive_protocol_v2 import read_shard


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 1024
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    embedding_dim: int = 32
    dropout: float = 0.1
    temperature: float = 0.05
    max_train_queries: int = 0
    max_validation_queries: int = 100_000
    validation_batches: int = 50
    patience: int = 2
    seed: int = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an author residual branch on top of frozen V4.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-validation-queries", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_video_author_map(static_path: Path, train_ids: set[int]) -> tuple[dict[int, int], AuthorVocabulary]:
    mapping: dict[int, int] = {}
    train_authors: set[int] = set()
    files = parquet_files(static_path)
    for index, file_path in enumerate(files, start=1):
        part = read_shard(file_path, ["video_id", "author_id"]).dropna()
        videos = part["video_id"].astype("int64")
        authors = part["author_id"].astype("int64")
        mapping.update(zip(videos, authors))
        train_authors.update(int(author) for video, author in zip(videos, authors) if int(video) in train_ids)
        print(f"author map: {index}/{len(files)} shards; videos={len(mapping):,}; train authors={len(train_authors):,}", flush=True)
    return mapping, AuthorVocabulary(np.asarray(sorted(train_authors), dtype=np.int64))


def iter_positive_batches(
    path: Path,
    batch_size: int,
    seed: int,
    maximum: int,
) -> Iterator[pd.DataFrame]:
    files = parquet_files(path)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    emitted = 0
    for shard_index in order:
        frame = read_shard(files[int(shard_index)], INPUT_COLUMNS)
        frame = frame[frame["target_class"].eq("STRONG_POSITIVE")]
        if frame.empty:
            continue
        frame = sanitize_point_in_time_sequences(frame)
        frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
        for start in range(0, len(frame), batch_size):
            part = frame.iloc[start : start + batch_size].reset_index(drop=True)
            if maximum and emitted + len(part) > maximum:
                part = part.head(maximum - emitted)
            if len(part):
                emitted += len(part)
                yield part
            if maximum and emitted >= maximum:
                return


def batch_logits(
    frame: pd.DataFrame,
    artifacts,
    branch: AuthorResidualRetrieval,
    video_to_author: dict[int, int],
    author_vocab: AuthorVocabulary,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    encoded = move(encode_frame(frame, artifacts.vocabs, artifacts.numeric, int(artifacts.v4_cfg["sequence_max_len"])), device)
    recent_authors, target_authors = encode_author_inputs(
        frame, video_to_author, author_vocab, int(artifacts.v4_cfg["sequence_max_len"])
    )
    recent_authors, target_authors = recent_authors.to(device), target_authors.to(device)
    with torch.no_grad():
        v4_user = artifacts.v4.encode_user(encoded)
        v4_item = artifacts.v4.encode_item(encoded)
        base_scores = v4_user @ v4_item.T
    return (base_scores + branch.score(recent_authors, target_authors)) / temperature


@torch.no_grad()
def evaluate(
    path: Path,
    artifacts,
    branch: AuthorResidualRetrieval,
    video_to_author: dict[int, int],
    author_vocab: AuthorVocabulary,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    branch.eval()
    losses, hits10, hits50, examples = [], 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    for batch_index, frame in enumerate(iter_positive_batches(path, cfg.batch_size, cfg.seed + 900, cfg.max_validation_queries)):
        if batch_index >= cfg.validation_batches:
            break
        logits = batch_logits(frame, artifacts, branch, video_to_author, author_vocab, device, cfg.temperature)
        labels = torch.arange(len(frame), device=device)
        losses.append(float(loss_fn(logits, labels)))
        order = torch.argsort(logits, dim=1, descending=True)
        hits10 += int(order[:, :10].eq(labels[:, None]).any(1).sum())
        hits50 += int(order[:, :50].eq(labels[:, None]).any(1).sum())
        examples += len(frame)
    return {
        "loss": float(np.mean(losses)),
        "hitrate@10": hits10 / max(examples, 1),
        "hitrate@50": hits50 / max(examples, 1),
        "examples": examples,
    }


def save_checkpoint(path: Path, branch, author_vocab, cfg, epoch, metric) -> None:
    torch.save(
        {
            "model_state_dict": branch.state_dict(),
            "author_values": author_vocab.values,
            "config": asdict(cfg),
            "epoch": epoch,
            "validation_hitrate@50": metric,
            "model_type": "v4_1_frozen_v4_author_residual",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        max_train_queries=args.max_train_queries,
        max_validation_queries=args.max_validation_queries,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print("device/config:", device, json.dumps(asdict(cfg), indent=2), flush=True)

    train_ids = load_distinct_ids(args.adaptive_root / "sequence_examples/train")
    video_to_author, author_vocab = load_video_author_map(
        args.base_gold_root / "item_features/static", train_ids
    )
    print("author vocabulary:", author_vocab.size, flush=True)
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, device=device)
    artifacts.v4.eval()
    for parameter in artifacts.v4.parameters():
        parameter.requires_grad_(False)

    branch = AuthorResidualRetrieval(author_vocab.size, cfg.embedding_dim, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history_path = args.output_dir / "training_history.csv"
    history: list[dict[str, float | int]] = []
    if args.resume and history_path.exists():
        previous = pd.read_csv(history_path)
        history = previous.to_dict(orient="records")
    start_epoch, best_metric, bad_epochs = 1, -1.0, 0
    last_path = args.output_dir / "last_checkpoint.pt"
    if args.resume and last_path.exists():
        # The resume checkpoint is created by this script and includes optimizer
        # and NumPy-backed metadata, so request the trusted full checkpoint.
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        branch.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
        print("resuming at epoch", start_epoch, flush=True)
        if start_epoch > cfg.epochs and (args.output_dir / "best_model.pt").exists():
            print(
                f"training already complete at epoch {start_epoch - 1}; "
                "keeping existing checkpoints and reports",
                flush=True,
            )
            return

    for epoch in range(start_epoch, cfg.epochs + 1):
        branch.train()
        started, total_loss, examples = time.time(), 0.0, 0
        for step, frame in enumerate(
            iter_positive_batches(
                args.adaptive_root / "sequence_examples/train",
                cfg.batch_size,
                cfg.seed + epoch,
                cfg.max_train_queries,
            ),
            start=1,
        ):
            logits = batch_logits(frame, artifacts, branch, video_to_author, author_vocab, device, cfg.temperature)
            labels = torch.arange(len(frame), device=device)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite author-branch loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss) * len(frame)
            examples += len(frame)
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} examples={examples:,} loss={total_loss/examples:.4f}", flush=True)
        validation = evaluate(
            args.adaptive_root / "sequence_examples/validation",
            artifacts,
            branch,
            video_to_author,
            author_vocab,
            cfg,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(examples, 1),
            "train_examples": examples,
            **{f"validation_{key}": value for key, value in validation.items()},
            "elapsed_minutes": (time.time() - started) / 60,
            "author_scale": float(branch.scale.detach().cpu()),
        }
        history.append(row)
        print(row, flush=True)
        metric = float(validation["hitrate@50"])
        if metric > best_metric:
            best_metric, bad_epochs = metric, 0
            save_checkpoint(args.output_dir / "best_model.pt", branch, author_vocab, cfg, epoch, metric)
        else:
            bad_epochs += 1
        torch.save(
            {
                "model_state_dict": branch.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "author_values": author_vocab.values,
                "config": asdict(cfg),
                "epoch": epoch,
                "best_metric": best_metric,
            },
            last_path,
        )
        if bad_epochs >= cfg.patience:
            print("early stopping", flush=True)
            break

    pd.DataFrame(history).to_csv(history_path, index=False)
    save_json(args.output_dir / "config.json", asdict(cfg))
    save_json(
        args.output_dir / "metrics.json",
        {"history": history, "best_validation_hitrate@50": best_metric, "author_vocab_size": author_vocab.size},
    )
    print("wrote:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
