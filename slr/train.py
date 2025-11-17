from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import SignSequenceDataset, sign_sequence_collate
from .model import SignLanguageXLSTM, SignModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-stream xLSTM sign language classifier.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV.")
    parser.add_argument("--json-root", type=str, required=True, help="Root directory containing per-sample JSON folders.")
    parser.add_argument("--output-dir", type=str, default="experiments/xlstm", help="Directory to store checkpoints and logs.")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--max-seq-len", type=int, default=64, help="Clip or pad sequences to this many frames.")
    parser.add_argument("--min-seq-len", type=int, default=8)
    parser.add_argument("--no-random-clip", action="store_true", help="Disable random temporal crops for training.")
    parser.add_argument("--cache", action="store_true", help="Cache sequences in memory.")

    parser.add_argument("--train-split", type=str, help="Use metadata 'split' column and select this split for training.")
    parser.add_argument("--val-split", type=str, help="Use metadata 'split' column and select this split for validation.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Random validation ratio if split column is absent.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--stream-hidden", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=512)
    parser.add_argument("--xlstm-hidden", type=int, default=512)
    parser.add_argument("--xlstm-layers", type=int, default=2)
    parser.add_argument("--xlstm-variant", choices=["mlstm", "slstm"], default="mlstm")
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--resume", type=str, help="Path to a checkpoint to resume from.")
    parser.add_argument("--save-best-only", action="store_true", help="Only keep the best validation checkpoint.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_datasets(args: argparse.Namespace, label_mapping: Dict[str, int]):
    metadata_df = pd.read_csv(args.metadata)
    has_split = "split" in metadata_df.columns
    train_dataset = None
    val_dataset = None

    if args.train_split:
        if not has_split:
            raise ValueError("metadata does not contain a 'split' column but --train-split was provided")
        train_dataset = SignSequenceDataset(
            metadata_path=args.metadata,
            json_root=args.json_root,
            split=args.train_split,
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
            random_clip=not args.no_random_clip,
            cache_sequences=args.cache,
            label_mapping=label_mapping,
        )
        if args.val_split:
            val_dataset = SignSequenceDataset(
                metadata_path=args.metadata,
                json_root=args.json_root,
                split=args.val_split,
                max_seq_len=args.max_seq_len,
                min_seq_len=args.min_seq_len,
                random_clip=False,
                cache_sequences=args.cache,
                label_mapping=label_mapping,
            )
    else:
        sample_ids = metadata_df["sample_id"].astype(str).tolist()
        rng = np.random.default_rng(args.seed)
        rng.shuffle(sample_ids)
        val_count = int(len(sample_ids) * args.val_ratio)
        if val_count == 0:
            val_count = 1 if len(sample_ids) > 1 else 0
        val_ids = sample_ids[:val_count] if val_count > 0 else []
        train_ids = sample_ids[val_count:] if val_count > 0 else sample_ids
        if not train_ids:
            raise ValueError("No samples available for training after applying val_ratio.")

        train_dataset = SignSequenceDataset(
            metadata_path=args.metadata,
            json_root=args.json_root,
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
            random_clip=not args.no_random_clip,
            cache_sequences=args.cache,
            label_mapping=label_mapping,
            allowed_sample_ids=train_ids,
        )
        if val_ids:
            val_dataset = SignSequenceDataset(
                metadata_path=args.metadata,
                json_root=args.json_root,
                max_seq_len=args.max_seq_len,
                min_seq_len=args.min_seq_len,
                random_clip=False,
                cache_sequences=args.cache,
                label_mapping=label_mapping,
                allowed_sample_ids=val_ids,
            )

    return train_dataset, val_dataset


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ["pose", "hand", "face"]}
        mask = batch["mask"].to(device)
        inputs["mask"] = mask
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)["logits"]
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ["pose", "hand", "face"]}
        mask = batch["mask"].to(device)
        inputs["mask"] = mask
        labels = batch["labels"].to(device)

        outputs = model(inputs)["logits"]
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"

    metadata_df = pd.read_csv(args.metadata)
    label_mapping = {label: idx for idx, label in enumerate(sorted(metadata_df["label"].astype(str).unique()))}
    num_classes = len(label_mapping)

    train_dataset, val_dataset = create_datasets(args, label_mapping)
    if train_dataset is None:
        raise ValueError("Training dataset could not be created. Check metadata and split parameters.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=sign_sequence_collate,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=sign_sequence_collate,
        )

    config = SignModelConfig(
        num_classes=num_classes,
        stream_hidden=args.stream_hidden,
        fusion_dim=args.fusion_dim,
        xlstm_hidden=args.xlstm_hidden,
        xlstm_layers=args.xlstm_layers,
        xlstm_variant=args.xlstm_variant,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        pose_dim=train_dataset.pose_points * 3,
        hand_dim=train_dataset.hand_points * 3,
        face_dim=train_dataset.face_points * 3,
    )
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    labels_path = output_dir / "labels.json"
    label_list = [None] * num_classes
    for label, idx in label_mapping.items():
        label_list[idx] = str(label)
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump({"labels": label_list}, f, indent=2, ensure_ascii=False)

    model = SignLanguageXLSTM(config).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    start_epoch = 0
    best_acc = 0.0
    checkpoint_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "best.pt"

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        if scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.device, args.grad_clip)
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, args.device)
        if scheduler:
            scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
        }
        if val_metrics:
            record["val_loss"] = val_metrics["loss"]
            record["val_acc"] = val_metrics["accuracy"]
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f}"
            + (
                f" | val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}"
                if val_metrics
                else ""
            )
        )

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
        }
        if scheduler:
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, checkpoint_path)

        if val_metrics and val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            state["best_acc"] = best_acc
            torch.save(state, best_path)
        elif not args.save_best_only:
            torch.save(state, output_dir / f"epoch_{epoch+1:03d}.pt")

    print("Training complete.")


if __name__ == "__main__":
    main()
