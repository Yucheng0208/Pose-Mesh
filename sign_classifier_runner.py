"""
CLI utilities to train, evaluate, and run the multi-stream sign classifier.

Expected data format
--------------------
Provide a manifest JSON file containing a list of samples. Each sample should
specify file paths (absolute or relative to the manifest) for body, hand, and
face keypoint sequences plus an optional label:

[
  {
    "id": "sample-0001",
    "body": "sequences/sample-0001_body.npy",
    "hand": "sequences/sample-0001_hand.npy",
    "face": "sequences/sample-0001_face.npy",
    "label": "hello"
  }
]

Each referenced file must store a numpy array of shape [T, F] where F equals the
number of joints multiplied by the coordinate dimension (typically 3 for
x/y/confidence). All three streams are trimmed to the shortest length so they
stay temporally aligned.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sign_classifier import (
    MultiStreamSignClassifier,
    StreamConfig,
    XlstmConfig,
)


# -----------------------------------------------------------------------------
# Dataset + dataloader utilities
# -----------------------------------------------------------------------------


def _load_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # pick the first array entry
        key = arr.files[0]
        arr = arr[key]
    return np.asarray(arr, dtype=np.float32)


def _normalize_path(sample_path: str, root: Path) -> str:
    path = Path(sample_path)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def load_manifest(path: str) -> List[Dict]:
    manifest_path = Path(path).resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base = manifest_path.parent
    normalized = []
    for sample in data:
        item = sample.copy()
        for key in ("body", "hand", "face"):
            if key not in item:
                raise ValueError(f"Sample missing required key '{key}': {sample}")
            item[key] = _normalize_path(item[key], base)
        normalized.append(item)
    return normalized


def build_label_map(samples: Iterable[Dict]) -> Dict[str, int]:
    labels = sorted({s["label"] for s in samples if "label" in s})
    if not labels:
        raise ValueError("Training manifest must include at least one labeled sample.")
    return {label: idx for idx, label in enumerate(labels)}


class KeypointSequenceDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        label_to_index: Optional[Dict[str, int]] = None,
        require_labels: bool = False,
    ) -> None:
        self.samples = samples
        self.label_to_index = label_to_index
        self.require_labels = require_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        body = torch.from_numpy(_load_array(sample["body"]))
        hand = torch.from_numpy(_load_array(sample["hand"]))
        face = torch.from_numpy(_load_array(sample["face"]))

        seq_len = min(body.shape[0], hand.shape[0], face.shape[0])
        if seq_len <= 0:
            raise ValueError(f"Empty sequence encountered in sample {sample}")
        body = body[:seq_len]
        hand = hand[:seq_len]
        face = face[:seq_len]

        label_name = sample.get("label")
        label_idx = -1
        if label_name is not None and self.label_to_index is not None:
            if label_name not in self.label_to_index:
                raise KeyError(f"Label '{label_name}' not found in label map.")
            label_idx = self.label_to_index[label_name]
        elif self.require_labels:
            raise ValueError(f"Sample {sample} is missing a label.")

        return {
            "body": body,
            "hand": hand,
            "face": face,
            "length": seq_len,
            "label": label_idx,
            "id": sample.get("id", f"sample-{idx:06d}"),
        }


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max())

    def pad_stream(key: str) -> torch.Tensor:
        feat_dim = batch[0][key].shape[1]
        padded = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        for i, item in enumerate(batch):
            seq = item[key]
            padded[i, : seq.shape[0]] = seq
        return padded

    body = pad_stream("body")
    hand = pad_stream("hand")
    face = pad_stream("face")
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "body": body,
        "hand": hand,
        "face": face,
        "lengths": lengths,
        "labels": labels,
        "ids": [item["id"] for item in batch],
    }


def create_dataloader(
    samples: List[Dict],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    label_to_index: Optional[Dict[str, int]],
    require_labels: bool,
) -> DataLoader:
    dataset = KeypointSequenceDataset(
        samples=samples,
        label_to_index=label_to_index,
        require_labels=require_labels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )


# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------


def build_model(args: argparse.Namespace, num_classes: int) -> MultiStreamSignClassifier:
    coord_dim = args.coordinate_dims
    body_cfg = StreamConfig(
        input_dim=args.num_body_joints * coord_dim,
        hidden_dim=args.body_hidden,
        dropout=args.stream_dropout,
    )
    hand_cfg = StreamConfig(
        input_dim=args.num_hand_joints * coord_dim,
        hidden_dim=args.hand_hidden,
        dropout=args.stream_dropout,
    )
    face_cfg = StreamConfig(
        input_dim=args.num_face_joints * coord_dim,
        hidden_dim=args.face_hidden,
        dropout=args.stream_dropout,
    )
    xlstm_cfg = XlstmConfig(
        hidden_dims=tuple(args.xlstm_hidden),
        variant=args.xlstm_variant,
        dropout=args.xlstm_dropout,
        bidirectional=args.xlstm_bidirectional,
    )
    model = MultiStreamSignClassifier(
        body_stream=body_cfg,
        hand_stream=hand_cfg,
        face_stream=face_cfg,
        num_classes=num_classes,
        fusion_dim=args.fusion_dim,
        xlstm_config=xlstm_cfg,
        attn_dim=args.attn_dim,
        final_dropout=args.final_dropout,
    )
    return model


def _forward_batch(
    model: MultiStreamSignClassifier,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = {
        "body": batch["body"].to(device),
        "hand": batch["hand"].to(device),
        "face": batch["face"].to(device),
    }
    lengths = batch["lengths"].to(device)
    output = model(inputs, lengths=lengths)
    logits = output["logits"]
    labels = batch["labels"].to(device)
    return logits, labels


def train_epoch(
    model: MultiStreamSignClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        logits, labels = _forward_batch(model, batch, device)
        supervised_mask = labels >= 0
        if not supervised_mask.any():
            continue

        loss = criterion(logits[supervised_mask], labels[supervised_mask])
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * supervised_mask.sum().item()
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions[supervised_mask] == labels[supervised_mask]).sum().item()
        total_count += supervised_mask.sum().item()

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: MultiStreamSignClassifier,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        logits, labels = _forward_batch(model, batch, device)
        supervised_mask = labels >= 0
        if not supervised_mask.any():
            continue
        loss = criterion(logits[supervised_mask], labels[supervised_mask])
        total_loss += loss.item() * supervised_mask.sum().item()
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions[supervised_mask] == labels[supervised_mask]).sum().item()
        total_count += supervised_mask.sum().item()

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def save_checkpoint(
    path: Path,
    model: MultiStreamSignClassifier,
    optimizer: Optional[torch.optim.Optimizer],
    label_map: Dict[str, int],
    args: argparse.Namespace,
    epoch: int,
    best_metric: float,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "label_to_index": label_map,
        "args": vars(args),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: torch.device,
    args_override: Optional[argparse.Namespace] = None,
) -> Tuple[MultiStreamSignClassifier, Dict[str, int]]:
    checkpoint = torch.load(path, map_location=device)
    label_map = checkpoint["label_to_index"]
    ckpt_args = argparse.Namespace(**checkpoint["args"])
    if args_override is not None:
        for key, value in vars(args_override).items():
            setattr(ckpt_args, key, value)
    model = build_model(ckpt_args, num_classes=len(label_map))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, label_map


# -----------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------


def run_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    train_samples = load_manifest(args.manifest)
    label_map = build_label_map(train_samples)
    val_samples = load_manifest(args.val_manifest) if args.val_manifest else None

    train_loader = create_dataloader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        label_to_index=label_map,
        require_labels=True,
    )
    val_loader = (
        create_dataloader(
            val_samples,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            label_to_index=label_map,
            require_labels=False,
        )
        if val_samples
        else None
    )

    model = build_model(args, num_classes=len(label_map)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_metric = -math.inf
    patience_counter = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip
        )
        if scheduler:
            scheduler.step()

        log = f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
        if val_loader:
            val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
            log += f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            metric = val_acc
        else:
            metric = train_acc
        print(log)

        if metric > best_metric:
            best_metric = metric
            patience_counter = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                label_map,
                args,
                epoch,
                best_metric,
            )
            print(f"✅ Saved new best checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("⏹️ Early stopping triggered.")
                break


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for eval mode.")
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    model, label_map = load_checkpoint(args.checkpoint, device)
    manifest = load_manifest(args.manifest)
    dataloader = create_dataloader(
        manifest,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        label_to_index=label_map,
        require_labels=False,
    )
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = evaluate_epoch(model, dataloader, criterion, device)
    if acc > 0:
        print(f"Evaluation results — loss: {loss:.4f}, accuracy: {acc:.4f}")
    else:
        print(f"Evaluation loss (no labels to score against): {loss:.4f}")


@torch.no_grad()
def run_predict(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for predict mode.")
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    model, label_map = load_checkpoint(args.checkpoint, device)
    manifest = load_manifest(args.manifest)
    dataloader = create_dataloader(
        manifest,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        label_to_index=None,
        require_labels=False,
    )
    idx_to_label = {idx: label for label, idx in label_map.items()}
    top_k = args.top_k
    outputs = []

    for batch in dataloader:
        logits, _ = _forward_batch(model, batch, device)
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=top_k, dim=-1)
        for sample_id, sample_values, sample_indices in zip(
            batch["ids"], values.cpu(), indices.cpu()
        ):
            predictions = [
                {
                    "label": idx_to_label[idx.item()],
                    "prob": float(val.item()),
                }
                for val, idx in zip(sample_values, sample_indices)
            ]
            outputs.append({"id": sample_id, "predictions": predictions})

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved predictions to {out_path}")
    else:
        for item in outputs:
            preds = ", ".join(
                f"{pred['label']} ({pred['prob']:.3f})" for pred in item["predictions"]
            )
            print(f"{item['id']}: {preds}")


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or run the multi-stream sign classifier."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "predict"],
        required=True,
        help="Operation mode.",
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    parser.add_argument(
        "--val_manifest",
        help="Optional validation manifest (train mode only).",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path for eval/predict or to resume training.",
    )
    parser.add_argument(
        "--save_dir",
        default="checkpoints",
        help="Directory to store checkpoints (train mode).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference/training device; falls back to CPU if unavailable.",
    )

    # Data parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")

    # Model parameters
    parser.add_argument("--num_body_joints", type=int, default=17)
    parser.add_argument("--num_hand_joints", type=int, default=42)
    parser.add_argument("--num_face_joints", type=int, default=478)
    parser.add_argument("--coordinate_dims", type=int, default=3)

    parser.add_argument("--body_hidden", type=int, default=256)
    parser.add_argument("--hand_hidden", type=int, default=192)
    parser.add_argument("--face_hidden", type=int, default=256)
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--final_dropout", type=float, default=0.2)
    parser.add_argument("--stream_dropout", type=float, default=0.1)

    parser.add_argument(
        "--xlstm_hidden",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden sizes for each xLSTM layer.",
    )
    parser.add_argument(
        "--xlstm_variant",
        choices=["mlstm", "slstm"],
        default="mlstm",
        help="Use multiplicative LSTM or standard stacked LSTM.",
    )
    parser.add_argument(
        "--xlstm_bidirectional",
        action="store_true",
        help="Enable bidirectional LSTM (only relevant for sLSTM).",
    )
    parser.add_argument("--xlstm_dropout", type=float, default=0.1)

    # Optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)

    # Prediction
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_path", help="Optional JSON output path for predictions.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
