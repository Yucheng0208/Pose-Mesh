from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data import SignSequenceDataset, sign_sequence_collate
from .model import SignLanguageXLSTM, SignModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained xLSTM sign language model.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV.")
    parser.add_argument("--json-root", type=str, required=True, help="Root directory of JSON sequences.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt file from training.")
    parser.add_argument("--config", type=str, help="Path to config.json (defaults to checkpoint directory).")
    parser.add_argument("--labels", type=str, help="Path to labels.json (defaults to checkpoint directory).")
    parser.add_argument("--split", type=str, help="Name of split in metadata to evaluate.")
    parser.add_argument("--allowed-ids", type=str, nargs="+", help="Optional subset of sample_ids to evaluate.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Clip sequences to this many frames.")
    parser.add_argument("--min-seq-len", type=int, default=8)
    return parser.parse_args()


def load_config(config_path: Path) -> SignModelConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "tcn_dilations" in data:
        data["tcn_dilations"] = tuple(data["tcn_dilations"])
    return SignModelConfig(**data)


def load_labels(labels_path: Path) -> List[str]:
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "labels" in data:
        labels = data["labels"]
    elif isinstance(data, list):
        labels = data
    else:
        raise ValueError(f"Unrecognized label format at {labels_path}")
    if any(label is None for label in labels):
        raise ValueError("labels.json contains null entries; ensure training exported label order.")
    return [str(label) for label in labels]


def build_model(config: SignModelConfig, checkpoint_path: Path, device: str) -> SignLanguageXLSTM:
    model = SignLanguageXLSTM(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, label_names: List[str]) -> Dict[str, float]:
    total = 0
    correct = 0
    per_class_total = torch.zeros(len(label_names), dtype=torch.long)
    per_class_correct = torch.zeros(len(label_names), dtype=torch.long)

    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ["pose", "hand", "face"]}
        mask = batch["mask"].to(device)
        inputs["mask"] = mask
        labels = batch["labels"].to(device)

        outputs = model(inputs)["logits"]
        preds = outputs.argmax(dim=-1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        per_class_total += torch.bincount(labels, minlength=len(label_names))
        per_class_correct += torch.bincount(labels[preds == labels], minlength=len(label_names))

    overall_acc = correct / total if total > 0 else 0.0
    per_class_acc = {
        label_names[i]: (per_class_correct[i].item() / per_class_total[i].item()) if per_class_total[i] > 0 else 0.0
        for i in range(len(label_names))
    }
    return {"accuracy": overall_acc, "per_class_accuracy": per_class_acc}


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent
    config_path = Path(args.config) if args.config else base_dir / "config.json"
    labels_path = Path(args.labels) if args.labels else base_dir / "labels.json"

    if not config_path.exists() or not labels_path.exists():
        raise FileNotFoundError("config.json or labels.json not found. Provide --config/--labels explicitly.")

    config = load_config(config_path)
    label_names = load_labels(labels_path)
    label_mapping = {label: idx for idx, label in enumerate(label_names)}

    dataset = SignSequenceDataset(
        metadata_path=args.metadata,
        json_root=args.json_root,
        split=args.split,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        random_clip=False,
        cache_sequences=False,
        label_mapping=label_mapping,
        allowed_sample_ids=args.allowed_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=sign_sequence_collate,
    )

    model = build_model(config, checkpoint_path, args.device)
    metrics = evaluate(model, loader, args.device, label_names)

    print(f"Overall accuracy: {metrics['accuracy']*100:.2f}%")
    print("Per-class accuracy:")
    for label, acc in sorted(metrics["per_class_accuracy"].items()):
        print(f"  {label}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
