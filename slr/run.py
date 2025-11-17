from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from .data import load_sequence_from_dir
from .model import SignLanguageXLSTM, SignModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on JSON keypoint folders using the trained xLSTM model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="One or more JSON directories to classify.")
    parser.add_argument("--config", type=str, help="Path to config.json (defaults to checkpoint directory).")
    parser.add_argument("--labels", type=str, help="Path to labels.json (defaults to checkpoint directory).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len", type=int, default=96, help="Clip sequences to this many frames before inference.")
    parser.add_argument("--min-seq-len", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3, help="Show top-K predictions.")
    return parser.parse_args()


def load_config_and_labels(checkpoint_path: Path, config_path: Path | None, labels_path: Path | None):
    base_dir = checkpoint_path.parent
    config_file = config_path or (base_dir / "config.json")
    labels_file = labels_path or (base_dir / "labels.json")

    if not config_file.exists() or not labels_file.exists():
        raise FileNotFoundError("config.json or labels.json not found; pass --config/--labels explicitly.")

    with config_file.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    if "tcn_dilations" in config_dict:
        config_dict["tcn_dilations"] = tuple(config_dict["tcn_dilations"])
    config = SignModelConfig(**config_dict)

    with labels_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "labels" in data:
        labels = data["labels"]
    elif isinstance(data, list):
        labels = data
    else:
        raise ValueError("labels.json must contain either a list or {\"labels\": [...]}")
    labels = [str(label) for label in labels]
    return config, labels


def build_model(config: SignModelConfig, checkpoint_path: Path, device: str) -> SignLanguageXLSTM:
    model = SignLanguageXLSTM(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def classify_directory(
    model: SignLanguageXLSTM,
    json_dir: Path,
    config: SignModelConfig,
    device: str,
    max_seq_len: int,
    min_seq_len: int,
) -> torch.Tensor:
    pose_points = config.pose_dim // 3
    hand_points = config.hand_dim // 3
    face_points = config.face_dim // 3

    sequence = load_sequence_from_dir(
        json_dir,
        pose_points=pose_points,
        hand_points=hand_points,
        face_points=face_points,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        random_clip=False,
    )

    inputs = {k: v.unsqueeze(0).to(device) for k, v in sequence.items()}
    outputs = model(inputs)["logits"]
    return outputs.squeeze(0)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    config, labels = load_config_and_labels(checkpoint_path, Path(args.config) if args.config else None, Path(args.labels) if args.labels else None)

    model = build_model(config, checkpoint_path, args.device)
    top_k = min(args.top_k, len(labels))

    for input_dir in args.inputs:
        dir_path = Path(input_dir)
        if not dir_path.exists():
            print(f"[WARN] {dir_path} does not exist, skipping.")
            continue

        logits = classify_directory(model, dir_path, config, args.device, args.max_seq_len, args.min_seq_len)
        probs = F.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=top_k)

        print(f"\nResults for {dir_path}:")
        for rank in range(top_k):
            label_idx = indices[rank].item()
            label = labels[label_idx] if label_idx < len(labels) else f"class_{label_idx}"
            score = values[rank].item() * 100
            print(f"  Top{rank+1}: {label} ({score:.2f}%)")


if __name__ == "__main__":
    main()
