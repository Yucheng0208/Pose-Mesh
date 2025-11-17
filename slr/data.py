from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _extract_landmarks(
    keypoints: List[Dict],
    expected: int,
) -> np.ndarray:
    """Convert a list of dicts to a dense (N, 3) array."""
    arr = np.zeros((expected, 3), dtype=np.float32)
    if not keypoints:
        return arr

    for kp in keypoints:
        idx = kp.get("id")
        if idx is None or idx < 0 or idx >= expected:
            continue
        arr[idx, 0] = kp.get("x", 0.0)
        arr[idx, 1] = kp.get("y", 0.0)
        arr[idx, 2] = kp.get("confidence", 1.0)
    return arr


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_sequence(
    frame_paths: List[Path],
    pose_points: int,
    hand_points: int,
    face_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose_frames: List[np.ndarray] = []
    hand_frames: List[np.ndarray] = []
    face_frames: List[np.ndarray] = []
    valid_mask: List[float] = []

    half_hand = hand_points // 2

    for frame_path in frame_paths:
        frame = _load_json(frame_path)
        persons = frame.get("persons", [])
        if not persons:
            pose_frames.append(np.zeros((pose_points, 3), dtype=np.float32))
            hand_frames.append(np.zeros((hand_points, 3), dtype=np.float32))
            face_frames.append(np.zeros((face_points, 3), dtype=np.float32))
            valid_mask.append(0.0)
            continue

        person = persons[0]
        keypoints: Dict[str, List[Dict]] = person.get("keypoints", {})
        pose_frames.append(_extract_landmarks(keypoints.get("pose", []), pose_points))

        left = _extract_landmarks(keypoints.get("left_hand", []), half_hand)
        right = _extract_landmarks(keypoints.get("right_hand", []), half_hand)
        hands = np.concatenate([left, right], axis=0)
        hand_frames.append(hands)

        face_frames.append(_extract_landmarks(keypoints.get("face", []), face_points))
        valid_mask.append(1.0)

    pose_seq = np.stack(pose_frames, axis=0)
    hand_seq = np.stack(hand_frames, axis=0)
    face_seq = np.stack(face_frames, axis=0)
    mask = np.array(valid_mask, dtype=np.float32)
    return pose_seq, hand_seq, face_seq, mask


def _clip_sequences(
    pose_seq: np.ndarray,
    hand_seq: np.ndarray,
    face_seq: np.ndarray,
    mask: np.ndarray,
    max_seq_len: Optional[int],
    random_clip: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_seq_len is None or pose_seq.shape[0] <= max_seq_len:
        return pose_seq, hand_seq, face_seq, mask

    start = 0
    total = pose_seq.shape[0]
    if random_clip:
        start = np.random.randint(0, total - max_seq_len + 1)
    else:
        start = total - max_seq_len
    end = start + max_seq_len
    return (
        pose_seq[start:end],
        hand_seq[start:end],
        face_seq[start:end],
        mask[start:end],
    )


def _tensorize_sequences(
    pose_seq: np.ndarray,
    hand_seq: np.ndarray,
    face_seq: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, torch.Tensor]:
    pose_tensor = torch.from_numpy(pose_seq.reshape(pose_seq.shape[0], -1))
    hand_tensor = torch.from_numpy(hand_seq.reshape(hand_seq.shape[0], -1))
    face_tensor = torch.from_numpy(face_seq.reshape(face_seq.shape[0], -1))
    mask_tensor = torch.from_numpy(mask)
    return {
        "pose": pose_tensor,
        "hand": hand_tensor,
        "face": face_tensor,
        "mask": mask_tensor,
    }


def load_sequence_from_dir(
    json_dir: str | Path,
    pose_points: int = 17,
    hand_points: int = 42,
    face_points: int = 478,
    max_seq_len: Optional[int] = None,
    min_seq_len: int = 1,
    random_clip: bool = False,
) -> Dict[str, torch.Tensor]:
    dir_path = Path(json_dir)
    frame_paths = sorted(dir_path.glob("*.json"))
    if len(frame_paths) < min_seq_len:
        raise ValueError(f"Sequence at {dir_path} has {len(frame_paths)} frames < min_seq_len={min_seq_len}")

    pose_seq, hand_seq, face_seq, mask = _decode_sequence(frame_paths, pose_points, hand_points, face_points)
    pose_seq, hand_seq, face_seq, mask = _clip_sequences(
        pose_seq, hand_seq, face_seq, mask, max_seq_len=max_seq_len, random_clip=random_clip
    )
    return _tensorize_sequences(pose_seq, hand_seq, face_seq, mask)


@dataclass
class SequenceSample:
    sample_id: str
    label: str
    json_dir: Path


class SignSequenceDataset(Dataset):
    """
    Dataset that loads per-frame JSON keypoints exported by `sign_detector.py`.

    Expected directory layout:
        metadata.csv  # columns: sample_id,label[,split]
        json_root/
            sample_id_0001/
                000000000001.json
                ...
    """

    def __init__(
        self,
        metadata_path: str | Path,
        json_root: str | Path,
        split: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        min_seq_len: int = 8,
        random_clip: bool = True,
        cache_sequences: bool = False,
        pose_points: int = 17,
        hand_points: int = 42,
        face_points: int = 478,
        label_mapping: Optional[Dict[str, int]] = None,
        allowed_sample_ids: Optional[List[str]] = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.json_root = Path(json_root)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata file not found: {self.metadata_path}")
        if not self.json_root.exists():
            raise FileNotFoundError(f"JSON root not found: {self.json_root}")

        metadata = pd.read_csv(self.metadata_path)
        if "sample_id" not in metadata.columns or "label" not in metadata.columns:
            raise ValueError("metadata must include 'sample_id' and 'label' columns")
        if split is not None:
            if "split" not in metadata.columns:
                raise ValueError("metadata does not contain a 'split' column")
            metadata = metadata[metadata["split"] == split]
            if metadata.empty:
                raise ValueError(f"No samples found for split '{split}'")

        if allowed_sample_ids is not None:
            allowed = set(str(sid) for sid in allowed_sample_ids)
            metadata = metadata[metadata["sample_id"].astype(str).isin(allowed)]
            if metadata.empty:
                raise ValueError("No samples left after applying allowed_sample_ids filter")

        metadata = metadata.reset_index(drop=True)
        self.samples: List[SequenceSample] = []
        for row in metadata.itertuples(index=False):
            sample_id = getattr(row, "sample_id")
            label = getattr(row, "label")
            sample_dir = self.json_root / str(sample_id)
            if not sample_dir.exists():
                raise FileNotFoundError(f"Expected folder for sample '{sample_id}' not found: {sample_dir}")
            self.samples.append(SequenceSample(sample_id=sample_id, label=label, json_dir=sample_dir))

        self.pose_points = pose_points
        self.hand_points = hand_points
        self.face_points = face_points
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.random_clip = random_clip
        self.cache_sequences = cache_sequences

        labels = metadata["label"].astype(str).unique().tolist()
        if label_mapping is None:
            labels = sorted(labels)
            self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_index = label_mapping
            missing = sorted({lbl for lbl in metadata["label"].unique() if lbl not in self.label_to_index})
            if missing:
                raise ValueError(f"Provided label_mapping is missing labels: {missing}")
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        self.cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        if self.cache_sequences and sample.sample_id in self.cache:
            sequence = self.cache[sample.sample_id]
        else:
            sequence = self._load_sequence(sample)
            if self.cache_sequences:
                self.cache[sample.sample_id] = sequence

        label_idx = self.label_to_index[sample.label]
        sequence["label"] = torch.tensor(label_idx, dtype=torch.long)
        sequence["sample_id"] = sample.sample_id
        return sequence

    def _load_sequence(self, sample: SequenceSample) -> Dict[str, torch.Tensor]:
        frame_paths = sorted(sample.json_dir.glob("*.json"))
        if len(frame_paths) < self.min_seq_len:
            raise ValueError(
                f"Sample '{sample.sample_id}' has {len(frame_paths)} frames but min_seq_len={self.min_seq_len}"
            )

        pose_seq, hand_seq, face_seq, mask = _decode_sequence(
            frame_paths,
            pose_points=self.pose_points,
            hand_points=self.hand_points,
            face_points=self.face_points,
        )
        pose_seq, hand_seq, face_seq, mask = _clip_sequences(
            pose_seq,
            hand_seq,
            face_seq,
            mask,
            max_seq_len=self.max_seq_len,
            random_clip=self.random_clip,
        )
        return _tensorize_sequences(pose_seq, hand_seq, face_seq, mask)


def sign_sequence_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences within a batch."""
    max_len = max(item["pose"].shape[0] for item in batch)

    def _pad_sequence(seq: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        pad_len = max_len - seq.shape[0]
        if pad_len == 0:
            return seq
        pad = torch.full((pad_len, seq.shape[1]), pad_value, dtype=seq.dtype)
        return torch.cat([seq, pad], dim=0)

    def _pad_mask(mask: torch.Tensor) -> torch.Tensor:
        pad_len = max_len - mask.shape[0]
        if pad_len == 0:
            return mask
        pad = torch.zeros(pad_len, dtype=mask.dtype)
        return torch.cat([mask, pad], dim=0)

    pose_batch, hand_batch, face_batch, mask_batch, label_batch = [], [], [], [], []
    sample_ids: List[str] = []
    for item in batch:
        pose_batch.append(_pad_sequence(item["pose"]))
        hand_batch.append(_pad_sequence(item["hand"]))
        face_batch.append(_pad_sequence(item["face"]))
        mask_batch.append(_pad_mask(item["mask"]))
        label_batch.append(item["label"])
        sample_ids.append(item["sample_id"])

    pose = torch.stack(pose_batch, dim=0)  # (B, T, D)
    hand = torch.stack(hand_batch, dim=0)
    face = torch.stack(face_batch, dim=0)
    mask = torch.stack(mask_batch, dim=0)
    labels = torch.stack(label_batch, dim=0)

    return {
        "pose": pose.float(),
        "hand": hand.float(),
        "face": face.float(),
        "mask": mask.float(),
        "labels": labels,
        "sample_ids": sample_ids,
    }
