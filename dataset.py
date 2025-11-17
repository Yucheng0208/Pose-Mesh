"""
Sign Language Dataset Loader
Loads JSON keypoint sequences from the Multi-Pose system
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os


class SignLanguageDataset(Dataset):
    """
    Dataset for sign language recognition from JSON keypoint files
    
    Expected directory structure:
    data_root/
        class_0/
            video_0/
                000000000001.json
                000000000002.json
                ...
            video_1/
                ...
        class_1/
            ...
    """
    def __init__(self,
                 data_root: str,
                 sequence_length: int = 32,
                 stride: int = 1,
                 mode: str = 'train',
                 augment: bool = False):
        """
        Args:
            data_root: Root directory containing class folders
            sequence_length: Number of frames per sequence
            stride: Stride for sliding window sampling
            mode: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.mode = mode
        self.augment = augment
        
        # Find all classes and videos
        self.classes = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build dataset index
        self.samples = self._build_dataset_index()
        
        print(f"Loaded {len(self.samples)} sequences from {len(self.classes)} classes")
        
    def _build_dataset_index(self) -> List[Dict]:
        """
        Build index of all valid sequences
        """
        samples = []
        
        for class_name in self.classes:
            class_dir = self.data_root / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Find all video directories in this class
            video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
            
            for video_dir in video_dirs:
                # Find all JSON files in video directory
                json_files = sorted(video_dir.glob("*.json"))
                
                if len(json_files) < self.sequence_length:
                    continue  # Skip videos that are too short
                
                # Create sequences with sliding window
                for start_idx in range(0, len(json_files) - self.sequence_length + 1, self.stride):
                    end_idx = start_idx + self.sequence_length
                    sequence_files = json_files[start_idx:end_idx]
                    
                    samples.append({
                        'class_name': class_name,
                        'class_idx': class_idx,
                        'video_dir': video_dir,
                        'sequence_files': sequence_files,
                        'start_frame': start_idx,
                        'end_frame': end_idx
                    })
        
        return samples
    
    def _load_keypoints_from_json(self, json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load keypoints from a single JSON file
        
        Returns:
            body_kpts: (17, 2) array
            hand_kpts: (42, 2) array - 21 left + 21 right
            face_kpts: (478, 2) array
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Initialize arrays with zeros
        body_kpts = np.zeros((17, 2), dtype=np.float32)
        hand_kpts = np.zeros((42, 2), dtype=np.float32)
        face_kpts = np.zeros((478, 2), dtype=np.float32)
        
        # Check if there are any persons detected
        if data['num_persons'] == 0 or len(data['persons']) == 0:
            return body_kpts, hand_kpts, face_kpts
        
        # Get first person's keypoints
        person = data['persons'][0]
        keypoints = person['keypoints']
        
        # Load body keypoints
        if 'pose' in keypoints and keypoints['pose']:
            for kp in keypoints['pose']:
                idx = kp['id']
                if idx < 17:
                    body_kpts[idx] = [kp['x'], kp['y']]
        
        # Load hand keypoints
        if 'left_hand' in keypoints and keypoints['left_hand']:
            for kp in keypoints['left_hand']:
                idx = kp['id']
                if idx < 21:
                    hand_kpts[idx] = [kp['x'], kp['y']]
        
        if 'right_hand' in keypoints and keypoints['right_hand']:
            for kp in keypoints['right_hand']:
                idx = kp['id']
                if idx < 21:
                    hand_kpts[21 + idx] = [kp['x'], kp['y']]
        
        # Load face keypoints
        if 'face' in keypoints and keypoints['face']:
            for kp in keypoints['face']:
                idx = kp['id']
                if idx < 478:
                    face_kpts[idx] = [kp['x'], kp['y']]
        
        return body_kpts, hand_kpts, face_kpts
    
    def _augment_keypoints(self, body, hand, face):
        """
        Apply data augmentation to keypoints
        """
        if not self.augment:
            return body, hand, face
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            body[:, :, 0] = -body[:, :, 0]
            hand[:, :, 0] = -hand[:, :, 0]
            face[:, :, 0] = -face[:, :, 0]
            
            # Swap left and right hands
            left_hand = hand[:, :21, :].copy()
            right_hand = hand[:, 21:, :].copy()
            hand[:, :21, :] = right_hand
            hand[:, 21:, :] = left_hand
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        body = body * scale
        hand = hand * scale
        face = face * scale
        
        # Random temporal shift (slight time warping)
        if np.random.random() > 0.7:
            shift = np.random.randint(-2, 3)
            if shift > 0:
                body = np.concatenate([body[shift:], body[-shift:]], axis=0)
                hand = np.concatenate([hand[shift:], hand[-shift:]], axis=0)
                face = np.concatenate([face[shift:], face[-shift:]], axis=0)
            elif shift < 0:
                body = np.concatenate([body[:shift], body[:-shift]], axis=0)
                hand = np.concatenate([hand[:shift], hand[:-shift]], axis=0)
                face = np.concatenate([face[:shift], face[:-shift]], axis=0)
        
        return body, hand, face
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single sequence sample
        
        Returns:
            body_seq: (sequence_length, 17, 2)
            hand_seq: (sequence_length, 42, 2)
            face_seq: (sequence_length, 478, 2)
            label: Class index
        """
        sample = self.samples[idx]
        
        # Load all frames in sequence
        body_seq = []
        hand_seq = []
        face_seq = []
        
        for json_file in sample['sequence_files']:
            body_kpts, hand_kpts, face_kpts = self._load_keypoints_from_json(json_file)
            body_seq.append(body_kpts)
            hand_seq.append(hand_kpts)
            face_seq.append(face_kpts)
        
        # Convert to numpy arrays
        body_seq = np.array(body_seq, dtype=np.float32)  # (seq_len, 17, 2)
        hand_seq = np.array(hand_seq, dtype=np.float32)  # (seq_len, 42, 2)
        face_seq = np.array(face_seq, dtype=np.float32)  # (seq_len, 478, 2)
        
        # Apply augmentation
        body_seq, hand_seq, face_seq = self._augment_keypoints(body_seq, hand_seq, face_seq)
        
        # Convert to tensors
        body_seq = torch.from_numpy(body_seq)
        hand_seq = torch.from_numpy(hand_seq)
        face_seq = torch.from_numpy(face_seq)
        
        label = sample['class_idx']
        
        return body_seq, hand_seq, face_seq, label


class VideoKeypointSequence:
    """
    Helper class to load a complete video sequence from JSON files
    """
    def __init__(self, json_dir: str):
        """
        Args:
            json_dir: Directory containing JSON keypoint files
        """
        self.json_dir = Path(json_dir)
        self.json_files = sorted(self.json_dir.glob("*.json"))
        
    def load_sequence(self, max_frames: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load complete video sequence
        
        Args:
            max_frames: Maximum number of frames to load (None = all)
        
        Returns:
            body_seq: (num_frames, 17, 2)
            hand_seq: (num_frames, 42, 2)
            face_seq: (num_frames, 478, 2)
        """
        if max_frames:
            json_files = self.json_files[:max_frames]
        else:
            json_files = self.json_files
        
        body_seq = []
        hand_seq = []
        face_seq = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Initialize arrays
            body_kpts = np.zeros((17, 2), dtype=np.float32)
            hand_kpts = np.zeros((42, 2), dtype=np.float32)
            face_kpts = np.zeros((478, 2), dtype=np.float32)
            
            # Load keypoints if person detected
            if data['num_persons'] > 0 and len(data['persons']) > 0:
                person = data['persons'][0]
                keypoints = person['keypoints']
                
                # Body
                if 'pose' in keypoints:
                    for kp in keypoints['pose']:
                        idx = kp['id']
                        if idx < 17:
                            body_kpts[idx] = [kp['x'], kp['y']]
                
                # Hands
                if 'left_hand' in keypoints:
                    for kp in keypoints['left_hand']:
                        idx = kp['id']
                        if idx < 21:
                            hand_kpts[idx] = [kp['x'], kp['y']]
                
                if 'right_hand' in keypoints:
                    for kp in keypoints['right_hand']:
                        idx = kp['id']
                        if idx < 21:
                            hand_kpts[21 + idx] = [kp['x'], kp['y']]
                
                # Face
                if 'face' in keypoints:
                    for kp in keypoints['face']:
                        idx = kp['id']
                        if idx < 478:
                            face_kpts[idx] = [kp['x'], kp['y']]
            
            body_seq.append(body_kpts)
            hand_seq.append(hand_kpts)
            face_seq.append(face_kpts)
        
        # Convert to tensors
        body_seq = torch.from_numpy(np.array(body_seq))
        hand_seq = torch.from_numpy(np.array(hand_seq))
        face_seq = torch.from_numpy(np.array(face_seq))
        
        return body_seq, hand_seq, face_seq


def create_dataloaders(data_root: str,
                       batch_size: int = 16,
                       sequence_length: int = 32,
                       num_workers: int = 4,
                       train_split: float = 0.8,
                       val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root: Root directory of dataset
        batch_size: Batch size
        sequence_length: Sequence length
        num_workers: Number of data loading workers
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = SignLanguageDataset(
        data_root=data_root,
        sequence_length=sequence_length,
        stride=sequence_length // 2,  # 50% overlap
        mode='train',
        augment=True
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    # This is an example - adjust path to your actual data
    data_root = "data/sign_language"
    
    if Path(data_root).exists():
        dataset = SignLanguageDataset(
            data_root=data_root,
            sequence_length=32,
            stride=16,
            augment=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Classes: {dataset.classes}")
        
        # Load a sample
        if len(dataset) > 0:
            body, hand, face, label = dataset[0]
            print(f"\nSample shapes:")
            print(f"  Body: {body.shape}")
            print(f"  Hand: {hand.shape}")
            print(f"  Face: {face.shape}")
            print(f"  Label: {label}")
    else:
        print(f"Data directory not found: {data_root}")
        print("Please organize your data in the following structure:")
        print("data_root/")
        print("  class_0/")
        print("    video_0/")
        print("      000000000001.json")
        print("      ...")
