"""
Keypoint Normalization Module
Normalizes body, hand, and face keypoints for invariance to position, scale, and rotation
"""
import torch
import torch.nn as nn
import numpy as np


class KeypointNormalizer(nn.Module):
    """
    Normalizes keypoint sequences for translation, scale, and rotation invariance
    """
    def __init__(self, normalize_scale=True, normalize_rotation=False):
        super(KeypointNormalizer, self).__init__()
        self.normalize_scale = normalize_scale
        self.normalize_rotation = normalize_rotation
        
    def forward(self, keypoints, confidences=None):
        """
        Args:
            keypoints: Tensor of shape (batch, time, num_keypoints, 2) for (x, y)
            confidences: Optional confidence scores (batch, time, num_keypoints)
        Returns:
            Normalized keypoints of same shape
        """
        batch_size, time_steps, num_keypoints, coords = keypoints.shape
        assert coords == 2, "Expected (x, y) coordinates"
        
        normalized = keypoints.clone()
        
        # Handle confidence masking if provided
        if confidences is not None:
            mask = confidences > 0.3  # Threshold for valid keypoints
            mask = mask.unsqueeze(-1).expand_as(keypoints)
        else:
            mask = torch.ones_like(keypoints, dtype=torch.bool)
        
        # Normalize each frame independently
        for b in range(batch_size):
            for t in range(time_steps):
                frame_kpts = keypoints[b, t]  # (num_keypoints, 2)
                frame_mask = mask[b, t] if confidences is not None else mask[b, t]
                
                # Get valid keypoints
                valid_kpts = frame_kpts[frame_mask[:, 0]]  # Filter by first dimension
                
                if len(valid_kpts) < 2:
                    # Not enough valid keypoints, skip normalization
                    continue
                
                # 1. Translation normalization (center at mean)
                mean = valid_kpts.mean(dim=0)
                normalized[b, t] = frame_kpts - mean
                
                # 2. Scale normalization (normalize by standard deviation)
                if self.normalize_scale:
                    std = valid_kpts.std(dim=0).mean() + 1e-6
                    normalized[b, t] = normalized[b, t] / std
                
                # 3. Rotation normalization (align to principal component)
                if self.normalize_rotation:
                    centered_valid = valid_kpts - mean
                    # Compute covariance matrix
                    cov = torch.mm(centered_valid.T, centered_valid) / len(centered_valid)
                    # Get principal component (largest eigenvector)
                    try:
                        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                        principal = eigenvectors[:, -1]
                        
                        # Compute rotation angle
                        angle = torch.atan2(principal[1], principal[0])
                        
                        # Create rotation matrix
                        cos_a = torch.cos(-angle)
                        sin_a = torch.sin(-angle)
                        rotation_matrix = torch.tensor([
                            [cos_a, -sin_a],
                            [sin_a, cos_a]
                        ], device=keypoints.device, dtype=keypoints.dtype)
                        
                        # Apply rotation
                        normalized[b, t] = torch.mm(normalized[b, t], rotation_matrix)
                    except:
                        # If eigendecomposition fails, skip rotation
                        pass
        
        return normalized


class BodyKeypointNormalizer(nn.Module):
    """
    Specialized normalizer for body keypoints (17 points from YOLO-Pose)
    Uses hip/shoulder center as reference point
    """
    def __init__(self):
        super(BodyKeypointNormalizer, self).__init__()
        # YOLO-Pose keypoint indices
        self.left_shoulder_idx = 5
        self.right_shoulder_idx = 6
        self.left_hip_idx = 11
        self.right_hip_idx = 12
        
    def forward(self, keypoints, confidences=None):
        """
        Args:
            keypoints: (batch, time, 17, 2)
            confidences: (batch, time, 17)
        Returns:
            Normalized keypoints
        """
        batch_size, time_steps, num_keypoints, _ = keypoints.shape
        normalized = keypoints.clone()
        
        for b in range(batch_size):
            for t in range(time_steps):
                frame_kpts = keypoints[b, t]
                
                # Compute torso center (average of shoulders and hips)
                shoulders = (frame_kpts[self.left_shoulder_idx] + frame_kpts[self.right_shoulder_idx]) / 2
                hips = (frame_kpts[self.left_hip_idx] + frame_kpts[self.right_hip_idx]) / 2
                torso_center = (shoulders + hips) / 2
                
                # Center at torso
                normalized[b, t] = frame_kpts - torso_center
                
                # Scale by torso height
                torso_height = torch.norm(shoulders - hips) + 1e-6
                normalized[b, t] = normalized[b, t] / torso_height
        
        return normalized


class HandKeypointNormalizer(nn.Module):
    """
    Specialized normalizer for hand keypoints (42 points: 21 left + 21 right)
    Normalizes each hand independently
    """
    def __init__(self):
        super(HandKeypointNormalizer, self).__init__()
        self.wrist_idx = 0  # Wrist is keypoint 0 in MediaPipe hand
        
    def forward(self, keypoints, confidences=None):
        """
        Args:
            keypoints: (batch, time, 42, 2) - 21 left + 21 right
            confidences: (batch, time, 42)
        Returns:
            Normalized keypoints
        """
        batch_size, time_steps, num_keypoints, _ = keypoints.shape
        assert num_keypoints == 42, "Expected 42 hand keypoints (21 per hand)"
        
        normalized = keypoints.clone()
        
        # Split into left and right hands
        left_hand = keypoints[:, :, :21, :]
        right_hand = keypoints[:, :, 21:, :]
        
        # Normalize left hand
        for b in range(batch_size):
            for t in range(time_steps):
                # Left hand
                wrist = left_hand[b, t, self.wrist_idx]
                centered = left_hand[b, t] - wrist
                palm_size = torch.norm(centered).mean() + 1e-6
                normalized[b, t, :21] = centered / palm_size
                
                # Right hand
                wrist = right_hand[b, t, self.wrist_idx]
                centered = right_hand[b, t] - wrist
                palm_size = torch.norm(centered).mean() + 1e-6
                normalized[b, t, 21:] = centered / palm_size
        
        return normalized


class FaceKeypointNormalizer(nn.Module):
    """
    Specialized normalizer for face keypoints (478 points from MediaPipe Face Mesh)
    Uses face bounding box for normalization
    """
    def __init__(self):
        super(FaceKeypointNormalizer, self).__init__()
        
    def forward(self, keypoints, confidences=None):
        """
        Args:
            keypoints: (batch, time, 478, 2)
            confidences: (batch, time, 478)
        Returns:
            Normalized keypoints
        """
        batch_size, time_steps, num_keypoints, _ = keypoints.shape
        normalized = keypoints.clone()
        
        for b in range(batch_size):
            for t in range(time_steps):
                frame_kpts = keypoints[b, t]
                
                # Compute face center and scale using bounding box
                min_coords = frame_kpts.min(dim=0)[0]
                max_coords = frame_kpts.max(dim=0)[0]
                center = (min_coords + max_coords) / 2
                scale = (max_coords - min_coords).max() + 1e-6
                
                # Normalize
                normalized[b, t] = (frame_kpts - center) / scale
        
        return normalized


class MultiStreamNormalizer(nn.Module):
    """
    Combined normalizer for all three streams
    """
    def __init__(self):
        super(MultiStreamNormalizer, self).__init__()
        self.body_normalizer = BodyKeypointNormalizer()
        self.hand_normalizer = HandKeypointNormalizer()
        self.face_normalizer = FaceKeypointNormalizer()
        
    def forward(self, body_kpts, hand_kpts, face_kpts, 
                body_conf=None, hand_conf=None, face_conf=None):
        """
        Normalize all three keypoint streams
        
        Args:
            body_kpts: (batch, time, 17, 2)
            hand_kpts: (batch, time, 42, 2)
            face_kpts: (batch, time, 478, 2)
            *_conf: Optional confidence scores
        Returns:
            Normalized body, hand, face keypoints
        """
        body_norm = self.body_normalizer(body_kpts, body_conf)
        hand_norm = self.hand_normalizer(hand_kpts, hand_conf)
        face_norm = self.face_normalizer(face_kpts, face_conf)
        
        return body_norm, hand_norm, face_norm
    
    def flatten_keypoints(self, body_kpts, hand_kpts, face_kpts):
        """
        Flatten normalized keypoints for feeding into TCN
        
        Args:
            body_kpts: (batch, time, 17, 2)
            hand_kpts: (batch, time, 42, 2)
            face_kpts: (batch, time, 478, 2)
        Returns:
            Flattened tensors:
            body: (batch, time, 34)
            hand: (batch, time, 84)
            face: (batch, time, 956)
        """
        batch_size, time_steps, _, _ = body_kpts.shape
        
        body_flat = body_kpts.reshape(batch_size, time_steps, -1)  # 17*2 = 34
        hand_flat = hand_kpts.reshape(batch_size, time_steps, -1)  # 42*2 = 84
        face_flat = face_kpts.reshape(batch_size, time_steps, -1)  # 478*2 = 956
        
        return body_flat, hand_flat, face_flat


if __name__ == "__main__":
    # Test normalizers
    batch_size = 4
    time_steps = 32
    
    print("Testing MultiStreamNormalizer...")
    
    # Create dummy data
    body_kpts = torch.randn(batch_size, time_steps, 17, 2) * 100 + 200
    hand_kpts = torch.randn(batch_size, time_steps, 42, 2) * 50 + 150
    face_kpts = torch.randn(batch_size, time_steps, 478, 2) * 30 + 180
    
    normalizer = MultiStreamNormalizer()
    
    # Normalize
    body_norm, hand_norm, face_norm = normalizer(body_kpts, hand_kpts, face_kpts)
    
    print(f"Body keypoints:")
    print(f"  Input range: [{body_kpts.min():.2f}, {body_kpts.max():.2f}]")
    print(f"  Normalized range: [{body_norm.min():.2f}, {body_norm.max():.2f}]")
    print(f"  Shape: {body_norm.shape}")
    
    print(f"\nHand keypoints:")
    print(f"  Input range: [{hand_kpts.min():.2f}, {hand_kpts.max():.2f}]")
    print(f"  Normalized range: [{hand_norm.min():.2f}, {hand_norm.max():.2f}]")
    print(f"  Shape: {hand_norm.shape}")
    
    print(f"\nFace keypoints:")
    print(f"  Input range: [{face_kpts.min():.2f}, {face_kpts.max():.2f}]")
    print(f"  Normalized range: [{face_norm.min():.2f}, {face_norm.max():.2f}]")
    print(f"  Shape: {face_norm.shape}")
    
    # Test flattening
    body_flat, hand_flat, face_flat = normalizer.flatten_keypoints(body_norm, hand_norm, face_norm)
    print(f"\nFlattened shapes:")
    print(f"  Body: {body_flat.shape} (expected: {batch_size}, {time_steps}, 34)")
    print(f"  Hand: {hand_flat.shape} (expected: {batch_size}, {time_steps}, 84)")
    print(f"  Face: {face_flat.shape} (expected: {batch_size}, {time_steps}, 956)")
