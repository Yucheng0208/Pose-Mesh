"""
Temporal Convolutional Network (TCN) Module
Implements multi-dilation TCN encoder for temporal feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        # First conv layer
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
        Returns:
            Output tensor of shape (batch, channels, time)
        """
        # First conv block
        out = self.conv1(x)
        # Remove future information (causal)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Multi-layer TCN Encoder with increasing dilation
    kernel_size=3, dilations=[1, 2, 4] as specified in the flowchart
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3, kernel_size=3, dropout=0.2):
        super(TCNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # TCN layers with exponentially increasing dilation
        dilations = [2**i for i in range(num_layers)]  # [1, 2, 4]
        
        layers = []
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim
            dilation = dilations[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, 
                    dilation, dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, time, features)
        Returns:
            Output tensor of shape (batch, time, hidden_dim)
        """
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch, time, hidden_dim)
        
        # Transpose for conv1d: (batch, hidden_dim, time)
        x = x.transpose(1, 2)
        
        # Pass through TCN layers
        x = self.network(x)
        
        # Transpose back: (batch, time, hidden_dim)
        x = x.transpose(1, 2)
        
        return x


class MultiStreamTCN(nn.Module):
    """
    Multi-stream TCN encoder for body, hand, and face keypoints
    """
    def __init__(self, 
                 body_dim=17*2,      # 17 keypoints * (x, y)
                 hand_dim=42*2,      # 21*2 hands * (x, y)
                 face_dim=478*2,     # 478 keypoints * (x, y)
                 hidden_dim=256,
                 num_layers=3,
                 kernel_size=3,
                 dropout=0.2):
        super(MultiStreamTCN, self).__init__()
        
        # Individual TCN encoders for each stream
        self.body_tcn = TCNEncoder(body_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.hand_tcn = TCNEncoder(hand_dim, hidden_dim, num_layers, kernel_size, dropout)
        self.face_tcn = TCNEncoder(face_dim, hidden_dim, num_layers, kernel_size, dropout)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, body_seq, hand_seq, face_seq):
        """
        Args:
            body_seq: Body keypoint sequence (batch, time, body_dim)
            hand_seq: Hand keypoint sequence (batch, time, hand_dim)
            face_seq: Face keypoint sequence (batch, time, face_dim)
        Returns:
            Fused feature tensor (batch, time, hidden_dim)
        """
        # Encode each stream
        body_feat = self.body_tcn(body_seq)    # (batch, time, hidden_dim)
        hand_feat = self.hand_tcn(hand_seq)    # (batch, time, hidden_dim)
        face_feat = self.face_tcn(face_seq)    # (batch, time, hidden_dim)
        
        # Concatenate features
        concat_feat = torch.cat([body_feat, hand_feat, face_feat], dim=-1)  # (batch, time, hidden_dim*3)
        
        # Fuse features
        fused_feat = self.fusion(concat_feat)  # (batch, time, hidden_dim)
        fused_feat = self.fusion_norm(fused_feat)
        
        return fused_feat


if __name__ == "__main__":
    # Test the TCN modules
    batch_size = 4
    seq_len = 32
    
    # Create dummy data
    body_seq = torch.randn(batch_size, seq_len, 34)   # 17 keypoints * 2
    hand_seq = torch.randn(batch_size, seq_len, 84)   # 42 keypoints * 2
    face_seq = torch.randn(batch_size, seq_len, 956)  # 478 keypoints * 2
    
    # Test MultiStreamTCN
    model = MultiStreamTCN(
        body_dim=34,
        hand_dim=84,
        face_dim=956,
        hidden_dim=256,
        num_layers=3
    )
    
    output = model(body_seq, hand_seq, face_seq)
    print(f"Input shapes:")
    print(f"  Body: {body_seq.shape}")
    print(f"  Hand: {hand_seq.shape}")
    print(f"  Face: {face_seq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, time={seq_len}, hidden_dim=256)")
