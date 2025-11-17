"""
Sign Language Recognition Model
Complete model architecture combining TCN, xLSTM, and attention mechanisms
Based on the multi-stream architecture flowchart
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from tcn_module import MultiStreamTCN
from xlstm_module import sLSTM, mLSTM
from attention import TemporalAttentionAggregator, PositionalEncoding
from normalization import MultiStreamNormalizer


class SignLanguageRecognitionModel(nn.Module):
    """
    Complete Sign Language Recognition Model
    
    Architecture:
    1. Input: RGB Frames -> Pose/Hand/Face Keypoints
    2. Multi-stream Normalization
    3. Multi-stream TCN Encoders (with dilations 1, 2, 4)
    4. Feature Fusion
    5. xLSTM Layers (mLSTM or sLSTM)
    6. Temporal Attention Aggregation
    7. Classification Head
    """
    def __init__(self,
                 num_classes,
                 # Keypoint dimensions
                 body_dim=17*2,      # 17 keypoints * (x, y)
                 hand_dim=42*2,      # 21*2 hands * (x, y)
                 face_dim=478*2,     # 478 keypoints * (x, y)
                 # TCN parameters
                 tcn_hidden_dim=256,
                 tcn_num_layers=3,
                 tcn_kernel_size=3,
                 tcn_dropout=0.2,
                 # xLSTM parameters
                 xlstm_type='mlstm',  # 'mlstm' or 'slstm'
                 xlstm_hidden_dim=256,
                 xlstm_num_layers=2,
                 xlstm_head_dim=32,
                 xlstm_dropout=0.2,
                 # Attention parameters
                 attn_hidden_dim=128,
                 attn_num_heads=8,
                 use_multi_pool=True,
                 # Classification head
                 classifier_hidden_dim=256,
                 classifier_dropout=0.3,
                 # Other
                 use_positional_encoding=True):
        super(SignLanguageRecognitionModel, self).__init__()
        
        self.num_classes = num_classes
        self.xlstm_type = xlstm_type
        self.use_positional_encoding = use_positional_encoding
        
        # 1. Keypoint Normalization
        self.normalizer = MultiStreamNormalizer()
        
        # 2. Multi-stream TCN Encoders
        self.tcn_encoder = MultiStreamTCN(
            body_dim=body_dim,
            hand_dim=hand_dim,
            face_dim=face_dim,
            hidden_dim=tcn_hidden_dim,
            num_layers=tcn_num_layers,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout
        )
        
        # 3. Positional Encoding (optional)
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=tcn_hidden_dim,
                dropout=0.1
            )
        
        # 4. xLSTM Layers
        if xlstm_type == 'mlstm':
            self.xlstm = mLSTM(
                input_size=tcn_hidden_dim,
                hidden_size=xlstm_hidden_dim,
                num_layers=xlstm_num_layers,
                head_dim=xlstm_head_dim,
                dropout=xlstm_dropout
            )
        elif xlstm_type == 'slstm':
            self.xlstm = sLSTM(
                input_size=tcn_hidden_dim,
                hidden_size=xlstm_hidden_dim,
                num_layers=xlstm_num_layers,
                dropout=xlstm_dropout
            )
        else:
            raise ValueError(f"Unknown xLSTM type: {xlstm_type}")
        
        # 5. Temporal Attention Aggregation
        self.temporal_aggregator = TemporalAttentionAggregator(
            input_dim=xlstm_hidden_dim,
            hidden_dim=attn_hidden_dim,
            num_heads=attn_num_heads,
            use_multi_pool=use_multi_pool
        )
        
        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(xlstm_hidden_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                body_keypoints, 
                hand_keypoints, 
                face_keypoints,
                body_confidence=None,
                hand_confidence=None,
                face_confidence=None,
                mask=None):
        """
        Forward pass
        
        Args:
            body_keypoints: (batch, time, 17, 2)
            hand_keypoints: (batch, time, 42, 2)
            face_keypoints: (batch, time, 478, 2)
            *_confidence: Optional confidence scores
            mask: Optional temporal mask (batch, time)
        
        Returns:
            logits: Class logits (batch, num_classes)
            attention_weights: Attention weights from aggregator
        """
        # 1. Normalize keypoints
        body_norm, hand_norm, face_norm = self.normalizer(
            body_keypoints, hand_keypoints, face_keypoints,
            body_confidence, hand_confidence, face_confidence
        )
        
        # 2. Flatten keypoints for TCN input
        body_flat, hand_flat, face_flat = self.normalizer.flatten_keypoints(
            body_norm, hand_norm, face_norm
        )
        
        # 3. Multi-stream TCN encoding
        tcn_features = self.tcn_encoder(body_flat, hand_flat, face_flat)
        # tcn_features: (batch, time, tcn_hidden_dim)
        
        # 4. Add positional encoding
        if self.use_positional_encoding:
            tcn_features = self.pos_encoder(tcn_features)
        
        # 5. xLSTM processing
        xlstm_output, _ = self.xlstm(tcn_features)
        # xlstm_output: (batch, time, xlstm_hidden_dim)
        
        # 6. Temporal attention aggregation
        aggregated_features, attention_weights = self.temporal_aggregator(
            xlstm_output, mask
        )
        # aggregated_features: (batch, xlstm_hidden_dim)
        
        # 7. Classification
        logits = self.classifier(aggregated_features)
        # logits: (batch, num_classes)
        
        return logits, attention_weights
    
    def predict(self, 
                body_keypoints, 
                hand_keypoints, 
                face_keypoints,
                body_confidence=None,
                hand_confidence=None,
                face_confidence=None):
        """
        Make predictions (inference mode)
        
        Returns:
            predictions: Predicted class indices (batch,)
            probabilities: Class probabilities (batch, num_classes)
            attention_weights: Attention weights
        """
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(
                body_keypoints, hand_keypoints, face_keypoints,
                body_confidence, hand_confidence, face_confidence
            )
            
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities, attention_weights


class SignLanguageSequenceModel(nn.Module):
    """
    Sign Language Sequence-to-Sequence Model
    For continuous sign language recognition (outputs sequence of glosses)
    """
    def __init__(self,
                 num_classes,
                 max_seq_len=50,
                 **kwargs):
        super(SignLanguageSequenceModel, self).__init__()
        
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        # Use the base model as encoder
        self.encoder = SignLanguageRecognitionModel(
            num_classes=num_classes,
            **kwargs
        )
        
        # Remove the classification head from encoder
        self.encoder.classifier = nn.Identity()
        
        # Decoder for sequence generation
        self.decoder = nn.LSTM(
            input_size=kwargs.get('xlstm_hidden_dim', 256),
            hidden_size=kwargs.get('xlstm_hidden_dim', 256),
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output projection
        self.output_proj = nn.Linear(kwargs.get('xlstm_hidden_dim', 256), num_classes)
        
        # Start/end tokens
        self.start_token = nn.Parameter(torch.randn(1, 1, kwargs.get('xlstm_hidden_dim', 256)))
        self.end_token_id = num_classes - 1  # Assuming last class is END token
        
    def forward(self, 
                body_keypoints, 
                hand_keypoints, 
                face_keypoints,
                target_seq=None,
                body_confidence=None,
                hand_confidence=None,
                face_confidence=None):
        """
        Forward pass for sequence model
        
        Args:
            *_keypoints: Keypoint sequences
            target_seq: Target sequence for training (batch, target_len)
            *_confidence: Optional confidence scores
        
        Returns:
            output_seq: Output logits (batch, seq_len, num_classes)
        """
        batch_size = body_keypoints.size(0)
        
        # Encode input
        encoded, _ = self.encoder(
            body_keypoints, hand_keypoints, face_keypoints,
            body_confidence, hand_confidence, face_confidence
        )
        # encoded: (batch, xlstm_hidden_dim)
        
        # Prepare decoder input
        if target_seq is not None:
            # Training mode: teacher forcing
            seq_len = target_seq.size(1)
            decoder_input = torch.cat([
                self.start_token.expand(batch_size, 1, -1),
                encoded.unsqueeze(1).expand(-1, seq_len - 1, -1)
            ], dim=1)
        else:
            # Inference mode: autoregressive
            seq_len = self.max_seq_len
            decoder_input = self.start_token.expand(batch_size, 1, -1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        
        # Project to vocabulary
        output_seq = self.output_proj(decoder_output)
        
        return output_seq


def create_sign_language_model(model_type='classifier', **kwargs):
    """
    Factory function to create sign language recognition models
    
    Args:
        model_type: 'classifier' or 'sequence'
        **kwargs: Model parameters
    
    Returns:
        model: Sign language recognition model
    """
    if model_type == 'classifier':
        return SignLanguageRecognitionModel(**kwargs)
    elif model_type == 'sequence':
        return SignLanguageSequenceModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    print("Testing SignLanguageRecognitionModel...")
    
    batch_size = 4
    seq_len = 32
    num_classes = 100
    
    # Create dummy data
    body_kpts = torch.randn(batch_size, seq_len, 17, 2)
    hand_kpts = torch.randn(batch_size, seq_len, 42, 2)
    face_kpts = torch.randn(batch_size, seq_len, 478, 2)
    
    # Create model
    model = SignLanguageRecognitionModel(
        num_classes=num_classes,
        xlstm_type='mlstm',
        tcn_hidden_dim=256,
        xlstm_hidden_dim=256
    )
    
    # Forward pass
    logits, attn_weights = model(body_kpts, hand_kpts, face_kpts)
    
    print(f"Input shapes:")
    print(f"  Body: {body_kpts.shape}")
    print(f"  Hand: {hand_kpts.shape}")
    print(f"  Face: {face_kpts.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: (batch={batch_size}, num_classes={num_classes})")
    
    # Test predictions
    predictions, probabilities, _ = model.predict(body_kpts, hand_kpts, face_kpts)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
