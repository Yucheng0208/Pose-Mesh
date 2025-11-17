"""
Sign Language Recognition Models Package
Multi-stream TCN-xLSTM architecture for sign language recognition
"""

from .sign_language_model import (
    SignLanguageRecognitionModel,
    SignLanguageSequenceModel,
    create_sign_language_model
)

from .tcn_module import (
    TemporalBlock,
    TCNEncoder,
    MultiStreamTCN
)

from .xlstm_module import (
    sLSTMCell,
    mLSTMCell,
    sLSTM,
    mLSTM
)

from .attention import (
    MultiHeadAttention,
    AttentionPooling,
    SelfAttentionPooling,
    TemporalAttentionAggregator,
    PositionalEncoding
)

from .normalization import (
    KeypointNormalizer,
    BodyKeypointNormalizer,
    HandKeypointNormalizer,
    FaceKeypointNormalizer,
    MultiStreamNormalizer
)

__all__ = [
    # Main models
    'SignLanguageRecognitionModel',
    'SignLanguageSequenceModel',
    'create_sign_language_model',
    
    # TCN components
    'TemporalBlock',
    'TCNEncoder',
    'MultiStreamTCN',
    
    # xLSTM components
    'sLSTMCell',
    'mLSTMCell',
    'sLSTM',
    'mLSTM',
    
    # Attention components
    'MultiHeadAttention',
    'AttentionPooling',
    'SelfAttentionPooling',
    'TemporalAttentionAggregator',
    'PositionalEncoding',
    
    # Normalization components
    'KeypointNormalizer',
    'BodyKeypointNormalizer',
    'HandKeypointNormalizer',
    'FaceKeypointNormalizer',
    'MultiStreamNormalizer',
]

__version__ = '1.0.0'
