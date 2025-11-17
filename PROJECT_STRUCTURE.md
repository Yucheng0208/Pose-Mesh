# Project Structure Overview

## 📁 File Organization

```
/workspace/
├── models/                           # Neural network modules
│   ├── __init__.py                  # Package initialization
│   ├── tcn_module.py                # Temporal Convolutional Networks
│   ├── xlstm_module.py              # Extended LSTM (mLSTM/sLSTM)
│   ├── attention.py                 # Attention mechanisms
│   ├── normalization.py             # Keypoint normalization
│   └── sign_language_model.py       # Main model architecture
│
├── dataset.py                        # Data loading and preprocessing
├── train.py                          # Training script
├── inference.py                      # Inference script
├── test_pipeline.py                  # System verification tests
│
├── sign_detector.py                  # Keypoint extraction (existing)
├── mediapipe-hands.py               # Hand detection demo (existing)
├── mediapipe-face.py                # Face detection demo (existing)
├── yolo.py                          # YOLO pose demo (existing)
│
├── class_names.json                  # Example class labels
├── requirements.txt                  # Python dependencies
│
├── README.md                         # Original Multi-Pose README
├── SIGN_LANGUAGE_README.md          # Detailed documentation
├── QUICKSTART.md                    # Quick start guide
├── PROJECT_STRUCTURE.md             # This file
└── LICENSE                          # MIT License

Data Directories (to be created):
├── data/                            # Training data
│   └── sign_language/
│       ├── class_0/
│       │   ├── video_001/
│       │   │   ├── 000000000001.json
│       │   │   └── ...
│       │   └── video_002/
│       └── class_1/
│
├── checkpoints/                     # Saved model checkpoints
├── logs/                           # TensorBoard logs
└── outputs/                        # Keypoint extraction outputs
    ├── json/                       # JSON keypoint sequences
    └── media/                      # Recorded videos
```

## 🧠 Core Modules

### 1. models/tcn_module.py
**Temporal Convolutional Network Encoder**

Components:
- `TemporalBlock`: Single TCN block with dilated causal convolution
- `TCNEncoder`: Multi-layer TCN with exponential dilation [1, 2, 4]
- `MultiStreamTCN`: Three parallel TCN encoders for body/hand/face

Key Features:
- Causal convolutions (no future information leakage)
- Residual connections
- Batch normalization
- Configurable kernel size and dilation

### 2. models/xlstm_module.py
**Extended LSTM Implementations**

Components:
- `sLSTMCell`: Scalar LSTM with exponential gating
- `mLSTMCell`: Matrix LSTM with multi-head attention
- `sLSTM`: Multi-layer sLSTM
- `mLSTM`: Multi-layer mLSTM

Key Features:
- Enhanced memory capacity
- Stabilized gating mechanisms
- Better long-term dependencies
- Compatible API with PyTorch LSTM

### 3. models/attention.py
**Attention-based Temporal Aggregation**

Components:
- `MultiHeadAttention`: Standard multi-head attention
- `AttentionPooling`: Learnable weighted temporal pooling
- `SelfAttentionPooling`: Self-attention with learnable query
- `TemporalAttentionAggregator`: Combined multi-strategy pooling
- `PositionalEncoding`: Sinusoidal position encoding

Key Features:
- Multi-head attention mechanism
- Combines attention, max, and average pooling
- Mask support for variable-length sequences

### 4. models/normalization.py
**Keypoint Normalization**

Components:
- `KeypointNormalizer`: General keypoint normalizer
- `BodyKeypointNormalizer`: Torso-centered body normalization
- `HandKeypointNormalizer`: Wrist-centered hand normalization
- `FaceKeypointNormalizer`: Bounding-box face normalization
- `MultiStreamNormalizer`: Combined normalizer for all streams

Key Features:
- Translation invariance (centering)
- Scale invariance (normalization)
- Optional rotation invariance
- Specialized for each keypoint type

### 5. models/sign_language_model.py
**Complete Model Architecture**

Components:
- `SignLanguageRecognitionModel`: Main classification model
- `SignLanguageSequenceModel`: Sequence-to-sequence model
- `create_sign_language_model()`: Factory function

Architecture Flow:
```
Input Keypoints
    ↓
Normalization (MultiStreamNormalizer)
    ↓
TCN Encoders (MultiStreamTCN)
    ↓
Positional Encoding (optional)
    ↓
xLSTM Layers (mLSTM or sLSTM)
    ↓
Temporal Attention (TemporalAttentionAggregator)
    ↓
Classification Head (FC + Softmax)
    ↓
Output Predictions
```

## 📊 Training & Inference

### dataset.py
**Data Loading**

Components:
- `SignLanguageDataset`: PyTorch Dataset for keypoint sequences
- `VideoKeypointSequence`: Load complete video sequences
- `create_dataloaders()`: Create train/val/test loaders

Features:
- Sliding window sampling
- Data augmentation (flip, scale, temporal shift)
- Handles variable-length videos
- Efficient batching

### train.py
**Training Script**

Components:
- `Trainer`: Training loop manager
- Command-line argument parsing
- TensorBoard logging
- Checkpoint management

Features:
- Learning rate scheduling
- Gradient clipping
- Best model saving
- Resume training support
- Validation monitoring

### inference.py
**Inference Script**

Components:
- `SignLanguagePredictor`: Prediction engine
- Multiple inference modes
- Real-time buffer management

Modes:
1. JSON mode: Process extracted keypoints
2. Video mode: Process video files
3. Real-time mode: Live camera input

## 🔧 Utility Scripts

### test_pipeline.py
Comprehensive system tests:
- Module integration tests
- Data flow verification
- Model save/load testing
- Performance benchmarks

### sign_detector.py (existing)
Keypoint extraction system:
- Real-time keypoint detection
- JSON output generation
- Video recording
- Multi-person tracking

## 📚 Documentation

### QUICKSTART.md
- 5-minute setup guide
- Basic usage examples
- Common issues and solutions
- Tips for better results

### SIGN_LANGUAGE_README.md
- Complete architecture details
- Training parameters
- Performance optimization
- Deployment guide
- API reference

### PROJECT_STRUCTURE.md (this file)
- File organization
- Module descriptions
- Architecture overview
- Contribution guidelines

## 🎯 Model Parameters

### Default Configuration

```python
{
    # Keypoint dimensions
    'body_dim': 34,        # 17 keypoints × 2 (x,y)
    'hand_dim': 84,        # 42 keypoints × 2 (21 per hand)
    'face_dim': 956,       # 478 keypoints × 2
    
    # TCN parameters
    'tcn_hidden_dim': 256,
    'tcn_num_layers': 3,
    'tcn_kernel_size': 3,
    'tcn_dilations': [1, 2, 4],
    'tcn_dropout': 0.2,
    
    # xLSTM parameters
    'xlstm_type': 'mlstm',  # or 'slstm'
    'xlstm_hidden_dim': 256,
    'xlstm_num_layers': 2,
    'xlstm_head_dim': 32,   # for mLSTM
    'xlstm_dropout': 0.2,
    
    # Attention parameters
    'attn_hidden_dim': 128,
    'attn_num_heads': 8,
    'use_multi_pool': True,
    
    # Classifier parameters
    'classifier_hidden_dim': 256,
    'classifier_dropout': 0.3,
    
    # Other
    'use_positional_encoding': True
}
```

## 📈 Model Statistics

### Typical Model Size
- **Parameters**: ~15-20M (depending on configuration)
- **Memory**: ~500MB-1GB GPU memory
- **Inference Speed**: 15-30 FPS on GPU

### Input Specifications
- **Sequence Length**: 32 frames (configurable)
- **Body Keypoints**: 17 points (COCO format)
- **Hand Keypoints**: 42 points (21 per hand)
- **Face Keypoints**: 478 points (MediaPipe Face Mesh)
- **Total Keypoints**: 537 points per frame

### Output Specifications
- **Classification**: Class probabilities (num_classes,)
- **Confidence Score**: Float [0, 1]
- **Attention Weights**: For visualization

## 🔄 Data Flow

### Training Pipeline
```
1. Load JSON keypoint files
   ↓
2. Create sequences (sliding window)
   ↓
3. Apply augmentation
   ↓
4. Normalize keypoints
   ↓
5. Feed to model
   ↓
6. Compute loss
   ↓
7. Backpropagation
   ↓
8. Update weights
```

### Inference Pipeline
```
1. Video/Camera input
   ↓
2. Extract keypoints (YOLO + MediaPipe)
   ↓
3. Buffer frames
   ↓
4. Normalize keypoints
   ↓
5. Model forward pass
   ↓
6. Softmax predictions
   ↓
7. Display results
```

## 🧪 Testing

### Unit Tests
```bash
# Test individual modules
python models/tcn_module.py
python models/xlstm_module.py
python models/attention.py
python models/normalization.py
python models/sign_language_model.py
```

### Integration Test
```bash
# Test complete pipeline
python test_pipeline.py
```

### System Test
```bash
# Test with real data
python dataset.py
python train.py --help
python inference.py --help
```

## 🚀 Getting Started Workflow

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect Data**
   ```bash
   python sign_detector.py --mode realtime
   ```

3. **Organize Data**
   ```bash
   mkdir -p data/sign_language/class_name/video_001
   mv outputs/json/timestamp/* data/sign_language/class_name/video_001/
   ```

4. **Train Model**
   ```bash
   python train.py --data_root data/sign_language --num_classes N
   ```

5. **Evaluate Model**
   ```bash
   python inference.py --mode video --model_path checkpoints/best_acc.pth
   ```

## 🤝 Contributing

To add new features:
1. Create new module in `models/`
2. Add imports to `models/__init__.py`
3. Update tests in `test_pipeline.py`
4. Document in appropriate README

## 📝 Notes

- All modules use PyTorch conventions
- Models are GPU-compatible (CUDA)
- Supports batch processing
- Implements gradient checkpointing for memory efficiency
- Modular design for easy experimentation

## 🔗 Dependencies

Critical dependencies:
- **PyTorch**: Deep learning framework
- **MediaPipe**: Hand and face landmark detection
- **Ultralytics**: YOLO-Pose for body detection
- **OpenCV**: Video processing
- **TensorBoard**: Training visualization

See `requirements.txt` for complete list.
