# Sign Language Recognition System

A state-of-the-art sign language recognition system based on multi-stream Temporal Convolutional Networks (TCN) and extended LSTM (xLSTM) architecture.

## Architecture Overview

```
RGB Frames → Pose/Hand/Face Extraction → Multi-stream TCN → xLSTM → Attention Pooling → Classification
```

### Key Components

1. **Multi-stream Keypoint Extraction** (537 landmarks total)
   - Body: 17 keypoints (YOLO-Pose)
   - Hands: 42 keypoints (21 per hand, MediaPipe)
   - Face: 478 keypoints (MediaPipe Face Mesh)

2. **Keypoint Normalization**
   - Translation, scale, and rotation invariance
   - Specialized normalizers for each stream

3. **TCN Encoders** (per stream)
   - Kernel size: 3
   - Dilations: [1, 2, 4]
   - Multi-layer temporal convolutions

4. **Feature Fusion**
   - Concatenation + Linear projection
   - Layer normalization

5. **xLSTM Layers**
   - mLSTM (matrix LSTM) or sLSTM (scalar LSTM)
   - Enhanced long-term memory capability
   - 2-layer stack

6. **Temporal Attention Aggregation**
   - Multi-head self-attention
   - Attention pooling
   - Combines multiple pooling strategies

7. **Classification Head**
   - Fully connected layers
   - Softmax output

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python
- mediapipe
- ultralytics
- numpy
- tensorboard
- tqdm

## Data Preparation

### 1. Collect Keypoint Data

Use the existing `sign_detector.py` to extract keypoints from videos:

```bash
python sign_detector.py --mode realtime --camera 0
```

This will generate JSON files with keypoint sequences in `outputs/json/`.

### 2. Organize Dataset

Structure your dataset as follows:

```
data/
  sign_language/
    class_0_hello/
      video_001/
        000000000001.json
        000000000002.json
        ...
      video_002/
        ...
    class_1_thank_you/
      video_001/
        ...
    ...
```

### 3. Create Class Names File

Create a `class_names.json` file:

```json
{
  "0": "hello",
  "1": "thank_you",
  "2": "please",
  ...
}
```

## Training

### Basic Training

```bash
python train.py \
  --data_root data/sign_language \
  --num_classes 30 \
  --sequence_length 32 \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 1e-3
```

### Advanced Training Options

```bash
python train.py \
  --data_root data/sign_language \
  --num_classes 100 \
  --sequence_length 32 \
  --batch_size 16 \
  --num_epochs 200 \
  --lr 1e-3 \
  --xlstm_type mlstm \
  --tcn_hidden_dim 256 \
  --xlstm_hidden_dim 256 \
  --xlstm_num_layers 2 \
  --weight_decay 1e-4 \
  --device cuda \
  --save_dir checkpoints \
  --log_dir logs
```

### Resume Training

```bash
python train.py \
  --data_root data/sign_language \
  --num_classes 30 \
  --resume checkpoints/checkpoint_epoch_50.pth
```

### Monitor Training

Use TensorBoard to monitor training:

```bash
tensorboard --logdir logs
```

## Inference

### 1. Process JSON Keypoints

If you already have extracted keypoints:

```bash
python inference.py \
  --mode json \
  --model_path checkpoints/best_acc.pth \
  --class_names class_names.json \
  --input outputs/json/2024-11-17_10-30-00 \
  --output predictions.json
```

### 2. Process Video File

Process a video file with automatic keypoint extraction:

```bash
python inference.py \
  --mode video \
  --model_path checkpoints/best_acc.pth \
  --class_names class_names.json \
  --input input_video.mp4 \
  --output output_video.mp4
```

### 3. Real-time Recognition

Real-time sign language recognition from camera:

```bash
python inference.py \
  --mode video \
  --model_path checkpoints/best_acc.pth \
  --class_names class_names.json \
  --input 0
```

## Model Architecture Details

### TCN Encoder
- **Input**: Normalized keypoint sequences
- **Layers**: 3 temporal convolutional layers
- **Dilations**: [1, 2, 4] for multi-scale temporal receptive field
- **Activation**: ReLU
- **Normalization**: Batch normalization
- **Residual connections**: Yes

### xLSTM Variants

#### mLSTM (Matrix LSTM)
- Uses matrix-valued memory for enhanced capacity
- Multi-head attention mechanism
- Better for capturing complex temporal dependencies

#### sLSTM (Scalar LSTM)
- Enhanced version of standard LSTM
- Exponential gating with stabilization
- More memory efficient

### Attention Aggregation
- **Multi-head self-attention**: 8 heads
- **Attention pooling**: Learnable weighted aggregation
- **Multi-pooling**: Combines attention, max, and average pooling

## Performance Optimization

### GPU Memory Optimization
- Use gradient accumulation for larger effective batch size
- Mixed precision training with `torch.cuda.amp`

### Data Augmentation
- Horizontal flipping (with left/right hand swap)
- Random scaling (0.9-1.1)
- Temporal shifting

### Training Tips
1. Start with a small learning rate (1e-3) and use learning rate scheduling
2. Use gradient clipping (max_norm=1.0) to prevent exploding gradients
3. Monitor validation loss and use early stopping
4. Fine-tune on specific sign vocabulary for better performance

## Model Export

Export trained model for deployment:

```python
import torch
from models.sign_language_model import create_sign_language_model

# Load model
model = create_sign_language_model(
    model_type='classifier',
    num_classes=30,
    xlstm_type='mlstm'
)

checkpoint = torch.load('checkpoints/best_acc.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Export to TorchScript
model.eval()
example_body = torch.randn(1, 32, 17, 2)
example_hand = torch.randn(1, 32, 42, 2)
example_face = torch.randn(1, 32, 478, 2)

traced_model = torch.jit.trace(
    model, 
    (example_body, example_hand, example_face)
)
traced_model.save('model.pt')
```

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test TCN module
cd models
python tcn_module.py

# Test xLSTM module
python xlstm_module.py

# Test attention module
python attention.py

# Test normalization module
python normalization.py

# Test complete model
python sign_language_model.py
```

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing
- Use mixed precision training

### Poor Convergence
- Check data quality and labels
- Increase model capacity
- Adjust learning rate
- Add more data augmentation

### Low Accuracy
- Collect more training data
- Balance class distribution
- Increase sequence length
- Use ensemble of models

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{signlanguage-tcn-xlstm,
  title={Multi-Stream TCN-xLSTM for Sign Language Recognition},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/sign-language-recognition}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO-Pose for body keypoint detection
- Google MediaPipe for hand and face landmarks
- xLSTM architecture based on recent advances in LSTM
