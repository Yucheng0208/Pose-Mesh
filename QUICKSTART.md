# Quick Start Guide - Sign Language Recognition System

This guide will help you get started with the sign language recognition system quickly.

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Models

Make sure you have the following model files in the `models/` directory:
- `yolo11n-pose.pt` - Body pose detection
- `hand_landmarker.task` - Hand keypoint detection  
- `face_landmarker.task` - Face landmark detection

These should already be present from the Multi-Pose system.

## 📹 Collect Training Data (10 minutes)

### Record Sign Language Videos

```bash
python sign_detector.py --mode realtime --camera 0
```

- Perform different signs in front of the camera
- Each recording creates a directory in `outputs/json/` with keypoint sequences
- Record at least 10-20 examples per sign for good results

### Organize Your Data

Move the JSON directories to create your dataset:

```bash
mkdir -p data/sign_language/hello/video_001
mkdir -p data/sign_language/thank_you/video_001

# Move your recordings
mv outputs/json/2024-11-17_10-00-00/* data/sign_language/hello/video_001/
mv outputs/json/2024-11-17_10-05-00/* data/sign_language/thank_you/video_001/
```

## 🎯 Train Your First Model (30 minutes)

### Update Class Names

Edit `class_names.json` to match your signs:

```json
{
  "0": "hello",
  "1": "thank_you",
  "2": "please"
}
```

### Start Training

```bash
python train.py \
  --data_root data/sign_language \
  --num_classes 3 \
  --sequence_length 32 \
  --batch_size 8 \
  --num_epochs 50 \
  --lr 1e-3 \
  --device cuda
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs
```

## 🔮 Test Your Model

### Test on New Video

```bash
python inference.py \
  --mode video \
  --model_path checkpoints/best_acc.pth \
  --class_names class_names.json \
  --input test_video.mp4 \
  --output result.mp4
```

### Real-time Recognition

```bash
python inference.py \
  --mode video \
  --model_path checkpoints/best_acc.pth \
  --class_names class_names.json \
  --input 0
```

## 📊 Expected Results

With proper data collection:
- **Training accuracy**: 85-95% after 50 epochs
- **Validation accuracy**: 75-85% with 10+ examples per class
- **Inference speed**: 15-30 FPS on GPU

## 💡 Tips for Better Results

### Data Collection
1. **Variety**: Record from different angles and distances
2. **Consistency**: Perform signs clearly and consistently
3. **Quantity**: More data = better results (aim for 20+ examples per sign)
4. **Background**: Vary backgrounds to improve generalization

### Training
1. Start with small number of classes (3-5)
2. Use validation set to prevent overfitting
3. Monitor loss curves in TensorBoard
4. Stop when validation loss stops improving

### Inference
1. Ensure good lighting conditions
2. Position yourself similar to training data
3. Perform signs clearly and deliberately
4. Wait for confidence score > 0.7 for reliable predictions

## 🐛 Common Issues

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --batch_size 4 ...
```

### "No module named 'models'"
```bash
# Make sure you're in the project root directory
cd /workspace
python train.py ...
```

### Low accuracy
1. Collect more training data
2. Check data quality (keypoints detected properly?)
3. Increase training epochs
4. Reduce number of classes initially

## 📚 Next Steps

1. **Add More Signs**: Collect data for more sign language vocabulary
2. **Fine-tune**: Adjust hyperparameters for better performance
3. **Deploy**: Export model for production use
4. **Evaluate**: Test on independent test set

For detailed information, see [SIGN_LANGUAGE_README.md](SIGN_LANGUAGE_README.md).

## 🎓 Architecture Summary

```
Input (RGB Frames)
    ↓
Keypoint Extraction (537 landmarks)
    ↓
Normalization (Body, Hand, Face)
    ↓
TCN Encoders (3 streams, dilations: 1,2,4)
    ↓
Feature Fusion
    ↓
xLSTM Layers (2 layers, mLSTM)
    ↓
Attention Aggregation
    ↓
Classification Head
    ↓
Output (Sign Label)
```

## 🤝 Need Help?

- Check the detailed README: `SIGN_LANGUAGE_README.md`
- Test individual modules: `python models/tcn_module.py`
- Verify data loading: `python dataset.py`

Happy signing! 🤟
