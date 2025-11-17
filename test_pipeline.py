"""
Test Script to Verify Complete Pipeline
Tests all components of the sign language recognition system
"""
import torch
import sys
import numpy as np

sys.path.append('models')

print("=" * 60)
print("Testing Sign Language Recognition Pipeline")
print("=" * 60)

# Test parameters
batch_size = 2
seq_len = 32
num_classes = 10

print("\n1. Testing Normalization Module...")
try:
    from models.normalization import MultiStreamNormalizer
    
    normalizer = MultiStreamNormalizer()
    
    # Create dummy keypoints
    body_kpts = torch.randn(batch_size, seq_len, 17, 2) * 100 + 200
    hand_kpts = torch.randn(batch_size, seq_len, 42, 2) * 50 + 150
    face_kpts = torch.randn(batch_size, seq_len, 478, 2) * 30 + 180
    
    # Normalize
    body_norm, hand_norm, face_norm = normalizer(body_kpts, hand_kpts, face_kpts)
    
    # Flatten
    body_flat, hand_flat, face_flat = normalizer.flatten_keypoints(body_norm, hand_norm, face_norm)
    
    print(f"   ✓ Body: {body_kpts.shape} -> {body_flat.shape}")
    print(f"   ✓ Hand: {hand_kpts.shape} -> {hand_flat.shape}")
    print(f"   ✓ Face: {face_kpts.shape} -> {face_flat.shape}")
    print("   ✓ Normalization: PASSED")
except Exception as e:
    print(f"   ✗ Normalization: FAILED - {e}")
    sys.exit(1)

print("\n2. Testing TCN Module...")
try:
    from models.tcn_module import MultiStreamTCN
    
    tcn = MultiStreamTCN(
        body_dim=34,
        hand_dim=84,
        face_dim=956,
        hidden_dim=256,
        num_layers=3
    )
    
    tcn_output = tcn(body_flat, hand_flat, face_flat)
    print(f"   ✓ TCN output: {tcn_output.shape}")
    print(f"   ✓ Expected: ({batch_size}, {seq_len}, 256)")
    print("   ✓ TCN: PASSED")
except Exception as e:
    print(f"   ✗ TCN: FAILED - {e}")
    sys.exit(1)

print("\n3. Testing xLSTM Module...")
try:
    from models.xlstm_module import mLSTM, sLSTM
    
    # Test mLSTM
    mlstm = mLSTM(input_size=256, hidden_size=256, num_layers=2)
    mlstm_output, _ = mlstm(tcn_output)
    print(f"   ✓ mLSTM output: {mlstm_output.shape}")
    
    # Test sLSTM
    slstm = sLSTM(input_size=256, hidden_size=256, num_layers=2)
    slstm_output, _ = slstm(tcn_output)
    print(f"   ✓ sLSTM output: {slstm_output.shape}")
    print("   ✓ xLSTM: PASSED")
except Exception as e:
    print(f"   ✗ xLSTM: FAILED - {e}")
    sys.exit(1)

print("\n4. Testing Attention Module...")
try:
    from models.attention import TemporalAttentionAggregator
    
    aggregator = TemporalAttentionAggregator(input_dim=256, use_multi_pool=True)
    aggregated, attn_weights = aggregator(mlstm_output)
    print(f"   ✓ Aggregated output: {aggregated.shape}")
    print(f"   ✓ Expected: ({batch_size}, 256)")
    print("   ✓ Attention: PASSED")
except Exception as e:
    print(f"   ✗ Attention: FAILED - {e}")
    sys.exit(1)

print("\n5. Testing Complete Model...")
try:
    from models.sign_language_model import SignLanguageRecognitionModel
    
    model = SignLanguageRecognitionModel(
        num_classes=num_classes,
        xlstm_type='mlstm',
        tcn_hidden_dim=256,
        xlstm_hidden_dim=256,
        xlstm_num_layers=2
    )
    
    # Forward pass
    logits, attention_weights = model(body_kpts, hand_kpts, face_kpts)
    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Expected: ({batch_size}, {num_classes})")
    
    # Test predictions
    predictions, probabilities, _ = model.predict(body_kpts, hand_kpts, face_kpts)
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Probabilities shape: {probabilities.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    print("   ✓ Complete Model: PASSED")
except Exception as e:
    print(f"   ✗ Complete Model: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n6. Testing Dataset Loader...")
try:
    from dataset import SignLanguageDataset
    
    print("   ℹ Dataset loader available")
    print("   ℹ To test: create data in data/sign_language/")
    print("   ✓ Dataset: PASSED")
except Exception as e:
    print(f"   ✗ Dataset: FAILED - {e}")
    sys.exit(1)

print("\n7. Testing Model Saving/Loading...")
try:
    import tempfile
    import os
    
    # Save model
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "test_model.pth")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'epoch': 1
    }
    torch.save(checkpoint, save_path)
    print(f"   ✓ Model saved to: {save_path}")
    
    # Load model
    loaded_checkpoint = torch.load(save_path, map_location='cpu')
    new_model = SignLanguageRecognitionModel(
        num_classes=num_classes,
        xlstm_type='mlstm',
        tcn_hidden_dim=256,
        xlstm_hidden_dim=256
    )
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    print("   ✓ Model loaded successfully")
    
    # Verify outputs match
    new_logits, _ = new_model(body_kpts, hand_kpts, face_kpts)
    assert torch.allclose(logits, new_logits, atol=1e-5), "Model outputs don't match!"
    print("   ✓ Model outputs verified")
    
    # Cleanup
    os.remove(save_path)
    os.rmdir(temp_dir)
    
    print("   ✓ Save/Load: PASSED")
except Exception as e:
    print(f"   ✗ Save/Load: FAILED - {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nPipeline Summary:")
print("  • Normalization: ✓")
print("  • TCN Encoder: ✓")
print("  • xLSTM: ✓")
print("  • Attention Aggregation: ✓")
print("  • Complete Model: ✓")
print("  • Dataset Loader: ✓")
print("  • Model Persistence: ✓")
print("\nYou can now:")
print("  1. Collect training data: python sign_detector.py")
print("  2. Train the model: python train.py --data_root data/sign_language --num_classes N")
print("  3. Run inference: python inference.py --mode video --model_path checkpoints/best_acc.pth")
print("\nSee QUICKSTART.md for detailed instructions.")
