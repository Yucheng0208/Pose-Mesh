"""
Inference Script for Sign Language Recognition
Performs sign language recognition on videos or real-time camera input
"""
import torch
import torch.nn.functional as F
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from collections import deque
import time

# Add models directory to path
sys.path.append('models')

from models.sign_language_model import create_sign_language_model
from dataset import VideoKeypointSequence
from sign_detector import SignLanguageDetector


class SignLanguagePredictor:
    """
    Real-time sign language prediction from keypoint sequences
    """
    def __init__(self,
                 model_path: str,
                 class_names_path: str,
                 device: str = 'cuda',
                 sequence_length: int = 32,
                 confidence_threshold: float = 0.5):
        """
        Args:
            model_path: Path to trained model checkpoint
            class_names_path: Path to JSON file with class names
            device: Device to run inference on
            sequence_length: Length of input sequence
            confidence_threshold: Minimum confidence for prediction
        """
        self.device = device
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        self.num_classes = len(self.class_names)
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        self.model = create_sign_language_model(
            model_type='classifier',
            num_classes=self.num_classes,
            xlstm_type='mlstm',  # Should match training config
            tcn_hidden_dim=256,
            xlstm_hidden_dim=256,
            xlstm_num_layers=2
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Number of classes: {self.num_classes}")
        
        # Frame buffer for sliding window
        self.frame_buffer = deque(maxlen=sequence_length)
        
    def reset_buffer(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()
    
    def predict_from_keypoints(self, body_seq, hand_seq, face_seq):
        """
        Make prediction from keypoint sequences
        
        Args:
            body_seq: (time, 17, 2)
            hand_seq: (time, 42, 2)
            face_seq: (time, 478, 2)
        
        Returns:
            predicted_class: Predicted class index
            confidence: Confidence score
            probabilities: Full probability distribution
        """
        # Add batch dimension
        body_seq = body_seq.unsqueeze(0).to(self.device)  # (1, time, 17, 2)
        hand_seq = hand_seq.unsqueeze(0).to(self.device)
        face_seq = face_seq.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits, attention_weights = self.model(body_seq, hand_seq, face_seq)
            probabilities = F.softmax(logits, dim=-1)
            confidence, predicted_class = probabilities.max(1)
        
        confidence = confidence.item()
        predicted_class = predicted_class.item()
        probabilities = probabilities.squeeze(0).cpu().numpy()
        
        return predicted_class, confidence, probabilities
    
    def predict_from_json_dir(self, json_dir: str, stride: int = 8):
        """
        Make predictions from a directory of JSON keypoint files
        
        Args:
            json_dir: Directory containing JSON files
            stride: Stride for sliding window
        
        Returns:
            predictions: List of (frame_idx, class_idx, class_name, confidence)
        """
        # Load keypoint sequence
        video_seq = VideoKeypointSequence(json_dir)
        body_seq, hand_seq, face_seq = video_seq.load_sequence()
        
        num_frames = body_seq.size(0)
        predictions = []
        
        # Sliding window prediction
        for start_idx in range(0, num_frames - self.sequence_length + 1, stride):
            end_idx = start_idx + self.sequence_length
            
            # Extract window
            body_window = body_seq[start_idx:end_idx]
            hand_window = hand_seq[start_idx:end_idx]
            face_window = face_seq[start_idx:end_idx]
            
            # Predict
            class_idx, confidence, probs = self.predict_from_keypoints(
                body_window, hand_window, face_window
            )
            
            if confidence >= self.confidence_threshold:
                class_name = self.class_names[class_idx]
                predictions.append({
                    'frame_start': start_idx,
                    'frame_end': end_idx,
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'confidence': confidence
                })
        
        return predictions
    
    def update_buffer(self, body_kpts, hand_kpts, face_kpts):
        """
        Update frame buffer with new keypoints
        
        Args:
            body_kpts: (17, 2) numpy array
            hand_kpts: (42, 2) numpy array
            face_kpts: (478, 2) numpy array
        """
        frame = {
            'body': torch.from_numpy(body_kpts).float(),
            'hand': torch.from_numpy(hand_kpts).float(),
            'face': torch.from_numpy(face_kpts).float()
        }
        self.frame_buffer.append(frame)
    
    def predict_from_buffer(self):
        """
        Make prediction from current buffer
        
        Returns:
            prediction dict or None if buffer not full
        """
        if len(self.frame_buffer) < self.sequence_length:
            return None
        
        # Stack frames
        body_seq = torch.stack([f['body'] for f in self.frame_buffer])
        hand_seq = torch.stack([f['hand'] for f in self.frame_buffer])
        face_seq = torch.stack([f['face'] for f in self.frame_buffer])
        
        # Predict
        class_idx, confidence, probs = self.predict_from_keypoints(
            body_seq, hand_seq, face_seq
        )
        
        if confidence >= self.confidence_threshold:
            return {
                'class_idx': class_idx,
                'class_name': self.class_names[class_idx],
                'confidence': confidence,
                'probabilities': probs
            }
        
        return None


def process_video_with_detector(video_path: str,
                                predictor: SignLanguagePredictor,
                                output_path: str = None,
                                display: bool = True):
    """
    Process video with keypoint detection and sign recognition
    
    Args:
        video_path: Path to input video
        predictor: SignLanguagePredictor instance
        output_path: Path to save output video (optional)
        display: Whether to display results
    """
    # Initialize detector
    detector = SignLanguageDetector(
        yolo_model_path="models/yolo11n-pose.pt",
        hand_model_path="models/hand_landmarker.task",
        face_model_path="models/face_landmarker.task",
        device='cuda',
        confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    predictor.reset_buffer()
    current_prediction = None
    prediction_history = deque(maxlen=10)
    
    frame_idx = 0
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect keypoints
            processed_frame, keypoints_data, num_persons = detector.process_frame(frame)
            
            # Extract keypoints as numpy arrays
            body_kpts = np.zeros((17, 2), dtype=np.float32)
            hand_kpts = np.zeros((42, 2), dtype=np.float32)
            face_kpts = np.zeros((478, 2), dtype=np.float32)
            
            if keypoints_data['pose'] is not None:
                pose = keypoints_data['pose']
                body_kpts = pose[:, :2]  # Extract x, y
            
            if keypoints_data['hands']['left'] is not None:
                left_hand_lms, _ = keypoints_data['hands']['left']
                for i, lm in enumerate(left_hand_lms):
                    hand_kpts[i] = [lm.x * width, lm.y * height]
            
            if keypoints_data['hands']['right'] is not None:
                right_hand_lms, _ = keypoints_data['hands']['right']
                for i, lm in enumerate(right_hand_lms):
                    hand_kpts[21 + i] = [lm.x * width, lm.y * height]
            
            if keypoints_data['face'] is not None:
                face_lms, _ = keypoints_data['face']
                roi_coords = detector.current_roi_coords.get('face')
                if roi_coords:
                    x_min, y_min, x_max, y_max = roi_coords
                    roi_w, roi_h = x_max - x_min, y_max - y_min
                    for i, lm in enumerate(face_lms):
                        face_kpts[i] = [x_min + lm.x * roi_w, y_min + lm.y * roi_h]
            
            # Update buffer and predict
            predictor.update_buffer(body_kpts, hand_kpts, face_kpts)
            prediction = predictor.predict_from_buffer()
            
            if prediction:
                prediction_history.append(prediction['class_name'])
                # Use majority vote from history
                if len(prediction_history) >= 5:
                    most_common = max(set(prediction_history), key=prediction_history.count)
                    current_prediction = {
                        'class_name': most_common,
                        'confidence': prediction['confidence']
                    }
            
            # Draw prediction on frame
            if current_prediction:
                text = f"{current_prediction['class_name']}: {current_prediction['confidence']:.2f}"
                cv2.rectangle(processed_frame, (10, 10), (500, 60), (0, 0, 0), -1)
                cv2.putText(processed_frame, text, (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Display
            if display:
                cv2.imshow('Sign Language Recognition', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write output
            if out:
                out.write(processed_frame)
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if output_path:
            print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Sign Language Recognition Inference')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['json', 'video', 'realtime'],
                        help='Inference mode')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--class_names', type=str, required=True,
                        help='Path to JSON file with class names')
    parser.add_argument('--input', type=str,
                        help='Input path (JSON dir or video file)')
    parser.add_argument('--output', type=str,
                        help='Output path for video')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Sequence length')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID for realtime mode')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create predictor
    predictor = SignLanguagePredictor(
        model_path=args.model_path,
        class_names_path=args.class_names,
        device=device,
        sequence_length=args.sequence_length,
        confidence_threshold=args.confidence
    )
    
    if args.mode == 'json':
        # Process from JSON directory
        if not args.input:
            print("Error: --input required for JSON mode")
            return
        
        print(f"Processing JSON directory: {args.input}")
        predictions = predictor.predict_from_json_dir(args.input)
        
        print(f"\nFound {len(predictions)} predictions:")
        for pred in predictions:
            print(f"  Frames {pred['frame_start']}-{pred['frame_end']}: "
                  f"{pred['class_name']} ({pred['confidence']:.2f})")
        
        # Save predictions
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"\nPredictions saved to: {args.output}")
    
    elif args.mode == 'video':
        # Process video
        if not args.input:
            print("Error: --input required for video mode")
            return
        
        process_video_with_detector(
            video_path=args.input,
            predictor=predictor,
            output_path=args.output,
            display=True
        )
    
    elif args.mode == 'realtime':
        # Real-time processing
        print("Real-time mode not yet implemented")
        print("Use video mode with camera input: --mode video --input 0")


if __name__ == "__main__":
    main()
