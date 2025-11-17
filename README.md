# Multi-Pose: A Mesh-Structured Network of Keypoints for Landmark Detection

**Multi-Pose** is a high-performance, real-time framework designed to capture comprehensive dynamic human feature points. The core idea of this framework is to merge the strengths of different models—combining the fast full-body pose estimation of **YOLOv11-Pose** with the high-precision hand and face details from **Google MediaPipe**—to generate a unified, mesh-structured network of **537 Landmarks** for a single person.

This system is not merely a detection tool; it is a powerful data acquisition engine that provides an extremely rich, structured data source for applications such as Sign Language Recognition (SLR), Virtual Avatar Control, Emotion Analysis, and advanced Human-Computer Interaction (HCI).

## Core Features

  - **Real-Time High Performance**: Achieves smooth real-time detection under GPU acceleration and accurately calculates and displays the true processing Frame Rate (Real FPS) to evaluate system performance.
  - **Hybrid Model Architecture**:
      - **Full Body Pose**: Uses `YOLOv11-Pose` for fast and robust human detection and localization of 17 key joints.
      - **High-Fidelity Hands**: Uses `MediaPipe Hands` to detect both hands across the full frame, capturing 21 fine-grained finger joint points per hand.
      - **Dense Face Mesh**: Uses `MediaPipe Face Mesh` within the YOLO-localized face region to generate a dense mesh of up to 478 landmarks, accurately capturing expression details.
  - **Comprehensive Landmark Coverage - 537 Points**:
      - **Body Pose**: 17 Landmarks
      - **Hands**: 42 Landmarks (21 Left + 21 Right)
      - **Face Mesh**: 478 Landmarks
  - **Structured JSON Output**:
      - Generates a separate, sequentially numbered JSON file for every frame of the video (e.g., `000000000001.json`).
      - The data format is clear, detailing the Frame ID, number of detected persons, Person ID, and the index, `x, y` coordinates, and confidence score for all body, hand, and face landmarks.
  - **Informative Visualization**:
      - Real-time rendering of detection results from all models, including skeleton connections, hand joints, and the face mesh.
      - Dynamically displays key information such as Real FPS, detection status for each module (OK/X), and the total number of persons.

## Tech Stack

  - **Pose Estimation**: [Ultralytics YOLOv11-Pose](https://github.com/ultralytics/ultralytics)
  - **Hand & Face Landmarks**: [Google MediaPipe](https://developers.google.com/mediapipe)
  - **Core Framework**: PyTorch
  - **Image Processing**: OpenCV
  - **Numerical Computing**: NumPy

## Setup and Installation

### 1\. Prerequisites

  - Python 3.8+
  - **Strongly Recommended**: NVIDIA GPU with CUDA & cuDNN for real-time performance.

### 2\. Clone the Repository

```bash
git clone [Your GitHub Repository Link]
cd Multi-Pose
```

### 3\. Create and Activate Python Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4\. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install ultralytics mediapipe opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

*(Please choose the PyTorch command corresponding to your CUDA version)*

### 5\. Download Pre-trained Models

Create a `models` folder in the project root and place the following model files inside.

```
Multi-Pose/
└── models/
    ├── yolo11n-pose.pt        # YOLOv8-Pose model
    ├── hand_landmarker.task  # MediaPipe Hands model
    └── face_landmarker.task  # MediaPipe Face Mesh model
```

  - YOLO Models: [Ultralytics GitHub Releases](https://github.com/ultralytics/assets/releases)
  - MediaPipe Task Models: [MediaPipe for Python Models Page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#models)

## Running the System

Start the system via `main.py`. Currently, the real-time webcam mode is the primary supported mode.

### Real-Time Webcam Mode

This mode will activate the default camera and save the structured JSON data in real-time to `output_json/`.

```bash
python main.py --mode realtime
```

  - Specify a different camera:
    ```bash
    python main.py --mode realtime --camera 1
    ```
  - Run with CPU (Performance will be significantly reduced):
    ```bash
    python main.py --device cpu
    ```

## Output JSON Data Structure

A JSON file is generated for every frame, with a clear data structure for easy parsing and use.

```json
{
    "frame_id": 123,
    "num_persons": 1,
    "persons": [
        {
            "person_id": 0,
            "keypoints": {
                "pose": [
                    { "id": 0, "x": 640.5, "y": 320.1, "confidence": 0.95 },
                    ... 16 more ...
                ],
                "left_hand": [
                    { "id": 0, "x": 410.2, "y": 450.7, "confidence": 0.99 },
                    ... 20 more ...
                ],
                "right_hand": [
                    { "id": 0, "x": 810.2, "y": 450.7, "confidence": 0.99 },
                    ... 20 more ...
                ],
                "face": [
                    { "id": 0, "x": 630.1, "y": 280.6, "confidence": 1.0 },
                    ... 477 more ...
                ]
            }
        }
    ]
}
```

  - **`frame_id`**: Sequential number of the frame.
  - **`num_persons`**: Total number of persons detected in the frame.
  - **`persons`**: A list containing data for all persons.
      - **`person_id`**: Unique ID for the person (currently primarily tracking ID 0).
      - **`keypoints`**: An object containing all landmarks for this person.
          - **`pose`**, **`left_hand`**, **`right_hand`**, **`face`**: Lists of landmarks for each part.
              - **`id`**: Index of the landmark within that part (e.g., 0 for nose in pose).
              - **`x`, `y`**: Absolute pixel coordinates in the original image frame.
              - **`confidence`**: Confidence score for the landmark.

## Multi-Stream Sign Classification (TCN + xLSTM)

Once landmark sequences are produced, the `sign_classifier.py` module provides a trainable classifier that matches the flow you described:

1. **Per-stream Normalization + Temporal Convolution**  
   Body, hand, and face sequences each pass through independent normalization and TCN encoders (kernel size 3, dilations 1/2/4) to capture short- and mid-range motion.
2. **Feature Fusion**  
   Encoded features are concatenated and linearly projected to create a fused representation of the signer state at every timestep.
3. **xLSTM Stack**  
   One or more xLSTM layers (multiplicative LSTM by default, switchable to stacked LSTM) provide long-term temporal reasoning. Optional bidirectionality is available on the sLSTM variant.
4. **Temporal Attention Pooling**  
   A lightweight attention pooling compresses the full sequence into a single context vector.
5. **Classification Head**  
   A final layer-normalized MLP produces logits for gloss/sign labels.

### Quick Start

```python
import torch
from sign_classifier import build_default_classifier

model = build_default_classifier(
    num_body_joints=17,
    num_hand_joints=42,   # 21 per hand
    num_face_joints=478,
    num_classes=200,      # adjust to your vocabulary size
)

batch_size, time_steps = 4, 64
inputs = {
    "body": torch.randn(batch_size, time_steps, 17 * 3),
    "hand": torch.randn(batch_size, time_steps, 42 * 3),
    "face": torch.randn(batch_size, time_steps, 478 * 3),
}

outputs = model(inputs, return_attention=True)
logits = outputs["logits"]     # [B, num_classes]
attention = outputs["attention_weights"]  # optional visualization
```

Feed the flattened `(x, y, confidence)` values from the JSON files into the corresponding tensors before training. You can swap the xLSTM variant by passing `xlstm_config=XlstmConfig(variant="slstm", bidirectional=True)` when building the classifier.

### Training, Evaluation, and Running Inference

Use `sign_classifier_runner.py` for complete workflows.

1. Create a manifest JSON; each entry references NumPy arrays (shape `[T, F]`) for body, hand, and face streams and can optionally include a label:
   ```json
   [
     {
       "id": "sample-0001",
       "body": "sequences/sample-0001_body.npy",
       "hand": "sequences/sample-0001_hand.npy",
       "face": "sequences/sample-0001_face.npy",
       "label": "hello"
     }
   ]
   ```
2. Run the desired mode:
   - **Train (with optional validation split)**  
     ```bash
     python sign_classifier_runner.py \
         --mode train \
         --manifest data/train_manifest.json \
         --val_manifest data/val_manifest.json \
         --save_dir checkpoints/slr
     ```
   - **Evaluate a checkpoint**  
     ```bash
     python sign_classifier_runner.py \
         --mode eval \
         --manifest data/test_manifest.json \
         --checkpoint checkpoints/slr/best_classifier.pt
     ```
   - **Run inference / deployment**  
     ```bash
     python sign_classifier_runner.py \
         --mode predict \
         --manifest data/live_manifest.json \
         --checkpoint checkpoints/slr/best_classifier.pt \
         --top_k 3 \
         --output_path outputs/predictions.json
     ```

## Contributing

Contributions in any form are welcome! Reporting issues, requesting new features, or submitting Pull Requests are all greatly helpful to this project.

## License

This project is licensed under the [MIT License](LICENSE).

-----

## Citation

If you use this system or the data generated by it for academic purposes, please cite the project using the following BibTeX format.

**Note:** Since this is a GitHub project and not a published paper, we use the `@misc` type. Please replace the bracketed placeholders with the actual information (e.g., GitHub username, specific release version, and year).

```bibtex
@misc{Multipose-Yucheng0208,
  author       = {Yucheng0208},
  title        = {{Multi-Pose}: A Mesh-Structured Network of Keypoints for Landmark Detection},
  howpublished = {\url{https://github.com/Yucheng0208/Multi-Pose}},
  year         = {2026}, % 
  note         = {GitHub Repository. Accessed: \today}
}
```
