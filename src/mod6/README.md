# Module 6: Real-Time Prediction & User Interface

## Overview
This module implements real-time Tamil sign language recognition using webcam, MediaPipe hand landmark detection, and the trained Random Forest model from Module 5.

## Features
- **Real-time webcam capture** with OpenCV
- **Hand landmark detection** using MediaPipe
- **Live prediction** of Tamil alphabet signs
- **Prediction smoothing** for stable results
- **Visual feedback** with hand landmarks overlay
- **FPS monitoring** and performance tracking
- **Frame capture** functionality
- **Single image prediction** capability

## Components

### 1. RealtimePredictor Class
Main class for real-time prediction.

**Key Features:**
- Webcam initialization and configuration
- MediaPipe Hands integration
- Model loading and inference
- Temporal prediction smoothing
- Visual overlay rendering
- Session statistics tracking

**Configuration Parameters:**
- `model_path`: Path to trained model file (.pkl)
- `min_detection_confidence`: Hand detection threshold (default: 0.7)
- `min_tracking_confidence`: Hand tracking threshold (default: 0.5)
- `prediction_smoothing`: Frames for smoothing (default: 5)
- `camera_index`: Camera device (default: 0)
- `frame_width`: Video width (default: 1280)
- `frame_height`: Video height (default: 720)

### 2. Single Image Prediction
Function to predict Tamil sign from a static image file.

## Usage

### Real-Time Webcam Prediction

```python
from mod6 import RealtimePredictor

# Initialize predictor
predictor = RealtimePredictor(
    model_path="path/to/model.pkl",
    prediction_smoothing=5
)

# Run real-time prediction
predictor.run(output_dir="output")
```

### Single Image Prediction

```python
from mod6.realtime_predictor import load_and_predict_image

# Predict from image
prediction, confidence = load_and_predict_image(
    model_path="path/to/model.pkl",
    image_path="path/to/image.jpg"
)

print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
```

### Running Tests

```bash
# Run all tests (interactive)
python test_module6.py

# Test webcam directly
python test_module6.py --webcam

# Test single image only
python test_module6.py --image
```

## Controls

During webcam prediction:
- **'q'** - Quit application
- **'s'** - Save current frame with prediction
- **'r'** - Reset session statistics

## Output

### On-Screen Display
- Video feed with hand landmarks overlay
- Current prediction (Tamil character)
- Confidence score
- Real-time FPS
- Session statistics

### Saved Frames
Captured frames are saved to `output/` directory with format:
```
capture_{prediction}_{timestamp}.jpg
```

## Requirements

- OpenCV (cv2)
- MediaPipe
- NumPy
- Trained Random Forest model from Module 5
- Webcam or camera device

## Technical Details

### Pipeline Flow
1. **Frame Capture**: Read frame from webcam
2. **Preprocessing**: Flip frame, convert BGR to RGB
3. **Hand Detection**: Process with MediaPipe Hands
4. **Landmark Extraction**: Extract 21 landmarks (63 features)
5. **Feature Scaling**: Apply scaler from training
6. **Prediction**: Random Forest inference
7. **Smoothing**: Temporal averaging over N frames
8. **Visualization**: Draw landmarks and prediction info
9. **Display**: Show processed frame

### Prediction Smoothing
Uses a sliding window buffer to smooth predictions over multiple frames:
- Stores last N predictions
- Returns most frequent prediction in window
- Averages confidence scores
- Clears buffer when no hand detected

### Performance Optimization
- Uses `static_image_mode=False` for video tracking
- Efficient landmark extraction with NumPy vectorization
- FPS tracking with rolling average
- Minimal overhead on frame processing

## Troubleshooting

### Camera Not Found
- Check camera index (try 0, 1, 2...)
- Verify camera permissions
- Test camera with other applications

### Low FPS
- Reduce frame resolution
- Decrease `prediction_smoothing` value
- Use faster computer/GPU

### No Hand Detected
- Ensure good lighting conditions
- Keep hand within camera frame
- Adjust `min_detection_confidence` threshold
- Clean camera lens

### Inaccurate Predictions
- Increase `prediction_smoothing` for stability
- Ensure proper hand gesture positioning
- Re-train model with more data
- Check lighting and background

## Future Enhancements

Potential improvements:
- Support for two-hand gestures
- Gesture sequence recording
- Word formation from letter sequences
- Multi-language UI
- GPU acceleration
- Mobile deployment
- Cloud-based inference

## Notes

- First frame may be slower due to initialization
- Prediction stability improves with smoothing
- Best results with clear hand visibility
- Lighting and background affect detection quality
