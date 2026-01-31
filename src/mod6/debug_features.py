"""
Debug script to check if feature extraction matches training data format
"""
import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path

# Load model
model_path = Path(__file__).parent.parent / "mod5" / "output" / "random_forest_model.pkl"
print(f"Loading model from: {model_path}")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
label_mapping = model_data['label_mapping']

print(f"\n✓ Model loaded!")
print(f"  - Classes: {len(label_mapping)}")
print(f"  - Scaler mean shape: {scaler.mean_.shape}")
print(f"  - Expected features: {scaler.mean_.shape[0]}")

# Load a sample from the dataset to compare
dataset_path = Path(__file__).parent.parent / "mod4" / "output" / "scaled_dataset.pkl"
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['X_train']
y_train = dataset['y_train']

print(f"\n✓ Training data loaded!")
print(f"  - Training samples shape: {X_train.shape}")
print(f"  - First sample (BEFORE scaling):")
print(f"    Min: {X_train[0].min():.4f}, Max: {X_train[0].max():.4f}, Mean: {X_train[0].mean():.4f}")

# Scale it
X_train_scaled = scaler.transform(X_train[:1])
print(f"  - First sample (AFTER scaling):")
print(f"    Min: {X_train_scaled[0].min():.4f}, Max: {X_train_scaled[0].max():.4f}, Mean: {X_train_scaled[0].mean():.4f}")

# Predict on training sample
pred = model.predict(X_train_scaled)[0]
proba = model.predict_proba(X_train_scaled)[0]
max_conf = np.max(proba)
pred_char = label_mapping.get(pred, "Unknown")
actual_char = label_mapping.get(y_train[0], "Unknown")

print(f"\n✓ Training sample prediction:")
print(f"  - Actual: {actual_char} (Class {y_train[0]})")
print(f"  - Predicted: {pred_char} (Class {pred})")
print(f"  - Confidence: {max_conf:.1%}")
print(f"  - Match: {'✓ CORRECT' if pred == y_train[0] else '✗ WRONG'}")

# Now test with webcam
print(f"\n" + "="*70)
print("Testing live webcam feature extraction...")
print("="*70)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nShow your hand gesture. Press 'q' to quit.")
print("Press 'SPACE' to capture and analyze...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Extract features
        features = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        features = np.array(features).reshape(1, -1)
        
        # Draw landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, "Hand detected! Press SPACE to analyze", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            print("\n" + "-"*70)
            print("WEBCAM FEATURE ANALYSIS")
            print("-"*70)
            print(f"Raw features shape: {features.shape}")
            print(f"  Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")
            
            # Scale
            features_scaled = scaler.transform(features)
            print(f"Scaled features:")
            print(f"  Min: {features_scaled.min():.4f}, Max: {features_scaled.max():.4f}, Mean: {features_scaled.mean():.4f}")
            
            # Predict
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            # Top 5
            top5_indices = np.argsort(proba)[-5:][::-1]
            print(f"\nTop 5 predictions:")
            for i, idx in enumerate(top5_indices):
                char = label_mapping.get(idx, "Unknown")
                conf = proba[idx]
                print(f"  {i+1}. {char} (Class {idx}) - {conf:.2%}")
            
            print("-"*70)
    else:
        cv2.putText(frame, "No hand detected", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Debug - Press SPACE to analyze', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
