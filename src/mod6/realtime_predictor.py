"""
Module 6: Real-Time Tamil Sign Language Prediction
Professional version with Tamil text display and 2-hand support.
"""

import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import deque
import time
from PIL import Image, ImageDraw, ImageFont


class RealtimePredictor:
    """Professional Tamil sign language predictor with proper Tamil text rendering."""
    
    def __init__(self, model_path: str, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, prediction_smoothing: int = 3,
                 camera_index: int = 0, frame_width: int = 1280, frame_height: int = 720):
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.prediction_smoothing = prediction_smoothing
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.model = None
        self.scaler = None
        self.label_mapping = None
        self.hands = None
        self.cap = None
        
        self.prediction_buffer = deque(maxlen=prediction_smoothing)
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_label_id = None
        self.is_paused = False
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        self.last_printed_prediction = None
        self.prediction_history = []
        
        # Load Tamil font
        try:
            self.tamil_font = ImageFont.truetype("C:\\Windows\\Fonts\\NotoSansTamil-Regular.ttf", 50)
            self.small_tamil_font = ImageFont.truetype("C:\\Windows\\Fonts\\NotoSansTamil-Regular.ttf", 28)
        except:
            try:
                self.tamil_font = ImageFont.truetype("C:\\Windows\\Fonts\\LATHA.TTF", 50)
                self.small_tamil_font = ImageFont.truetype("C:\\Windows\\Fonts\\LATHA.TTF", 28)
            except:
                self.tamil_font = ImageFont.load_default()
                self.small_tamil_font = ImageFont.load_default()
        
        self.stats = {'total_frames': 0, 'hands_detected': 0, 'predictions_made': 0}
        
    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_mapping = model_data['label_mapping']
        print(f"✓ Model loaded! Classes: {len(self.label_mapping)}")
        
    def initialize_mediapipe(self):
        print("Initializing MediaPipe for 1-2 hands...")
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp_hands
        print("✓ MediaPipe initialized! (Supports 1-2 hands)")
        
    def initialize_camera(self):
        print(f"Initializing camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera initialized! Resolution: {actual_width}x{actual_height}")
        
    def extract_landmarks(self, hand_landmarks_list: List) -> np.ndarray:
        """Extract features from first hand (63 features)."""
        features = []
        for landmark in hand_landmarks_list[0].landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features).reshape(1, -1)
    
    def predict_sign(self, features: np.ndarray) -> Tuple[str, float, int, List]:
        features_scaled = self.scaler.transform(features)
        predicted_label = self.model.predict(features_scaled)[0]
        predicted_proba = self.model.predict_proba(features_scaled)[0]
        
        # Gentle confidence calibration: Only boost if top prediction is strong
        # If top probability is already high (>0.3), boost it more
        # If it's low, keep it realistic
        max_prob = np.max(predicted_proba)
        if max_prob > 0.3:
            # Strong prediction - apply moderate boost
            temperature = 0.5
        elif max_prob > 0.2:
            # Medium prediction - gentle boost
            temperature = 0.7
        else:
            # Weak prediction - minimal boost
            temperature = 0.9
        
        calibrated_proba = np.power(predicted_proba, 1.0/temperature)
        calibrated_proba = calibrated_proba / np.sum(calibrated_proba)
        
        # Get top 5 predictions with calibrated probabilities
        top5_indices = np.argsort(calibrated_proba)[-5:][::-1]
        top5_predictions = []
        for idx in top5_indices:
            char = self.label_mapping.get(idx, "Unknown")
            conf = calibrated_proba[idx]
            top5_predictions.append((char, conf, idx))
        
        confidence = np.max(calibrated_proba)
        predicted_char = self.label_mapping.get(predicted_label, "Unknown")
        return predicted_char, confidence, predicted_label, top5_predictions
    
    def smooth_prediction(self, prediction: str, confidence: float, label_id: int, top5: List) -> Tuple[str, float, int, List]:
        self.prediction_buffer.append((prediction, confidence, label_id))
        predictions = [p for p, c, l in self.prediction_buffer]
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        smoothed_pred = max(pred_counts, key=pred_counts.get)
        smoothed_label = [l for p, c, l in self.prediction_buffer if p == smoothed_pred][0]
        avg_confidence = np.mean([c for p, c, l in self.prediction_buffer if p == smoothed_pred])
        return smoothed_pred, avg_confidence, smoothed_label, top5
    
    def calculate_fps(self) -> float:
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)
    
    def put_tamil_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                       font, color: Tuple[int, int, int] = (255, 255, 255)):
        """Render Tamil Unicode text on frame using PIL."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=font, fill=color)
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def draw_info(self, frame: np.ndarray, prediction: str, confidence: float, label_id: int, fps: float, num_hands: int, top5_preds: List = None):
        height, width = frame.shape[:2]
        
        # Create professional side panel
        overlay = frame.copy()
        panel_width = 380
        cv2.rectangle(overlay, (0, 0), (panel_width, height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Header
        cv2.putText(frame, "Tamil Sign Language", (15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Recognition System", (15, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
        
        cv2.line(frame, (15, 90), (panel_width - 15, 90), (100, 100, 100), 2)
        
        y_pos = 130
        
        # Status
        if self.is_paused:
            cv2.rectangle(frame, (15, y_pos-5), (180, y_pos+25), (0, 165, 255), -1)
            cv2.putText(frame, "PAUSED", (25, y_pos+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
        
        # Hands detected
        hand_color = (0, 255, 0) if num_hands > 0 else (100, 100, 100)
        cv2.putText(frame, f"Hands Detected: {num_hands}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        y_pos += 50
        
        if prediction and confidence > 0.0:
            # Section header
            cv2.putText(frame, "Predicted Gesture:", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
            y_pos += 40
            
            # LARGE Tamil character
            frame = self.put_tamil_text(frame, prediction, (20, y_pos), 
                                       self.tamil_font, (0, 255, 0))
            y_pos += 80
            
            # Class ID with background
            cv2.rectangle(frame, (15, y_pos-5), (200, y_pos+25), (50, 50, 50), -1)
            cv2.putText(frame, f"Class ID: {label_id}", (20, y_pos+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_pos += 45
            
            # Confidence bar
            conf_color = (0, 255, 0) if confidence > 0.75 else (0, 255, 255) if confidence > 0.5 else (0, 165, 255)
            cv2.putText(frame, "Confidence:", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            y_pos += 25
            
            # Progress bar
            bar_width = int((panel_width - 40) * confidence)
            cv2.rectangle(frame, (15, y_pos), (panel_width - 15, y_pos + 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, y_pos), (15 + bar_width, y_pos + 20), conf_color, -1)
            cv2.putText(frame, f"{confidence:.1%}", (20, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 40
            
            # Stability
            if len(self.prediction_buffer) >= self.prediction_smoothing:
                predictions = [p for p, c, l in self.prediction_buffer]
                stability = predictions.count(prediction) / len(predictions)
                if stability >= 0.7:
                    cv2.rectangle(frame, (15, y_pos), (165, y_pos + 28), (0, 200, 0), -1)
                    cv2.putText(frame, "✓ STABLE", (25, y_pos + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 38
            
            # Top 5 Predictions
            if top5_preds:
                y_pos += 15
                cv2.line(frame, (15, y_pos), (panel_width - 15, y_pos), (100, 100, 100), 1)
                y_pos += 20
                
                cv2.putText(frame, "Top 5 Predictions:", (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                y_pos += 30
                
                for i, (pred_char, pred_conf, pred_id) in enumerate(top5_preds):
                    rank_color = (0, 255, 0) if i == 0 else (200, 200, 200) if i == 1 else (150, 150, 150)
                    frame = self.put_tamil_text(frame, f"{i+1}. {pred_char}  ({pred_conf:.1%})", 
                                               (20, y_pos), self.small_tamil_font, rank_color)
                    y_pos += 35
                    if y_pos > height - 120:
                        break
        else:
            cv2.rectangle(frame, (15, y_pos-5), (panel_width-15, y_pos+30), (50, 50, 100), -1)
            cv2.putText(frame, "No hand detected", (25, y_pos+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Footer
        cv2.line(frame, (15, height - 80), (panel_width - 15, height - 80), (100, 100, 100), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "SPACE: pause  's': save  'q': quit", (15, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        return frame
    
    def save_frame(self, frame: np.ndarray, prediction: str, output_dir: str = "output"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{prediction}_{timestamp}.jpg"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), frame)
        print(f"\n✓ Frame saved: {filepath}")
    
    def run(self, output_dir: Optional[str] = None):
        print("\n" + "=" * 70)
        print("MODULE 6: TAMIL SIGN LANGUAGE RECOGNITION")
        print("=" * 70)
        
        self.load_model()
        self.initialize_mediapipe()
        self.initialize_camera()
        
        print("\n" + "=" * 70)
        print("CONTROLS:")
        print("  SPACE - Pause/unpause")
        print("  's'   - Save frame")
        print("  'q'   - Quit")
        print("=" * 70)
        print("\nStarting...\n")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.stats['total_frames'] += 1
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                prediction = None
                confidence = 0.0
                label_id = None
                num_hands = 0
                top5_predictions = None
                
                if not self.is_paused:
                    results = self.hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks:
                        num_hands = len(results.multi_hand_landmarks)
                        self.stats['hands_detected'] += 1
                        
                        # Draw all hands
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
                        
                        # Predict using first hand
                        features = self.extract_landmarks(results.multi_hand_landmarks)
                        prediction, confidence, label_id, top5_predictions = self.predict_sign(features)
                        prediction, confidence, label_id, top5_predictions = self.smooth_prediction(prediction, confidence, label_id, top5_predictions)
                        
                        self.stats['predictions_made'] += 1
                        self.current_prediction = prediction
                        self.current_confidence = confidence
                        self.current_label_id = label_id
                        
                        # Update history
                        if confidence > 0.3 and prediction != self.last_printed_prediction:
                            self.prediction_history.append((prediction, confidence, label_id))
                            if len(self.prediction_history) > 10:
                                self.prediction_history.pop(0)
                            print(f"📌 {prediction} (Class {label_id}) - {confidence:.1%} confidence - {num_hands} hand(s)")
                            self.last_printed_prediction = prediction
                    else:
                        self.prediction_buffer.clear()
                        self.current_prediction = None
                        self.current_confidence = 0.0
                        self.current_label_id = None
                        self.last_printed_prediction = None
                else:
                    prediction = self.current_prediction
                    confidence = self.current_confidence
                    label_id = self.current_label_id
                
                fps = self.calculate_fps()
                frame = self.draw_info(frame, self.current_prediction, self.current_confidence, 
                                      self.current_label_id if self.current_label_id is not None else -1, 
                                      fps, num_hands, top5_predictions)
                
                cv2.imshow('Tamil Sign Language Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord(' '):
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        print(f"\n⏸ PAUSED - {self.current_prediction} (Class {self.current_label_id}) - {self.current_confidence:.1%}")
                    else:
                        print("\n▶ RESUMED")
                elif key == ord('s') and self.current_prediction:
                    if output_dir:
                        self.save_frame(frame, self.current_prediction, output_dir)
                        
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.cleanup()
            self.print_statistics()
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        cv2.destroyAllWindows()
    
    def print_statistics(self):
        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Hands Detected: {self.stats['hands_detected']}")
        print(f"Predictions Made: {self.stats['predictions_made']}")
        if self.stats['total_frames'] > 0:
            detection_rate = (self.stats['hands_detected'] / self.stats['total_frames']) * 100
            print(f"Detection Rate: {detection_rate:.2f}%")
        print("=" * 70)


def load_and_predict_image(model_path: str, image_path: str) -> Tuple[str, float]:
    """Predict from a single image."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_mapping = model_data['label_mapping']
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        hands.close()
        return None, 0.0
    
    features = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    predicted_label = model.predict(features_scaled)[0]
    predicted_proba = model.predict_proba(features_scaled)[0]
    confidence = np.max(predicted_proba)
    predicted_char = label_mapping.get(predicted_label, "Unknown")
    
    hands.close()
    return predicted_char, confidence
