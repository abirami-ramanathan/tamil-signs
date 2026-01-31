"""
Module 6: Test Script for Real-Time Prediction

This script tests the real-time Tamil sign language prediction system.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from mod6 import RealtimePredictor


def test_webcam_prediction():
    """Test real-time prediction with webcam."""
    print("=" * 70)
    print("TEST: WEBCAM REAL-TIME PREDICTION")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_path = Path(__file__).parent.parent / "mod5" / "output" / "random_forest_model.pkl"
    output_dir = Path(__file__).parent / "output"
    
    # Check if model exists
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("  Please run Module 5 tests first to train the model!")
        return
    
    print(f"\nModel path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize predictor
    predictor = RealtimePredictor(
        model_path=str(model_path),
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        prediction_smoothing=5,
        camera_index=0,
        frame_width=1280,
        frame_height=720
    )
    
    # Run real-time prediction
    print("\n" + "=" * 70)
    print("Starting webcam prediction...")
    print("=" * 70)
    
    try:
        predictor.run(output_dir=str(output_dir))
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Webcam prediction test completed!")


def test_image_prediction():
    """Test prediction on a single image."""
    print("\n" + "=" * 70)
    print("TEST: SINGLE IMAGE PREDICTION")
    print("=" * 70)
    
    from mod6.realtime_predictor import load_and_predict_image
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_path = Path(__file__).parent.parent / "mod5" / "output" / "random_forest_model.pkl"
    
    # Find a test image from the dataset
    dataset_path = project_root / "TLFS23 - Tamil Language Finger Spelling Image Dataset" / "Dataset Folders"
    
    if not dataset_path.exists():
        print(f"\n✗ Dataset not found: {dataset_path}")
        return
    
    # Get first class folder
    class_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    if not class_folders:
        print("\n✗ No class folders found in dataset")
        return
    
    # Get first image from first class
    test_class = class_folders[0]
    images = list(test_class.glob("*.jpg")) + list(test_class.glob("*.png"))
    
    if not images:
        print(f"\n✗ No images found in {test_class}")
        return
    
    test_image = images[0]
    
    print(f"\nTest image: {test_image}")
    print(f"Model: {model_path}")
    
    # Predict
    print("\nPredicting...")
    try:
        prediction, confidence = load_and_predict_image(str(model_path), str(test_image))
        
        if prediction:
            print(f"\n✓ Prediction successful!")
            print(f"  Character: {prediction}")
            print(f"  Confidence: {confidence:.2%}")
        else:
            print("\n✗ No hand detected in image")
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("=" * 70)
    print("MODULE 6: REAL-TIME PREDICTION TESTS")
    print("=" * 70)
    
    # Test 1: Single image prediction
    test_image_prediction()
    
    # Test 2: Webcam real-time prediction
    print("\n" + "=" * 70)
    print("Would you like to test webcam prediction? (y/n)")
    print("=" * 70)
    
    response = input("Enter choice: ").strip().lower()
    
    if response == 'y':
        test_webcam_prediction()
    else:
        print("\nSkipping webcam test.")
        print("\nTo run webcam test manually, use:")
        print("  python test_module6.py --webcam")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Module 6: Real-Time Prediction")
    parser.add_argument('--webcam', action='store_true', help='Run webcam test directly')
    parser.add_argument('--image', action='store_true', help='Run image prediction test only')
    
    args = parser.parse_args()
    
    if args.webcam:
        test_webcam_prediction()
    elif args.image:
        test_image_prediction()
    else:
        main()
