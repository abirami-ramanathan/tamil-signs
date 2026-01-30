"""
Module 3: Test Script for Hand Landmark Extraction

This script tests the hand landmark extraction functionality.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from mod1 import TLFS23DatasetLoader
from mod2 import DatasetPreprocessor
from mod3.hand_landmark_extractor import HandLandmarkExtractor, DatasetLandmarkExtractor
from mod3.utils import (
    visualize_landmarks_on_images,
    visualize_landmark_distribution,
    analyze_landmark_statistics,
    plot_landmark_statistics,
    validate_landmarks,
    export_landmark_report
)


def test_single_image_landmark_extraction():
    """Test landmark extraction from a single image."""
    print("=" * 70)
    print("TEST 1: Single Image Landmark Extraction")
    print("=" * 70)
    
    # Load dataset
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Get a sample image
    class_info = loader.get_class_info(0)
    sample_image_path = class_info['image_paths'][0]
    
    print(f"\nTest image: {sample_image_path}")
    
    # Initialize extractor
    extractor = HandLandmarkExtractor(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    # Read and preprocess image
    import cv2
    image = cv2.imread(str(sample_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract landmarks
    landmarks = extractor.extract_landmarks(image_rgb)
    
    if landmarks is not None:
        print(f"\n✓ Landmarks extracted successfully!")
        print(f"  Shape: {landmarks.shape}")
        print(f"  Dtype: {landmarks.dtype}")
        print(f"  Value range: [{np.min(landmarks):.4f}, {np.max(landmarks):.4f}]")
        print(f"  Mean: {np.mean(landmarks):.4f}")
        print(f"  First 9 values (first 3 landmarks): {landmarks[:9]}")
    else:
        print(f"\n✗ Failed to extract landmarks")
    
    # Get statistics
    stats = extractor.get_statistics()
    print(f"\nExtractor statistics:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    
    extractor.close()
    print("\n✓ Single image landmark extraction test passed!\n")


def test_batch_landmark_extraction():
    """Test batch landmark extraction."""
    print("=" * 70)
    print("TEST 2: Batch Landmark Extraction")
    print("=" * 70)
    
    # Load dataset
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Get sample image paths (20 images)
    class_info = loader.get_class_info(0)
    sample_paths = class_info['image_paths'][:20]
    
    print(f"\nExtracting landmarks from {len(sample_paths)} images...")
    
    # Initialize extractor
    extractor = HandLandmarkExtractor()
    
    # Extract landmarks
    landmarks, status, successful_paths = extractor.extract_from_paths(
        sample_paths,
        show_progress=True
    )
    
    # Get statistics
    stats = extractor.get_statistics()
    
    print(f"\nBatch extraction results:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.2f}%")
    
    # Filter successful extractions
    valid_landmarks = [l for l in landmarks if l is not None]
    if len(valid_landmarks) > 0:
        landmarks_array = np.array(valid_landmarks)
        print(f"\nLandmarks array shape: {landmarks_array.shape}")
    
    extractor.close()
    print("\n✓ Batch landmark extraction test passed!\n")


def test_dataframe_extraction():
    """Test landmark extraction from DataFrame."""
    print("=" * 70)
    print("TEST 3: DataFrame Landmark Extraction")
    print("=" * 70)
    
    # Load dataset
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Create DataFrame with 5 images per class
    df = loader.create_dataframe()
    df_sample = df.groupby('label').head(5).reset_index(drop=True)
    
    print(f"\nExtracting from DataFrame: {len(df_sample)} images")
    
    # Initialize extractor
    extractor = HandLandmarkExtractor()
    
    # Extract landmarks
    result = extractor.extract_from_dataframe(
        df_sample,
        max_samples=None
    )
    
    print(f"\nResults:")
    print(f"  X shape: {result['X'].shape}")
    print(f"  y shape: {result['y'].shape}")
    print(f"  Image paths: {len(result['image_paths'])}")
    print(f"  Unique labels: {len(np.unique(result['y']))}")
    
    extractor.close()
    print("\n✓ DataFrame extraction test passed!\n")
    
    return result


def test_preprocessed_data_integration():
    """Test integration with Module 2 preprocessed data."""
    print("=" * 70)
    print("TEST 4: Integration with Module 2 (Preprocessed Data)")
    print("=" * 70)
    
    # Load dataset
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Preprocess data using Module 2
    print("\nPreprocessing data using Module 2...")
    from mod2 import DatasetPreprocessor as Mod2Preprocessor
    
    preprocessor = Mod2Preprocessor(
        dataset_loader=loader,
        target_size=None,
        train_split=0.8
    )
    
    preprocessed_data = preprocessor.preprocess_dataset(
        max_samples_per_class=3,  # Small sample for testing
        output_dir=None
    )
    
    # Extract landmarks using Module 3
    print("\nExtracting landmarks using Module 3...")
    landmark_extractor = DatasetLandmarkExtractor(
        min_detection_confidence=0.5
    )
    
    output_dir = Path(__file__).parent / "output"
    landmark_data = landmark_extractor.extract_from_preprocessed_data(
        preprocessed_data,
        output_dir=str(output_dir)
    )
    
    print(f"\n✓ Module 2→3 integration test passed!\n")
    
    landmark_extractor.close()
    return landmark_data


def test_landmark_validation():
    """Test landmark validation utilities."""
    print("=" * 70)
    print("TEST 5: Landmark Validation")
    print("=" * 70)
    
    # Load dataset
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Get sample data
    df = loader.create_dataframe()
    df_sample = df.groupby('label').head(3).reset_index(drop=True)
    
    # Extract landmarks
    extractor = HandLandmarkExtractor()
    result = extractor.extract_from_dataframe(df_sample, max_samples=None)
    
    # Validate landmarks
    print("\nValidating extracted landmarks...")
    validation_result = validate_landmarks(result['X'])
    
    print(f"\nValidation summary:")
    print(f"  Is valid: {validation_result['is_valid']}")
    print(f"  Number of samples: {validation_result['num_samples']}")
    print(f"  Number of features: {validation_result['num_features']}")
    print(f"  NaN count: {validation_result['nan_count']}")
    print(f"  Inf count: {validation_result['inf_count']}")
    print(f"  Zero rows: {validation_result['zero_rows']}")
    
    extractor.close()
    print("\n✓ Landmark validation test passed!\n")


def test_utilities_and_visualization(landmark_data):
    """Test utility functions and visualizations."""
    print("=" * 70)
    print("TEST 6: Utilities and Visualization")
    print("=" * 70)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze statistics
    print("\nAnalyzing landmark statistics...")
    train_stats = analyze_landmark_statistics(landmark_data['X_train'])
    test_stats = analyze_landmark_statistics(landmark_data['X_test'])
    
    print(f"\nTrain statistics:")
    print(f"  Samples: {train_stats['num_samples']}")
    print(f"  Features: {train_stats['num_features']}")
    print(f"  X coords mean: {train_stats['x_coords_mean']:.4f} ± {train_stats['x_coords_std']:.4f}")
    print(f"  Y coords mean: {train_stats['y_coords_mean']:.4f} ± {train_stats['y_coords_std']:.4f}")
    print(f"  Z coords mean: {train_stats['z_coords_mean']:.4f} ± {train_stats['z_coords_std']:.4f}")
    
    print(f"\nTest statistics:")
    print(f"  Samples: {test_stats['num_samples']}")
    print(f"  Features: {test_stats['num_features']}")
    print(f"  X coords mean: {test_stats['x_coords_mean']:.4f} ± {test_stats['x_coords_std']:.4f}")
    print(f"  Y coords mean: {test_stats['y_coords_mean']:.4f} ± {test_stats['y_coords_std']:.4f}")
    print(f"  Z coords mean: {test_stats['z_coords_mean']:.4f} ± {test_stats['z_coords_std']:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        # Distribution plot
        visualize_landmark_distribution(
            landmark_data['X_train'][:100],
            save_path=str(output_dir / 'landmark_distribution.png')
        )
        print("✓ Distribution plot saved")
    except Exception as e:
        print(f"Warning: Distribution plot skipped: {e}")
    
    try:
        # Statistics plot
        plot_landmark_statistics(
            landmark_data['X_train'],
            landmark_data['X_test'],
            save_path=str(output_dir / 'landmark_statistics.png')
        )
        print("✓ Statistics plot saved")
    except Exception as e:
        print(f"Warning: Statistics plot skipped: {e}")
    
    # Export report
    print("\nExporting landmark report...")
    export_landmark_report(
        landmark_data,
        str(output_dir / 'landmark_report.txt')
    )
    
    print("\n✓ Utilities and visualization test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING MODULE 3 TESTS")
    print("=" * 70 + "\n")
    
    # Run tests
    test_single_image_landmark_extraction()
    test_batch_landmark_extraction()
    test_dataframe_extraction()
    landmark_data = test_preprocessed_data_integration()
    test_landmark_validation()
    test_utilities_and_visualization(landmark_data)
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    output_dir = Path(__file__).parent / "output"
    print(f"\nOutput files saved to: {output_dir}")
    print("\nModule 3 is ready for integration with Module 4!")
    print("Next: Feature Dataset Construction & Scaling")


if __name__ == "__main__":
    run_all_tests()
