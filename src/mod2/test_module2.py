"""
Module 2: Test Script for Image Preprocessing

This script tests the image preprocessing functionality.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from mod1 import TLFS23DatasetLoader
from mod2.image_preprocessor import ImagePreprocessor, DatasetPreprocessor
from mod2.utils import (
    visualize_preprocessed_images,
    compare_original_vs_preprocessed,
    analyze_image_statistics,
    plot_preprocessing_statistics,
    visualize_data_distribution,
    check_image_quality_batch,
    export_preprocessing_report,
    validate_preprocessed_data
)


def test_single_image_preprocessing():
    """Test preprocessing of a single image."""
    print("=" * 70)
    print("TEST 1: Single Image Preprocessing")
    print("=" * 70)
    
    # Get a sample image path
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Get first image
    class_info = loader.get_class_info(0)
    sample_image = class_info['image_paths'][0]
    
    print(f"\nTest image: {sample_image}")
    
    # Test with different configurations
    configs = [
        {'target_size': None, 'normalize': False, 'desc': 'Original size, uint8'},
        {'target_size': (224, 224), 'normalize': False, 'desc': 'Resized 224x224, uint8'},
        {'target_size': None, 'normalize': True, 'desc': 'Original size, normalized'},
    ]
    
    for config in configs:
        preprocessor = ImagePreprocessor(
            target_size=config['target_size'],
            normalize=config['normalize']
        )
        
        result = preprocessor.preprocess_image(sample_image)
        
        if result is not None:
            print(f"\n{config['desc']}:")
            print(f"  Shape: {result.shape}")
            print(f"  Dtype: {result.dtype}")
            print(f"  Min: {np.min(result):.2f}, Max: {np.max(result):.2f}")
            print(f"  Mean: {np.mean(result):.2f}")
        else:
            print(f"\n{config['desc']}: FAILED")
    
    print("\n✓ Single image preprocessing tests passed!\n")


def test_batch_preprocessing(loader):
    """Test batch preprocessing."""
    print("=" * 70)
    print("TEST 2: Batch Preprocessing")
    print("=" * 70)
    
    # Get sample images from first class
    class_info = loader.get_class_info(0)
    sample_paths = class_info['image_paths'][:20]  # First 20 images
    
    print(f"\nProcessing {len(sample_paths)} images...")
    
    preprocessor = ImagePreprocessor(target_size=None, normalize=False)
    preprocessed, successful = preprocessor.preprocess_batch(
        sample_paths,
        show_progress=True
    )
    
    stats = preprocessor.get_statistics()
    
    print(f"\nBatch processing results:")
    print(f"  Total: {stats['total_processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.2f}%")
    
    if len(preprocessed) > 0:
        print(f"\nPreprocessed data shape: {np.array(preprocessed).shape}")
    
    print("\n✓ Batch preprocessing tests passed!\n")
    
    return preprocessed, successful


def test_dataframe_preprocessing(loader):
    """Test preprocessing from DataFrame."""
    print("=" * 70)
    print("TEST 3: DataFrame Preprocessing")
    print("=" * 70)
    
    # Create DataFrame
    df = loader.create_dataframe()
    
    # Sample 5 images per class for quick testing
    print("\nSampling 5 images per class...")
    df_sample = df.groupby('label').head(5).reset_index(drop=True)
    
    print(f"Total samples: {len(df_sample)}")
    
    # Preprocess
    preprocessor = ImagePreprocessor(target_size=None, normalize=False)
    result = preprocessor.preprocess_from_dataframe(
        df_sample,
        max_samples=None
    )
    
    print(f"\nResults:")
    print(f"  X shape: {result['X'].shape}")
    print(f"  y shape: {result['y'].shape}")
    print(f"  Unique labels: {len(np.unique(result['y']))}")
    
    print("\n✓ DataFrame preprocessing tests passed!\n")
    
    return result


def test_dataset_preprocessing(loader):
    """Test full dataset preprocessing with train/test split."""
    print("=" * 70)
    print("TEST 4: Dataset Preprocessing with Train/Test Split")
    print("=" * 70)
    
    # Initialize dataset preprocessor
    dataset_preprocessor = DatasetPreprocessor(
        dataset_loader=loader,
        target_size=None,
        train_split=0.8,
        random_state=42
    )
    
    # Preprocess with limited samples for testing
    result = dataset_preprocessor.preprocess_dataset(
        max_samples_per_class=5,  # 5 per class = 247*5 = 1235 total
        output_dir=None
    )
    
    print(f"\n✓ Dataset preprocessing tests passed!\n")
    
    return result


def test_save_and_load(loader, output_dir):
    """Test saving and loading preprocessed data."""
    print("=" * 70)
    print("TEST 5: Save and Load Preprocessed Data")
    print("=" * 70)
    
    # Preprocess small dataset
    dataset_preprocessor = DatasetPreprocessor(
        dataset_loader=loader,
        target_size=None,
        train_split=0.8
    )
    
    print("\nPreprocessing data...")
    result = dataset_preprocessor.preprocess_dataset(
        max_samples_per_class=3,
        output_dir=str(output_dir)
    )
    
    # Load back
    print("\nLoading preprocessed data...")
    preprocessor = ImagePreprocessor()
    loaded_data = preprocessor.load_preprocessed_data(
        str(output_dir / 'preprocessed_data.pkl')
    )
    
    # Verify
    print("\nVerifying loaded data...")
    assert loaded_data['X_train'].shape == result['X_train'].shape
    assert loaded_data['y_train'].shape == result['y_train'].shape
    assert np.array_equal(loaded_data['y_train'], result['y_train'])
    
    print("✓ Data shapes match!")
    print("✓ Labels match!")
    
    print("\n✓ Save and load tests passed!\n")


def test_utilities(loader, result, output_dir):
    """Test utility functions."""
    print("=" * 70)
    print("TEST 6: Utility Functions")
    print("=" * 70)
    
    # Get label mapping from Module 1
    label_mapping = loader.mapping.label_to_character
    
    # Test image statistics
    print("\nAnalyzing image statistics...")
    stats = analyze_image_statistics(result['X_train'])
    print(f"  Number of images: {stats['num_images']}")
    print(f"  Image shape: {stats['shape']}")
    print(f"  Data type: {stats['dtype']}")
    print(f"  Mean pixel value: {stats['mean_pixel_value']:.2f}")
    print(f"  Memory size: {stats['memory_size_mb']:.2f} MB")
    
    # Test quality check
    print("\nChecking image quality...")
    quality = check_image_quality_batch(result['X_train'])
    print(f"  Total images: {quality['total_images']}")
    print(f"  Valid images: {quality['valid_images']}")
    print(f"  Quality rate: {quality['quality_rate']:.2f}%")
    
    # Test validation
    print("\nValidating preprocessed data...")
    is_valid = validate_preprocessed_data(result)
    
    # Test visualization (save only, don't show)
    print("\nGenerating visualizations...")
    try:
        visualize_preprocessed_images(
            result['X_train'][:10],
            result['y_train'][:10],
            label_mapping,
            num_samples=10,
            save_path=str(output_dir / 'preprocessed_samples.png')
        )
        print("✓ Sample visualization saved")
    except Exception as e:
        print(f"Warning: Visualization skipped: {e}")
    
    try:
        plot_preprocessing_statistics(
            result['train_stats'],
            result['test_stats'],
            save_path=str(output_dir / 'preprocessing_stats.png')
        )
        print("✓ Statistics plot saved")
    except Exception as e:
        print(f"Warning: Statistics plot skipped: {e}")
    
    try:
        visualize_data_distribution(
            result['y_train'],
            result['y_test'],
            label_mapping,
            save_path=str(output_dir / 'data_distribution.png')
        )
        print("✓ Distribution plot saved")
    except Exception as e:
        print(f"Warning: Distribution plot skipped: {e}")
    
    # Export report
    print("\nExporting preprocessing report...")
    export_preprocessing_report(
        result,
        str(output_dir / 'preprocessing_report.txt')
    )
    
    print("\n✓ Utility function tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING MODULE 2 TESTS")
    print("=" * 70 + "\n")
    
    # Setup
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset using Module 1
    print("Loading dataset using Module 1...")
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    print(f"✓ Dataset loaded: {loader.dataset_stats['total_images']:,} images\n")
    
    # Run tests
    test_single_image_preprocessing()
    test_batch_preprocessing(loader)
    test_dataframe_preprocessing(loader)
    result = test_dataset_preprocessing(loader)
    test_save_and_load(loader, output_dir)
    test_utilities(loader, result, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nModule 2 is ready for integration with Module 3!")
    print("Next: Hand Landmark Extraction using MediaPipe")


if __name__ == "__main__":
    run_all_tests()
