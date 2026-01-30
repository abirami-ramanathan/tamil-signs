"""
Module 2: Quick Start Examples for Image Preprocessing

This script demonstrates common use cases for Module 2.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from mod1 import TLFS23DatasetLoader
from mod2 import (
    ImagePreprocessor,
    DatasetPreprocessor,
    visualize_preprocessed_images,
    analyze_image_statistics,
    export_preprocessing_report
)


def example_1_single_image():
    """Example 1: Preprocess a single image."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Image Preprocessing")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Get a sample image using Module 1
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    class_info = loader.get_class_info(0)
    image_path = class_info['image_paths'][0]
    
    print(f"\nOriginal image: {image_path}")
    
    # Preprocess
    preprocessor = ImagePreprocessor(
        target_size=None,  # Keep original size
        normalize=False
    )
    
    preprocessed = preprocessor.preprocess_image(image_path)
    
    print(f"\nPreprocessed image:")
    print(f"  Shape: {preprocessed.shape}")
    print(f"  Dtype: {preprocessed.dtype}")
    print(f"  Value range: [{np.min(preprocessed)}, {np.max(preprocessed)}]")


def example_2_batch_processing():
    """Example 2: Batch processing multiple images."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Get sample images
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Get first 50 images from class 0
    class_info = loader.get_class_info(0)
    image_paths = class_info['image_paths'][:50]
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    # Batch preprocess
    preprocessor = ImagePreprocessor()
    preprocessed, successful = preprocessor.preprocess_batch(
        image_paths,
        show_progress=True
    )
    
    # Get statistics
    stats = preprocessor.get_statistics()
    print(f"\nResults:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.2f}%")


def example_3_resize_and_normalize():
    """Example 3: Resize and normalize images."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Resize and Normalize")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Get a sample image
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    class_info = loader.get_class_info(0)
    image_path = class_info['image_paths'][0]
    
    # Preprocess with resizing and normalization
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),  # Resize to 224x224
        normalize=True           # Normalize to [0, 1]
    )
    
    preprocessed = preprocessor.preprocess_image(image_path)
    
    print(f"\nResized and normalized image:")
    print(f"  Shape: {preprocessed.shape}")
    print(f"  Dtype: {preprocessed.dtype}")
    print(f"  Value range: [{np.min(preprocessed):.3f}, {np.max(preprocessed):.3f}]")
    print(f"  Mean: {np.mean(preprocessed):.3f}")


def example_4_dataframe_preprocessing():
    """Example 4: Preprocess from DataFrame."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: DataFrame Preprocessing")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Load dataset and create DataFrame
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    df = loader.create_dataframe()
    
    print(f"\nDataFrame info:")
    print(f"  Total images: {len(df)}")
    print(f"  Unique labels: {df['label'].nunique()}")
    
    # Sample 10 images per class
    df_sample = df.groupby('label').head(10).reset_index(drop=True)
    print(f"  Sampled images: {len(df_sample)}")
    
    # Preprocess
    print("\nPreprocessing...")
    preprocessor = ImagePreprocessor()
    result = preprocessor.preprocess_from_dataframe(df_sample)
    
    print(f"\nResults:")
    print(f"  X shape: {result['X'].shape}")
    print(f"  y shape: {result['y'].shape}")
    print(f"  Unique labels in y: {len(np.unique(result['y']))}")


def example_5_train_test_split():
    """Example 5: Full dataset preprocessing with train/test split."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Train/Test Split")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Load dataset
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Initialize dataset preprocessor
    dataset_preprocessor = DatasetPreprocessor(
        dataset_loader=loader,
        target_size=None,
        train_split=0.8,
        random_state=42
    )
    
    # Preprocess with limited samples for demonstration
    print("\nPreprocessing dataset (5 images per class)...")
    result = dataset_preprocessor.preprocess_dataset(
        max_samples_per_class=5,
        output_dir=None
    )
    
    print(f"\nTrain set:")
    print(f"  X_train: {result['X_train'].shape}")
    print(f"  y_train: {result['y_train'].shape}")
    print(f"  Unique labels: {len(np.unique(result['y_train']))}")
    
    print(f"\nTest set:")
    print(f"  X_test: {result['X_test'].shape}")
    print(f"  y_test: {result['y_test'].shape}")
    print(f"  Unique labels: {len(np.unique(result['y_test']))}")


def example_6_save_and_load():
    """Example 6: Save and load preprocessed data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Save and Load Preprocessed Data")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    output_dir = Path(__file__).parent / "output" / "quick_start"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    dataset_preprocessor = DatasetPreprocessor(
        dataset_loader=loader,
        target_size=None
    )
    
    print("\nPreprocessing and saving...")
    result = dataset_preprocessor.preprocess_dataset(
        max_samples_per_class=3,
        output_dir=str(output_dir)
    )
    
    print(f"\n✓ Data saved to: {output_dir}")
    print(f"  Files: preprocessed_data.pkl, train_data.pkl, test_data.pkl")
    
    # Load back
    print("\nLoading preprocessed data...")
    preprocessor = ImagePreprocessor()
    loaded = preprocessor.load_preprocessed_data(
        str(output_dir / 'preprocessed_data.pkl')
    )
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  X_train shape: {loaded['X_train'].shape}")
    print(f"  y_train shape: {loaded['y_train'].shape}")


def example_7_analysis_and_reporting():
    """Example 7: Analyze and generate reports."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Analysis and Reporting")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    output_dir = Path(__file__).parent / "output" / "quick_start"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess small sample
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    dataset_preprocessor = DatasetPreprocessor(
        dataset_loader=loader,
        target_size=None
    )
    
    print("\nPreprocessing data...")
    result = dataset_preprocessor.preprocess_dataset(
        max_samples_per_class=5,
        output_dir=None
    )
    
    # Analyze statistics
    print("\nAnalyzing statistics...")
    train_stats = analyze_image_statistics(result['X_train'])
    test_stats = analyze_image_statistics(result['X_test'])
    
    print(f"\nTrain statistics:")
    print(f"  Images: {train_stats['num_images']}")
    print(f"  Shape: {train_stats['shape']}")
    print(f"  Memory: {train_stats['memory_size_mb']:.2f} MB")
    
    print(f"\nTest statistics:")
    print(f"  Images: {test_stats['num_images']}")
    print(f"  Shape: {test_stats['shape']}")
    print(f"  Memory: {test_stats['memory_size_mb']:.2f} MB")
    
    # Export report
    report_path = output_dir / 'preprocessing_report.txt'
    export_preprocessing_report(result, str(report_path))
    print(f"\n✓ Report exported to: {report_path}")


def example_8_visualization():
    """Example 8: Visualize preprocessed images."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Visualization")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    output_dir = Path(__file__).parent / "output" / "quick_start"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess sample
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    dataset_preprocessor = DatasetPreprocessor(
        dataset_loader=loader,
        target_size=None
    )
    
    print("\nPreprocessing data...")
    result = dataset_preprocessor.preprocess_dataset(
        max_samples_per_class=5,
        output_dir=None
    )
    
    # Visualize
    print("\nGenerating visualizations...")
    try:
        visualize_preprocessed_images(
            result['X_train'][:20],
            result['y_train'][:20],
            result['label_mapping'],
            num_samples=20,
            save_path=str(output_dir / 'sample_visualization.png')
        )
        print(f"✓ Visualization saved to: {output_dir / 'sample_visualization.png'}")
    except Exception as e:
        print(f"Warning: Visualization skipped: {e}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MODULE 2: QUICK START EXAMPLES")
    print("=" * 70)
    
    examples = [
        example_1_single_image,
        example_2_batch_processing,
        example_3_resize_and_normalize,
        example_4_dataframe_preprocessing,
        example_5_train_test_split,
        example_6_save_and_load,
        example_7_analysis_and_reporting,
        example_8_visualization,
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n❌ Example {i} failed: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nNext: Explore Module 3 for hand landmark extraction!")


if __name__ == "__main__":
    run_all_examples()
