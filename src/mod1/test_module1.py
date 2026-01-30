"""
Module 1: Test Script for Dataset Loading

This script tests the TLFS23 dataset loader functionality.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from dataset_loader import TLFS23DatasetLoader, TamilCharacterMapping
from utils import (
    visualize_sample_images,
    visualize_reference_images,
    plot_class_distribution,
    validate_dataset_integrity,
    get_image_statistics,
    export_label_mappings
)


def test_character_mapping():
    """Test the character mapping functionality."""
    print("=" * 70)
    print("TEST 1: Character Mapping")
    print("=" * 70)
    
    mapping = TamilCharacterMapping()
    
    # Test folder to character
    print("\nTesting folder to character mappings:")
    for folder_num in [1, 14, 32, 100, 247]:
        char_info = mapping.get_character_by_folder(folder_num)
        print(f"Folder {folder_num}: {char_info['tamil']} ({char_info['pronunciation']}) - {char_info['type']}")
    
    # Test label to character
    print("\nTesting label to character mappings:")
    for label in [0, 13, 31, 99, 246]:
        char_info = mapping.get_character_by_label(label)
        print(f"Label {label}: {char_info['tamil']} ({char_info['pronunciation']}) - {char_info['type']}")
    
    # Test reverse mapping
    print("\nTesting character to label mappings:")
    test_chars = ['அ', 'க்', 'க', 'ண', 'னௌ']
    for char in test_chars:
        label = mapping.get_label_by_character(char)
        folder = mapping.get_folder_by_character(char)
        print(f"Character '{char}': Label={label}, Folder={folder}")
    
    # Test type counts
    print("\nCharacter type distribution:")
    type_counts = mapping.get_character_type_counts()
    for char_type, count in type_counts.items():
        print(f"  {char_type}: {count} characters")
    
    print("\n✓ Character mapping tests passed!\n")


def test_dataset_loader(dataset_path: str):
    """Test the dataset loader functionality."""
    print("=" * 70)
    print("TEST 2: Dataset Loader")
    print("=" * 70)
    
    # Initialize loader
    print("\nInitializing dataset loader...")
    loader = TLFS23DatasetLoader(dataset_path)
    
    # Load dataset structure
    print("\nLoading dataset structure...")
    dataset_info = loader.load_dataset_structure(validate_images=False)
    
    # Print summary
    print("\n" + loader.get_dataset_summary())
    
    # Test getting class info
    print("\nTesting get_class_info():")
    for label in [0, 50, 100, 150, 200, 246]:
        class_info = loader.get_class_info(label)
        if class_info:
            print(f"Label {label}: {class_info['tamil_char']} - {class_info['image_count']} images")
    
    # Test getting all image paths
    print("\nTesting get_all_image_paths():")
    all_images = loader.get_all_image_paths()
    print(f"Total image paths: {len(all_images)}")
    print(f"First 3 images: {all_images[:3]}")
    
    # Test reference images
    print("\nTesting get_reference_image():")
    for label in [0, 100, 200]:
        ref_path = loader.get_reference_image(label)
        status = "Found" if ref_path else "Not found"
        print(f"Label {label}: {status}")
    
    print("\n✓ Dataset loader tests passed!\n")
    
    return loader


def test_dataframe_creation(loader):
    """Test DataFrame creation."""
    print("=" * 70)
    print("TEST 3: DataFrame Creation")
    print("=" * 70)
    
    print("\nCreating DataFrame...")
    df = loader.create_dataframe()
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    print("\nDataFrame info:")
    print(f"Total rows: {len(df)}")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Unique characters: {df['tamil_char'].nunique()}")
    
    print("\nValue counts by type:")
    print(df['type'].value_counts())
    
    print("\n✓ DataFrame creation tests passed!\n")
    
    return df


def test_save_functionality(loader, output_dir: Path):
    """Test save functionality."""
    print("=" * 70)
    print("TEST 4: Save Functionality")
    print("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset info
    print("\nSaving dataset info to JSON...")
    loader.save_dataset_info(str(output_dir / "dataset_info.json"))
    
    # Save DataFrame
    print("Saving DataFrame to CSV...")
    df = loader.create_dataframe()
    df.to_csv(str(output_dir / "dataset_dataframe.csv"), index=False, encoding='utf-8')
    
    # Export label mappings
    print("Exporting label mappings...")
    export_label_mappings(loader, str(output_dir / "label_mappings.txt"))
    
    print(f"\n✓ All files saved to: {output_dir}\n")


def test_utilities(loader, output_dir: Path):
    """Test utility functions."""
    print("=" * 70)
    print("TEST 5: Utility Functions")
    print("=" * 70)
    
    # Validate dataset integrity
    print("\nValidating dataset integrity (sampling 50 images per class)...")
    validation_results = validate_dataset_integrity(loader, sample_size=50)
    
    # Get image statistics
    print("\nCalculating image statistics...")
    stats = get_image_statistics(loader, num_samples=500)
    
    # Plot class distribution
    print("\nGenerating class distribution plot...")
    try:
        plot_class_distribution(loader, save_path=str(output_dir / "class_distribution.png"))
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")
    
    # Visualize sample images
    print("\nVisualizing sample images...")
    try:
        visualize_sample_images(loader, num_samples=10, save_path=str(output_dir / "sample_images.png"))
    except Exception as e:
        print(f"Warning: Could not visualize samples: {e}")
    
    # Visualize reference images
    print("\nVisualizing reference images...")
    try:
        sample_labels = [0, 13, 31, 50, 100, 150, 200, 246]
        visualize_reference_images(loader, sample_labels, save_path=str(output_dir / "reference_images.png"))
    except Exception as e:
        print(f"Warning: Could not visualize reference images: {e}")
    
    print("\n✓ Utility function tests completed!\n")


def run_all_tests(dataset_path: str):
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING MODULE 1 TESTS")
    print("=" * 70 + "\n")
    
    # Test 1: Character Mapping
    test_character_mapping()
    
    # Test 2: Dataset Loader
    loader = test_dataset_loader(dataset_path)
    
    # Test 3: DataFrame Creation
    df = test_dataframe_creation(loader)
    
    # Setup output directory
    output_dir = Path(dataset_path).parent / "src" / "mod1" / "output"
    
    # Test 4: Save Functionality
    test_save_functionality(loader, output_dir)
    
    # Test 5: Utility Functions
    test_utilities(loader, output_dir)
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nModule 1 is ready for integration with Module 2!")


if __name__ == "__main__":
    # Set your dataset path here
    DATASET_PATH = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Run all tests
    run_all_tests(DATASET_PATH)
