"""
Quick Start Guide for Module 1: Dataset Loading

This script demonstrates the basic usage of Module 1 with simple examples.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

from pathlib import Path
import sys

# Add module to path
sys.path.append(str(Path(__file__).parent.parent))

from mod1 import TLFS23DatasetLoader, TamilCharacterMapping


def example_1_basic_loading():
    """Example 1: Basic dataset loading"""
    print("="*70)
    print("EXAMPLE 1: Basic Dataset Loading")
    print("="*70)
    
    # Set dataset path
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Create loader
    loader = TLFS23DatasetLoader(dataset_path)
    
    # Load dataset
    loader.load_dataset_structure()
    
    # Print summary
    print(loader.get_dataset_summary())


def example_2_character_mapping():
    """Example 2: Working with character mappings"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Character Mapping")
    print("="*70)
    
    mapping = TamilCharacterMapping()
    
    # Get character info by label
    print("\nGetting character info by label:")
    char_info = mapping.get_character_by_label(0)
    print(f"Label 0: {char_info['tamil']} ({char_info['pronunciation']}) - {char_info['type']}")
    
    # Get label by character
    print("\nGetting label by character:")
    label = mapping.get_label_by_character('அ')
    print(f"Character 'அ' has label: {label}")
    
    # Get all characters
    print("\nTotal characters in dataset:")
    all_chars = mapping.get_all_characters()
    print(f"Total: {len(all_chars)} characters")
    print(f"First 10: {all_chars[:10]}")


def example_3_accessing_images():
    """Example 3: Accessing image paths"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Accessing Image Paths")
    print("="*70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Get info for a specific class
    print("\nClass information for Label 0 (அ):")
    class_info = loader.get_class_info(0)
    print(f"Tamil Character: {class_info['tamil_char']}")
    print(f"Pronunciation: {class_info['pronunciation']}")
    print(f"Type: {class_info['type']}")
    print(f"Number of images: {class_info['image_count']}")
    print(f"First 3 image paths:")
    for i, path in enumerate(class_info['image_paths'][:3], 1):
        print(f"  {i}. {path}")


def example_4_create_dataframe():
    """Example 4: Creating a pandas DataFrame"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Creating DataFrame")
    print("="*70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Create DataFrame
    print("\nCreating DataFrame...")
    df = loader.create_dataframe()
    
    print(f"DataFrame shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df[['label', 'tamil_char', 'pronunciation', 'type']].head())
    
    print(f"\nDataset statistics:")
    print(f"Total images: {len(df):,}")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"\nImages by type:")
    print(df['type'].value_counts())


def example_5_filter_by_type():
    """Example 5: Filter dataset by character type"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Filter by Character Type")
    print("="*70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    df = loader.create_dataframe()
    
    # Get only vowels
    vowels_df = df[df['type'] == 'vowel']
    print(f"\nVowels:")
    print(f"Total vowel images: {len(vowels_df):,}")
    print(f"Unique vowel characters: {vowels_df['label'].nunique()}")
    print(f"Vowel characters: {vowels_df['tamil_char'].unique().tolist()}")
    
    # Get only consonants
    consonants_df = df[df['type'] == 'consonant']
    print(f"\nConsonants:")
    print(f"Total consonant images: {len(consonants_df):,}")
    print(f"Unique consonant characters: {consonants_df['label'].nunique()}")
    
    # Get only compounds
    compounds_df = df[df['type'] == 'compound']
    print(f"\nCompound Characters:")
    print(f"Total compound images: {len(compounds_df):,}")
    print(f"Unique compound characters: {compounds_df['label'].nunique()}")


def example_6_save_data():
    """Example 6: Save dataset information"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Save Dataset Information")
    print("="*70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    # Create output directory
    output_dir = Path(__file__).parent / "quick_start_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save dataset info
    print(f"\nSaving files to: {output_dir}")
    
    loader.save_dataset_info(str(output_dir / "dataset_info.json"))
    print("✓ Saved dataset_info.json")
    
    df = loader.create_dataframe()
    df.to_csv(str(output_dir / "dataset.csv"), index=False, encoding='utf-8')
    print("✓ Saved dataset.csv")
    
    # Save a subset (first 1000 rows for quick testing)
    df.head(1000).to_csv(str(output_dir / "dataset_sample.csv"), index=False, encoding='utf-8')
    print("✓ Saved dataset_sample.csv (first 1000 rows)")
    
    print(f"\n✓ All files saved successfully!")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("MODULE 1 - QUICK START GUIDE")
    print("="*70)
    print("\nThis script demonstrates basic usage of Module 1")
    print("Dataset Loading and Label Mapping functionality.\n")
    
    try:
        # Run examples
        example_1_basic_loading()
        example_2_character_mapping()
        example_3_accessing_images()
        example_4_create_dataframe()
        example_5_filter_by_type()
        example_6_save_data()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nYou can now:")
        print("1. Use the loader in your own scripts")
        print("2. Integrate with Module 2 (Image Preprocessing)")
        print("3. Explore the saved output files")
        print("\nFor more details, see:")
        print("- src/mod1/README.md")
        print("- src/mod1/test_module1.py")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nMake sure:")
        print("1. Dataset path is correct")
        print("2. All dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
