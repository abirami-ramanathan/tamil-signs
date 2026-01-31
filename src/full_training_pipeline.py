"""
Full Training Pipeline: Extract landmarks from FULL dataset for proper training

This script processes a large number of samples per class to achieve
high-quality model training results.

Memory-Efficient Implementation: Processes data in batches to handle
large datasets without memory overflow, following research paper methodology.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import gc

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from mod1 import TLFS23DatasetLoader
from mod2 import DatasetPreprocessor
from mod3.hand_landmark_extractor import DatasetLandmarkExtractor
from mod4 import DatasetConstructor
from mod5 import ModelComparator


def full_training_pipeline():
    """
    Memory-efficient training pipeline using batch processing.
    Follows research paper methodology to handle large datasets.
    """
    
    print("=" * 70)
    print("MEMORY-EFFICIENT TRAINING PIPELINE")
    print("(Following Research Paper Methodology)")
    print("=" * 70)
    
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Configuration - Can now handle more samples due to batch processing!
    SAMPLES_PER_CLASS = 350  # Research paper used ~1000, we'll use 350 for good accuracy
    BATCH_SIZE = 30  # Process 30 classes at a time to avoid memory overflow
    
    print(f"\nConfiguration:")
    print(f"  Samples per class: {SAMPLES_PER_CLASS}")
    print(f"  Batch size: {BATCH_SIZE} classes")
    print(f"  Expected total: ~{SAMPLES_PER_CLASS * 247} images")
    print(f"  Memory-efficient: ✓ (Batch processing)")
    
    # Step 1: Load Dataset
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATASET STRUCTURE")
    print("=" * 70)
    
    loader = TLFS23DatasetLoader(dataset_path)
    loader.load_dataset_structure()
    
    total_classes = len(loader.class_paths)
    print(f"\n✓ Found {total_classes} classes")
    
    # Step 2 & 3: Process in Batches (Preprocess + Extract Landmarks)
    print("\n" + "=" * 70)
    print("STEP 2 & 3: BATCH PROCESSING (Preprocess + Landmarks)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent / "mod3" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collections for final data
    all_train_features = []
    all_train_labels = []
    all_test_features = []
    all_test_labels = []
    
    # Get all classes and process in batches
    all_classes = sorted(loader.class_paths.keys())
    num_batches = (len(all_classes) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\nProcessing {len(all_classes)} classes in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(all_classes))
        batch_classes = all_classes[start_idx:end_idx]
        
        print(f"\n{'─' * 70}")
        print(f"BATCH {batch_idx + 1}/{num_batches}: Classes {batch_classes[0]} to {batch_classes[-1]}")
        print(f"{'─' * 70}")
        
        # Create temporary loader for this batch
        batch_loader = TLFS23DatasetLoader(dataset_path)
        batch_loader.load_dataset_structure()
        
        # Filter to only batch classes
        batch_loader.class_paths = {
            k: v for k, v in batch_loader.class_paths.items() 
            if k in batch_classes
        }
        
        # Preprocess batch
        print(f"  Preprocessing {len(batch_classes)} classes...")
        preprocessor = DatasetPreprocessor(
            dataset_loader=batch_loader,
            target_size=None,
            train_split=0.8
        )
        
        batch_preprocessed = preprocessor.preprocess_dataset(
            max_samples_per_class=SAMPLES_PER_CLASS,
            output_dir=None
        )
        
        print(f"    Train: {batch_preprocessed['X_train'].shape}")
        print(f"    Test: {batch_preprocessed['X_test'].shape}")
        
        # Extract landmarks from batch
        print(f"  Extracting landmarks...")
        landmark_extractor = DatasetLandmarkExtractor(
            min_detection_confidence=0.3,
            enable_retry=False,  # Disabled for speed - dataset is high quality
            enhance_image=False  # Disabled for speed - images already preprocessed
        )
        
        batch_landmarks = landmark_extractor.extract_from_preprocessed_data(
            batch_preprocessed,
            output_dir=None  # Don't save intermediate files
        )
        
        landmark_extractor.close()
        
        print(f"    Train landmarks: {batch_landmarks['X_train'].shape}")
        print(f"    Test landmarks: {batch_landmarks['X_test'].shape}")
        
        # Accumulate results
        all_train_features.append(batch_landmarks['X_train'])
        all_train_labels.append(batch_landmarks['y_train'])
        all_test_features.append(batch_landmarks['X_test'])
        all_test_labels.append(batch_landmarks['y_test'])
        
        # Clear memory
        del batch_preprocessed, batch_landmarks, batch_loader, preprocessor, landmark_extractor
        gc.collect()
        
        print(f"  ✓ Batch {batch_idx + 1} complete (memory cleared)")
    
    # Combine all batches
    print(f"\n{'=' * 70}")
    print("COMBINING ALL BATCHES")
    print(f"{'=' * 70}")
    
    X_train_combined = np.vstack(all_train_features)
    y_train_combined = np.concatenate(all_train_labels)
    X_test_combined = np.vstack(all_test_features)
    y_test_combined = np.concatenate(all_test_labels)
    
    print(f"\n✓ Combined dataset:")
    print(f"  Train: {X_train_combined.shape}")
    print(f"  Test: {X_test_combined.shape}")
    
    # Save combined landmarks
    # Get label mapping from the character mapping
    label_mapping = {i: loader.mapping.get_character_by_label(i)['tamil'] 
                     for i in range(247)}
    
    landmark_data = {
        'X_train': X_train_combined,
        'y_train': y_train_combined,
        'X_test': X_test_combined,
        'y_test': y_test_combined,
        'label_mapping': label_mapping
    }
    
    landmark_pickle = output_dir / 'landmark_features.pkl'
    with open(landmark_pickle, 'wb') as f:
        pickle.dump(landmark_data, f)
    
    print(f"\n✓ Landmarks saved to: {landmark_pickle}")
    
    # Clear memory before next steps
    del all_train_features, all_train_labels, all_test_features, all_test_labels
    gc.collect()
    
    # Step 4: Feature Scaling
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE SCALING")
    print("=" * 70)
    
    constructor = DatasetConstructor(
        test_size=0.2,
        random_state=42,
        scaler_type='standard',
        stratify=True
    )
    
    landmark_pickle = output_dir / 'landmark_features.pkl'
    scaled_output_dir = Path(__file__).parent.parent / "mod4" / "output"
    
    dataset = constructor.construct_dataset(
        landmark_pickle_path=str(landmark_pickle),
        output_dir=str(scaled_output_dir)
    )
    
    print(f"\n✓ Dataset scaled:")
    print(f"  Train: {dataset['X_train'].shape}")
    print(f"  Test: {dataset['X_test'].shape}")
    
    # Step 5: Model Training
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING")
    print("=" * 70)
    
    comparator = ModelComparator(random_state=42)
    
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    print(f"\nTraining on {len(X_train)} samples...")
    comparator.train_all_models(X_train, y_train, X_test, y_test)
    
    # Compare and save
    comparison_df = comparator.compare_models()
    comparator.print_detailed_reports()
    
    model_output_dir = Path(__file__).parent.parent / "mod5" / "output"
    comparator.save_results(str(model_output_dir), dataset)
    
    print("\n" + "=" * 70)
    print("FULL TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nModels saved to: {model_output_dir}")


if __name__ == "__main__":
    full_training_pipeline()
