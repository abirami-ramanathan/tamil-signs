"""
Module 2 Utilities: Helper functions for image preprocessing

This module provides utility functions for image analysis, visualization,
and quality assessment.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import random
from pathlib import Path


def visualize_preprocessed_images(images: np.ndarray, 
                                   labels: np.ndarray,
                                   label_mapping: dict,
                                   num_samples: int = 10,
                                   save_path: Optional[str] = None):
    """
    Visualize preprocessed images with their labels.
    
    Args:
        images: Array of preprocessed images
        labels: Array of labels
        label_mapping: Dictionary mapping labels to character info
        num_samples: Number of samples to display
        save_path: Optional path to save the visualization
    """
    # Random sample
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    # Calculate grid size
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, sample_idx in enumerate(indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get image and label
        img = images[sample_idx]
        label = labels[sample_idx]
        
        # Display image
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Normalized image
            ax.imshow(img)
        else:
            # uint8 image
            ax.imshow(img)
        
        # Get character info
        char_info = label_mapping.get(label, {'tamil': '?', 'pronunciation': '?'})
        title = f"{char_info['tamil']} ({char_info['pronunciation']})\nLabel: {label}"
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def compare_original_vs_preprocessed(original_path: str,
                                     preprocessed_image: np.ndarray,
                                     save_path: Optional[str] = None):
    """
    Compare original and preprocessed images side by side.
    
    Args:
        original_path: Path to original image
        preprocessed_image: Preprocessed image array
        save_path: Optional path to save the comparison
    """
    # Load original
    original = cv2.imread(original_path)
    if original is not None:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original
    if original is not None:
        ax1.imshow(original)
        ax1.set_title(f'Original\nSize: {original.shape[:2]}', fontsize=12)
    else:
        ax1.text(0.5, 0.5, 'Original not found', ha='center', va='center')
    ax1.axis('off')
    
    # Preprocessed
    ax2.imshow(preprocessed_image)
    ax2.set_title(f'Preprocessed\nSize: {preprocessed_image.shape[:2]}', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()


def analyze_image_statistics(images: np.ndarray) -> dict:
    """
    Analyze statistics of preprocessed images.
    
    Args:
        images: Array of images
    
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'num_images': len(images),
        'shape': images[0].shape,
        'dtype': str(images.dtype),
        'mean_pixel_value': np.mean(images),
        'std_pixel_value': np.std(images),
        'min_pixel_value': np.min(images),
        'max_pixel_value': np.max(images),
        'memory_size_mb': images.nbytes / (1024 * 1024)
    }
    
    # Channel-wise statistics if RGB
    if len(images.shape) == 4 and images.shape[3] == 3:
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            stats[f'{channel.lower()}_mean'] = np.mean(images[:, :, :, i])
            stats[f'{channel.lower()}_std'] = np.std(images[:, :, :, i])
    
    return stats


def plot_preprocessing_statistics(train_stats: dict, test_stats: dict,
                                  save_path: Optional[str] = None):
    """
    Plot preprocessing statistics for train and test sets.
    
    Args:
        train_stats: Training set statistics
        test_stats: Testing set statistics
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rates
    ax1 = axes[0]
    categories = ['Training', 'Testing']
    success = [train_stats['successful'], test_stats['successful']]
    failed = [train_stats['failed'], test_stats['failed']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, success, width, label='Successful', color='#4CAF50')
    ax1.bar(x + width/2, failed, width, label='Failed', color='#F44336')
    
    ax1.set_xlabel('Dataset Split', fontsize=12)
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_title('Preprocessing Results', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (s, f) in enumerate(zip(success, failed)):
        ax1.text(i - width/2, s, str(s), ha='center', va='bottom', fontsize=10)
        ax1.text(i + width/2, f, str(f), ha='center', va='bottom', fontsize=10)
    
    # Success rates percentage
    ax2 = axes[1]
    train_rate = (train_stats['successful'] / train_stats['total_processed'] * 100 
                  if train_stats['total_processed'] > 0 else 0)
    test_rate = (test_stats['successful'] / test_stats['total_processed'] * 100
                 if test_stats['total_processed'] > 0 else 0)
    
    rates = [train_rate, test_rate]
    colors = ['#2196F3', '#FF9800']
    
    bars = ax2.bar(categories, rates, color=colors, alpha=0.7)
    ax2.set_xlabel('Dataset Split', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Preprocessing Success Rate', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    
    plt.show()


def visualize_data_distribution(y_train: np.ndarray, y_test: np.ndarray,
                                label_mapping: dict,
                                save_path: Optional[str] = None):
    """
    Visualize the distribution of labels in train and test sets.
    
    Args:
        y_train: Training labels
        y_test: Testing labels
        label_mapping: Dictionary mapping labels to character info
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Training set distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    ax1.bar(unique_train, counts_train, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Class Label', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=np.mean(counts_train), color='r', linestyle='--', 
                label=f'Mean: {np.mean(counts_train):.1f}')
    ax1.legend()
    
    # Testing set distribution
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    ax2.bar(unique_test, counts_test, alpha=0.7, color='orange')
    ax2.set_xlabel('Class Label', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Testing Set - Class Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=np.mean(counts_test), color='r', linestyle='--',
                label=f'Mean: {np.mean(counts_test):.1f}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()


def check_image_quality_batch(images: np.ndarray) -> dict:
    """
    Check quality metrics for a batch of images.
    
    Args:
        images: Array of images
    
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        'total_images': len(images),
        'valid_images': 0,
        'issues': {
            'too_dark': 0,
            'too_bright': 0,
            'low_contrast': 0,
            'suspicious': []
        }
    }
    
    for idx, img in enumerate(images):
        # Check brightness
        mean_brightness = np.mean(img)
        
        if mean_brightness < 30:
            quality_metrics['issues']['too_dark'] += 1
        elif mean_brightness > 225:
            quality_metrics['issues']['too_bright'] += 1
        else:
            quality_metrics['valid_images'] += 1
        
        # Check contrast
        contrast = np.std(img)
        if contrast < 10:
            quality_metrics['issues']['low_contrast'] += 1
    
    quality_metrics['quality_rate'] = (quality_metrics['valid_images'] / 
                                       quality_metrics['total_images'] * 100)
    
    return quality_metrics


def export_preprocessing_report(result: dict, output_path: str):
    """
    Export a detailed preprocessing report to text file.
    
    Args:
        result: Preprocessing result dictionary
        output_path: Path to save the report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MODULE 2: IMAGE PREPROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training samples: {len(result['y_train'])}\n")
        f.write(f"Testing samples: {len(result['y_test'])}\n")
        f.write(f"Total samples: {len(result['y_train']) + len(result['y_test'])}\n")
        f.write(f"Number of classes: {len(np.unique(result['y_train']))}\n")
        f.write(f"Image shape: {result['X_train'][0].shape}\n")
        f.write(f"Image dtype: {result['X_train'].dtype}\n\n")
        
        f.write("TRAINING SET STATISTICS\n")
        f.write("-" * 80 + "\n")
        train_stats = result['train_stats']
        f.write(f"Total processed: {train_stats['total_processed']}\n")
        f.write(f"Successful: {train_stats['successful']}\n")
        f.write(f"Failed: {train_stats['failed']}\n")
        success_rate = (train_stats['successful'] / train_stats['total_processed'] * 100
                       if train_stats['total_processed'] > 0 else 0)
        f.write(f"Success rate: {success_rate:.2f}%\n\n")
        
        f.write("TESTING SET STATISTICS\n")
        f.write("-" * 80 + "\n")
        test_stats = result['test_stats']
        f.write(f"Total processed: {test_stats['total_processed']}\n")
        f.write(f"Successful: {test_stats['successful']}\n")
        f.write(f"Failed: {test_stats['failed']}\n")
        success_rate = (test_stats['successful'] / test_stats['total_processed'] * 100
                       if test_stats['total_processed'] > 0 else 0)
        f.write(f"Success rate: {success_rate:.2f}%\n\n")
        
        f.write("PREPROCESSING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        config = result['preprocessing_config']
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PREPROCESSING COMPLETE\n")
        f.write("=" * 80 + "\n")
    
    print(f"Preprocessing report saved to: {output_path}")


def validate_preprocessed_data(result: dict) -> bool:
    """
    Validate preprocessed data for consistency and correctness.
    
    Args:
        result: Preprocessing result dictionary
    
    Returns:
        True if validation passes, False otherwise
    """
    print("Validating preprocessed data...")
    
    checks = []
    
    # Check shapes
    print("✓ Checking data shapes...")
    checks.append(len(result['X_train'].shape) == 4)  # (N, H, W, C)
    checks.append(len(result['y_train'].shape) == 1)  # (N,)
    checks.append(len(result['X_test'].shape) == 4)
    checks.append(len(result['y_test'].shape) == 1)
    
    # Check matching samples
    print("✓ Checking sample counts...")
    checks.append(len(result['X_train']) == len(result['y_train']))
    checks.append(len(result['X_test']) == len(result['y_test']))
    
    # Check data types
    print("✓ Checking data types...")
    checks.append(result['X_train'].dtype in [np.uint8, np.float32, np.float64])
    checks.append(result['y_train'].dtype in [np.int32, np.int64])
    
    # Check value ranges
    print("✓ Checking value ranges...")
    if result['X_train'].dtype == np.uint8:
        checks.append(np.min(result['X_train']) >= 0 and np.max(result['X_train']) <= 255)
    else:
        checks.append(np.min(result['X_train']) >= 0 and np.max(result['X_train']) <= 1.0)
    
    # Check labels
    print("✓ Checking label consistency...")
    checks.append(np.min(result['y_train']) >= 0)
    checks.append(np.min(result['y_test']) >= 0)
    
    # Overall validation
    all_passed = all(checks)
    
    if all_passed:
        print("\n✓ All validation checks passed!")
    else:
        print(f"\n✗ Validation failed! {sum(checks)}/{len(checks)} checks passed")
    
    return all_passed
