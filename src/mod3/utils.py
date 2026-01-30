"""
Module 3: Utility Functions for Hand Landmark Analysis

This module provides visualization, analysis, and validation utilities
for hand landmark features.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Union, List, Optional, Dict
import mediapipe as mp


def visualize_landmarks_on_images(
    images: List[np.ndarray],
    landmarks_list: List[np.ndarray],
    labels: Optional[List[int]] = None,
    label_mapping: Optional[Dict[int, str]] = None,
    num_samples: int = 10,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Visualize hand landmarks overlaid on images.
    
    Args:
        images: List of RGB images
        landmarks_list: List of landmark feature vectors (63-dim)
        labels: Optional list of labels
        label_mapping: Optional label to character mapping
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        image = images[i].copy()
        landmarks = landmarks_list[i]
        
        if landmarks is not None and len(landmarks) == 63:
            # Reconstruct landmarks for MediaPipe drawing
            # Create a dummy hand landmarks object
            # Note: This is simplified visualization using matplotlib
            
            # Extract x, y coordinates (ignore z for 2D visualization)
            x_coords = landmarks[0::3]  # Every 3rd element starting from 0
            y_coords = landmarks[1::3]  # Every 3rd element starting from 1
            
            # Denormalize coordinates to image dimensions
            h, w = image.shape[:2]
            x_pixels = (x_coords * w).astype(int)
            y_pixels = (y_coords * h).astype(int)
            
            # Draw landmarks on image
            for j in range(21):
                cv2.circle(image, (x_pixels[j], y_pixels[j]), 5, (0, 255, 0), -1)
            
            # Draw connections (simplified)
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                cv2.line(
                    image,
                    (x_pixels[start_idx], y_pixels[start_idx]),
                    (x_pixels[end_idx], y_pixels[end_idx]),
                    (255, 0, 0),
                    2
                )
        
        axes[i].imshow(image)
        axes[i].axis('off')
        
        if labels is not None and label_mapping is not None:
            title = f"Label: {label_mapping.get(labels[i], labels[i])}"
            axes[i].set_title(title, fontsize=10)
        elif labels is not None:
            axes[i].set_title(f"Label: {labels[i]}", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_landmark_distribution(
    landmarks: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Visualize distribution of landmark features.
    
    Args:
        landmarks: Landmark feature matrix (n_samples, 63)
        feature_names: Optional feature names
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # X coordinates distribution
    x_coords = landmarks[:, 0::3]  # Every 3rd column starting from 0
    axes[0].boxplot(x_coords, labels=[f'L{i}' for i in range(21)])
    axes[0].set_title('X Coordinate Distribution Across Landmarks')
    axes[0].set_xlabel('Landmark Index')
    axes[0].set_ylabel('X Coordinate')
    axes[0].grid(True, alpha=0.3)
    
    # Y coordinates distribution
    y_coords = landmarks[:, 1::3]  # Every 3rd column starting from 1
    axes[1].boxplot(y_coords, labels=[f'L{i}' for i in range(21)])
    axes[1].set_title('Y Coordinate Distribution Across Landmarks')
    axes[1].set_xlabel('Landmark Index')
    axes[1].set_ylabel('Y Coordinate')
    axes[1].grid(True, alpha=0.3)
    
    # Z coordinates distribution
    z_coords = landmarks[:, 2::3]  # Every 3rd column starting from 2
    axes[2].boxplot(z_coords, labels=[f'L{i}' for i in range(21)])
    axes[2].set_title('Z Coordinate Distribution Across Landmarks')
    axes[2].set_xlabel('Landmark Index')
    axes[2].set_ylabel('Z Coordinate (Depth)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_landmark_statistics(landmarks: np.ndarray) -> Dict[str, any]:
    """
    Analyze statistical properties of landmark features.
    
    Args:
        landmarks: Landmark feature matrix (n_samples, 63)
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_samples': landmarks.shape[0],
        'num_features': landmarks.shape[1],
        'mean': np.mean(landmarks, axis=0),
        'std': np.std(landmarks, axis=0),
        'min': np.min(landmarks, axis=0),
        'max': np.max(landmarks, axis=0),
        'median': np.median(landmarks, axis=0),
        'x_coords_mean': np.mean(landmarks[:, 0::3]),
        'y_coords_mean': np.mean(landmarks[:, 1::3]),
        'z_coords_mean': np.mean(landmarks[:, 2::3]),
        'x_coords_std': np.std(landmarks[:, 0::3]),
        'y_coords_std': np.std(landmarks[:, 1::3]),
        'z_coords_std': np.std(landmarks[:, 2::3]),
    }
    
    return stats


def plot_landmark_statistics(
    train_landmarks: np.ndarray,
    test_landmarks: np.ndarray,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plot comparative statistics for train and test landmarks.
    
    Args:
        train_landmarks: Training landmark features
        test_landmarks: Test landmark features
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean comparison
    train_mean = np.mean(train_landmarks, axis=0)
    test_mean = np.mean(test_landmarks, axis=0)
    
    axes[0, 0].plot(train_mean, label='Train', alpha=0.7)
    axes[0, 0].plot(test_mean, label='Test', alpha=0.7)
    axes[0, 0].set_title('Mean Feature Values')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviation comparison
    train_std = np.std(train_landmarks, axis=0)
    test_std = np.std(test_landmarks, axis=0)
    
    axes[0, 1].plot(train_std, label='Train', alpha=0.7)
    axes[0, 1].plot(test_std, label='Test', alpha=0.7)
    axes[0, 1].set_title('Feature Standard Deviation')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coordinate-wise mean
    coord_names = ['X', 'Y', 'Z']
    train_coord_means = [
        np.mean(train_landmarks[:, 0::3]),
        np.mean(train_landmarks[:, 1::3]),
        np.mean(train_landmarks[:, 2::3])
    ]
    test_coord_means = [
        np.mean(test_landmarks[:, 0::3]),
        np.mean(test_landmarks[:, 1::3]),
        np.mean(test_landmarks[:, 2::3])
    ]
    
    x = np.arange(len(coord_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, train_coord_means, width, label='Train')
    axes[1, 0].bar(x + width/2, test_coord_means, width, label='Test')
    axes[1, 0].set_title('Mean Coordinate Values')
    axes[1, 0].set_xlabel('Coordinate')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(coord_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample counts
    data = [train_landmarks.shape[0], test_landmarks.shape[0]]
    labels = ['Train', 'Test']
    colors = ['#3498db', '#e74c3c']
    
    axes[1, 1].bar(labels, data, color=colors)
    axes[1, 1].set_title('Dataset Size')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(data):
        axes[1, 1].text(i, v + 10, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def validate_landmarks(landmarks: np.ndarray) -> Dict[str, any]:
    """
    Validate landmark features for correctness.
    
    Args:
        landmarks: Landmark feature matrix (n_samples, 63)
    
    Returns:
        Dictionary with validation results
    """
    print("Validating landmark features...")
    
    issues = []
    
    # Check shape
    if landmarks.shape[1] != 63:
        issues.append(f"Invalid feature dimension: {landmarks.shape[1]} (expected 63)")
    
    # Check for NaN values
    nan_count = np.isnan(landmarks).sum()
    if nan_count > 0:
        issues.append(f"Found {nan_count} NaN values")
    
    # Check for infinite values
    inf_count = np.isinf(landmarks).sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values")
    
    # Check coordinate ranges (normalized coordinates should be in [0, 1] for x, y)
    x_coords = landmarks[:, 0::3]
    y_coords = landmarks[:, 1::3]
    
    x_out_of_range = ((x_coords < -0.1) | (x_coords > 1.1)).sum()
    y_out_of_range = ((y_coords < -0.1) | (y_coords > 1.1)).sum()
    
    if x_out_of_range > 0:
        issues.append(f"X coordinates out of range: {x_out_of_range} values")
    if y_out_of_range > 0:
        issues.append(f"Y coordinates out of range: {y_out_of_range} values")
    
    # Check for all-zero rows
    zero_rows = np.all(landmarks == 0, axis=1).sum()
    if zero_rows > 0:
        issues.append(f"Found {zero_rows} all-zero samples")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        print("✓ All validation checks passed!")
    else:
        print("✗ Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'num_samples': landmarks.shape[0],
        'num_features': landmarks.shape[1],
        'nan_count': nan_count,
        'inf_count': inf_count,
        'zero_rows': zero_rows
    }


def export_landmark_report(
    landmark_data: Dict[str, np.ndarray],
    output_path: Union[str, Path]
):
    """
    Export comprehensive landmark extraction report.
    
    Args:
        landmark_data: Dictionary with X_train, X_test, y_train, y_test, etc.
        output_path: Path to save report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HAND LANDMARK EXTRACTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Training data
        f.write("TRAINING DATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of samples: {landmark_data['X_train'].shape[0]}\n")
        f.write(f"Feature dimensions: {landmark_data['X_train'].shape[1]}\n")
        f.write(f"Number of classes: {len(np.unique(landmark_data['y_train']))}\n")
        
        train_stats = analyze_landmark_statistics(landmark_data['X_train'])
        f.write(f"\nStatistics:\n")
        f.write(f"  X coordinates mean: {train_stats['x_coords_mean']:.4f} ± {train_stats['x_coords_std']:.4f}\n")
        f.write(f"  Y coordinates mean: {train_stats['y_coords_mean']:.4f} ± {train_stats['y_coords_std']:.4f}\n")
        f.write(f"  Z coordinates mean: {train_stats['z_coords_mean']:.4f} ± {train_stats['z_coords_std']:.4f}\n")
        
        # Test data
        f.write("\n" + "=" * 70 + "\n")
        f.write("TEST DATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of samples: {landmark_data['X_test'].shape[0]}\n")
        f.write(f"Feature dimensions: {landmark_data['X_test'].shape[1]}\n")
        f.write(f"Number of classes: {len(np.unique(landmark_data['y_test']))}\n")
        
        test_stats = analyze_landmark_statistics(landmark_data['X_test'])
        f.write(f"\nStatistics:\n")
        f.write(f"  X coordinates mean: {test_stats['x_coords_mean']:.4f} ± {test_stats['x_coords_std']:.4f}\n")
        f.write(f"  Y coordinates mean: {test_stats['y_coords_mean']:.4f} ± {test_stats['y_coords_std']:.4f}\n")
        f.write(f"  Z coordinates mean: {test_stats['z_coords_mean']:.4f} ± {test_stats['z_coords_std']:.4f}\n")
        
        # Extraction stats
        if 'train_stats' in landmark_data and 'test_stats' in landmark_data:
            f.write("\n" + "=" * 70 + "\n")
            f.write("EXTRACTION STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write("Training:\n")
            f.write(f"  Total processed: {landmark_data['train_stats']['total_processed']}\n")
            f.write(f"  Successful: {landmark_data['train_stats']['successful']}\n")
            f.write(f"  Failed: {landmark_data['train_stats']['failed']}\n")
            f.write(f"  Success rate: {landmark_data['train_stats']['success_rate']:.2f}%\n")
            
            f.write("\nTesting:\n")
            f.write(f"  Total processed: {landmark_data['test_stats']['total_processed']}\n")
            f.write(f"  Successful: {landmark_data['test_stats']['successful']}\n")
            f.write(f"  Failed: {landmark_data['test_stats']['failed']}\n")
            f.write(f"  Success rate: {landmark_data['test_stats']['success_rate']:.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Landmark report exported to: {output_path}")


def compare_landmark_features(
    landmarks1: np.ndarray,
    landmarks2: np.ndarray,
    label1: str = "Set 1",
    label2: str = "Set 2",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Compare two sets of landmark features.
    
    Args:
        landmarks1: First landmark feature set
        landmarks2: Second landmark feature set
        label1: Label for first set
        label2: Label for second set
        save_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature-wise mean comparison
    mean1 = np.mean(landmarks1, axis=0)
    mean2 = np.mean(landmarks2, axis=0)
    
    axes[0, 0].plot(mean1, label=label1, alpha=0.7, linewidth=2)
    axes[0, 0].plot(mean2, label=label2, alpha=0.7, linewidth=2)
    axes[0, 0].set_title('Feature-wise Mean Comparison')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature-wise std comparison
    std1 = np.std(landmarks1, axis=0)
    std2 = np.std(landmarks2, axis=0)
    
    axes[0, 1].plot(std1, label=label1, alpha=0.7, linewidth=2)
    axes[0, 1].plot(std2, label=label2, alpha=0.7, linewidth=2)
    axes[0, 1].set_title('Feature-wise Std Comparison')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram comparison for first feature
    axes[1, 0].hist(landmarks1[:, 0], bins=50, alpha=0.5, label=label1, density=True)
    axes[1, 0].hist(landmarks2[:, 0], bins=50, alpha=0.5, label=label2, density=True)
    axes[1, 0].set_title('Distribution of First Feature')
    axes[1, 0].set_xlabel('Feature Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall statistics comparison
    stats_labels = ['Mean', 'Std', 'Min', 'Max']
    stats1 = [np.mean(landmarks1), np.std(landmarks1), np.min(landmarks1), np.max(landmarks1)]
    stats2 = [np.mean(landmarks2), np.std(landmarks2), np.min(landmarks2), np.max(landmarks2)]
    
    x = np.arange(len(stats_labels))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, stats1, width, label=label1)
    axes[1, 1].bar(x + width/2, stats2, width, label=label2)
    axes[1, 1].set_title('Overall Statistics Comparison')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(stats_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
