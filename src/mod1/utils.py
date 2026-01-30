"""
Module 1 Utilities: Helper functions for dataset operations

This module provides utility functions for dataset visualization,
validation, and analysis operations.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import random


def visualize_sample_images(loader, num_samples: int = 10, save_path: Optional[str] = None):
    """
    Visualize random sample images from the dataset with labels.
    
    Args:
        loader: TLFS23DatasetLoader instance
        num_samples: Number of random samples to display
        save_path: Optional path to save the visualization
    """
    # Get all image paths
    all_images = loader.get_all_image_paths()
    
    # Random sample
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    # Calculate grid size
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (img_path, label) in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Read and display image
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            
            # Get character info
            class_info = loader.get_class_info(label)
            title = f"{class_info['tamil_char']} ({class_info['pronunciation']})\nLabel: {label}"
            ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Error loading', ha='center', va='center')
        
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


def visualize_reference_images(loader, labels: List[int], save_path: Optional[str] = None):
    """
    Visualize reference images for specified labels.
    
    Args:
        loader: TLFS23DatasetLoader instance
        labels: List of labels to visualize
        save_path: Optional path to save the visualization
    """
    cols = min(5, len(labels))
    rows = (len(labels) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, label in enumerate(labels):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get reference image
        ref_path = loader.get_reference_image(label)
        if ref_path and os.path.exists(ref_path):
            img = cv2.imread(ref_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                
                # Get character info
                class_info = loader.get_class_info(label)
                title = f"{class_info['tamil_char']} ({class_info['pronunciation']})\nLabel: {label}"
                ax.set_title(title, fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Error loading', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center')
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(labels), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reference images saved to: {save_path}")
    
    plt.show()


def plot_class_distribution(loader, save_path: Optional[str] = None):
    """
    Plot the distribution of images across classes.
    
    Args:
        loader: TLFS23DatasetLoader instance
        save_path: Optional path to save the plot
    """
    labels = sorted(loader.class_paths.keys())
    counts = [loader.class_paths[label]['image_count'] for label in labels]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Bar plot
    ax1.bar(labels, counts, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Class Label', fontsize=12)
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_title('Image Distribution Across All Classes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(counts), color='r', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    ax1.legend()
    
    # Distribution by type
    type_counts = {}
    for label, class_info in loader.class_paths.items():
        char_type = class_info['type']
        if char_type not in type_counts:
            type_counts[char_type] = 0
        type_counts[char_type] += class_info['image_count']
    
    types = list(type_counts.keys())
    type_values = list(type_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    ax2.bar(types, type_values, alpha=0.8, color=colors[:len(types)])
    ax2.set_xlabel('Character Type', fontsize=12)
    ax2.set_ylabel('Total Images', fontsize=12)
    ax2.set_title('Image Distribution by Character Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (t, v) in enumerate(zip(types, type_values)):
        ax2.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()


def validate_dataset_integrity(loader, sample_size: int = 100) -> dict:
    """
    Validate dataset integrity by checking if images can be read.
    
    Args:
        loader: TLFS23DatasetLoader instance
        sample_size: Number of images to sample from each class
    
    Returns:
        Dictionary with validation results
    """
    print("Validating dataset integrity...")
    
    validation_results = {
        'total_checked': 0,
        'successful': 0,
        'failed': 0,
        'corrupted_files': [],
        'empty_files': [],
        'unreadable_files': []
    }
    
    for label, class_info in loader.class_paths.items():
        # Sample images from this class
        image_paths = class_info['image_paths']
        sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))
        
        for img_path in sample_paths:
            validation_results['total_checked'] += 1
            
            try:
                # Check if file exists
                if not os.path.exists(img_path):
                    validation_results['failed'] += 1
                    validation_results['unreadable_files'].append(img_path)
                    continue
                
                # Check file size
                file_size = os.path.getsize(img_path)
                if file_size == 0:
                    validation_results['failed'] += 1
                    validation_results['empty_files'].append(img_path)
                    continue
                
                # Try to read the image
                img = cv2.imread(img_path)
                if img is None:
                    validation_results['failed'] += 1
                    validation_results['corrupted_files'].append(img_path)
                    continue
                
                # Check if image has valid dimensions
                if img.shape[0] == 0 or img.shape[1] == 0:
                    validation_results['failed'] += 1
                    validation_results['corrupted_files'].append(img_path)
                    continue
                
                validation_results['successful'] += 1
                
            except Exception as e:
                validation_results['failed'] += 1
                validation_results['unreadable_files'].append(img_path)
    
    # Calculate success rate
    if validation_results['total_checked'] > 0:
        success_rate = (validation_results['successful'] / validation_results['total_checked']) * 100
        validation_results['success_rate'] = success_rate
    else:
        validation_results['success_rate'] = 0.0
    
    print(f"\nValidation Results:")
    print(f"Total checked: {validation_results['total_checked']}")
    print(f"Successful: {validation_results['successful']}")
    print(f"Failed: {validation_results['failed']}")
    print(f"Success rate: {validation_results['success_rate']:.2f}%")
    
    if validation_results['corrupted_files']:
        print(f"\nFound {len(validation_results['corrupted_files'])} corrupted files")
    if validation_results['empty_files']:
        print(f"Found {len(validation_results['empty_files'])} empty files")
    if validation_results['unreadable_files']:
        print(f"Found {len(validation_results['unreadable_files'])} unreadable files")
    
    return validation_results


def get_image_statistics(loader, num_samples: int = 1000) -> dict:
    """
    Calculate image statistics (dimensions, channels, etc.).
    
    Args:
        loader: TLFS23DatasetLoader instance
        num_samples: Number of images to sample for statistics
    
    Returns:
        Dictionary with image statistics
    """
    print(f"Calculating image statistics from {num_samples} samples...")
    
    all_images = loader.get_all_image_paths()
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    widths = []
    heights = []
    channels = []
    file_sizes = []
    
    for img_path, _ in samples:
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                c = img.shape[2] if len(img.shape) == 3 else 1
                
                widths.append(w)
                heights.append(h)
                channels.append(c)
                
                # Get file size
                file_size = os.path.getsize(img_path)
                file_sizes.append(file_size)
        except Exception as e:
            continue
    
    stats = {
        'num_samples': len(widths),
        'width': {
            'min': min(widths) if widths else 0,
            'max': max(widths) if widths else 0,
            'mean': np.mean(widths) if widths else 0,
            'std': np.std(widths) if widths else 0
        },
        'height': {
            'min': min(heights) if heights else 0,
            'max': max(heights) if heights else 0,
            'mean': np.mean(heights) if heights else 0,
            'std': np.std(heights) if heights else 0
        },
        'channels': {
            'unique': list(set(channels)) if channels else []
        },
        'file_size_kb': {
            'min': min(file_sizes) / 1024 if file_sizes else 0,
            'max': max(file_sizes) / 1024 if file_sizes else 0,
            'mean': np.mean(file_sizes) / 1024 if file_sizes else 0
        }
    }
    
    print("\nImage Statistics:")
    print(f"Width: {stats['width']['mean']:.1f} ± {stats['width']['std']:.1f} "
          f"(range: {stats['width']['min']} - {stats['width']['max']})")
    print(f"Height: {stats['height']['mean']:.1f} ± {stats['height']['std']:.1f} "
          f"(range: {stats['height']['min']} - {stats['height']['max']})")
    print(f"Channels: {stats['channels']['unique']}")
    print(f"File size: {stats['file_size_kb']['mean']:.1f} KB "
          f"(range: {stats['file_size_kb']['min']:.1f} - {stats['file_size_kb']['max']:.1f} KB)")
    
    return stats


def export_label_mappings(loader, output_path: str):
    """
    Export label mappings to a text file for reference.
    
    Args:
        loader: TLFS23DatasetLoader instance
        output_path: Path to save the mappings
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TLFS23 DATASET - LABEL TO CHARACTER MAPPINGS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Label':<8} {'Tamil':<10} {'Pronunciation':<20} {'Type':<15} {'Images':<10}\n")
        f.write("-" * 80 + "\n")
        
        for label in sorted(loader.class_paths.keys()):
            class_info = loader.get_class_info(label)
            f.write(f"{label:<8} {class_info['tamil_char']:<10} "
                   f"{class_info['pronunciation']:<20} {class_info['type']:<15} "
                   f"{class_info['image_count']:<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Classes: {len(loader.class_paths)}\n")
        f.write(f"Total Images: {loader.dataset_stats['total_images']:,}\n")
        f.write(f"Average Images per Class: {loader.dataset_stats['avg_images_per_class']:.2f}\n")
        
        f.write("\nCharacter Type Distribution:\n")
        for char_type, count in loader.dataset_stats['character_type_counts'].items():
            f.write(f"  - {char_type.capitalize()}: {count} classes\n")
    
    print(f"Label mappings exported to: {output_path}")
