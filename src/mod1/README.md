# Module 1: TLFS23 Dataset Loading and Label Mapping

## Overview

This module implements **Module 1** of the Tamil Sign Language Recognition System. It is responsible for loading the TLFS23 dataset (Tamil Language Fingerspelling dataset) which contains 254,147 images across 247 Tamil alphabet classes. It handles dataset initialization, class mapping, and organizes the data structure for subsequent processing stages.

## Features

### Core Functionality
- **Dataset Loading**: Efficiently loads and organizes 247 Tamil alphabet classes
- **Label Mapping**: Bidirectional mapping between numerical labels (0-246) and Tamil characters
- **Dataset Validation**: Validates dataset integrity and image accessibility
- **Statistics Generation**: Comprehensive dataset statistics and analysis
- **Data Export**: Export to CSV, JSON, and text formats

### Character Types Supported
- **Vowels (உயிர் எழுத்துக்கள்)**: 13 classes
- **Consonants (மெய் எழுத்துக்கள்)**: 18 classes
- **Compound Characters (உயிர்மெய் எழுத்துக்கள்)**: 216 classes

## Files

```
mod1/
├── __init__.py              # Package initialization
├── dataset_loader.py        # Main dataset loader implementation
├── utils.py                 # Utility functions for visualization & validation
├── test_module1.py          # Comprehensive test suite
├── README.md                # This file
└── output/                  # Generated output files (created on first run)
    ├── dataset_info.json
    ├── dataset_dataframe.csv
    ├── label_mappings.txt
    ├── class_distribution.png
    ├── sample_images.png
    └── reference_images.png
```

## Installation

### 1. Install Dependencies

```bash
cd c:\Users\Abirami Ramanathan\Desktop\tamil-signs
pip install -r requirements.txt
```

### 2. Verify Dataset Structure

Ensure your dataset is organized as:
```
TLFS23 - Tamil Language Finger Spelling Image Dataset/
├── ReadMe.txt
├── Dataset Folders/
│   ├── 1/              # அ (a) - ~1000 images
│   ├── 2/              # ஆ (ā) - ~1000 images
│   ├── ...
│   └── 247/            # னௌ (Ṉau) - ~1000 images
└── Refrence Image/     # Reference images (1 per class)
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

## Usage

### Basic Usage

```python
from src.mod1 import TLFS23DatasetLoader

# Initialize the loader
dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
loader = TLFS23DatasetLoader(dataset_path)

# Load dataset structure
dataset_info = loader.load_dataset_structure()

# Print summary
print(loader.get_dataset_summary())

# Create DataFrame
df = loader.create_dataframe()
print(df.head())

# Get class information
class_info = loader.get_class_info(label=0)  # Get info for label 0 (அ)
print(f"Character: {class_info['tamil_char']}")
print(f"Images: {class_info['image_count']}")
```

### Character Mapping

```python
from src.mod1 import TamilCharacterMapping

# Initialize mapping
mapping = TamilCharacterMapping()

# Get character by label
char_info = mapping.get_character_by_label(0)
print(f"Label 0: {char_info['tamil']} ({char_info['pronunciation']})")

# Get label by character
label = mapping.get_label_by_character('அ')
print(f"Character 'அ' has label: {label}")

# Get all characters
all_chars = mapping.get_all_characters()
print(f"Total characters: {len(all_chars)}")
```

### Visualization

```python
from src.mod1 import TLFS23DatasetLoader
from src.mod1.utils import (
    visualize_sample_images,
    plot_class_distribution,
    visualize_reference_images
)

loader = TLFS23DatasetLoader(dataset_path)
loader.load_dataset_structure()

# Visualize random samples
visualize_sample_images(loader, num_samples=10, save_path='samples.png')

# Plot class distribution
plot_class_distribution(loader, save_path='distribution.png')

# Visualize reference images
labels = [0, 13, 31, 50, 100, 150, 200, 246]
visualize_reference_images(loader, labels, save_path='references.png')
```

### Dataset Validation

```python
from src.mod1.utils import validate_dataset_integrity, get_image_statistics

# Validate dataset
validation_results = validate_dataset_integrity(loader, sample_size=100)
print(f"Success rate: {validation_results['success_rate']:.2f}%")

# Get image statistics
stats = get_image_statistics(loader, num_samples=1000)
print(f"Average width: {stats['width']['mean']:.1f} pixels")
print(f"Average height: {stats['height']['mean']:.1f} pixels")
```

### Export Data

```python
from src.mod1.utils import export_label_mappings

# Save dataset info to JSON
loader.save_dataset_info('output/dataset_info.json')

# Save DataFrame to CSV
df = loader.create_dataframe()
df.to_csv('output/dataset.csv', index=False, encoding='utf-8')

# Export label mappings to text
export_label_mappings(loader, 'output/mappings.txt')
```

## Running Tests

Run the comprehensive test suite:

```bash
cd c:\Users\Abirami Ramanathan\Desktop\tamil-signs\src\mod1
python test_module1.py
```

The test suite will:
1. Test character mapping functionality
2. Test dataset loader
3. Test DataFrame creation
4. Test save functionality
5. Test utility functions
6. Generate visualizations and reports

## API Reference

### TamilCharacterMapping

| Method | Description | Returns |
|--------|-------------|---------|
| `get_character_by_folder(folder_num)` | Get character info by folder number (1-247) | Dict |
| `get_character_by_label(label)` | Get character info by ML label (0-246) | Dict |
| `get_label_by_character(character)` | Get ML label by Tamil character | int |
| `get_folder_by_character(character)` | Get folder number by Tamil character | int |
| `get_all_characters()` | Get list of all Tamil characters | List[str] |
| `get_character_type_counts()` | Get count of each character type | Dict |

### TLFS23DatasetLoader

| Method | Description | Returns |
|--------|-------------|---------|
| `load_dataset_structure(validate_images)` | Load and organize dataset structure | Dict |
| `get_class_info(label)` | Get information about a specific class | Dict |
| `get_all_image_paths()` | Get all image paths with labels | List[Tuple] |
| `get_reference_image(label)` | Get reference image path for a class | Optional[str] |
| `save_dataset_info(output_path)` | Save dataset info to JSON | None |
| `create_dataframe()` | Create pandas DataFrame with all data | DataFrame |
| `get_dataset_summary()` | Generate detailed dataset summary | str |

## Output Format

### Dataset Info JSON Structure
```json
{
  "dataset_path": "path/to/dataset",
  "class_paths": {
    "0": {
      "folder_num": 1,
      "tamil_char": "அ",
      "pronunciation": "a",
      "type": "vowel",
      "image_count": 1029,
      "image_paths": ["path1", "path2", ...]
    }
  },
  "dataset_stats": {
    "total_classes": 247,
    "total_images": 254147,
    "avg_images_per_class": 1029.0
  }
}
```

### DataFrame Columns
- `image_path`: Full path to image file
- `label`: ML label (0-246)
- `tamil_char`: Tamil character
- `pronunciation`: Pronunciation in English
- `type`: Character type (vowel/consonant/compound)
- `folder_num`: Original folder number (1-247)

## Dataset Statistics

Expected statistics for TLFS23 dataset:
- **Total Classes**: 247
- **Total Images**: ~254,147
- **Average Images per Class**: ~1,029
- **Character Types**:
  - Vowels: 13 classes
  - Consonants: 18 classes
  - Compound Characters: 216 classes

## Integration with Other Modules

This module outputs data that will be consumed by:
- **Module 2**: Image Preprocessing
- **Module 3**: Hand Landmark Extraction using MediaPipe

The DataFrame and image paths can be directly used in the preprocessing pipeline.

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Verify dataset path is correct
2. **UnicodeDecodeError**: Ensure Python supports UTF-8 encoding
3. **Memory Issues**: Use batch processing for large operations

### Performance Tips

- Set `validate_images=False` for faster loading
- Use sampling for validation and statistics on large datasets
- Cache the DataFrame after first load

## License

Part of the Tamil Sign Language Recognition System project.

## Authors

Tamil Sign Language Recognition Team  
January 2026
