# Tamil Sign Language Recognition System
## Project Structure

```
tamil-signs/
│
├── README.md                                   # Main project README
├── Tamil_Sign_Language_Project_Modules.md      # Detailed module specifications
├── requirements.txt                             # Python dependencies
│
├── TLFS23 - Tamil Language Finger Spelling Image Dataset/
│   ├── ReadMe.txt                              # Dataset documentation
│   ├── Dataset Folders/                        # 247 class folders (1-247)
│   │   ├── 1/                                  # அ (a) - ~2,128 images
│   │   ├── 2/                                  # ஆ (ā) - ~2,000+ images
│   │   ├── ...
│   │   └── 247/                                # னௌ (Ṉau) - ~2,038 images
│   └── Refrence Image/                         # Reference images (1-247)
│
└── src/
    └── mod1/                                    # ✅ MODULE 1 - COMPLETE
        ├── __init__.py                          # Package initialization
        ├── dataset_loader.py                    # Core dataset loading classes
        ├── utils.py                             # Utility functions
        ├── test_module1.py                      # Comprehensive test suite
        ├── quick_start.py                       # Quick start examples
        ├── README.md                            # Module documentation
        ├── IMPLEMENTATION_SUMMARY.md            # Implementation details
        │
        └── output/                              # Generated output files
            ├── dataset_info.json                # Complete dataset metadata
            ├── dataset_dataframe.csv            # Full dataset CSV (508K rows)
            ├── label_mappings.txt               # Human-readable mappings
            ├── class_distribution.png           # Distribution visualization
            ├── sample_images.png                # Sample image visualization
            └── reference_images.png             # Reference image visualization
```

## Module Implementation Status

| Module | Status | Files | Features |
|--------|--------|-------|----------|
| **Module 1**: Dataset Loading & Label Mapping | ✅ Complete | 7 files | All features implemented & tested |
| **Module 2**: Image Preprocessing | 🔜 Next | - | Ready to implement |
| **Module 3**: Hand Landmark Extraction | 🔜 Pending | - | MediaPipe ready |
| **Module 4**: Feature Construction & Scaling | 🔜 Pending | - | - |
| **Module 5**: Model Training & Selection | 🔜 Pending | - | - |
| **Module 6**: Real-Time Prediction & UI | 🔜 Pending | - | - |
| **Module 7**: Word Generation | 🔜 Pending | - | - |

## Module 1 Details

### Core Classes
- **`TamilCharacterMapping`**: Manages 247 Tamil character mappings
- **`TLFS23DatasetLoader`**: Main dataset loading functionality

### Key Functions
- `load_dataset_structure()`: Load all 247 classes
- `get_class_info()`: Get information for specific class
- `create_dataframe()`: Create pandas DataFrame
- `save_dataset_info()`: Export to JSON
- Visualization utilities (samples, distributions, references)
- Validation utilities (integrity checks, statistics)

### Dataset Statistics (Loaded Successfully)
- **Total Classes**: 247
- **Total Images**: 508,294
- **Image Resolution**: 640 × 480 pixels
- **Format**: RGB (3 channels)
- **Average Images per Class**: 2,057.87
- **Range**: 2,000 - 2,162 images per class

### Character Distribution
- **Vowels**: 13 classes (27,442 images)
- **Consonants**: 18 classes (37,978 images)
- **Compound Characters**: 216 classes (442,874 images)

## Installation & Usage

### 1. Install Dependencies
```bash
cd "c:\Users\Abirami Ramanathan\Desktop\tamil-signs"
pip install -r requirements.txt
```

### 2. Run Tests
```bash
cd src\mod1
python test_module1.py
```

### 3. Quick Start
```bash
python quick_start.py
```

### 4. Use in Your Code
```python
from src.mod1 import TLFS23DatasetLoader

loader = TLFS23DatasetLoader(dataset_path)
loader.load_dataset_structure()
df = loader.create_dataframe()
```

## Next Module: Module 2

**Image Preprocessing** will:
1. Use `loader.get_all_image_paths()` for batch processing
2. Resize/normalize images for MediaPipe
3. Convert to RGB format
4. Prepare for hand landmark extraction

## Dependencies Installed

✅ All dependencies installed successfully:
- numpy >= 1.21.0
- pandas >= 1.3.0
- opencv-python >= 4.5.0
- pillow >= 8.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- mediapipe >= 0.8.9
- joblib >= 1.0.0
- tqdm >= 4.62.0

## Project Guidelines

### Code Quality
- ✅ Type hints for all functions
- ✅ Comprehensive docstrings
- ✅ Error handling and validation
- ✅ Progress bars for long operations
- ✅ Unicode support for Tamil characters

### Documentation
- ✅ Module-level README
- ✅ API reference
- ✅ Usage examples
- ✅ Test suite with clear outputs
- ✅ Implementation summary

### Testing
- ✅ Unit tests for all major functions
- ✅ Integration tests
- ✅ Dataset validation tests
- ✅ 100% success rate on validation

## Contact & Support

For issues or questions about Module 1:
1. Check `src/mod1/README.md` for detailed documentation
2. Run `test_module1.py` to verify installation
3. See `quick_start.py` for usage examples

---

**Status**: Module 1 Complete ✅ | Ready for Module 2 Development 🚀
