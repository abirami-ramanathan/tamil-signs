# Tamil Sign Language Recognition System

A machine learning system for recognizing Tamil sign language alphabets using MediaPipe and Random Forest classifier.

## Project Overview

This project implements a Tamil sign language recognition system that processes static hand gesture images to classify 247 Tamil alphabet characters. The system uses:

- **MediaPipe Hands**: For extracting 21 hand landmarks (63-dimensional feature vectors)
- **Random Forest Classifier**: For alphabet classification
- **TLFS23 Dataset**: 254,147 images across 247 classes from 120 individuals

## Dataset

This project uses the **TLFS23 (Tamil Language Finger Spelling)** dataset, which is publicly available on [Mendeley Data](https://data.mendeley.com/datasets/). 

**Note**: The dataset is NOT included in this repository due to its size (254,147 images). Please download it separately from the official source.

## Modules

The project is organized into 7 modules:

1. **Dataset Loading & Label Mapping** - TLFS23 dataset initialization and class mapping
2. **Image Preprocessing** - RGB conversion and format standardization
3. **Hand Landmark Extraction** - MediaPipe-based feature extraction (21 landmarks × 3 coordinates)
4. **Feature Engineering** - Scaling, train-test split, and serialization
5. **Model Training** - Random Forest classifier training and evaluation
6. **Real-time Prediction** - Webcam-based inference with visual feedback
7. **Word Recognition** - Alphabet sequence aggregation for word formation

See [Tamil_Sign_Language_Project_Modules.md](Tamil_Sign_Language_Project_Modules.md) for detailed documentation.

## Getting Started

### Prerequisites

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

### Download Dataset

1. Download the TLFS23 dataset from [Mendeley Data](https://data.mendeley.com/)
2. Extract it to the project root folder
3. Ensure the folder structure matches:
   ```
   TLFS23 - Tamil Language Finger Spelling Image Dataset/
   ├── Dataset Folders/
   │   ├── 1/
   │   ├── 2/
   │   └── ... (247 class folders)
   └── Refrence Image/
   ```

## License

This project uses the publicly available TLFS23 dataset. Please refer to the dataset's license terms for usage.

## Contributors

- Project by Abirami Ramanathan

---

**Note**: This is a research/educational project for Tamil sign language recognition.
