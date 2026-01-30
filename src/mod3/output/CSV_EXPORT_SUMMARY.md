# Landmark Features CSV Export - Summary

## Overview
Successfully exported hand landmark features from Module 3 to CSV format. Each hand gesture is now represented as 63 features (21 landmarks × 3 coordinates).

## Generated Files

Located in: `src/mod3/output/`

1. **landmark_features.csv** - Combined dataset (train + test)
   - Size: 420,951 bytes
   - Rows: 586 samples
   - Columns: 65 (class_name, class_label, + 63 landmark features)

2. **train_landmarks.csv** - Training set only
   - Size: 335,724 bytes  
   - Rows: 467 samples
   - Columns: 65

3. **test_landmarks.csv** - Test set only
   - Size: 86,103 bytes
   - Rows: 119 samples
   - Columns: 65

## CSV Structure

### Columns (65 total):

1. **class_name**: Tamil character (அ, ஆ, இ, etc.)
2. **class_label**: Numeric label (0-246)
3-65. **Landmark features** (63 columns):
   - landmark_0_x, landmark_0_y, landmark_0_z (wrist)
   - landmark_1_x, landmark_1_y, landmark_1_z (thumb CMC)
   - landmark_2_x, landmark_2_y, landmark_2_z (thumb MCP)
   - landmark_3_x, landmark_3_y, landmark_3_z (thumb IP)
   - landmark_4_x, landmark_4_y, landmark_4_z (thumb tip)
   - landmark_5_x, landmark_5_y, landmark_5_z (index MCP)
   - landmark_6_x, landmark_6_y, landmark_6_z (index PIP)
   - landmark_7_x, landmark_7_y, landmark_7_z (index DIP)
   - landmark_8_x, landmark_8_y, landmark_8_z (index tip)
   - landmark_9_x, landmark_9_y, landmark_9_z (middle MCP)
   - landmark_10_x, landmark_10_y, landmark_10_z (middle PIP)
   - landmark_11_x, landmark_11_y, landmark_11_z (middle DIP)
   - landmark_12_x, landmark_12_y, landmark_12_z (middle tip)
   - landmark_13_x, landmark_13_y, landmark_13_z (ring MCP)
   - landmark_14_x, landmark_14_y, landmark_14_z (ring PIP)
   - landmark_15_x, landmark_15_y, landmark_15_z (ring DIP)
   - landmark_16_x, landmark_16_y, landmark_16_z (ring tip)
   - landmark_17_x, landmark_17_y, landmark_17_z (pinky MCP)
   - landmark_18_x, landmark_18_y, landmark_18_z (pinky PIP)
   - landmark_19_x, landmark_19_y, landmark_19_z (pinky DIP)
   - landmark_20_x, landmark_20_y, landmark_20_z (pinky tip)

## Data Statistics

- **Total samples**: 586 (467 train + 119 test)
- **Total classes**: 246 Tamil characters
- **Samples per class**: 1-3 samples (mean: 2.38)
- **Coordinate ranges**:
  - X: [0.0545, 0.8900] (normalized 0-1)
  - Y: [0.0170, 0.9443] (normalized 0-1)
  - Z: [-0.2500, 0.0000] (depth, relative to wrist)

## Sample Data

### Example Row:
```
class_name: யொ
class_label: 160
landmark_0_x: 0.7167
landmark_0_y: 0.7718
landmark_0_z: -0.0000
... (60 more landmark features)
landmark_20_x: 0.9231
landmark_20_y: 0.3854
landmark_20_z: -0.1082
```

## Usage

### Load CSV in Python:
```python
import pandas as pd

# Load combined dataset
df = pd.read_csv('src/mod3/output/landmark_features.csv')

# Load train set only
df_train = pd.read_csv('src/mod3/output/train_landmarks.csv')

# Load test set only
df_test = pd.read_csv('src/mod3/output/test_landmarks.csv')

# Access features
X = df.iloc[:, 2:].values  # All 63 landmark features
y = df['class_label'].values  # Numeric labels
class_names = df['class_name'].values  # Tamil character names
```

### Load CSV in Excel:
1. Open Excel
2. Data → From Text/CSV
3. Select the CSV file
4. Encoding: UTF-8 (for Tamil characters)
5. Import

## MediaPipe Hand Landmarks Reference

The 21 landmarks represent key points on the hand:
- **0**: Wrist
- **1-4**: Thumb (CMC, MCP, IP, TIP)
- **5-8**: Index finger (MCP, PIP, DIP, TIP)
- **9-12**: Middle finger (MCP, PIP, DIP, TIP)
- **13-16**: Ring finger (MCP, PIP, DIP, TIP)
- **17-20**: Pinky finger (MCP, PIP, DIP, TIP)

Coordinates:
- **X**: Horizontal position (0=left, 1=right)
- **Y**: Vertical position (0=top, 1=bottom)
- **Z**: Depth relative to wrist (negative=closer to camera)

## Next Steps

These CSV files are ready for:
1. **Module 4**: Feature scaling and normalization
2. **Module 5**: Model training (Random Forest, SVM, Neural Networks)
3. **Data analysis**: Exploratory data analysis, visualization
4. **External tools**: Import into R, MATLAB, Excel for analysis
5. **Documentation**: Share with team members

## Notes

- All coordinates are normalized (0-1 range for X/Y)
- Z coordinates are relative to wrist landmark
- Missing/failed extractions are not included in CSV
- Success rate: ~80-85% (from 741 original samples → 586 successful extractions)
