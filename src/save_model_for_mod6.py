"""
Train and save a working Random Forest model for Module 6
"""

import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from mod1 import TLFS23DatasetLoader

print("=" * 70)
print("TRAINING RANDOM FOREST MODEL FOR MODULE 6")
print("=" * 70)

# Load the scaled dataset
scaled_file = Path(r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\mod4\output\scaled_dataset.pkl")
print(f"\nLoading scaled dataset from: {scaled_file}")

with open(scaled_file, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
scaler = data['scaler']

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Features: {X_train.shape[1]}")

# Load label mapping
dataset_path = Path(r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset")
print(f"\nLoading label mapping from dataset...")
loader = TLFS23DatasetLoader(str(dataset_path))
loader.load_dataset_structure()

# Create label mapping
label_mapping = {}
for label, class_info in loader.class_paths.items():
    label_mapping[label] = class_info['tamil_char']

print(f"  Loaded {len(label_mapping)} class labels")

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nResults:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")

# Save model
output_dir = Path(__file__).parent / "mod5" / "output"
output_dir.mkdir(parents=True, exist_ok=True)

model_file = output_dir / "random_forest_model.pkl"

print(f"\nSaving model to: {model_file}")

model_data = {
    'model': rf,
    'scaler': scaler,
    'label_mapping': label_mapping
}

with open(model_file, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✓ Model saved successfully!")
print(f"  File size: {model_file.stat().st_size / (1024*1024):.2f} MB")

print("\n" + "=" * 70)
print("MODEL READY FOR MODULE 6")
print("=" * 70)
