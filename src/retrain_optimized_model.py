"""
Retrain Random Forest model with optimized hyperparameters using GridSearchCV
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

print("="*80)
print("RETRAINING RANDOM FOREST WITH HYPERPARAMETER OPTIMIZATION")
print("="*80)

# Load dataset
dataset_path = Path(__file__).parent / "mod4" / "output" / "scaled_dataset.pkl"
print(f"\n1. Loading dataset from: {dataset_path}")

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['X_train']
X_test = dataset['X_test']
y_train = dataset['y_train']
y_test = dataset['y_test']
scaler = dataset['scaler']

print(f"   ✓ Training samples: {X_train.shape[0]}")
print(f"   ✓ Testing samples: {X_test.shape[0]}")
print(f"   ✓ Features: {X_train.shape[1]}")
print(f"   ✓ Classes: {len(np.unique(y_train))}")

# Get label mapping from old model if exists
old_model_path = Path(__file__).parent / "mod5" / "output" / "random_forest_model.pkl"
if old_model_path.exists():
    with open(old_model_path, 'rb') as f:
        old_model_data = pickle.load(f)
        label_mapping = old_model_data.get('label_mapping', {})
else:
    # Create simple label mapping
    label_mapping = {i: f"Class_{i}" for i in range(len(np.unique(y_train)))}

print(f"   ✓ Label mapping loaded: {len(label_mapping)} classes")

print(f"\n2. Setting up hyperparameter grid...")
# Comprehensive parameter grid
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [30, 50, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': ['balanced', None]
}

print("   Parameter grid:")
for param, values in param_grid.items():
    print(f"      - {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n   Total combinations to test: {total_combinations}")

# Initialize Random Forest
print(f"\n3. Initializing GridSearchCV with 3-fold cross-validation...")
rf = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

# Train with grid search
print(f"\n4. Starting grid search training...")
print("   This may take 30-60 minutes...\n")

start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"\n✓ Grid search completed in {training_time/60:.1f} minutes")

# Best parameters
print(f"\n5. Best hyperparameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   - {param}: {value}")

print(f"\n   Best cross-validation score: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# Evaluate on training set
print(f"\n6. Evaluating best model...")
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Evaluate on test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check for overfitting
if train_accuracy - test_accuracy > 0.15:
    print(f"   ⚠ Warning: Possible overfitting (gap: {(train_accuracy-test_accuracy)*100:.2f}%)")
else:
    print(f"   ✓ Good generalization (gap: {(train_accuracy-test_accuracy)*100:.2f}%)")

# Prediction probabilities analysis
print(f"\n7. Analyzing prediction confidence...")
y_test_proba = best_model.predict_proba(X_test)
max_probas = np.max(y_test_proba, axis=1)

print(f"   Mean confidence: {np.mean(max_probas):.4f} ({np.mean(max_probas)*100:.2f}%)")
print(f"   Median confidence: {np.median(max_probas):.4f} ({np.median(max_probas)*100:.2f}%)")
print(f"   Min confidence: {np.min(max_probas):.4f} ({np.min(max_probas)*100:.2f}%)")
print(f"   Max confidence: {np.max(max_probas):.4f} ({np.max(max_probas)*100:.2f}%)")

# Confidence distribution
conf_ranges = [
    (0.0, 0.2, "Very Low (0-20%)"),
    (0.2, 0.4, "Low (20-40%)"),
    (0.4, 0.6, "Medium (40-60%)"),
    (0.6, 0.8, "Good (60-80%)"),
    (0.8, 1.0, "Excellent (80-100%)")
]

print(f"\n   Confidence distribution:")
for low, high, label in conf_ranges:
    count = np.sum((max_probas >= low) & (max_probas < high))
    pct = (count / len(max_probas)) * 100
    print(f"      {label}: {count:,} samples ({pct:.1f}%)")

# Feature importances
print(f"\n8. Top 10 most important features:")
feature_importances = best_model.feature_importances_
feature_names = [f"Landmark_{i//3}_{['x','y','z'][i%3]}" for i in range(63)]
top_indices = np.argsort(feature_importances)[-10:][::-1]

for idx in top_indices:
    print(f"   - {feature_names[idx]}: {feature_importances[idx]:.4f}")

# Save optimized model
output_path = Path(__file__).parent / "mod5" / "output" / "random_forest_model.pkl"
print(f"\n9. Saving optimized model to: {output_path}")

model_data = {
    'model': best_model,
    'scaler': scaler,
    'label_mapping': label_mapping,
    'best_params': grid_search.best_params_,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'mean_confidence': np.mean(max_probas),
    'cv_score': grid_search.best_score_
}

with open(output_path, 'wb') as f:
    pickle.dump(model_data, f)

file_size = output_path.stat().st_size / (1024**3)
print(f"   ✓ Model saved! Size: {file_size:.2f} GB")

# Save detailed report
report_path = Path(__file__).parent / "mod5" / "output" / "training_report_optimized.txt"
print(f"\n10. Saving detailed report to: {report_path}")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("OPTIMIZED RANDOM FOREST MODEL - TRAINING REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Training samples: {X_train.shape[0]:,}\n")
    f.write(f"Testing samples: {X_test.shape[0]:,}\n")
    f.write(f"Features: {X_train.shape[1]}\n")
    f.write(f"Classes: {len(np.unique(y_train))}\n\n")
    
    f.write("BEST HYPERPARAMETERS\n")
    f.write("-"*80 + "\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\n")
    f.write(f"\nBest CV Score: {grid_search.best_score_:.4f}\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("-"*80 + "\n")
    f.write(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
    f.write(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
    f.write(f"Overfitting Gap: {(train_accuracy-test_accuracy)*100:.2f}%\n\n")
    
    f.write("CONFIDENCE STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Mean Confidence: {np.mean(max_probas):.4f} ({np.mean(max_probas)*100:.2f}%)\n")
    f.write(f"Median Confidence: {np.median(max_probas):.4f} ({np.median(max_probas)*100:.2f}%)\n")
    f.write(f"Min Confidence: {np.min(max_probas):.4f} ({np.min(max_probas)*100:.2f}%)\n")
    f.write(f"Max Confidence: {np.max(max_probas):.4f} ({np.max(max_probas)*100:.2f}%)\n\n")
    
    f.write("CONFIDENCE DISTRIBUTION\n")
    f.write("-"*80 + "\n")
    for low, high, label in conf_ranges:
        count = np.sum((max_probas >= low) & (max_probas < high))
        pct = (count / len(max_probas)) * 100
        f.write(f"{label}: {count:,} samples ({pct:.1f}%)\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"   ✓ Report saved!")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\nKey Improvements:")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Mean Confidence: {np.mean(max_probas)*100:.2f}%")
print(f"  - Training Time: {training_time/60:.1f} minutes")
print(f"\nOptimized model ready for deployment!")
print("="*80)
