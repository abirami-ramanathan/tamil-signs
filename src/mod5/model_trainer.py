"""
Comprehensive model trainer with focus on reducing overfitting
"""
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE MODEL TRAINER - TAMIL SIGN LANGUAGE RECOGNITION")
print("=" * 80)

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("\n[1/5] LOADING AND PREPARING DATA")
        print("-" * 40)
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Combine train and test for fresh split
        X = np.vstack([data['X_train'], data['X_test']])
        y = np.concatenate([data['y_train'], data['y_test']])
        
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"  Min samples per class: {counts.min()}")
        print(f"  Max samples per class: {counts.max()}")
        print(f"  Avg samples per class: {counts.mean():.1f}")
        
        # Split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n  Training set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        
        # Apply PCA for SVM
        print("\n  Applying PCA for dimensionality reduction...")
        self.pca = PCA(n_components=0.95, random_state=42)
        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca = self.pca.transform(self.X_test)
        
        print(f"  PCA reduced to {self.X_train_pca.shape[1]} features "
              f"({self.pca.explained_variance_ratio_.sum():.1%} variance retained)")
        
        return True
    
    def train_random_forest(self):
        """Train optimized Random Forest"""
        print("\n[2/5] TRAINING OPTIMIZED RANDOM FOREST")
        print("-" * 40)
        
        # Define parameter grid
        param_grid = [
            {
                'n_estimators': [200, 300],
                'max_depth': [15, 20, 25],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 4, 6],
                'max_features': ['sqrt', 'log2', 0.5]
            }
        ]
        
        # Base model
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        print("  Training with cross-validation...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Simple manual search for speed
        best_score = 0
        best_params = None
        
        # Test a few combinations
        param_combinations = [
            {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10, 
             'min_samples_leaf': 4, 'max_features': 'sqrt'},
            {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 5, 
             'min_samples_leaf': 2, 'max_features': 'log2'},
            {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 15, 
             'min_samples_leaf': 6, 'max_features': 0.5}
        ]
        
        for params in param_combinations:
            rf = RandomForestClassifier(
                **params,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            scores = cross_val_score(rf, self.X_train, self.y_train, 
                                    cv=cv, scoring='f1_weighted', n_jobs=-1)
            mean_score = scores.mean()
            
            print(f"    Params: {params}")
            print(f"    CV F1 Score: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        # Train with best parameters
        print(f"\n  Best parameters: {best_params}")
        print(f"  Best CV score: {best_score:.4f}")
        
        self.models['random_forest'] = RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        self.models['random_forest'].fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = self.models['random_forest'].predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        self.results['random_forest'] = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'train_time': train_time
        }
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        print(f"  Training Time: {train_time:.2f}s")
        
        # Check for overfitting
        y_pred_train = self.models['random_forest'].predict(self.X_train)
        train_acc = accuracy_score(self.y_train, y_pred_train)
        overfitting_gap = train_acc - test_acc
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Overfitting Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.1:
            print("  ⚠️ Warning: Significant overfitting detected")
        
        return test_acc
    
    def train_svm(self):
        """Train optimized SVM"""
        print("\n[3/5] TRAINING OPTIMIZED SVM")
        print("-" * 40)
        
        # Use PCA-transformed data for SVM
        print("  Using PCA-transformed features for SVM")
        
        # LinearSVC is faster and often works well for high-dimensional data
        svm_model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=5000,
            dual=False  # Better for n_samples > n_features
        )
        
        print("  Training LinearSVC...")
        start_time = time.time()
        svm_model.fit(self.X_train_pca, self.y_train)
        train_time = time.time() - start_time
        
        self.models['svm'] = svm_model
        
        # Evaluate
        y_pred = svm_model.predict(self.X_test_pca)
        test_acc = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        self.results['svm'] = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'train_time': train_time
        }
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        print(f"  Training Time: {train_time:.2f}s")
        
        return test_acc
    
    def train_extra_trees(self):
        """Train ExtraTrees (more random than Random Forest)"""
        print("\n[4/5] TRAINING EXTRA TREES")
        print("-" * 40)
        
        # ExtraTrees often generalizes better
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        print("  Training ExtraTrees...")
        start_time = time.time()
        et_model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        self.models['extra_trees'] = et_model
        
        # Evaluate
        y_pred = et_model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        self.results['extra_trees'] = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'train_time': train_time
        }
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        print(f"  Training Time: {train_time:.2f}s")
        
        return test_acc
    
    def train_logistic_regression(self):
        """Train Logistic Regression as baseline"""
        print("\n[5/5] TRAINING LOGISTIC REGRESSION")
        print("-" * 40)
        
        # Use PCA-transformed data
        lr_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            n_jobs=-1,
            solver='saga'  # Good for large datasets
        )
        
        print("  Training Logistic Regression...")
        start_time = time.time()
        lr_model.fit(self.X_train_pca, self.y_train)
        train_time = time.time() - start_time
        
        self.models['logistic_regression'] = lr_model
        
        # Evaluate
        y_pred = lr_model.predict(self.X_test_pca)
        test_acc = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        self.results['logistic_regression'] = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'train_time': train_time
        }
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        print(f"  Training Time: {train_time:.2f}s")
        
        return test_acc
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
                'Test F1-Score': f"{metrics['test_f1']:.4f}",
                'Training Time (s)': f"{metrics['train_time']:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_accuracy'])
        self.best_model = self.models[best_model_name]
        self.best_score = self.results[best_model_name]['test_accuracy']
        
        print(f"\n★ BEST MODEL: {best_model_name.replace('_', ' ').title()}")
        print(f"  Test Accuracy: {self.best_score:.4f}")
        
        # Plot comparison
        self.plot_comparison()
        
        return df_comparison
    
    def plot_comparison(self):
        """Plot model comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        models = list(self.results.keys())
        accuracies = [self.results[m]['test_accuracy'] for m in models]
        f1_scores = [self.results[m]['test_f1'] for m in models]
        times = [self.results[m]['train_time'] for m in models]
        
        # Format model names
        model_names = [m.replace('_', ' ').title() for m in models]
        
        # Accuracy comparison
        bars1 = axes[0].bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'])
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        bars2 = axes[1].bar(model_names, times, color=['blue', 'green', 'orange', 'red'])
        axes[1].set_ylabel('Training Time (seconds)')
        axes[1].set_title('Training Time Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def save_models(self, output_dir='output'):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            filename = output_path / f"{model_name}_model.pkl"
            joblib.dump(model, filename)
            print(f"  ✓ {model_name}: {filename}")
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(output_path / 'model_results.csv')
        
        # Save best model separately
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_accuracy'])
        best_model_path = output_path / 'best_model.pkl'
        joblib.dump(self.models[best_model_name], best_model_path)
        
        print(f"\n  ★ Best model ({best_model_name}) saved to: {best_model_path}")
    
    def generate_detailed_report(self, best_model_name):
        """Generate detailed report for best model"""
        print(f"\n" + "=" * 80)
        print(f"DETAILED REPORT FOR {best_model_name.upper()}")
        print("=" * 80)
        
        best_model = self.models[best_model_name]
        
        if best_model_name in ['random_forest', 'extra_trees']:
            # Feature importance for tree-based models
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\nTop 20 Most Important Features:")
            for i in range(min(20, len(importances))):
                print(f"  Feature {indices[i]:3d}: {importances[indices[i]]:.4f}")
            
            # Plot feature importances
            plt.figure(figsize=(12, 6))
            plt.bar(range(30), importances[indices[:30]])
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title(f'Top 30 Feature Importances - {best_model_name.title()}')
            plt.tight_layout()
            plt.savefig(f'{best_model_name}_feature_importances.png', dpi=100)
            plt.show()
        
        # Detailed classification report
        if best_model_name in ['svm', 'logistic_regression']:
            X_test_used = self.X_test_pca
        else:
            X_test_used = self.X_test
        
        y_pred = best_model.predict(X_test_used)
        
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, digits=4))
        
        # Confusion matrix (simplified)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {best_model_name.title()}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{best_model_name}_confusion_matrix.png', dpi=100)
        plt.show()

def main():
    """Main training pipeline"""
    # Initialize trainer - use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "mod4" / "output" / "scaled_dataset.pkl"
    trainer = ModelTrainer(data_path)
    
    # Load and prepare data
    trainer.load_and_prepare_data()
    
    # Train models
    trainer.train_random_forest()
    trainer.train_svm()
    trainer.train_extra_trees()
    trainer.train_logistic_regression()
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Find best model
    best_model_name = max(trainer.results.keys(), 
                         key=lambda x: trainer.results[x]['test_accuracy'])
    
    # Generate detailed report
    trainer.generate_detailed_report(best_model_name)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    # Final recommendations
    best_acc = trainer.results[best_model_name]['test_accuracy']
    print(f"\n★ FINAL RESULTS:")
    print(f"  Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"  Best Accuracy: {best_acc:.4f}")
    
    if best_acc < 0.5:
        print(f"\n⚠️  RECOMMENDATIONS FOR IMPROVEMENT:")
        print(f"  1. Collect more data (aim for 50+ samples per class)")
        print(f"  2. Improve feature extraction (check landmark quality)")
        print(f"  3. Try data augmentation techniques")
        print(f"  4. Consider simpler model architecture")
        print(f"  5. Check for class imbalance issues")
    elif best_acc < 0.7:
        print(f"\n✓ Decent performance achieved. For further improvement:")
        print(f"  1. Hyperparameter tuning with GridSearchCV")
        print(f"  2. Ensemble methods (voting/stacking classifiers)")
        print(f"  3. Feature engineering")
        print(f"  4. Try neural networks (MLP, CNN)")
    else:
        print(f"\n🎉 Excellent performance! Consider:")
        print(f"  1. Deploying the model")
        print(f"  2. Creating a real-time recognition system")
        print(f"  3. Optimizing for speed if needed")

if __name__ == "__main__":
    main()