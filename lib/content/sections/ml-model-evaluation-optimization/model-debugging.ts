/**
 * Section: Model Debugging
 * Module: Model Evaluation & Optimization
 *
 * Covers debugging techniques, error analysis, and systematic troubleshooting
 */

export const modelDebugging = {
  id: 'model-debugging',
  title: 'Model Debugging',
  content: `
# Model Debugging

## Introduction

Your model performs poorly. Now what? Debugging ML models is different from debugging code—there's no stack trace pointing to the bug. You need systematic approaches to identify and fix issues.

**Common Symptoms:**
- Poor performance (low accuracy/AUC)
- High training error (underfitting)
- Large train-test gap (overfitting)
- Model works offline but fails in production
- Performance degraded over time

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Create a problematic dataset (for demonstration)
np.random.seed(42)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_classes=2, class_sep=0.8,
    weights=[0.9, 0.1],  # Imbalanced
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Model Debugging: Systematic Troubleshooting")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {np.bincount (y)}")
print(f"Imbalance ratio: {(y==0).sum() / (y==1).sum():.1f}:1")
\`\`\`

## Step 1: Establish Baseline Performance

\`\`\`python
print("\\n" + "="*70)
print("Step 1: Establish Baseline Performance")
print("="*70)

from sklearn.dummy import DummyClassifier

# Baseline: Always predict majority class
baseline = DummyClassifier (strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)

# Baseline: Random guessing
random_baseline = DummyClassifier (strategy='uniform', random_state=42)
random_baseline.fit(X_train, y_train)
random_acc = random_baseline.score(X_test, y_test)

print(f"\\nBaseline Performance:")
print(f"  Always predict majority class: {baseline_acc:.3f} accuracy")
print(f"  Random guessing: {random_acc:.3f} accuracy")

# Train actual model
model = RandomForestClassifier (n_estimators=100, random_state=42)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\\nModel Performance:")
print(f"  Training accuracy: {train_acc:.3f}")
print(f"  Test accuracy: {test_acc:.3f}")

if test_acc < baseline_acc + 0.05:
    print("\\n⚠️  WARNING: Model barely beats baseline!")
    print("   → Model may not be learning useful patterns")
    print("   → Check: data quality, feature engineering, model complexity")
else:
    print(f"\\n✅ Model improves over baseline by {(test_acc - baseline_acc)*100:.1f}%")
\`\`\`

## Step 2: Diagnose Bias vs Variance

\`\`\`python
print("\\n" + "="*70)
print("Step 2: Diagnose Bias vs Variance")
print("="*70)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
gap = train_acc - test_acc

print(f"\\nPerformance Analysis:")
print(f"  Training accuracy: {train_acc:.3f}")
print(f"  Test accuracy: {test_acc:.3f}")
print(f"  Gap (train - test): {gap:.3f}")

print("\\nDiagnosis:")
if train_acc < 0.85 and gap < 0.05:
    diagnosis = "HIGH BIAS (Underfitting)"
    print(f"  🔴 {diagnosis}")
    print("  → Both training and test errors are high")
    print("  → Model is too simple to capture patterns")
    print("\\n  Solutions:")
    print("    • Increase model complexity (more layers, deeper trees)")
    print("    • Add more features or polynomial features")
    print("    • Reduce regularization")
    print("    • Train longer")
elif train_acc > 0.95 and gap > 0.10:
    diagnosis = "HIGH VARIANCE (Overfitting)"
    print(f"  🔴 {diagnosis}")
    print("  → Training accuracy is high but test accuracy is low")
    print("  → Model memorized training data")
    print("\\n  Solutions:")
    print("    • Add regularization (L1/L2, dropout)")
    print("    • Get more training data")
    print("    • Reduce model complexity")
    print("    • Use data augmentation")
    print("    • Early stopping")
else:
    diagnosis = "GOOD FIT"
    print(f"  ✅ {diagnosis}")
    print("  → Model is appropriately complex")
    print("  → If performance is still low, issue is elsewhere (data quality, features)")

# Learning curves
print("\\n📊 Generating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

plt.figure (figsize=(10, 6))
plt.plot (train_sizes, train_scores.mean (axis=1), label='Training score', marker='o')
plt.fill_between (train_sizes, train_scores.mean (axis=1) - train_scores.std (axis=1),
                 train_scores.mean (axis=1) + train_scores.std (axis=1), alpha=0.1)

plt.plot (train_sizes, val_scores.mean (axis=1), label='Validation score', marker='o')
plt.fill_between (train_sizes, val_scores.mean (axis=1) - val_scores.std (axis=1),
                 val_scores.mean (axis=1) + val_scores.std (axis=1), alpha=0.1)

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
print("   Saved: learning_curves.png")

print("\\n💡 Interpreting Learning Curves:")
print("  • Curves converge at high value → good fit")
print("  • Both plateau at low value → high bias (need more complexity)")
print("  • Large gap between curves → high variance (need more data/regularization)")
\`\`\`

## Step 3: Error Analysis

\`\`\`python
print("\\n" + "="*70)
print("Step 3: Error Analysis - Where Does the Model Fail?")
print("="*70)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix (y_test, y_pred)

print("\\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure (figsize=(8, 6))
sns.heatmap (cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_debug.png', dpi=150, bbox_inches='tight')
print("📊 Saved: confusion_matrix_debug.png")

# Classification report
print("\\nDetailed Classification Report:")
print(classification_report (y_test, y_pred, target_names=['Class 0', 'Class 1']))

# Identify types of errors
tn, fp, fn, tp = cm.ravel()
total = len (y_test)

print(f"\\nError Breakdown:")
print(f"  True Negatives (correct): {tn} ({tn/total*100:.1f}%)")
print(f"  True Positives (correct): {tp} ({tp/total*100:.1f}%)")
print(f"  False Positives (Type I error): {fp} ({fp/total*100:.1f}%)")
print(f"  False Negatives (Type II error): {fn} ({fn/total*100:.1f}%)")

if fp > fn:
    print("\\n⚠️  More False Positives than False Negatives")
    print("   → Model is too aggressive (predicts positive too often)")
    print("   → Consider: increasing threshold, adding precision penalty")
elif fn > fp:
    print("\\n⚠️  More False Negatives than False Positives")
    print("   → Model is too conservative (misses positive cases)")
    print("   → Consider: decreasing threshold, adding recall penalty")

# Analyze prediction confidence for errors
errors_idx = y_test != y_pred
errors_confidence = y_pred_proba[errors_idx]
correct_confidence = y_pred_proba[~errors_idx]

print(f"\\nPrediction Confidence Analysis:")
print(f"  Errors - Mean confidence: {errors_confidence.mean():.3f} (+/- {errors_confidence.std():.3f})")
print(f"  Correct - Mean confidence: {correct_confidence.mean():.3f} (+/- {correct_confidence.std():.3f})")

if errors_confidence.mean() > 0.7:
    print("\\n⚠️  Model is overconfident in its errors!")
    print("   → Calibration issue: predicted probabilities don't match reality")
    print("   → Consider: probability calibration (Platt scaling, isotonic regression)")
\`\`\`

## Step 4: Analyze Misclassified Examples

\`\`\`python
print("\\n" + "="*70)
print("Step 4: Analyze Misclassified Examples")
print("="*70)

# Create DataFrame with predictions
test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
test_df['true_label'] = y_test
test_df['predicted_label'] = y_pred
test_df['predicted_proba'] = y_pred_proba
test_df['correct'] = y_test == y_pred

# Analyze false positives
false_positives = test_df[(test_df['true_label'] == 0) & (test_df['predicted_label'] == 1)]
false_negatives = test_df[(test_df['true_label'] == 1) & (test_df['predicted_label'] == 0)]

print(f"\\nFalse Positives (n={len (false_positives)}):")
if len (false_positives) > 0:
    print("  Feature statistics (mean):")
    for col in [f'feature_{i}' for i in range (min(5, X_test.shape[1]))]:
        fp_mean = false_positives[col].mean()
        all_mean = test_df[col].mean()
        diff = fp_mean - all_mean
        print(f"    {col}: {fp_mean:.3f} (overall: {all_mean:.3f}, diff: {diff:+.3f})")

print(f"\\nFalse Negatives (n={len (false_negatives)}):")
if len (false_negatives) > 0:
    print("  Feature statistics (mean):")
    for col in [f'feature_{i}' for i in range (min(5, X_test.shape[1]))]:
        fn_mean = false_negatives[col].mean()
        all_mean = test_df[col].mean()
        diff = fn_mean - all_mean
        print(f"    {col}: {fn_mean:.3f} (overall: {all_mean:.3f}, diff: {diff:+.3f})")

# High-confidence errors (most problematic)
high_conf_errors = test_df[~test_df['correct'] & (test_df['predicted_proba'].apply (lambda x: max (x, 1-x)) > 0.8)]

print(f"\\nHigh-Confidence Errors (n={len (high_conf_errors)}):")
print("  These are the most problematic - model is confident but wrong")
if len (high_conf_errors) > 0:
    print(f"\\n  Sample of high-confidence errors:")
    sample = high_conf_errors.head(3)[['true_label', 'predicted_label', 'predicted_proba']]
    for idx, row in sample.iterrows():
        print(f"    True: {row['true_label']}, Predicted: {row['predicted_label']}, Confidence: {row['predicted_proba']:.3f}")

print("\\n💡 Action Items:")
print("  • Investigate patterns in misclassified examples")
print("  • Check for data quality issues (mislabeling, outliers)")
print("  • Consider creating features that distinguish these cases")
print("  • For high-confidence errors: may need different model architecture")
\`\`\`

## Step 5: Check Data Quality

\`\`\`python
print("\\n" + "="*70)
print("Step 5: Data Quality Checks")
print("="*70)

# Missing values
missing = pd.DataFrame(X_train).isnull().sum()
if missing.sum() > 0:
    print(f"\\n⚠️  Missing values detected:")
    print(missing[missing > 0])
    print("   → Ensure proper imputation")
else:
    print("✅ No missing values")

# Duplicate rows
duplicates = pd.DataFrame(X_train).duplicated().sum()
if duplicates > 0:
    print(f"\\n⚠️  {duplicates} duplicate rows detected")
    print("   → May cause overfitting or data leakage")
else:
    print("✅ No duplicate rows")

# Feature distributions
print("\\nFeature Distribution Checks:")
for i in range (min(5, X_train.shape[1])):
    feat = X_train[:, i]
    skew = pd.Series (feat).skew()
    print(f"  Feature {i}: mean={feat.mean():.2f}, std={feat.std():.2f}, skew={skew:.2f}")
    if abs (skew) > 2:
        print(f"    ⚠️  Highly skewed - consider log transform")

# Target distribution
print(f"\\nTarget Distribution:")
print(f"  Training: {np.bincount (y_train)}")
print(f"  Test: {np.bincount (y_test)}")

train_ratio = y_train.sum() / len (y_train)
test_ratio = y_test.sum() / len (y_test)
ratio_diff = abs (train_ratio - test_ratio)

if ratio_diff > 0.05:
    print(f"\\n⚠️  Train/test distribution mismatch: {ratio_diff:.3f}")
    print("   → May indicate poor splitting or data shift")
else:
    print("✅ Train/test distributions are similar")

# Outlier detection
print("\\nOutlier Detection (using z-score):")
from scipy import stats
z_scores = np.abs (stats.zscore(X_train))
outliers = (z_scores > 3).sum (axis=0)
total_outliers = (outliers > 0).sum()

if total_outliers > 0:
    print(f"  ⚠️  {total_outliers} features have outliers (|z-score| > 3)")
    print(f"     Consider: robust scaling, outlier removal, or robust models")
else:
    print("  ✅ No significant outliers detected")
\`\`\`

## Step 6: Feature Analysis

\`\`\`python
print("\\n" + "="*70)
print("Step 6: Feature Analysis")
print("="*70)

# Feature importance
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range (len (importances))],
    'importance': importances
}).sort_values('importance', ascending=False)

print("\\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Check for zero-importance features
zero_importance = (importances == 0).sum()
if zero_importance > 0:
    print(f"\\n⚠️  {zero_importance} features have zero importance")
    print("   → Consider removing these features")

# Feature correlation
from scipy.stats import spearmanr
print("\\nFeature Correlation Analysis:")
X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
corr_matrix = X_train_df.corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range (len (corr_matrix.columns)):
    for j in range (i+1, len (corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print(f"  ⚠️  {len (high_corr_pairs)} highly correlated feature pairs (>0.9):")
    for feat1, feat2, corr in high_corr_pairs[:3]:
        print(f"     {feat1} <-> {feat2}: {corr:.3f}")
    print("   → Consider removing redundant features")
else:
    print("  ✅ No highly correlated features")
\`\`\`

## Debugging Checklist

\`\`\`python
print("\\n" + "="*70)
print("Complete Model Debugging Checklist")
print("="*70)

checklist = {
    "Data Quality": [
        "☐ Check for missing values",
        "☐ Check for duplicate rows",
        "☐ Verify target labels are correct",
        "☐ Check for outliers",
        "☐ Verify train/test distributions match",
    ],
    "Model Performance": [
        "☐ Compare to baseline (majority class, random)",
        "☐ Check training vs test error (bias-variance)",
        "☐ Examine learning curves",
        "☐ Validate on separate holdout set",
    ],
    "Error Analysis": [
        "☐ Analyze confusion matrix",
        "☐ Review false positives and false negatives",
        "☐ Check prediction confidence distribution",
        "☐ Investigate high-confidence errors",
        "☐ Look for patterns in misclassified examples",
    ],
    "Features": [
        "☐ Check feature importance",
        "☐ Remove zero-importance features",
        "☐ Check for highly correlated features",
        "☐ Verify feature engineering makes sense",
        "☐ Check for data leakage (target in features)",
    ],
    "Hyperparameters": [
        "☐ Try different learning rates",
        "☐ Tune regularization strength",
        "☐ Adjust model complexity",
        "☐ Cross-validate hyperparameters",
    ],
    "Implementation": [
        "☐ Verify data preprocessing is correct",
        "☐ Check that transformations are fit on train only",
        "☐ Validate stratified splitting for imbalanced data",
        "☐ Ensure random seeds are set for reproducibility",
    ],
}

for category, items in checklist.items():
    print(f"\\n{category}:")
    for item in items:
        print(f"  {item}")

print("\\n" + "="*70)
print("Common Issues and Solutions")
print("="*70)

issues = {
    "Model worse than baseline": {
        "cause": "Not learning or data issues",
        "solutions": ["Check data quality", "Verify labels", "Increase model complexity"],
    },
    "High training error": {
        "cause": "Underfitting (high bias)",
        "solutions": ["Increase complexity", "Add features", "Reduce regularization"],
    },
    "Large train-test gap": {
        "cause": "Overfitting (high variance)",
        "solutions": ["Get more data", "Add regularization", "Reduce complexity"],
    },
    "Good offline, bad online": {
        "cause": "Distribution shift or data leakage",
        "solutions": ["Check for leakage", "Temporal validation", "Monitor drift"],
    },
    "Imbalanced errors": {
        "cause": "Class imbalance or wrong threshold",
        "solutions": ["Adjust threshold", "Class weights", "Resampling"],
    },
}

for issue, details in issues.items():
    print(f"\\n{issue}:")
    print(f"  Cause: {details['cause']}")
    print(f"  Solutions:")
    for sol in details['solutions']:
        print(f"    • {sol}")
\`\`\`

## Key Takeaways

1. **Debugging is systematic**: Follow a checklist, don't just try random things
2. **Start with baselines**: Know if your model is actually learning
3. **Diagnose bias vs variance**: Determines which direction to go
4. **Error analysis is crucial**: Understand where and why the model fails
5. **Check data quality**: Garbage in, garbage out
6. **Feature importance**: Identify which features drive predictions
7. **Learning curves**: Visualize if you need more data or different complexity
8. **Document findings**: Keep track of what you've tried

**Debugging Workflow:**
1. Establish baseline performance
2. Diagnose bias vs variance
3. Analyze error patterns
4. Check data quality
5. Review features
6. Tune hyperparameters
7. Iterate based on findings

Remember: Debugging ML models requires patience and systematic investigation. Most issues are data-related, not algorithm-related!
`,
};
