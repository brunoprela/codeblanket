/**
 * Imbalanced Data Handling Section
 */

export const imbalanceddataSection = {
  id: 'imbalanced-data',
  title: 'Imbalanced Data Handling',
  content: `# Imbalanced Data Handling

## Introduction

Imbalanced datasets have significantly more examples of one class than others. This is common in real-world problems where positive events are rare:

**Examples**:
- Fraud detection (0.1% fraud)
- Disease diagnosis (5% diseased)
- Click-through rate (2% clicks)
- Defect detection (1% defective)
- Customer churn (10% churn)

**Problem**: Models trained on imbalanced data often predict only the majority class, achieving high accuracy but failing on the minority class (which is usually more important).

**Accuracy Paradox**: 99% accuracy might be useless if all predictions are "not fraud" when detecting fraud is the goal!

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.95, 0.05],  # 95% class 0, 5% class 1
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Class distribution:")
print(f"Class 0: {np.sum (y_train == 0)} ({np.sum (y_train == 0)/len (y_train)*100:.1f}%)")
print(f"Class 1: {np.sum (y_train == 1)} ({np.sum (y_train == 1)/len (y_train)*100:.1f}%)")

# Naive model
naive_clf = LogisticRegression (random_state=42, max_iter=1000)
naive_clf.fit(X_train, y_train)

y_pred = naive_clf.predict(X_test)

print("\\n" + "="*60)
print("NAIVE MODEL ON IMBALANCED DATA")
print("="*60)
print(f"Accuracy: {naive_clf.score(X_test, y_test):.4f}")
print("\\nClassification Report:")
print(classification_report (y_test, y_pred, target_names=['Class 0', 'Class 1']))
print("\\nConfusion Matrix:")
print(confusion_matrix (y_test, y_pred))

print("\\nNotice: High accuracy but poor recall for minority class!")
\`\`\`

## Evaluation Metrics for Imbalanced Data

**Don't use accuracy!** Use metrics that focus on minority class:

### Confusion Matrix

\\[
\\begin{bmatrix}
TN & FP \\\\
FN & TP
\\end{bmatrix}
\\]

- **True Positive (TP)**: Correctly predicted positive
- **False Positive (FP)**: Incorrectly predicted positive (Type I error)
- **True Negative (TN)**: Correctly predicted negative
- **False Negative (FN)**: Incorrectly predicted negative (Type II error)

### Key Metrics

**Precision**: Of predicted positives, how many are correct?
\\[ Precision = \\frac{TP}{TP + FP} \\]

**Recall (Sensitivity)**: Of actual positives, how many did we find?
\\[ Recall = \\frac{TP}{TP + FN} \\]

**F1-Score**: Harmonic mean of precision and recall
\\[ F1 = 2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall} \\]

**ROC AUC**: Area Under ROC Curve (TPR vs FPR)
**PR AUC**: Area Under Precision-Recall Curve (better for imbalanced)

\`\`\`python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc

# Calculate metrics
precision = precision_score (y_test, y_pred)
recall = recall_score (y_test, y_pred)
f1 = f1_score (y_test, y_pred)

y_pred_proba = naive_clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score (y_test, y_pred_proba)

# Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve (y_test, y_pred_proba)
pr_auc = auc (recall_curve, precision_curve)

print("="*60)
print("DETAILED METRICS")
print("="*60)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC: {pr_auc:.4f}")

# Visualize ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, _ = roc_curve (y_test, y_pred_proba)
axes[0].plot (fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Precision-Recall Curve
axes[1].plot (recall_curve, precision_curve, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
axes[1].axhline (y=np.sum (y_test==1)/len (y_test), color='k', linestyle='--', label='Baseline')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nFor imbalanced data, PR AUC is more informative than ROC AUC!")
\`\`\`

## Resampling Techniques

### 1. Random Undersampling

Remove majority class samples to balance classes.

\`\`\`python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler (random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"Original: {len (y_train)} samples")
print(f"After undersampling: {len (y_train_rus)} samples")
print(f"Class distribution: {np.bincount (y_train_rus)}")

# Train model
clf_rus = LogisticRegression (random_state=42, max_iter=1000)
clf_rus.fit(X_train_rus, y_train_rus)

y_pred_rus = clf_rus.predict(X_test)

print("\\nUndersampling Results:")
print(classification_report (y_test, y_pred_rus, target_names=['Class 0', 'Class 1']))
\`\`\`

### 2. Random Oversampling

Duplicate minority class samples.

\`\`\`python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler (random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print(f"\\nAfter oversampling: {len (y_train_ros)} samples")
print(f"Class distribution: {np.bincount (y_train_ros)}")

clf_ros = LogisticRegression (random_state=42, max_iter=1000)
clf_ros.fit(X_train_ros, y_train_ros)

y_pred_ros = clf_ros.predict(X_test)

print("\\nOversampling Results:")
print(classification_report (y_test, y_pred_ros, target_names=['Class 0', 'Class 1']))
\`\`\`

### 3. SMOTE (Synthetic Minority Over-sampling Technique)

Generate synthetic minority samples using k-nearest neighbors.

\`\`\`python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\\nAfter SMOTE: {len (y_train_smote)} samples")
print(f"Class distribution: {np.bincount (y_train_smote)}")

clf_smote = LogisticRegression (random_state=42, max_iter=1000)
clf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = clf_smote.predict(X_test)

print("\\nSMOTE Results:")
print(classification_report (y_test, y_pred_smote, target_names=['Class 0', 'Class 1']))
\`\`\`

### 4. ADASYN (Adaptive Synthetic Sampling)

Similar to SMOTE but focuses on harder-to-learn samples.

\`\`\`python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

clf_adasyn = LogisticRegression (random_state=42, max_iter=1000)
clf_adasyn.fit(X_train_adasyn, y_train_adasyn)

y_pred_adasyn = clf_adasyn.predict(X_test)

print("\\nADASYN Results:")
print(classification_report (y_test, y_pred_adasyn, target_names=['Class 0', 'Class 1']))
\`\`\`

### 5. Combined Approaches

SMOTEENN and SMOTETomek combine over and undersampling.

\`\`\`python
from imblearn.combine import SMOTEENN, SMOTETomek

# SMOTE + Edited Nearest Neighbors
smote_enn = SMOTEENN(random_state=42)
X_train_senn, y_train_senn = smote_enn.fit_resample(X_train, y_train)

clf_senn = LogisticRegression (random_state=42, max_iter=1000)
clf_senn.fit(X_train_senn, y_train_senn)

y_pred_senn = clf_senn.predict(X_test)

print("\\nSMOTEENN Results:")
print(classification_report (y_test, y_pred_senn, target_names=['Class 0', 'Class 1']))
\`\`\`

## Class Weights

Adjust model to penalize errors on minority class more.

\`\`\`python
# Balanced class weights
clf_balanced = LogisticRegression (class_weight='balanced', random_state=42, max_iter=1000)
clf_balanced.fit(X_train, y_train)

y_pred_balanced = clf_balanced.predict(X_test)

print("="*60)
print("CLASS WEIGHTS")
print("="*60)
print("\\nBalanced Class Weights Results:")
print(classification_report (y_test, y_pred_balanced, target_names=['Class 0', 'Class 1']))

# Custom weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique (y_train), y=y_train)
print(f"\\nComputed class weights: {dict (zip (np.unique (y_train), class_weights))}")

# Manual weights
clf_weighted = LogisticRegression (class_weight={0: 1, 1: 10}, random_state=42, max_iter=1000)
clf_weighted.fit(X_train, y_train)

y_pred_weighted = clf_weighted.predict(X_test)

print("\\nCustom Weights (1:10) Results:")
print(classification_report (y_test, y_pred_weighted, target_names=['Class 0', 'Class 1']))
\`\`\`

## Threshold Adjustment

Adjust classification threshold (default 0.5) to trade precision for recall.

\`\`\`python
# Get predicted probabilities
y_pred_proba = naive_clf.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("="*60)
print("THRESHOLD ADJUSTMENT")
print("="*60)

threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype (int)
    
    precision = precision_score (y_test, y_pred_threshold)
    recall = recall_score (y_test, y_pred_threshold)
    f1 = f1_score (y_test, y_pred_threshold)
    
    threshold_results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

df_thresholds = pd.DataFrame (threshold_results)
print(df_thresholds.to_string (index=False))

# Visualize
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (df_thresholds['Threshold'], df_thresholds['Precision'], 'o-', label='Precision', linewidth=2)
plt.plot (df_thresholds['Threshold'], df_thresholds['Recall'], 's-', label='Recall', linewidth=2)
plt.plot (df_thresholds['Threshold'], df_thresholds['F1-Score'], '^-', label='F1-Score', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Classification Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
precision_curve, recall_curve, thresholds_pr = precision_recall_curve (y_test, y_pred_proba)
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
optimal_idx = np.argmax (f1_scores)
optimal_threshold = thresholds_pr[optimal_idx]

plt.plot (thresholds_pr, precision_curve[:-1], label='Precision', linewidth=2)
plt.plot (thresholds_pr, recall_curve[:-1], label='Recall', linewidth=2)
plt.axvline (x=optimal_threshold, color='r', linestyle='--', label=f'Optimal={optimal_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Optimal Threshold by F1-Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\nOptimal threshold: {optimal_threshold:.3f}")
\`\`\`

## Ensemble Methods for Imbalanced Data

### Balanced Random Forest

\`\`\`python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier (n_estimators=100, random_state=42)
brf.fit(X_train, y_train)

y_pred_brf = brf.predict(X_test)

print("="*60)
print("BALANCED RANDOM FOREST")
print("="*60)
print(classification_report (y_test, y_pred_brf, target_names=['Class 0', 'Class 1']))
\`\`\`

### Easy Ensemble

\`\`\`python
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier (n_estimators=10, random_state=42)
eec.fit(X_train, y_train)

y_pred_eec = eec.predict(X_test)

print("="*60)
print("EASY ENSEMBLE")
print("="*60)
print(classification_report (y_test, y_pred_eec, target_names=['Class 0', 'Class 1']))
\`\`\`

## Comparison of All Methods

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

methods = {
    'Naive': (LogisticRegression (random_state=42, max_iter=1000), X_train, y_train),
    'Undersampling': (LogisticRegression (random_state=42, max_iter=1000), X_train_rus, y_train_rus),
    'Oversampling': (LogisticRegression (random_state=42, max_iter=1000), X_train_ros, y_train_ros),
    'SMOTE': (LogisticRegression (random_state=42, max_iter=1000), X_train_smote, y_train_smote),
    'Class Weights': (LogisticRegression (class_weight='balanced', random_state=42, max_iter=1000), X_train, y_train),
    'Balanced RF': (BalancedRandomForestClassifier (n_estimators=100, random_state=42), X_train, y_train),
}

comparison_results = []

for name, (clf, X_tr, y_tr) in methods.items():
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_test)
    
    precision = precision_score (y_test, y_pred)
    recall = recall_score (y_test, y_pred)
    f1 = f1_score (y_test, y_pred)
    
    if hasattr (clf, 'predict_proba'):
        roc_auc = roc_auc_score (y_test, clf.predict_proba(X_test)[:, 1])
    else:
        roc_auc = 0
    
    comparison_results.append({
        'Method': name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    })

df_comparison = pd.DataFrame (comparison_results).sort_values('F1-Score', ascending=False)

print("\\n" + "="*60)
print("COMPREHENSIVE COMPARISON")
print("="*60)
print(df_comparison.to_string (index=False))

# Visualize
plt.figure (figsize=(12, 6))
x = range (len (df_comparison))
width = 0.2

plt.bar([i - 1.5*width for i in x], df_comparison['Precision'], width, label='Precision', alpha=0.8)
plt.bar([i - 0.5*width for i in x], df_comparison['Recall'], width, label='Recall', alpha=0.8)
plt.bar([i + 0.5*width for i in x], df_comparison['F1-Score'], width, label='F1-Score', alpha=0.8)
plt.bar([i + 1.5*width for i in x], df_comparison['ROC AUC'], width, label='ROC AUC', alpha=0.8)

plt.xlabel('Method')
plt.ylabel('Score')
plt.title('Comparison of Imbalanced Data Handling Techniques')
plt.xticks (x, df_comparison['Method'], rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
\`\`\`

## Best Practices

**General Guidelines**:
1. **Always use appropriate metrics** (F1, PR AUC, not accuracy)
2. **Start with class weights** (simplest, often sufficient)
3. **Try SMOTE** if class weights insufficient
4. **Adjust threshold** based on business cost of FP vs FN
5. **Use stratified splits** in train/test and CV

**Practical Decision Framework**:
- Slight imbalance (1:3 to 1:10): Class weights
- Moderate imbalance (1:10 to 1:100): SMOTE or undersampling
- Severe imbalance (>1:100): Ensemble methods, anomaly detection
- Fraud/anomaly: Consider one-class SVM, isolation forest

\`\`\`python
# Production pipeline for imbalanced data
from imblearn.pipeline import Pipeline as ImbPipeline

imbalanced_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier (class_weight='balanced', n_estimators=200, random_state=42))
])

imbalanced_pipeline.fit(X_train, y_train)

y_pred_pipeline = imbalanced_pipeline.predict(X_test)

print("="*60)
print("PRODUCTION PIPELINE")
print("="*60)
print(classification_report (y_test, y_pred_pipeline, target_names=['Class 0', 'Class 1']))
\`\`\`

## Summary

Imbalanced data requires special handling:

**Evaluation**:
- Use precision, recall, F1, PR AUC (not accuracy)
- Focus on minority class performance

**Techniques**:
- **Resampling**: Under/oversampling, SMOTE, ADASYN
- **Class weights**: Penalize minority class errors more
- **Threshold tuning**: Adjust classification threshold
- **Ensemble methods**: Balanced RF, Easy Ensemble

**Best Practices**:
- Start with class weights
- Use SMOTE for moderate imbalance
- Adjust threshold based on cost
- Always use stratified splits
- Consider business cost of FP vs FN

Next: Time series forecasting with classical methods!
`,
  codeExample: `# Complete Imbalanced Data Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Comprehensive pipeline
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"F1-Score: {f1_score (y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score (y_test, pipeline.predict_proba(X_test)[:, 1]):.4f}")
`,
};
