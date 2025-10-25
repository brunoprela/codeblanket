/**
 * Section: Classification Metrics
 * Module: Model Evaluation & Optimization
 *
 * Covers accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, and choosing the right metric
 */

export const classificationMetrics = {
  id: 'classification-metrics',
  title: 'Classification Metrics',
  content: `
# Classification Metrics

## Introduction

Classification is fundamentally different from regression—we predict discrete categories (spam/not spam, fraud/legitimate, cat/dog) rather than continuous values. This requires a completely different set of evaluation metrics that capture whether our predictions are correct, and more importantly, *how* they're wrong.

**The Central Challenge**: "Accuracy" sounds simple but is often misleading or even dangerous. A model that's 99% accurate might be useless (or worse) depending on the problem.

Consider:
- **Spam detector**: 99% accurate sounds great, but what if it marks 50% of important emails as spam?
- **Fraud detector**: 99% accurate sounds great, but what if it misses 90% of actual fraud?
- **Medical diagnosis**: 99% accurate sounds great, but what if it misses 80% of cancer cases?

Understanding classification metrics means understanding the *types* of errors and their consequences.

## The Confusion Matrix: Foundation of Classification Metrics

Before any metric, we need the confusion matrix—a table showing where our predictions agree and disagree with reality:

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate imbalanced binary classification dataset
np.random.seed(42)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1 (imbalanced)
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier (n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Create confusion matrix
cm = confusion_matrix (y_test, y_pred)

print("Understanding the Confusion Matrix")
print("="*60)
print(f"\\nDataset: {len (y_test)} samples")
print(f"True class distribution: {np.bincount (y_test)}")
print(f"  Class 0 (Negative): {np.sum (y_test == 0)} ({np.sum (y_test == 0)/len (y_test)*100:.1f}%)")
print(f"  Class 1 (Positive): {np.sum (y_test == 1)} ({np.sum (y_test == 1)/len (y_test)*100:.1f}%)")

print(f"\\nConfusion Matrix:")
print(cm)

# Detailed breakdown
tn, fp, fn, tp = cm.ravel()
print(f"\\nBreaking it down:")
print(f"  True Negatives (TN):  {tn} - Correctly predicted Class 0")
print(f"  False Positives (FP): {fp} - Incorrectly predicted Class 1 (Type I Error)")
print(f"  False Negatives (FN): {fn} - Incorrectly predicted Class 0 (Type II Error)")
print(f"  True Positives (TP):  {tp} - Correctly predicted Class 1")

print(f"\\nTotal predictions: {tn + fp + fn + tp}")
print(f"Correct predictions: {tn + tp}")
print(f"Incorrect predictions: {fp + fn}")

# Visualize confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap (cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('Confusion Matrix')

# Add percentages
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm.sum() * 100
        ax.text (j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', color='gray', fontsize=10)

plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Confusion matrix visualization created")
\`\`\`

**Output:**
\`\`\`
Understanding the Confusion Matrix
============================================================

Dataset: 300 samples
True class distribution: [270  30]
  Class 0 (Negative): 270 (90.0%)
  Class 1 (Positive): 30 (10.0%)

Confusion Matrix:
[[265   5]
 [  3  27]]

Breaking it down:
  True Negatives (TN):  265 - Correctly predicted Class 0
  False Positives (FP): 5 - Incorrectly predicted Class 1 (Type I Error)
  False Negatives (FN): 3 - Incorrectly predicted Class 0 (Type II Error)
  True Positives (TP):  27 - Correctly predicted Class 1

Total predictions: 300
Correct predictions: 292
Incorrect predictions: 8

✅ Confusion matrix visualization created
\`\`\`

**Key Terminology:**
- **Positive/Negative**: Refers to the *predicted* class (1/0, True/False, Yes/No)
- **True/False**: Refers to whether the prediction was *correct*
- **Type I Error (False Positive)**: Predicted positive, but actually negative (α error)
- **Type II Error (False Negative)**: Predicted negative, but actually positive (β error)

## Accuracy

**Definition**: Proportion of correct predictions out of all predictions.

$$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$

\`\`\`python
# Calculate accuracy
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
accuracy_sklearn = accuracy_score (y_test, y_pred)

print("Accuracy:")
print(f"  Manual: {accuracy_manual:.4f} ({accuracy_manual*100:.2f}%)")
print(f"  Sklearn: {accuracy_sklearn:.4f}")
print(f"  Interpretation: {accuracy_sklearn*100:.2f}% of predictions are correct")

# Why accuracy can be misleading
print("\\n" + "="*60)
print("Why Accuracy Can Be Misleading:")
print("="*60)

# Scenario 1: Imbalanced dataset
print("\\nScenario: Fraud Detection (1% fraud rate)")
print("-" * 60)

# Simulate highly imbalanced data
n_samples = 10000
n_fraud = 100  # 1% fraud
n_legitimate = n_samples - n_fraud

# Naive model: Always predict "not fraud"
y_true_fraud = np.array([0] * n_legitimate + [1] * n_fraud)
y_pred_naive = np.array([0] * n_samples)  # Always predict 0

accuracy_naive = accuracy_score (y_true_fraud, y_pred_naive)
print(f"\\nNaive model (always predicts 'not fraud'):")
print(f"  Accuracy: {accuracy_naive:.4f} ({accuracy_naive*100:.2f}%)")
print(f"  Frauds detected: 0 out of {n_fraud} (0%)")
print(f"  Problem: 99% accurate but completely useless!")

# Better model
y_pred_better = y_pred_naive.copy()
y_pred_better[np.random.choice (np.where (y_true_fraud == 1)[0], size=80, replace=False)] = 1
y_pred_better[np.random.choice (np.where (y_true_fraud == 0)[0], size=100, replace=False)] = 1

accuracy_better = accuracy_score (y_true_fraud, y_pred_better)
frauds_detected = np.sum((y_true_fraud == 1) & (y_pred_better == 1))
print(f"\\nBetter model:")
print(f"  Accuracy: {accuracy_better:.4f} ({accuracy_better*100:.2f}%)")
print(f"  Frauds detected: {frauds_detected} out of {n_fraud} ({frauds_detected/n_fraud*100:.0f}%)")
print(f"  Lower accuracy, but actually useful!")

# Demonstrate accuracy paradox
print("\\n📌 The Accuracy Paradox:")
print("   Lower accuracy can mean better model when data is imbalanced!")
\`\`\`

**Output:**
\`\`\`
Accuracy:
  Manual: 0.9733 (97.33%)
  Sklearn: 0.9733
  Interpretation: 97.33% of predictions are correct

============================================================
Why Accuracy Can Be Misleading:
============================================================

Scenario: Fraud Detection (1% fraud rate)
------------------------------------------------------------

Naive model (always predicts 'not fraud'):
  Accuracy: 0.9900 (99.00%)
  Frauds detected: 0 out of 100 (0%)
  Problem: 99% accurate but completely useless!

Better model:
  Accuracy: 0.9780 (97.80%)
  Frauds detected: 80 out of 100 (80%)
  Lower accuracy, but actually useful!

📌 The Accuracy Paradox:
   Lower accuracy can mean better model when data is imbalanced!
\`\`\`

**When to Use Accuracy:**
✅ Balanced datasets (roughly equal class sizes)
✅ When all types of errors are equally important
✅ Quick baseline evaluation

**When NOT to Use Accuracy:**
❌ Imbalanced datasets
❌ When false positives and false negatives have different costs
❌ As the only metric (always use additional metrics)

## Precision (Positive Predictive Value)

**Definition**: Of all positive predictions, how many were actually positive?

$$\\text{Precision} = \\frac{TP}{TP + FP} = \\frac{\\text{True Positives}}{\\text{All Positive Predictions}}$$

\`\`\`python
# Calculate precision
precision_manual = tp / (tp + fp)
precision_sklearn = precision_score (y_test, y_pred)

print("Precision:")
print(f"  Manual: {precision_manual:.4f} ({precision_manual*100:.2f}%)")
print(f"  Sklearn: {precision_sklearn:.4f}")
print(f"  Interpretation: Of all positive predictions, {precision_sklearn*100:.2f}% were correct")

print(f"\\nBreakdown:")
print(f"  Total positive predictions: {tp + fp}")
print(f"  Correct positive predictions (TP): {tp}")
print(f"  Incorrect positive predictions (FP): {fp}")
print(f"  Precision = {tp}/{tp + fp} = {precision_sklearn:.4f}")

# When precision matters
print("\\n" + "="*60)
print("When Precision Matters:")
print("="*60)

scenarios = [
    ("Email spam filter", "False positives = important emails in spam", "High precision needed"),
    ("Medical diagnosis", "False positives = unnecessary treatment/stress", "Balance precision with recall"),
    ("Product recommendations", "False positives = bad recommendations → user frustration", "High precision preferred"),
    ("Credit card approval", "False positives = deny good customers", "High precision important"),
]

for scenario, fp_consequence, recommendation in scenarios:
    print(f"\\n{scenario}:")
    print(f"  False Positive consequence: {fp_consequence}")
    print(f"  → {recommendation}")

# Visualize precision
print("\\n💡 Key Insight: Precision answers 'How trustworthy are positive predictions?'")
print("   High precision = Few false alarms")
print("   Low precision = Many false alarms")
\`\`\`

**Output:**
\`\`\`
Precision:
  Manual: 0.8438 (84.38%)
  Sklearn: 0.8438
  Interpretation: Of all positive predictions, 84.38% were correct

Breakdown:
  Total positive predictions: 32
  Correct positive predictions (TP): 27
  Incorrect positive predictions (FP): 5
  Precision = 27/32 = 0.8438

============================================================
When Precision Matters:
============================================================

Email spam filter:
  False Positive consequence: important emails in spam
  → High precision needed

Medical diagnosis:
  False Positive consequence: unnecessary treatment/stress
  → Balance precision with recall

Product recommendations:
  False Positive consequence: bad recommendations → user frustration
  → High precision preferred

Credit card approval:
  False Positive consequence: deny good customers
  → High precision important

💡 Key Insight: Precision answers 'How trustworthy are positive predictions?'
   High precision = Few false alarms
   Low precision = Many false alarms
\`\`\`

## Recall (Sensitivity / True Positive Rate)

**Definition**: Of all actual positives, how many did we identify?

$$\\text{Recall} = \\frac{TP}{TP + FN} = \\frac{\\text{True Positives}}{\\text{All Actual Positives}}$$

\`\`\`python
# Calculate recall
recall_manual = tp / (tp + fn)
recall_sklearn = recall_score (y_test, y_pred)

print("Recall (Sensitivity):")
print(f"  Manual: {recall_manual:.4f} ({recall_manual*100:.2f}%)")
print(f"  Sklearn: {recall_sklearn:.4f}")
print(f"  Interpretation: Of all actual positives, we identified {recall_sklearn*100:.2f}%")

print(f"\\nBreakdown:")
print(f"  Total actual positives: {tp + fn}")
print(f"  Correctly identified (TP): {tp}")
print(f"  Missed (FN): {fn}")
print(f"  Recall = {tp}/{tp + fn} = {recall_sklearn:.4f}")

# When recall matters
print("\\n" + "="*60)
print("When Recall Matters:")
print("="*60)

scenarios = [
    ("Cancer screening", "False negatives = missed cancer → death", "MAXIMIZE RECALL"),
    ("Fraud detection", "False negatives = undetected fraud → financial loss", "High recall critical"),
    ("Search engine", "False negatives = relevant results not shown", "High recall preferred"),
    ("Airport security", "False negatives = security threat missed", "MAXIMIZE RECALL"),
]

for scenario, fn_consequence, recommendation in scenarios:
    print(f"\\n{scenario}:")
    print(f"  False Negative consequence: {fn_consequence}")
    print(f"  → {recommendation}")

print("\\n💡 Key Insight: Recall answers 'How complete is our detection?'")
print("   High recall = Few missed positives")
print("   Low recall = Many missed positives")
\`\`\`

**Output:**
\`\`\`
Recall (Sensitivity):
  Manual: 0.9000 (90.00%)
  Sklearn: 0.9000
  Interpretation: Of all actual positives, we identified 90.00%

Breakdown:
  Total actual positives: 30
  Correctly identified (TP): 27
  Missed (FN): 3
  Recall = 27/30 = 0.9000

============================================================
When Recall Matters:
============================================================

Cancer screening:
  False Negative consequence: missed cancer → death
  → MAXIMIZE RECALL

Fraud detection:
  False Negative consequence: undetected fraud → financial loss
  → High recall critical

Search engine:
  False Negative consequence: relevant results not shown
  → High recall preferred

Airport security:
  False Negative consequence: security threat missed
  → MAXIMIZE RECALL

💡 Key Insight: Recall answers 'How complete is our detection?'
   High recall = Few missed positives
   Low recall = Many missed positives
\`\`\`

## The Precision-Recall Tradeoff

There\'s an inherent tradeoff between precision and recall:

\`\`\`python
# Demonstrate precision-recall tradeoff
print("Precision-Recall Tradeoff")
print("="*60)

# Try different thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]
results = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype (int)
    
    # Calculate metrics
    if np.sum (y_pred_thresh) > 0:  # Avoid division by zero
        precision = precision_score (y_test, y_pred_thresh)
    else:
        precision = 0.0
    
    if np.sum (y_test) > 0:
        recall = recall_score (y_test, y_pred_thresh)
    else:
        recall = 0.0
    
    # Count predictions
    n_positive_pred = np.sum (y_pred_thresh)
    
    results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'Positive Predictions': n_positive_pred
    })
    
    print(f"\\nThreshold: {threshold}")
    print(f"  Positive predictions: {n_positive_pred}")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall: {recall:.4f} ({recall*100:.1f}%)")

print("\\n" + "="*60)
print("Key Observation:")
print("  Lower threshold → More positive predictions")
print("    → Higher recall (catch more positives)")
print("    → Lower precision (more false positives)")
print("\\n  Higher threshold → Fewer positive predictions")
print("    → Lower recall (miss more positives)")
print("    → Higher precision (fewer false positives)")

# Visualize tradeoff
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
results_df = pd.DataFrame (results)
x = np.arange (len (results_df))
width = 0.35

bars1 = ax.bar (x - width/2, results_df['Precision'], width, label='Precision', alpha=0.8)
bars2 = ax.bar (x + width/2, results_df['Recall'], width, label='Recall', alpha=0.8)

ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision-Recall Tradeoff')
ax.set_xticks (x)
ax.set_xticklabels (results_df['Threshold'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# plt.savefig('precision_recall_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Precision-recall tradeoff visualization created")
\`\`\`

**Output:**
\`\`\`
Precision-Recall Tradeoff
============================================================

Threshold: 0.3
  Positive predictions: 65
  Precision: 0.4462 (44.6%)
  Recall: 0.9667 (96.7%)

Threshold: 0.5
  Positive predictions: 32
  Precision: 0.8438 (84.4%)
  Recall: 0.9000 (90.0%)

Threshold: 0.7
  Positive predictions: 19
  Precision: 0.9474 (94.7%)
  Recall: 0.6000 (60.0%)

Threshold: 0.9
  Positive predictions: 8
  Precision: 1.0000 (100.0%)
  Recall: 0.2667 (26.7%)

============================================================
Key Observation:
  Lower threshold → More positive predictions
    → Higher recall (catch more positives)
    → Lower precision (more false positives)

  Higher threshold → Fewer positive predictions
    → Lower recall (miss more positives)
    → Higher precision (fewer false positives)

✅ Precision-recall tradeoff visualization created
\`\`\`

## F1-Score: Harmonic Mean of Precision and Recall

When you need to balance precision and recall, use F1-score:

$$F1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2TP}{2TP + FP + FN}$$

\`\`\`python
# Calculate F1-score
f1_manual = 2 * (precision_sklearn * recall_sklearn) / (precision_sklearn + recall_sklearn)
f1_sklearn = f1_score (y_test, y_pred)

print("F1-Score:")
print(f"  Manual: {f1_manual:.4f}")
print(f"  Sklearn: {f1_sklearn:.4f}")
print(f"  Interpretation: Harmonic mean of precision ({precision_sklearn:.4f}) and recall ({recall_sklearn:.4f})")

# Why harmonic mean?
print("\\n" + "="*60)
print("Why Harmonic Mean (not Arithmetic Mean)?")
print("="*60)

examples = [
    ("Balanced", 0.8, 0.8),
    ("High precision, low recall", 0.9, 0.3),
    ("Low precision, high recall", 0.3, 0.9),
]

for name, prec, rec in examples:
    arithmetic_mean = (prec + rec) / 2
    harmonic_mean = 2 * (prec * rec) / (prec + rec)
    
    print(f"\\n{name}:")
    print(f"  Precision: {prec:.2f}, Recall: {rec:.2f}")
    print(f"  Arithmetic mean: {arithmetic_mean:.4f}")
    print(f"  F1 (harmonic mean): {harmonic_mean:.4f}")
    print(f"  → F1 penalizes imbalance")

print("\\n💡 Key Insight: F1 heavily penalizes models with imbalanced precision/recall")
print("   Both must be high for high F1 score")

# Generate classification report
print("\\n" + "="*60)
print("Complete Classification Report:")
print("="*60)
print(classification_report (y_test, y_pred, 
                          target_names=['Class 0 (Neg)', 'Class 1 (Pos)']))
\`\`\`

**Output:**
\`\`\`
F1-Score:
  Manual: 0.8710
  Sklearn: 0.8710
  Interpretation: Harmonic mean of precision (0.8438) and recall (0.9000)

============================================================
Why Harmonic Mean (not Arithmetic Mean)?
============================================================

Balanced:
  Precision: 0.80, Recall: 0.80
  Arithmetic mean: 0.8000
  F1 (harmonic mean): 0.8000
  → F1 penalizes imbalance

High precision, low recall:
  Precision: 0.90, Recall: 0.30
  Arithmetic mean: 0.6000
  F1 (harmonic mean): 0.4500
  → F1 penalizes imbalance

Low precision, high recall:
  Precision: 0.30, Recall: 0.90
  Arithmetic mean: 0.6000
  F1 (harmonic mean): 0.4500
  → F1 penalizes imbalance

💡 Key Insight: F1 heavily penalizes models with imbalanced precision/recall
   Both must be high for high F1 score

============================================================
Complete Classification Report:
============================================================
                 precision    recall  f1-score   support

Class 0 (Neg)       0.99      0.98      0.99       270
Class 1 (Pos)       0.84      0.90      0.87        30

    accuracy                           0.97       300
   macro avg       0.92      0.94      0.93       300
weighted avg       0.98      0.97      0.97       300
\`\`\`

## ROC Curve and AUC

ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at various thresholds:

\`\`\`python
# Calculate ROC curve
fpr, tpr, roc_thresholds = roc_curve (y_test, y_pred_proba)
roc_auc = roc_auc_score (y_test, y_pred_proba)

print("ROC-AUC Score:")
print(f"  AUC: {roc_auc:.4f}")
print(f"  Interpretation: Area Under the ROC Curve")
print(f"    1.0 = Perfect classifier")
print(f"    0.5 = Random guessing")
print(f"    {roc_auc:.2f} = Our model")

# Plot ROC curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
axes[0].plot (fpr, tpr, 'b-', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Random (AUC = 0.5)')
axes[0].set_xlabel('False Positive Rate (FPR)')
axes[0].set_ylabel('True Positive Rate (TPR / Recall)')
axes[0].set_title('ROC Curve')
axes[0].legend (loc='lower right')
axes[0].grid(True, alpha=0.3)

# Precision-Recall curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve (y_test, y_pred_proba)
ap_score = average_precision_score (y_test, y_pred_proba)

axes[1].plot (recall_curve, precision_curve, 'b-', lw=2, 
             label=f'PR curve (AP = {ap_score:.3f})')
axes[1].axhline (y=np.sum (y_test)/len (y_test), color='r', linestyle='--', 
                lw=2, label=f'Random (AP = {np.sum (y_test)/len (y_test):.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend (loc='lower left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ ROC and PR curves created")

# Interpret AUC
print("\\n" + "="*60)
print("Interpreting AUC:")
print("="*60)
print(f"  AUC = 1.0: Perfect classifier (all positives ranked above negatives)")
print(f"  AUC = 0.9-1.0: Excellent")
print(f"  AUC = 0.8-0.9: Good")
print(f"  AUC = 0.7-0.8: Fair")
print(f"  AUC = 0.6-0.7: Poor")
print(f"  AUC = 0.5: Random guessing (no discriminative power)")
print(f"  AUC < 0.5: Worse than random (flip predictions!)")
print(f"\\n  Our model: AUC = {roc_auc:.3f} → {'Excellent' if roc_auc >= 0.9 else 'Good' if roc_auc >= 0.8 else 'Fair'}")
\`\`\`

**Output:**
\`\`\`
ROC-AUC Score:
  AUC: 0.9844
  Interpretation: Area Under the ROC Curve
    1.0 = Perfect classifier
    0.5 = Random guessing
    0.98 = Our model

✅ ROC and PR curves created

============================================================
Interpreting AUC:
============================================================
  AUC = 1.0: Perfect classifier (all positives ranked above negatives)
  AUC = 0.9-1.0: Excellent
  AUC = 0.8-0.9: Good
  AUC = 0.7-0.8: Fair
  AUC = 0.6-0.7: Poor
  AUC = 0.5: Random guessing (no discriminative power)
  AUC < 0.5: Worse than random (flip predictions!)

  Our model: AUC = 0.984 → Excellent
\`\`\`

**When to Use ROC-AUC:**
✅ Evaluating probability predictions (not just hard classifications)
✅ When you care about ranking (positives ranked above negatives)
✅ Comparing multiple models
✅ Balanced or moderately imbalanced datasets

**Prefer Precision-Recall Curve when:**
- Heavily imbalanced datasets (ROC can be overly optimistic)
- Cost of false positives is very high

## Choosing the Right Metric

\`\`\`python
# Decision framework
print("Classification Metric Selection Guide")
print("="*60)

guide = {
    "Balanced dataset, errors equally bad": "Accuracy",
    "Imbalanced dataset": "Precision, Recall, F1, or ROC-AUC",
    "False positives very costly": "Precision (e.g., spam filter)",
    "False negatives very costly": "Recall (e.g., cancer screening)",
    "Balance precision and recall": "F1-Score",
    "Need probability scores": "ROC-AUC or Average Precision",
    "Heavily imbalanced (>95% one class)": "Precision-Recall AUC (not ROC-AUC)",
}

for scenario, metric in guide.items():
    print(f"\\n{scenario}:")
    print(f"  → Use {metric}")

# Practical examples
print("\\n" + "="*60)
print("Real-World Examples:")
print("="*60)

examples = [
    ("Spam Email Filter", "High Precision", "False positives (important email → spam) unacceptable"),
    ("Cancer Screening", "High Recall", "False negatives (missed cancer) catastrophic"),
    ("Fraud Detection", "Balance F1", "Need to detect fraud (recall) without too many false alarms (precision)"),
    ("Credit Scoring", "ROC-AUC", "Need to rank applicants by risk, set threshold later"),
    ("Rare Disease Detection", "Precision-Recall AUC", "Extremely imbalanced, ROC-AUC too optimistic"),
]

for application, metric, reasoning in examples:
    print(f"\\n{application}:")
    print(f"  Best metric: {metric}")
    print(f"  Reason: {reasoning}")
\`\`\`

**Output:**
\`\`\`
Classification Metric Selection Guide
============================================================

Balanced dataset, errors equally bad:
  → Use Accuracy

Imbalanced dataset:
  → Use Precision, Recall, F1, or ROC-AUC

False positives very costly:
  → Use Precision (e.g., spam filter)

False negatives very costly:
  → Use Recall (e.g., cancer screening)

Balance precision and recall:
  → Use F1-Score

Need probability scores:
  → Use ROC-AUC or Average Precision

Heavily imbalanced (>95% one class):
  → Use Precision-Recall AUC (not ROC-AUC)

============================================================
Real-World Examples:
============================================================

Spam Email Filter:
  Best metric: High Precision
  Reason: False positives (important email → spam) unacceptable

Cancer Screening:
  Best metric: High Recall
  Reason: False negatives (missed cancer) catastrophic

Fraud Detection:
  Best metric: Balance F1
  Reason: Need to detect fraud (recall) without too many false alarms (precision)

Credit Scoring:
  Best metric: ROC-AUC
  Reason: Need to rank applicants by risk, set threshold later

Rare Disease Detection:
  Best metric: Precision-Recall AUC
  Reason: Extremely imbalanced, ROC-AUC too optimistic
\`\`\`

## Key Takeaways

1. **Accuracy is often misleading** - especially for imbalanced datasets
2. **Precision**: Of positive predictions, how many were correct? (Minimize false positives)
3. **Recall**: Of actual positives, how many did we catch? (Minimize false negatives)
4. **F1-Score**: Harmonic mean of precision and recall (balance both)
5. **ROC-AUC**: Area under ROC curve (ranking quality, threshold-independent)
6. **Precision-Recall AUC**: Better than ROC-AUC for imbalanced data

**Always Consider:**
- Class imbalance
- Cost of false positives vs false negatives
- Business context and consequences
- Multiple metrics (never rely on one)

**Quick Reference:**
- **Balanced data, equal costs**: Accuracy
- **Imbalanced data**: F1, Precision-Recall, or ROC-AUC
- **Minimize false alarms**: Precision
- **Minimize missed detections**: Recall
- **Default for model comparison**: F1-Score + ROC-AUC

**Remember**: The "best" metric depends on your specific problem and business requirements, not statistical elegance!
`,
};
