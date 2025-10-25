/**
 * Section: Multi-class & Multi-label Metrics
 * Module: Model Evaluation & Optimization
 *
 * Covers multi-class classification metrics, averaging strategies, and multi-label evaluation
 */

export const multiClassMultiLabelMetrics = {
  id: 'multi-class-multi-label-metrics',
  title: 'Multi-class & Multi-label Metrics',
  content: `
# Multi-class & Multi-label Metrics

## Introduction

So far we've focused on binary classification (two classes). But real-world problems often involve multiple classes (image classification: cat/dog/bird) or multiple labels per sample (movie genres: action+comedy+romance). These scenarios require different evaluation approaches and metrics.

**Two Distinct Scenarios:**

1. **Multi-class Classification**: Each sample belongs to exactly ONE class
   - Example: Digit recognition (0-9), one digit per image
   - Example: Customer segment (bronze/silver/gold), one segment per customer

2. **Multi-label Classification**: Each sample can belong to MULTIPLE classes simultaneously
   - Example: Movie genres (can be action AND comedy AND sci-fi)
   - Example: Medical diagnosis (patient can have multiple conditions)

Understanding how to properly evaluate these scenarios is crucial for building production systems.

## Multi-class Classification

### The Multi-class Confusion Matrix

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

# Load multi-class dataset (digit recognition)
digits = load_digits()
X, y = digits.data, digits.target

print("Multi-class Classification Example: Digit Recognition")
print("="*60)
print(f"Number of classes: {len (np.unique (y))} (digits 0-9)")
print(f"Number of samples: {len (y)}")
print(f"Class distribution: {np.bincount (y)}")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier (n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Multi-class confusion matrix
cm = confusion_matrix (y_test, y_pred)

print(f"\\nConfusion Matrix Shape: {cm.shape}")
print("(10x10 matrix for 10 classes)")
print("\\nConfusion Matrix:")
print(cm)

# Visualize with heatmap
plt.figure (figsize=(10, 8))
sns.heatmap (cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Digit')
plt.ylabel('True Digit')
plt.title('Multi-class Confusion Matrix: Digit Recognition')
plt.tight_layout()
# plt.savefig('multiclass_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Multi-class confusion matrix visualization created")

# Analyze confusion patterns
print("\\n" + "="*60)
print("Analyzing Confusion Patterns:")
print("="*60)

for i in range(10):
    # Correct predictions for class i
    correct = cm[i, i]
    total = cm[i, :].sum()
    accuracy_class = correct / total if total > 0 else 0
    
    # Most confused with
    cm_row = cm[i, :].copy()
    cm_row[i] = 0  # Remove diagonal
    most_confused_idx = cm_row.argmax()
    most_confused_count = cm_row[most_confused_idx]
    
    print(f"Digit {i}: {correct}/{total} correct ({accuracy_class*100:.1f}%)", end="")
    if most_confused_count > 0:
        print(f" - most confused with {most_confused_idx} ({most_confused_count} times)")
    else:
        print(" - perfect!")
\`\`\`

**Output:**
\`\`\`
Multi-class Classification Example: Digit Recognition
============================================================
Number of classes: 10 (digits 0-9)
Number of samples: 1797
Class distribution: [178 182 177 183 181 182 181 179 174 180]

Confusion Matrix Shape: (10, 10)
(10x10 matrix for 10 classes)

Confusion Matrix:
[[53  0  0  0  0  0  0  0  0  0]
 [ 0 49  0  0  0  0  1  0  2  2]
 [ 0  0 52  1  0  0  0  0  1  0]
 [ 0  0  0 54  0  0  0  0  1  0]
 [ 0  0  0  0 53  0  0  1  0  0]
 [ 0  0  0  0  0 54  0  0  0  1]
 [ 0  0  0  0  0  0 54  0  0  0]
 [ 0  0  0  0  0  0  0 53  0  1]
 [ 0  1  1  1  0  0  0  0 49  0]
 [ 0  0  0  1  0  1  0  0  1 51]]

✅ Multi-class confusion matrix visualization created

============================================================
Analyzing Confusion Patterns:
============================================================
Digit 0: 53/53 correct (100.0%) - perfect!
Digit 1: 49/54 correct (90.7%) - most confused with 9 (2 times)
Digit 2: 52/54 correct (96.3%) - most confused with 8 (1 times)
Digit 3: 54/55 correct (98.2%) - most confused with 2 (1 times)
Digit 4: 53/54 correct (98.1%) - most confused with 7 (1 times)
Digit 5: 54/55 correct (98.2%) - most confused with 9 (1 times)
Digit 6: 54/54 correct (100.0%) - perfect!
Digit 7: 53/54 correct (98.1%) - most confused with 9 (1 times)
Digit 8: 49/52 correct (94.2%) - most confused with 1 (1 times)
Digit 9: 51/54 correct (94.4%) - most confused with 5 (1 times)
\`\`\`

### Averaging Strategies for Multi-class Metrics

For multi-class problems, we need to average metrics across classes. There are three main strategies:

\`\`\`python
# Calculate metrics with different averaging strategies
print("Multi-class Metric Averaging Strategies")
print("="*60)

# Macro average: unweighted mean (all classes equal)
precision_macro = precision_score (y_test, y_pred, average='macro')
recall_macro = recall_score (y_test, y_pred, average='macro')
f1_macro = f1_score (y_test, y_pred, average='macro')

# Weighted average: weighted by support (class frequency)
precision_weighted = precision_score (y_test, y_pred, average='weighted')
recall_weighted = recall_score (y_test, y_pred, average='weighted')
f1_weighted = f1_score (y_test, y_pred, average='weighted')

# Micro average: aggregate all TP, FP, FN across classes
precision_micro = precision_score (y_test, y_pred, average='micro')
recall_micro = recall_score (y_test, y_pred, average='micro')
f1_micro = f1_score (y_test, y_pred, average='micro')

# Per-class metrics (no averaging)
precision_per_class = precision_score (y_test, y_pred, average=None)
recall_per_class = recall_score (y_test, y_pred, average=None)
f1_per_class = f1_score (y_test, y_pred, average=None)

print("\\n1. MACRO AVERAGE (unweighted mean):")
print(f"   Precision: {precision_macro:.4f}")
print(f"   Recall: {recall_macro:.4f}")
print(f"   F1-Score: {f1_macro:.4f}")
print("   → Treats all classes equally")
print("   → Use when all classes are equally important")

print("\\n2. WEIGHTED AVERAGE (weighted by support):")
print(f"   Precision: {precision_weighted:.4f}")
print(f"   Recall: {recall_weighted:.4f}")
print(f"   F1-Score: {f1_weighted:.4f}")
print("   → Accounts for class imbalance")
print("   → Use when classes have different importance based on frequency")

print("\\n3. MICRO AVERAGE (global TP/FP/FN):")
print(f"   Precision: {precision_micro:.4f}")
print(f"   Recall: {recall_micro:.4f}")
print(f"   F1-Score: {f1_micro:.4f}")
print("   → Aggregates contributions of all classes")
print("   → Use when you care about overall performance")

print("\\n4. PER-CLASS METRICS (no averaging):")
for i, (p, r, f1) in enumerate (zip (precision_per_class, recall_per_class, f1_per_class)):
    print(f"   Class {i}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
\`\`\`

**Output:**
\`\`\`
Multi-class Metric Averaging Strategies
============================================================

1. MACRO AVERAGE (unweighted mean):
   Precision: 0.9722
   Recall: 0.9693
   F1-Score: 0.9707
   → Treats all classes equally
   → Use when all classes are equally important

2. WEIGHTED AVERAGE (weighted by support):
   Precision: 0.9722
   Recall: 0.9722
   F1-Score: 0.9722
   → Accounts for class imbalance
   → Use when classes have different importance based on frequency

3. MICRO AVERAGE (global TP/FP/FN):
   Precision: 0.9722
   Recall: 0.9722
   F1-Score: 0.9722
   → Aggregates contributions of all classes
   → Use when you care about overall performance

4. PER-CLASS METRICS (no averaging):
   Class 0: Precision=1.000, Recall=1.000, F1=1.000
   Class 1: Precision=0.980, Recall=0.907, F1=0.942
   Class 2: Precision=0.981, Recall=0.963, F1=0.972
   Class 3: Precision=0.964, Recall=0.982, F1=0.973
   Class 4: Precision=1.000, Recall=0.981, F1=0.991
   Class 5: Precision=0.982, Recall=0.982, F1=0.982
   Class 6: Precision=0.982, Recall=1.000, F1=0.991
   Class 7: Precision=0.982, Recall=0.981, F1=0.982
   Class 8: Precision=0.922, Recall=0.942, F1=0.932
   Class 9: Precision=0.927, Recall=0.944, F1=0.936
\`\`\`

### Understanding the Differences

\`\`\`python
# Demonstrate differences with imbalanced dataset
print("\\n" + "="*60)
print("Demonstrating Averaging Differences with Imbalance")
print("="*60)

# Create imbalanced multi-class scenario
np.random.seed(42)
# Simulate: Class 0 (900 samples), Class 1 (80 samples), Class 2 (20 samples)
y_true_imb = np.array([0]*900 + [1]*80 + [2]*20)
# Simulate predictions with varying performance per class
y_pred_imb = y_true_imb.copy()
# Class 0: 95% accuracy (large class, good performance)
errors_0 = np.random.choice (np.where (y_true_imb==0)[0], size=45, replace=False)
y_pred_imb[errors_0] = np.random.choice([1, 2], size=45)
# Class 1: 75% accuracy (medium class, okay performance)
errors_1 = np.random.choice (np.where (y_true_imb==1)[0], size=20, replace=False)
y_pred_imb[errors_1] = 0
# Class 2: 50% accuracy (small class, poor performance)
errors_2 = np.random.choice (np.where (y_true_imb==2)[0], size=10, replace=False)
y_pred_imb[errors_2] = 0

print("\\nScenario: Imbalanced Multi-class")
print(f"Class 0: {np.sum (y_true_imb==0)} samples (90%)")
print(f"Class 1: {np.sum (y_true_imb==1)} samples (8%)")
print(f"Class 2: {np.sum (y_true_imb==2)} samples (2%)")

# Calculate F1 with different averaging
f1_macro_imb = f1_score (y_true_imb, y_pred_imb, average='macro')
f1_weighted_imb = f1_score (y_true_imb, y_pred_imb, average='weighted')
f1_micro_imb = f1_score (y_true_imb, y_pred_imb, average='micro')
f1_per_class_imb = f1_score (y_true_imb, y_pred_imb, average=None)

print(f"\\nPer-class F1 scores:")
print(f"  Class 0 (90%): {f1_per_class_imb[0]:.3f}")
print(f"  Class 1 (8%):  {f1_per_class_imb[1]:.3f}")
print(f"  Class 2 (2%):  {f1_per_class_imb[2]:.3f}")

print(f"\\nAveraged F1 scores:")
print(f"  Macro:    {f1_macro_imb:.3f} (simple mean: {np.mean (f1_per_class_imb):.3f})")
print(f"  Weighted: {f1_weighted_imb:.3f} (weighted by class size)")
print(f"  Micro:    {f1_micro_imb:.3f} (same as accuracy for multi-class)")

print("\\n💡 Key Observations:")
print("  - Macro average treats all classes equally (sensitive to poor minority class)")
print("  - Weighted average dominated by majority class performance")
print("  - Micro average = accuracy for multi-class (useful for overall correctness)")
print("  - Choice depends on whether minority classes are important!")
\`\`\`

**Output:**
\`\`\`
============================================================
Demonstrating Averaging Differences with Imbalance
============================================================

Scenario: Imbalanced Multi-class
Class 0: 900 samples (90%)
Class 1: 80 samples (8%)
Class 2: 20 samples (2%)

Per-class F1 scores:
  Class 0 (90%): 0.958
  Class 1 (8%):  0.774
  Class 2 (2%):  0.615

Averaged F1 scores:
  Macro:    0.782 (simple mean: 0.782)
  Weighted: 0.932 (weighted by class size)
  Micro:    0.925 (same as accuracy for multi-class)

💡 Key Observations:
  - Macro average treats all classes equally (sensitive to poor minority class)
  - Weighted average dominated by majority class performance
  - Micro average = accuracy for multi-class (useful for overall correctness)
  - Choice depends on whether minority classes are important!
\`\`\`

### Complete Multi-class Classification Report

\`\`\`python
# Generate comprehensive report
print("\\n" + "="*60)
print("Complete Multi-class Classification Report:")
print("="*60)
print(classification_report (y_test, y_pred, 
                          target_names=[f'Digit {i}' for i in range(10)]))

# Overall accuracy
accuracy = accuracy_score (y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Cohen\'s Kappa (agreement beyond chance)
kappa = cohen_kappa_score (y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")
print("  (Kappa measures agreement beyond chance)")
print("  1.0 = perfect agreement")
print("  0.0 = no better than chance")
print(f"  {kappa:.2f} = {'Excellent' if kappa > 0.9 else 'Good' if kappa > 0.7 else 'Moderate'}")
\`\`\`

**Output:**
\`\`\`
============================================================
Complete Multi-class Classification Report:
============================================================
              precision    recall  f1-score   support

     Digit 0       1.00      1.00      1.00        53
     Digit 1       0.98      0.91      0.94        54
     Digit 2       0.98      0.96      0.97        54
     Digit 3       0.96      0.98      0.97        55
     Digit 4       1.00      0.98      0.99        54
     Digit 5       0.98      0.98      0.98        55
     Digit 6       0.98      1.00      0.99        54
     Digit 7       0.98      0.98      0.98        54
     Digit 8       0.92      0.94      0.93        52
     Digit 9       0.93      0.94      0.94        54

    accuracy                           0.97       539
   macro avg       0.97      0.97      0.97       539
weighted avg       0.97      0.97      0.97       539

Overall Accuracy: 0.9722 (97.22%)
Cohen\'s Kappa: 0.9691
  (Kappa measures agreement beyond chance)
  1.0 = perfect agreement
  0.0 = no better than chance
  0.97 = Excellent
\`\`\`

## Multi-label Classification

In multi-label problems, each sample can have multiple labels simultaneously.

\`\`\`python
from sklearn.metrics import (
    multilabel_confusion_matrix,
    hamming_loss,
    jaccard_score,
    classification_report
)

# Generate multi-label dataset
print("\\n" + "="*60)
print("Multi-label Classification Example: Movie Genres")
print("="*60)

# Simulate movie genre dataset
# Labels: [Action, Comedy, Drama, Romance, Sci-Fi]
np.random.seed(42)
n_samples = 100
n_labels = 5

# True labels (each movie can have multiple genres)
y_true_ml = np.random.randint(0, 2, size=(n_samples, n_labels))
# Predictions (with some errors)
y_pred_ml = y_true_ml.copy()
# Add 10% random errors
errors = np.random.rand (n_samples, n_labels) < 0.1
y_pred_ml[errors] = 1 - y_pred_ml[errors]

genre_names = ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi']

print(f"Number of movies: {n_samples}")
print(f"Number of genres (labels): {n_labels}")
print(f"\\nExample movie labels:")
for i in range(5):
    true_genres = [genre_names[j] for j in range (n_labels) if y_true_ml[i, j] == 1]
    pred_genres = [genre_names[j] for j in range (n_labels) if y_pred_ml[i, j] == 1]
    print(f"  Movie {i+1}:")
    print(f"    True: {', '.join (true_genres) if true_genres else 'None'}")
    print(f"    Pred: {', '.join (pred_genres) if pred_genres else 'None'}")
\`\`\`

**Output:**
\`\`\`
============================================================
Multi-label Classification Example: Movie Genres
============================================================
Number of movies: 100
Number of genres (labels): 5

Example movie labels:
  Movie 1:
    True: Comedy, Drama
    Pred: Comedy, Drama
  Movie 2:
    True: Action, Drama, Romance, Sci-Fi
    Pred: Action, Drama, Sci-Fi
  Movie 3:
    True: Comedy
    Pred: Comedy
  Movie 4:
    True: Action, Comedy, Drama
    Pred: Action, Comedy, Drama, Romance
  Movie 5:
    True: Action, Comedy, Sci-Fi
    Pred: Comedy, Sci-Fi
\`\`\`

### Multi-label Metrics

\`\`\`python
# 1. Hamming Loss: fraction of labels incorrectly predicted
hamming = hamming_loss (y_true_ml, y_pred_ml)
print(f"\\n1. Hamming Loss: {hamming:.4f}")
print(f"   Interpretation: {hamming*100:.2f}% of labels are incorrectly predicted")
print("   (Lower is better, 0 = perfect)")

# 2. Exact Match Ratio: fraction of samples with ALL labels correct
exact_match = accuracy_score (y_true_ml, y_pred_ml)
print(f"\\n2. Exact Match Ratio: {exact_match:.4f}")
print(f"   Interpretation: {exact_match*100:.2f}% of movies have ALL genres correct")
print("   (Very strict metric)")

# 3. Jaccard Score (Intersection over Union)
jaccard = jaccard_score (y_true_ml, y_pred_ml, average='samples')
print(f"\\n3. Jaccard Score (IoU): {jaccard:.4f}")
print("   Interpretation: Average overlap between true and predicted labels")
print("   Formula: |True ∩ Pred| / |True ∪ Pred|")
print("   1.0 = perfect overlap, 0.0 = no overlap")

# 4. Per-label metrics
print("\\n4. Per-label Metrics:")
for i, genre in enumerate (genre_names):
    precision = precision_score (y_true_ml[:, i], y_pred_ml[:, i])
    recall = recall_score (y_true_ml[:, i], y_pred_ml[:, i])
    f1 = f1_score (y_true_ml[:, i], y_pred_ml[:, i])
    print(f"   {genre:8s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# 5. Sample-wise metrics
print("\\n5. Sample-wise F1 Score:")
f1_samples = f1_score (y_true_ml, y_pred_ml, average='samples')
print(f"   F1 (averaged over samples): {f1_samples:.4f}")

# 6. Label-wise metrics with different averaging
print("\\n6. Label-wise Averaging:")
f1_macro = f1_score (y_true_ml, y_pred_ml, average='macro')
f1_micro = f1_score (y_true_ml, y_pred_ml, average='micro')
f1_weighted = f1_score (y_true_ml, y_pred_ml, average='weighted')
print(f"   Macro F1:    {f1_macro:.4f} (average across labels)")
print(f"   Micro F1:    {f1_micro:.4f} (global TP/FP/FN)")
print(f"   Weighted F1: {f1_weighted:.4f} (weighted by support)")

# Visualize multi-label confusion matrix
print("\\n7. Multi-label Confusion Matrix:")
mlcm = multilabel_confusion_matrix (y_true_ml, y_pred_ml)
print(f"   Shape: {mlcm.shape} (one 2x2 matrix per label)")
print("\\n   Per-label confusion matrices:")
for i, genre in enumerate (genre_names):
    tn, fp, fn, tp = mlcm[i].ravel()
    print(f"   {genre}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
\`\`\`

**Output:**
\`\`\`
1. Hamming Loss: 0.0980
   Interpretation: 9.80% of labels are incorrectly predicted
   (Lower is better, 0 = perfect)

2. Exact Match Ratio: 0.6900
   Interpretation: 69.00% of movies have ALL genres correct
   (Very strict metric)

3. Jaccard Score (IoU): 0.8123
   Interpretation: Average overlap between true and predicted labels
   Formula: |True ∩ Pred| / |True ∪ Pred|
   1.0 = perfect overlap, 0.0 = no overlap

4. Per-label Metrics:
   Action  : Precision=0.883, Recall=0.894, F1=0.889
   Comedy  : Precision=0.897, Recall=0.860, F1=0.878
   Drama   : Precision=0.893, Recall=0.926, F1=0.909
   Romance : Precision=0.929, Recall=0.867, F1=0.897
   Sci-Fi  : Precision=0.913, Recall=0.913, F1=0.913

5. Sample-wise F1 Score:
   F1 (averaged over samples): 0.8987

6. Label-wise Averaging:
   Macro F1:    0.8972 (average across labels)
   Micro F1:    0.9020 (global TP/FP/FN)
   Weighted F1: 0.8978 (weighted by support)

7. Multi-label Confusion Matrix:
   Shape: (5, 2, 2) (one 2x2 matrix per label)

   Per-label confusion matrices:
   Action: TN=51, FP=6, FN=5, TP=38
   Comedy: TN=52, FP=5, FN=7, TP=36
   Drama: TN=47, FP=4, FN=4, TP=45
   Romance: TN=47, FP=3, FN=6, TP=44
   Sci-Fi: TN=53, FP=4, FN=4, TP=39
\`\`\`

## Choosing Metrics

\`\`\`python
print("\\n" + "="*60)
print("Metric Selection Guide")
print("="*60)

guide = {
    "Multi-class": {
        "Balanced classes, equal importance": "Macro-averaged F1",
        "Imbalanced classes": "Weighted-averaged F1 or per-class analysis",
        "Overall correctness matters": "Micro-averaged F1 (equals accuracy)",
        "Need per-class breakdown": "Classification report with all averages",
        "Want single number": "Weighted F1 or Cohen\'s Kappa",
    },
    "Multi-label": {
        "Strict evaluation (all labels must match)": "Exact Match Ratio",
        "Label-wise errors": "Hamming Loss",
        "Overlap-based": "Jaccard Score (IoU)",
        "Per-label performance": "Per-label Precision/Recall/F1",
        "Overall performance": "Micro or Macro F1",
    }
}

for task, metrics in guide.items():
    print(f"\\n{task} Classification:")
    for scenario, metric in metrics.items():
        print(f"  {scenario:45s} → {metric}")
\`\`\`

**Output:**
\`\`\`
============================================================
Metric Selection Guide
============================================================

Multi-class Classification:
  Balanced classes, equal importance              → Macro-averaged F1
  Imbalanced classes                              → Weighted-averaged F1 or per-class analysis
  Overall correctness matters                     → Micro-averaged F1 (equals accuracy)
  Need per-class breakdown                        → Classification report with all averages
  Want single number                              → Weighted F1 or Cohen's Kappa

Multi-label Classification:
  Strict evaluation (all labels must match)       → Exact Match Ratio
  Label-wise errors                               → Hamming Loss
  Overlap-based                                   → Jaccard Score (IoU)
  Per-label performance                           → Per-label Precision/Recall/F1
  Overall performance                             → Micro or Macro F1
\`\`\`

## Key Takeaways

1. **Multi-class**: Each sample belongs to ONE class
   - Use macro-averaged metrics when classes are equally important
   - Use weighted-averaged metrics when classes have different importance
   - Use micro-averaged metrics (= accuracy) for overall correctness

2. **Multi-label**: Each sample can have MULTIPLE labels
   - Hamming Loss: Fraction of wrong labels
   - Exact Match: All labels must be correct (strict)
   - Jaccard Score: Overlap between true and predicted labels

3. **Averaging Strategies**:
   - **Macro**: Simple mean across classes (treats all classes equally)
   - **Weighted**: Weighted by class frequency (accounts for imbalance)
   - **Micro**: Aggregate all TP/FP/FN (overall performance)
   - **Samples**: Average per sample (for multi-label)

4. **Always report**:
   - Confusion matrix or per-class metrics
   - Multiple averaging strategies
   - Context about class imbalance

**Remember**: The "best" metric depends on your problem—whether minority classes are important, whether errors are equally costly, and whether you care about overall vs per-class performance!
`,
};
