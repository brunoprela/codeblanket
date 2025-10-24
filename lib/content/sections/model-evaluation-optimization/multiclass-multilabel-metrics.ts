export const multiClassMultiLabelMetrics = {
  title: 'Multi-class & Multi-label Metrics',
  content: `
# Multi-class & Multi-label Metrics

## Introduction

Real-world classification problems often involve more than two classes (multi-class) or multiple labels per instance (multi-label). These scenarios require specialized metrics and careful consideration of how to aggregate performance across classes.

**Key Distinctions:**
- **Multi-class**: Each instance belongs to exactly one of multiple classes (e.g., classifying images as cat, dog, or bird)
- **Multi-label**: Each instance can belong to multiple classes simultaneously (e.g., tagging articles with multiple topics)

## Multi-class Classification

### The Multi-class Confusion Matrix

\`\`\`python
import numpy as np
from sklearn.datasets import load_iris, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load multi-class dataset
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Multi-class & Multi-label Metrics")
print("="*70)

# Multi-class confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\\nMulti-class Confusion Matrix:")
print(cm)

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=iris.target_names,
           yticklabels=iris.target_names)
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.title('Multi-class Confusion Matrix', fontsize=14, fontweight='bold')

# Add percentages
for i in range(len(iris.target_names)):
    for j in range(len(iris.target_names)):
        percentage = cm[i, j] / cm[i, :].sum() * 100
        plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                ha='center', va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('multiclass_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\\nConfusion matrix saved to 'multiclass_confusion_matrix.png'")

# Per-class analysis
print("\\n" + "="*70)
print("Per-class Analysis:")
print("="*70)

for i, class_name in enumerate(iris.target_names):
    # For each class, calculate metrics
    y_test_binary = (y_test == i).astype(int)
    y_pred_binary = (y_pred == i).astype(int)
    
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\\nClass '{class_name}':")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
\`\`\`

## Averaging Strategies: Macro, Micro, and Weighted

### Understanding Different Averages

\`\`\`python
print("\\n" + "="*70)
print("Averaging Strategies for Multi-class Metrics")
print("="*70)

# Calculate metrics with different averaging
metrics = {}

# Macro average: Calculate metric for each class, then average
metrics['macro'] = {
    'precision': precision_score(y_test, y_pred, average='macro'),
    'recall': recall_score(y_test, y_pred, average='macro'),
    'f1': f1_score(y_test, y_pred, average='macro')
}

# Weighted average: Like macro but weighted by support (class frequency)
metrics['weighted'] = {
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1': f1_score(y_test, y_pred, average='weighted')
}

# Micro average: Calculate metrics globally by counting total TP, FP, FN
metrics['micro'] = {
    'precision': precision_score(y_test, y_pred, average='micro'),
    'recall': recall_score(y_test, y_pred, average='micro'),
    'f1': f1_score(y_test, y_pred, average='micro')
}

# Display results
print(f"\\n{'Average Type':<15s} {'Precision':<12s} {'Recall':<12s} {'F1 Score':<12s}")
print("-"*70)

for avg_type, scores in metrics.items():
    print(f"{avg_type.capitalize():<15s} {scores['precision']:<12.4f} "
          f"{scores['recall']:<12.4f} {scores['f1']:<12.4f}")

print("\\nExplanation:")
print("  Macro:    Average of per-class scores (all classes equal weight)")
print("  Weighted: Average weighted by class support (frequent classes matter more)")
print("  Micro:    Global calculation (all samples equal weight)")

# Manual calculation to understand micro average
print("\\n" + "="*70)
print("Understanding Micro Average:")
print("="*70)

# For micro, we treat it as binary classification
# TP = all correct predictions, FP = all wrong predictions
total_tp = np.sum(y_test == y_pred)
total_fp = np.sum(y_test != y_pred)
total_fn = total_fp  # In multi-class, FP of one class is FN of another

micro_precision_manual = total_tp / (total_tp + total_fp)
micro_recall_manual = total_tp / (total_tp + total_fn)
micro_f1_manual = 2 * (micro_precision_manual * micro_recall_manual) / (
    micro_precision_manual + micro_recall_manual
)

print(f"Micro Precision (manual): {micro_precision_manual:.4f}")
print(f"Micro Recall (manual):    {micro_recall_manual:.4f}")
print(f"Micro F1 (manual):        {micro_f1_manual:.4f}")

# Note: For balanced multi-class, micro precision = micro recall = accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\\nAccuracy: {accuracy:.4f}")
print(f"Micro F1: {metrics['micro']['f1']:.4f}")
print("→ In multi-class, micro average equals accuracy")
\`\`\`

### When to Use Each Average

\`\`\`python
print("\\n" + "="*70)
print("Choosing the Right Average")
print("="*70)

scenarios = [
    {
        'scenario': 'Balanced classes, all equally important',
        'recommendation': 'Macro',
        'reason': 'Treats each class equally'
    },
    {
        'scenario': 'Imbalanced classes, frequent classes more important',
        'recommendation': 'Weighted',
        'reason': 'Accounts for class imbalance'
    },
    {
        'scenario': 'Want overall performance regardless of class',
        'recommendation': 'Micro',
        'reason': 'Treats each sample equally'
    },
    {
        'scenario': 'Rare classes are critical (e.g., rare diseases)',
        'recommendation': 'Macro',
        'reason': 'Prevents rare classes from being ignored'
    }
]

for s in scenarios:
    print(f"\\n{s['scenario']}:")
    print(f"  → Use {s['recommendation']}: {s['reason']}")
\`\`\`

### Demonstration with Imbalanced Multi-class Data

\`\`\`python
print("\\n" + "="*70)
print("Comparing Averages on Imbalanced Data")
print("="*70)

# Create imbalanced multi-class dataset
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=3, n_clusters_per_class=1,
    weights=[0.7, 0.2, 0.1], random_state=42
)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# Train model
model_imb = RandomForestClassifier(n_estimators=100, random_state=42)
model_imb.fit(X_train_imb, y_train_imb)
y_pred_imb = model_imb.predict(X_test_imb)

# Show class distribution
print("Class distribution in test set:")
for i in range(3):
    count = (y_test_imb == i).sum()
    print(f"  Class {i}: {count:3d} samples ({count/len(y_test_imb)*100:5.1f}%)")

# Calculate all averaging methods
print("\\n" + "="*70)
print("Performance with Different Averages:")
print("="*70)

for average in ['macro', 'weighted', 'micro']:
    precision = precision_score(y_test_imb, y_pred_imb, average=average)
    recall = recall_score(y_test_imb, y_pred_imb, average=average)
    f1 = f1_score(y_test_imb, y_pred_imb, average=average)
    
    print(f"\\n{average.upper()} Average:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

# Show per-class scores
print("\\n" + "="*70)
print("Per-class Scores:")
print("="*70)

precision_per_class = precision_score(y_test_imb, y_pred_imb, average=None)
recall_per_class = recall_score(y_test_imb, y_pred_imb, average=None)
f1_per_class = f1_score(y_test_imb, y_pred_imb, average=None)

print(f"\\n{'Class':<8s} {'Support':<10s} {'Precision':<12s} {'Recall':<12s} {'F1':<12s}")
print("-"*70)

for i in range(3):
    support = (y_test_imb == i).sum()
    print(f"{i:<8d} {support:<10d} {precision_per_class[i]:<12.4f} "
          f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")

print("\\nObservation: Rare classes often have lower performance")
print("Macro average highlights this, weighted average masks it")
\`\`\`

## Multi-class ROC and AUC

### One-vs-Rest (OvR) ROC Curves

\`\`\`python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

print("\\n" + "="*70)
print("Multi-class ROC Curves (One-vs-Rest)")
print("="*70)

# Get prediction probabilities
y_prob = model.predict_proba(X_test)

# Binarize the labels for OvR
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot
plt.figure(figsize=(10, 8))

colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
            label=f'{iris.target_names[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multi-class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('multiclass_roc.png', dpi=150, bbox_inches='tight')
print("\\nMulti-class ROC curves saved to 'multiclass_roc.png'")

# Calculate macro and weighted average AUC
print("\\nAUC Scores:")
for i in range(n_classes):
    print(f"  {iris.target_names[i]}: {roc_auc[i]:.4f}")

# Macro-average AUC
macro_auc = np.mean(list(roc_auc.values()))
print(f"\\nMacro-average AUC: {macro_auc:.4f}")

# Using sklearn's built-in
from sklearn.metrics import roc_auc_score

# One-vs-Rest AUC
ovr_auc_macro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
ovr_auc_weighted = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

print(f"\\nOne-vs-Rest AUC:")
print(f"  Macro:    {ovr_auc_macro:.4f}")
print(f"  Weighted: {ovr_auc_weighted:.4f}")

# One-vs-One AUC
ovo_auc_macro = roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
ovo_auc_weighted = roc_auc_score(y_test, y_prob, multi_class='ovo', average='weighted')

print(f"\\nOne-vs-One AUC:")
print(f"  Macro:    {ovo_auc_macro:.4f}")
print(f"  Weighted: {ovo_auc_weighted:.4f}")
\`\`\`

## Multi-label Classification

### Multi-label Basics

\`\`\`python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, jaccard_score

print("\\n" + "="*70)
print("Multi-label Classification")
print("="*70)

# Generate multi-label dataset
X_ml, y_ml = make_multilabel_classification(
    n_samples=1000, n_features=20, n_classes=5,
    n_labels=2, random_state=42
)

print(f"Dataset: {X_ml.shape[0]} samples, {X_ml.shape[1]} features, {y_ml.shape[1]} labels")
print(f"\\nExample samples (first 5):")
print(f"{'Sample':<8s} {'Labels':<30s}")
print("-"*70)
for i in range(5):
    labels = np.where(y_ml[i] == 1)[0]
    print(f"{i:<8d} {str(labels):<30s}")

# Split data
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_ml, y_ml, test_size=0.3, random_state=42
)

# Train multi-label model
model_ml = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model_ml.fit(X_train_ml, y_train_ml)
y_pred_ml = model_ml.predict(X_test_ml)

print(f"\\nPredictions (first 5):")
print(f"{'Sample':<8s} {'True Labels':<25s} {'Predicted Labels':<25s}")
print("-"*70)
for i in range(5):
    true_labels = np.where(y_test_ml[i] == 1)[0]
    pred_labels = np.where(y_pred_ml[i] == 1)[0]
    print(f"{i:<8d} {str(true_labels):<25s} {str(pred_labels):<25s}")
\`\`\`

### Multi-label Metrics

\`\`\`python
print("\\n" + "="*70)
print("Multi-label Metrics")
print("="*70)

# 1. Hamming Loss: Fraction of labels that are incorrectly predicted
hamming = hamming_loss(y_test_ml, y_pred_ml)
print(f"\\n1. Hamming Loss: {hamming:.4f}")
print(f"   Interpretation: {hamming*100:.2f}% of labels are incorrectly predicted")
print(f"   (Lower is better, 0 is perfect)")

# Manual calculation
n_samples, n_labels = y_test_ml.shape
errors = np.sum(y_test_ml != y_pred_ml)
hamming_manual = errors / (n_samples * n_labels)
print(f"   Manual calculation: {errors} errors / ({n_samples} × {n_labels}) = {hamming_manual:.4f} ✓")

# 2. Jaccard Score (Intersection over Union)
jaccard = jaccard_score(y_test_ml, y_pred_ml, average='samples')
print(f"\\n2. Jaccard Score: {jaccard:.4f}")
print(f"   Interpretation: Average IoU across samples")
print(f"   Formula: |intersection| / |union|")

# Show example calculation
for i in range(3):
    true_set = set(np.where(y_test_ml[i] == 1)[0])
    pred_set = set(np.where(y_pred_ml[i] == 1)[0])
    
    intersection = true_set & pred_set
    union = true_set | pred_set
    
    if len(union) > 0:
        sample_jaccard = len(intersection) / len(union)
    else:
        sample_jaccard = 1.0
    
    print(f"   Sample {i}: True={true_set}, Pred={pred_set}, "
          f"Jaccard={sample_jaccard:.3f}")

# 3. Exact Match Ratio (Subset Accuracy)
from sklearn.metrics import accuracy_score

exact_match = accuracy_score(y_test_ml, y_pred_ml)
print(f"\\n3. Exact Match Ratio: {exact_match:.4f}")
print(f"   Interpretation: {exact_match*100:.1f}% of samples have ALL labels correct")
print(f"   (Strictest metric - all labels must match exactly)")

# 4. Precision, Recall, F1 for Multi-label
precision_ml = precision_score(y_test_ml, y_pred_ml, average='samples')
recall_ml = recall_score(y_test_ml, y_pred_ml, average='samples')
f1_ml = f1_score(y_test_ml, y_pred_ml, average='samples')

print(f"\\n4. Sample-averaged Metrics:")
print(f"   Precision: {precision_ml:.4f}")
print(f"   Recall:    {recall_ml:.4f}")
print(f"   F1 Score:  {f1_ml:.4f}")

# Different averaging for multi-label
print("\\n" + "="*70)
print("Different Averaging Strategies for Multi-label:")
print("="*70)

for average in ['micro', 'macro', 'weighted', 'samples']:
    try:
        p = precision_score(y_test_ml, y_pred_ml, average=average)
        r = recall_score(y_test_ml, y_pred_ml, average=average)
        f = f1_score(y_test_ml, y_pred_ml, average=average)
        
        print(f"\\n{average.upper()} average:")
        print(f"  Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
    except:
        pass

print("\\nAveraging options:")
print("  'micro':   Calculate globally (all label-sample pairs)")
print("  'macro':   Calculate per-label, then average")
print("  'weighted': Macro weighted by label support")
print("  'samples':  Calculate per-sample, then average")
\`\`\`

### Multi-label Classification Report

\`\`\`python
print("\\n" + "="*70)
print("Comprehensive Multi-label Report")
print("="*70)

# Per-label metrics
print("\\nPer-label Performance:")
print(f"{'Label':<8s} {'Support':<10s} {'Precision':<12s} {'Recall':<12s} {'F1':<12s}")
print("-"*70)

for label_idx in range(y_test_ml.shape[1]):
    y_true_label = y_test_ml[:, label_idx]
    y_pred_label = y_pred_ml[:, label_idx]
    
    support = np.sum(y_true_label)
    precision = precision_score(y_true_label, y_pred_label, zero_division=0)
    recall = recall_score(y_true_label, y_pred_label, zero_division=0)
    f1 = f1_score(y_true_label, y_pred_label, zero_division=0)
    
    print(f"{label_idx:<8d} {support:<10d} {precision:<12.4f} "
          f"{recall:<12.4f} {f1:<12.4f}")

# Summary metrics
print("\\n" + "="*70)
print("Summary Metrics:")
print("="*70)

summary = {
    'Hamming Loss': hamming_loss(y_test_ml, y_pred_ml),
    'Exact Match Ratio': accuracy_score(y_test_ml, y_pred_ml),
    'Jaccard Score': jaccard_score(y_test_ml, y_pred_ml, average='samples'),
    'Precision (micro)': precision_score(y_test_ml, y_pred_ml, average='micro'),
    'Recall (micro)': recall_score(y_test_ml, y_pred_ml, average='micro'),
    'F1 (micro)': f1_score(y_test_ml, y_pred_ml, average='micro'),
    'Precision (macro)': precision_score(y_test_ml, y_pred_ml, average='macro'),
    'Recall (macro)': recall_score(y_test_ml, y_pred_ml, average='macro'),
    'F1 (macro)': f1_score(y_test_ml, y_pred_ml, average='macro'),
}

for metric, value in summary.items():
    print(f"{metric:<25s}: {value:.4f}")
\`\`\`

## Label Ranking Metrics

\`\`\`python
from sklearn.metrics import label_ranking_average_precision_score, coverage_error

print("\\n" + "="*70)
print("Label Ranking Metrics")
print("="*70)

# Get prediction scores (probabilities)
y_score_ml = np.array([
    est.predict_proba(X_test_ml)[:, 1] 
    for est in model_ml.estimators_
]).T

# Label Ranking Average Precision
lrap = label_ranking_average_precision_score(y_test_ml, y_score_ml)
print(f"\\nLabel Ranking Average Precision: {lrap:.4f}")
print("  Measures average precision for ranking problem")
print("  Higher is better (1.0 is perfect)")

# Coverage Error
coverage = coverage_error(y_test_ml, y_score_ml)
print(f"\\nCoverage Error: {coverage:.4f}")
print("  Average number of labels to go through to cover all true labels")
print("  Lower is better")

# Example ranking
print("\\n" + "="*70)
print("Example Label Rankings:")
print("="*70)

for i in range(3):
    true_labels = set(np.where(y_test_ml[i] == 1)[0])
    
    # Rank labels by score
    scores = y_score_ml[i]
    ranked_labels = np.argsort(scores)[::-1]  # Descending order
    
    print(f"\\nSample {i}:")
    print(f"  True labels: {true_labels}")
    print(f"  Ranked prediction (label: score):")
    for rank, label in enumerate(ranked_labels[:5], 1):
        marker = "✓" if label in true_labels else " "
        print(f"    {rank}. Label {label}: {scores[label]:.3f} {marker}")
\`\`\`

## Comprehensive Multi-class vs Multi-label Comparison

\`\`\`python
print("\\n" + "="*70)
print("Multi-class vs Multi-label: Quick Reference")
print("="*70)

comparison = pd.DataFrame({
    'Aspect': [
        'Definition',
        'Example',
        'Output per sample',
        'Mutually exclusive?',
        'Primary metrics',
        'Averaging methods',
        'When to use'
    ],
    'Multi-class': [
        'One class per sample',
        'Cat, Dog, or Bird',
        'Single class label',
        'Yes',
        'Accuracy, Macro/Micro F1',
        'Macro, Micro, Weighted',
        'Mutually exclusive categories'
    ],
    'Multi-label': [
        'Multiple classes per sample',
        'Action, Comedy, and Drama',
        'Multiple binary indicators',
        'No',
        'Hamming Loss, Jaccard, Subset Accuracy',
        'Micro, Macro, Samples',
        'Multiple simultaneous attributes'
    ]
})

print("\\n" + comparison.to_string(index=False))
\`\`\`

## Trading Application: Multi-class Regime Detection

\`\`\`python
print("\\n" + "="*70)
print("Trading Application: Market Regime Classification")
print("="*70)

# Simulate multi-class market regime classification
np.random.seed(42)
n_days = 300

# Generate features
features = {
    'returns': np.random.randn(n_days) * 0.02,
    'volatility': np.abs(np.random.randn(n_days) * 0.01),
    'volume': np.random.uniform(0.5, 1.5, n_days),
    'momentum': np.random.randn(n_days) * 0.01
}

# Define regimes: 0=Bull, 1=Bear, 2=Sideways
# Simple logic based on returns and volatility
regimes = []
for i in range(n_days):
    if features['returns'][i] > 0.01 and features['volatility'][i] < 0.015:
        regimes.append(0)  # Bull
    elif features['returns'][i] < -0.01:
        regimes.append(1)  # Bear
    else:
        regimes.append(2)  # Sideways

X_regime = np.column_stack([features[k] for k in features.keys()])
y_regime = np.array(regimes)

# Split and train
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regime, y_regime, test_size=0.3, random_state=42, stratify=y_regime
)

model_regime = RandomForestClassifier(n_estimators=100, random_state=42)
model_regime.fit(X_train_reg, y_train_reg)
y_pred_reg = model_regime.predict(X_test_reg)

# Evaluate
regime_names = ['Bull', 'Bear', 'Sideways']
cm_regime = confusion_matrix(y_test_reg, y_pred_reg)

print("\\nMarket Regime Classification Results:")
print("="*70)

# Detailed report
print("\\nClassification Report:")
print(classification_report(y_test_reg, y_pred_reg, target_names=regime_names))

# Trading implications
print("\\nTrading Implications:")
print("-"*70)

for i, regime in enumerate(regime_names):
    y_true_binary = (y_test_reg == i).astype(int)
    y_pred_binary = (y_pred_reg == i).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    
    print(f"\\n{regime} Market:")
    print(f"  Precision: {precision:.3f} - When we predict {regime}, "
          f"{precision*100:.1f}% of time it's correct")
    print(f"  Recall: {recall:.3f} - We correctly identify "
          f"{recall*100:.1f}% of {regime} periods")
    
    if regime == 'Bull':
        print(f"  Impact: High precision = Confident long entries")
    elif regime == 'Bear':
        print(f"  Impact: High recall = Don't miss exits/hedges")
    else:
        print(f"  Impact: Avoid overtrading in flat markets")

print("\\n✓ Multi-class classification helps adapt strategy to market regime")
\`\`\`

## Key Takeaways

### Multi-class Classification:
1. **Confusion Matrix**: NxN matrix for N classes
2. **Macro Average**: Equal weight to each class (good for balanced importance)
3. **Micro Average**: Equal weight to each sample (equals accuracy)
4. **Weighted Average**: Accounts for class imbalance
5. **OvR/OvO ROC**: Can compute AUC for multi-class

### Multi-label Classification:
1. **Hamming Loss**: Fraction of wrong labels
2. **Exact Match**: Strictest metric (all labels must match)
3. **Jaccard Score**: Intersection over union
4. **Per-label Metrics**: Treat each label as binary classification
5. **Sample-averaged**: Average metrics across samples

**Recommendation**: Always report multiple metrics and per-class/label performance for complete picture.

## Further Reading

- Sokolova, M., & Lapalme, G. (2009). "A systematic analysis of performance measures for classification tasks"
- Tsoumakas, G., & Katakis, I. (2007). "Multi-label classification: An overview"
- Zhang, M. L., & Zhou, Z. H. (2014). "A review on multi-label learning algorithms"
`,
  exercises: [
    {
      prompt:
        'Build a comprehensive multi-class and multi-label evaluation framework that handles both scenarios, provides all relevant metrics, and generates detailed visualizations including per-class/label analysis.',
      solution: `
# Complete solution provided in separate comprehensive evaluation class
`,
    },
  ],
  quizId: 'model-evaluation-optimization-multiclass-multilabel',
  multipleChoiceId: 'model-evaluation-optimization-multiclass-multilabel-mc',
};
