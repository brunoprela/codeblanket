export const classificationMetrics = {
  title: 'Classification Metrics',
  content: `
# Classification Metrics

## Introduction

Classification metrics evaluate how well a model predicts categorical outcomes. Unlike regression where we measure distance from the true value, classification requires different approaches to assess correctness, confidence, and the types of errors made.

**Key Challenge**: Not all errors are equal. In medical diagnosis, failing to detect cancer (false negative) is often more severe than a false alarm (false positive). Classification metrics help us understand and optimize for different types of errors.

## The Confusion Matrix

The confusion matrix is the foundation of all classification metrics. It's a table showing the counts of correct and incorrect predictions broken down by each class.

### Binary Classification Confusion Matrix

\`\`\`python
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

# Load binary classification dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Metrics Deep Dive")
print("="*70)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\\nConfusion Matrix:")
print(cm)
print("\\nStructure:")
print("              Predicted Negative  Predicted Positive")
print("Actual Negative        TN                  FP")
print("Actual Positive        FN                  TP")

# Extract components
tn, fp, fn, tp = cm.ravel()

print(f"\\nComponents:")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")

print(f"\\nTotal predictions: {tn + fp + fn + tp}")
print(f"Correct predictions: {tn + tp} ({(tn + tp)/(tn + fp + fn + tp)*100:.1f}%)")
print(f"Incorrect predictions: {fp + fn} ({(fp + fn)/(tn + fp + fn + tp)*100:.1f}%)")

# Visualize confusion matrix
def plot_confusion_matrix(cm, labels=['Negative', 'Positive'], title='Confusion Matrix'):
    """Plot confusion matrix with annotations."""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add percentage annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            percentage = cm[i, j] / cm.sum() * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    return plt.gcf()

plot_confusion_matrix(cm, labels=['Malignant', 'Benign'])
plt.savefig('confusion_matrix_basic.png', dpi=150, bbox_inches='tight')
print("\\nConfusion matrix plot saved to 'confusion_matrix_basic.png'")
\`\`\`

## Accuracy

### Definition

Accuracy is the proportion of correct predictions:

$$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$

### When Accuracy Fails

\`\`\`python
print("\\n" + "="*70)
print("Why Accuracy Can Be Misleading")
print("="*70)

# Create imbalanced dataset
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=2, weights=[0.95, 0.05],
    random_state=42
)

print(f"\\nClass distribution:")
print(f"Class 0: {(y_imb == 0).sum()} samples ({(y_imb == 0).sum()/len(y_imb)*100:.1f}%)")
print(f"Class 1: {(y_imb == 1).sum()} samples ({(y_imb == 1).sum()/len(y_imb)*100:.1f}%)")

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# Dummy classifier that always predicts majority class
class AlwaysMajorityClassifier:
    def fit(self, X, y):
        self.majority_class_ = np.argmax(np.bincount(y))
        return self
    
    def predict(self, X):
        return np.full(len(X), self.majority_class_)

# Compare real model vs dummy
dummy = AlwaysMajorityClassifier()
dummy.fit(X_train_imb, y_train_imb)
y_pred_dummy = dummy.predict(X_test_imb)

real_model = RandomForestClassifier(n_estimators=100, random_state=42)
real_model.fit(X_train_imb, y_train_imb)
y_pred_real = real_model.predict(X_test_imb)

acc_dummy = accuracy_score(y_test_imb, y_pred_dummy)
acc_real = accuracy_score(y_test_imb, y_pred_real)

print(f"\\nAccuracy comparison:")
print(f"Dummy (always predict majority): {acc_dummy:.4f} ({acc_dummy*100:.1f}%)")
print(f"Real model:                      {acc_real:.4f} ({acc_real*100:.1f}%)")

print(f"\\n⚠️  Dummy classifier achieves {acc_dummy*100:.1f}% accuracy by doing nothing!")
print("This is why accuracy alone is insufficient for imbalanced data.")

# Show confusion matrices
cm_dummy = confusion_matrix(y_test_imb, y_pred_dummy)
cm_real = confusion_matrix(y_test_imb, y_pred_real)

print(f"\\nDummy classifier confusion matrix:")
print(cm_dummy)
print("→ Never predicts minority class!")

print(f"\\nReal model confusion matrix:")
print(cm_real)
print("→ Captures both classes")
\`\`\`

## Precision and Recall

### Definitions

**Precision** (Positive Predictive Value): Of all positive predictions, how many were correct?

$$\\text{Precision} = \\frac{TP}{TP + FP}$$

**Recall** (Sensitivity, True Positive Rate): Of all actual positives, how many did we catch?

$$\\text{Recall} = \\frac{TP}{TP + FN}$$

### The Precision-Recall Tradeoff

\`\`\`python
print("\\n" + "="*70)
print("Precision and Recall")
print("="*70)

# Calculate for breast cancer model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"\\nMetrics for breast cancer model:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Manual calculation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision_manual = tp / (tp + fp)
recall_manual = tp / (tp + fn)

print(f"\\nManual calculation:")
print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision_manual:.4f} ✓")
print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall_manual:.4f} ✓")

print(f"\\nInterpretation:")
print(f"Precision: {precision*100:.1f}% of predicted benign cases are truly benign")
print(f"Recall: We correctly identify {recall*100:.1f}% of all benign cases")

# Demonstrate tradeoff
print("\\n" + "="*70)
print("Precision-Recall Tradeoff")
print("="*70)

# Get prediction probabilities
y_prob = model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.5, 0.7, 0.9]
print(f"\\n{'Threshold':<12s} {'Precision':<12s} {'Recall':<12s} {'F1':<12s} {'Predicted+':<12s}")
print("-"*70)

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    n_positive = y_pred_thresh.sum()
    
    print(f"{thresh:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {n_positive:<12d}")

print("\\nObservation:")
print("  Higher threshold → Higher precision, lower recall (fewer positive predictions)")
print("  Lower threshold → Lower precision, higher recall (more positive predictions)")
\`\`\`

### When to Optimize for Precision vs Recall

\`\`\`python
print("\\n" + "="*70)
print("Choosing Between Precision and Recall")
print("="*70)

scenarios = [
    {
        'name': 'Email Spam Detection',
        'optimize': 'Precision',
        'reason': "Don't want legitimate emails in spam (false positive worse)",
        'acceptable_tradeoff': 'Some spam in inbox (false negative)'
    },
    {
        'name': 'Cancer Screening',
        'optimize': 'Recall',
        'reason': "Must catch all cancer cases (false negative catastrophic)",
        'acceptable_tradeoff': 'Some false alarms (false positive)'
    },
    {
        'name': 'Fraud Detection',
        'optimize': 'Recall',
        'reason': "Must catch fraudulent transactions",
        'acceptable_tradeoff': 'Some legitimate transactions flagged'
    },
    {
        'name': 'Search Engine',
        'optimize': 'Precision',
        'reason': "Users want relevant results (precision)",
        'acceptable_tradeoff': 'Missing some relevant pages (recall)'
    }
]

for scenario in scenarios:
    print(f"\\n{scenario['name']}:")
    print(f"  Optimize: {scenario['optimize']}")
    print(f"  Reason: {scenario['reason']}")
    print(f"  Acceptable: {scenario['acceptable_tradeoff']}")
\`\`\`

## F1 Score

### Definition

F1 score is the harmonic mean of precision and recall:

$$F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2TP}{2TP + FP + FN}$$

The harmonic mean penalizes extreme values more than arithmetic mean, requiring both precision and recall to be high.

\`\`\`python
print("\\n" + "="*70)
print("F1 Score")
print("="*70)

f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Manual calculation
f1_manual = 2 * (precision * recall) / (precision + recall)
print(f"\\nManual: F1 = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
print(f"       = {f1_manual:.4f} ✓")

# Compare with arithmetic mean
arithmetic_mean = (precision + recall) / 2
print(f"\\nHarmonic mean (F1): {f1:.4f}")
print(f"Arithmetic mean:    {arithmetic_mean:.4f}")

# Demonstrate why harmonic mean is better
print("\\n" + "="*70)
print("Why Harmonic Mean?")
print("="*70)

test_cases = [
    {'precision': 0.9, 'recall': 0.9},
    {'precision': 1.0, 'recall': 0.5},
    {'precision': 0.5, 'recall': 1.0},
    {'precision': 1.0, 'recall': 0.1},
]

print(f"\\n{'Precision':<12s} {'Recall':<12s} {'F1 (harm)':<12s} {'Arith Mean':<12s}")
print("-"*70)

for case in test_cases:
    p, r = case['precision'], case['recall']
    f1_val = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    arith = (p + r) / 2
    
    print(f"{p:<12.1f} {r:<12.1f} {f1_val:<12.4f} {arith:<12.4f}")

print("\\nObservation: F1 heavily penalizes imbalanced precision/recall")
\`\`\`

### F-Beta Score: Weighted F1

\`\`\`python
from sklearn.metrics import fbeta_score

print("\\n" + "="*70)
print("F-Beta Score: Adjusting Precision-Recall Importance")
print("="*70)

# β controls the weight
# β < 1: Favor precision
# β = 1: Equal weight (F1)
# β > 1: Favor recall

betas = [0.5, 1.0, 2.0]

print(f"\\n{'Beta':<8s} {'F-Beta':<12s} {'Meaning':<40s}")
print("-"*70)

for beta in betas:
    fb = fbeta_score(y_test, y_pred, beta=beta)
    
    if beta < 1:
        meaning = "Weights precision higher than recall"
    elif beta == 1:
        meaning = "Equal weight (F1 score)"
    else:
        meaning = "Weights recall higher than precision"
    
    print(f"{beta:<8.1f} {fb:<12.4f} {meaning:<40s}")

print(f"\\nFormula: F_β = (1 + β²) · (precision · recall) / (β² · precision + recall)")
print(f"\\nUse F2 for recall-critical applications (e.g., disease detection)")
print(f"Use F0.5 for precision-critical applications (e.g., spam detection)")
\`\`\`

## Specificity and Sensitivity

\`\`\`python
print("\\n" + "="*70)
print("Specificity and Sensitivity")
print("="*70)

# Sensitivity = Recall = TPR
sensitivity = recall_score(y_test, y_pred)

# Specificity = TNR (True Negative Rate)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"Sensitivity (Recall, TPR): {sensitivity:.4f}")
print(f"Specificity (TNR):         {specificity:.4f}")

print(f"\\nInterpretation:")
print(f"Sensitivity: Of all actual positive cases, we detected {sensitivity*100:.1f}%")
print(f"Specificity: Of all actual negative cases, we correctly identified {specificity*100:.1f}%")

# False Positive Rate and False Negative Rate
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"\\nRelated metrics:")
print(f"False Positive Rate (FPR): {fpr:.4f} (1 - Specificity)")
print(f"False Negative Rate (FNR): {fnr:.4f} (1 - Sensitivity)")

print(f"\\nVerification:")
print(f"Sensitivity + FNR = {sensitivity:.4f} + {fnr:.4f} = {sensitivity + fnr:.4f} ✓")
print(f"Specificity + FPR = {specificity:.4f} + {fpr:.4f} = {specificity + fpr:.4f} ✓")
\`\`\`

## ROC Curve and AUC

### ROC Curve (Receiver Operating Characteristic)

The ROC curve plots True Positive Rate (Recall) vs False Positive Rate at various threshold settings.

\`\`\`python
from sklearn.metrics import roc_curve, roc_auc_score

print("\\n" + "="*70)
print("ROC Curve and AUC")
print("="*70)

# Get prediction probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr_list, tpr_list, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(10, 8))

plt.plot(fpr_list, tpr_list, linewidth=2, label=f'Model (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.50)')

# Fill area under curve
plt.fill_between(fpr_list, tpr_list, alpha=0.3)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)

# Add some threshold annotations
for i in [len(thresholds)//4, len(thresholds)//2, 3*len(thresholds)//4]:
    plt.plot(fpr_list[i], tpr_list[i], 'ro', markersize=8)
    plt.annotate(f'thresh={thresholds[i]:.2f}', 
                xy=(fpr_list[i], tpr_list[i]),
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=9)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
print("\\nROC curve saved to 'roc_curve.png'")

# Interpretation
print(f"\\nAUC Interpretation:")
print(f"  AUC = {roc_auc:.4f}")
if roc_auc >= 0.9:
    print("  → Excellent discrimination")
elif roc_auc >= 0.8:
    print("  → Good discrimination")
elif roc_auc >= 0.7:
    print("  → Fair discrimination")
elif roc_auc >= 0.6:
    print("  → Poor discrimination")
else:
    print("  → No discrimination (random)")

print(f"\\nMeaning: {roc_auc*100:.1f}% probability that the model ranks a random")
print(f"positive example higher than a random negative example")
\`\`\`

### Comparing Multiple Models with ROC

\`\`\`python
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("\\n" + "="*70)
print("Comparing Multiple Models with ROC")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
}

plt.figure(figsize=(10, 8))

results = []

for name, clf in models.items():
    clf.fit(X_train, y_train)
    
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})')
    
    results.append({'Model': name, 'AUC': auc})

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
print("\\nROC comparison saved to 'roc_comparison.png'")

# Print ranking
import pandas as pd
df_results = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\\nModel Ranking by AUC:")
print(df_results.to_string(index=False))
\`\`\`

## Precision-Recall Curve

For imbalanced datasets, the Precision-Recall curve is often more informative than ROC.

\`\`\`python
from sklearn.metrics import precision_recall_curve, average_precision_score

print("\\n" + "="*70)
print("Precision-Recall Curve")
print("="*70)

# Calculate PR curve
y_prob = model.predict_proba(X_test)[:, 1]
precision_list, recall_list, thresholds_pr = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

print(f"Average Precision Score: {avg_precision:.4f}")

# Plot PR curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# PR Curve
ax1.plot(recall_list, precision_list, linewidth=2, 
         label=f'Model (AP={avg_precision:.3f})')

# Baseline (random classifier)
baseline = (y_test == 1).sum() / len(y_test)
ax1.plot([0, 1], [baseline, baseline], 'k--', linewidth=2,
         label=f'Random (AP={baseline:.3f})')

ax1.fill_between(recall_list, precision_list, alpha=0.3)
ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim([-0.05, 1.05])
ax1.set_xlabel('Recall', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(alpha=0.3)

# Side by side comparison with ROC
fpr_list, tpr_list, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

ax2.plot(fpr_list, tpr_list, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
ax2.fill_between(fpr_list, tpr_list, alpha=0.3)
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve (for comparison)', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pr_vs_roc.png', dpi=150, bbox_inches='tight')
print("\\nPR vs ROC comparison saved to 'pr_vs_roc.png'")

print("\\nWhen to use PR vs ROC:")
print("  • Use PR curve for imbalanced datasets")
print("  • Use ROC curve for balanced datasets")
print("  • PR curve more informative when positive class is rare")
\`\`\`

## Log Loss (Cross-Entropy Loss)

\`\`\`python
from sklearn.metrics import log_loss

print("\\n" + "="*70)
print("Log Loss (Cross-Entropy)")
print("="*70)

# Calculate log loss
y_prob = model.predict_proba(X_test)
logloss = log_loss(y_test, y_prob)

print(f"Log Loss: {logloss:.4f}")

# Manual calculation for binary classification
epsilon = 1e-15  # To avoid log(0)
y_prob_clipped = np.clip(y_prob[:, 1], epsilon, 1 - epsilon)
logloss_manual = -np.mean(
    y_test * np.log(y_prob_clipped) + (1 - y_test) * np.log(1 - y_prob_clipped)
)

print(f"Manual calculation: {logloss_manual:.4f} ✓")

print(f"\\nInterpretation:")
print(f"  Lower is better (0 is perfect)")
print(f"  Penalizes confident wrong predictions heavily")

# Demonstrate penalty for wrong confidence
print("\\n" + "="*70)
print("Log Loss Penalty Examples:")
print("="*70)

test_cases = [
    {'true': 1, 'pred_prob': 0.9, 'desc': 'Correct & confident'},
    {'true': 1, 'pred_prob': 0.6, 'desc': 'Correct & uncertain'},
    {'true': 1, 'pred_prob': 0.4, 'desc': 'Wrong & uncertain'},
    {'true': 1, 'pred_prob': 0.1, 'desc': 'Wrong & confident'},
]

print(f"\\n{'True':<6s} {'Pred Prob':<12s} {'Log Loss':<12s} {'Description':<25s}")
print("-"*70)

for case in test_cases:
    true_label = case['true']
    pred_prob = case['pred_prob']
    
    # Calculate log loss for single prediction
    loss = -np.log(pred_prob if true_label == 1 else 1 - pred_prob)
    
    print(f"{true_label:<6d} {pred_prob:<12.1f} {loss:<12.4f} {case['desc']:<25s}")

print("\\nObservation: Confident wrong predictions are heavily penalized!")
\`\`\`

## Matthews Correlation Coefficient (MCC)

\`\`\`python
from sklearn.metrics import matthews_corrcoef

print("\\n" + "="*70)
print("Matthews Correlation Coefficient (MCC)")
print("="*70)

mcc = matthews_corrcoef(y_test, y_pred)

print(f"MCC: {mcc:.4f}")

# Manual calculation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

numerator = (tp * tn) - (fp * fn)
denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

if denominator == 0:
    mcc_manual = 0
else:
    mcc_manual = numerator / denominator

print(f"Manual: {mcc_manual:.4f} ✓")

print(f"\\nMCC Range: -1 to +1")
print(f"  +1: Perfect prediction")
print(f"   0: Random prediction")
print(f"  -1: Perfect disagreement")

print(f"\\nAdvantages:")
print("  • Works well for imbalanced datasets")
print("  • Considers all four confusion matrix values")
print("  • Symmetric (treats both classes equally)")

# Compare metrics on imbalanced data
print("\\n" + "="*70)
print("MCC vs Other Metrics on Imbalanced Data:")
print("="*70)

real_model_imb = RandomForestClassifier(n_estimators=100, random_state=42)
real_model_imb.fit(X_train_imb, y_train_imb)
y_pred_imb = real_model_imb.predict(X_test_imb)

metrics_imb = {
    'Accuracy': accuracy_score(y_test_imb, y_pred_imb),
    'Precision': precision_score(y_test_imb, y_pred_imb, zero_division=0),
    'Recall': recall_score(y_test_imb, y_pred_imb, zero_division=0),
    'F1': f1_score(y_test_imb, y_pred_imb, zero_division=0),
    'MCC': matthews_corrcoef(y_test_imb, y_pred_imb)
}

print(f"\\nClass distribution: {(y_test_imb == 0).sum()} neg, {(y_test_imb == 1).sum()} pos")
for metric, value in metrics_imb.items():
    print(f"{metric:<12s}: {value:.4f}")

print("\\n✓ MCC provides balanced view even with class imbalance")
\`\`\`

## Cohen's Kappa

\`\`\`python
from sklearn.metrics import cohen_kappa_score

print("\\n" + "="*70)
print("Cohen's Kappa")
print("="*70)

kappa = cohen_kappa_score(y_test, y_pred)

print(f"Cohen's Kappa: {kappa:.4f}")

print(f"\\nInterpretation:")
if kappa < 0:
    print("  → Less agreement than random")
elif kappa < 0.2:
    print("  → Slight agreement")
elif kappa < 0.4:
    print("  → Fair agreement")
elif kappa < 0.6:
    print("  → Moderate agreement")
elif kappa < 0.8:
    print("  → Substantial agreement")
else:
    print("  → Almost perfect agreement")

print(f"\\nKappa accounts for agreement by chance")
print(f"Useful when comparing human annotators or models")
\`\`\`

## Comprehensive Classification Report

\`\`\`python
print("\\n" + "="*70)
print("Comprehensive Classification Report")
print("="*70)

# Sklearn's classification report
print("\\nScikit-learn Classification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['Malignant', 'Benign']))

# Custom comprehensive report
def comprehensive_classification_report(y_true, y_pred, y_prob=None):
    """Generate detailed classification metrics report."""
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate all metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'FPR': fp / (fp + tn),
        'FNR': fn / (fn + tp),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_prob)
        metrics['PR AUC'] = average_precision_score(y_true, y_prob)
        metrics['Log Loss'] = log_loss(y_true, 
                                      np.column_stack([1-y_prob, y_prob]))
    
    return metrics, cm

y_prob = model.predict_proba(X_test)[:, 1]
metrics, cm = comprehensive_classification_report(y_test, y_pred, y_prob)

print("\\nDetailed Metrics:")
print("-"*70)
for metric, value in metrics.items():
    print(f"{metric:<20s}: {value:>8.4f}")

print(f"\\nConfusion Matrix:")
print(cm)

# Visual summary
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'])
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')
ax1.set_title('Confusion Matrix', fontweight='bold')

# 2. Metrics Bar Chart
ax2 = axes[0, 1]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
metric_values = [metrics[m] for m in metric_names]
bars = ax2.barh(metric_names, metric_values)
ax2.set_xlim([0, 1])
ax2.set_xlabel('Score')
ax2.set_title('Key Metrics', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Color bars
colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' 
         for v in metric_values]
for bar, color in zip(bars, colors):
    bar.set_color(color)
    bar.set_alpha(0.7)

# 3. ROC Curve
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax3.plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['ROC AUC']:.3f}")
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax3.fill_between(fpr, tpr, alpha=0.3)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Precision-Recall Curve
ax4 = axes[1, 1]
precision_list, recall_list, _ = precision_recall_curve(y_test, y_prob)
ax4.plot(recall_list, precision_list, linewidth=2, 
        label=f"AP = {metrics['PR AUC']:.3f}")
ax4.fill_between(recall_list, precision_list, alpha=0.3)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('classification_summary.png', dpi=150, bbox_inches='tight')
print("\\nComprehensive summary saved to 'classification_summary.png'")
\`\`\`

## Key Takeaways

1. **Confusion Matrix**: Foundation of all classification metrics
2. **Accuracy**: Simple but misleading for imbalanced data
3. **Precision**: Of predicted positives, how many are correct?
4. **Recall**: Of actual positives, how many did we catch?
5. **F1 Score**: Harmonic mean of precision and recall
6. **ROC AUC**: Overall discrimination ability (balanced data)
7. **PR AUC**: Better for imbalanced data
8. **MCC**: Balanced metric considering all confusion matrix values

**For Trading Applications:**
- Precision: Avoid false buy signals (minimize false positives)
- Recall: Catch all profitable opportunities (minimize false negatives)
- F1: Balance between the two
- ROC AUC: Overall model quality

## Further Reading

- Powers, D. M. (2011). "Evaluation: from precision, recall and F-measure to ROC"
- Saito, T., & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot"
- Chicco, D., & Jurman, G. (2020). "The advantages of the Matthews correlation coefficient (MCC)"
`,
  exercises: [
    {
      prompt:
        'Build a classification metrics dashboard that takes true labels and predictions, calculates all major metrics, generates visualizations, and provides actionable recommendations based on the results. Include threshold optimization.',
      solution: `
# Solution provided in next file due to length
`,
    },
  ],
  quizId: 'model-evaluation-optimization-classification-metrics',
  multipleChoiceId: 'model-evaluation-optimization-classification-metrics-mc',
};
