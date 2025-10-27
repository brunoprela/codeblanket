/**
 * Section: Anomaly Detection
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive coverage of anomaly detection using unsupervised methods
 */

export const anomalyDetection = {
  id: 'anomaly-detection',
  title: 'Anomaly Detection',
  content: `
# Anomaly Detection

## Introduction

Anomaly detection (also called outlier detection or novelty detection) is the identification of rare items, events, or observations that differ significantly from the majority of the data. It\'s a critical unsupervised learning technique used in:

- **Fraud Detection**: Credit card fraud, insurance fraud
- **Intrusion Detection**: Network security, cybersecurity
- **Manufacturing**: Defect detection, quality control
- **Healthcare**: Disease outbreak detection, medical diagnosis
- **Predictive Maintenance**: Equipment failure prediction

**Key Challenge**: Anomalies are rare by definition, often< 1% of data!

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate normal data
np.random.seed(42)
X_normal = np.random.randn(300, 2) * 0.5 + np.array([0, 0])

# Generate anomalies
X_anomalies = np.random.uniform (low=-4, high=4, size=(20, 2))

# Combine
X = np.vstack([X_normal, X_anomalies])
y = np.array([0]*300 + [1]*20)  # 0=normal, 1=anomaly

plt.figure (figsize=(10, 8))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.5, s=30, label='Normal')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.8, s=100,
           marker='*', edgecolors='black', linewidths=1, label='Anomalies')
plt.title('Example Data with Anomalies')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Normal samples: {sum (y==0)} ({sum (y==0)/len (y)*100:.1f}%)")
print(f"Anomalies: {sum (y==1)} ({sum (y==1)/len (y)*100:.1f}%)")
\`\`\`

## Statistical Methods

### Z-Score Method

Identifies points that are many standard deviations away from the mean

**Threshold**: Typically |z| > 3 indicates anomaly

\`\`\`python
# Calculate Z-scores
from scipy import stats

z_scores = np.abs (stats.zscore(X))
threshold = 3

# Points with z-score > 3 in any dimension are anomalies
anomalies_zscore = (z_scores > threshold).any (axis=1)

print(f"Z-score method detected: {sum (anomalies_zscore)} anomalies")

plt.figure (figsize=(10, 8))
plt.scatter(X[~anomalies_zscore, 0], X[~anomalies_zscore, 1],
           c='blue', alpha=0.5, s=30, label='Normal')
plt.scatter(X[anomalies_zscore, 0], X[anomalies_zscore, 1],
           c='red', alpha=0.8, s=100, marker='*',
           edgecolors='black', linewidths=1, label='Detected Anomalies')
plt.title('Anomaly Detection: Z-Score Method')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Assumes normal distribution!
# Doesn't work well for multivariate outliers
\`\`\`

### IQR (Interquartile Range) Method

Less sensitive to extreme values than z-score

**Threshold**: Points outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are outliers

\`\`\`python
def detect_outliers_iqr (data):
    ''Detect outliers using IQR method''
    Q1 = np.percentile (data, 25, axis=0)
    Q3 = np.percentile (data, 75, axis=0)
    IQR = Q3 - Q1

    # Outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Points outside boundaries
    outliers = ((data < lower_bound) | (data > upper_bound)).any (axis=1)
    return outliers

anomalies_iqr = detect_outliers_iqr(X)
print(f"IQR method detected: {sum (anomalies_iqr)} anomalies")

plt.figure (figsize=(10, 8))
plt.scatter(X[~anomalies_iqr, 0], X[~anomalies_iqr, 1],
           c='blue', alpha=0.5, s=30, label='Normal')
plt.scatter(X[anomalies_iqr, 0], X[anomalies_iqr, 1],
           c='red', alpha=0.8, s=100, marker='*',
           edgecolors='black', linewidths=1, label='Detected Anomalies')
plt.title('Anomaly Detection: IQR Method')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# More robust than z-score
# Still univariate (checks each feature independently)
\`\`\`

## Isolation Forest

A powerful tree-based anomaly detection method

**Core Idea**: Anomalies are easier to isolate (fewer splits needed) than normal points

\`\`\`python
from sklearn.ensemble import IsolationForest

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of anomalies
    random_state=42,
    n_estimators=100
)

# Fit and predict
y_pred_iso = iso_forest.fit_predict(X)
# -1 = anomaly, 1 = normal

anomalies_iso = (y_pred_iso == -1)
print(f"Isolation Forest detected: {sum (anomalies_iso)} anomalies")

# Get anomaly scores
anomaly_scores = iso_forest.score_samples(X)
# More negative = more anomalous

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[~anomalies_iso, 0], X[~anomalies_iso, 1],
           c='blue', alpha=0.5, s=30, label='Normal')
plt.scatter(X[anomalies_iso, 0], X[anomalies_iso, 1],
           c='red', alpha=0.8, s=100, marker='*',
           edgecolors='black', linewidths=1, label='Anomalies')
plt.title('Isolation Forest: Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=anomaly_scores,
                     cmap='RdYlBu_r', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.colorbar (scatter, label='Anomaly Score\\n (lower = more anomalous)')
plt.title('Isolation Forest: Anomaly Scores')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Advantages:
# - Fast and scalable
# - Works well in high dimensions
# - Doesn't assume any distribution
# - Provides anomaly scores
\`\`\`

### How Isolation Forest Works

\`\`\`python
# Visualize isolation concept
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Normal point (hard to isolate)
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.3, s=20)
normal_point = X[y==0][0]
axes[0].scatter (normal_point[0], normal_point[1], c='green', s=200,
               marker='o', edgecolors='black', linewidths=2, label='Normal Point')
axes[0].set_title('Normal Point\\n(Many splits needed to isolate)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Anomaly (easy to isolate)
axes[1].scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.3, s=20)
anomaly_point = X[y==1][0]
axes[1].scatter (anomaly_point[0], anomaly_point[1], c='red', s=200,
               marker='*', edgecolors='black', linewidths=2, label='Anomaly')
axes[1].set_title('Anomaly\\n(Few splits needed to isolate)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Isolation Forest builds random decision trees")
print("Anomalies are isolated faster (shorter path lengths)")
\`\`\`

### Tuning Isolation Forest

\`\`\`python
# Effect of contamination parameter
contamination_values = [0.05, 0.1, 0.15, 0.2]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, cont in enumerate (contamination_values):
    iso_temp = IsolationForest (contamination=cont, random_state=42)
    y_pred_temp = iso_temp.fit_predict(X)
    anomalies_temp = (y_pred_temp == -1)

    axes[idx].scatter(X[~anomalies_temp, 0], X[~anomalies_temp, 1],
                     c='blue', alpha=0.5, s=30)
    axes[idx].scatter(X[anomalies_temp, 0], X[anomalies_temp, 1],
                     c='red', alpha=0.8, s=100, marker='*',
                     edgecolors='black', linewidths=1)
    axes[idx].set_title (f'Contamination = {cont}\\n{sum (anomalies_temp)} anomalies detected')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Effect of Contamination Parameter', fontsize=14)
plt.tight_layout()
plt.show()

print("Contamination = expected proportion of anomalies")
print("Set based on domain knowledge or cross-validation")
\`\`\`

## Local Outlier Factor (LOF)

Identifies outliers based on local density

**Core Idea**: Anomalies have lower density than their neighbors

\`\`\`python
from sklearn.neighbors import LocalOutlierFactor

# Train LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=False  # False = fit_predict, True = fit then predict
)

# Fit and predict
y_pred_lof = lof.fit_predict(X)
anomalies_lof = (y_pred_lof == -1)

# Get LOF scores
lof_scores = -lof.negative_outlier_factor_  # Higher = more anomalous

print(f"LOF detected: {sum (anomalies_lof)} anomalies")

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[~anomalies_lof, 0], X[~anomalies_lof, 1],
           c='blue', alpha=0.5, s=30, label='Normal')
plt.scatter(X[anomalies_lof, 0], X[anomalies_lof, 1],
           c='red', alpha=0.8, s=100, marker='*',
           edgecolors='black', linewidths=1, label='Anomalies')
plt.title('LOF: Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=lof_scores,
                     cmap='RdYlBu_r', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.colorbar (scatter, label='LOF Score\\n (higher = more anomalous)')
plt.title('LOF: Anomaly Scores')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Advantages:
# - Detects local anomalies (points in low-density regions)
# - Works well with varying density
# - Provides interpretable scores
\`\`\`

### LOF vs Isolation Forest

\`\`\`python
# Generate data with varying density
np.random.seed(42)
X_dense = np.random.randn(200, 2) * 0.3
X_sparse = np.random.randn(100, 2) * 1.5 + np.array([4, 4])
X_varying = np.vstack([X_dense, X_sparse])

# Add a local anomaly in dense region
X_local_anomaly = np.array([[0.5, 1.5]])
X_varying = np.vstack([X_varying, X_local_anomaly])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Original data
axes[0].scatter(X_dense[:, 0], X_dense[:, 1], c='blue', alpha=0.5, s=20, label='Dense cluster')
axes[0].scatter(X_sparse[:, 0], X_sparse[:, 1], c='lightblue', alpha=0.5, s=20, label='Sparse cluster')
axes[0].scatter(X_local_anomaly[0, 0], X_local_anomaly[0, 1],
               c='red', s=200, marker='*', edgecolors='black', linewidths=2, label='Local anomaly')
axes[0].set_title('Data with Varying Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Isolation Forest
iso_varying = IsolationForest (contamination=0.05, random_state=42)
y_pred_iso_varying = iso_varying.fit_predict(X_varying)
anomalies_iso_varying = (y_pred_iso_varying == -1)

axes[1].scatter(X_varying[~anomalies_iso_varying, 0], X_varying[~anomalies_iso_varying, 1],
               c='blue', alpha=0.5, s=30)
axes[1].scatter(X_varying[anomalies_iso_varying, 0], X_varying[anomalies_iso_varying, 1],
               c='red', alpha=0.8, s=100, marker='*', edgecolors='black', linewidths=1)
axes[1].set_title (f'Isolation Forest\\n{sum (anomalies_iso_varying)} anomalies')
axes[1].grid(True, alpha=0.3)

# LOF
lof_varying = LocalOutlierFactor (n_neighbors=20, contamination=0.05)
y_pred_lof_varying = lof_varying.fit_predict(X_varying)
anomalies_lof_varying = (y_pred_lof_varying == -1)

axes[2].scatter(X_varying[~anomalies_lof_varying, 0], X_varying[~anomalies_lof_varying, 1],
               c='blue', alpha=0.5, s=30)
axes[2].scatter(X_varying[anomalies_lof_varying, 0], X_varying[anomalies_lof_varying, 1],
               c='red', alpha=0.8, s=100, marker='*', edgecolors='black', linewidths=1)
axes[2].set_title (f'LOF\\n{sum (anomalies_lof_varying)} anomalies')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("LOF is better at detecting local anomalies in varying density data")
\`\`\`

## One-Class SVM

Uses SVM to learn the boundary of normal data

\`\`\`python
from sklearn.svm import OneClassSVM

# Train One-Class SVM
oc_svm = OneClassSVM(
    nu=0.1,  # Upper bound on fraction of outliers (similar to contamination)
    kernel='rbf',
    gamma='auto'
)

y_pred_svm = oc_svm.fit_predict(X)
anomalies_svm = (y_pred_svm == -1)

print(f"One-Class SVM detected: {sum (anomalies_svm)} anomalies")

# Decision function (distance to separating hyperplane)
decision_scores = oc_svm.decision_function(X)

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[~anomalies_svm, 0], X[~anomalies_svm, 1],
           c='blue', alpha=0.5, s=30, label='Normal')
plt.scatter(X[anomalies_svm, 0], X[anomalies_svm, 1],
           c='red', alpha=0.8, s=100, marker='*',
           edgecolors='black', linewidths=1, label='Anomalies')
plt.title('One-Class SVM: Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=decision_scores,
                     cmap='RdYlBu_r', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.colorbar (scatter, label='Decision Score\\n (lower = more anomalous)')
plt.title('One-Class SVM: Decision Scores')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Advantages:
# - Flexible with kernel trick
# - Works well in high dimensions
# - Theoretical foundation (SVM)

# Disadvantages:
# - Slower than Isolation Forest
# - Sensitive to parameters (nu, gamma)
\`\`\`

## DBSCAN for Anomaly Detection

DBSCAN can identify outliers as noise points

\`\`\`python
from sklearn.cluster import DBSCAN

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# -1 = noise (potential anomalies)
anomalies_dbscan = (labels == -1)

print(f"DBSCAN identified: {sum (anomalies_dbscan)} noise points (potential anomalies)")

plt.figure (figsize=(10, 8))

# Plot clusters
for label in set (labels):
    if label == -1:
        # Noise points
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1],
                   c='red', marker='*', s=200,
                   edgecolors='black', linewidths=2, label='Noise/Anomalies')
    else:
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1],
                   s=30, alpha=0.6, label=f'Cluster {label}')

plt.title('DBSCAN for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("DBSCAN noise points = points in low-density regions")
\`\`\`

## Comparison of Methods

\`\`\`python
# Compare all methods
from sklearn.metrics import precision_score, recall_score, f1_score

methods = {
    'Z-Score': anomalies_zscore,
    'IQR': anomalies_iqr,
    'Isolation Forest': anomalies_iso,
    'LOF': anomalies_lof,
    'One-Class SVM': anomalies_svm,
    'DBSCAN': anomalies_dbscan
}

results = []
for name, predictions in methods.items():
    precision = precision_score (y, predictions)
    recall = recall_score (y, predictions)
    f1 = f1_score (y, predictions)
    n_detected = sum (predictions)

    results.append({
        'Method': name,
        'Detected': n_detected,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

import pandas as pd
results_df = pd.DataFrame (results)
print(results_df.to_string (index=False))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

results_df.plot (x='Method', y='Precision', kind='bar', ax=axes[0], legend=False, color='steelblue')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision')
axes[0].set_xticklabels (axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')

results_df.plot (x='Method', y='Recall', kind='bar', ax=axes[1], legend=False, color='orange')
axes[1].set_ylabel('Recall')
axes[1].set_title('Recall')
axes[1].set_xticklabels (axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')

results_df.plot (x='Method', y='F1', kind='bar', ax=axes[2], legend=False, color='green')
axes[2].set_ylabel('F1 Score')
axes[2].set_title('F1 Score')
axes[2].set_xticklabels (axes[2].get_xticklabels(), rotation=45, ha='right')
axes[2].set_ylim([0, 1])
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('Comparison of Anomaly Detection Methods', fontsize=14)
plt.tight_layout()
plt.show()
\`\`\`

## Evaluation Metrics for Anomaly Detection

### Confusion Matrix

\`\`\`python
from sklearn.metrics import confusion_matrix, classification_report

# Example: Isolation Forest
cm = confusion_matrix (y, anomalies_iso)

plt.figure (figsize=(8, 6))
plt.imshow (cm, cmap='Blues', alpha=0.7)
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text (j, i, cm[i, j], ha='center', va='center', fontsize=16, fontweight='bold')

plt.xticks([0, 1], ['Predicted Normal', 'Predicted Anomaly'])
plt.yticks([0, 1], ['Actual Normal', 'Actual Anomaly'])
plt.title('Confusion Matrix: Isolation Forest')
plt.tight_layout()
plt.show()

print("Classification Report:")
print(classification_report (y, anomalies_iso, target_names=['Normal', 'Anomaly']))

# In real scenarios, we often don't have labels!
# Use domain experts to validate detected anomalies
\`\`\`

### Precision-Recall Trade-off

\`\`\`python
from sklearn.metrics import precision_recall_curve, auc

# Use anomaly scores from Isolation Forest
anomaly_scores_iso = -iso_forest.score_samples(X)  # Negate so higher = more anomalous

precision, recall, thresholds = precision_recall_curve (y, anomaly_scores_iso)

plt.figure (figsize=(10, 6))
plt.plot (recall, precision, linewidth=2, label=f'PR curve (AUC={auc (recall, precision):.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve: Isolation Forest', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Choose threshold based on business requirements
# High precision: Minimize false alarms
# High recall: Catch all anomalies
\`\`\`

## Real-World Applications

### Application 1: Credit Card Fraud Detection

\`\`\`python
# Simulated credit card transactions
np.random.seed(42)
n_transactions = 1000

# Normal transactions
normal_transactions = np.random.randn(950, 5) * np.array([100, 50, 10, 5, 2])

# Fraudulent transactions (unusual patterns)
fraud_transactions = np.random.randn(50, 5) * np.array([500, 200, 50, 20, 10]) + np.array([200, 100, 30, 15, 5])

X_transactions = np.vstack([normal_transactions, fraud_transactions])
y_transactions = np.array([0]*950 + [1]*50)

print(f"Total transactions: {len(X_transactions)}")
print(f"Fraud rate: {sum (y_transactions)/len (y_transactions)*100:.2f}%")

# Scale features
scaler = StandardScaler()
X_transactions_scaled = scaler.fit_transform(X_transactions)

# Apply Isolation Forest
iso_fraud = IsolationForest (contamination=0.05, random_state=42)
y_pred_fraud = iso_fraud.fit_predict(X_transactions_scaled)
detected_fraud = (y_pred_fraud == -1)

# Evaluate
from sklearn.metrics import classification_report

print("\\nFraud Detection Results:")
print(classification_report (y_transactions, detected_fraud,
                          target_names=['Legitimate', 'Fraud']))

# In production:
# - Flag detected frauds for manual review
# - Update model with confirmed frauds
# - Monitor precision/recall over time
\`\`\`

### Application 2: Manufacturing Defect Detection

\`\`\`python
# Simulated sensor data from manufacturing line
n_samples = 500

# Normal products
normal_products = np.random.randn(480, 3) * np.array([0.5, 0.3, 0.2])

# Defective products (out of spec)
defective_products = np.random.randn(20, 3) * np.array([1.5, 1.0, 0.8]) + np.array([2, 1.5, 1])

X_manufacturing = np.vstack([normal_products, defective_products])
y_manufacturing = np.array([0]*480 + [1]*20)

feature_names = ['Temperature', 'Pressure', 'Vibration']

# Scale
X_manufacturing_scaled = StandardScaler().fit_transform(X_manufacturing)

# Apply LOF
lof_defects = LocalOutlierFactor (n_neighbors=20, contamination=0.05)
y_pred_defects = lof_defects.fit_predict(X_manufacturing_scaled)
detected_defects = (y_pred_defects == -1)

print(f"Defect Detection Results:")
print(f"Actual defects: {sum (y_manufacturing)}")
print(f"Detected defects: {sum (detected_defects)}")

# Visualize first two features
plt.figure (figsize=(10, 8))
plt.scatter(X_manufacturing[~detected_defects, 0], X_manufacturing[~detected_defects, 1],
           c='blue', alpha=0.5, s=30, label='Normal')
plt.scatter(X_manufacturing[detected_defects, 0], X_manufacturing[detected_defects, 1],
           c='red', alpha=0.8, s=100, marker='*',
           edgecolors='black', linewidths=1, label='Defects')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('Manufacturing Defect Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Best Practices

1. **Understand Your Data**:
   - What constitutes an anomaly?
   - How rare are anomalies?
   - What features are relevant?

2. **Choose the Right Method**:
   - **Isolation Forest**: General purpose, fast, high-dimensional
   - **LOF**: Varying density, local anomalies
   - **One-Class SVM**: Non-linear boundaries, medium-sized data
   - **DBSCAN**: Clustering + anomaly detection
   - **Statistical**: Simple baseline, assumes distribution

3. **Feature Engineering**:
   - Domain-specific features
   - Time-based features (for time series)
   - Aggregations and ratios

4. **Validation**:
   - Use domain experts
   - Monitor false positive rate
   - Update model with feedback

5. **Handle Imbalance**:
   - Anomalies are rare
   - Use appropriate metrics (precision, recall, F1)
   - Don't rely solely on accuracy

## Summary

**Key Takeaways**:

- **Anomaly detection** identifies rare, unusual patterns
- **Many algorithms available**: Statistical, Isolation Forest, LOF, One-Class SVM, DBSCAN
- **No labels required**: Unsupervised learning
- **Critical applications**: Fraud, security, manufacturing, healthcare

**Method Selection**:
- **Start with Isolation Forest**: Fast, scalable, no assumptions
- **Use LOF** for local anomalies in varying density
- **Consider One-Class SVM** for complex boundaries
- **Try DBSCAN** if you also need clustering

**Best Practices**:
1. Always scale features
2. Tune contamination parameter based on domain knowledge
3. Use anomaly scores, not just binary predictions
4. Validate with domain experts
5. Monitor and update over time

**Next**: We'll explore Association Rule Learning for discovering interesting patterns in transactional data.
`,
};
