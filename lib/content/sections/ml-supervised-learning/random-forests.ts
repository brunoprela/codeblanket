/**
 * Random Forests Section
 */

export const randomforestsSection = {
  id: 'random-forests',
  title: 'Random Forests',
  content: `# Random Forests

## Introduction

Random Forests are ensemble learning methods that combine multiple decision trees to create more robust and accurate predictions. By aggregating many weak learners, Random Forests overcome the limitations of individual trees while maintaining interpretability through feature importance.

**Key Innovation**: Randomness at two levels:
1. **Bagging**: Each tree trained on random sample of data
2. **Random feature selection**: Each split considers random subset of features

**Why It Works**:
- Reduces variance (overcomes tree instability)
- Maintains low bias (trees still flexible)
- Decorrelates trees (randomness prevents similar trees)

**Applications**:
- Kaggle competitions (baseline model)
- Feature selection and ranking
- Medical diagnosis
- Credit risk modeling
- Remote sensing (satellite imagery)

## Bootstrap Aggregating (Bagging)

**Concept**: Train multiple models on random samples with replacement, then average predictions.

\\[ \\text{For regression: } \\hat{y} = \\frac{1}{B}\\sum_{b=1}^{B}f_b(x) \\]
\\[ \\text{For classification: Majority vote} \\]

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_test_split import train_test_split

# Generate non-linear data
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare single tree vs bagged trees vs Random Forest
models = {
    'Single Tree': DecisionTreeClassifier(random_state=42),
    'Bagging (10 trees)': BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=10, random_state=42
    ),
    'Random Forest (10 trees)': RandomForestClassifier(
        n_estimators=10, random_state=42
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=20, edgecolors='k', label='Class 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=20, edgecolors='k', label='Class 1')
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    ax.set_title(f'{name}\\nTrain: {train_acc:.3f}, Test: {test_acc:.3f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

plt.tight_layout()
plt.show()

print("Notice: RF has smoother boundaries and better test accuracy!")
\`\`\`

## Random Feature Selection

At each split, Random Forest considers only a random subset of features. This decorrelates trees.

\\[ m_{try} = \\sqrt{p} \\text{ (classification)} \\]
\\[ m_{try} = \\frac{p}{3} \\text{ (regression)} \\]

Where \\( p \\) is total number of features.

\`\`\`python
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

print(f"Total features: {X_train.shape[1]}")
print(f"Default max_features for classification: sqrt({X_train.shape[1]}) ≈ {int(np.sqrt(X_train.shape[1]))}")

# Compare different max_features settings
max_features_options = [1, 5, 10, 20, 30, 'sqrt', 'log2', None]
results = []

for max_feat in max_features_options:
    rf = RandomForestClassifier(n_estimators=100, max_features=max_feat, random_state=42)
    rf.fit(X_train, y_train)
    
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    
    results.append({
        'max_features': str(max_feat),
        'train_acc': train_acc,
        'test_acc': test_acc
    })

import pandas as pd
df_results = pd.DataFrame(results)
print("\\nImpact of max_features:")
print(df_results.to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
x_pos = range(len(results))
plt.plot(x_pos, df_results['train_acc'], 'o-', label='Training', linewidth=2)
plt.plot(x_pos, df_results['test_acc'], 's-', label='Test', linewidth=2)
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.title('Effect of Random Feature Selection')
plt.xticks(x_pos, df_results['max_features'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Number of Trees

More trees = better performance (up to a point), but longer training.

\`\`\`python
# Study impact of n_estimators
n_estimators_range = [1, 5, 10, 25, 50, 100, 200, 500]
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='Training', linewidth=2)
plt.plot(n_estimators_range, test_scores, 's-', label='Test', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Performance vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.show()

print("Test accuracy plateaus after ~100 trees typically.")
\`\`\`

## Out-of-Bag (OOB) Error

Each tree is trained on ~63% of data (bootstrap sample). Remaining ~37% is "out-of-bag" and can be used for validation without separate test set!

\`\`\`python
# OOB error estimation
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print("Out-of-Bag Score (internal validation):", rf.oob_score_)
print("Test Score:", rf.score(X_test, y_test))
print("\\nOOB score provides free validation estimate!")

# Track OOB error vs number of trees
oob_errors = []
n_trees = range(1, 201, 10)

for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

plt.figure(figsize=(10, 6))
plt.plot(n_trees, oob_errors, 'o-', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error vs Number of Trees')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Feature Importance

Random Forests provide robust feature importance by averaging across many trees.

\`\`\`python
# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(10, 8))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['importance'])
plt.yticks(range(len(top_15)), top_15['feature'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Compare: single tree vs Random Forest feature importance
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

comparison = pd.DataFrame({
    'feature': cancer.feature_names,
    'tree_importance': tree.feature_importances_,
    'rf_importance': rf.feature_importances_
})

print("\\nFeature Importance Correlation:")
print(f"Pearson correlation: {comparison['tree_importance'].corr(comparison['rf_importance']):.3f}")
print("\\nRandom Forest importance is more stable!")
\`\`\`

## Hyperparameter Tuning

Key hyperparameters:
- **n_estimators**: Number of trees (more is better, 100-500 typical)
- **max_depth**: Tree depth (None for full, or limit to prevent overfitting)
- **min_samples_split**: Minimum samples to split node
- **min_samples_leaf**: Minimum samples in leaf
- **max_features**: Features to consider at each split

\`\`\`python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Randomized search (faster than grid search)
rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,  # Try 20 random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_random.fit(X_train, y_train)

print("Best parameters:", rf_random.best_params_)
print(f"Best CV score: {rf_random.best_score_:.4f}")
print(f"Test score: {rf_random.score(X_test, y_test):.4f}")
\`\`\`

## Practical Considerations

### Class Imbalance

\`\`\`python
# Handling imbalanced classes
from sklearn.utils import class_weight

# Balanced Random Forest
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Adjust for class imbalance
    random_state=42
)

# Or use balanced_subsample for bootstrap sampling
rf_balanced_subsample = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced_subsample',
    random_state=42
)
\`\`\`

### Memory and Speed

\`\`\`python
import time

# Measure training time
start = time.time()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
train_time = time.time() - start

# Measure prediction time
start = time.time()
predictions = rf.predict(X_test)
predict_time = time.time() - start

print(f"Training time: {train_time:.2f}s")
print(f"Prediction time: {predict_time:.4f}s")
print(f"Time per prediction: {predict_time/len(X_test)*1000:.4f}ms")

# Memory usage
import sys
model_size_mb = sys.getsizeof(rf) / (1024**2)
print(f"Model size: {model_size_mb:.2f} MB")
\`\`\`

## Regression Example

\`\`\`python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = rf_reg.predict(X_test_reg)

print("="*60)
print("RANDOM FOREST REGRESSION")
print("="*60)
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
print(f"OOB Score: {RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42).fit(X_train_reg, y_train_reg).oob_score_:.4f}")

# Prediction intervals (using quantile regression)
from sklearn.ensemble import RandomForestQuantileRegressor

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression: Predicted vs Actual')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Advantages vs Disadvantages

**Advantages:**
- Highly accurate (state-of-the-art for many problems)
- Reduces overfitting compared to single trees
- Handles high-dimensional data
- Provides feature importance
- Robust to outliers and noise
- Little hyperparameter tuning needed
- Parallelizable (fast with n_jobs=-1)

**Disadvantages:**
- Less interpretable than single tree
- Larger model size (stores all trees)
- Slower prediction than single tree
- Can overfit noisy datasets
- Biased toward categorical features with many levels

## Real-World Example: Credit Risk

\`\`\`python
# Simulated credit risk dataset
np.random.seed(42)
n = 5000

credit_data = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.lognormal(10, 1, n),
    'debt_to_income': np.random.uniform(0, 1, n),
    'credit_score': np.random.randint(300, 850, n),
    'num_accounts': np.random.poisson(5, n),
    'delinquencies': np.random.poisson(0.5, n),
    'employment_years': np.random.randint(0, 30, n),
})

# Generate default probability
default_prob = 1 / (1 + np.exp(-(
    -5
    - 0.02 * credit_data['age']
    - 0.0001 * credit_data['income']
    + 2 * credit_data['debt_to_income']
    - 0.01 * credit_data['credit_score']
    + 0.1 * credit_data['delinquencies']
)))

credit_data['default'] = (np.random.random(n) < default_prob).astype(int)

X = credit_data.drop('default', axis=1)
y = credit_data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train optimized Random Forest
rf_credit = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_credit.fit(X_train, y_train)

from sklearn.metrics import classification_report, roc_auc_score, roc_curve

y_pred = rf_credit.predict(X_test)
y_pred_proba = rf_credit.predict_proba(X_test)[:, 1]

print("="*60)
print("CREDIT RISK PREDICTION WITH RANDOM FOREST")
print("="*60)
print(f"Test Accuracy: {rf_credit.score(X_test, y_test):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_credit.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nFeature Importance:")
print(importance_df.to_string(index=False))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'RF (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Credit Default Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## Summary

Random Forests combine multiple decision trees using:
- Bagging (bootstrap samples)
- Random feature selection

**Key Benefits:**
- Higher accuracy than single trees
- Robust feature importance
- Little tuning needed (use defaults!)
- State-of-the-art for tabular data

**Best Practices:**
- Start with n_estimators=100-200
- Use n_jobs=-1 for parallel training
- Use OOB score for quick validation
- Feature engineering still matters!

Next: Gradient Boosting for even better performance!
`,
  codeExample: `# Complete Random Forest Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

# Train with best practices
rf = RandomForestClassifier(
    n_estimators=200,        # Enough trees
    max_depth=15,            # Prevent overfitting
    min_samples_split=10,    # Robust splits
    min_samples_leaf=4,      # Smooth predictions
    max_features='sqrt',     # Decorrelate trees
    class_weight='balanced', # Handle imbalance
    oob_score=True,         # Free validation
    n_jobs=-1,              # Parallel
    random_state=42
)

rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"Test Score: {rf.score(X_test, y_test):.4f}")
`,
};
