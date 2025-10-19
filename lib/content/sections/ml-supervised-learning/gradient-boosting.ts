/**
 * Gradient Boosting Machines Section
 */

export const gradientboostingSection = {
  id: 'gradient-boosting',
  title: 'Gradient Boosting Machines',
  content: `# Gradient Boosting Machines

## Introduction

Gradient Boosting is a powerful ensemble technique that builds models sequentially, with each new model correcting errors of previous models. Unlike Random Forests (which build trees independently), Gradient Boosting creates trees that specifically target residual errors.

**Key Idea**: Fit new models to residuals (errors) of ensemble so far.

\\[ F_m(x) = F_{m-1}(x) + \\nu \\cdot h_m(x) \\]

Where:
- \\( F_m(x) \\): Ensemble after m iterations
- \\( h_m(x) \\): New weak learner
- \\( \\nu \\): Learning rate (shrinkage)

**Popular Implementations**:
- scikit-learn: GradientBoostingClassifier/Regressor
- **XGBoost**: Extreme Gradient Boosting (most popular)
- **LightGBM**: Light Gradient Boosting Machine (Microsoft)
- **CatBoost**: Categorical Boosting (Yandex)

**Applications**:
- Kaggle competition winner
- Click-through rate prediction
- Fraud detection
- Customer lifetime value
- Ranking systems

## Boosting vs Bagging

**Random Forest (Bagging)**:
- Trees built independently in parallel
- Each tree sees different bootstrap sample
- Reduces variance through averaging
- Trees can be deep (low bias)

**Gradient Boosting (Boosting)**:
- Trees built sequentially
- Each tree fits residuals of previous trees
- Reduces bias by focusing on mistakes
- Trees are shallow (weak learners)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Generate data with complex pattern
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(200) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare models
models = {
    'Single Tree': DecisionTreeRegressor(max_depth=3),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

X_plot = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_plot)
    
    ax.scatter(X_train, y_train, s=20, edgecolor="black", c="blue", alpha=0.5, label="Train")
    ax.scatter(X_test, y_test, s=20, edgecolor="black", c="red", alpha=0.5, label="Test")
    ax.plot(X_plot, y_pred, color="green", linewidth=2, label="Prediction")
    ax.plot(X_plot, np.sin(X_plot), color="black", linestyle="--", label="True function")
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    ax.set_title(f'{name}\\nTrain R²: {train_score:.3f}, Test R²: {test_score:.3f}')
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Gradient Boosting typically achieves best test performance!")
\`\`\`

## How Gradient Boosting Works

**Algorithm**:
1. Start with initial prediction (mean for regression, log-odds for classification)
2. For m = 1 to M iterations:
   a. Compute residuals: \\( r_i = y_i - F_{m-1}(x_i) \\)
   b. Fit weak learner \\( h_m \\) to residuals
   c. Update: \\( F_m(x) = F_{m-1}(x) + \\nu \\cdot h_m(x) \\)
3. Final prediction: \\( F_M(x) \\)

\`\`\`python
# Manual implementation to illustrate concept
from sklearn.tree import DecisionTreeRegressor

# Simple 1D example
X_simple = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_simple = np.array([2, 4, 3, 8, 7, 10, 9, 14, 13, 16])

# Initialize with mean
F = np.full(len(y_simple), y_simple.mean())
learning_rate = 0.5
n_iterations = 5

# Store predictions for visualization
predictions_over_time = [F.copy()]

print("Gradient Boosting Iterations:")
print(f"Initial prediction (mean): {F[0]:.2f}")
print()

for iteration in range(n_iterations):
    # Compute residuals
    residuals = y_simple - F
    
    # Fit tree to residuals
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X_simple, residuals)
    
    # Update predictions
    update = tree.predict(X_simple)
    F = F + learning_rate * update
    
    predictions_over_time.append(F.copy())
    
    mse = np.mean((y_simple - F)**2)
    print(f"Iteration {iteration+1}: MSE = {mse:.4f}")

# Visualize progression
plt.figure(figsize=(15, 10))

for i, preds in enumerate(predictions_over_time):
    plt.subplot(2, 3, i+1)
    plt.scatter(X_simple, y_simple, s=100, color='blue', label='Actual', zorder=3)
    plt.scatter(X_simple, preds, s=100, color='red', marker='x', s=100, label='Predicted', zorder=3)
    plt.plot(X_simple, preds, 'r--', alpha=0.5)
    
    if i == 0:
        plt.title(f'Initial (Mean): MSE={np.mean((y_simple-preds)**2):.2f}')
    else:
        plt.title(f'After {i} iterations: MSE={np.mean((y_simple-preds)**2):.2f}')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nEach iteration reduces error by targeting residuals!")
\`\`\`

## Learning Rate (Shrinkage)

Learning rate \\( \\nu \\) controls how much each tree contributes:

\`\`\`python
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Compare different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
n_estimators = 200

plt.figure(figsize=(12, 8))

for lr in learning_rates:
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    # Track test score over iterations
    test_scores = []
    for i in range(1, n_estimators + 1):
        y_pred = gb.predict(X_test[:, :])  # Predict with first i trees
        # Note: scikit-learn doesn't easily support staged predictions for single estimator
        # Using staged_predict for simplicity
    
    # Use staged_predict_proba
    test_scores = [gb.score(X_test, y_test) for _ in gb.staged_predict(X_test)]
    
    plt.plot(range(1, len(test_scores)+1), test_scores, label=f'lr={lr}', linewidth=2)

plt.xlabel('Number of Trees')
plt.ylabel('Test Accuracy')
plt.title('Learning Rate Impact on Gradient Boosting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Small learning rate needs more trees but generalizes better!")
\`\`\`

## XGBoost: Extreme Gradient Boosting

XGBoost is the most popular GB implementation with many optimizations:

**Improvements over standard GB**:
- Regularization (L1/L2 on leaf weights)
- Parallel tree construction
- Tree pruning (max_depth then prune back)
- Built-in cross-validation
- Handling missing values
- Early stopping

\`\`\`python
# First install: pip install xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Prepare data in XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    'max_depth': 3,
    'eta': 0.1,  # learning rate
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

# Train with evaluation
evals = [(dtrain, 'train'), (dtest, 'test')]
num_rounds = 200

xgb_model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=evals,
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose_eval=20  # Print every 20 rounds
)

# Predictions
y_pred_proba = xgb_model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\\n" + "="*60)
print("XGBOOST RESULTS")
print("="*60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Feature importance
xgb.plot_importance(xgb_model, max_num_features=15, height=0.8, title="XGBoost Feature Importance")
plt.tight_layout()
plt.show()
\`\`\`

## Scikit-learn API for XGBoost

\`\`\`python
from xgboost import XGBClassifier

# Sklearn-compatible interface
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,  # Fraction of samples for each tree
    colsample_bytree=0.8,  # Fraction of features for each tree
    gamma=0,  # Minimum loss reduction for split
    reg_alpha=0,  # L1 regularization
    reg_lambda=1,  # L2 regularization
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

print(f"Best iteration: {xgb_clf.best_iteration}")
print(f"Best score: {xgb_clf.best_score:.4f}")
print(f"Test accuracy: {xgb_clf.score(X_test, y_test):.4f}")
\`\`\`

## LightGBM: Microsoft's Fast GB

LightGBM is faster and more memory-efficient than XGBoost for large datasets.

**Key Innovations**:
- Leaf-wise tree growth (vs level-wise)
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)

\`\`\`python
# Install: pip install lightgbm
import lightgbm as lgb

# LightGBM Dataset
ltrain = lgb.Dataset(X_train, label=y_train)
ltest = lgb.Dataset(X_test, label=y_test, reference=ltrain)

# Parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,  # -1 means no limit
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

# Train
lgb_model = lgb.train(
    lgb_params,
    ltrain,
    num_boost_round=200,
    valid_sets=[ltest],
    early_stopping_rounds=10
)

# Predictions
y_pred_lgb = lgb_model.predict(X_test)
y_pred_lgb_binary = (y_pred_lgb > 0.5).astype(int)

print(f"LightGBM Test Accuracy: {accuracy_score(y_test, y_pred_lgb_binary):.4f}")

# Scikit-learn API
from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

lgbm_clf.fit(X_train, y_train)
print(f"LightGBM (sklearn API): {lgbm_clf.score(X_test, y_test):.4f}")
\`\`\`

## CatBoost: Handling Categorical Features

CatBoost excels with categorical features (no need for one-hot encoding).

\`\`\`python
# Install: pip install catboost
from catboost import CatBoostClassifier

# Create dataset with categorical features
import pandas as pd

# Add categorical features for demonstration
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# CatBoost
cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=3,
    loss_function='Logloss',
    random_seed=42,
    verbose=0
)

cat_model.fit(X_train_df, y_train, eval_set=(X_test_df, y_test), early_stopping_rounds=10)

print(f"CatBoost Test Accuracy: {cat_model.score(X_test_df, y_test):.4f}")
\`\`\`

## Hyperparameter Tuning

\`\`\`python
from sklearn.model_selection import GridSearchCV

# XGBoost hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Grid search
xgb_search = GridSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)

print("Best parameters:", xgb_search.best_params_)
print(f"Best CV score: {xgb_search.best_score_:.4f}")
print(f"Test score: {xgb_search.score(X_test, y_test):.4f}")
\`\`\`

## Gradient Boosting for Regression

\`\`\`python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# Load data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Compare GB implementations
models = {
    'scikit-learn GB': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
}

import pandas as pd
results = []

for name, model in models.items():
    import time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Model': name,
        'R²': r2,
        'RMSE': rmse,
        'Train Time (s)': train_time
    })

df_results = pd.DataFrame(results)
print("\\nRegression Comparison:")
print(df_results.to_string(index=False))
\`\`\`

## Overfitting Prevention

**Strategies**:
1. **Learning rate**: Lower learning rate (requires more trees)
2. **Max depth**: Shallow trees (3-5 typical)
3. **Subsampling**: Use fraction of data per tree
4. **Feature sampling**: Use fraction of features per tree
5. **Early stopping**: Stop when validation error stops improving
6. **Regularization**: L1/L2 on leaf weights (XGBoost)

\`\`\`python
# Demonstrate overfitting
gb_overfit = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.5,  # High learning rate
    max_depth=10,  # Deep trees
    random_state=42
)

gb_optimal = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,  # Lower learning rate
    max_depth=3,  # Shallow trees
    subsample=0.8,  # Subsampling
    max_features='sqrt',  # Feature sampling
    random_state=42
)

gb_overfit.fit(X_train, y_train)
gb_optimal.fit(X_train, y_train)

print("Overfitting Example:")
print(f"High LR + Deep Trees - Train: {gb_overfit.score(X_train, y_train):.4f}, Test: {gb_overfit.score(X_test, y_test):.4f}")
print(f"Optimal Settings - Train: {gb_optimal.score(X_train, y_train):.4f}, Test: {gb_optimal.score(X_test, y_test):.4f}")
\`\`\`

## Summary

Gradient Boosting builds models sequentially, with each tree correcting previous errors:

**Key Concepts**:
- Sequential ensemble (not parallel like RF)
- Fits trees to residuals
- Learning rate controls contribution
- Shallow trees (weak learners)

**Implementations**:
- **XGBoost**: Most popular, regularization, parallel
- **LightGBM**: Fastest, best for large data
- **CatBoost**: Best for categorical features

**Best Practices**:
- Start with learning_rate=0.1, max_depth=3
- Use early stopping with validation set
- XGBoost for general use
- LightGBM for speed
- More trees + lower learning rate = better generalization

Next: Ensemble methods that combine different model types!
`,
  codeExample: `# Complete XGBoost Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Best practice configuration
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,              # Shallow trees
    learning_rate=0.1,        # Moderate LR
    subsample=0.8,            # Row sampling
    colsample_bytree=0.8,     # Column sampling
    gamma=0.1,                # Regularization
    reg_lambda=1,             # L2 regularization
    early_stopping_rounds=10, # Stop early
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"Test accuracy: {xgb_model.score(X_test, y_test):.4f}")
`,
};
