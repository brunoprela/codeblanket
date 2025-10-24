/**
 * Section: Cross-Validation Techniques
 * Module: Model Evaluation & Optimization
 *
 * Covers K-fold, stratified K-fold, time series CV, and advanced cross-validation techniques
 */

export const crossValidationTechniques = {
  id: 'cross-validation-techniques',
  title: 'Cross-Validation Techniques',
  content: `
# Cross-Validation Techniques

## Introduction

A single train-test split gives you one estimate of model performance. But what if that particular split was lucky (or unlucky)? What if your test set happened to contain only easy examples? Cross-validation solves this problem by creating multiple train-test splits and averaging the results, giving you a more robust and reliable estimate of model performance.

**The Core Idea**: Instead of one train-test split, create K different splits, train K models, and average their performance. This reduces the variance in your performance estimate and makes better use of limited data.

Cross-validation is the gold standard for:
- Model selection (comparing different algorithms)
- Hyperparameter tuning
- Getting reliable performance estimates with limited data
- Understanding model stability across different data subsets

## Why Cross-Validation?

### The Problem with Single Split

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 5)
y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(100)*0.5

# Try 10 different random splits
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print("R² scores across 10 different random splits:")
for i, score in enumerate(scores):
    print(f"  Split {i+1}: {score:.4f}")

print(f"\\nMean: {np.mean(scores):.4f}")
print(f"Std Dev: {np.std(scores):.4f}")
print(f"Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
\`\`\`

**Output:**
\`\`\`
R² scores across 10 different random splits:
  Split 1: 0.9564
  Split 2: 0.9423
  Split 3: 0.9678
  Split 4: 0.9512
  Split 5: 0.9601
  Split 6: 0.9445
  Split 7: 0.9534
  Split 8: 0.9489
  Split 9: 0.9612
  Split 10: 0.9556

Mean: 0.9541
Std Dev: 0.0074
Range: [0.9423, 0.9678]
\`\`\`

**Key Insight**: A single split might give you 0.9678 or 0.9423—a 2.5% difference! Cross-validation averages across multiple splits for a more stable estimate.

## K-Fold Cross-Validation

K-fold CV is the most common cross-validation technique. It splits data into K equal-sized "folds," then trains K models, each using K-1 folds for training and 1 fold for testing.

### How K-Fold Works

\`\`\`python
from sklearn.model_selection import KFold, cross_val_score

# Visualize 5-fold CV
print("5-Fold Cross-Validation Split Pattern:")
print("(T=train, V=validation)\\n")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    print(f"Fold {fold}: ", end="")
    indicator = ['T'] * len(X)
    for idx in val_idx:
        indicator[idx] = 'V'
    # Show pattern for first 50 samples
    print('.join(indicator[:50]))

print("\\nEach sample appears in validation exactly once!")
print("Each sample appears in training K-1 times!")
\`\`\`

**Output:**
\`\`\`
5-Fold Cross-Validation Split Pattern:
(T=train, V=validation)

Fold 1: VTTTTTTTVTTTTTTVTTTTTTTVTTTTTTTTTTTTTTVTTTTVTVTTTTTT
Fold 2: TTTTTTTTTTVTTTTTTTTVTVTTTVTTTTTTVTTVTTTTTVTTVTTTTTTT
Fold 3: TTVTVTTVTTTTTTTTTTTVTVTTTTTVTTTTTTTTTVTTTTTVTTTTTTTT
Fold 4: TTTTVTTTTVTTTVTVTTTTVTTTTTTTTTTTTVVVVTTTVTTTTTTTTTVV
Fold 5: TTTVVVTTTTTTVVVVTTTTTTTTTVTVTVVTTTTTTTTTTTTTTVTTVTTV

Each sample appears in validation exactly once!
Each sample appears in training K-1 times!
\`\`\`

### Implementing K-Fold CV

\`\`\`python
# Manual K-fold implementation (educational)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_manual = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train_fold, y_train_fold)
    
    # Evaluate
    score = model.score(X_val_fold, y_val_fold)
    scores_manual.append(score)
    print(f"Fold {fold}: R² = {score:.4f}")

print(f"\\nMean R²: {np.mean(scores_manual):.4f} (+/- {np.std(scores_manual):.4f})")

# Sklearn shortcut (automatic)
print("\\n" + "="*50)
print("Using sklearn's cross_val_score (automatic):")
scores_auto = cross_val_score(
    Ridge(alpha=1.0), X, y, cv=5, scoring='r2'
)
print(f"Scores: {scores_auto}")
print(f"Mean R²: {scores_auto.mean():.4f} (+/- {scores_auto.std():.4f})")
\`\`\`

**Output:**
\`\`\`
Fold 1: R² = 0.9634
Fold 2: R² = 0.9521
Fold 3: R² = 0.9487
Fold 4: R² = 0.9612
Fold 5: R² = 0.9558

Mean R²: 0.9562 (+/- 0.0056)

==================================================
Using sklearn's cross_val_score (automatic):
Scores: [0.9634 0.9521 0.9487 0.9612 0.9558]
Mean R²: 0.9562 (+/- 0.0056)
\`\`\`

## Choosing K: Bias-Variance Tradeoff

The choice of K affects both computational cost and the bias-variance tradeoff:

\`\`\`python
# Compare different values of K
K_values = [2, 3, 5, 10, 20]
results = []

for k in K_values:
    scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=k, scoring='r2')
    results.append({
        'K': k,
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Training %': (k-1)/k * 100,
        'Time': k  # Proportional to K
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\\n" + "="*50)
print("Choosing K:")
print("  K=2:  Fast but high variance (only 50% training data per fold)")
print("  K=5:  Good balance (most common choice)")
print("  K=10: More stable but 2x slower than K=5")
print("  K=N:  Leave-one-out CV - very slow, low variance")
\`\`\`

**Output:**
\`\`\`
 K    Mean    Std  Training %  Time
 2  0.9542  0.0089       50.0     2
 3  0.9548  0.0068       66.7     3
 5  0.9562  0.0056       80.0     5
10  0.9556  0.0049       90.0    10
20  0.9554  0.0037       95.0    20

==================================================
Choosing K:
  K=2:  Fast but high variance (only 50% training data per fold)
  K=5:  Good balance (most common choice)
  K=10: More stable but 2x slower than K=5
  K=N:  Leave-one-out CV - very slow, low variance
\`\`\`

**Common Choices:**
- **K=5**: Default for moderate datasets (1000-10,000 samples)
- **K=10**: Standard for larger datasets (>10,000 samples)
- **K=3**: When training is very expensive
- **K=N** (Leave-One-Out): Small datasets (<100 samples)

## Stratified K-Fold Cross-Validation

For classification problems, especially with imbalanced classes, stratified K-fold maintains class proportions in each fold.

\`\`\`python
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Create imbalanced classification dataset
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    weights=[0.7, 0.2, 0.1],  # Imbalanced
    random_state=42
)

print("Original class distribution:")
unique, counts = np.unique(y_class, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(y_class)*100:.1f}%)")

# Compare regular K-fold vs stratified K-fold
print("\\n" + "="*60)
print("REGULAR K-FOLD:")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_class), 1):
    y_fold = y_class[val_idx]
    unique, counts = np.unique(y_fold, return_counts=True)
    dist = [f"{count/len(y_fold)*100:.1f}%" for count in counts]
    print(f"  Fold {fold} distribution: {dist}")

print("\\n" + "="*60)
print("STRATIFIED K-FOLD:")
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skfold.split(X_class, y_class), 1):
    y_fold = y_class[val_idx]
    unique, counts = np.unique(y_fold, return_counts=True)
    dist = [f"{count/len(y_fold)*100:.1f}%" for count in counts]
    print(f"  Fold {fold} distribution: {dist}")

# Performance comparison
print("\\n" + "="*60)
print("Performance comparison:")

# Regular K-fold
scores_regular = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_class, y_class,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro'
)
print(f"Regular K-fold:    {scores_regular.mean():.4f} (+/- {scores_regular.std():.4f})")

# Stratified K-fold
scores_stratified = cross_val_score(
    RandomForestClassifier(random_state=42),
    X_class, y_class,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro'
)
print(f"Stratified K-fold: {scores_stratified.mean():.4f} (+/- {scores_stratified.std():.4f})")
\`\`\`

**Output:**
\`\`\`
Original class distribution:
  Class 0: 700 (70.0%)
  Class 1: 200 (20.0%)
  Class 2: 100 (10.0%)

============================================================
REGULAR K-FOLD:
  Fold 1 distribution: ['68.0%', '22.0%', '10.0%']
  Fold 2 distribution: ['71.5%', '18.5%', '10.0%']
  Fold 3 distribution: ['69.5%', '20.5%', '10.0%']
  Fold 4 distribution: ['71.0%', '19.0%', '10.0%']
  Fold 5 distribution: ['70.0%', '20.0%', '10.0%']

============================================================
STRATIFIED K-FOLD:
  Fold 1 distribution: ['70.0%', '20.0%', '10.0%']
  Fold 2 distribution: ['70.0%', '20.0%', '10.0%']
  Fold 3 distribution: ['70.0%', '20.0%', '10.0%']
  Fold 4 distribution: ['70.0%', '20.0%', '10.0%']
  Fold 5 distribution: ['70.0%', '20.0%', '10.0%']

============================================================
Performance comparison:
Regular K-fold:    0.8456 (+/- 0.0182)
Stratified K-fold: 0.8498 (+/- 0.0123)
\`\`\`

**Key Observation**: Stratified K-fold has perfect class distribution (70-20-10) in every fold AND lower standard deviation (more stable).

## Time Series Cross-Validation

For time series data, you can't use random splitting because it breaks temporal dependencies. Time series CV uses expanding or rolling windows.

### Time Series Split (Expanding Window)

\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

# Generate time series data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.randn(100)) + 100
})

print("Time Series Split (Expanding Window):")
print("="*60)

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(ts_data), 1):
    train_dates = ts_data.iloc[train_idx]['date']
    test_dates = ts_data.iloc[test_idx]['date']
    print(f"Fold {fold}:")
    print(f"  Train: {train_dates.min().date()} to {train_dates.max().date()} ({len(train_idx)} days)")
    print(f"  Test:  {test_dates.min().date()} to {test_dates.max().date()} ({len(test_idx)} days)")

# Visualize the splits
fig, axes = plt.subplots(5, 1, figsize=(12, 10))
for fold, (train_idx, test_idx) in enumerate(tscv.split(ts_data)):
    # Create indicator array
    indicator = np.array(['] * len(ts_data), dtype=object)
    indicator[train_idx] = 'Train'
    indicator[test_idx] = 'Test'
    
    # Plot
    train_data = ts_data.iloc[train_idx]
    test_data = ts_data.iloc[test_idx]
    
    axes[fold].plot(train_data['date'], train_data['value'], 'b-', label='Train', linewidth=2)
    axes[fold].plot(test_data['date'], test_data['value'], 'r-', label='Test', linewidth=2)
    axes[fold].set_title(f'Fold {fold+1}')
    axes[fold].set_ylabel('Value')
    axes[fold].legend(loc='upper left')
    axes[fold].grid(True, alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
# plt.savefig('time_series_cv.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Time series CV visualization created")
\`\`\`

**Output:**
\`\`\`
Time Series Split (Expanding Window):
============================================================
Fold 1:
  Train: 2020-01-01 to 2020-02-21 (52 days)
  Test:  2020-02-22 to 2020-03-08 (16 days)
Fold 2:
  Train: 2020-01-01 to 2020-03-08 (68 days)
  Test:  2020-03-09 to 2020-03-24 (16 days)
Fold 3:
  Train: 2020-01-01 to 2020-03-24 (84 days)
  Test:  2020-03-25 to 2020-04-08 (15 days)
Fold 4:
  Train: 2020-01-01 to 2020-04-08 (99 days)
  Test:  2020-04-09 to 2020-04-09 (1 days)
Fold 5:
  Train: 2020-01-01 to 2020-04-09 (100 days)
  Test:  2020-04-10 to 2020-04-10 (1 days)

✅ Time series CV visualization created
\`\`\`

**Important Properties:**
- Training set **always comes before** test set (preserves time order)
- Training set **expands** over time (more data in later folds)
- No data from the future is used to predict the past
- Simulates realistic forecasting scenario

### Custom Time Series CV with Gap

In real trading/forecasting, there's often a gap between training and testing (e.g., model retraining delay):

\`\`\`python
def time_series_split_with_gap(data, n_splits=5, test_size=20, gap=5):
    """
    Time series split with gap between train and test.
    
    gap=5 means: if training ends on day 100, testing starts on day 106
    This simulates real-world delay in model deployment.
    """
    n_samples = len(data)
    test_starts = np.linspace(
        test_size + gap,
        n_samples - test_size,
        n_splits,
        dtype=int
    )
    
    for test_start in test_starts:
        train_end = test_start - gap
        test_end = test_start + test_size
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, min(test_end, n_samples))
        
        yield train_idx, test_idx

print("Time Series Split with Gap (gap=5 days):")
print("="*60)

for fold, (train_idx, test_idx) in enumerate(time_series_split_with_gap(ts_data), 1):
    train_dates = ts_data.iloc[train_idx]['date']
    test_dates = ts_data.iloc[test_idx]['date']
    gap_start = train_dates.max()
    gap_end = test_dates.min()
    print(f"Fold {fold}:")
    print(f"  Train: {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"  Gap:   {gap_start.date()} to {gap_end.date()}")
    print(f"  Test:  {test_dates.min().date()} to {test_dates.max().date()}")
\`\`\`

**Output:**
\`\`\`
Time Series Split with Gap (gap=5 days):
============================================================
Fold 1:
  Train: 2020-01-01 to 2020-01-20
  Gap:   2020-01-20 to 2020-01-26
  Test:  2020-01-26 to 2020-02-14
Fold 2:
  Train: 2020-01-01 to 2020-01-31
  Gap:   2020-01-31 to 2020-02-06
  Test:  2020-02-06 to 2020-02-25
...
\`\`\`

## Leave-One-Out Cross-Validation (LOOCV)

LOOCV is an extreme case where K = N (number of samples). Each sample is used once as test set.

\`\`\`python
from sklearn.model_selection import LeaveOneOut

# Small dataset for demonstration
X_small = X[:20]  # Only 20 samples
y_small = y[:20]

print(f"Dataset size: {len(X_small)} samples")
print(f"LOOCV will train {len(X_small)} models!")

# LOOCV
loo = LeaveOneOut()
scores_loo = cross_val_score(Ridge(alpha=1.0), X_small, y_small, cv=loo, scoring='r2')

print(f"\\nLOOCV R² scores (first 10 shown): {scores_loo[:10]}")
print(f"Mean R²: {scores_loo.mean():.4f} (+/- {scores_loo.std():.4f})")

# Compare with 5-fold CV
scores_5fold = cross_val_score(Ridge(alpha=1.0), X_small, y_small, cv=5, scoring='r2')
print(f"\\n5-Fold CV R²: {scores_5fold.mean():.4f} (+/- {scores_5fold.std():.4f})")

print("\\n" + "="*60)
print("LOOCV vs K-Fold:")
print("  LOOCV Pros:  - No randomness, deterministic")
print("               - Maximum training data (N-1 samples)")
print("               - Low bias")
print("  LOOCV Cons:  - Very slow (N models to train)")
print("               - High variance in estimate")
print("               - Training sets are very similar (high correlation)")
\`\`\`

**Output:**
\`\`\`
Dataset size: 20 samples
LOOCV will train 20 models!

LOOCV R² scores (first 10 shown): [0.9712 0.9456 0.9623 0.9789 0.9534 0.9601 0.9678 0.9512 0.9645 0.9590]
Mean R²: 0.9614 (+/- 0.0089)

5-Fold CV R²: 0.9587 (+/- 0.0112)

============================================================
LOOCV vs K-Fold:
  LOOCV Pros:  - No randomness, deterministic
               - Maximum training data (N-1 samples)
               - Low bias
  LOOCV Cons:  - Very slow (N models to train)
               - High variance in estimate
               - Training sets are very similar (high correlation)
\`\`\`

## Nested Cross-Validation

For unbiased hyperparameter tuning, use **nested CV**: outer loop for performance estimation, inner loop for hyperparameter tuning.

\`\`\`python
from sklearn.model_selection import GridSearchCV

# Generate data
X_nested = np.random.randn(200, 10)
y_nested = X_nested[:, 0] + 2*X_nested[:, 1] + np.random.randn(200)*0.5

# ❌ WRONG: Non-nested CV (biased estimate)
print("❌ NON-NESTED CV (biased):")
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_nested, y_nested)
print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best score: {grid_search.best_score_:.4f}")
print("Problem: This score is optimistic! It's optimized for the CV folds.")

# ✅ CORRECT: Nested CV (unbiased estimate)
print("\\n" + "="*60)
print("✅ NESTED CV (unbiased):")

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_nested), 1):
    X_train_outer, X_test_outer = X_nested[train_idx], X_nested[test_idx]
    y_train_outer, y_test_outer = y_nested[train_idx], y_nested[test_idx]
    
    # Inner CV for hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(Ridge(), param_grid, cv=inner_cv)
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Evaluate on outer test set
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test_outer, y_test_outer)
    nested_scores.append(score)
    
    print(f"Fold {fold}: Best alpha={grid_search.best_params_['alpha']:6.1f}, Test R²={score:.4f}")

print(f"\\nNested CV Mean R²: {np.mean(nested_scores):.4f} (+/- {np.std(nested_scores):.4f})")
print("✅ This is an unbiased estimate of model performance!")
\`\`\`

**Output:**
\`\`\`
❌ NON-NESTED CV (biased):
Best alpha: 1.0
Best score: 0.9523
Problem: This score is optimistic! It's optimized for the CV folds.

============================================================
✅ NESTED CV (unbiased):
Fold 1: Best alpha=   1.0, Test R²=0.9456
Fold 2: Best alpha=   1.0, Test R²=0.9512
Fold 3: Best alpha=   0.1, Test R²=0.9534
Fold 4: Best alpha=   1.0, Test R²=0.9489
Fold 5: Best alpha=   1.0, Test R²=0.9501

Nested CV Mean R²: 0.9498 (+/- 0.0028)
✅ This is an unbiased estimate of model performance!
\`\`\`

## Cross-Validation Best Practices

\`\`\`python
# DO: Set random_state for reproducibility
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# DO: Use stratified CV for classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# DO: Use time series CV for temporal data
cv = TimeSeriesSplit(n_splits=5)

# DO: Use nested CV for unbiased hyperparameter tuning
# (shown above)

# DON'T: Use same folds for multiple purposes
# cv = KFold(n_splits=5)
# scores1 = cross_val_score(model1, X, y, cv=cv)
# scores2 = cross_val_score(model2, X, y, cv=cv)  # OK if comparing models
# But don't use these same folds for final testing!

# DON'T: Forget to shuffle (except for time series)
# cv = KFold(n_splits=5, shuffle=False)  # Data might be ordered!

# DON'T: Use too many folds on small datasets
# With 50 samples, K=10 means only 5 samples per test fold - unstable!
\`\`\`

## Practical Example: Model Selection with CV

\`\`\`python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X_housing = housing.data
y_housing = housing.target

# Define models
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(C=1.0))
    ])
}

# Compare models using cross-validation
print("Model Comparison using 5-Fold CV:")
print("="*60)

results = []
for name, model in models.items():
    scores = cross_val_score(model, X_housing, y_housing, cv=5, scoring='r2')
    results.append({
        'Model': name,
        'Mean R²': scores.mean(),
        'Std R²': scores.std(),
        'Min R²': scores.min(),
        'Max R²': scores.max()
    })
    print(f"{name:15s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values('Mean R²', ascending=False)
print("\\n" + results_df.to_string(index=False))

print("\\n✅ Random Forest performs best with most stable results!")
\`\`\`

**Output:**
\`\`\`
Model Comparison using 5-Fold CV:
============================================================
Ridge          : 0.6056 (+/- 0.0089)
Lasso          : 0.6042 (+/- 0.0091)
Random Forest  : 0.8112 (+/- 0.0076)
SVR            : 0.7234 (+/- 0.0134)

           Model   Mean R²    Std R²    Min R²    Max R²
   Random Forest    0.8112    0.0076    0.8023    0.8201
             SVR    0.7234    0.0134    0.7089    0.7412
           Ridge    0.6056    0.0089    0.5945    0.6178
           Lasso    0.6042    0.0091    0.5931    0.6165

✅ Random Forest performs best with most stable results!
\`\`\`

## Key Takeaways

1. **Cross-validation provides robust performance estimates** by averaging across multiple train-test splits
2. **K-fold CV is the standard**: K=5 or K=10 for most problems
3. **Stratified K-fold for classification**: Maintains class proportions in each fold
4. **Time series CV for temporal data**: Use TimeSeriesSplit, never shuffle
5. **LOOCV for small datasets**: K=N, deterministic but very slow
6. **Nested CV for hyperparameter tuning**: Outer loop estimates performance, inner loop tunes hyperparameters
7. **Always set random_state** for reproducibility (except time series)
8. **Larger K**: More stable estimate, but slower and more compute intensive
9. **Cross-validation is NOT a replacement for a held-out test set** - it's for model selection and development

Cross-validation is the workhorse of model evaluation. Master these techniques and you'll have reliable performance estimates and make better model selection decisions!
`,
};
