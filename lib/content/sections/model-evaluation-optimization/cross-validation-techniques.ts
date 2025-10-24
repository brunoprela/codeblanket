export const crossValidationTechniques = {
  title: 'Cross-Validation Techniques',
  content: `
# Cross-Validation Techniques

## Introduction

Cross-validation is one of the most powerful techniques in machine learning for assessing model performance and reducing variance in performance estimates. While a single train-test split provides one estimate of model performance, cross-validation provides multiple estimates, giving us a more robust and reliable assessment.

**The Core Idea**: Instead of splitting data once, split it multiple times in different ways, train and evaluate the model on each split, and aggregate the results.

**Why Cross-Validation?**

1. **More reliable estimates**: Single split can be lucky or unlucky; averaging multiple splits reduces variance
2. **Better use of data**: Every sample gets to be in both training and test sets
3. **Model stability assessment**: Variance across folds indicates model robustness
4. **Small dataset utility**: Particularly valuable when data is limited

## K-Fold Cross-Validation

K-fold CV is the most common cross-validation technique. The data is divided into K equal-sized folds, and the model is trained K times, each time using K-1 folds for training and 1 fold for testing.

### Mathematical Foundation

For K folds, the cross-validation score is:

$$\\text{CV Score} = \\frac{1}{K} \\sum_{i=1}^{K} \\text{Score}_i$$

where Score_i is the performance metric on fold i.

The standard error provides uncertainty:

$$\\text{SE} = \\frac{\\sigma}{\\sqrt{K}}$$

where σ is the standard deviation of scores across folds.

### Implementation from Scratch

\`\`\`python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class KFoldCV:
    """K-Fold Cross-Validation from scratch."""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """Generate train/test indices for K-fold CV."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
    
    def cross_val_score(self, model, X, y, scoring='r2'):
        """Perform cross-validation and return scores."""
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X, y), 1):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone and train model
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            # Evaluate
            if scoring == 'r2':
                score = model_clone.score(X_test, y_test)
            elif scoring == 'neg_mse':
                y_pred = model_clone.predict(X_test)
                score = -mean_squared_error(y_test, y_pred)
            elif scoring == 'neg_rmse':
                y_pred = model_clone.predict(X_test)
                score = -np.sqrt(mean_squared_error(y_test, y_pred))
            
            scores.append(score)
            print(f"Fold {fold}: {score:.4f}")
        
        return np.array(scores)

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print("K-Fold Cross-Validation Demo")
print("=" * 70)

# Custom implementation
cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)
model = Ridge(alpha=1.0)

print("\\nCustom K-Fold CV:")
scores = cv.cross_val_score(model, X, y, scoring='r2')
print(f"\\nMean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"95% CI: [{scores.mean() - 1.96*scores.std():.4f}, "
      f"{scores.mean() + 1.96*scores.std():.4f}]")
\`\`\`

### Using Scikit-learn's KFold

\`\`\`python
from sklearn.model_selection import KFold, cross_val_score, cross_validate

# Simple cross-validation score
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

print("\\nScikit-learn K-Fold CV:")
print(f"Scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")

# More detailed cross-validation with multiple metrics
scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                            return_train_score=True)

print("\\nDetailed Cross-Validation Results:")
print("-" * 70)
for metric in scoring:
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']
    print(f"{metric}:")
    print(f"  Test:  {test_scores.mean():7.4f} (+/- {test_scores.std():.4f})")
    print(f"  Train: {train_scores.mean():7.4f} (+/- {train_scores.std():.4f})")
\`\`\`

### Visualizing K-Fold Splits

\`\`\`python
def visualize_cv_splits(cv, X, y, n_samples_to_plot=100):
    """Visualize how cross-validation splits the data."""
    fig, axes = plt.subplots(cv.get_n_splits(), 1, figsize=(12, 8))
    
    # Use subset for visualization
    indices = np.arange(min(n_samples_to_plot, len(X)))
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X[:n_samples_to_plot])):
        # Create array for visualization
        split_array = np.full(len(indices), 0)
        split_array[train_idx] = 1  # Training
        split_array[test_idx] = 2   # Testing
        
        axes[fold].barh(0, len(indices), color='white', edgecolor='black')
        
        # Color code: training (blue), testing (red)
        for idx, val in enumerate(split_array):
            color = 'steelblue' if val == 1 else 'coral'
            axes[fold].barh(0, 1, left=idx, color=color)
        
        axes[fold].set_xlim(0, len(indices))
        axes[fold].set_ylim(-0.5, 0.5)
        axes[fold].set_ylabel(f'Fold {fold+1}', rotation=0, labelpad=30)
        axes[fold].set_yticks([])
        axes[fold].set_xticks([])
    
    axes[0].set_title('K-Fold Cross-Validation Data Splits\\n(Blue=Train, Red=Test)', 
                      fontsize=14, pad=20)
    axes[-1].set_xlabel('Sample Index')
    plt.tight_layout()
    plt.savefig('kfold_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'kfold_visualization.png'")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
visualize_cv_splits(cv, X, y)
\`\`\`

## Stratified K-Fold Cross-Validation

For classification problems, especially with imbalanced classes, stratified K-fold ensures each fold maintains the same class distribution as the full dataset.

### Why Stratification Matters

\`\`\`python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# Load imbalanced dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print("Dataset class distribution:")
print(Counter(y))
print(f"Class 0: {(y==0).sum()/len(y)*100:.1f}%")
print(f"Class 1: {(y==1).sum()/len(y)*100:.1f}%")

# Compare regular K-Fold vs Stratified K-Fold
print("\\n" + "="*70)
print("Regular K-Fold (not stratified):")
print("="*70)

cv_regular = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(cv_regular.split(X, y), 1):
    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]
    
    train_dist = Counter(y_train_fold)
    test_dist = Counter(y_test_fold)
    
    print(f"Fold {fold}:")
    print(f"  Train - Class 0: {train_dist[0]/len(y_train_fold)*100:5.1f}%, "
          f"Class 1: {train_dist[1]/len(y_train_fold)*100:5.1f}%")
    print(f"  Test  - Class 0: {test_dist[0]/len(y_test_fold)*100:5.1f}%, "
          f"Class 1: {test_dist[1]/len(y_test_fold)*100:5.1f}%")

print("\\n" + "="*70)
print("Stratified K-Fold:")
print("="*70)

cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(cv_stratified.split(X, y), 1):
    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]
    
    train_dist = Counter(y_train_fold)
    test_dist = Counter(y_test_fold)
    
    print(f"Fold {fold}:")
    print(f"  Train - Class 0: {train_dist[0]/len(y_train_fold)*100:5.1f}%, "
          f"Class 1: {train_dist[1]/len(y_train_fold)*100:5.1f}%")
    print(f"  Test  - Class 0: {test_dist[0]/len(y_test_fold)*100:5.1f}%, "
          f"Class 1: {test_dist[1]/len(y_test_fold)*100:5.1f}%")

# Compare model performance
model = LogisticRegression(max_iter=10000, random_state=42)

scores_regular = cross_val_score(model, X, y, cv=cv_regular, scoring='f1')
scores_stratified = cross_val_score(model, X, y, cv=cv_stratified, scoring='f1')

print("\\n" + "="*70)
print("Model Performance Comparison:")
print("="*70)
print(f"Regular K-Fold:    F1 = {scores_regular.mean():.4f} (+/- {scores_regular.std():.4f})")
print(f"Stratified K-Fold: F1 = {scores_stratified.mean():.4f} (+/- {scores_stratified.std():.4f})")
print(f"\\nVariance reduction: {(scores_regular.std() - scores_stratified.std())/scores_regular.std()*100:.1f}%")
\`\`\`

## Leave-One-Out Cross-Validation (LOOCV)

LOOCV is an extreme case of K-fold where K = N (number of samples). Each sample is used once as a test set.

### When to Use LOOCV

**Advantages:**
- Maximum training data (N-1 samples)
- Deterministic (no randomness)
- Nearly unbiased estimate

**Disadvantages:**
- Computationally expensive (N model trainings)
- High variance in estimates
- Not suitable for large datasets

\`\`\`python
from sklearn.model_selection import LeaveOneOut
import time

# Small dataset example
from sklearn.datasets import load_iris
iris = load_iris()
X_small, y_small = iris.data, iris.target

print("Leave-One-Out Cross-Validation")
print("="*70)

# LOOCV
loo = LeaveOneOut()
n_splits = loo.get_n_splits(X_small)
print(f"Number of splits (folds): {n_splits}")
print(f"Each fold: {n_splits-1} train, 1 test sample")

model = LogisticRegression(max_iter=10000)

# Time comparison: LOOCV vs 5-Fold
start_time = time.time()
scores_loo = cross_val_score(model, X_small, y_small, cv=loo)
loo_time = time.time() - start_time

start_time = time.time()
scores_5fold = cross_val_score(model, X_small, y_small, cv=5)
fold5_time = time.time() - start_time

print(f"\\nResults:")
print(f"LOOCV:  Accuracy = {scores_loo.mean():.4f} (+/- {scores_loo.std():.4f}), Time = {loo_time:.3f}s")
print(f"5-Fold: Accuracy = {scores_5fold.mean():.4f} (+/- {scores_5fold.std():.4f}), Time = {fold5_time:.3f}s")
print(f"\\nLOOCV took {loo_time/fold5_time:.1f}x longer")

# Demonstrate computational cost scaling
print("\\n" + "="*70)
print("Computational Cost Analysis:")
print("="*70)

dataset_sizes = [50, 100, 200, 500]
for size in dataset_sizes:
    X_subset = X_small[:min(size, len(X_small))]
    y_subset = y_small[:min(size, len(X_small))]
    
    # Repeat to fill if needed
    if size > len(X_small):
        reps = size // len(X_small) + 1
        X_subset = np.tile(X_small, (reps, 1))[:size]
        y_subset = np.tile(y_small, reps)[:size]
    
    loo = LeaveOneOut()
    
    start = time.time()
    # Just split, don't train (to show split overhead)
    splits = list(loo.split(X_subset))
    elapsed = time.time() - start
    
    print(f"N={size:4d}: {len(splits):4d} folds, estimated training time ≈ {elapsed*100:.2f}s")

print("\\n⚠️  LOOCV not recommended for N > 1000")
\`\`\`

## Time Series Cross-Validation

For temporal data, standard cross-validation violates the temporal ordering. Time series CV ensures training data always precedes test data.

### TimeSeriesSplit

\`\`\`python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

# Generate time series data
np.random.seed(42)
n_samples = 200
dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

# Simulate stock price with trend
price = 100 + np.cumsum(np.random.randn(n_samples) * 2)
X_ts = price[:-1].reshape(-1, 1)
y_ts = price[1:]  # Predict next day

print("Time Series Cross-Validation")
print("="*70)

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

print(f"\\nTimeSeriesSplit with {tscv.n_splits} splits:")
print("-"*70)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts), 1):
    train_dates = dates[train_idx]
    test_dates = dates[test_idx]
    
    print(f"Fold {fold}:")
    print(f"  Train: {train_dates.min().date()} to {train_dates.max().date()} "
          f"({len(train_idx):3d} samples)")
    print(f"  Test:  {test_dates.min().date()} to {test_dates.max().date()} "
          f"({len(test_idx):3d} samples)")
    
    # Verify no overlap and correct order
    assert train_dates.max() < test_dates.min(), "Training must precede testing!"

# Visualize time series splits
def visualize_ts_splits(tscv, X):
    """Visualize time series CV splits."""
    fig, axes = plt.subplots(tscv.n_splits, 1, figsize=(12, 8))
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        split_array = np.full(len(X), 0)
        split_array[train_idx] = 1
        split_array[test_idx] = 2
        
        axes[fold].barh(0, len(X), color='white', edgecolor='black')
        
        for idx, val in enumerate(split_array):
            if val == 1:
                color = 'steelblue'
            elif val == 2:
                color = 'coral'
            else:
                color = 'lightgray'
            axes[fold].barh(0, 1, left=idx, color=color)
        
        axes[fold].set_xlim(0, len(X))
        axes[fold].set_ylim(-0.5, 0.5)
        axes[fold].set_ylabel(f'Fold {fold+1}', rotation=0, labelpad=30)
        axes[fold].set_yticks([])
    
    axes[0].set_title('Time Series Cross-Validation\\n(Blue=Train, Red=Test, Gray=Not Used)', 
                      fontsize=14, pad=20)
    axes[-1].set_xlabel('Time Index')
    plt.tight_layout()
    plt.savefig('timeseries_cv.png', dpi=150, bbox_inches='tight')
    print("\\nVisualization saved to 'timeseries_cv.png'")

visualize_ts_splits(tscv, X_ts)

# Evaluate model with time series CV
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
scores = cross_val_score(model, X_ts, y_ts, cv=tscv, scoring='neg_mean_squared_error')

print(f"\\nTime Series CV Results:")
print(f"Neg MSE: {scores.mean():.2f} (+/- {scores.std():.2f})")
print(f"RMSE: {np.sqrt(-scores.mean()):.2f}")
\`\`\`

### Advanced: Blocked Time Series CV

For financial data with intraday patterns or weekly seasonality:

\`\`\`python
def blocked_time_series_split(X, y, n_splits=5, test_size=20, gap=5):
    """
    Time series CV with gap to prevent data leakage.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    n_splits : int
        Number of splits
    test_size : int
        Size of test set
    gap : int
        Gap between train and test to prevent leakage
    
    Yields:
    -------
    train_idx, test_idx : arrays
    """
    n_samples = len(X)
    
    # Calculate split points
    test_starts = np.linspace(n_samples // 2, n_samples - test_size, n_splits, dtype=int)
    
    for test_start in test_starts:
        train_end = test_start - gap  # Gap between train and test
        test_end = min(test_start + test_size, n_samples)
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        
        yield train_idx, test_idx

print("\\n" + "="*70)
print("Blocked Time Series Split with Gap:")
print("="*70)

for fold, (train_idx, test_idx) in enumerate(
    blocked_time_series_split(X_ts, y_ts, n_splits=5, test_size=20, gap=5), 1
):
    print(f"Fold {fold}:")
    print(f"  Train: indices 0 to {train_idx[-1]} ({len(train_idx)} samples)")
    print(f"  Gap:   indices {train_idx[-1]+1} to {test_idx[0]-1} (not used)")
    print(f"  Test:  indices {test_idx[0]} to {test_idx[-1]} ({len(test_idx)} samples)")

print("\\n✓ Gap prevents leakage from auto-correlated features")
\`\`\`

## Nested Cross-Validation

Nested CV provides unbiased performance estimates when performing hyperparameter tuning.

**Structure:**
- Outer loop: Estimates generalization performance
- Inner loop: Performs hyperparameter tuning

\`\`\`python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

# Load data
X, y = cancer.data, cancer.target

print("Nested Cross-Validation")
print("="*70)

# Non-nested (WRONG for unbiased estimate)
print("\\n❌ Non-Nested CV (biased estimate):")
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
scores_non_nested = cross_val_score(grid_search, X, y, cv=5)
print(f"Accuracy: {scores_non_nested.mean():.4f} (+/- {scores_non_nested.std():.4f})")
print("⚠️  This estimate is optimistically biased!")

# Nested CV (CORRECT)
print("\\n✓ Nested CV (unbiased estimate):")

def nested_cv_score(X, y, outer_cv=5, inner_cv=5):
    """Perform nested cross-validation."""
    outer_scores = []
    
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV: Hyperparameter tuning
        inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            SVC(), param_grid, cv=inner_cv_splitter, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Evaluate best model on outer test fold
        score = grid_search.score(X_test, y_test)
        outer_scores.append(score)
        
        print(f"Outer Fold {fold}: Accuracy = {score:.4f}, Best params = {grid_search.best_params_}")
    
    return np.array(outer_scores)

nested_scores = nested_cv_score(X, y, outer_cv=5, inner_cv=3)
print(f"\\nNested CV Accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")
print("✓ This is an unbiased estimate of generalization performance")

print(f"\\nBias difference: {(scores_non_nested.mean() - nested_scores.mean())*100:.2f} percentage points")
\`\`\`

## Choosing the Right CV Strategy

\`\`\`python
def recommend_cv_strategy(n_samples, task_type, has_time_component, is_balanced=True):
    """
    Recommend appropriate cross-validation strategy.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples in dataset
    task_type : str
        'classification' or 'regression'
    has_time_component : bool
        Whether data has temporal ordering
    is_balanced : bool
        Whether classes are balanced (for classification)
    
    Returns:
    --------
    recommendation : dict
    """
    if has_time_component:
        return {
            'method': 'TimeSeriesSplit',
            'reason': 'Data has temporal ordering',
            'n_splits': 5,
            'notes': 'Consider adding gap to prevent leakage'
        }
    
    if n_samples < 50:
        return {
            'method': 'LeaveOneOut',
            'reason': 'Very small dataset',
            'n_splits': n_samples,
            'notes': 'Maximum use of limited data'
        }
    
    if n_samples < 200:
        return {
            'method': 'StratifiedKFold' if task_type == 'classification' else 'KFold',
            'reason': 'Small dataset',
            'n_splits': 10,
            'notes': '10-fold recommended for small datasets'
        }
    
    if task_type == 'classification' and not is_balanced:
        return {
            'method': 'StratifiedKFold',
            'reason': 'Imbalanced classification',
            'n_splits': 5,
            'notes': 'Maintains class distribution'
        }
    
    return {
        'method': 'KFold',
        'reason': 'Standard case',
        'n_splits': 5,
        'notes': '5-fold is efficient and reliable'
    }

# Example recommendations
scenarios = [
    (1000, 'classification', False, True),
    (50, 'regression', False, True),
    (500, 'classification', False, False),
    (1000, 'regression', True, True),
    (100, 'classification', False, True),
]

print("\\nCV Strategy Recommendations:")
print("="*70)

for n, task, time_dep, balanced in scenarios:
    rec = recommend_cv_strategy(n, task, time_dep, balanced)
    print(f"\\nScenario: N={n}, {task}, time={'Yes' if time_dep else 'No'}, "
          f"balanced={'Yes' if balanced else 'No'}")
    print(f"→ Use: {rec['method']} (k={rec['n_splits']})")
    print(f"  Reason: {rec['reason']}")
    print(f"  Note: {rec['notes']}")
\`\`\`

## Common Pitfalls and Best Practices

### Pitfall 1: Data Leakage in Preprocessing

\`\`\`python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\\nData Leakage in Cross-Validation:")
print("="*70)

# ❌ WRONG: Fit scaler on full dataset before CV
scaler = StandardScaler()
X_scaled_wrong = scaler.fit_transform(X)  # Leakage!
scores_wrong = cross_val_score(Ridge(), X_scaled_wrong, y, cv=5)
print(f"❌ Wrong (with leakage): R² = {scores_wrong.mean():.4f}")

# ✓ CORRECT: Use Pipeline so scaling happens within each fold
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
scores_correct = cross_val_score(pipeline, X, y, cv=5)
print(f"✓ Correct (no leakage): R² = {scores_correct.mean():.4f}")

print(f"\\nOptimistic bias from leakage: {(scores_wrong.mean() - scores_correct.mean()):.4f}")
\`\`\`

### Pitfall 2: Not Enough Folds

\`\`\`python
# Analyze effect of number of folds
print("\\n" + "="*70)
print("Effect of Number of Folds:")
print("="*70)

k_values = [2, 3, 5, 10, 20]
results = []

for k in k_values:
    # Run CV multiple times with different random states
    score_distributions = []
    for seed in range(10):
        cv = KFold(n_splits=k, shuffle=True, random_state=seed)
        scores = cross_val_score(Ridge(), X, y, cv=cv)
        score_distributions.append(scores.mean())
    
    results.append({
        'k': k,
        'mean_cv_score': np.mean(score_distributions),
        'variance_across_runs': np.var(score_distributions),
        'mean_fold_std': np.std(score_distributions)
    })

print(f"{'K':>3s} {'Mean Score':>12s} {'Variance':>12s} {'Stability':>12s}")
print("-"*70)
for r in results:
    print(f"{r['k']:3d} {r['mean_cv_score']:12.4f} {r['variance_across_runs']:12.6f} "
          f"{'High' if r['variance_across_runs'] < 0.0001 else 'Low':>12s}")

print("\\nRecommendation: Use k=5 or k=10 for good bias-variance tradeoff")
\`\`\`

## Trading Application: Strategy Validation

\`\`\`python
# Example: Validating a trading strategy with proper CV
import pandas as pd

def create_trading_features(prices):
    """Create technical indicators."""
    df = pd.DataFrame({'price': prices})
    df['returns'] = df['price'].pct_change()
    df['sma_5'] = df['price'].rolling(5).mean()
    df['sma_20'] = df['price'].rolling(20).mean()
    df['volatility'] = df['returns'].rolling(10).std()
    df['momentum'] = df['price'].pct_change(10)
    
    # Target: next period return
    df['target'] = df['price'].shift(-1) / df['price'] - 1
    df['target_binary'] = (df['target'] > 0).astype(int)
    
    return df.dropna()

# Generate synthetic price data
np.random.seed(42)
n_days = 500
price_data = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))

trading_df = create_trading_features(price_data)
X_trading = trading_df[['returns', 'sma_5', 'sma_20', 'volatility', 'momentum']].values
y_trading = trading_df['target_binary'].values

print("\\nTrading Strategy Cross-Validation:")
print("="*70)

# Must use TimeSeriesSplit for trading!
tscv = TimeSeriesSplit(n_splits=5)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_results = cross_validate(
    model, X_trading, y_trading, 
    cv=tscv,
    scoring=['accuracy', 'precision', 'recall'],
    return_train_score=True
)

print("Strategy Performance Across Time Periods:")
print("-"*70)
print(f"{'Metric':<12s} {'Train Mean':<12s} {'Test Mean':<12s} {'Test Std':<12s}")
print("-"*70)

for metric in ['accuracy', 'precision', 'recall']:
    train_mean = cv_results[f'train_{metric}'].mean()
    test_mean = cv_results[f'test_{metric}'].mean()
    test_std = cv_results[f'test_{metric}'].std()
    print(f"{metric:<12s} {train_mean:>11.4f} {test_mean:>11.4f} {test_std:>11.4f}")

print("\\n✓ TimeSeriesSplit ensures realistic backtest")
print("✓ Performance metrics show strategy stability over time")
\`\`\`

## Key Takeaways

1. **K-fold CV** (k=5 or 10) is the standard for most problems
2. **Stratified K-fold** for classification, especially with imbalanced data
3. **TimeSeriesSplit** is mandatory for temporal data
4. **LOOCV** only for very small datasets (N < 100)
5. **Nested CV** for unbiased estimates during hyperparameter tuning
6. **Always use Pipeline** to prevent data leakage
7. More folds = less bias but more variance and computation
8. Report mean ± std to show estimate uncertainty

## Further Reading

- Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation"
- Bergmeir, C., & Benítez, J. M. (2012). "On the use of cross-validation for time series predictor evaluation"
- Varma, S., & Simon, R. (2006). "Bias in error estimation when using cross-validation for model selection"
`,
  exercises: [
    {
      prompt:
        'Implement a custom cross-validation function that performs stratified time series splits - combining the benefits of stratification (for maintaining class balance) with time series ordering (for temporal data). Test it on a financial classification problem.',
      solution: `
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

class StratifiedTimeSeriesSplit:
    """
    Time series cross-validation with stratification.
    Useful for classification problems with temporal ordering.
    """
    
    def __init__(self, n_splits=5, test_size=0.2, gap=0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y):
        """
        Generate stratified time series splits.
        
        Strategy:
        1. Divide time into sequential blocks
        2. Within each block, stratify the test set
        3. Use all previous data for training
        """
        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)
        
        # Calculate split points
        min_train_size = int(n_samples * 0.3)
        
        for i in range(self.n_splits):
            # Expanding window
            split_point = min_train_size + int((n_samples - test_samples - min_train_size) * (i + 1) / self.n_splits)
            
            # Training: all data up to split point
            train_end = split_point - self.gap
            train_indices = np.arange(0, train_end)
            
            # Test: stratified sample from remaining data
            remaining_indices = np.arange(split_point, min(split_point + test_samples * 2, n_samples))
            
            if len(remaining_indices) == 0:
                continue
            
            y_remaining = y[remaining_indices]
            
            # Stratified sampling
            test_size_fold = min(test_samples, len(remaining_indices))
            
            # Group by class
            class_indices = {}
            for cls in np.unique(y_remaining):
                class_indices[cls] = remaining_indices[y_remaining == cls]
            
            # Sample proportionally from each class
            test_indices = []
            for cls, indices in class_indices.items():
                n_samples_cls = int(test_size_fold * len(indices) / len(remaining_indices))
                n_samples_cls = min(n_samples_cls, len(indices))
                if n_samples_cls > 0:
                    selected = np.random.choice(indices, n_samples_cls, replace=False)
                    test_indices.extend(selected)
            
            test_indices = np.array(test_indices)
            
            yield train_indices, test_indices
    
    def get_n_splits(self):
        return self.n_splits

# Generate synthetic financial data
np.random.seed(42)
n_days = 500

# Simulate price with regime changes
regime_changes = [0, 150, 300, 450, 500]
regimes = ['bull', 'bear', 'bull', 'bear']
prices = []

for i in range(len(regimes)):
    start = regime_changes[i]
    end = regime_changes[i + 1]
    n_samples = end - start
    
    if regimes[i] == 'bull':
        returns = np.random.randn(n_samples) * 0.015 + 0.001
    else:
        returns = np.random.randn(n_samples) * 0.02 - 0.001
    
    if i == 0:
        prices.extend([100])
    
    prices.extend(prices[-1] * np.exp(np.cumsum(returns)))

prices = np.array(prices)[:n_days]

# Create features and target
df = pd.DataFrame({'price': prices})
df['returns'] = df['price'].pct_change()
df['sma_10'] = df['price'].rolling(10).mean()
df['sma_30'] = df['price'].rolling(30).mean()
df['volatility'] = df['returns'].rolling(20).std()
df['momentum'] = df['price'].pct_change(10)

# Binary classification: will price go up?
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
df = df.dropna()

X = df[['returns', 'sma_10', 'sma_30', 'volatility', 'momentum']].values
y = df['target'].values

print("Stratified Time Series Split Demo")
print("="*70)
print(f"Dataset: {len(X)} samples")
print(f"Class distribution: {Counter(y)}")
print(f"Class 0: {(y==0).sum()/len(y)*100:.1f}%, Class 1: {(y==1).sum()/len(y)*100:.1f}%")

# Test our custom splitter
cv = StratifiedTimeSeriesSplit(n_splits=5, test_size=0.2, gap=5)

print("\\n" + "="*70)
print("Cross-Validation Folds:")
print("="*70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

overall_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Check stratification
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    
    print(f"\\nFold {fold}:")
    print(f"  Train: indices 0-{train_idx[-1]} ({len(train_idx)} samples)")
    print(f"    Class 0: {train_dist[0]} ({train_dist[0]/len(y_train)*100:.1f}%)")
    print(f"    Class 1: {train_dist[1]} ({train_dist[1]/len(y_train)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} samples from indices {test_idx.min()}-{test_idx.max()}")
    print(f"    Class 0: {test_dist[0]} ({test_dist[0]/len(y_test)*100:.1f}%)")
    print(f"    Class 1: {test_dist[1]} ({test_dist[1]/len(y_test)*100:.1f}%)")
    
    # Train and evaluate
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    overall_scores.append({'accuracy': acc, 'f1': f1})

print("\\n" + "="*70)
print("Overall Performance:")
print("="*70)
acc_scores = [s['accuracy'] for s in overall_scores]
f1_scores = [s['f1'] for s in overall_scores]
print(f"Accuracy: {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
print(f"F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

print("\\n✓ Time series ordering maintained (no future data in training)")
print("✓ Class balance preserved in test sets (stratification)")
print("✓ Realistic evaluation for financial classification")
`,
    },
  ],
  quizId: 'model-evaluation-optimization-cross-validation',
  multipleChoiceId: 'model-evaluation-optimization-cross-validation-mc',
};
