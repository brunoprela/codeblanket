export const trainTestSplitValidation = {
  title: 'Train-Test Split & Validation',
  content: `
# Train-Test Split & Validation

## Introduction

Model evaluation is the cornerstone of building reliable machine learning systems. Without proper evaluation, you cannot know whether your model will perform well on unseen data, whether you're overfitting, or how to compare different modeling approaches. This section covers the fundamental techniques for splitting data and validating model performance.

**Why Split Data?**

The primary goal of machine learning is to build models that **generalize** well to new, unseen data. If we train and evaluate a model on the same data, we're measuring how well the model memorizes the training examples, not how well it generalizes. This is analogous to studying for an exam using only the exact questions that will appear on the test—you'll ace the test but won't actually understand the subject.

**The Fundamental Principle**: Never evaluate your model on data it has seen during training.

## The Three-Way Split: Train, Validation, and Test

### Training Set
The training set is used to fit model parameters (e.g., weights in neural networks, coefficients in linear regression, tree structures in decision trees).

**Purpose**: Learn patterns and relationships in the data.

**Typical Size**: 60-80% of total data.

### Validation Set
The validation set is used for model selection and hyperparameter tuning. It provides an unbiased evaluation during the development phase.

**Purpose**: 
- Compare different models
- Tune hyperparameters
- Decide when to stop training (early stopping)
- Make decisions about feature engineering

**Typical Size**: 10-20% of total data.

### Test Set
The test set provides the final, unbiased evaluation of the model's performance. It should only be used once, at the very end of your model development process.

**Purpose**: 
- Estimate real-world performance
- Report final metrics
- Compare with benchmark models

**Typical Size**: 10-20% of total data.

**Critical Rule**: The test set is sacred. Touch it only once at the end!

## Random Splitting

Random splitting is the simplest approach, suitable when your data points are independent and identically distributed (i.i.d.).

### Implementation with Scikit-learn

\`\`\`python
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load example dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training (20% of remaining)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
)

print(f"\\nAfter splitting:")
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on all three sets
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"\\nModel Performance (RMSE):")
print(f"Training:   {train_rmse:.2f}")
print(f"Validation: {val_rmse:.2f}")
print(f"Test:       {test_rmse:.2f}")

# Check for overfitting
if train_rmse < val_rmse * 0.8:
    print("\\n⚠️  Warning: Significant gap between train and val suggests overfitting")
elif train_rmse > val_rmse * 1.2:
    print("\\n⚠️  Warning: Val better than train is unusual - check data leakage")
else:
    print("\\n✓ Reasonable train/val gap")
\`\`\`

### The Importance of random_state

The \`random_state\` parameter ensures reproducibility. Always set it in your experiments:

\`\`\`python
# Without random_state: different splits each time
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)

print(f"First split overlap: {np.intersect1d(X_train1[:, 0], X_train2[:, 0]).shape[0]}")
print("Results will vary each run!")

# With random_state: reproducible splits
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\\nWith random_state:")
print(f"Splits identical: {np.array_equal(X_train1, X_train2)}")
\`\`\`

## Stratified Splitting

When dealing with classification problems, especially with imbalanced classes, stratified splitting maintains the same class distribution across train, validation, and test sets.

### Why Stratify?

Consider a binary classification problem with 95% negative and 5% positive examples. With random splitting, you might end up with:
- Training set: 94% negative, 6% positive
- Test set: 98% negative, 2% positive

This imbalance can lead to misleading evaluations.

### Implementation

\`\`\`python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter

# Load imbalanced dataset example
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("Original class distribution:")
print(Counter(y))
print(f"Class 0: {(y==0).sum()/len(y)*100:.1f}%")
print(f"Class 1: {(y==1).sum()/len(y)*100:.1f}%")

# Random split (not stratified)
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\\n--- Random Split ---")
print(f"Train - Class 0: {(y_train_rand==0).sum()/len(y_train_rand)*100:.1f}%")
print(f"Train - Class 1: {(y_train_rand==1).sum()/len(y_train_rand)*100:.1f}%")
print(f"Test  - Class 0: {(y_test_rand==0).sum()/len(y_test_rand)*100:.1f}%")
print(f"Test  - Class 1: {(y_test_rand==1).sum()/len(y_test_rand)*100:.1f}%")

# Stratified split
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\\n--- Stratified Split ---")
print(f"Train - Class 0: {(y_train_strat==0).sum()/len(y_train_strat)*100:.1f}%")
print(f"Train - Class 1: {(y_train_strat==1).sum()/len(y_train_strat)*100:.1f}%")
print(f"Test  - Class 0: {(y_test_strat==0).sum()/len(y_test_strat)*100:.1f}%")
print(f"Test  - Class 1: {(y_test_strat==1).sum()/len(y_test_strat)*100:.1f}%")
\`\`\`

### Stratified Multi-class Example

\`\`\`python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

print("Multi-class distribution:")
for class_id in range(3):
    count = (y == class_id).sum()
    print(f"Class {class_id}: {count} samples ({count/len(y)*100:.1f}%)")

# Stratified split preserves distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\\nAfter stratified split:")
print("Training set:")
for class_id in range(3):
    count = (y_train == class_id).sum()
    print(f"  Class {class_id}: {count} samples ({count/len(y_train)*100:.1f}%)")

print("Test set:")
for class_id in range(3):
    count = (y_test == class_id).sum()
    print(f"  Class {class_id}: {count} samples ({count/len(y_test)*100:.1f}%)")
\`\`\`

## Time-Based Splitting

For time series data or any temporal dataset, random splitting violates the temporal order and creates data leakage. You must use time-based splitting.

### The Problem with Random Splits in Time Series

\`\`\`python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate synthetic stock price data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n = len(dates)

# Simulate price with trend and noise
trend = np.linspace(100, 150, n)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.randn(n) * 5
price = trend + seasonality + noise

df = pd.DataFrame({
    'date': dates,
    'price': price
})

print("Stock price dataset:")
print(df.head())
print(f"Total days: {len(df)}")

# WRONG: Random split on time series
X = df[['price']].values[:-1]  # Use price[t] to predict price[t+1]
y = df['price'].values[1:]
dates_X = dates[:-1]

X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong, dates_train_wrong, dates_test_wrong = train_test_split(
    X, y, dates_X, test_size=0.2, random_state=42
)

print("\\n❌ WRONG: Random split creates data leakage!")
print(f"Training date range: {dates_train_wrong.min()} to {dates_train_wrong.max()}")
print(f"Test date range: {dates_test_wrong.min()} to {dates_test_wrong.max()}")
print("Problem: Using future data to predict the past!")

# CORRECT: Time-based split
train_size = int(0.8 * len(X))
X_train_correct = X[:train_size]
y_train_correct = y[:train_size]
X_test_correct = X[train_size:]
y_test_correct = y[train_size:]
dates_train_correct = dates_X[:train_size]
dates_test_correct = dates_X[train_size:]

print("\\n✓ CORRECT: Time-based split maintains temporal order")
print(f"Training: {dates_train_correct.min()} to {dates_train_correct.max()}")
print(f"Test:     {dates_test_correct.min()} to {dates_test_correct.max()}")
print("Training always comes before test!")
\`\`\`

### Walk-Forward Validation for Time Series

Walk-forward validation (also called rolling window) is essential for realistic time series evaluation:

\`\`\`python
def walk_forward_split(X, y, n_splits=5, test_size=0.2):
    """
    Create walk-forward validation splits for time series.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    n_splits : int
        Number of splits
    test_size : float
        Proportion of data in test set
    
    Yields:
    -------
    train_idx, test_idx : arrays
        Indices for train and test
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    train_samples = n_samples - test_samples
    
    # Initial split
    min_train = int(train_samples * 0.3)  # Minimum training size
    
    for i in range(n_splits):
        # Expanding window: training set grows
        train_end = min_train + int((train_samples - min_train) * (i + 1) / n_splits)
        test_start = train_end
        test_end = min(test_start + test_samples // n_splits, n_samples)
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        
        yield train_idx, test_idx

# Example usage
from sklearn.linear_model import Ridge

scores = []
split_info = []

print("Walk-Forward Validation:")
print("-" * 70)

for fold, (train_idx, test_idx) in enumerate(walk_forward_split(X, y, n_splits=5), 1):
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_test_fold = X[test_idx]
    y_test_fold = y[test_idx]
    
    # Train and evaluate
    model = Ridge(alpha=1.0)
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_test_fold, y_test_fold)
    scores.append(score)
    
    train_dates = dates_X[train_idx]
    test_dates = dates_X[test_idx]
    
    split_info.append({
        'fold': fold,
        'train_start': train_dates.min(),
        'train_end': train_dates.max(),
        'test_start': test_dates.min(),
        'test_end': test_dates.max(),
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        'r2_score': score
    })
    
    print(f"Fold {fold}:")
    print(f"  Train: {train_dates.min().date()} to {train_dates.max().date()} ({len(train_idx)} samples)")
    print(f"  Test:  {test_dates.min().date()} to {test_dates.max().date()} ({len(test_idx)} samples)")
    print(f"  R² Score: {score:.4f}")
    print()

print(f"Average R² Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
\`\`\`

## Validation Set Purpose and Best Practices

### When to Use Validation Set

1. **Hyperparameter Tuning**: Use validation set to find optimal hyperparameters
2. **Model Selection**: Compare different algorithms
3. **Feature Engineering**: Test different feature combinations
4. **Early Stopping**: Prevent overfitting in iterative models

### Example: Complete Model Development Workflow

\`\`\`python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Prepare data
X_train_scaled = StandardScaler().fit_transform(X_train)
X_val_scaled = StandardScaler().fit_transform(X_val)
X_test_scaled = StandardScaler().fit_transform(X_test)

# Hyperparameter tuning on validation set
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

best_score = float('inf')
best_params = None
results = []

print("Hyperparameter tuning using validation set...")
print("-" * 60)

# Grid search manually (to show the process)
for n_est in param_grid['n_estimators']:
    for depth in param_grid['max_depth']:
        for min_split in param_grid['min_samples_split']:
            # Train model
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=depth,
                min_samples_split=min_split,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            results.append({
                'n_estimators': n_est,
                'max_depth': depth,
                'min_samples_split': min_split,
                'val_rmse': val_rmse
            })
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'min_samples_split': min_split
                }

print(f"Best parameters: {best_params}")
print(f"Best validation RMSE: {best_score:.4f}")

# Train final model with best parameters
final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train_scaled, y_train)

# Evaluate on all three sets
train_pred = final_model.predict(X_train_scaled)
val_pred = final_model.predict(X_val_scaled)
test_pred = final_model.predict(X_test_scaled)

print("\\nFinal Model Performance:")
print("-" * 60)
for name, y_true, y_pred in [
    ('Training', y_train, train_pred),
    ('Validation', y_val, val_pred),
    ('Test', y_test, test_pred)
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:12s} - RMSE: {rmse:7.2f}, MAE: {mae:7.2f}, R²: {r2:6.4f}")
\`\`\`

## Common Pitfalls and How to Avoid Them

### 1. Data Leakage

**Problem**: Information from the test set leaking into training.

\`\`\`python
# ❌ WRONG: Scaling before splitting
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses statistics from entire dataset!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

print("❌ Wrong approach: scaler saw test data")

# ✓ CORRECT: Split first, then scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)  # Transform test data with training statistics

print("✓ Correct approach: scaler only saw training data")
\`\`\`

### 2. Using Test Set Multiple Times

**Problem**: Repeatedly evaluating on test set during development.

\`\`\`python
# ❌ WRONG: Tuning hyperparameters using test set
best_alpha = None
best_test_score = 0

for alpha in [0.01, 0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)  # Using test set for tuning!
    
    if test_score > best_test_score:
        best_test_score = test_score
        best_alpha = alpha

print(f"❌ Best alpha: {best_alpha} (chosen using test set - biased!)")

# ✓ CORRECT: Use validation set for tuning
best_alpha = None
best_val_score = 0

for alpha in [0.01, 0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)  # Using validation set
    
    if val_score > best_val_score:
        best_val_score = val_score
        best_alpha = alpha

# Final evaluation on test set (only once!)
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)
final_test_score = final_model.score(X_test, y_test)

print(f"✓ Best alpha: {best_alpha} (chosen using validation set)")
print(f"✓ Final test score: {final_test_score:.4f} (unbiased estimate)")
\`\`\`

### 3. Insufficient Test Set Size

**Problem**: Test set too small to provide reliable estimates.

\`\`\`python
# Rule of thumb: Need at least 30-50 samples per class for classification
# For regression: at least 100-200 samples for stable estimates

def evaluate_split_sizes(X, y, test_sizes):
    """Evaluate how test set size affects estimate stability."""
    results = []
    
    for test_size in test_sizes:
        scores = []
        # Multiple random splits to see variance
        for seed in range(10):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )
            model = Ridge()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        results.append({
            'test_size': test_size,
            'test_samples': int(len(X) * test_size),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        })
    
    return results

# Test different split sizes
test_sizes = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
results = evaluate_split_sizes(X, y, test_sizes)

print("Effect of test set size on estimate stability:")
print("-" * 70)
print(f"{'Test %':>8s} {'N samples':>10s} {'Mean R²':>10s} {'Std R²':>10s}")
print("-" * 70)

for r in results:
    print(f"{r['test_size']*100:>7.0f}% {r['test_samples']:>10d} "
          f"{r['mean_score']:>10.4f} {r['std_score']:>10.4f}")

print("\\nRecommendation: Use 15-25% for test set with sufficient samples")
\`\`\`

## Best Practices Summary

### 1. **Split Strategy**
- Use 60/20/20 or 70/15/15 train/val/test split
- For small datasets (<1000 samples), consider cross-validation instead
- For time series, always use time-based splits

### 2. **Preprocessing**
- Always split data BEFORE any preprocessing
- Fit preprocessing only on training data
- Transform validation and test data using training statistics

### 3. **Evaluation Protocol**
- Use validation set for all development decisions
- Touch test set only once at the end
- Report all three set performances to check for overfitting

### 4. **Special Cases**
- Imbalanced data: Use stratified splits
- Time series: Use time-based or walk-forward splits
- Grouped data (e.g., multiple samples from same person): Use GroupShuffleSplit

## Trading Application: Backtesting with Proper Validation

\`\`\`python
# Example: Validating a trading strategy
import pandas as pd
import numpy as np

def create_trading_features(df):
    """Create technical indicators as features."""
    df = df.copy()
    
    # Simple moving averages
    df['sma_5'] = df['price'].rolling(5).mean()
    df['sma_20'] = df['price'].rolling(20).mean()
    
    # Returns
    df['returns'] = df['price'].pct_change()
    df['returns_5'] = df['price'].pct_change(5)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Target: next day return
    df['target'] = df['price'].shift(-1) / df['price'] - 1
    
    return df.dropna()

# Prepare trading data
trading_df = create_trading_features(df)

# Time-based split for trading (CRITICAL!)
train_end = '2022-12-31'
val_end = '2023-06-30'

train_data = trading_df[trading_df['date'] <= train_end]
val_data = trading_df[(trading_df['date'] > train_end) & (trading_df['date'] <= val_end)]
test_data = trading_df[trading_df['date'] > val_end]

print("Trading Model Validation:")
print(f"Training:   {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} days)")
print(f"Validation: {val_data['date'].min().date()} to {val_data['date'].max().date()} ({len(val_data)} days)")
print(f"Test:       {test_data['date'].min().date()} to {test_data['date'].max().date()} ({len(test_data)} days)")

# This ensures we never use future information to predict the past
# Critical for realistic backtest results!
\`\`\`

## Key Takeaways

1. **Never evaluate on training data** - it measures memorization, not generalization
2. **Use three-way split** - train, validation, test for proper model development
3. **Stratify for classification** - maintain class distribution across splits
4. **Time-based for temporal data** - never use future to predict past
5. **Avoid data leakage** - split before preprocessing
6. **Test set is sacred** - use only once for final evaluation
7. **Sufficient size matters** - ensure test set is large enough for stable estimates

## Further Reading

- Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation"
- Bergstra, J. & Bengio, Y. (2012). "Random search for hyper-parameter optimization"
- Time Series Cross-Validation: https://robjhyndman.com/hyndsight/tscv/
`,
  exercises: [
    {
      prompt:
        'Create a function that performs a proper three-way split with stratification for a classification dataset, and verify that class distributions are preserved across all three sets.',
      solution: `
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def three_way_stratified_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Perform three-way stratified split maintaining class distribution.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like  
        Target labels
    train_size : float
        Proportion for training (default 0.6)
    val_size : float
        Proportion for validation (default 0.2)
    test_size : float
        Proportion for test (default 0.2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1"
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: separate validation from training
    # val_size / (train_size + val_size) of the temp set
    val_size_adjusted = val_size / (train_size + val_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def verify_stratification(y_train, y_val, y_test, y_full):
    """Verify that class distributions are preserved."""
    print("Class Distribution Verification:")
    print("=" * 70)
    
    # Get unique classes
    classes = np.unique(y_full)
    
    # Calculate distributions
    full_dist = Counter(y_full)
    train_dist = Counter(y_train)
    val_dist = Counter(y_val)
    test_dist = Counter(y_test)
    
    print(f"{'Class':<10s} {'Full':<12s} {'Train':<12s} {'Val':<12s} {'Test':<12s}")
    print("-" * 70)
    
    for cls in classes:
        full_pct = full_dist[cls] / len(y_full) * 100
        train_pct = train_dist[cls] / len(y_train) * 100
        val_pct = val_dist[cls] / len(y_val) * 100
        test_pct = test_dist[cls] / len(y_test) * 100
        
        print(f"{cls:<10d} {full_pct:>10.2f}% {train_pct:>10.2f}% "
              f"{val_pct:>10.2f}% {test_pct:>10.2f}%")
    
    # Check if distributions are close
    max_deviation = 0
    for cls in classes:
        full_pct = full_dist[cls] / len(y_full)
        for dist, name in [(train_dist, 'train'), (val_dist, 'val'), (test_dist, 'test')]:
            n_samples = len(y_train) if name == 'train' else (len(y_val) if name == 'val' else len(y_test))
            set_pct = dist[cls] / n_samples
            deviation = abs(full_pct - set_pct)
            max_deviation = max(max_deviation, deviation)
    
    print(f"\\nMaximum deviation from full distribution: {max_deviation*100:.2f}%")
    
    if max_deviation < 0.05:  # Less than 5% deviation
        print("✓ Stratification successful!")
    else:
        print("⚠️  Warning: Significant deviation detected")

# Example usage
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Perform three-way split
X_train, X_val, X_test, y_train, y_val, y_test = three_way_stratified_split(
    X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
)

print("Split sizes:")
print(f"Training:   {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test:       {len(X_test)} samples")
print(f"Total:      {len(X)} samples\\n")

# Verify stratification
verify_stratification(y_train, y_val, y_test, y)
`,
    },
  ],
  quizId: 'model-evaluation-optimization-train-test',
  multipleChoiceId: 'model-evaluation-optimization-train-test-mc',
};
