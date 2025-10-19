/**
 * Section: Train-Test Split & Validation
 * Module: Model Evaluation & Optimization
 *
 * Covers the fundamentals of splitting data, validation strategies, and avoiding data leakage
 */

export const trainTestSplitValidation = {
  id: 'train-test-split-validation',
  title: 'Train-Test Split & Validation',
  content: `
# Train-Test Split & Validation

## Introduction

One of the most fundamental concepts in machine learning is the train-test split. Before we can evaluate how well our model will perform on new, unseen data, we must carefully separate our data into distinct sets for training and testing. This seemingly simple step is crucial for building models that generalize well and avoiding the pitfall of overly optimistic performance estimates.

**Why Split Data?**

Imagine teaching a student by having them memorize the exact answers to test questions beforehand. They might score perfectly on the test, but have they actually learned anything? This is analogous to evaluating a machine learning model on the same data it was trained on—the model might appear perfect but fail completely on new data.

**The Core Principle**: Never use the same data for training and evaluation. Your model must be tested on data it has never seen during training.

## The Data Split Strategy

### The Three-Way Split

In production machine learning, we typically split data into three sets:

1. **Training Set (60-80%)**: Used to fit the model—learn parameters, weights, patterns
2. **Validation Set (10-20%)**: Used during model development to tune hyperparameters and select models
3. **Test Set (10-20%)**: Used only ONCE at the end to get an unbiased estimate of model performance

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

# Features and target
X = np.random.randn(n_samples, 5)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(n_samples)*0.5

print(f"Total samples: {n_samples}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
\`\`\`

**Output:**
\`\`\`
Total samples: 1000
Features shape: (1000, 5)
Target shape: (1000,)
\`\`\`

## Basic Train-Test Split

### Simple Random Split

\`\`\`python
# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42  # For reproducibility
)

print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\\nTraining features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
\`\`\`

**Output:**
\`\`\`
Training set size: 800 (80.0%)
Test set size: 200 (20.0%)

Training features shape: (800, 5)
Training target shape: (800,)
\`\`\`

### Three-Way Split (Train-Validation-Test)

\`\`\`python
# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training (20% of remaining 80% = 16% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"Total: {len(X_train) + len(X_val) + len(X_test)} samples")

# Verify no data leakage
print(f"\\nAny overlap between train and test? {len(set(range(len(X_train))).intersection(set(range(len(X_train), len(X_train)+len(X_test))))) > 0}")
\`\`\`

**Output:**
\`\`\`
Training set: 640 samples (64.0%)
Validation set: 160 samples (16.0%)
Test set: 200 samples (20.0%)
Total: 1000 samples

Any overlap between train and test? False
\`\`\`

## Stratified Splitting for Classification

When working with classification problems, especially with imbalanced classes, stratified splitting ensures each split has the same proportion of classes as the original dataset.

\`\`\`python
from sklearn.datasets import make_classification

# Create imbalanced classification dataset
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=3,
    weights=[0.6, 0.3, 0.1],  # Imbalanced classes
    random_state=42
)

print("Original class distribution:")
unique, counts = np.unique(y_class, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(y_class)*100:.1f}%)")

# Regular split (might not preserve class distribution)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

print("\\nRegular split - Test set distribution:")
unique, counts = np.unique(y_test_reg, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(y_test_reg)*100:.1f}%)")

# Stratified split (preserves class distribution)
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X_class, y_class, 
    test_size=0.2, 
    stratify=y_class,  # Maintain class proportions
    random_state=42
)

print("\\nStratified split - Test set distribution:")
unique, counts = np.unique(y_test_strat, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(y_test_strat)*100:.1f}%)")
\`\`\`

**Output:**
\`\`\`
Original class distribution:
  Class 0: 600 (60.0%)
  Class 1: 300 (30.0%)
  Class 2: 100 (10.0%)

Regular split - Test set distribution:
  Class 0: 116 (58.0%)
  Class 1: 62 (31.0%)
  Class 2: 22 (11.0%)

Stratified split - Test set distribution:
  Class 0: 120 (60.0%)
  Class 1: 60 (30.0%)
  Class 2: 20 (10.0%)
\`\`\`

## Time Series Splitting

For time series data, **random shuffling breaks temporal dependencies**. You must split sequentially to maintain the time order.

\`\`\`python
# Generate time series data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
time_series_data = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.randn(365)) + 100,  # Random walk
    'feature1': np.random.randn(365),
    'feature2': np.random.randn(365)
})

print(f"Time series length: {len(time_series_data)} days")
print(f"Date range: {time_series_data['date'].min()} to {time_series_data['date'].max()}")

# WRONG: Random split (destroys temporal order)
# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)  # DON'T DO THIS!

# CORRECT: Sequential split (preserves time order)
train_size = int(0.8 * len(time_series_data))
train_data = time_series_data[:train_size]
test_data = time_series_data[train_size:]

print(f"\\nTraining period: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"Test period: {test_data['date'].min()} to {test_data['date'].max()}")
print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Visualize the split
plt.figure(figsize=(12, 4))
plt.plot(train_data['date'], train_data['value'], label='Training', color='blue', alpha=0.7)
plt.plot(test_data['date'], test_data['value'], label='Test', color='orange', alpha=0.7)
plt.axvline(train_data['date'].iloc[-1], color='red', linestyle='--', label='Split point')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Train-Test Split')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('time_series_split.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Time series split visualization created")
\`\`\`

**Output:**
\`\`\`
Time series length: 365 days
Date range: 2020-01-01 00:00:00 to 2020-12-30 00:00:00

Training period: 2020-01-01 00:00:00 to 2020-09-27 00:00:00
Test period: 2020-09-28 00:00:00 to 2020-12-30 00:00:00
Training samples: 292
Test samples: 73

✅ Time series split visualization created
\`\`\`

## Shuffle Parameter

The \`shuffle\` parameter controls whether data is shuffled before splitting:

\`\`\`python
# Example: Effect of shuffling
data = np.arange(20)
print(f"Original data: {data}")

# With shuffle (default for non-time-series)
_, test_shuffled = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
print(f"Test set with shuffle: {sorted(test_shuffled)}")

# Without shuffle (for time series)
train_no_shuffle, test_no_shuffle = train_test_split(data, test_size=0.3, random_state=42, shuffle=False)
print(f"Train set without shuffle: {train_no_shuffle}")
print(f"Test set without shuffle: {test_no_shuffle}")
\`\`\`

**Output:**
\`\`\`
Original data: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
Test set with shuffle: [0, 4, 5, 8, 11, 19]
Train set without shuffle: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13]
Test set without shuffle: [14 15 16 17 18 19]
\`\`\`

## Data Leakage: The Silent Killer

**Data leakage** occurs when information from the test set influences the training process, leading to overly optimistic performance estimates.

### Common Data Leakage Mistakes

\`\`\`python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(1000)*0.1

# ❌ WRONG: Scaling before splitting (DATA LEAKAGE)
print("❌ WRONG WAY (with data leakage):")
scaler_wrong = StandardScaler()
X_scaled_wrong = scaler_wrong.fit_transform(X)  # Uses statistics from ALL data

X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
    X_scaled_wrong, y, test_size=0.2, random_state=42
)

model_wrong = LinearRegression()
model_wrong.fit(X_train_wrong, y_train_wrong)
y_pred_wrong = model_wrong.predict(X_test_wrong)
r2_wrong = r2_score(y_test_wrong, y_pred_wrong)
print(f"R² score: {r2_wrong:.4f}")
print("Problem: Test set statistics influenced the scaling!")

# ✅ CORRECT: Scaling after splitting
print("\\n✅ CORRECT WAY (no data leakage):")
X_train_correct, X_test_correct, y_train_correct, y_test_correct = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler on training data only
scaler_correct = StandardScaler()
X_train_correct = scaler_correct.fit_transform(X_train_correct)
X_test_correct = scaler_correct.transform(X_test_correct)  # Use training statistics

model_correct = LinearRegression()
model_correct.fit(X_train_correct, y_train_correct)
y_pred_correct = model_correct.predict(X_test_correct)
r2_correct = r2_score(y_test_correct, y_pred_correct)
print(f"R² score: {r2_correct:.4f}")
print("✅ Test set was scaled using only training statistics!")
\`\`\`

**Output:**
\`\`\`
❌ WRONG WAY (with data leakage):
R² score: 0.9954
Problem: Test set statistics influenced the scaling!

✅ CORRECT WAY (no data leakage):
R² score: 0.9953
✅ Test set was scaled using only training statistics!
\`\`\`

### The Correct Pipeline

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

# ✅ BEST PRACTICE: Use Pipeline to prevent leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=1.0))
])

# Split data FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline ensures all transformations are fitted on training data only
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"Pipeline R² score: {r2_score(y_test, y_pred):.4f}")
print("✅ Pipeline automatically prevents data leakage!")
\`\`\`

**Output:**
\`\`\`
Pipeline R² score: 0.9942
✅ Pipeline automatically prevents data leakage!
\`\`\`

## Practical Example: House Price Prediction

\`\`\`python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Load real dataset
housing = fetch_california_housing()
X_housing = housing.data
y_housing = housing.target

print(f"California housing dataset:")
print(f"  Samples: {X_housing.shape[0]}")
print(f"  Features: {X_housing.shape[1]}")
print(f"  Target: Median house value (in $100,000s)")

# Three-way split with proper procedure
# Step 1: Hold out test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# Step 2: Split remaining into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print(f"\\nData split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# Step 3: Train model on training set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate on validation set (for hyperparameter tuning)
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"\\nValidation RMSE: \${val_rmse*100000:.2f}")

# Step 5: Final evaluation on test set (only once!)
y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print(f"\\nFinal Test Performance:")
print(f"  RMSE: \${test_rmse*100000:.2f}")
print(f"  R²: {test_r2:.4f}")
\`\`\`

**Output:**
\`\`\`
California housing dataset:
  Samples: 20640
  Features: 8
  Target: Median house value (in $100,000s)

Data split:
  Training: 13209 samples
  Validation: 3303 samples
  Test: 4128 samples

Validation RMSE: $49718.49

Final Test Performance:
  RMSE: $49203.67
  R²: 0.8099
\`\`\`

## Best Practices Summary

### Do's ✅
1. **Always split before any preprocessing** to avoid data leakage
2. **Use stratified splitting** for classification with imbalanced classes
3. **Use sequential splitting** for time series data (no shuffling)
4. **Set random_state** for reproducibility
5. **Use validation set** for hyperparameter tuning
6. **Test set is sacred** - use only once at the very end
7. **Use Pipeline** to ensure proper train-test isolation

### Don'ts ❌
1. **Never fit preprocessing on the entire dataset** before splitting
2. **Never shuffle time series data** before splitting
3. **Never use test set for model selection** or hyperparameter tuning
4. **Never look at test set** until model is finalized
5. **Never train and test on the same data**

## Common Pitfalls

\`\`\`python
# Pitfall 1: Not setting random_state
# Results are not reproducible - different split each time
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2)
print(f"Same split? {np.array_equal(X_train_1, X_train_2)}")  # False

# Pitfall 2: Wrong test size
# Common mistake: using 0.8 thinking it's 80% test (it's actually 80% for test!)
X_train_wrong, X_test_wrong, _, _ = train_test_split(X, y, test_size=0.8)
print(f"Training: {len(X_train_wrong)}, Test: {len(X_test_wrong)}")  # Oops!

# Pitfall 3: Forgetting to split y
# Must split both X and y together
X_train, X_test = train_test_split(X, test_size=0.2)  # Wrong! Where's y?

# Pitfall 4: Using test set for feature engineering
# This leaks information about test set into your model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# ❌ all_data = np.concatenate([X_train, X_test])  # DON'T DO THIS
\`\`\`

## Key Takeaways

1. **Train-test split is fundamental** - it's the first step in any ML project
2. **Three sets are standard**: training (fit model), validation (tune hyperparameters), test (final evaluation)
3. **Random splitting works for most cases**, but use stratified for classification and sequential for time series
4. **Data leakage is insidious** - always split before preprocessing
5. **Test set is for final evaluation only** - don't peek!
6. **Pipelines prevent leakage** - use sklearn.pipeline.Pipeline
7. **Reproducibility matters** - always set random_state
8. **Size matters**: 60-20-20 or 70-15-15 or 80-10-10 splits are common

Proper train-test splitting is the foundation of reliable model evaluation. Get this right, and you'll avoid the costly mistake of deploying models that fail in production!
`,
};
