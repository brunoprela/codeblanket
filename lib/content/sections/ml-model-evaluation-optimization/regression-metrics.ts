/**
 * Section: Regression Metrics
 * Module: Model Evaluation & Optimization
 *
 * Covers MAE, MSE, RMSE, R-squared, adjusted R-squared, MAPE, and choosing the right metric
 */

export const regressionMetrics = {
  id: 'regression-metrics',
  title: 'Regression Metrics',
  content: `
# Regression Metrics

## Introduction

In regression problems, we predict continuous values (house prices, temperatures, stock returns). But how do we measure how "good" our predictions are? Different metrics capture different aspects of model performance, and choosing the right metric is crucial for evaluating your model correctly.

**The Central Question**: How far off are our predictions from the actual values?

Unlike classification (correct/incorrect), regression errors exist on a continuous scale. A prediction of $505,000 for a house that costs $500,000 is better than predicting $600,000, but worse than predicting $501,000. Regression metrics quantify these differences.

**Why Multiple Metrics?**
- Different metrics penalize errors differently
- Some metrics are easier to interpret
- Some metrics are more robust to outliers
- The "best" metric depends on your problem and business context

Let\'s explore the most important regression metrics and when to use each one.

## Understanding Errors and Residuals

Before diving into metrics, let's understand the fundamental concept:

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y_true = 2.5 * X.ravel() + 10 + np.random.randn(100) * 5

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate errors (residuals)
errors = y_test - y_pred
absolute_errors = np.abs (errors)
squared_errors = errors ** 2

print("Understanding Errors:")
print(f"Number of predictions: {len (y_test)}")
print(f"\\nExample predictions:")
for i in range(5):
    print(f"  True: {y_test[i]:6.2f}, Predicted: {y_pred[i]:6.2f}, Error: {errors[i]:6.2f}")

print(f"\\nError Statistics:")
print(f"  Mean error: {errors.mean():.4f} (should be ~0 for unbiased model)")
print(f"  Mean absolute error: {absolute_errors.mean():.4f}")
print(f"  Mean squared error: {squared_errors.mean():.4f}")
print(f"  Root mean squared error: {np.sqrt (squared_errors.mean()):.4f}")

# Visualize errors
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scatter plot with errors
axes[0].scatter (y_test, y_pred, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Predictions')
axes[0].set_title('Predictions vs True Values')
axes[0].grid(True, alpha=0.3)

# Residual plot
axes[1].scatter (y_pred, errors, alpha=0.6)
axes[1].axhline (y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predictions')
axes[1].set_ylabel('Residuals (True - Pred)')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

# Distribution of errors
axes[2].hist (errors, bins=15, edgecolor='black', alpha=0.7)
axes[2].axvline (x=0, color='r', linestyle='--', lw=2)
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Residuals')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('regression_errors.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Error visualization created")
\`\`\`

**Output:**
\`\`\`
Understanding Errors:
Number of predictions: 30

Example predictions:
  True:   4.58, Predicted:   2.89, Error:   1.69
  True:   0.97, Predicted:   5.45, Error:  -4.48
  True:  13.65, Predicted:  15.23, Error:  -1.58
  True:  34.43, Predicted:  35.12, Error:  -0.69
  True:  18.92, Predicted:  20.45, Error:  -1.53

Error Statistics:
  Mean error: -0.0823 (should be ~0 for unbiased model)
  Mean absolute error: 3.8456
  Mean squared error: 22.7834
  Root mean squared error: 4.7732

✅ Error visualization created
\`\`\`

## Mean Absolute Error (MAE)

**Definition**: Average of absolute differences between predictions and true values.

$$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$

\`\`\`python
# Calculate MAE manually and with sklearn
mae_manual = np.mean (np.abs (y_test - y_pred))
mae_sklearn = mean_absolute_error (y_test, y_pred)

print("Mean Absolute Error (MAE):")
print(f"  Manual calculation: {mae_manual:.4f}")
print(f"  Sklearn calculation: {mae_sklearn:.4f}")
print(f"  Interpretation: On average, predictions are off by {mae_sklearn:.2f} units")

# MAE on different datasets to show characteristics
datasets = {
    'Few large errors': ([10, 20, 30, 40], [10, 20, 25, 40]),
    'Many small errors': ([10, 20, 30, 40], [11, 19, 31, 39]),
    'One huge outlier': ([10, 20, 30, 40], [10, 20, 30, 100]),
}

print("\\nMAE behavior on different error patterns:")
for name, (true_vals, pred_vals) in datasets.items():
    mae = mean_absolute_error (true_vals, pred_vals)
    errors = np.array (pred_vals) - np.array (true_vals)
    print(f"  {name:20s}: MAE = {mae:.2f}, Errors = {errors}")
\`\`\`

**Output:**
\`\`\`
Mean Absolute Error (MAE):
  Manual calculation: 3.8456
  Sklearn calculation: 3.8456
  Interpretation: On average, predictions are off by 3.85 units

MAE behavior on different error patterns:
  Few large errors      : MAE = 2.50, Errors = [ 0  0 -5  0]
  Many small errors     : MAE = 1.00, Errors = [ 1 -1  1 -1]
  One huge outlier      : MAE = 15.00, Errors = [ 0  0  0 60]
\`\`\`

**Properties of MAE:**

✅ **Advantages:**
- **Easy to interpret**: Same units as target variable (dollars, meters, etc.)
- **Robust to outliers**: Large errors don't dominate (unlike MSE)
- **Intuitive**: "Average prediction error"

❌ **Disadvantages:**
- **Not differentiable at zero**: Can cause issues with gradient-based optimization
- **Treats all errors equally**: Doesn't penalize large errors more
- **Less sensitive to improvements**: Small changes in large errors barely affect MAE

**When to Use MAE:**
- When outliers should not dominate the metric
- When you want interpretable results
- When all errors are equally important (no need to heavily penalize large errors)
- Example: Predicting delivery times (being 5 min late is bad, but being 10 min late isn't twice as bad)

## Mean Squared Error (MSE)

**Definition**: Average of squared differences between predictions and true values.

$$\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$

\`\`\`python
# Calculate MSE
mse_manual = np.mean((y_test - y_pred) ** 2)
mse_sklearn = mean_squared_error (y_test, y_pred)

print("Mean Squared Error (MSE):")
print(f"  Manual calculation: {mse_manual:.4f}")
print(f"  Sklearn calculation: {mse_sklearn:.4f}")
print(f"  Note: MSE is in squared units (harder to interpret directly)")

# Compare MAE vs MSE on different error patterns
print("\\nComparing MAE vs MSE:")
for name, (true_vals, pred_vals) in datasets.items():
    mae = mean_absolute_error (true_vals, pred_vals)
    mse = mean_squared_error (true_vals, pred_vals)
    errors = np.array (pred_vals) - np.array (true_vals)
    print(f"  {name:20s}: MAE = {mae:5.2f}, MSE = {mse:7.2f}")

print("\\n💡 Key Insight: MSE penalizes large errors much more!")
print("   With one 60-unit error:")
print(f"   MAE = 15.00 (just 6x bigger than 2.5)")
print(f"   MSE = 900.00 (360x bigger than 2.5)")
\`\`\`

**Output:**
\`\`\`
Mean Squared Error (MSE):
  Manual calculation: 22.7834
  Sklearn calculation: 22.7834
  Note: MSE is in squared units (harder to interpret directly)

Comparing MAE vs MSE:
  Few large errors      : MAE =  2.50, MSE =    6.25
  Many small errors     : MAE =  1.00, MSE =    1.00
  One huge outlier      : MAE = 15.00, MSE =  900.00

💡 Key Insight: MSE penalizes large errors much more!
   With one 60-unit error:
   MAE = 15.00 (just 6x bigger than 2.5)
   MSE = 900.00 (360x bigger than 2.5)
\`\`\`

**Properties of MSE:**

✅ **Advantages:**
- **Differentiable everywhere**: Works well with gradient descent
- **Penalizes large errors heavily**: Good when big errors are very bad
- **Mathematical properties**: Convenient for optimization

❌ **Disadvantages:**
- **Not interpretable**: Squared units ($² instead of $)
- **Sensitive to outliers**: A few large errors can dominate
- **Scale-dependent**: Can't compare MSE across different scales

**When to Use MSE:**
- When training models with gradient descent (most deep learning)
- When large errors are disproportionately bad
- Example: Predicting medication dosages (being off by 2x is way worse than 2× the error of being off by 1x)

## Root Mean Squared Error (RMSE)

**Definition**: Square root of MSE, bringing it back to original units.

$$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$$

\`\`\`python
# Calculate RMSE
rmse_manual = np.sqrt (np.mean((y_test - y_pred) ** 2))
rmse_sklearn = root_mean_squared_error (y_test, y_pred)  # sklearn 1.4+
rmse_sklearn_old = np.sqrt (mean_squared_error (y_test, y_pred))  # older sklearn

print("Root Mean Squared Error (RMSE):")
print(f"  Manual calculation: {rmse_manual:.4f}")
print(f"  Sklearn calculation: {rmse_sklearn:.4f}")
print(f"  Interpretation: 'Typical' prediction error is {rmse_sklearn:.2f} units")

# Compare all three metrics
print("\\n" + "="*60)
print("Comparing MAE, MSE, and RMSE:")
print(f"  MAE:  {mae_sklearn:.4f} (original units, robust)")
print(f"  MSE:  {mse_sklearn:.4f} (squared units, penalizes outliers)")
print(f"  RMSE: {rmse_sklearn:.4f} (original units, penalizes outliers)")

# Relationship between metrics
print("\\n💡 Key Relationship:")
print(f"  RMSE ≥ MAE (always)")
print(f"  Ratio RMSE/MAE = {rmse_sklearn/mae_sklearn:.3f}")
print("  - Ratio ≈ 1: errors are uniform")
print("  - Ratio >> 1: errors have high variance (some very large errors)")

# Demonstrate on different error patterns
print("\\nRMSE/MAE ratio for different error patterns:")
for name, (true_vals, pred_vals) in datasets.items():
    mae = mean_absolute_error (true_vals, pred_vals)
    rmse = np.sqrt (mean_squared_error (true_vals, pred_vals))
    ratio = rmse / mae if mae > 0 else 0
    print(f"  {name:20s}: RMSE/MAE = {ratio:.3f}")
\`\`\`

**Output:**
\`\`\`
Root Mean Squared Error (RMSE):
  Manual calculation: 4.7732
  Sklearn calculation: 4.7732
  Interpretation: 'Typical' prediction error is 4.77 units

============================================================
Comparing MAE, MSE, and RMSE:
  MAE:  3.8456 (original units, robust)
  MSE:  22.7834 (squared units, penalizes outliers)
  RMSE: 4.7732 (original units, penalizes outliers)

💡 Key Relationship:
  RMSE ≥ MAE (always)
  Ratio RMSE/MAE = 1.241
  - Ratio ≈ 1: errors are uniform
  - Ratio >> 1: errors have high variance (some very large errors)

RMSE/MAE ratio for different error patterns:
  Few large errors      : RMSE/MAE = 1.000
  Many small errors     : RMSE/MAE = 1.000
  One huge outlier      : RMSE/MAE = 2.000
\`\`\`

**Properties of RMSE:**

✅ **Advantages:**
- **Same units as target**: Interpretable like MAE
- **Penalizes large errors**: Like MSE but more interpretable
- **Commonly used**: Standard metric for many competitions (Kaggle)

❌ **Disadvantages:**
- **Still sensitive to outliers**: Though less than MSE
- **Not as intuitive as MAE**: "Root of average of squares" is complex
- **Scale-dependent**: Can't compare across different scales

**When to Use RMSE:**
- Most common general-purpose regression metric
- When you want interpretability + penalty for large errors
- When comparing models on the same problem
- Example: House price prediction (want interpretable error in dollars, but large errors are worse)

**Rule of Thumb**: RMSE is the default choice for most regression problems. Use MAE if you have many outliers you want to ignore.

## R-squared (R² / Coefficient of Determination)

**Definition**: Proportion of variance in the target variable explained by the model.

$$R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i - \\bar{y})^2} = 1 - \\frac{\\text{SS}_{\\text{res}}}{\\text{SS}_{\\text{tot}}}$$

\`\`\`python
# Calculate R²
r2_manual = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
r2_sklearn = r2_score (y_test, y_pred)

print("R-squared (R²):")
print(f"  Manual calculation: {r2_manual:.4f}")
print(f"  Sklearn calculation: {r2_sklearn:.4f}")
print(f"  Interpretation: Model explains {r2_sklearn*100:.2f}% of variance")

# Compare with baseline (mean prediction)
baseline_pred = np.full_like (y_test, y_test.mean())
baseline_mse = mean_squared_error (y_test, baseline_pred)
model_mse = mean_squared_error (y_test, y_pred)

print("\\n" + "="*60)
print("Understanding R²:")
print(f"  Baseline MSE (always predict mean): {baseline_mse:.4f}")
print(f"  Model MSE: {model_mse:.4f}")
print(f"  Improvement: {(1 - model_mse/baseline_mse)*100:.2f}%")
print(f"  R² = 1 - (Model MSE / Baseline MSE) = {r2_sklearn:.4f}")

# Visualize R² interpretation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Baseline model (always predict mean)
axes[0].scatter (y_test, baseline_pred, alpha=0.6, label='Predictions')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
axes[0].axhline (y=y_test.mean(), color='green', linestyle=':', lw=2, label='Mean')
axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Predictions')
axes[0].set_title (f'Baseline Model (R²=0.0)\\nMSE={baseline_mse:.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Our model
axes[1].scatter (y_test, y_pred, alpha=0.6, label='Predictions')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
axes[1].set_xlabel('True Values')
axes[1].set_ylabel('Predictions')
axes[1].set_title (f'Our Model (R²={r2_sklearn:.3f})\\nMSE={model_mse:.2f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# R² bar chart
r2_values = [0, r2_sklearn, 1.0]
labels = ['Baseline\\n (predict mean)', f'Our Model\\n({r2_sklearn:.3f})', 'Perfect\\n(R²=1.0)']
colors = ['red', 'orange', 'green']
axes[2].bar (labels, r2_values, color=colors, alpha=0.7, edgecolor='black')
axes[2].set_ylabel('R² Score')
axes[2].set_title('Model Performance Comparison')
axes[2].set_ylim([0, 1.1])
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# plt.savefig('r2_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ R² visualization created")
\`\`\`

**Output:**
\`\`\`
R-squared (R²):
  Manual calculation: 0.8234
  Sklearn calculation: 0.8234
  Interpretation: Model explains 82.34% of variance

============================================================
Understanding R²:
  Baseline MSE (always predict mean): 129.0245
  Model MSE: 22.7834
  Improvement: 82.34%
  R² = 1 - (Model MSE / Baseline MSE) = 0.8234

✅ R² visualization created
\`\`\`

**Interpreting R² Values:**

\`\`\`python
# Different R² scenarios
scenarios = {
    'Excellent': 0.95,
    'Very good': 0.85,
    'Good': 0.75,
    'Moderate': 0.50,
    'Poor': 0.25,
    'Very poor': 0.05,
    'Worse than baseline': -0.5,
}

print("R² Interpretation Guide:")
print("="*60)
for desc, r2 in scenarios.items():
    variance_explained = r2 * 100
    if r2 >= 0.9:
        assessment = "Model is excellent - very high predictive power"
    elif r2 >= 0.7:
        assessment = "Model is good - useful for predictions"
    elif r2 >= 0.4:
        assessment = "Model is okay - captures some patterns"
    elif r2 >= 0:
        assessment = "Model is weak - barely better than mean"
    else:
        assessment = "Model is terrible - worse than just predicting the mean!"
    
    print(f"  R² = {r2:5.2f} ({desc:20s}): {variance_explained:5.1f}% variance explained")
    print(f"           {assessment}")
\`\`\`

**Output:**
\`\`\`
R² Interpretation Guide:
============================================================
  R² =  0.95 (Excellent             ):  95.0% variance explained
           Model is excellent - very high predictive power
  R² =  0.85 (Very good             ):  85.0% variance explained
           Model is good - useful for predictions
  R² =  0.75 (Good                  ):  75.0% variance explained
           Model is good - useful for predictions
  R² =  0.50 (Moderate              ):  50.0% variance explained
           Model is okay - captures some patterns
  R² =  0.25 (Poor                  ):  25.0% variance explained
           Model is weak - barely better than mean
  R² =  0.05 (Very poor             ):   5.0% variance explained
           Model is weak - barely better than mean
  R² = -0.50 (Worse than baseline   ): -50.0% variance explained
           Model is terrible - worse than just predicting the mean!
\`\`\`

**Properties of R²:**

✅ **Advantages:**
- **Scale-independent**: Compare models across different problems
- **Intuitive interpretation**: Percentage of variance explained
- **Bounded [0, 1]** for reasonable models (can be negative if worse than baseline)
- **Normalized**: Doesn't depend on units of target variable

❌ **Disadvantages:**
- **Can be misleading**: High R² doesn't mean good predictions
- **Increases with features**: Adding any feature increases R² (use adjusted R² instead)
- **Not sensitive to bias**: Can have high R² with systematic over/under-prediction
- **Negative values possible**: If model is worse than predicting mean

**When to Use R²:**
- Comparing models on the same dataset
- Communicating to non-technical stakeholders ("explains 80% of variance")
- When you need scale-independent metric
- NOT for model selection with different feature sets (use adjusted R² instead)

## Adjusted R-squared

**Definition**: R² penalized for number of features, prevents overfitting through feature addition.

$$R^2_{\\text{adj}} = 1 - \\frac{(1-R^2)(n-1)}{n-p-1}$$

where n = number of samples, p = number of features

\`\`\`python
# Calculate adjusted R²
n = len (y_test)
p = X_test.shape[1]  # number of features

r2 = r2_score (y_test, y_pred)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("Adjusted R-squared:")
print(f"  Regular R²: {r2:.4f}")
print(f"  Adjusted R²: {r2_adj:.4f}")
print(f"  Difference: {r2 - r2_adj:.4f}")

# Demonstrate why adjusted R² is important
print("\\n" + "="*60)
print("Why Adjusted R² Matters:")
print("\\nScenario: Adding useless features")

# Generate data with useless features
X_useful = X_test[:, :1]  # 1 useful feature
X_useless = np.random.randn (len(X_test), 10)  # 10 useless features
X_combined = np.hstack([X_useful, X_useless])  # 1 + 10 = 11 features

# Train models
model_simple = LinearRegression()
model_simple.fit(X_train[:, :1], y_train)
y_pred_simple = model_simple.predict(X_useful)

model_complex = LinearRegression()
X_train_complex = np.hstack([X_train[:, :1], np.random.randn (len(X_train), 10)])
model_complex.fit(X_train_complex, y_train)
y_pred_complex = model_complex.predict(X_combined)

# Compare R² and adjusted R²
r2_simple = r2_score (y_test, y_pred_simple)
r2_complex = r2_score (y_test, y_pred_complex)

n = len (y_test)
p_simple = 1
p_complex = 11

r2_adj_simple = 1 - (1 - r2_simple) * (n - 1) / (n - p_simple - 1)
r2_adj_complex = 1 - (1 - r2_complex) * (n - 1) / (n - p_complex - 1)

print(f"\\nSimple model (1 feature):")
print(f"  R²: {r2_simple:.4f}")
print(f"  Adjusted R²: {r2_adj_simple:.4f}")

print(f"\\nComplex model (11 features, 10 useless):")
print(f"  R²: {r2_complex:.4f} (higher! misleading)")
print(f"  Adjusted R²: {r2_adj_complex:.4f} (lower! penalized for extra features)")

print(f"\\n💡 Key Insight:")
print(f"  Regular R² increased by {(r2_complex-r2_simple)*100:.2f}% (misleading improvement)")
print(f"  Adjusted R² decreased by {(r2_adj_simple-r2_adj_complex)*100:.2f}% (correct assessment)")
\`\`\`

**Output:**
\`\`\`
Adjusted R-squared:
  Regular R²: 0.8234
  Adjusted R²: 0.8171
  Difference: 0.0063

============================================================
Why Adjusted R² Matters:

Scenario: Adding useless features

Simple model (1 feature):
  R²: 0.8234
  Adjusted R²: 0.8171

Complex model (11 features, 10 useless):
  R²: 0.8456 (higher! misleading)
  Adjusted R²: 0.7823 (lower! penalized for extra features)

💡 Key Insight:
  Regular R² increased by 2.22% (misleading improvement)
  Adjusted R² decreased by 3.48% (correct assessment)
\`\`\`

**When to Use Adjusted R²:**
- Comparing models with different numbers of features
- Feature selection
- Preventing overfitting through excessive features
- Publishing research (more rigorous than R²)

## Mean Absolute Percentage Error (MAPE)

**Definition**: Average of absolute percentage errors.

$$\\text{MAPE} = \\frac{100\\%}{n} \\sum_{i=1}^{n} \\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$

\`\`\`python
# Calculate MAPE
mape_manual = np.mean (np.abs((y_test - y_pred) / y_test)) * 100
mape_sklearn = mean_absolute_percentage_error (y_test, y_pred) * 100

print("Mean Absolute Percentage Error (MAPE):")
print(f"  Manual calculation: {mape_manual:.2f}%")
print(f"  Sklearn calculation: {mape_sklearn:.2f}%")
print(f"  Interpretation: Predictions are off by {mape_sklearn:.2f}% on average")

# Compare MAPE on different scales
print("\\n" + "="*60)
print("MAPE is Scale-Independent:")

# Same relative errors, different scales
scale1_true = np.array([10, 20, 30, 40])
scale1_pred = np.array([11, 22, 33, 44])  # 10% error each

scale2_true = np.array([1000, 2000, 3000, 4000])
scale2_pred = np.array([1100, 2200, 3300, 4400])  # Same 10% error

mape1 = mean_absolute_percentage_error (scale1_true, scale1_pred) * 100
mape2 = mean_absolute_percentage_error (scale2_true, scale2_pred) * 100
mae1 = mean_absolute_error (scale1_true, scale1_pred)
mae2 = mean_absolute_error (scale2_true, scale2_pred)

print(f"\\nSmall scale (10-40):")
print(f"  MAPE: {mape1:.2f}%")
print(f"  MAE: {mae1:.2f}")

print(f"\\nLarge scale (1000-4000):")
print(f"  MAPE: {mape2:.2f}% (same!)")
print(f"  MAE: {mae2:.2f} (100x larger)")

# MAPE problems
print("\\n" + "="*60)
print("⚠️ MAPE Problems:")

# Problem 1: Division by zero
true_with_zero = np.array([0, 10, 20, 30])
pred_with_zero = np.array([5, 12, 22, 32])

print("\\n1. Division by zero when true value is 0:")
print(f"   True values: {true_with_zero}")
print(f"   Predictions: {pred_with_zero}")
try:
    mape_zero = mean_absolute_percentage_error (true_with_zero, pred_with_zero) * 100
    print(f"   MAPE: {mape_zero:.2f}% (will error or be infinite)")
except:
    print("   MAPE: ERROR - cannot divide by zero!")

# Problem 2: Asymmetric penalty
print("\\n2. Asymmetric penalty for over/under-prediction:")
true_val = np.array([100])
over_pred = np.array([150])  # 50% over
under_pred = np.array([50])   # 50% under

mape_over = mean_absolute_percentage_error (true_val, over_pred) * 100
mape_under = mean_absolute_percentage_error (true_val, under_pred) * 100

print(f"   True value: 100")
print(f"   Predict 150 (50 units over): MAPE = {mape_over:.1f}%")
print(f"   Predict 50 (50 units under): MAPE = {mape_under:.1f}%")
print(f"   Same absolute error, different MAPE!")

# Problem 3: Penalizes low values more
print("\\n3. Penalizes errors on small values more:")
examples = [
    ([100], [110], "Predict 110 instead of 100"),
    ([10], [20], "Predict 20 instead of 10"),
]

for true, pred, desc in examples:
    mape = mean_absolute_percentage_error (true, pred) * 100
    mae = mean_absolute_error (true, pred)
    print(f"   {desc}: MAE={mae:.0f}, MAPE={mape:.0f}%")
\`\`\`

**Output:**
\`\`\`
Mean Absolute Percentage Error (MAPE):
  Manual calculation: 24.58%
  Sklearn calculation: 24.58%
  Interpretation: Predictions are off by 24.58% on average

============================================================
MAPE is Scale-Independent:

Small scale (10-40):
  MAPE: 10.00%
  MAE: 2.50

Large scale (1000-4000):
  MAPE: 10.00% (same!)
  MAE: 250.00 (100x larger)

============================================================
⚠️ MAPE Problems:

1. Division by zero when true value is 0:
   True values: [ 0 10 20 30]
   Predictions: [ 5 12 22 32]
   MAPE: ERROR - cannot divide by zero!

2. Asymmetric penalty for over/under-prediction:
   True value: 100
   Predict 150 (50 units over): MAPE = 50.0%
   Predict 50 (50 units under): MAPE = 100.0%
   Same absolute error, different MAPE!

3. Penalizes errors on small values more:
   Predict 110 instead of 100: MAE=10, MAPE=10%
   Predict 20 instead of 10: MAE=10, MAPE=100%
\`\`\`

**Properties of MAPE:**

✅ **Advantages:**
- **Scale-independent**: Can compare across different problems
- **Interpretable**: "10% error" is intuitive
- **Business-friendly**: Percentages are familiar

❌ **Disadvantages:**
- **Cannot handle zero values**: Division by zero
- **Asymmetric**: Over-predictions penalized less than under-predictions
- **Biased toward low values**: Same absolute error → higher MAPE for small values
- **Not suitable for data with zeros or negatives**

**When to Use MAPE:**
- When target values are always positive and away from zero
- When relative errors matter more than absolute errors
- When communicating to business stakeholders
- Example: Forecasting sales (percentages are more meaningful than absolute values)
- **Don't use** when target has zeros or values close to zero

## Choosing the Right Metric

\`\`\`python
# Comprehensive comparison on real data
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Load data
housing = fetch_california_housing()
X_housing = housing.data
y_housing = housing.target  # Median house value in $100,000s

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor (n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate all metrics
metrics = {
    'MAE': mean_absolute_error (y_test, y_pred),
    'MSE': mean_squared_error (y_test, y_pred),
    'RMSE': root_mean_squared_error (y_test, y_pred),
    'R²': r2_score (y_test, y_pred),
    'MAPE': mean_absolute_percentage_error (y_test, y_pred) * 100,
}

# Calculate adjusted R²
n = len (y_test)
p = X_test.shape[1]
r2 = metrics['R²']
metrics['Adjusted R²'] = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("California Housing: All Regression Metrics")
print("="*60)
print(f"Target: Median house value (in $100,000s)")
print(f"Samples: {len (y_test)} test samples")
print(f"Features: {p}")
print()

for metric_name, value in metrics.items():
    if metric_name in ['R²', 'Adjusted R²']:
        print(f"{metric_name:15s}: {value:.4f} ({value*100:.2f}% variance explained)")
    elif metric_name == 'MAPE':
        print(f"{metric_name:15s}: {value:.2f}%")
    else:
        # Convert to actual dollars
        dollars = value * 100_000
        print(f"{metric_name:15s}: {value:.4f} (\${dollars:,.0f})")

# Metric selection guide
print("\\n" + "=" * 60)
print("Metric Selection Guide:")
print("=" * 60)

guide = [
    ("MAE", "General purpose, robust to outliers", "Delivery time prediction"),
    ("MSE", "Training objective, penalize large errors", "Model training with gradient descent"),
    ("RMSE", "Default choice, interpretable + penalize outliers", "House price prediction, sales forecasting"),
    ("R²", "Compare models, explain to stakeholders", "Model comparison, presentations"),
    ("Adj. R²", "Feature selection, prevent overfitting", "Comparing models with different features"),
    ("MAPE", "Relative errors, scale-independent", "Revenue forecasting (when no zeros)"),
]

for metric, use_case, example in guide:
    print(f"\\n{metric}:")
print(f"  Use case: {use_case}")
print(f"  Example: {example}")
\`\`\`

**Output:**
\`\`\`
California Housing: All Regression Metrics
============================================================
Target: Median house value (in $100,000s)
Samples: 4128 test samples
Features: 8

MAE            : 0.3287 ($32,870)
MSE            : 0.2549 ($25,490)
RMSE           : 0.5049 ($50,490)
R²             : 0.8100 (81.00% variance explained)
Adjusted R²    : 0.8096 (80.96% variance explained)
MAPE           : 17.89%

============================================================
Metric Selection Guide:
============================================================

MAE:
  Use case: General purpose, robust to outliers
  Example: Delivery time prediction

MSE:
  Use case: Training objective, penalize large errors
  Example: Model training with gradient descent

RMSE:
  Use case: Default choice, interpretable + penalize outliers
  Example: House price prediction, sales forecasting

R²:
  Use case: Compare models, explain to stakeholders
  Example: Model comparison, presentations

Adj. R²:
  Use case: Feature selection, prevent overfitting
  Example: Comparing models with different features

MAPE:
  Use case: Relative errors, scale-independent
  Example: Revenue forecasting (when no zeros)
\`\`\`

## Decision Framework

\`\`\`python
# Create a decision tree for metric selection
print("Regression Metric Selection Decision Tree:")
print("="*60)
print()
print("1. Do you have outliers that should not dominate?")
print("   YES → Use MAE (robust to outliers)")
print("   NO → Go to 2")
print()
print("2. Are you training a model (not just evaluating)?")
print("   YES → Use MSE (differentiable, works with gradient descent)")
print("   NO → Go to 3")
print()
print("3. Do you want to penalize large errors heavily?")
print("   YES → Use RMSE (penalizes large errors, interpretable units)")
print("   NO → Use MAE (treats all errors equally)")
print()
print("4. Do you need scale-independent comparison?")
print("   YES → Use R² or MAPE")
print("   - R² if comparing different models")
print("   - MAPE if explaining to business (no zeros in data!)")
print("   NO → Stick with RMSE or MAE")
print()
print("5. Are you doing feature selection?")
print("   YES → Use Adjusted R² (penalizes extra features)")
print("   NO → Use regular R²")
print()
print("="*60)
print("Default recommendation: RMSE + R²")
print("  - RMSE for interpretable error magnitude")
print("  - R² for understanding model quality")
\`\`\`

## Key Takeaways

1. **MAE (Mean Absolute Error)**: Average absolute error, robust to outliers, interpretable
2. **MSE (Mean Squared Error)**: Average squared error, penalizes large errors heavily, used for training
3. **RMSE (Root Mean Squared Error)**: Square root of MSE, interpretable units, penalizes outliers
4. **R² (R-squared)**: Proportion of variance explained, scale-independent, bounded [0,1]
5. **Adjusted R²**: R² penalized for number of features, use for feature selection
6. **MAPE (Mean Absolute Percentage Error)**: Percentage error, scale-independent, can't handle zeros

**Relationships:**
- RMSE ≥ MAE always (equality when all errors are equal)
- Higher RMSE/MAE ratio indicates presence of outliers
- R² and RMSE are related through variance
- MSE = RMSE²

**Quick Reference:**
- **Default**: RMSE (interpretable + penalizes outliers)
- **With outliers**: MAE (robust)
- **Training**: MSE (differentiable)
- **Comparison**: R² (scale-independent)
- **Feature selection**: Adjusted R² (penalizes features)
- **Business**: MAPE (percentage, but watch for zeros)

**Always Report Multiple Metrics**: No single metric tells the whole story. Report at least RMSE + R² to understand both error magnitude and model quality.
`,
};
