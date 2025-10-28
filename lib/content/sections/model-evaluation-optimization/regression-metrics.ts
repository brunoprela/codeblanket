export const regressionMetrics = {
  title: 'Regression Metrics',
  content: `
# Regression Metrics

## Introduction

Evaluating regression models requires different metrics than classification. While classification deals with discrete categories, regression predicts continuous values, requiring metrics that measure the **magnitude and direction of errors**.

**Key Question**: How do we quantify how close our predictions are to the true values?

The choice of metric can dramatically affect model development, optimization objectives, and business decisions. Different metrics penalize errors differently, and understanding these differences is crucial for building effective models.

## Mean Absolute Error (MAE)

### Definition

MAE is the average of absolute differences between predictions and actual values:

$$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$

where:
- $y_i$ is the actual value
- $\\hat{y}_i$ is the predicted value  
- $n$ is the number of samples

### Properties

**Advantages:**
- Easy to understand (average error in original units)
- Robust to outliers
- All errors weighted equally

**Disadvantages:**
- Not differentiable at zero (optimization challenges)
- Doesn't heavily penalize large errors

### Implementation and Interpretation

\`\`\`python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error (y_test, y_pred)

print("Mean Absolute Error (MAE)")
print("="*70)
print(f"MAE: {mae:.2f}")
print(f"\\nInterpretation: On average, predictions are off by {mae:.2f} units")
print(f"Target range: [{y_test.min():.1f}, {y_test.max():.1f}]")
print(f"Relative error: {mae / (y_test.max() - y_test.min()) * 100:.1f}% of range")

# Manual calculation
mae_manual = np.mean (np.abs (y_test - y_pred))
print(f"\\nManual calculation: {mae_manual:.2f}")
print(f"✓ Matches sklearn: {np.isclose (mae, mae_manual)}")

# Error distribution
errors = np.abs (y_test - y_pred)
print(f"\\nError Distribution:")
print(f"  Min error:    {errors.min():.2f}")
print(f"  25th percentile: {np.percentile (errors, 25):.2f}")
print(f"  Median error: {np.median (errors):.2f}")
print(f"  75th percentile: {np.percentile (errors, 75):.2f}")
print(f"  Max error:    {errors.max():.2f}")
\`\`\`

## Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

### Definitions

**MSE**: Average of squared differences

$$\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$

**RMSE**: Square root of MSE (returns to original units)

$$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$$

### Properties

**MSE Advantages:**
- Differentiable (good for optimization)
- Heavily penalizes large errors
- Mathematical properties (unbiased estimator)

**MSE Disadvantages:**
- Not in original units (hard to interpret)
- Sensitive to outliers
- Larger numbers can be misleading

**RMSE Advantages:**
- Same units as target variable
- Penalizes large errors
- Most commonly used

\`\`\`python
# Calculate MSE and RMSE
mse = mean_squared_error (y_test, y_pred)
rmse = np.sqrt (mse)

print("\\n" + "="*70)
print("Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)")
print("="*70)
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# Manual calculations
mse_manual = np.mean((y_test - y_pred)**2)
rmse_manual = np.sqrt (mse_manual)

print(f"\\nManual calculations:")
print(f"MSE:  {mse_manual:.2f} ✓")
print(f"RMSE: {rmse_manual:.2f} ✓")

# Comparison with MAE
print(f"\\nMAE vs RMSE:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"RMSE/MAE ratio: {rmse/mae:.2f}")

if rmse / mae > 1.5:
    print("⚠️  Large RMSE/MAE ratio suggests presence of outliers")
else:
    print("✓ RMSE/MAE ratio suggests errors are relatively uniform")
\`\`\`

### MAE vs MSE: Sensitivity to Outliers

\`\`\`python
print("\\n" + "="*70)
print("MAE vs MSE: Outlier Sensitivity Demo")
print("="*70)

# Create predictions with and without outliers
np.random.seed(42)
n_samples = 100

y_true = np.linspace(0, 100, n_samples)
y_pred_good = y_true + np.random.randn (n_samples) * 5

# Version with outliers
y_pred_outliers = y_pred_good.copy()
outlier_indices = [10, 25, 50, 75, 90]
y_pred_outliers[outlier_indices] += np.random.choice([-50, 50], len (outlier_indices))

# Calculate metrics
mae_good = mean_absolute_error (y_true, y_pred_good)
rmse_good = np.sqrt (mean_squared_error (y_true, y_pred_good))

mae_outliers = mean_absolute_error (y_true, y_pred_outliers)
rmse_outliers = np.sqrt (mean_squared_error (y_true, y_pred_outliers))

print("\\nWithout outliers:")
print(f"  MAE:  {mae_good:.2f}")
print(f"  RMSE: {rmse_good:.2f}")

print("\\nWith 5 large outliers:")
print(f"  MAE:  {mae_outliers:.2f} (increase: {(mae_outliers/mae_good - 1)*100:.1f}%)")
print(f"  RMSE: {rmse_outliers:.2f} (increase: {(rmse_outliers/rmse_good - 1)*100:.1f}%)")

print("\\nObservation:")
print(f"  RMSE increased {(rmse_outliers/rmse_good - 1)*100:.1f}% vs MAE {(mae_outliers/mae_good - 1)*100:.1f}%")
print("  RMSE is MORE sensitive to outliers due to squaring")
\`\`\`

## R-squared (Coefficient of Determination)

### Definition

R² measures the proportion of variance in the target variable explained by the model:

$$R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i - \\bar{y})^2} = 1 - \\frac{SS_{res}}{SS_{tot}}$$

where:
- $SS_{res}$ = Residual sum of squares
- $SS_{tot}$ = Total sum of squares
- $\\bar{y}$ = mean of observed values

### Properties

**Range:**
- Best possible score: 1.0
- Baseline (predicting mean): 0.0
- Worse than baseline: negative

**Interpretation:**
- R² = 0.8 means model explains 80% of variance
- R² = 0.0 means model is no better than predicting the mean
- R² < 0 means model is worse than predicting the mean

\`\`\`python
print("\\n" + "="*70)
print("R-squared (R²)")
print("="*70)

r2 = r2_score (y_test, y_pred)
print(f"R²: {r2:.4f}")

# Manual calculation
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean (y_test))**2)
r2_manual = 1 - (ss_res / ss_tot)

print(f"\\nManual calculation:")
print(f"SS_res (residual): {ss_res:.2f}")
print(f"SS_tot (total):    {ss_tot:.2f}")
print(f"R² = 1 - ({ss_res:.2f} / {ss_tot:.2f}) = {r2_manual:.4f} ✓")

# Interpretation
print(f"\\nInterpretation:")
print(f"  Model explains {r2*100:.1f}% of variance in target")
print(f"  {(1-r2)*100:.1f}% of variance remains unexplained")

# Baseline comparison
baseline_pred = np.full_like (y_test, np.mean (y_train))
baseline_mse = mean_squared_error (y_test, baseline_pred)
model_mse = mean_squared_error (y_test, y_pred)

print(f"\\nComparison to baseline (predicting mean):")
print(f"  Baseline MSE: {baseline_mse:.2f}")
print(f"  Model MSE:    {model_mse:.2f}")
print(f"  Improvement:  {(1 - model_mse/baseline_mse)*100:.1f}%")

# R² and MSE relationship
r2_from_mse = 1 - (model_mse / baseline_mse)
print(f"\\nR² from MSE ratio: {r2_from_mse:.4f} ✓")
\`\`\`

### R² Limitations and Adjusted R²

\`\`\`python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

print("\\n" + "="*70)
print("R² Limitations and Adjusted R²")
print("="*70)

def adjusted_r2(r2, n_samples, n_features):
    """Calculate adjusted R²."""
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# Compare models with different numbers of features
results = []

for degree in [1, 2, 3, 5, 10]:
    poly = PolynomialFeatures (degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model_poly = Ridge (alpha=1.0)
    model_poly.fit(X_train_poly, y_train)
    y_pred_poly = model_poly.predict(X_test_poly)
    
    r2 = r2_score (y_test, y_pred_poly)
    adj_r2 = adjusted_r2(r2, len (y_test), X_test_poly.shape[1])
    
    results.append({
        'degree': degree,
        'n_features': X_test_poly.shape[1],
        'r2': r2,
        'adj_r2': adj_r2
    })

print(f"{'Degree':<8s} {'Features':<10s} {'R²':<10s} {'Adj R²':<10s} {'Difference':<12s}")
print("-"*70)
for r in results:
    diff = r['r2'] - r['adj_r2']
    print(f"{r['degree']:<8d} {r['n_features']:<10d} {r['r2']:<10.4f} "
          f"{r['adj_r2']:<10.4f} {diff:<12.4f}")

print("\\nObservation:")
print("  Regular R² always increases with more features")
print("  Adjusted R² penalizes model complexity")
print("  Use adjusted R² when comparing models with different feature counts")
\`\`\`

## Mean Absolute Percentage Error (MAPE)

### Definition

MAPE expresses error as a percentage of actual values:

$$\\text{MAPE} = \\frac{100\\%}{n} \\sum_{i=1}^{n} \\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$

### Properties

**Advantages:**
- Easy to interpret (percentage)
- Scale-independent
- Business-friendly metric

**Disadvantages:**
- Undefined when $y_i = 0$
- Asymmetric (penalizes over-predictions more than under-predictions)
- Sensitive to small denominators

\`\`\`python
print("\\n" + "="*70)
print("Mean Absolute Percentage Error (MAPE)")
print("="*70)

# MAPE calculation (avoid division by zero)
def safe_mape (y_true, y_pred, epsilon=1e-10):
    """Calculate MAPE with protection against division by zero."""
    mask = np.abs (y_true) > epsilon
    return np.mean (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape = safe_mape (y_test, y_pred)
mape_sklearn = mean_absolute_percentage_error (y_test, y_pred) * 100

print(f"MAPE: {mape:.2f}%")
print(f"MAPE (sklearn): {mape_sklearn:.2f}%")

print(f"\\nInterpretation:")
print(f"  On average, predictions are off by {mape:.1f}% of actual value")

# Show some examples
print(f"\\nSample Predictions:")
print(f"{'Actual':<10s} {'Predicted':<10s} {'Error':<10s} {'% Error':<10s}")
print("-"*50)
for i in range (min(10, len (y_test))):
    actual = y_test[i]
    predicted = y_pred[i]
    error = actual - predicted
    pct_error = abs (error / actual * 100)
    print(f"{actual:<10.1f} {predicted:<10.1f} {error:<10.1f} {pct_error:<10.1f}%")

# MAPE asymmetry demonstration
print("\\n" + "="*70)
print("MAPE Asymmetry:")
print("="*70)

true_val = 100
over_pred = 150  # 50% over-prediction
under_pred = 50  # 50% under-prediction

over_error = abs((true_val - over_pred) / true_val) * 100
under_error = abs((true_val - under_pred) / true_val) * 100

print(f"True value: {true_val}")
print(f"Over-prediction by 50: MAPE = {over_error:.1f}%")
print(f"Under-prediction by 50: MAPE = {under_error:.1f}%")
print("\\n⚠️  MAPE treats over/under predictions differently!")
\`\`\`

## Symmetric MAPE and Other Percentage Metrics

\`\`\`python
def smape (y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs (y_true - y_pred)
    denominator = (np.abs (y_true) + np.abs (y_pred)) / 2
    return np.mean (numerator / denominator) * 100

def mpe (y_true, y_pred):
    """Mean Percentage Error (shows bias)."""
    return np.mean((y_true - y_pred) / y_true) * 100

print("\\n" + "="*70)
print("Alternative Percentage Metrics")
print("="*70)

smape_score = smape (y_test, y_pred)
mpe_score = mpe (y_test, y_pred)

print(f"MAPE:  {mape:.2f}%")
print(f"SMAPE: {smape_score:.2f}%")
print(f"MPE:   {mpe_score:.2f}%")

print(f"\\nMPE Interpretation:")
if abs (mpe_score) < 1:
    print("  ✓ Model is unbiased (no systematic over/under-prediction)")
elif mpe_score > 0:
    print(f"  ⚠️  Model under-predicts by {mpe_score:.1f}% on average")
else:
    print(f"  ⚠️  Model over-predicts by {abs (mpe_score):.1f}% on average")
\`\`\`

## Median Absolute Error (MedAE)

### Definition

MedAE is the median of absolute errors:

$$\\text{MedAE} = \\text{median}(|y_1 - \\hat{y}_1|, ..., |y_n - \\hat{y}_n|)$$

### Properties

**Advantages:**
- Very robust to outliers
- Better than MAE for skewed error distributions

**Disadvantages:**
- Less commonly used
- Not differentiable

\`\`\`python
from sklearn.metrics import median_absolute_error

print("\\n" + "="*70)
print("Median Absolute Error (MedAE)")
print("="*70)

medae = median_absolute_error (y_test, y_pred)

print(f"MAE:    {mae:.2f}")
print(f"MedAE:  {medae:.2f}")
print(f"Ratio:  {mae/medae:.2f}")

if mae / medae > 1.3:
    print("\\n⚠️  Large MAE/MedAE ratio suggests outliers are present")
    print("   MedAE may be more representative of typical error")
else:
    print("\\n✓ Similar MAE and MedAE suggest symmetric, outlier-free errors")

# Visualize error distribution
errors = np.abs (y_test - y_pred)
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist (errors, bins=30, edgecolor='black', alpha=0.7)
plt.axvline (mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.2f}')
plt.axvline (medae, color='blue', linestyle='--', linewidth=2, label=f'MedAE = {medae:.2f}')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.grid (alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter (y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Predictions vs Actual')
plt.legend()
plt.grid (alpha=0.3)

plt.tight_layout()
plt.savefig('regression_metrics_visualization.png', dpi=150, bbox_inches='tight')
print("\\nVisualization saved to 'regression_metrics_visualization.png'")
\`\`\`

## Comparing Multiple Models

\`\`\`python
print("\\n" + "="*70)
print("Comprehensive Model Comparison")
print("="*70)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)': Ridge (alpha=1.0),
    'Ridge (α=10)': Ridge (alpha=10.0),
    'Lasso (α=1)': Lasso (alpha=1.0, max_iter=10000),
    'Random Forest': RandomForestRegressor (n_estimators=100, random_state=42, max_depth=5)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate all metrics
    metrics = {
        'Model': name,
        'Train MAE': mean_absolute_error (y_train, y_pred_train),
        'Test MAE': mean_absolute_error (y_test, y_pred_test),
        'Train RMSE': np.sqrt (mean_squared_error (y_train, y_pred_train)),
        'Test RMSE': np.sqrt (mean_squared_error (y_test, y_pred_test)),
        'Train R²': r2_score (y_train, y_pred_train),
        'Test R²': r2_score (y_test, y_pred_test),
        'Test MAPE': safe_mape (y_test, y_pred_test),
        'Test MedAE': median_absolute_error (y_test, y_pred_test)
    }
    
    results.append (metrics)

# Display results
import pandas as pd
df_results = pd.DataFrame (results)

print("\\nAll Metrics:")
print(df_results.to_string (index=False))

# Find best model for each metric
print("\\n" + "="*70)
print("Best Models by Metric:")
print("="*70)

for metric in ['Test MAE', 'Test RMSE', 'Test R²', 'Test MAPE']:
    if metric == 'Test R²':
        best_idx = df_results[metric].idxmax()
    else:
        best_idx = df_results[metric].idxmin()
    
    best_model = df_results.loc[best_idx, 'Model']
    best_value = df_results.loc[best_idx, metric]
    
    print(f"{metric:<12s}: {best_model:<20s} ({best_value:.4f})")

# Check for overfitting
print("\\n" + "="*70)
print("Overfitting Analysis:")
print("="*70)

for _, row in df_results.iterrows():
    train_rmse = row['Train RMSE']
    test_rmse = row['Test RMSE']
    gap = test_rmse - train_rmse
    gap_pct = (gap / train_rmse) * 100
    
    status = "✓ Good" if gap_pct < 20 else "⚠️  Overfitting" if gap_pct < 50 else "❌ Severe Overfitting"
    
    print(f"{row['Model']:<20s}: Train {train_rmse:6.2f} → Test {test_rmse:6.2f} "
          f"(+{gap_pct:5.1f}%) {status}")
\`\`\`

## Residual Analysis

\`\`\`python
print("\\n" + "="*70)
print("Residual Analysis")
print("="*70)

# Calculate residuals
residuals = y_test - y_pred

# Statistics
print("Residual Statistics:")
print(f"  Mean:     {np.mean (residuals):7.2f} (should be ≈ 0)")
print(f"  Std Dev:  {np.std (residuals):7.2f}")
print(f"  Min:      {np.min (residuals):7.2f}")
print(f"  Max:      {np.max (residuals):7.2f}")
print(f"  Skewness: {pd.Series (residuals).skew():7.2f} (should be ≈ 0)")
print(f"  Kurtosis: {pd.Series (residuals).kurtosis():7.2f} (should be ≈ 0)")

# Tests
from scipy import stats

# Normality test
_, p_value_normality = stats.normaltest (residuals)
print(f"\\nNormality test p-value: {p_value_normality:.4f}")
if p_value_normality > 0.05:
    print("  ✓ Residuals appear normally distributed")
else:
    print("  ⚠️  Residuals may not be normally distributed")

# Check for patterns
residuals_sorted_indices = np.argsort (y_test)
residuals_sorted = residuals[residuals_sorted_indices]

# Simple trend check
from scipy.stats import spearmanr
corr, p_value_trend = spearmanr (y_test, np.abs (residuals))

print(f"\\nCorrelation between |residuals| and actual values: {corr:.4f} (p={p_value_trend:.4f})")
if abs (corr) > 0.3 and p_value_trend < 0.05:
    print("  ⚠️  Heteroscedasticity detected (error variance depends on y)")
else:
    print("  ✓ Homoscedasticity (constant error variance)")
\`\`\`

## Choosing the Right Metric

\`\`\`python
def recommend_metric (characteristics):
    """
    Recommend appropriate regression metric based on problem characteristics.
    
    Parameters:
    -----------
    characteristics : dict
        Problem characteristics with keys:
        - 'has_outliers': bool
        - 'needs_interpretability': bool  
        - 'scale_matters': bool
        - 'penalize_large_errors': bool
        - 'percentage_based': bool
    
    Returns:
    --------
    recommendation : dict
    """
    recommendations = []
    
    if characteristics.get('percentage_based', False):
        recommendations.append({
            'metric': 'MAPE',
            'reason': 'Percentage-based interpretation needed',
            'caveat': 'Avoid if target can be zero or near-zero'
        })
    
    if characteristics.get('has_outliers', False):
        recommendations.append({
            'metric': 'MAE or MedAE',
            'reason': 'Robust to outliers',
            'caveat': 'MedAE even more robust but less common'
        })
    else:
        recommendations.append({
            'metric': 'RMSE',
            'reason': 'No outliers, RMSE is standard choice',
            'caveat': 'More sensitive to large errors'
        })
    
    if characteristics.get('needs_interpretability', False):
        recommendations.append({
            'metric': 'MAE',
            'reason': 'Most interpretable (average error in original units)',
            'caveat': 'Easy to explain to stakeholders'
        })
    
    if characteristics.get('penalize_large_errors', True):
        recommendations.append({
            'metric': 'RMSE or MSE',
            'reason': 'Quadratic penalty for large errors',
            'caveat': 'MSE not in original units'
        })
    
    if not characteristics.get('scale_matters', True):
        recommendations.append({
            'metric': 'R²',
            'reason': 'Scale-independent, proportion of variance explained',
            'caveat': 'Use adjusted R² when comparing models with different features'
        })
    
    return recommendations

# Example recommendations
scenarios = [
    {
        'name': 'Stock price prediction',
        'characteristics': {
            'has_outliers': True,
            'needs_interpretability': True,
            'scale_matters': True,
            'penalize_large_errors': True,
            'percentage_based': False
        }
    },
    {
        'name': 'House price prediction',
        'characteristics': {
            'has_outliers': False,
            'needs_interpretability': True,
            'scale_matters': True,
            'penalize_large_errors': True,
            'percentage_based': False
        }
    },
    {
        'name': 'Sales forecasting',
        'characteristics': {
            'has_outliers': False,
            'needs_interpretability': True,
            'scale_matters': False,
            'penalize_large_errors': False,
            'percentage_based': True
        }
    }
]

print("\\n" + "="*70)
print("Metric Recommendations by Scenario:")
print("="*70)

for scenario in scenarios:
    print(f"\\n{scenario['name']}:")
    recommendations = recommend_metric (scenario['characteristics'])
    for i, rec in enumerate (recommendations[:2], 1):  # Top 2
        print(f"  {i}. {rec['metric']}: {rec['reason']}")
        print(f"     Note: {rec['caveat']}")
\`\`\`

## Trading Application: Evaluating Price Predictions

\`\`\`python
print("\\n" + "="*70)
print("Trading Application: Price Prediction Evaluation")
print("="*70)

# Simulate stock price predictions
np.random.seed(42)
n_days = 100

true_prices = 100 + np.cumsum (np.random.randn (n_days) * 2)
predicted_prices = true_prices + np.random.randn (n_days) * 5

# Standard metrics
mae_price = mean_absolute_error (true_prices, predicted_prices)
rmse_price = np.sqrt (mean_squared_error (true_prices, predicted_prices))
mape_price = safe_mape (true_prices, predicted_prices)

print("Standard Metrics:")
print(f"  MAE:  \\$\{mae_price:.2f}")
print(f"  RMSE: \\$\{rmse_price:.2f}")
print(f"  MAPE: {mape_price:.2f}%")

# Trading - specific metrics
def directional_accuracy (y_true, y_pred):
"""Percentage of times direction is predicted correctly."""
true_direction = np.diff (y_true) > 0
pred_direction = np.diff (y_pred) > 0
return np.mean (true_direction == pred_direction) * 100

def trading_profit (y_true, y_pred, transaction_cost = 0.001):
"""Simulate trading profit based on predictions."""
positions = np.zeros (len (y_pred))
positions[1:] = np.where (np.diff (y_pred) > 0, 1, -1)  # Long if predicting up, short if down
    
    returns = np.diff (y_true) / y_true[: -1]
position_returns = positions[1:] * returns
    
    # Subtract transaction costs
position_changes = np.abs (np.diff (positions))
n_trades = np.sum (position_changes > 0)

gross_return = np.sum (position_returns)
net_return = gross_return - (n_trades * transaction_cost)

return {
    'gross_return': gross_return * 100,
    'net_return': net_return * 100,
    'n_trades': n_trades
}

dir_acc = directional_accuracy (true_prices, predicted_prices)
profit_metrics = trading_profit (true_prices, predicted_prices)

print("\\nTrading-Specific Metrics:")
print(f"  Directional Accuracy: {dir_acc:.1f}%")
print(f"  Gross Return: {profit_metrics['gross_return']:+.2f}%")
print(f"  Net Return: {profit_metrics['net_return']:+.2f}%")
print(f"  Number of Trades: {profit_metrics['n_trades']}")

print("\\n💡 Insight: For trading, directional accuracy often matters more than")
print("   absolute price accuracy. A model with higher MAE but better")
print("   directional accuracy can be more profitable!")
\`\`\`

## Key Takeaways

1. **MAE**: Robust, interpretable, equal weight to all errors
2. **RMSE**: Penalizes large errors, most common, same units as target
3. **R²**: Scale-independent, shows variance explained (0 to 1)
4. **MAPE**: Percentage-based, business-friendly, but asymmetric
5. **MedAE**: Very robust to outliers

**Recommendations:**
- Use RMSE as default for most regression problems
- Use MAE when you have outliers or want interpretability
- Use R² for model comparison (adjusted R² for different feature counts)
- Use MAPE for business stakeholders (when appropriate)
- Always look at multiple metrics together

**For Trading:**
- Directional accuracy often matters more than absolute error
- Consider domain-specific metrics (profit, Sharpe ratio, etc.)
- MAPE useful for relative price movements

## Further Reading

- Willmott, C. J., & Matsuura, K. (2005). "Advantages of the mean absolute error (MAE)"
- Chai, T., & Draxler, R. R. (2014). "Root mean square error (RMSE) or mean absolute error (MAE)?"
- Hyndman, R. J., & Koehler, A. B. (2006). "Another look at measures of forecast accuracy"
`,
  exercises: [
    {
      prompt:
        'Create a comprehensive regression evaluation framework that calculates all major metrics, performs residual analysis, and provides a detailed report with visualizations. Test it on multiple models and datasets.',
      solution: `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from scipy import stats

class RegressionEvaluator:
    """Comprehensive regression model evaluation framework."""
    
    def __init__(self, y_true, y_pred, model_name="Model"):
        self.y_true = np.array (y_true)
        self.y_pred = np.array (y_pred)
        self.model_name = model_name
        self.residuals = y_true - y_pred
        
    def calculate_all_metrics (self):
        """Calculate all common regression metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['MAE'] = mean_absolute_error (self.y_true, self.y_pred)
        metrics['MSE'] = mean_squared_error (self.y_true, self.y_pred)
        metrics['RMSE'] = np.sqrt (metrics['MSE'])
        metrics['R²'] = r2_score (self.y_true, self.y_pred)
        metrics['MedAE'] = median_absolute_error (self.y_true, self.y_pred)
        
        # Percentage metrics (with safety)
        try:
            metrics['MAPE'] = mean_absolute_percentage_error (self.y_true, self.y_pred) * 100
        except:
            metrics['MAPE'] = np.nan
        
        # Additional metrics
        metrics['Max Error'] = max_error (self.y_true, self.y_pred)
        metrics['Explained Variance'] = explained_variance_score (self.y_true, self.y_pred)
        
        # Custom metrics
        metrics['RMSE/MAE Ratio'] = metrics['RMSE'] / metrics['MAE']
        metrics['Mean Residual'] = np.mean (self.residuals)
        metrics['Std Residual'] = np.std (self.residuals)
        
        return metrics
    
    def residual_analysis (self):
        """Perform comprehensive residual analysis."""
        analysis = {}
        
        # Basic statistics
        analysis['Mean'] = np.mean (self.residuals)
        analysis['Median'] = np.median (self.residuals)
        analysis['Std'] = np.std (self.residuals)
        analysis['Min'] = np.min (self.residuals)
        analysis['Max'] = np.max (self.residuals)
        analysis['Skewness'] = pd.Series (self.residuals).skew()
        analysis['Kurtosis'] = pd.Series (self.residuals).kurtosis()
        
        # Normality test
        _, p_normal = stats.normaltest (self.residuals)
        analysis['Normality p-value'] = p_normal
        analysis['Normal?'] = p_normal > 0.05
        
        # Heteroscedasticity check (correlation between |residuals| and y)
        corr, p_hetero = stats.spearmanr (self.y_true, np.abs (self.residuals))
        analysis['Heteroscedasticity correlation'] = corr
        analysis['Heteroscedasticity p-value'] = p_hetero
        analysis['Homoscedastic?'] = abs (corr) < 0.3 or p_hetero > 0.05
        
        # Autocorrelation (for time series)
        if len (self.residuals) > 1:
            analysis['Residual Autocorrelation'] = np.corrcoef(
                self.residuals[:-1], self.residuals[1:]
            )[0, 1]
        
        return analysis
    
    def generate_report (self):
        """Generate comprehensive text report."""
        print("="*80)
        print(f"REGRESSION EVALUATION REPORT: {self.model_name}")
        print("="*80)
        
        # Metrics
        metrics = self.calculate_all_metrics()
        print("\\nPERFORMANCE METRICS:")
        print("-"*80)
        for metric, value in metrics.items():
            if not np.isnan (value):
                print(f"  {metric:<25s}: {value:>12.4f}")
        
        # Residual analysis
        residual_stats = self.residual_analysis()
        print("\\nRESIDUAL ANALYSIS:")
        print("-"*80)
        
        print("  Statistics:")
        for key in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis']:
            print(f"    {key:<20s}: {residual_stats[key]:>12.4f}")
        
        print("\\n  Diagnostics:")
        print(f"    Normality test p-value: {residual_stats['Normality p-value']:.4f}")
        print(f"    Residuals normal? {residual_stats['Normal?']}")
        print(f"    Homoscedastic? {residual_stats['Homoscedastic?']}")
        if 'Residual Autocorrelation' in residual_stats:
            print(f"    Autocorrelation: {residual_stats['Residual Autocorrelation']:.4f}")
        
        # Interpretation
        print("\\nINTERPRETATION:")
        print("-"*80)
        
        # Error magnitude
        error_pct = (metrics['RMSE'] / (self.y_true.max() - self.y_true.min())) * 100
        print(f"  Average error: {metrics['MAE']:.2f} units")
        print(f"  RMSE: {metrics['RMSE']:.2f} ({error_pct:.1f}% of target range)")
        print(f"  Variance explained: {metrics['R²']*100:.1f}%")
        
        # Outlier detection
        if metrics['RMSE/MAE Ratio'] > 1.5:
            print("  ⚠️  Large errors present (RMSE >> MAE)")
        
        # Bias
        if abs (residual_stats['Mean']) > metrics['MAE'] * 0.1:
            if residual_stats['Mean'] > 0:
                print("  ⚠️  Systematic under-prediction detected")
            else:
                print("  ⚠️  Systematic over-prediction detected")
        else:
            print("  ✓ No systematic bias")
        
        # Residual assumptions
        if not residual_stats['Normal?']:
            print("  ⚠️  Residuals not normally distributed")
        if not residual_stats['Homoscedastic?']:
            print("  ⚠️  Heteroscedasticity detected (non-constant variance)")
        
        return metrics, residual_stats
    
    def plot_diagnostics (self, save_path=None):
        """Create comprehensive diagnostic plots."""
        fig = plt.figure (figsize=(16, 12))
        
        # 1. Predicted vs Actual
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter (self.y_true, self.y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.plot([self.y_true.min(), self.y_true.max()], 
                 [self.y_true.min(), self.y_true.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values', fontsize=11)
        ax1.set_ylabel('Predicted Values', fontsize=11)
        ax1.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid (alpha=0.3)
        
        # Calculate R²
        r2 = r2_score (self.y_true, self.y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
                verticalalignment='top', bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals vs Predicted
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter (self.y_pred, self.residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.axhline (y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values', fontsize=11)
        ax2.set_ylabel('Residuals', fontsize=11)
        ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax2.grid (alpha=0.3)
        
        # 3. Residual Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist (self.residuals, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline (x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Residuals', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax3.grid (alpha=0.3)
        
        # Add normal curve
        mu, sigma = np.mean (self.residuals), np.std (self.residuals)
        x = np.linspace (self.residuals.min(), self.residuals.max(), 100)
        ax3_twin = ax3.twinx()
        ax3_twin.plot (x, stats.norm.pdf (x, mu, sigma), 'r-', linewidth=2, label='Normal')
        ax3_twin.set_ylabel('Probability Density', fontsize=11)
        ax3_twin.legend()
        
        # 4. Q-Q Plot
        ax4 = plt.subplot(2, 3, 4)
        stats.probplot (self.residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax4.grid (alpha=0.3)
        
        # 5. Scale-Location Plot
        ax5 = plt.subplot(2, 3, 5)
        standardized_residuals = self.residuals / np.std (self.residuals)
        ax5.scatter (self.y_pred, np.sqrt (np.abs (standardized_residuals)), 
                   alpha=0.6, edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('Predicted Values', fontsize=11)
        ax5.set_ylabel('√|Standardized Residuals|', fontsize=11)
        ax5.set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
        ax5.grid (alpha=0.3)
        
        # Add smoothed line
        from scipy.interpolate import make_interp_spline
        sorted_indices = np.argsort (self.y_pred)
        x_smooth = self.y_pred[sorted_indices]
        y_smooth = np.sqrt (np.abs (standardized_residuals))[sorted_indices]
        
        # Smooth with rolling average
        window = max (len (x_smooth) // 10, 5)
        y_rolling = pd.Series (y_smooth).rolling (window, center=True).mean()
        ax5.plot (x_smooth, y_rolling, 'r-', linewidth=2)
        
        # 6. Error Distribution by Magnitude
        ax6 = plt.subplot(2, 3, 6)
        errors = np.abs (self.residuals)
        ax6.scatter (self.y_true, errors, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax6.set_xlabel('Actual Values', fontsize=11)
        ax6.set_ylabel('Absolute Error', fontsize=11)
        ax6.set_title('Error by Actual Value', fontsize=12, fontweight='bold')
        ax6.grid (alpha=0.3)
        
        plt.suptitle (f'Diagnostic Plots: {self.model_name}', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig (save_path, dpi=150, bbox_inches='tight')
            print(f"\\nDiagnostic plots saved to '{save_path}'")
        
        return fig

# Example usage with multiple models
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge (alpha=1.0),
    'Random Forest': RandomForestRegressor (n_estimators=100, random_state=42)
}

print("\\n" + "="*80)
print("COMPREHENSIVE REGRESSION EVALUATION")
print("="*80)

all_metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    evaluator = RegressionEvaluator (y_test, y_pred, model_name=name)
    metrics, residual_stats = evaluator.generate_report()
    
    # Store metrics
    metrics['Model'] = name
    all_metrics.append (metrics)
    
    # Generate plots
    evaluator.plot_diagnostics (save_path=f'diagnostics_{name.replace(" ", "_").lower()}.png')
    
    print("\\n")

# Compare models
print("="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

df_metrics = pd.DataFrame (all_metrics)
print(df_metrics[['Model', 'MAE', 'RMSE', 'R²', 'MAPE']].to_string (index=False))

print("\\nBest model by metric:")
for metric in ['MAE', 'RMSE', 'R²']:
    if metric == 'R²':
        best_idx = df_metrics[metric].idxmax()
    else:
        best_idx = df_metrics[metric].idxmin()
    print(f"  {metric}: {df_metrics.loc[best_idx, 'Model']}")
`,
    },
  ],
  quizId: 'model-evaluation-optimization-regression-metrics',
  multipleChoiceId: 'model-evaluation-optimization-regression-metrics-mc',
};
