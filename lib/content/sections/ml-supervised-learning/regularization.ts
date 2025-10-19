/**
 * Regularization: Ridge and Lasso Section
 */

export const regularizationSection = {
  id: 'regularization',
  title: 'Regularization: Ridge & Lasso',
  content: `# Regularization: Ridge & Lasso

## Introduction

Regularization is a fundamental technique for preventing overfitting by adding a penalty term to the loss function that discourages overly complex models. While standard linear regression minimizes only prediction error, regularized regression also penalizes large coefficient values, leading to simpler, more generalizable models.

**Why Regularization Matters:**
- Prevents overfitting, especially with many features
- Handles multicollinearity (correlated features)
- Enables feature selection (Lasso)
- Provides better generalization to unseen data
- Essential for high-dimensional problems

**Real-World Applications:**
- Genomics: thousands of genes, hundreds of samples
- Text classification: vocabulary of 10,000+ words
- Financial modeling: hundreds of potential predictors
- Image recognition: millions of pixel features
- Trading strategies: many technical indicators

## The Overfitting Problem Revisited

Standard linear regression minimizes:

\\[ \\text{Loss} = \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 \\]

With many features or polynomial terms, this can lead to:
- Very large coefficient values
- Model fitting noise in training data
- Poor generalization to test data

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate data with noise
np.random.seed(42)
n_samples = 30
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_true = 2 * np.sin(X).ravel()
y = y_true + np.random.randn(n_samples) * 0.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create high-degree polynomial features (prone to overfitting)
poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Standard regression (overfits)
model_standard = LinearRegression()
model_standard.fit(X_train_poly, y_train)

# Predictions
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot_standard = model_standard.predict(X_plot_poly)

# Calculate errors
train_pred = model_standard.predict(X_train_poly)
test_pred = model_standard.predict(X_test_poly)

print("Standard Linear Regression (Degree 15 Polynomial):")
print(f"Training R²: {r2_score(y_train, train_pred):.4f}")
print(f"Test R²: {r2_score(y_test, test_pred):.4f}")
print(f"Max coefficient magnitude: {np.max(np.abs(model_standard.coef_)):.2e}")

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='Training data', alpha=0.7)
plt.scatter(X_test, y_test, label='Test data', alpha=0.7, color='red')
plt.plot(X_plot, y_plot_standard, 'g-', linewidth=2, label='Fitted curve')
plt.ylim(-4, 4)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Standard Regression: Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)

# Coefficient magnitude
plt.subplot(1, 2, 2)
plt.bar(range(len(model_standard.coef_)), np.abs(model_standard.coef_))
plt.xlabel('Coefficient Index')
plt.ylabel('Absolute Value')
plt.title('Coefficient Magnitudes (Note the scale!)')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Ridge Regression (L2 Regularization)

### The Concept

Ridge regression adds an L2 penalty term that penalizes the sum of squared coefficients:

\\[ \\text{Loss}_{\\text{Ridge}} = \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 + \\alpha \\sum_{j=1}^{p}\\beta_j^2 \\]

**Components:**
- First term: Prediction error (same as linear regression)
- Second term: L2 penalty on coefficient magnitudes
- \\( \\alpha \\): Regularization strength (hyperparameter)

**Effect:**
- Shrinks all coefficients toward zero
- Never sets coefficients exactly to zero
- Handles multicollinearity well
- Always has a unique solution

### How It Works

\\[ \\hat{\\boldsymbol{\\beta}}_{\\text{Ridge}} = (\\mathbf{X}^T\\mathbf{X} + \\alpha\\mathbf{I})^{-1}\\mathbf{X}^T\\mathbf{y} \\]

The \\( \\alpha\\mathbf{I} \\) term ensures the matrix is always invertible, even with perfect multicollinearity!

\`\`\`python
# Ridge Regression with different alpha values
from sklearn.linear_model import Ridge

alphas = [0.001, 0.1, 1.0, 10.0, 100.0]

plt.figure(figsize=(15, 10))

for idx, alpha in enumerate(alphas, 1):
    # Train Ridge model
    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X_train_poly, y_train)
    
    # Predictions
    y_plot_ridge = model_ridge.predict(X_plot_poly)
    train_pred = model_ridge.predict(X_train_poly)
    test_pred = model_ridge.predict(X_test_poly)
    
    # Plot
    plt.subplot(2, 3, idx)
    plt.scatter(X_train, y_train, label='Train', alpha=0.7, s=50)
    plt.scatter(X_test, y_test, label='Test', alpha=0.7, s=50, color='red')
    plt.plot(X_plot, y_plot_ridge, 'g-', linewidth=2.5, label='Ridge fit')
    plt.ylim(-4, 4)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'α = {alpha}\\nTest R² = {r2_score(y_test, test_pred):.3f}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    print(f"\\nRidge (α={alpha}):")
    print(f"  Train R²: {r2_score(y_train, train_pred):.4f}")
    print(f"  Test R²: {r2_score(y_test, test_pred):.4f}")
    print(f"  Max |coef|: {np.max(np.abs(model_ridge.coef_)):.4f}")

plt.subplot(2, 3, 6)
plt.axis('off')
plt.text(0.5, 0.5, 
         'Ridge Regularization:\\n\\n' +
         '• Small α: Less regularization\\n  (closer to standard regression)\\n\\n' +
         '• Large α: More regularization\\n  (simpler model, smaller coefficients)\\n\\n' +
         '• Choose α via cross-validation',
         ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
\`\`\`

## Lasso Regression (L1 Regularization)

### The Concept

Lasso uses an L1 penalty that penalizes the sum of absolute coefficient values:

\\[ \\text{Loss}_{\\text{Lasso}} = \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 + \\alpha \\sum_{j=1}^{p}|\\beta_j| \\]

**Key Difference from Ridge:**
- **Lasso can set coefficients exactly to zero** → Automatic feature selection
- Creates sparse models (many zero coefficients)
- Useful when you believe only a few features matter

### Why L1 Creates Sparsity

The L1 penalty has "corners" at zero in coefficient space, making it geometrically prefer solutions with some coefficients exactly zero.

\`\`\`python
from sklearn.linear_model import Lasso

# Lasso Regression with different alpha values
alphas_lasso = [0.001, 0.01, 0.05, 0.1, 0.5]

plt.figure(figsize=(15, 10))

for idx, alpha in enumerate(alphas_lasso, 1):
    # Train Lasso model
    model_lasso = Lasso(alpha=alpha, max_iter=10000)
    model_lasso.fit(X_train_poly, y_train)
    
    # Predictions
    y_plot_lasso = model_lasso.predict(X_plot_poly)
    train_pred = model_lasso.predict(X_train_poly)
    test_pred = model_lasso.predict(X_test_poly)
    
    # Count non-zero coefficients
    n_nonzero = np.sum(model_lasso.coef_ != 0)
    
    # Plot
    plt.subplot(2, 3, idx)
    plt.scatter(X_train, y_train, label='Train', alpha=0.7, s=50)
    plt.scatter(X_test, y_test, label='Test', alpha=0.7, s=50, color='red')
    plt.plot(X_plot, y_plot_lasso, 'b-', linewidth=2.5, label='Lasso fit')
    plt.ylim(-4, 4)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'α = {alpha}\\nNon-zero coefs: {n_nonzero}/{len(model_lasso.coef_)}\\nTest R² = {r2_score(y_test, test_pred):.3f}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    print(f"\\nLasso (α={alpha}):")
    print(f"  Train R²: {r2_score(y_train, train_pred):.4f}")
    print(f"  Test R²: {r2_score(y_test, test_pred):.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(model_lasso.coef_)}")
    print(f"  Max |coef|: {np.max(np.abs(model_lasso.coef_)):.4f}")

plt.subplot(2, 3, 6)
plt.axis('off')
plt.text(0.5, 0.5,
         'Lasso Regularization:\\n\\n' +
         '• Automatic feature selection\\n  (sets coefficients to exactly 0)\\n\\n' +
         '• Small α: More features retained\\n\\n' +
         '• Large α: Fewer features, simpler model\\n\\n' +
         '• Useful for interpretability',
         ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.show()
\`\`\`

## Ridge vs. Lasso: Visual Comparison

\`\`\`python
# Direct comparison on same data
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Choose optimal alpha for each via cross-validation
from sklearn.model_selection import cross_val_score

alpha_ridge_opt = 1.0
alpha_lasso_opt = 0.05

models = {
    'Standard': LinearRegression(),
    'Ridge': Ridge(alpha=alpha_ridge_opt),
    'Lasso': Lasso(alpha=alpha_lasso_opt, max_iter=10000)
}

results = []

for idx, (name, model) in enumerate(models.items()):
    # Train
    model.fit(X_train_poly, y_train)
    
    # Predictions
    y_plot_pred = model.predict(X_plot_poly)
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    # Metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        n_nonzero = np.sum(np.abs(coefs) > 1e-10)
    else:
        coefs = []
        n_nonzero = 0
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test RMSE': test_rmse,
        'Non-zero coefs': n_nonzero
    })
    
    # Plot predictions
    axes[0, idx].scatter(X_train, y_train, alpha=0.7, s=50, label='Train')
    axes[0, idx].scatter(X_test, y_test, alpha=0.7, s=50, color='red', label='Test')
    axes[0, idx].plot(X_plot, y_plot_pred, linewidth=2.5, label='Fit')
    axes[0, idx].set_ylim(-4, 4)
    axes[0, idx].set_xlabel('X')
    axes[0, idx].set_ylabel('y')
    axes[0, idx].set_title(f'{name}\\nTest R² = {test_r2:.3f}')
    axes[0, idx].legend()
    axes[0, idx].grid(True, alpha=0.3)
    
    # Plot coefficients
    if len(coefs) > 0:
        axes[1, idx].bar(range(len(coefs)), coefs, alpha=0.7)
        axes[1, idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, idx].set_xlabel('Coefficient Index')
        axes[1, idx].set_ylabel('Coefficient Value')
        axes[1, idx].set_title(f'{name} Coefficients\\n{n_nonzero} non-zero')
        axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print comparison
import pandas as pd
df_comparison = pd.DataFrame(results)
print("\\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(df_comparison.to_string(index=False))
\`\`\`

## Elastic Net: Combining L1 and L2

Elastic Net combines Ridge and Lasso:

\\[ \\text{Loss}_{\\text{ElasticNet}} = \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2 + \\alpha \\rho \\sum_{j=1}^{p}|\\beta_j| + \\frac{\\alpha(1-\\rho)}{2} \\sum_{j=1}^{p}\\beta_j^2 \\]

Where:
- \\( \\rho \\in [0,1] \\): L1 ratio (0 = Ridge, 1 = Lasso)
- \\( \\alpha \\): Overall regularization strength

**Benefits:**
- Handles correlated features better than Lasso
- Still performs feature selection
- More stable than Lasso with highly correlated features

\`\`\`python
from sklearn.linear_model import ElasticNet

# Elastic Net with different l1_ratio
l1_ratios = [0.1, 0.5, 0.9]  # 0 = Ridge, 1 = Lasso

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, l1_ratio in enumerate(l1_ratios):
    model = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    y_plot_pred = model.predict(X_plot_poly)
    test_pred = model.predict(X_test_poly)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
    
    axes[idx].scatter(X_train, y_train, alpha=0.7, s=50, label='Train')
    axes[idx].scatter(X_test, y_test, alpha=0.7, s=50, color='red', label='Test')
    axes[idx].plot(X_plot, y_plot_pred, linewidth=2.5, label='ElasticNet')
    axes[idx].set_ylim(-4, 4)
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].set_title(f'L1 ratio = {l1_ratio}\\nNon-zero: {n_nonzero}\\nTest R² = {r2_score(y_test, test_pred):.3f}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Cross-Validation for Hyperparameter Tuning

The key to regularization is choosing the right \\( \\alpha \\). Use cross-validation:

\`\`\`python
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score

# Ridge with built-in CV
alphas_ridge = np.logspace(-3, 3, 50)
model_ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5)
model_ridge_cv.fit(X_train_poly, y_train)

print(f"Optimal Ridge alpha: {model_ridge_cv.alpha_:.4f}")

# Lasso with built-in CV
model_lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
model_lasso_cv.fit(X_train_poly, y_train)

print(f"Optimal Lasso alpha: {model_lasso_cv.alpha_:.4f}")

# Manual CV for visualization
alpha_range = np.logspace(-3, 2, 30)
ridge_scores = []
lasso_scores = []

for alpha in alpha_range:
    # Ridge
    model_ridge = Ridge(alpha=alpha)
    scores_ridge = cross_val_score(model_ridge, X_train_poly, y_train, cv=5, scoring='r2')
    ridge_scores.append(scores_ridge.mean())
    
    # Lasso
    model_lasso = Lasso(alpha=alpha, max_iter=10000)
    scores_lasso = cross_val_score(model_lasso, X_train_poly, y_train, cv=5, scoring='r2')
    lasso_scores.append(scores_lasso.mean())

# Plot CV curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(alpha_range, ridge_scores, 'o-', label='Ridge', linewidth=2)
plt.plot(alpha_range, lasso_scores, 's-', label='Lasso', linewidth=2)
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Cross-Validation R² Score')
plt.title('Regularization Strength vs. Performance')
plt.xscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=model_ridge_cv.alpha_, color='blue', linestyle='--', alpha=0.5, label='Ridge optimal')
plt.axvline(x=model_lasso_cv.alpha_, color='orange', linestyle='--', alpha=0.5, label='Lasso optimal')

# Number of features vs alpha (Lasso)
n_features_lasso = []
for alpha in alpha_range:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)
    n_features_lasso.append(np.sum(np.abs(model.coef_) > 1e-10))

plt.subplot(1, 2, 2)
plt.plot(alpha_range, n_features_lasso, 'o-', color='orange', linewidth=2)
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Number of Non-Zero Coefficients')
plt.title('Feature Selection with Lasso')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.axvline(x=model_lasso_cv.alpha_, color='red', linestyle='--', alpha=0.5, label='Optimal α')
plt.legend()

plt.tight_layout()
plt.show()
\`\`\`

## Real-World Example: Stock Return Prediction with Many Factors

\`\`\`python
# Realistic scenario: predicting stock returns with many technical indicators
np.random.seed(42)
n_days = 252  # One year of trading days
n_features = 50  # Many technical indicators

# Generate features (technical indicators)
X_stock = np.random.randn(n_days, n_features)

# Only 5 features actually matter (sparse ground truth)
true_important_features = [0, 5, 12, 23, 41]
true_coefs = np.zeros(n_features)
true_coefs[true_important_features] = [0.8, -0.6, 0.5, 0.7, -0.4]

# Generate returns
y_stock = X_stock @ true_coefs + np.random.randn(n_days) * 0.3

# Split data (time-based for financial data)
split_idx = int(0.7 * n_days)
X_train_stock = X_stock[:split_idx]
y_train_stock = y_stock[:split_idx]
X_test_stock = X_stock[split_idx:]
y_test_stock = y_stock[split_idx:]

print("="*60)
print("FINANCIAL EXAMPLE: STOCK RETURN PREDICTION")
print("="*60)
print(f"Training samples: {len(X_train_stock)}")
print(f"Test samples: {len(X_test_stock)}")
print(f"Number of features: {n_features}")
print(f"True important features: {len(true_important_features)}")

# Compare models
models_stock = {
    'Standard': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5),
    'Lasso': LassoCV(cv=5, max_iter=10000),
}

results_stock = []

for name, model in models_stock.items():
    model.fit(X_train_stock, y_train_stock)
    
    train_pred = model.predict(X_train_stock)
    test_pred = model.predict(X_test_stock)
    
    train_r2 = r2_score(y_train_stock, train_pred)
    test_r2 = r2_score(y_test_stock, test_pred)
    test_mse = mean_squared_error(y_test_stock, test_pred)
    
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 0.01)
    else:
        n_nonzero = n_features
    
    results_stock.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MSE': test_mse,
        'Features Used': n_nonzero
    })

df_stock_results = pd.DataFrame(results_stock)
print("\\n" + df_stock_results.to_string(index=False))

# Visualize feature selection (Lasso)
model_lasso_stock = models_stock['Lasso']

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(range(n_features), np.abs(model_lasso_stock.coef_), alpha=0.7)
plt.bar(true_important_features, np.abs(model_lasso_stock.coef_[true_important_features]), 
        color='red', alpha=0.9, label='True important features')
plt.xlabel('Feature Index')
plt.ylabel('Absolute Coefficient Value')
plt.title('Lasso Feature Selection')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
selected_features = np.where(np.abs(model_lasso_stock.coef_) > 0.01)[0]
print(f"\\nLasso selected features: {selected_features}")
print(f"True important features: {true_important_features}")
print(f"Correctly identified: {len(set(selected_features) & set(true_important_features))}/{len(true_important_features)}")

# Plot predictions
plt.scatter(y_test_stock, test_pred, alpha=0.6)
plt.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.title(f'Lasso Predictions\\nTest R² = {r2_score(y_test_stock, test_pred):.3f}')
plt.grid(True, alpha=0.3)

# Time series of predictions
plt.subplot(1, 3, 3)
plt.plot(y_test_stock, label='Actual', alpha=0.7, linewidth=2)
plt.plot(test_pred, label='Predicted', alpha=0.7, linewidth=2)
plt.xlabel('Trading Day')
plt.ylabel('Return')
plt.title('Out-of-Sample Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## When to Use Which Regularization

**Ridge (L2):**
- All features potentially relevant
- Features are correlated
- Want smooth coefficient shrinkage
- Need stable solution
- Example: Text classification with all words

**Lasso (L1):**
- Sparse ground truth (few important features)
- Want interpretable model
- Need feature selection
- Features mostly independent
- Example: Genomics with few causal genes

**Elastic Net:**
- Mix of both scenarios
- Highly correlated features with sparse ground truth
- Want robust feature selection
- Example: High-dimensional data with feature groups

## Best Practices

1. **Always use cross-validation** to select \\( \\alpha \\)
2. **Scale features** before regularization (important!)
3. **Start with Ridge** if unsure
4. **Use Lasso for interpretability** when you need feature selection
5. **Try Elastic Net** for high-dimensional problems
6. **Visualize regularization path** to understand feature importance
7. **Don't regularize intercept** (already done by default in sklearn)

## Common Pitfalls

1. **Forgetting to scale**: Regularization is scale-dependent
2. **Wrong alpha**: Too small = underfitting, too large = overfitting
3. **Using test set for alpha selection**: Use validation/CV instead
4. **Over-interpreting Lasso selections**: Statistical properties tricky
5. **Ignoring multicollinearity**: Lasso arbitrarily picks among correlated features

## Summary

Regularization prevents overfitting by penalizing model complexity:

**Ridge (L2)**:
- Adds \\( \\alpha \\sum \\beta_j^2 \\) penalty
- Shrinks coefficients toward zero
- Never exactly zero
- Best for correlated features

**Lasso (L1)**:
- Adds \\( \\alpha \\sum |\\beta_j| \\) penalty
- Can set coefficients exactly to zero
- Automatic feature selection
- Best for sparse problems

**Elastic Net**:
- Combines L1 and L2
- Balance feature selection and stability

**Key insight**: Regularization trades training performance for better generalization!

Next, we'll move beyond regression to classification with Logistic Regression!
`,
  codeExample: `# Complete Regularization Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Generate high-dimensional dataset
np.random.seed(42)
n_samples = 200
n_features = 100  # Many features
n_informative = 10  # Only 10 actually matter

# Create features
X = np.random.randn(n_samples, n_features)

# True sparse coefficient vector (only 10 non-zero)
true_coef = np.zeros(n_features)
informative_idx = np.random.choice(n_features, n_informative, replace=False)
true_coef[informative_idx] = np.random.randn(n_informative) * 2

# Generate target
y = X @ true_coef + np.random.randn(n_samples) * 0.5

print("="*70)
print("COMPREHENSIVE REGULARIZATION COMPARISON")
print("="*70)
print(f"\\nDataset:")
print(f"  Samples: {n_samples}")
print(f"  Total features: {n_features}")
print(f"  Informative features: {n_informative}")
print(f"  True non-zero coefficients: {np.sum(true_coef != 0)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# CRITICAL: Scale features for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Define models with optimal hyperparameters via CV
models = {
    'Standard': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
    'Lasso': LassoCV(cv=5, max_iter=10000, random_state=42),
    'ElasticNet': ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, max_iter=10000, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\\nTraining {name}...")
    
    # Fit model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Coefficient analysis
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        n_nonzero = np.sum(np.abs(coefs) > 1e-5)
        max_coef = np.max(np.abs(coefs))
        
        # Feature recovery: how many true features identified?
        selected_features = np.where(np.abs(coefs) > 0.1)[0]
        true_features = np.where(true_coef != 0)[0]
        correctly_identified = len(set(selected_features) & set(true_features))
    else:
        n_nonzero = n_features
        max_coef = np.nan
        correctly_identified = np.nan
    
    # Get optimal hyperparameters
    if hasattr(model, 'alpha_'):
        alpha_opt = model.alpha_
    else:
        alpha_opt = np.nan
    
    if hasattr(model, 'l1_ratio_'):
        l1_ratio = model.l1_ratio_
    else:
        l1_ratio = np.nan
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Overfit Gap': train_r2 - test_r2,
        'Non-zero Coefs': n_nonzero,
        'Max |Coef|': max_coef,
        'Correct Features': correctly_identified,
        'Alpha': alpha_opt,
        'L1 Ratio': l1_ratio
    })

# Results table
df_results = pd.DataFrame(results)
print("\\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print(df_results.to_string(index=False))

# Find best model
best_idx = df_results['Test R²'].idxmax()
best_model_name = df_results.loc[best_idx, 'Model']
print(f"\\n{'='*70}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*70}")
print(f"Test R²: {df_results.loc[best_idx, 'Test R²']:.4f}")
print(f"Test RMSE: {df_results.loc[best_idx, 'Test RMSE']:.4f}")
print(f"Overfitting gap: {df_results.loc[best_idx, 'Overfit Gap']:.4f}")

# Visualization 1: Model Performance
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# R² comparison
ax = axes[0, 0]
x_pos = np.arange(len(df_results))
width = 0.35
ax.bar(x_pos - width/2, df_results['Train R²'], width, label='Train R²', alpha=0.8)
ax.bar(x_pos + width/2, df_results['Test R²'], width, label='Test R²', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('R² Score')
ax.set_title('Model Performance: R² Scores')
ax.set_xticks(x_pos)
ax.set_xticklabels(df_results['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Feature selection
ax = axes[0, 1]
colors = ['red' if x == n_informative else 'blue' for x in df_results['Non-zero Coefs']]
ax.bar(range(len(df_results)), df_results['Non-zero Coefs'], color=colors, alpha=0.7)
ax.axhline(y=n_informative, color='green', linestyle='--', linewidth=2, label='True # features')
ax.set_xlabel('Model')
ax.set_ylabel('Number of Non-Zero Coefficients')
ax.set_title('Feature Selection Sparsity')
ax.set_xticks(range(len(df_results)))
ax.set_xticklabels(df_results['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Coefficient magnitudes for each model
for idx, (name, model) in enumerate(models.items()):
    if hasattr(model, 'coef_'):
        ax = axes[1, idx % 2] if idx < 2 else plt.subplot(2, 4, 5 + idx - 2)
        
        # Plot coefficients
        coefs = model.coef_
        true_nonzero = np.where(true_coef != 0)[0]
        
        ax.bar(range(len(coefs)), np.abs(coefs), alpha=0.5, label='Estimated')
        ax.bar(true_nonzero, np.abs(coefs[true_nonzero]), 
               color='red', alpha=0.8, label='True important')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Absolute Coefficient Value')
        ax.set_title(f'{name} Coefficients\\n({int(df_results.iloc[idx]["Non-zero Coefs"])} non-zero)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if name == 'Standard':
            ax.set_yscale('log')

plt.tight_layout()
plt.show()

# Visualization 2: Predictions
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx // 2, idx % 2]
    
    y_pred = model.predict(X_test_scaled)
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{name}\\nTest R² = {r2_score(y_test, y_pred):.4f}, RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# Detailed analysis of best model
best_model = models[best_model_name]

print("\\n" + "="*70)
print(f"DETAILED ANALYSIS: {best_model_name}")
print("="*70)

if hasattr(best_model, 'alpha_'):
    print(f"\\nOptimal alpha: {best_model.alpha_:.6f}")
    
if hasattr(best_model, 'l1_ratio_'):
    print(f"Optimal L1 ratio: {best_model.l1_ratio_:.4f}")

if hasattr(best_model, 'coef_'):
    coefs = best_model.coef_
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': range(n_features),
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs),
        'Is_True_Feature': [i in informative_idx for i in range(n_features)]
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\\nTop 15 Features by Coefficient Magnitude:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Recovery analysis
    selected = feature_importance[feature_importance['Abs_Coefficient'] > 0.1]['Feature'].values
    true_features = set(informative_idx)
    selected_features = set(selected)
    
    true_positives = len(true_features & selected_features)
    false_positives = len(selected_features - true_features)
    false_negatives = len(true_features - selected_features)
    
    print(f"\\nFeature Selection Analysis:")
    print(f"  True informative features: {len(true_features)}")
    print(f"  Features selected by model: {len(selected_features)}")
    print(f"  Correctly identified: {true_positives}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")
    
    if len(selected_features) > 0:
        precision = true_positives / len(selected_features)
        recall = true_positives / len(true_features)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\\n  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-score: {f1:.3f}")

# Summary
print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
When dealing with high-dimensional data ({n_features} features, {n_samples} samples):

1. Standard linear regression overfits severely
   - High training R², low test R²
   - All {n_features} features used
   - Unstable coefficient estimates

2. Ridge regression improves generalization
   - Shrinks all coefficients
   - Uses all features (no feature selection)
   - Good when most features are somewhat relevant

3. Lasso performs automatic feature selection
   - Sets many coefficients to exactly zero
   - Identifies {df_results[df_results['Model']=='Lasso']['Correct Features'].values[0]}/{n_informative} true features
   - Best for sparse ground truth

4. ElasticNet balances both approaches
   - Combines L1 and L2 penalties
   - More stable than Lasso with correlated features
   - Flexible middle ground

Best model: {best_model_name}
- Achieves {df_results.loc[best_idx, 'Test R²']:.2%} test R²
- Uses {int(df_results.loc[best_idx, 'Non-zero Coefs'])}/{n_features} features
- Minimal overfitting (gap: {df_results.loc[best_idx, 'Overfit Gap']:.4f})

Key takeaway: Regularization is essential for high-dimensional problems!
""")
`,
};
