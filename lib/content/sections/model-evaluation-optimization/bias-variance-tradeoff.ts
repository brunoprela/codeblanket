export const biasVarianceTradeoff = {
  title: 'Bias-Variance Tradeoff',
  content: `
# Bias-Variance Tradeoff

## Introduction

The bias-variance tradeoff is one of the fundamental concepts in machine learning. It explains why models fail: they're either too simple (high bias) or too complex (high variance). Understanding this tradeoff is crucial for building models that generalize well.

**The Central Problem**: We want models that perform well on unseen data, but we can only train on seen data. The bias-variance decomposition helps us understand the sources of prediction error.

## Total Error Decomposition

For a regression problem, the expected prediction error can be decomposed into three components:

$$\\text{Expected Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Error}$$

### Understanding Each Component

**1. Bias**: Error from wrong assumptions in the learning algorithm
- High bias → Underfitting
- Model is too simple to capture true relationship
- Systematic errors

**2. Variance**: Error from sensitivity to small fluctuations in training data
- High variance → Overfitting
- Model captures noise as if it were signal
- Model changes dramatically with different training sets

**3. Irreducible Error**: Noise in the data that cannot be reduced
- Inherent randomness
- Measurement errors
- Unknown factors

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

print("Bias-Variance Tradeoff")
print("="*70)

# Generate synthetic data with known function
np.random.seed(42)
n_samples = 100

def true_function (x):
    """True underlying function"""
    return np.sin(2 * np.pi * x)

# Generate data
X = np.sort (np.random.uniform(0, 1, n_samples))
y = true_function(X) + np.random.normal(0, 0.1, n_samples)  # Add noise

# Reshape for sklearn
X = X.reshape(-1, 1)

print("Dataset:")
print(f"  Samples: {n_samples}")
print(f"  True function: y = sin(2πx)")
print(f"  Noise level: σ = 0.1")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create X values for plotting smooth curves
X_plot = np.linspace(0, 1, 300).reshape(-1, 1)
y_true = true_function(X_plot.ravel())
\`\`\`

## Visualizing Bias and Variance

\`\`\`python
print("\\n" + "="*70)
print("Demonstrating Bias and Variance with Different Models")
print("="*70)

# Train three models with different complexity
models = {
    'High Bias\\n(Underfit)': LinearRegression(),  # Too simple
    'Just Right': Ridge (alpha=0.01),  # Appropriate complexity
    'High Variance\\n(Overfit)': DecisionTreeRegressor (max_depth=10)  # Too complex
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip (axes, models.items()):
    # Train model
    if 'High Bias' in name:
        # Linear model
        model.fit(X_train, y_train)
        y_plot = model.predict(X_plot)
    elif 'Just Right' in name:
        # Polynomial with regularization
        poly = PolynomialFeatures (degree=7)
        X_train_poly = poly.fit_transform(X_train)
        X_plot_poly = poly.transform(X_plot)
        model.fit(X_train_poly, y_train)
        y_plot = model.predict(X_plot_poly)
    else:
        # Decision tree (high variance)
        model.fit(X_train, y_train)
        y_plot = model.predict(X_plot)
    
    # Plot
    ax.scatter(X_train, y_train, alpha=0.6, s=30, label='Training data', color='blue')
    ax.plot(X_plot, y_true, 'g--', linewidth=2, label='True function', alpha=0.7)
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model prediction')
    
    # Calculate errors
    if 'High Bias' in name:
        y_pred_test = model.predict(X_test)
    elif 'Just Right' in name:
        X_test_poly = poly.transform(X_test)
        y_pred_test = model.predict(X_test_poly)
    else:
        y_pred_test = model.predict(X_test)
    
    train_error = mean_squared_error (y_train, 
                                    model.predict(X_train) if 'High Bias' in name or 'High Variance' in name
                                    else model.predict(X_train_poly))
    test_error = mean_squared_error (y_test, y_pred_test)
    
    ax.set_title (f'{name}\\nTrain MSE: {train_error:.4f}, Test MSE: {test_error:.4f}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend (fontsize=9)
    ax.grid (alpha=0.3)

plt.tight_layout()
plt.savefig('bias_variance_demo.png', dpi=150, bbox_inches='tight')
print("\\nVisualization saved to 'bias_variance_demo.png'")

print("\\nObservations:")
print("  High Bias: Smooth curve, misses true pattern (underfitting)")
print("  Just Right: Captures signal without noise (good generalization)")
print("  High Variance: Fits training data too closely (overfitting)")
\`\`\`

## Quantifying Bias and Variance

\`\`\`python
print("\\n" + "="*70)
print("Measuring Bias and Variance Empirically")
print("="*70)

def bias_variance_decomposition (model, X_train, y_train, X_test, y_test, 
                                n_iterations=100, random_state=42):
    """
    Empirically estimate bias and variance of a model.
    
    Process:
    1. Create many bootstrap samples
    2. Train model on each
    3. Measure predictions on test set
    4. Calculate bias and variance from prediction distribution
    """
    np.random.seed (random_state)
    n_samples = len(X_train)
    predictions = np.zeros((n_iterations, len(X_test)))
    
    for i in range (n_iterations):
        # Bootstrap sample
        indices = np.random.choice (n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # Train and predict
        model_clone = type (model)(**model.get_params())
        model_clone.fit(X_boot, y_boot)
        predictions[i] = model_clone.predict(X_test)
    
    # Calculate bias and variance
    mean_predictions = predictions.mean (axis=0)
    
    # Bias^2: how far average prediction is from true value
    bias_squared = ((mean_predictions - y_test) ** 2).mean()
    
    # Variance: how much predictions vary across different training sets
    variance = predictions.var (axis=0).mean()
    
    # Total error
    mse = ((predictions - y_test) ** 2).mean()
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'mse': mse,
        'predictions': predictions,
        'mean_prediction': mean_predictions
    }

# Test with different models
test_models = {
    'Linear (d=1)': LinearRegression(),
    'Poly (d=3)': 'poly3',
    'Poly (d=7)': 'poly7',
    'Poly (d=15)': 'poly15',
    'Decision Tree (d=3)': DecisionTreeRegressor (max_depth=3, random_state=42),
    'Decision Tree (d=10)': DecisionTreeRegressor (max_depth=10, random_state=42),
}

results = []

for name, model in test_models.items():
    print(f"\\nAnalyzing: {name}")
    
    if isinstance (model, str) and 'poly' in model:
        # Polynomial regression
        degree = int (model.replace('poly', '))
        poly = PolynomialFeatures (degree=degree)
        X_train_trans = poly.fit_transform(X_train)
        X_test_trans = poly.transform(X_test)
        
        base_model = LinearRegression()
        
        # Custom training function for polynomial
        def train_and_predict(X_tr, y_tr, X_te):
            model_temp = LinearRegression()
            model_temp.fit(X_tr, y_tr)
            return model_temp.predict(X_te)
        
        # Manual bias-variance decomposition for polynomial
        n_iterations = 100
        predictions = np.zeros((n_iterations, len(X_test)))
        
        for i in range (n_iterations):
            indices = np.random.choice (len(X_train), size=len(X_train), replace=True)
            X_boot = X_train_trans[indices]
            y_boot = y_train[indices]
            
            model_temp = LinearRegression()
            model_temp.fit(X_boot, y_boot)
            predictions[i] = model_temp.predict(X_test_trans)
        
        mean_predictions = predictions.mean (axis=0)
        bias_squared = ((mean_predictions - y_test) ** 2).mean()
        variance = predictions.var (axis=0).mean()
        mse = ((predictions - y_test) ** 2).mean()
        
        result = {
            'bias_squared': bias_squared,
            'variance': variance,
            'mse': mse
        }
    else:
        result = bias_variance_decomposition (model, X_train, y_train, X_test, y_test)
    
    results.append({
        'Model': name,
        'Bias²': result['bias_squared'],
        'Variance': result['variance'],
        'MSE': result['mse'],
        'Bias²+Variance': result['bias_squared'] + result['variance']
    })
    
    print(f"  Bias²: {result['bias_squared']:.6f}")
    print(f"  Variance: {result['variance']:.6f}")
    print(f"  MSE: {result['mse']:.6f}")
    print(f"  Bias²+Variance: {result['bias_squared'] + result['variance']:.6f}")

# Display as table
df_results = pd.DataFrame (results)
print("\\n" + "="*70)
print("Bias-Variance Decomposition Summary:")
print("="*70)
print(df_results.to_string (index=False))

# Visualize tradeoff
plt.figure (figsize=(12, 6))

models_sorted = df_results.sort_values('Bias²', ascending=False)

x = np.arange (len (models_sorted))
width = 0.25

plt.bar (x - width, models_sorted['Bias²'], width, label='Bias²', alpha=0.8)
plt.bar (x, models_sorted['Variance'], width, label='Variance', alpha=0.8)
plt.bar (x + width, models_sorted['MSE'], width, label='Total MSE', alpha=0.8)

plt.xlabel('Model (increasing complexity →)', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
plt.xticks (x, models_sorted['Model'], rotation=45, ha='right')
plt.legend()
plt.grid (axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
print("\\nTradeoff visualization saved to 'bias_variance_tradeoff.png'")

print("\\nKey Insight:")
print("  As model complexity increases:")
print("    • Bias² decreases (model can fit true pattern)")
print("    • Variance increases (model fits noise)")
print("    • Optimal complexity minimizes total error")
\`\`\`

## The Tradeoff in Action: Learning Curves

\`\`\`python
from sklearn.model_selection import learning_curve

print("\\n" + "="*70)
print("Learning Curves: Visualizing Bias and Variance")
print("="*70)

def plot_learning_curves (model, X, y, title, cv=5):
    """Plot learning curves for a model."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    train_mean = -train_scores.mean (axis=1)
    train_std = train_scores.std (axis=1)
    val_mean = -val_scores.mean (axis=1)
    val_std = val_scores.std (axis=1)
    
    return train_sizes, train_mean, train_std, val_mean, val_std

# Create learning curves for different models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_lc = [
    ('High Bias (Linear)', LinearRegression()),
    ('Balanced (Ridge)', Ridge (alpha=1.0)),
    ('High Variance (Tree d=15)', DecisionTreeRegressor (max_depth=15, random_state=42))
]

for ax, (title, model) in zip (axes, models_lc):
    train_sizes, train_mean, train_std, val_mean, val_std = plot_learning_curves(
        model, X_train, y_train, title
    )
    
    ax.plot (train_sizes, train_mean, label='Training error', linewidth=2)
    ax.fill_between (train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    ax.plot (train_sizes, val_mean, label='Validation error', linewidth=2)
    ax.fill_between (train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    ax.set_xlabel('Training Size', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title (title, fontsize=12, fontweight='bold')
    ax.legend (fontsize=10)
    ax.grid (alpha=0.3)
    
    # Add interpretation
    if 'High Bias' in title:
        ax.text(0.5, 0.95, 'Both errors high & close\\n (underfitting)',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.7))
    elif 'High Variance' in title:
        ax.text(0.5, 0.95, 'Large gap between curves\\n (overfitting)',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax.text(0.5, 0.95, 'Small gap, low error\\n (good fit)',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
print("\\nLearning curves saved to 'learning_curves.png'")

print("\\nInterpreting Learning Curves:")
print("-"*70)
print("High Bias (Underfitting):")
print("  • Training and validation error are both high")
print("  • Small gap between curves")
print("  • More data won't help much")
print("  • Solution: Increase model complexity")

print("\\nHigh Variance (Overfitting):")
print("  • Training error is low, validation error is high")
print("  • Large gap between curves")
print("  • More data would help")
print("  • Solution: Reduce model complexity or add regularization")

print("\\nGood Fit:")
print("  • Both errors are low")
print("  • Small gap between curves")
print("  • Curves converge")
print("  • Model generalizes well")
\`\`\`

## Diagnosing and Fixing: A Practical Guide

\`\`\`python
print("\\n" + "="*70)
print("Practical Guide: Diagnosing and Fixing Models")
print("="*70)

def diagnose_model (train_error, val_error, threshold_ratio=1.3, threshold_absolute=0.1):
    """
    Diagnose if model has bias or variance problem.
    
    Parameters:
    -----------
    train_error : float
        Error on training set
    val_error : float
        Error on validation set
    threshold_ratio : float
        Ratio threshold for detecting overfitting
    threshold_absolute : float
        Absolute error threshold for detecting underfitting
    
    Returns:
    --------
    diagnosis : dict
    """
    ratio = val_error / train_error if train_error > 0 else float('inf')
    
    if ratio > threshold_ratio and val_error > threshold_absolute:
        diagnosis = {
            'problem': 'High Variance (Overfitting)',
            'symptoms': [
                f'Validation error ({val_error:.4f}) much higher than training ({train_error:.4f})',
                f'Error ratio: {ratio:.2f}x'
            ],
            'solutions': [
                '1. Get more training data',
                '2. Reduce model complexity',
                '3. Add regularization (L1/L2)',
                '4. Feature selection (remove irrelevant features)',
                '5. Early stopping',
                '6. Ensemble methods (bagging)'
            ]
        }
    elif train_error > threshold_absolute and ratio < threshold_ratio:
        diagnosis = {
            'problem': 'High Bias (Underfitting)',
            'symptoms': [
                f'Both training ({train_error:.4f}) and validation ({val_error:.4f}) errors are high',
                f'Errors are close together'
            ],
            'solutions': [
                '1. Increase model complexity',
                '2. Add more features or feature engineering',
                '3. Reduce regularization',
                '4. Train longer (for neural networks)',
                '5. Use more powerful model class',
                '6. Ensemble methods (boosting)'
            ]
        }
    else:
        diagnosis = {
            'problem': 'Good Fit',
            'symptoms': [
                f'Training error ({train_error:.4f}) and validation error ({val_error:.4f}) are both low',
                f'Small gap between errors'
            ],
            'solutions': [
                'Model appears to be performing well!',
                'Consider fine-tuning hyperparameters for marginal improvements',
                'Validate on held-out test set'
            ]
        }
    
    return diagnosis

# Test diagnosis on our models
print("\\nModel Diagnostics:")
print("="*70)

test_cases = [
    {'name': 'Linear Model', 'train_error': 0.15, 'val_error': 0.16},
    {'name': 'Deep Tree', 'train_error': 0.01, 'val_error': 0.25},
    {'name': 'Well-tuned RF', 'train_error': 0.05, 'val_error': 0.06},
]

for case in test_cases:
    diagnosis = diagnose_model (case['train_error'], case['val_error'])
    
    print(f"\\n{case['name']}:")
    print(f"  Problem: {diagnosis['problem']}")
    print("  Symptoms:")
    for symptom in diagnosis['symptoms']:
        print(f"    • {symptom}")
    print("  Recommended Solutions:")
    for solution in diagnosis['solutions']:
        print(f"    {solution}")
\`\`\`

## Regularization: Controlling Variance

\`\`\`python
print("\\n" + "="*70)
print("Regularization Effect on Bias-Variance")
print("="*70)

# Demonstrate regularization effect
from sklearn.linear_model import Ridge

alphas = [0, 0.01, 0.1, 1.0, 10.0, 100.0]
results_reg = []

poly = PolynomialFeatures (degree=15)  # High degree polynomial
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

for alpha in alphas:
    model = Ridge (alpha=alpha)
    model.fit(X_train_poly, y_train)
    
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error (y_train, train_pred)
    test_mse = mean_squared_error (y_test, test_pred)
    
    results_reg.append({
        'Alpha': alpha,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Gap': test_mse - train_mse
    })

df_reg = pd.DataFrame (results_reg)
print("\\nRegularization Strength vs Performance:")
print(df_reg.to_string (index=False))

# Plot
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (df_reg['Alpha'], df_reg['Train MSE'], 'o-', label='Train MSE', linewidth=2)
plt.plot (df_reg['Alpha'], df_reg['Test MSE'], 's-', label='Test MSE', linewidth=2)
plt.xscale('log')
plt.xlabel('Regularization Strength (α)', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Effect of Regularization', fontsize=14, fontweight='bold')
plt.legend()
plt.grid (alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot (df_reg['Alpha'], df_reg['Gap'], 'o-', color='red', linewidth=2)
plt.xscale('log')
plt.xlabel('Regularization Strength (α)', fontsize=12)
plt.ylabel('Test MSE - Train MSE (Gap)', fontsize=12)
plt.title('Overfitting Gap', fontsize=14, fontweight='bold')
plt.grid (alpha=0.3)
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_effect.png', dpi=150, bbox_inches='tight')
print("\\nRegularization effect saved to 'regularization_effect.png'")

# Find optimal alpha
optimal_idx = df_reg['Test MSE'].idxmin()
optimal_alpha = df_reg.loc[optimal_idx, 'Alpha']
print(f"\\nOptimal regularization: α = {optimal_alpha}")
print("  → Minimizes test error")
print("  → Balances bias and variance")
\`\`\`

## Model Complexity Curve

\`\`\`python
print("\\n" + "="*70)
print("Model Complexity Curve")
print("="*70)

# Vary model complexity systematically
complexities = [1, 2, 3, 5, 7, 10, 15, 20]
complexity_results = []

for degree in complexities:
    poly = PolynomialFeatures (degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    train_mse = mean_squared_error (y_train, model.predict(X_train_poly))
    test_mse = mean_squared_error (y_test, model.predict(X_test_poly))
    
    complexity_results.append({
        'Degree': degree,
        'Train MSE': train_mse,
        'Test MSE': test_mse
    })

df_complexity = pd.DataFrame (complexity_results)

# Plot
plt.figure (figsize=(12, 6))

plt.plot (df_complexity['Degree'], df_complexity['Train MSE'], 
        'o-', label='Training Error', linewidth=2, markersize=8)
plt.plot (df_complexity['Degree'], df_complexity['Test MSE'], 
        's-', label='Validation Error', linewidth=2, markersize=8)

# Find sweet spot
optimal_degree = df_complexity.loc[df_complexity['Test MSE'].idxmin(), 'Degree']
optimal_test_mse = df_complexity['Test MSE'].min()

plt.axvline (x=optimal_degree, color='green', linestyle='--', 
           linewidth=2, alpha=0.7, label=f'Optimal (degree={optimal_degree})')

# Annotate regions
plt.text(2, plt.ylim()[1]*0.9, 'High Bias\\n(Underfitting)', 
        fontsize=11, ha='center', bbox=dict (boxstyle='round', facecolor='red', alpha=0.3))
plt.text (optimal_degree, plt.ylim()[1]*0.9, 'Sweet Spot', 
        fontsize=11, ha='center', bbox=dict (boxstyle='round', facecolor='green', alpha=0.3))
plt.text(17, plt.ylim()[1]*0.9, 'High Variance\\n(Overfitting)', 
        fontsize=11, ha='center', bbox=dict (boxstyle='round', facecolor='orange', alpha=0.3))

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Model Complexity Curve', fontsize=14, fontweight='bold')
plt.legend (fontsize=11)
plt.grid (alpha=0.3)
plt.tight_layout()
plt.savefig('complexity_curve.png', dpi=150, bbox_inches='tight')
print("\\nComplexity curve saved to 'complexity_curve.png'")

print(f"\\nOptimal complexity: Polynomial degree = {optimal_degree}")
print(f"Corresponding test MSE: {optimal_test_mse:.6f}")
\`\`\`

## Trading Application: Strategy Complexity

\`\`\`python
print("\\n" + "="*70)
print("Trading Application: Strategy Complexity")
print("="*70)

# Simulate trading strategy with different number of features
np.random.seed(42)
n_days = 500

# Generate price data
returns = np.random.randn (n_days) * 0.02
prices = 100 * np.exp (np.cumsum (returns))

# Create features of varying complexity
def create_trading_features (prices, n_features):
    """Create n_features technical indicators."""
    features = {}
    
    # Always include basic features
    features['returns'] = np.diff (prices, prepend=prices[0]) / prices
    
    if n_features >= 2:
        features['sma_5'] = pd.Series (prices).rolling(5).mean().fillna (method='bfill').values
    
    if n_features >= 3:
        features['sma_20'] = pd.Series (prices).rolling(20).mean().fillna (method='bfill').values
    
    if n_features >= 4:
        features['volatility'] = pd.Series (features['returns']).rolling(10).std().fillna (method='bfill').values
    
    # Add increasingly complex/noisy features
    for i in range(4, n_features):
        # Random technical indicators (simulating overfitting to noise)
        period = np.random.randint(5, 50)
        features[f'indicator_{i}'] = pd.Series (prices).rolling (period).mean().fillna (method='bfill').values
    
    return np.column_stack (list (features.values()))

# Target: next day return
y_trading = np.diff (prices[1:]) / prices[1:-1]
y_trading_binary = (y_trading > 0).astype (int)

# Test different numbers of features
feature_counts = [1, 2, 3, 5, 10, 20, 50]
strategy_results = []

for n_feat in feature_counts:
    X_feat = create_trading_features (prices[:-2], n_feat)
    
    # Ensure same length
    min_len = min (len(X_feat), len (y_trading_binary))
    X_feat = X_feat[-min_len:]
    y_target = y_trading_binary[-min_len:]
    
    # Split
    split_point = int(0.7 * len(X_feat))
    X_train_strat = X_feat[:split_point]
    X_test_strat = X_feat[split_point:]
    y_train_strat = y_target[:split_point]
    y_test_strat = y_target[split_point:]
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model_strat = RandomForestClassifier (n_estimators=100, max_depth=5, random_state=42)
    model_strat.fit(X_train_strat, y_train_strat)
    
    # Evaluate
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score (y_train_strat, model_strat.predict(X_train_strat))
    test_acc = accuracy_score (y_test_strat, model_strat.predict(X_test_strat))
    
    strategy_results.append({
        'Features': n_feat,
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'Gap': train_acc - test_acc
    })

df_strategy = pd.DataFrame (strategy_results)
print("\\nTrading Strategy Complexity:")
print(df_strategy.to_string (index=False))

print("\\n💡 Trading Insight:")
print("  • Too few features: Miss important signals (high bias)")
print("  • Too many features: Overfit to noise (high variance)")
print("  • Optimal: Capture signal without noise")
print("  • Simpler strategies often more robust in live trading")
\`\`\`

## Key Takeaways

1. **Total Error = Bias² + Variance + Irreducible Error**2. **Bias**: Error from wrong assumptions (underfitting)
3. **Variance**: Error from sensitivity to training data (overfitting)
4. **Tradeoff**: Reducing one often increases the other
5. **Optimal Model**: Minimizes total error, not individual components
6. **Learning Curves**: Diagnose bias vs variance problems
7. **Regularization**: Primary tool for controlling variance
8. **Model Complexity**: Sweet spot between too simple and too complex

**Practical Tips:**
- High bias → Add complexity, features, or reduce regularization
- High variance → Add data, regularization, or reduce complexity
- Always validate on held-out test set
- Use cross-validation for robust estimates

## Further Reading

- Hastie, T., et al. (2009). "The Elements of Statistical Learning" - Chapter 7
- Domingos, P. (2000). "A unified bias-variance decomposition"
- Geman, S., et al. (1992). "Neural networks and the bias/variance dilemma"
`,
  exercises: [
    {
      prompt:
        'Create a comprehensive bias-variance analysis tool that empirically measures bias and variance for any model, generates all diagnostic plots, and provides actionable recommendations.',
      solution: `# Solution in comprehensive evaluation framework`,
    },
  ],
  quizId: 'model-evaluation-optimization-bias-variance',
  multipleChoiceId: 'model-evaluation-optimization-bias-variance-mc',
};
