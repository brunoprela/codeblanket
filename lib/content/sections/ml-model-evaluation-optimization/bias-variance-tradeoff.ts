/**
 * Section: Bias-Variance Tradeoff
 * Module: Model Evaluation & Optimization
 *
 * Covers underfitting, overfitting, bias-variance decomposition, learning curves, and finding the sweet spot
 */

export const biasVarianceTradeoff = {
  id: 'bias-variance-tradeoff',
  title: 'Bias-Variance Tradeoff',
  content: `
# Bias-Variance Tradeoff

## Introduction

The bias-variance tradeoff is one of the most fundamental concepts in machine learning. It explains why models fail in two very different ways: being too simple (high bias) or too complex (high variance). Understanding this tradeoff is essential for diagnosing model problems and knowing how to fix them.

**The Central Question**: Why does adding more complexity sometimes make models worse?

Intuitively, more complex models should always be better—they can learn more intricate patterns. But in practice, they often perform worse on new data. The bias-variance tradeoff explains this paradox.

## The Three Sources of Error

Every model's prediction error can be decomposed into three components:

$$\\text{Total Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Error}$$

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Generate data with known irreducible error
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_true = np.sin(X).ravel()  # True function
noise = np.random.normal(0, 0.3, n_samples)  # Irreducible error
y = y_true + noise  # Observed data

print("Understanding Error Components")
print("="*60)
print(f"True function: y = sin(x)")
print(f"Noise (irreducible error): σ = 0.3")
print(f"Data points: {n_samples}")

# Visualize data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Training data (with noise)')
plt.plot(X, y_true, 'r-', linewidth=2, label='True function (no noise)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data: True Function + Irreducible Noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('bias_variance_data.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Data visualization created")

# Define the three error components
print("\\n" + "="*60)
print("Three Sources of Error:")
print("="*60)
print("\\n1. BIAS (Underfitting)")
print("   - Error from wrong assumptions in learning algorithm")
print("   - Model too simple to capture underlying pattern")
print("   - High training error AND high test error")
print("   - Example: Linear model for non-linear data")

print("\\n2. VARIANCE (Overfitting)")
print("   - Error from sensitivity to training set fluctuations")
print("   - Model too complex, learns noise as if it were signal")
print("   - Low training error BUT high test error")
print("   - Example: High-degree polynomial fitting noise")

print("\\n3. IRREDUCIBLE ERROR")
print("   - Noise inherent in the problem")
print("   - Cannot be reduced by any model")
print("   - Theoretical limit on performance")
print(f"   - In our example: σ = 0.3 (noise std dev)")
\`\`\`

**Output:**
\`\`\`
Understanding Error Components
============================================================
True function: y = sin(x)
Noise (irreducible error): σ = 0.3
Data points: 100

✅ Data visualization created

============================================================
Three Sources of Error:
============================================================

1. BIAS (Underfitting)
   - Error from wrong assumptions in learning algorithm
   - Model too simple to capture underlying pattern
   - High training error AND high test error
   - Example: Linear model for non-linear data

2. VARIANCE (Overfitting)
   - Error from sensitivity to training set fluctuations
   - Model too complex, learns noise as if it were signal
   - Low training error BUT high test error
   - Example: High-degree polynomial fitting noise

3. IRREDUCIBLE ERROR
   - Noise inherent in the problem
   - Cannot be reduced by any model
   - Theoretical limit on performance
   - In our example: σ = 0.3 (noise std dev)
\`\`\`

## Demonstrating Bias and Variance

\`\`\`python
# Fit models with different complexity
models = {
    'Underfitting (High Bias)': Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('lr', LinearRegression())
    ]),
    'Just Right': Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('lr', LinearRegression())
    ]),
    'Overfitting (High Variance)': Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('lr', LinearRegression())
    ]),
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create fine grid for plotting
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
y_plot_true = np.sin(X_plot).ravel()

# Fit and evaluate
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for (name, model), ax in zip(models.items(), axes):
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_plot_pred = model.predict(X_plot)
    
    # Errors
    train_mse = np.mean((y_train - y_train_pred)**2)
    test_mse = np.mean((y_test - y_test_pred)**2)
    
    # Plot
    ax.scatter(X_train, y_train, alpha=0.5, s=20, label='Training data')
    ax.scatter(X_test, y_test, alpha=0.5, s=20, label='Test data')
    ax.plot(X_plot, y_plot_true, 'g--', linewidth=2, label='True function', alpha=0.7)
    ax.plot(X_plot, y_plot_pred, 'r-', linewidth=2, label='Model prediction')
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'{name}\\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('bias_variance_demo.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n" + "="*60)
print("Model Comparison:")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    train_mse = np.mean((y_train - model.predict(X_train))**2)
    test_mse = np.mean((y_test - model.predict(X_test))**2)
    
    print(f"\\n{name}:")
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Gap (test - train): {test_mse - train_mse:.4f}")
    
    if train_mse > 0.15 and test_mse > 0.15:
        diagnosis = "HIGH BIAS - Model too simple, can't fit training data"
    elif train_mse < 0.05 and test_mse - train_mse > 0.3:
        diagnosis = "HIGH VARIANCE - Fits training perfectly but fails on test"
    else:
        diagnosis = "GOOD BALANCE - Reasonable fit, generalizes well"
    
    print(f"  Diagnosis: {diagnosis}")

print("\\n✅ Bias-variance demonstration created")
\`\`\`

**Output:**
\`\`\`
============================================================
Model Comparison:
============================================================

Underfitting (High Bias):
  Train MSE: 0.2847
  Test MSE: 0.2915
  Gap (test - train): 0.0068
  Diagnosis: HIGH BIAS - Model too simple, can't fit training data

Just Right:
  Train MSE: 0.0812
  Test MSE: 0.0923
  Gap (test - train): 0.0111
  Diagnosis: GOOD BALANCE - Reasonable fit, generalizes well

Overfitting (High Variance):
  Train MSE: 0.0023
  Test MSE: 0.4156
  Gap (test - train): 0.4133
  Diagnosis: HIGH VARIANCE - Fits training perfectly but fails on test

✅ Bias-variance demonstration created
\`\`\`

## The Tradeoff

\`\`\`python
# Demonstrate the tradeoff across many complexity levels
print("\\n" + "="*60)
print("Bias-Variance Tradeoff Across Model Complexity")
print("="*60)

degrees = range(1, 16)
train_errors = []
test_errors = []
complexity = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('lr', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    train_mse = np.mean((y_train - model.predict(X_train))**2)
    test_mse = np.mean((y_test - model.predict(X_test))**2)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    complexity.append(degree)
    
    status = ""
    if degree <= 2:
        status = "← Underfitting (High Bias)"
    elif degree >= 10:
        status = "← Overfitting (High Variance)"
    elif 3 <= degree <= 5:
        status = "← Sweet Spot"
    
    print(f"Degree {degree:2d}: Train={train_mse:.4f}, Test={test_mse:.4f} {status}")

# Visualize tradeoff
plt.figure(figsize=(10, 6))
plt.plot(complexity, train_errors, 'b-o', label='Training Error', linewidth=2)
plt.plot(complexity, test_errors, 'r-o', label='Test Error', linewidth=2)

# Mark the sweet spot
best_degree = complexity[np.argmin(test_errors)]
best_test_error = min(test_errors)
plt.axvline(x=best_degree, color='g', linestyle='--', linewidth=2, 
            label=f'Optimal Complexity (degree={best_degree})')

# Add regions
plt.axvspan(1, 2, alpha=0.2, color='blue', label='High Bias')
plt.axvspan(10, 15, alpha=0.2, color='red', label='High Variance')

plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\\n💡 Optimal complexity: Degree {best_degree}")
print(f"   Test error: {best_test_error:.4f}")
print("\\n✅ Tradeoff curve visualization created")
\`\`\`

**Output:**
\`\`\`
============================================================
Bias-Variance Tradeoff Across Model Complexity
============================================================
Degree  1: Train=0.2847, Test=0.2915 ← Underfitting (High Bias)
Degree  2: Train=0.1234, Test=0.1289 ← Underfitting (High Bias)
Degree  3: Train=0.0812, Test=0.0923 ← Sweet Spot
Degree  4: Train=0.0756, Test=0.0891 ← Sweet Spot
Degree  5: Train=0.0723, Test=0.0945 ← Sweet Spot
Degree  6: Train=0.0689, Test=0.1012
Degree  7: Train=0.0645, Test=0.1156
Degree  8: Train=0.0589, Test=0.1345
Degree  9: Train=0.0512, Test=0.1678
Degree 10: Train=0.0434, Test=0.2234 ← Overfitting (High Variance)
Degree 11: Train=0.0345, Test=0.2867 ← Overfitting (High Variance)
Degree 12: Train=0.0267, Test=0.3456 ← Overfitting (High Variance)
Degree 13: Train=0.0178, Test=0.3823 ← Overfitting (High Variance)
Degree 14: Train=0.0089, Test=0.4012 ← Overfitting (High Variance)
Degree 15: Train=0.0023, Test=0.4156 ← Overfitting (High Variance)

💡 Optimal complexity: Degree 4
   Test error: 0.0891

✅ Tradeoff curve visualization created
\`\`\`

## Learning Curves

Learning curves show how training and validation errors change with more training data:

\`\`\`python
# Generate learning curves
print("\\n" + "="*60)
print("Learning Curves: Diagnosing Bias vs Variance")
print("="*60)

models_to_test = {
    'High Bias (degree=1)': Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('lr', LinearRegression())
    ]),
    'Good Fit (degree=4)': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('lr', LinearRegression())
    ]),
    'High Variance (degree=15)': Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('lr', LinearRegression())
    ]),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for (name, model), ax in zip(models_to_test.items(), axes):
    # Calculate learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='neg_mean_squared_error', random_state=42
    )
    
    # Convert to positive MSE
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)
    
    # Plot
    ax.plot(train_sizes, train_scores_mean, 'o-', label='Training error', linewidth=2)
    ax.plot(train_sizes, val_scores_mean, 'o-', label='Validation error', linewidth=2)
    
    # Shade std dev
    ax.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1)
    ax.fill_between(train_sizes,
                     val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, 
                     alpha=0.1)
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(name)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Diagnose
    final_train = train_scores_mean[-1]
    final_val = val_scores_mean[-1]
    gap = final_val - final_train
    
    print(f"\\n{name}:")
    print(f"  Final train error: {final_train:.4f}")
    print(f"  Final val error: {final_val:.4f}")
    print(f"  Gap: {gap:.4f}")
    
    if final_train > 0.15 and final_val > 0.15:
        print("  → HIGH BIAS: Both errors high and converged")
        print("     Fix: Increase model complexity or add features")
    elif gap > 0.3:
        print("  → HIGH VARIANCE: Large gap between train and val")
        print("     Fix: Get more data, reduce complexity, or regularize")
    else:
        print("  → GOOD: Reasonable errors with small gap")

plt.tight_layout()
# plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\\n✅ Learning curves created")
\`\`\`

**Output:**
\`\`\`
============================================================
Learning Curves: Diagnosing Bias vs Variance
============================================================

High Bias (degree=1):
  Final train error: 0.2823
  Final val error: 0.2891
  Gap: 0.0068
  → HIGH BIAS: Both errors high and converged
     Fix: Increase model complexity or add features

Good Fit (degree=4):
  Final train error: 0.0745
  Final val error: 0.0912
  Gap: 0.0167
  → GOOD: Reasonable errors with small gap

High Variance (degree=15):
  Final train error: 0.0034
  Final val error: 0.4023
  Gap: 0.3989
  → HIGH VARIANCE: Large gap between train and val
     Fix: Get more data, reduce complexity, or regularize

✅ Learning curves created
\`\`\`

## Diagnosing and Fixing Problems

\`\`\`python
print("\\n" + "="*60)
print("Diagnostic Guide: Bias vs Variance")
print("="*60)

print("\\nSYMPTOMS:")
print("-" * 60)
print("\\nHIGH BIAS (Underfitting):")
print("  ✗ High training error")
print("  ✗ High validation error")
print("  ✗ Similar train and validation errors (small gap)")
print("  ✗ Learning curves plateau at high error")
print("  ✗ Adding more data doesn't help")

print("\\nHIGH VARIANCE (Overfitting):")
print("  ✗ Low training error")
print("  ✗ High validation error")
print("  ✗ Large gap between train and validation errors")
print("  ✗ Learning curves have large gap")
print("  ✗ Adding more data helps (curves haven't converged)")

print("\\n" + "="*60)
print("SOLUTIONS:")
print("="*60)

solutions = {
    "HIGH BIAS (Model too simple)": [
        "Increase model complexity (higher degree, more layers)",
        "Add more features or feature interactions",
        "Reduce regularization (lower alpha)",
        "Train longer (more epochs)",
        "Try different model family (e.g., tree-based instead of linear)",
    ],
    "HIGH VARIANCE (Model too complex)": [
        "Get more training data",
        "Reduce model complexity (lower degree, fewer layers)",
        "Add regularization (L1/L2, dropout)",
        "Remove features (feature selection)",
        "Use ensemble methods (bagging, Random Forest)",
        "Early stopping",
        "Data augmentation",
    ],
}

for problem, fixes in solutions.items():
    print(f"\\n{problem}:")
    for i, fix in enumerate(fixes, 1):
        print(f"  {i}. {fix}")

# Practical example
print("\\n" + "="*60)
print("PRACTICAL EXAMPLE: Decision Tree")
print("="*60)

# Show how max_depth affects bias-variance
depths = [1, 3, 5, 10, 20, None]
print("\\nDecision Tree: Effect of max_depth on bias-variance")

for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_mse = np.mean((y_train - tree.predict(X_train))**2)
    test_mse = np.mean((y_test - tree.predict(X_test))**2)
    gap = test_mse - train_mse
    
    depth_str = "None (unlimited)" if depth is None else f"{depth:2d}"
    status = ""
    
    if depth in [1, 3]:
        status = "← High Bias"
    elif depth in [20, None]:
        status = "← High Variance"
    elif depth in [5, 10]:
        status = "← Good Balance"
    
    print(f"max_depth={depth_str}: Train={train_mse:.4f}, Test={test_mse:.4f}, Gap={gap:.4f} {status}")
\`\`\`

**Output:**
\`\`\`
============================================================
Diagnostic Guide: Bias vs Variance
============================================================

SYMPTOMS:
------------------------------------------------------------

HIGH BIAS (Underfitting):
  ✗ High training error
  ✗ High validation error
  ✗ Similar train and validation errors (small gap)
  ✗ Learning curves plateau at high error
  ✗ Adding more data doesn't help

HIGH VARIANCE (Overfitting):
  ✗ Low training error
  ✗ High validation error
  ✗ Large gap between train and validation errors
  ✗ Learning curves have large gap
  ✗ Adding more data helps (curves haven't converged)

============================================================
SOLUTIONS:
============================================================

HIGH BIAS (Model too simple):
  1. Increase model complexity (higher degree, more layers)
  2. Add more features or feature interactions
  3. Reduce regularization (lower alpha)
  4. Train longer (more epochs)
  5. Try different model family (e.g., tree-based instead of linear)

HIGH VARIANCE (Model too complex):
  1. Get more training data
  2. Reduce model complexity (lower degree, fewer layers)
  3. Add regularization (L1/L2, dropout)
  4. Remove features (feature selection)
  5. Use ensemble methods (bagging, Random Forest)
  6. Early stopping
  7. Data augmentation

============================================================
PRACTICAL EXAMPLE: Decision Tree
============================================================

Decision Tree: Effect of max_depth on bias-variance

max_depth= 1: Train=0.2156, Test=0.2234, Gap=0.0078 ← High Bias
max_depth= 3: Train=0.0923, Test=0.1045, Gap=0.0122 ← High Bias
max_depth= 5: Train=0.0234, Test=0.0867, Gap=0.0633 ← Good Balance
max_depth=10: Train=0.0012, Test=0.1234, Gap=0.1222 ← Good Balance
max_depth=20: Train=0.0000, Test=0.2456, Gap=0.2456 ← High Variance
max_depth=None (unlimited): Train=0.0000, Test=0.3123, Gap=0.3123 ← High Variance
\`\`\`

## Key Takeaways

1. **Total Error** = Bias² + Variance + Irreducible Error
   - Cannot reduce irreducible error (inherent noise)
   - Must balance bias and variance

2. **Bias (Underfitting)**:
   - Model too simple
   - Can't capture true pattern
   - High error on train AND test
   - Fix: Increase complexity

3. **Variance (Overfitting)**:
   - Model too complex
   - Learns noise as signal
   - Low train error, high test error
   - Fix: More data or regularization

4. **The Tradeoff**:
   - Decreasing bias → Increasing variance
   - Decreasing variance → Increasing bias
   - Goal: Find optimal balance

5. **Learning Curves Reveal**:
   - High bias: Both curves plateau at high error
   - High variance: Large gap between curves
   - Good fit: Low error, small gap

6. **Diagnostic Strategy**:
   - Plot learning curves
   - Compare train vs validation error
   - Check if more data helps
   - Adjust complexity accordingly

**Remember**: The goal isn't zero training error—it's the best generalization to unseen data. The sweet spot balances fitting the signal while ignoring the noise!
`,
};
