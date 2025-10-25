/**
 * Polynomial and Non-linear Regression Section
 */

export const polynomialregressionSection = {
  id: 'polynomial-regression',
  title: 'Polynomial & Non-linear Regression',
  content: `# Polynomial & Non-linear Regression

## Introduction

While linear regression assumes a straight-line relationship between features and targets, many real-world relationships are non-linear. Polynomial regression extends linear regression by adding polynomial terms, allowing the model to fit curved relationships while still using the same linear regression framework.

**Real-World Non-Linear Relationships**:
- House price vs. size: Larger houses don't increase linearly in value
- Marketing ROI: Diminishing returns at high ad spend
- Drug dosage response: Often follows sigmoid or polynomial curves
- Temperature effects: Many natural processes have optimal temperatures
- Stock volatility vs. market conditions: Non-linear regime changes

## Polynomial Features

### The Concept

Polynomial regression transforms features by adding powers:

For single feature \\( x \\):
\\[ y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\beta_3 x^3 + ... + \\beta_d x^d \\]

For multiple features, we can add:
- Individual powers: \\( x_1^2, x_2^2, x_1^3, ... \\)
- Interaction terms: \\( x_1 x_2, x_1 x_2^2, ... \\)

**Key Insight**: Despite using polynomial terms, this is still a **linear** model - linear in the coefficients \\( \\beta \\). We use standard linear regression algorithms after transforming features.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X.ravel()**2 - 3 * X.ravel() + 5 + np.random.randn(100) * 3

# Visualize true relationship
plt.figure (figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Non-linear Relationship')
plt.grid(True, alpha=0.3)

# Fit linear regression (underfitting)
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

plt.subplot(1, 3, 2)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_linear, 'r-', linewidth=2, label='Linear fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title (f'Linear Model (R²={r2_score (y, y_pred_linear):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Fit polynomial regression (degree 2)
poly = PolynomialFeatures (degree=2)
X_poly = poly.fit_transform(X)

model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

plt.subplot(1, 3, 3)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_poly, 'g-', linewidth=2, label='Polynomial fit (degree=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title (f'Polynomial Model (R²={r2_score (y, y_pred_poly):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Linear Model R²: {r2_score (y, y_pred_linear):.4f}")
print(f"Polynomial Model R²: {r2_score (y, y_pred_poly):.4f}")
\`\`\`

## Choosing Polynomial Degree

### The Tradeoff

**Too low degree (underfitting)**:
- Model too simple
- Poor fit on training data
- Bias error dominates

**Too high degree (overfitting)**:
- Model too complex
- Excellent fit on training data
- Poor generalization to test data
- Variance error dominates

\`\`\`python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different polynomial degrees
degrees = [1, 2, 3, 5, 10, 15]
results = []

for degree in degrees:
    # Transform features
    poly = PolynomialFeatures (degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    train_r2 = r2_score (y_train, train_pred)
    test_r2 = r2_score (y_test, test_pred)
    
    results.append({
        'degree': degree,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_features': X_train_poly.shape[1]
    })
    
    print(f"Degree {degree:2d} | Features: {X_train_poly.shape[1]:3d} | "
          f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")

# Plot the tradeoff
import pandas as pd

df_results = pd.DataFrame (results)

plt.figure (figsize=(10, 6))
plt.plot (df_results['degree'], df_results['train_r2'], 'o-', label='Training R²')
plt.plot (df_results['degree'], df_results['test_r2'], 's-', label='Test R²')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Bias-Variance Tradeoff: Model Complexity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline (y=1.0, color='r', linestyle='--', alpha=0.3, label='Perfect fit')
plt.show()

# Find optimal degree
optimal_idx = df_results['test_r2'].idxmax()
optimal_degree = df_results.loc[optimal_idx, 'degree']
print(f"\\nOptimal polynomial degree: {optimal_degree}")
\`\`\`

## Cross-Validation for Model Selection

Use cross-validation to select the best polynomial degree:

\`\`\`python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Create pipeline combining polynomial features and linear regression
def make_polynomial_pipeline (degree):
    return Pipeline([
        ('poly', PolynomialFeatures (degree=degree)),
        ('linear', LinearRegression())
    ])

# Test degrees using cross-validation
degrees = range(1, 16)
cv_scores = []

for degree in degrees:
    pipeline = make_polynomial_pipeline (degree)
    scores = cross_val_score (pipeline, X, y, cv=5, scoring='r2')
    cv_scores.append({
        'degree': degree,
        'mean_cv_score': scores.mean(),
        'std_cv_score': scores.std()
    })

df_cv = pd.DataFrame (cv_scores)

# Plot cross-validation scores
plt.figure (figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.errorbar (df_cv['degree'], df_cv['mean_cv_score'], 
             yerr=df_cv['std_cv_score'], capsize=5)
plt.xlabel('Polynomial Degree')
plt.ylabel('Cross-Validation R² Score')
plt.title('Model Selection via Cross-Validation')
plt.grid(True, alpha=0.3)

# Find best degree
best_degree = df_cv.loc[df_cv['mean_cv_score'].idxmax(), 'degree']
print(f"Best polynomial degree (CV): {best_degree}")

# Visualize predictions for different degrees
plt.subplot(1, 2, 2)
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
plt.scatter(X, y, alpha=0.3, label='Data')

for degree in [1, 2, 5, 10]:
    pipeline = make_polynomial_pipeline (degree)
    pipeline.fit(X, y)
    y_plot = pipeline.predict(X_plot)
    plt.plot(X_plot, y_plot, label=f'Degree {degree}', linewidth=2)

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Fits (Different Degrees)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Interaction Terms

For multiple features, we can include interaction terms:

\\[ y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\beta_3 x_1 x_2 + ... \\]

\`\`\`python
# Example: House price with interaction terms
np.random.seed(42)
n = 200

# Features: size and location quality
size = np.random.uniform(1000, 3000, n)
location = np.random.uniform(1, 10, n)  # 1-10 scale

# Price depends on size, location, AND their interaction
# (nice location makes size more valuable)
price = (
    100000 +
    80 * size +
    20000 * location +
    15 * size * location +  # Interaction effect
    np.random.randn (n) * 20000
)

X = np.column_stack([size, location])

# Model without interactions
model_no_interact = LinearRegression()
model_no_interact.fit(X, price)
pred_no_interact = model_no_interact.predict(X)
r2_no_interact = r2_score (price, pred_no_interact)

# Model with interactions (degree=2 includes x1*x2)
poly = PolynomialFeatures (degree=2, interaction_only=False)
X_poly = poly.fit_transform(X)
model_with_interact = LinearRegression()
model_with_interact.fit(X_poly, price)
pred_with_interact = model_with_interact.predict(X_poly)
r2_with_interact = r2_score (price, pred_with_interact)

print("Feature names:", poly.get_feature_names_out(['size', 'location']))
print(f"\\nWithout interactions R²: {r2_no_interact:.4f}")
print(f"With interactions R²: {r2_with_interact:.4f}")
print(f"\\nCoefficients (with interactions):")
for name, coef in zip (poly.get_feature_names_out(), model_with_interact.coef_):
    print(f"  {name:20s}: {coef:12,.2f}")
\`\`\`

## Other Non-Linear Transformations

### Logarithmic Transformation

Useful for:
- Relationships with diminishing returns
- Right-skewed distributions
- Multiplicative relationships

\`\`\`python
# Example: Log transformation for exponential growth
np.random.seed(42)
X_exp = np.linspace(1, 10, 100).reshape(-1, 1)
y_exp = 10 * np.exp(0.3 * X_exp.ravel()) + np.random.randn(100) * 10

# Linear model (poor fit)
model_linear = LinearRegression()
model_linear.fit(X_exp, y_exp)
y_pred_linear = model_linear.predict(X_exp)

# Log-transformed target
y_log = np.log (y_exp)
model_log = LinearRegression()
model_log.fit(X_exp, y_log)
y_pred_log_transformed = np.exp (model_log.predict(X_exp))

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_exp, y_exp, alpha=0.5, label='Data')
plt.plot(X_exp, y_pred_linear, 'r-', linewidth=2, 
         label=f'Linear (R²={r2_score (y_exp, y_pred_linear):.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model on Exponential Data')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_exp, y_exp, alpha=0.5, label='Data')
plt.plot(X_exp, y_pred_log_transformed, 'g-', linewidth=2,
         label=f'Log-transformed (R²={r2_score (y_exp, y_pred_log_transformed):.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Model with Log-Transformed Target')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

### Square Root Transformation

Useful for count data and variance stabilization:

\`\`\`python
# Example: Count data with increasing variance
X_count = np.linspace(0, 10, 100).reshape(-1, 1)
y_count = 2 * X_count.ravel()**2 + np.random.poisson(5 * X_count.ravel())

# Model with square root transformation
y_sqrt = np.sqrt (y_count)
model_sqrt = LinearRegression()
model_sqrt.fit(X_count, y_sqrt)
y_pred_sqrt = model_sqrt.predict(X_count)**2

print(f"R² with sqrt transformation: {r2_score (y_count, y_pred_sqrt):.4f}")
\`\`\`

## Splines and Piecewise Polynomials

Splines fit different polynomials in different regions:

\`\`\`python
from sklearn.preprocessing import SplineTransformer

# Generate data with different behavior in different regions
np.random.seed(42)
X_spline = np.linspace(0, 10, 200).reshape(-1, 1)
y_spline = np.where(X_spline.ravel() < 5, 
                     X_spline.ravel()**2,
                     50 - 2*X_spline.ravel())
y_spline += np.random.randn(200) * 2

# Fit spline model
spline = SplineTransformer (n_knots=5, degree=3)
X_spline_features = spline.fit_transform(X_spline)

model_spline = LinearRegression()
model_spline.fit(X_spline_features, y_spline)
y_pred_spline = model_spline.predict(X_spline_features)

# Compare with polynomial
poly = PolynomialFeatures (degree=5)
X_poly_global = poly.fit_transform(X_spline)
model_poly_global = LinearRegression()
model_poly_global.fit(X_poly_global, y_spline)
y_pred_poly = model_poly_global.predict(X_poly_global)

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_spline, y_spline, alpha=0.3, label='Data')
plt.plot(X_spline, y_pred_poly, 'r-', linewidth=2, 
         label=f'Global Polynomial (R²={r2_score (y_spline, y_pred_poly):.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Global Polynomial (Degree 5)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_spline, y_spline, alpha=0.3, label='Data')
plt.plot(X_spline, y_pred_spline, 'g-', linewidth=2,
         label=f'Spline (R²={r2_score (y_spline, y_pred_spline):.3f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Spline Regression')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Real-World Example: Marketing ROI

\`\`\`python
# Realistic marketing scenario: diminishing returns
np.random.seed(42)
n_campaigns = 100

# Ad spend (in thousands)
ad_spend = np.random.uniform(10, 200, n_campaigns).reshape(-1, 1)

# Revenue with diminishing returns (log relationship)
# ROI decreases as spend increases
revenue = 50 + 30 * np.log (ad_spend.ravel()) + np.random.randn (n_campaigns) * 10

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    ad_spend, revenue, test_size=0.2, random_state=42
)

# Compare models
models = {
    'Linear': Pipeline([
        ('linear', LinearRegression())
    ]),
    'Polynomial (degree=2)': Pipeline([
        ('poly', PolynomialFeatures (degree=2)),
        ('linear', LinearRegression())
    ]),
    'Log-transformed': Pipeline([
        ('log', FunctionTransformer (np.log, validate=True)),
        ('linear', LinearRegression())
    ]),
}

results = []

plt.figure (figsize=(15, 5))

for idx, (name, model) in enumerate (models.items(), 1):
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score (y_train, y_train_pred)
    test_r2 = r2_score (y_test, y_test_pred)
    test_rmse = np.sqrt (mean_squared_error (y_test, y_test_pred))
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test RMSE': test_rmse
    })
    
    # Plot
    plt.subplot(1, 3, idx)
    plt.scatter(X_train, y_train, alpha=0.5, label='Train')
    plt.scatter(X_test, y_test, alpha=0.5, label='Test')
    
    # Prediction line
    X_plot = np.linspace(10, 200, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Fit')
    
    plt.xlabel('Ad Spend ($1000s)')
    plt.ylabel('Revenue ($1000s)')
    plt.title (f'{name}\\nTest R² = {test_r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print results
print("\\nModel Comparison:")
print(pd.DataFrame (results).to_string (index=False))

# Business insights
best_model = models['Log-transformed']
print("\\n--- Business Insights (Log-transformed Model) ---")

# Marginal ROI at different spend levels
spend_levels = np.array([50, 100, 150, 200]).reshape(-1, 1)
base_revenue = best_model.predict (spend_levels)
marginal_revenue = best_model.predict (spend_levels + 10) - base_revenue

for spend, marginal in zip (spend_levels.ravel(), marginal_revenue):
    roi = (marginal / 10) * 100
    print(f"At \${spend}k spend: Additional $10k returns \${marginal:.2f}k (ROI: { roi: .1f }%) ")
\`\`\`

## Best Practices

1. **Start simple**: Try linear regression first as baseline
2. **Visualize**: Plot data to identify non-linear patterns
3. **Use domain knowledge**: Choose transformations that make sense
4. **Cross-validate**: Select degree/complexity using validation set
5. **Watch for overfitting**: Higher degree ≠ better model
6. **Feature scaling**: Important for polynomial features
7. **Interpretability**: Complex polynomials hard to interpret
8. **Regularization**: Often needed with high-degree polynomials

## Common Pitfalls

1. **Extrapolation danger**: Polynomials behave wildly outside training range
2. **Multicollinearity**: High-degree polynomials create correlated features
3. **Overfitting**: Easy to fit noise with high degrees
4. **Feature explosion**: Number of features grows rapidly with degree
5. **Numerical instability**: Very high/low powers can cause issues

## Summary

Polynomial and non-linear regression extends linear regression to curved relationships:
- **Polynomial features**: Add powers of features (x², x³, etc.)
- **Interaction terms**: Capture combined effects (x₁x₂)
- **Transformations**: Log, sqrt, and other functions
- **Splines**: Piecewise polynomials for local fitting
- **Model selection**: Use cross-validation to choose complexity
- **Regularization**: Often needed to prevent overfitting

**Key insight**: Many non-linear relationships can be modeled using linear regression with engineered features!

Next, we'll explore regularization techniques (Ridge and Lasso) to prevent overfitting in these complex models.
`,
  codeExample: `# Complete Polynomial Regression Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Generate realistic dataset: Temperature vs. Crop Yield
np.random.seed(42)
n_samples = 200

# Temperature (Celsius)
temperature = np.random.uniform(10, 35, n_samples)

# Crop yield has optimal temperature (quadratic relationship)
# Optimal around 22°C, decreases on both sides
optimal_temp = 22
yield_crop = (
    100 -  # Maximum yield
    0.5 * (temperature - optimal_temp)**2 +  # Quadratic penalty from optimal
    np.random.randn (n_samples) * 5  # Noise
)

# Create DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'yield': yield_crop
})

print("="*70)
print("POLYNOMIAL REGRESSION: CROP YIELD PREDICTION")
print("="*70)

print(f"\\nDataset: {len (df)} observations")
print(f"\\nYield statistics:")
print(df['yield'].describe())

# Visualize data
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter (df['temperature'], df['yield'], alpha=0.6)
plt.xlabel('Temperature (°C)')
plt.ylabel('Crop Yield (tons/hectare)')
plt.title('Temperature vs. Crop Yield\\n(Non-linear Relationship)')
plt.grid(True, alpha=0.3)

# Prepare data
X = df[['temperature']]
y = df['yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Test different polynomial degrees
print("\\n" + "="*70)
print("MODEL COMPARISON: DIFFERENT POLYNOMIAL DEGREES")
print("="*70)

degrees = [1, 2, 3, 5, 8]
models = {}
results = []

for degree in degrees:
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures (degree=degree)),
        ('regressor', LinearRegression())
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    models[degree] = pipeline
    
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Cross-validation
    cv_scores = cross_val_score (pipeline, X_train, y_train, cv=5, scoring='r2')
    
    # Calculate metrics
    train_r2 = r2_score (y_train, y_train_pred)
    test_r2 = r2_score (y_test, y_test_pred)
    test_rmse = np.sqrt (mean_squared_error (y_test, y_test_pred))
    test_mae = mean_absolute_error (y_test, y_test_pred)
    cv_mean = cv_scores.mean()
    
    # Get number of features after transformation
    poly_temp = PolynomialFeatures (degree=degree)
    n_features = poly_temp.fit_transform(X_train).shape[1]
    
    results.append({
        'Degree': degree,
        'Features': n_features,
        'Train R²': train_r2,
        'CV R²': cv_mean,
        'Test R²': test_r2,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Overfit Gap': train_r2 - test_r2
    })

# Display results
df_results = pd.DataFrame (results)
print("\\n", df_results.to_string (index=False))

# Find best model
best_idx = df_results['CV R²'].idxmax()
best_degree = df_results.loc[best_idx, 'Degree']

print(f"\\n{'='*70}")
print(f"BEST MODEL: Polynomial Degree = {best_degree}")
print(f"{'='*70}")
print(f"Cross-Validation R²: {df_results.loc[best_idx, 'CV R²']:.4f}")
print(f"Test R²: {df_results.loc[best_idx, 'Test R²']:.4f}")
print(f"Test RMSE: {df_results.loc[best_idx, 'Test RMSE']:.2f}")
print(f"Overfitting gap: {df_results.loc[best_idx, 'Overfit Gap']:.4f}")

# Visualize model comparisons
plt.subplot(1, 2, 2)
plt.plot (df_results['Degree'], df_results['Train R²'], 'o-', label='Train R²')
plt.plot (df_results['Degree'], df_results['CV R²'], 's-', label='CV R²')
plt.plot (df_results['Degree'], df_results['Test R²'], '^-', label='Test R²')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Complexity vs. Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline (x=best_degree, color='r', linestyle='--', alpha=0.5, label='Best')

plt.tight_layout()
plt.show()

# Detailed visualization of different fits
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

X_plot = np.linspace(10, 35, 200).reshape(-1, 1)

for idx, degree in enumerate (degrees):
    ax = axes[idx]
    
    # Get model
    model = models[degree]
    
    # Predictions
    y_plot = model.predict(X_plot)
    y_test_pred = model.predict(X_test)
    
    # Plot
    ax.scatter(X_train, y_train, alpha=0.4, s=30, label='Train')
    ax.scatter(X_test, y_test, alpha=0.6, s=30, color='red', label='Test')
    ax.plot(X_plot, y_plot, 'g-', linewidth=2.5, label='Fitted curve')
    
    # Metrics
    test_r2 = r2_score (y_test, y_test_pred)
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Crop Yield')
    ax.set_title (f'Degree {degree} | Test R² = {test_r2:.3f}')
    ax.legend (fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 35)
    
    # Highlight if best model
    if degree == best_degree:
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)

# Hide extra subplot
axes[-1].axis('off')

plt.suptitle('Polynomial Regression: Different Degrees', fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# Practical insights with best model
print("\\n" + "="*70)
print("PRACTICAL INSIGHTS")
print("="*70)

best_model = models[best_degree]

# Find optimal temperature
temp_range = np.linspace(10, 35, 200).reshape(-1, 1)
predicted_yields = best_model.predict (temp_range)
optimal_idx = predicted_yields.argmax()
optimal_temp_pred = temp_range[optimal_idx][0]
max_yield_pred = predicted_yields[optimal_idx]

print(f"\\nOptimal temperature: {optimal_temp_pred:.1f}°C")
print(f"Predicted maximum yield: {max_yield_pred:.1f} tons/hectare")

# Yield at different temperatures
print("\\nPredicted yields at different temperatures:")
for temp in [15, 20, 22, 25, 30]:
    pred_yield = best_model.predict([[temp]])[0]
    print(f"  {temp}°C: {pred_yield:.1f} tons/hectare")

# Test predictions
print("\\n" + "="*70)
print("EXAMPLE TEST PREDICTIONS")
print("="*70)

sample_indices = np.random.choice (len(X_test), 5, replace=False)
print(f"\\n{'Temperature':>12s} {'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
print("-" * 45)

for idx in sample_indices:
    temp = X_test.iloc[idx, 0]
    actual = y_test.iloc[idx]
    predicted = best_model.predict(X_test.iloc[[idx]])[0]
    error = predicted - actual
    
    print(f"{temp:>12.1f} {actual:>10.2f} {predicted:>10.2f} {error:>10.2f}")

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Best model: Polynomial degree {best_degree}
- Test R²: {df_results.loc[best_idx, 'Test R²']:.3f}
- Test RMSE: {df_results.loc[best_idx, 'Test RMSE']:.2f} tons/hectare
- Optimal temperature: {optimal_temp_pred:.1f}°C

The polynomial model successfully captures the non-linear relationship
between temperature and crop yield, identifying the optimal growing
temperature for maximum yield.

Key finding: Crop yield follows an inverted parabola, with optimal
performance around {optimal_temp_pred:.1f}°C. Both lower and higher
temperatures reduce yield.
""")
`,
};
