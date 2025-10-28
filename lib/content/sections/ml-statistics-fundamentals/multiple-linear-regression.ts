/**
 * Multiple Linear Regression Section
 */

export const multiplelinearregressionSection = {
  id: 'multiple-linear-regression',
  title: 'Multiple Linear Regression',
  content: `# Multiple Linear Regression

## Introduction

Multiple linear regression extends simple linear regression to model the relationship between **multiple predictors** and a response variable:

\\[ Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + ... + \\beta_p X_p + \\epsilon \\]

This is the workhorse of statistical modeling and the foundation of many ML algorithms.

**Why multiple regression?**
- Real-world: outcomes depend on multiple factors
- Control for confounders
- Compare effects of different predictors
- Foundation for GLMs, neural networks
- Interpretable predictions

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)

def demonstrate_multiple_regression():
    """Multiple regression with two predictors"""
    
    # Generate data
    n = 200
    X1 = np.random.rand (n) * 10  # e.g., years experience
    X2 = np.random.rand (n) * 10  # e.g., education level
    
    # Y depends on both (e.g., salary)
    Y = 30000 + 2000*X1 + 3000*X2 + np.random.normal(0, 5000, n)
    
    # Create design matrix
    X = np.column_stack([X1, X2])
    
    # Fit model
    model = LinearRegression()
    model.fit(X, Y)
    
    Y_pred = model.predict(X)
    r2 = model.score(X, Y)
    
    print("=== Multiple Linear Regression ===")
    print(f"Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}*X1 + {model.coef_[1]:.2f}*X2")
    print(f"\\nTrue equation: Y = 30000 + 2000*X1 + 3000*X2")
    print(f"R² = {r2:.4f}")
    
    # 3D visualization
    fig = plt.figure (figsize=(12, 5))
    
    # 3D scatter
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X1, X2, Y, alpha=0.5, s=20)
    
    # Create regression plane
    x1_range = np.linspace(X1.min(), X1.max(), 20)
    x2_range = np.linspace(X2.min(), X2.max(), 20)
    X1_grid, X2_grid = np.meshgrid (x1_range, x2_range)
    Y_grid = model.intercept_ + model.coef_[0]*X1_grid + model.coef_[1]*X2_grid
    
    ax1.plot_surface(X1_grid, X2_grid, Y_grid, alpha=0.3, cmap='viridis')
    ax1.set_xlabel('X1 (Experience)')
    ax1.set_ylabel('X2 (Education)')
    ax1.set_zlabel('Y (Salary)')
    ax1.set_title('Multiple Regression: 3D View')
    
    # Residual plot
    ax2 = fig.add_subplot(122)
    residuals = Y - Y_pred
    ax2.scatter(Y_pred, residuals, alpha=0.5)
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiple_regression_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

demonstrate_multiple_regression()
\`\`\`

## Matrix Formulation

\\[ \\mathbf{Y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon} \\]

\\[ \\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{Y} \\]

\`\`\`python
def ols_matrix_form(X, y):
    """OLS using matrix formulation"""
    
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones (len(X)), X])
    
    # β = (X'X)^(-1) X'y
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    beta = np.linalg.solve(XtX, Xty)  # More stable than inverse
    
    # Predictions
    y_pred = X_with_intercept @ beta
    
    # Residuals
    residuals = y - y_pred
    
    # MSE
    n, p = X_with_intercept.shape
    mse = np.sum (residuals**2) / (n - p)
    
    # Coefficient standard errors
    cov_matrix = mse * np.linalg.inv(XtX)
    se = np.sqrt (np.diag (cov_matrix))
    
    # t-statistics
    t_stats = beta / se
    
    # p-values
    p_values = 2 * (1 - stats.t.cdf (np.abs (t_stats), df=n-p))
    
    # R²
    ss_total = np.sum((y - y.mean())**2)
    ss_residual = np.sum (residuals**2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Adjusted R²
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
    
    return {
        'coefficients': beta,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'predictions': y_pred,
        'residuals': residuals
    }

# Example
n = 100
X = np.random.randn (n, 3)  # 3 predictors
true_beta = np.array([5, 2, -3, 1.5])  # intercept + 3 slopes
y = true_beta[0] + X @ true_beta[1:] + np.random.normal(0, 1, n)

results = ols_matrix_form(X, y)

print("=== OLS Matrix Formulation ===")
print(f"Coefficients: {results['coefficients']}")
print(f"True values: {true_beta}")
print(f"R²: {results['r_squared']:.4f}")
print(f"Adjusted R²: {results['adj_r_squared']:.4f}")
\`\`\`

## Coefficient Interpretation

**Holding other variables constant!**

\`\`\`python
def interpret_coefficients():
    """Demonstrate coefficient interpretation"""
    
    # Real estate example
    n = 200
    data = pd.DataFrame({
        'sqft': np.random.randint(800, 3000, n),
        'bedrooms': np.random.choice([1, 2, 3, 4], n),
        'age': np.random.randint(0, 50, n)
    })
    
    # Price depends on all three
    data['price'] = (
        50000 + 
        150 * data['sqft'] + 
        20000 * data['bedrooms'] - 
        1000 * data['age'] +
        np.random.normal(0, 30000, n)
    )
    
    # Fit model
    X = data[['sqft', 'bedrooms', 'age']]
    y = data['price']
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    print("=== Coefficient Interpretation ===")
    print(model.summary())
    print()
    print("Interpretation:")
    print(f"Intercept: \\$\{model.params['const']:.0f}")
print(f"  → Base price with sqft=0, bedrooms=0, age=0")
print()
print(f"sqft: \\$\{model.params['sqft']:.2f} per sq ft")
print(f"  → Holding bedrooms and age constant,")
print(f"    each additional sq ft increases price by \\$\{model.params['sqft']:.2f}")
print()
print(f"bedrooms: \\$\{model.params['bedrooms']:.0f} per bedroom")
print(f"  → Holding sqft and age constant,")
print(f"    each additional bedroom increases price by \\$\{model.params['bedrooms']:.0f}")
print()
print(f"age: \\$\{model.params['age']:.2f} per year")
print(f"  → Holding sqft and bedrooms constant,")
print(f"    each year older decreases price by \${-model.params['age']:.2f}")

interpret_coefficients()
\`\`\`

## Multicollinearity (VIF)

When predictors are highly correlated:

\`\`\`python
def analyze_multicollinearity():
    """Detect and handle multicollinearity"""
    
    n = 100
    X1 = np.random.randn (n)
    X2 = np.random.randn (n)
    X3 = 0.95 * X1 + 0.05 * np.random.randn (n)  # Highly correlated with X1!
    
    X = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})
    y = 2*X1 + 3*X2 + 1*X3 + np.random.randn (n)
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    
    print("=== Multicollinearity Analysis ===")
    print(vif_data)
    print()
    print("VIF Interpretation:")
    print("  VIF = 1: No correlation with other features")
    print("  VIF < 5: Acceptable")
    print("  VIF 5-10: Moderate multicollinearity")
    print("  VIF > 10: Severe multicollinearity (problematic!)")
    print()
    
    # Fit model with multicollinearity
    model_with = sm.OLS(y, sm.add_constant(X)).fit()
    
    # Fit without X3
    X_reduced = X[['X1', 'X2']]
    model_without = sm.OLS(y, sm.add_constant(X_reduced)).fit()
    
    print("With X3 (multicollinear):")
    print(f"  R² = {model_with.rsquared:.4f}")
    print(f"  X1 coef: {model_with.params['X1']:.4f} (SE: {model_with.bse['X1']:.4f})")
    print()
    print("Without X3 (reduced):")
    print(f"  R² = {model_without.rsquared:.4f}")
    print(f"  X1 coef: {model_without.params['X1']:.4f} (SE: {model_without.bse['X1']:.4f})")
    print()
    print("Note: Removing X3 barely affects R² but reduces SE!")

analyze_multicollinearity()
\`\`\`

## Adjusted R²

\\[ R^2_{adj} = 1 - \\frac{(1-R^2)(n-1)}{n-p-1} \\]

Penalizes model complexity!

\`\`\`python
def compare_r_squared_adjusted():
    """Show why adjusted R² is better for model comparison"""
    
    n = 100
    X_base = np.random.randn (n, 2)  # 2 useful predictors
    y = X_base @ np.array([2, 3]) + np.random.randn (n)
    
    results = []
    
    # Add noise predictors
    for n_noise in [0, 2, 5, 10]:
        X_noise = np.random.randn (n, n_noise)
        X = np.column_stack([X_base, X_noise]) if n_noise > 0 else X_base
        
        model = LinearRegression()
        model.fit(X, y)
        
        r2 = model.score(X, y)
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        results.append({
            'n_predictors': p,
            'R²': r2,
            'Adj R²': adj_r2
        })
    
    results_df = pd.DataFrame (results)
    print("=== R² vs Adjusted R² ===")
    print(results_df.to_string (index=False))
    print()
    print("Notice:")
    print("  • R² always increases with more predictors (even noise!)")
    print("  • Adjusted R² can decrease (penalizes complexity)")
    print("  • Use Adjusted R² for model comparison")

compare_r_squared_adjusted()
\`\`\`

## F-Test for Overall Significance

\\[ F = \\frac{(SS_{total} - SS_{residual})/p}{SS_{residual}/(n-p-1)} \\]

Tests if ANY predictor is useful.

\`\`\`python
def overall_f_test():
    """F-test for overall model significance"""
    
    n = 100
    X = np.random.randn (n, 3)
    y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn (n)
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    print("=== F-Test for Overall Significance ===")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"P-value: {model.f_pvalue:.6f}")
    print()
    
    if model.f_pvalue < 0.05:
        print("✓ At least one predictor is significant")
    else:
        print("✗ No predictors are significant")
    
    print("\\nIndividual predictor p-values:")
    for name, pval in model.pvalues.items():
        sig = "✓" if pval < 0.05 else "✗"
        print(f"  {name}: p={pval:.6f} {sig}")

overall_f_test()
\`\`\`

## Feature Selection Methods

\`\`\`python
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor

def feature_selection_comparison():
    """Compare feature selection methods"""
    
    # Generate data with some useless features
    n = 200
    X = np.random.randn (n, 10)
    # Only first 3 features are useful
    y = 2*X[:, 0] + 3*X[:, 1] - 1.5*X[:, 2] + np.random.randn (n)
    
    feature_names = [f'X{i}' for i in range(10)]
    
    # Method 1: Univariate (F-statistic)
    selector_univariate = SelectKBest (f_regression, k=5)
    selector_univariate.fit(X, y)
    selected_univariate = [feature_names[i] for i in selector_univariate.get_support (indices=True)]
    
    # Method 2: Recursive Feature Elimination
    model = LinearRegression()
    selector_rfe = RFE(model, n_features_to_select=5)
    selector_rfe.fit(X, y)
    selected_rfe = [feature_names[i] for i in selector_rfe.get_support (indices=True)]
    
    # Method 3: Feature importance (Random Forest)
    rf = RandomForestRegressor (n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    selected_rf = importances.head(5)['Feature'].tolist()
    
    print("=== Feature Selection Methods ===")
    print(f"True important features: X0, X1, X2")
    print()
    print(f"Univariate (F-test): {selected_univariate}")
    print(f"RFE (Recursive): {selected_rfe}")
    print(f"Random Forest importance: {selected_rf}")
    print()
    print("Feature Importances (Random Forest):")
    print(importances.to_string (index=False))

feature_selection_comparison()
\`\`\`

## ML Application: Real Estate Prediction

\`\`\`python
def real_estate_model():
    """Complete ML workflow with multiple regression"""
    
    # Generate realistic real estate data
    n = 500
    data = pd.DataFrame({
        'sqft': np.random.randint(600, 4000, n),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n),
        'age': np.random.randint(0, 100, n),
        'garage': np.random.choice([0, 1, 2, 3], n),
        'pool': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    
    # Price depends on features (with realistic coefficients)
    data['price'] = (
        50000 +
        150 * data['sqft'] +
        20000 * data['bedrooms'] +
        15000 * data['bathrooms'] -
        500 * data['age'] +
        10000 * data['garage'] +
        25000 * data['pool'] +
        np.random.normal(0, 30000, n)
    )
    
    # Split data
    X = data.drop('price', axis=1)
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    
    # Predictions
    y_pred = model.predict (sm.add_constant(X_test))
    
    # Evaluate
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score (y_test, y_pred)
    mae = mean_absolute_error (y_test, y_pred)
    rmse = np.sqrt (mean_squared_error (y_test, y_pred))
    
    print("=== Real Estate Price Prediction ===")
    print(model.summary())
    print()
    print("Test Set Performance:")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = \\$\{mae:,.0f}")
print(f"  RMSE = \\$\{rmse:,.0f}")
    
    # Feature importance (by absolute t - statistic)
importance = pd.DataFrame({
    'Feature': model.params.index[1:],  # Exclude intercept
        'Coefficient': model.params.values[1:],
    '|t-stat|': np.abs (model.tvalues.values[1:])
}).sort_values('|t-stat|', ascending = False)

print("\\nFeature Importance (by |t-statistic|):")
print(importance.to_string (index = False))

real_estate_model()
\`\`\`

## Key Takeaways

1. **Multiple regression**: Y = β₀ + β₁X₁ + ... + βₚXₚ + ε
2. **Coefficient interpretation**: Effect of Xᵢ **holding others constant**3. **R²**: Proportion of variance explained
4. **Adjusted R²**: Penalizes complexity, use for model comparison
5. **Multicollinearity**: High VIF (>10) indicates correlated predictors
6. **F-test**: Tests if any predictor is useful
7. **Feature selection**: Univariate, RFE, or importance-based
8. **Matrix form**: β = (X'X)⁻¹X'Y

## Connection to Machine Learning

- **Foundation of GLMs**: Logistic regression extends multiple regression
- **Neural networks**: First layer is essentially multiple regression
- **Feature engineering**: Interactions, polynomials extend basic model
- **Regularization**: Ridge/Lasso prevent overfitting in high dimensions
- **Interpretability**: Linear models are highly interpretable
- **Baseline**: Always try multiple regression first

Multiple regression bridges statistics and machine learning - master it to understand both fields!
`,
};
