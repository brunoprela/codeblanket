export const featureImportanceInterpretation = {
  title: 'Feature Importance & Interpretation',
  content: `
# Feature Importance & Interpretation

## Introduction

Understanding which features drive model predictions is crucial for:
- **Trust**: Verify the model uses sensible features
- **Debugging**: Identify data leakage or spurious correlations  
- **Feature Engineering**: Know which features to invest in
- **Regulatory Compliance**: Explain decisions (GDPR, fair lending)
- **Domain Insights**: Learn about the problem

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Feature Importance & Model Interpretation")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {feature_names}")
\`\`\`

## Built-in Feature Importance (Tree-based Models)

\`\`\`python
# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances_rf = rf.feature_importances_
indices = np.argsort(importances_rf)[::-1]

print("\\nRandom Forest Feature Importances:")
for i in range(len(feature_names)):
    print(f"  {i+1}. {feature_names[indices[i]]}: {importances_rf[indices[i]]:.4f}")

# Visualize
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances_rf)), importances_rf[indices])
plt.xticks(range(len(importances_rf)), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances', fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150)
print("\\nSaved to 'feature_importances.png'")
\`\`\`

## Permutation Importance

\`\`\`python
# Permutation importance works for any model
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42)

print("\\nPermutation Importance:")
perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
for i in range(len(feature_names)):
    idx = perm_indices[i]
    print(f"  {i+1}. {feature_names[idx]}: "
          f"{perm_importance.importances_mean[idx]:.4f} "
          f"(+/- {perm_importance.importances_std[idx]:.4f})")
\`\`\`

## SHAP Values

\`\`\`python
try:
    import shap
    
    # Create explainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    print("\\nSHAP summary plot saved")
    
except ImportError:
    print("\\nInstall SHAP: pip install shap")
\`\`\`

## Linear Model Coefficients

\`\`\`python
# Linear models have interpretable coefficients
lr = LinearRegression()
lr.fit(X_train, y_train)

print("\\nLinear Regression Coefficients:")
coef_indices = np.argsort(np.abs(lr.coef_))[::-1]
for i in range(len(feature_names)):
    idx = coef_indices[i]
    print(f"  {feature_names[idx]}: {lr.coef_[idx]:+.4f}")
\`\`\`

## Key Takeaways

1. **Tree Importance**: Based on splitting (may be biased)
2. **Permutation**: Model-agnostic, robust
3. **SHAP**: Individual prediction explanations
4. **Coefficients**: Direct for linear models

**Best Practices:**
- Use multiple methods
- Check consistency across methods  
- Validate with domain knowledge
- Watch for multicollinearity

## Further Reading

- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions" (SHAP)
- Breiman, L. (2001). "Random forests" (Built-in importance)
`,
  exercises: [
    { prompt: 'Compare importance methods', solution: '# Implementation' },
  ],
  quizId: 'model-evaluation-optimization-feature-importance',
  multipleChoiceId: 'model-evaluation-optimization-feature-importance-mc',
};

export const modelDebugging = {
  title: 'Model Debugging',
  content: `
# Model Debugging

## Introduction

Models fail for many reasons. Systematic debugging helps identify and fix issues.

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data  
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Model Debugging")
print("="*70)
\`\`\`

## Common Issues Checklist

**1. Data Issues:**
- Missing values
- Data leakage
- Distribution shifts
- Label errors

**2. Model Issues:**
- Underfitting (high bias)
- Overfitting (high variance)
- Wrong hyperparameters
- Poor feature engineering

**3. Evaluation Issues:**
- Wrong metric
- Data splitting errors
- Test set leakage

## Debugging Workflow

\`\`\`python
def debug_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model debugging."""
    
    # 1. Check data
    print("\\n1. DATA CHECKS:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Missing values: {np.isnan(X_train).sum()}")
    
    # 2. Train and evaluate
    print("\\n2. MODEL PERFORMANCE:")
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    print(f"  Train MSE: {train_mse:.2f}")
    print(f"  Test MSE: {test_mse:.2f}")
    print(f"  Gap: {test_mse - train_mse:.2f}")
    
    # 3. Diagnose
    if test_mse > train_mse * 1.5:
        print("\\n  ⚠️  HIGH VARIANCE (Overfitting)")
        print("  Solutions:")
        print("    - Add more data")
        print("    - Reduce model complexity")
        print("    - Add regularization")
    elif train_mse > 100:  # Example threshold
        print("\\n  ⚠️  HIGH BIAS (Underfitting)")  
        print("  Solutions:")
        print("    - Increase model complexity")
        print("    - Add more features")
        print("    - Reduce regularization")
    else:
        print("\\n  ✓ Model appears well-calibrated")
    
    # 4. Residual analysis
    residuals = y_test - test_pred
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_debugging.png', dpi=150)
    print("\\n  Diagnostic plots saved to 'model_debugging.png'")

# Run debugging
model = RandomForestRegressor(n_estimators=100, random_state=42)
debug_model(model, X_train, X_test, y_train, y_test)
\`\`\`

## Key Takeaways

1. **Systematic Approach**: Follow debugging checklist
2. **Check Data First**: Most issues are data problems
3. **Visualize**: Plots reveal patterns
4. **Compare Baselines**: Is model better than simple baseline?
5. **Track Everything**: Log all experiments

**Common Fixes:**
- Overfitting → Regularization, more data, simpler model
- Underfitting → More complexity, better features
- Poor generalization → Better validation strategy
- Slow training → Feature selection, sampling

## Further Reading

- Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems"
`,
  exercises: [{ prompt: 'Build debugging tool', solution: '# Implementation' }],
  quizId: 'model-evaluation-optimization-model-debugging',
  multipleChoiceId: 'model-evaluation-optimization-model-debugging-mc',
};
