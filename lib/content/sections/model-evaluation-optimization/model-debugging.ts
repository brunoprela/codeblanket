export const modelDebugging = {
  title: 'Model Debugging',
  content: `
# Model Debugging

## Introduction

Models fail for many reasons. Systematic debugging helps identify and fix issues quickly and effectively.

**Why Models Fail:**
- Poor data quality
- Incorrect preprocessing
- Data leakage
- Wrong model choice
- Hyperparameter issues
- Evaluation errors

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data  
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Model Debugging")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
\`\`\`

## Debugging Checklist

\`\`\`python
def comprehensive_debug_checklist(X_train, X_test, y_train, y_test, model=None):
    """Run through complete debugging checklist."""
    
    print("\\n" + "="*70)
    print("COMPREHENSIVE MODEL DEBUGGING CHECKLIST")
    print("="*70)
    
    issues_found = []
    
    # 1. DATA QUALITY CHECKS
    print("\\n1. DATA QUALITY CHECKS")
    print("-"*70)
    
    # Check for missing values
    train_missing = np.isnan(X_train).sum()
    test_missing = np.isnan(X_test).sum()
    print(f"  Missing values - Train: {train_missing}, Test: {test_missing}")
    if train_missing > 0 or test_missing > 0:
        issues_found.append("Missing values detected")
        print("  ⚠️  ACTION: Handle missing values before training")
    else:
        print("  ✓ No missing values")
    
    # Check for infinite values
    train_inf = np.isinf(X_train).sum()
    test_inf = np.isinf(X_test).sum()
    print(f"  Infinite values - Train: {train_inf}, Test: {test_inf}")
    if train_inf > 0 or test_inf > 0:
        issues_found.append("Infinite values detected")
        print("  ⚠️  ACTION: Replace or remove infinite values")
    else:
        print("  ✓ No infinite values")
    
    # Check data types
    if isinstance(X_train, pd.DataFrame):
        non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            issues_found.append(f"Non-numeric columns: {list(non_numeric)}")
            print(f"  ⚠️  Non-numeric columns: {list(non_numeric)}")
        else:
            print("  ✓ All columns numeric")
    
    # Check label distribution
    print(f"\\n  Target statistics:")
    print(f"    Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"    Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    # Check for label leakage (perfect correlation with features)
    from scipy.stats import pearsonr
    max_corr = 0
    max_corr_feature = None
    for i in range(X_train.shape[1]):
        corr, _ = pearsonr(X_train[:, i], y_train)
        if abs(corr) > abs(max_corr):
            max_corr = corr
            max_corr_feature = i
    
    print(f"\\n  Maximum correlation with target: {max_corr:.4f} (feature {max_corr_feature})")
    if abs(max_corr) > 0.95:
        issues_found.append(f"Suspiciously high correlation ({max_corr:.4f})")
        print("  ⚠️  WARNING: Possible data leakage!")
    
    # 2. DATA DISTRIBUTION CHECKS
    print("\\n2. DATA DISTRIBUTION CHECKS")
    print("-"*70)
    
    # Check train-test distribution similarity
    from scipy.stats import ks_2samp
    distribution_mismatches = []
    
    for i in range(min(X_train.shape[1], 10)):  # Check first 10 features
        stat, p_value = ks_2samp(X_train[:, i], X_test[:, i])
        if p_value < 0.05:
            distribution_mismatches.append(i)
    
    if distribution_mismatches:
        print(f"  ⚠️  Distribution mismatch in features: {distribution_mismatches}")
        issues_found.append("Train-test distribution mismatch")
        print("  ACTION: Check if test set is representative")
    else:
        print("  ✓ Train and test distributions appear similar")
    
    # Check for constant features
    constant_features = []
    for i in range(X_train.shape[1]):
        if X_train[:, i].std() < 1e-10:
            constant_features.append(i)
    
    if constant_features:
        print(f"  ⚠️  Constant features: {constant_features}")
        issues_found.append("Constant features detected")
        print("  ACTION: Remove constant features")
    else:
        print("  ✓ No constant features")
    
    # 3. MODEL PERFORMANCE CHECKS
    if model is not None:
        print("\\n3. MODEL PERFORMANCE CHECKS")
        print("-"*70)
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"  Train MSE: {train_mse:.2f}, R²: {train_r2:.4f}")
        print(f"  Test  MSE: {test_mse:.2f}, R²: {test_r2:.4f}")
        print(f"  Gap: {test_mse - train_mse:.2f} ({(test_mse/train_mse - 1)*100:.1f}%)")
        
        # Diagnose bias/variance
        if test_mse > train_mse * 1.5:
            print("  ⚠️  HIGH VARIANCE (Overfitting)")
            issues_found.append("Overfitting detected")
            print("  Actions:")
            print("    - Reduce model complexity")
            print("    - Add regularization")
            print("    - Get more training data")
            print("    - Remove noisy features")
        elif train_mse > np.var(y_train) * 0.8:  # Barely better than predicting mean
            print("  ⚠️  HIGH BIAS (Underfitting)")
            issues_found.append("Underfitting detected")
            print("  Actions:")
            print("    - Increase model complexity")
            print("    - Add more relevant features")
            print("    - Reduce regularization")
            print("    - Try different model")
        else:
            print("  ✓ Model appears well-balanced")
        
        # Check predictions
        if (test_pred < 0).any():
            print("  ⚠️  Negative predictions (may be problematic)")
            issues_found.append("Negative predictions")
        
        if np.isnan(test_pred).any():
            print("  ⚠️  NaN predictions!")
            issues_found.append("NaN predictions")
    
    # SUMMARY
    print("\\n" + "="*70)
    print("DEBUGGING SUMMARY")
    print("="*70)
    
    if not issues_found:
        print("\\n✓ No major issues detected!")
        print("  Model appears healthy.")
    else:
        print(f"\\n⚠️  {len(issues_found)} issue(s) found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    
    return issues_found

# Run checklist
model = RandomForestRegressor(n_estimators=100, random_state=42)
issues = comprehensive_debug_checklist(X_train, X_test, y_train, y_test, model)
\`\`\`

## Residual Analysis

\`\`\`python
def analyze_residuals(model, X_train, X_test, y_train, y_test):
    """Detailed residual analysis."""
    
    print("\\n" + "="*70)
    print("RESIDUAL ANALYSIS")
    print("="*70)
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_residuals = y_train - train_pred
    test_residuals = y_test - test_pred
    
    # Statistics
    print("\\nResidual Statistics:")
    print(f"  Train - Mean: {train_residuals.mean():7.2f}, Std: {train_residuals.std():7.2f}")
    print(f"  Test  - Mean: {test_residuals.mean():7.2f}, Std: {test_residuals.std():7.2f}")
    
    # Check for bias
    if abs(test_residuals.mean()) > test_residuals.std() * 0.1:
        if test_residuals.mean() > 0:
            print("  ⚠️  Model systematically UNDER-predicts")
        else:
            print("  ⚠️  Model systematically OVER-predicts")
    else:
        print("  ✓ No systematic bias detected")
    
    # Check for heteroscedasticity
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(test_pred, np.abs(test_residuals))
    print(f"\\n  Correlation(predictions, |residuals|): {corr:.4f} (p={p_value:.4f})")
    if abs(corr) > 0.3 and p_value < 0.05:
        print("  ⚠️  Heteroscedasticity detected (error variance not constant)")
        print("  ACTION: Consider transforming target variable")
    else:
        print("  ✓ Homoscedastic (constant error variance)")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_test, test_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', linewidth=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Predicted vs Actual', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. Residuals vs Predicted
    ax2 = axes[0, 1]
    ax2.scatter(test_pred, test_residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Residual Distribution
    ax3 = axes[1, 0]
    ax3.hist(test_residuals, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Q-Q Plot
    from scipy import stats
    ax4 = axes[1, 1]
    stats.probplot(test_residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
    print("\\n  Plots saved to 'residual_analysis.png'")

analyze_residuals(model, X_train, X_test, y_train, y_test)
\`\`\`

## Common Debugging Scenarios

\`\`\`python
print("\\n" + "="*70)
print("COMMON DEBUGGING SCENARIOS")
print("="*70)

scenarios = [
    {
        'symptoms': 'Training accuracy high, test accuracy low',
        'diagnosis': 'Overfitting',
        'solutions': [
            'Reduce model complexity',
            'Add regularization',
            'Get more training data',
            'Use dropout (neural networks)',
            'Early stopping',
            'Feature selection'
        ]
    },
    {
        'symptoms': 'Both training and test accuracy low',
        'diagnosis': 'Underfitting',
        'solutions': [
            'Increase model complexity',
            'Add more features',
            'Reduce regularization',
            'Train longer',
            'Try different model architecture'
        ]
    },
    {
        'symptoms': 'Test accuracy varies wildly',
        'diagnosis': 'High variance / unstable model',
        'solutions': [
            'Use cross-validation',
            'Increase training data',
            'Ensemble methods',
            'Set random seeds for reproducibility',
            'Check test set size'
        ]
    },
    {
        'symptoms': 'Good CV score, poor test score',
        'diagnosis': 'Data leakage or distribution shift',
        'solutions': [
            'Check for data leakage',
            'Verify preprocessing done correctly',
            'Ensure test set is representative',
            'Review feature engineering',
            'Check for time-based dependencies'
        ]
    },
    {
        'symptoms': 'Model worse than simple baseline',
        'diagnosis': 'Model or implementation error',
        'solutions': [
            'Check data preprocessing',
            'Verify labels are correct',
            'Review model hyperparameters',
            'Try scikit-learn default parameters first',
            'Check for bugs in custom code'
        ]
    },
    {
        'symptoms': 'NaN or infinite predictions',
        'diagnosis': 'Numerical instability',
        'solutions': [
            'Check for NaN in input data',
            'Scale features',
            'Reduce learning rate',
            'Add gradient clipping (neural networks)',
            'Check for division by zero'
        ]
    }
]

for i, scenario in enumerate(scenarios, 1):
    print(f"\\n{i}. {scenario['diagnosis'].upper()}")
    print(f"   Symptoms: {scenario['symptoms']}")
    print("   Solutions:")
    for solution in scenario['solutions']:
        print(f"     • {solution}")
\`\`\`

## Debugging Workflow

\`\`\`python
print("\\n" + "="*70)
print("SYSTEMATIC DEBUGGING WORKFLOW")
print("="*70)

workflow = """
Step 1: VERIFY DATA
  □ Check data loading (shapes, types)
  □ Verify no missing/infinite values
  □ Confirm labels are correct
  □ Check train-test split

Step 2: ESTABLISH BASELINE
  □ Train simplest possible model
  □ Compare with random/mean prediction
  □ Document baseline performance

Step 3: INITIAL DIAGNOSTICS
  □ Calculate train and validation metrics
  □ Check for obvious over/underfitting
  □ Examine learning curves
  □ Analyze residuals

Step 4: DEEP DIVE
  □ Feature importance analysis
  □ Check for data leakage
  □ Verify preprocessing steps
  □ Review hyperparameters

Step 5: EXPERIMENT
  □ Try different models
  □ Tune hyperparameters systematically
  □ Test different feature sets
  □ Document all experiments

Step 6: VALIDATE
  □ Cross-validation
  □ Hold-out test set
  □ Check on different data subsets
  □ Verify assumptions hold

Step 7: PRODUCTION CHECKS
  □ Test on real-world examples
  □ Monitor predictions over time
  □ Set up alerts for anomalies
  □ Plan for model updates
"""

print(workflow)
\`\`\`

## Quick Debugging Commands

\`\`\`python
print("\\n" + "="*70)
print("QUICK DEBUGGING COMMANDS")
print("="*70)

debugging_commands = """
# Check data
print(X_train.shape, X_test.shape)
print(X_train.isnull().sum())  # For pandas
print(np.isnan(X_train).sum())  # For numpy
print(X_train.describe())

# Check labels
print(y_train.min(), y_train.max(), y_train.mean())
print(pd.Series(y_train).value_counts())  # For classification

# Check predictions
pred = model.predict(X_test)
print(f"Predictions: min={pred.min()}, max={pred.max()}, mean={pred.mean()}")
print(f"NaN predictions: {np.isnan(pred).sum()}")

# Quick performance check
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
print(f"Train R²: {r2_score(y_train, model.predict(X_train)):.4f}")
print(f"Test R²: {r2_score(y_test, model.predict(X_test)):.4f}")

# Check feature importance (tree models)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    for i, imp in enumerate(importances):
        if imp > 0.1:  # Significant features
            print(f"Feature {i}: {imp:.4f}")

# Learning curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5
)
print(f"Learning curve - Train: {train_scores.mean(axis=1)}")
print(f"Learning curve - Val: {val_scores.mean(axis=1)}")
"""

print(debugging_commands)
\`\`\`

## Key Takeaways

1. **Systematic Approach**: Follow debugging checklist
2. **Check Data First**: 80% of issues are data problems
3. **Establish Baselines**: Know what "good" looks like
4. **Visualize Everything**: Plots reveal hidden issues
5. **Document Findings**: Track what you tried
6. **Start Simple**: Debug with simple model first
7. **One Change at a Time**: Isolate issues
8. **Verify Assumptions**: Check your assumptions about data

**Common Mistakes to Avoid:**
- Not checking for data leakage
- Incorrect train-test split (especially for time series)
- Forgetting to scale features
- Using test set for development decisions
- Not setting random seeds (non-reproducible results)
- Ignoring warning messages

## Trading-Specific Debugging

\`\`\`python
print("\\n" + "="*70)
print("TRADING-SPECIFIC DEBUGGING")
print("="*70)

trading_checks = """
1. TEMPORAL INTEGRITY
   □ Ensure no future data in features
   □ Verify time-based split (not random)
   □ Check for survivorship bias
   □ Validate lookahead bias

2. REALISTIC ASSUMPTIONS
   □ Include transaction costs
   □ Model slippage
   □ Consider liquidity constraints
   □ Account for data delays

3. OVERFITTING RISKS
   □ Many strategies will backtest well by chance
   □ Use walk-forward validation
   □ Test on multiple time periods
   □ Verify across different market regimes

4. PRACTICAL CONSTRAINTS
   □ Can execute at predicted prices?
   □ Position size limits
   □ Margin requirements
   □ Overnight holding costs

Common Trading Model Bugs:
• Using close-to-close returns but trading at open
• Not accounting for bid-ask spread
• Training on adjusted prices, testing on unadjusted
• Ignoring dividends and splits
• Data snooping (testing many strategies)
"""

print(trading_checks)
\`\`\`

## Debugging Tools Checklist

\`\`\`python
print("\\n" + "="*70)
print("ESSENTIAL DEBUGGING TOOLS")
print("="*70)

tools = {
    'Data Inspection': [
        'pandas.DataFrame.info()',
        'pandas.DataFrame.describe()',
        'pandas.DataFrame.isnull().sum()',
        'seaborn.pairplot()',
        'pandas.DataFrame.corr()'
    ],
    'Model Diagnostics': [
        'sklearn.model_selection.learning_curve()',
        'sklearn.model_selection.validation_curve()',
        'sklearn.metrics.classification_report()',
        'sklearn.metrics.confusion_matrix()',
        'sklearn.inspection.permutation_importance()'
    ],
    'Visualization': [
        'matplotlib for custom plots',
        'seaborn for statistical plots',
        'plotly for interactive viz',
        'yellowbrick for ML viz',
        'SHAP for explanations'
    ],
    'Logging & Tracking': [
        'MLflow for experiment tracking',
        'Weights & Biases (wandb)',
        'TensorBoard',
        'Custom logging with Python logging module'
    ]
}

for category, tool_list in tools.items():
    print(f"\\n{category}:")
    for tool in tool_list:
        print(f"  • {tool}")
\`\`\`

## Final Checklist

Before deploying to production:

- [ ] Model performs better than baseline
- [ ] Validated on held-out test set
- [ ] Cross-validation shows stable performance
- [ ] No data leakage
- [ ] Feature preprocessing documented
- [ ] Hyperparameters tuned systematically
- [ ] Model predictions make domain sense
- [ ] Edge cases tested
- [ ] Monitoring plan in place
- [ ] Rollback strategy defined

## Further Reading

- Breck, E., et al. (2017). "The ML Test Score: A Rubric for ML Production Readiness"
- Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems"
- Polyzotis, N., et al. (2017). "Data Management Challenges in Production Machine Learning"
`,
  exercises: [
    {
      prompt:
        'Build a comprehensive model debugging tool that automatically runs through all checks, generates diagnostic reports, and provides specific recommendations for fixing identified issues.',
      solution: `# Complete debugging framework implementation`,
    },
  ],
  quizId: 'model-evaluation-optimization-model-debugging',
  multipleChoiceId: 'model-evaluation-optimization-model-debugging-mc',
};
