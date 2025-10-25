/**
 * Section: Feature Importance & Interpretation
 * Module: Model Evaluation & Optimization
 *
 * Covers feature importance analysis, SHAP values, and model interpretation techniques
 */

export const featureImportanceInterpretation = {
  id: 'feature-importance-interpretation',
  title: 'Feature Importance & Interpretation',
  content: `
# Feature Importance & Interpretation

## Introduction

"Your model achieved 95% accuracy—great! But **why** did it make that prediction?" In many applications (healthcare, finance, legal), understanding **how** models make decisions is as important as the predictions themselves.

**The Challenge**: Modern ML models (especially deep learning) are "black boxes." We need techniques to interpret and explain their decisions.

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import shap

# Load data
data = load_breast_cancer()
X = pd.DataFrame (data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Feature Importance & Interpretation")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Task: Binary classification (benign vs malignant)")
print(f"\\nFeature names (first 5): {data.feature_names[:5]}")
\`\`\`

## Built-in Feature Importance (Tree Models)

Tree-based models provide built-in feature importance based on how much each feature reduces impurity.

\`\`\`python
print("\\n" + "="*70)
print("Method 1: Built-in Feature Importance (Random Forest)")
print("="*70)

# Train Random Forest
rf_model = RandomForestClassifier (n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\\nTop 10 Most Important Features:")
for idx, row in feature_importance_df.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Visualize
fig, ax = plt.subplots (figsize=(10, 6))
top_features = feature_importance_df.head(15)
ax.barh (top_features['feature'], top_features['importance'])
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance (Top 15)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=150, bbox_inches='tight')
print("\\n📊 Saved: feature_importance_rf.png")

# Calculate cumulative importance
feature_importance_df['cumulative'] = feature_importance_df['importance'].cumsum()
n_features_90 = (feature_importance_df['cumulative'] <= 0.9).sum() + 1
print(f"\\n💡 Insight: Top {n_features_90} features explain 90% of importance")
print(f"   Could potentially reduce from {X.shape[1]} to {n_features_90} features")
\`\`\`

**Interpretation:**
- **Importance = 0.15**: This feature accounts for 15% of total predictive power
- **Top features**: Most influential for predictions
- **Cumulative analysis**: How many features capture most information

**Limitations:**
- ❌ Biased toward high-cardinality features
- ❌ Can't distinguish between correlated features
- ❌ Doesn't show direction of effect (positive or negative)

## Permutation Importance (Model-Agnostic)

Permutation importance measures how much performance drops when you shuffle a feature's values.

\`\`\`python
print("\\n" + "="*70)
print("Method 2: Permutation Importance (Model-Agnostic)")
print("="*70)

# Calculate permutation importance
print("\\nCalculating permutation importance (this may take a moment)...")
perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Create DataFrame
perm_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print("\\nTop 10 Features by Permutation Importance:")
for idx, row in perm_importance_df.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f} (+/- {row['std']:.4f})")

# Compare with built-in importance
comparison = feature_importance_df.merge(
    perm_importance_df, on='feature', suffixes=('_builtin', '_permutation')
)

print("\\n📊 Correlation between methods:")
correlation = comparison['importance_builtin'].corr (comparison['importance_permutation'])
print(f"   Spearman correlation: {correlation:.3f}")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Built-in
top_builtin = comparison.nlargest(10, 'importance_builtin')
axes[0].barh (top_builtin['feature'], top_builtin['importance_builtin'])
axes[0].set_title('Built-in Importance (Training Set)')
axes[0].invert_yaxis()

# Permutation
top_perm = comparison.nlargest(10, 'importance_permutation')
axes[1].barh (top_perm['feature'], top_perm['importance_permutation'])
axes[1].errorbar (top_perm['importance_permutation'], top_perm['feature'], 
                 xerr=top_perm['std_permutation'], fmt='none', color='red', alpha=0.5)
axes[1].set_title('Permutation Importance (Test Set)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
print("\\n📊 Saved: feature_importance_comparison.png")

print("\\n💡 Key Insight:")
print("   Built-in: How features were used in training")
print("   Permutation: How features impact actual predictions")
\`\`\`

**How It Works:**
1. Measure baseline performance
2. Shuffle one feature's values (breaks relationship with target)
3. Measure new performance
4. Importance = baseline_score - shuffled_score

**Advantages:**
- ✅ Model-agnostic (works with any model)
- ✅ Based on actual prediction performance
- ✅ Provides uncertainty estimates (std)

## SHAP Values (SHapley Additive exPlanations)

SHAP values explain individual predictions by showing each feature's contribution.

\`\`\`python
print("\\n" + "="*70)
print("Method 3: SHAP Values (Individual Prediction Explanations)")
print("="*70)

# Create SHAP explainer
print("\\nCreating SHAP explainer...")
explainer = shap.TreeExplainer (rf_model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list [class_0, class_1]
# We'll use class 1 (malignant)
shap_values_class1 = shap_values[1] if isinstance (shap_values, list) else shap_values

print("✅ SHAP values calculated")

# Global feature importance (mean absolute SHAP)
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs (shap_values_class1).mean (axis=0)
}).sort_values('importance', ascending=False)

print("\\nTop 10 Features by SHAP Importance:")
for idx, row in shap_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Summary plot (shows distribution of SHAP values)
print("\\n📊 Creating SHAP summary plot...")
plt.figure (figsize=(10, 8))
shap.summary_plot (shap_values_class1, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
print("   Saved: shap_importance.png")

# Detailed summary plot (shows feature values)
plt.figure (figsize=(10, 8))
shap.summary_plot (shap_values_class1, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
print("   Saved: shap_summary.png")

print("\\n💡 Interpretation:")
print("   • Each dot = one prediction")
print("   • Red = high feature value, Blue = low feature value")
print("   • Position = SHAP value (impact on prediction)")
print("   • Wide spread = feature has variable impact")
\`\`\`

## Explaining Individual Predictions

\`\`\`python
print("\\n" + "="*70)
print("Explaining Individual Predictions with SHAP")
print("="*70)

# Select a few interesting examples
example_indices = [0, 10, 50]  # First few test samples

for idx in example_indices:
    sample = X_test.iloc[idx]
    true_label = y_test.iloc[idx]
    pred_proba = rf_model.predict_proba(X_test.iloc[[idx]])[0]
    
    print(f"\\n{'='*60}")
    print(f"Sample #{idx}")
    print(f"  True label: {'Malignant' if true_label == 1 else 'Benign'}")
    print(f"  Predicted probability: Malignant={pred_proba[1]:.3f}, Benign={pred_proba[0]:.3f}")
    
    # Get SHAP values for this sample
    sample_shap = shap_values_class1[idx]
    
    # Show top contributing features
    feature_contributions = pd.DataFrame({
        'feature': X.columns,
        'value': sample.values,
        'shap': sample_shap
    }).sort_values('shap', key=abs, ascending=False)
    
    print(f"\\n  Top 5 Contributing Features:")
    for _, row in feature_contributions.head(5).iterrows():
        direction = "increases" if row['shap'] > 0 else "decreases"
        print(f"    {row['feature']:30s} = {row['value']:.2f}")
        print(f"      → SHAP: {row['shap']:+.3f} ({direction} malignancy probability)")

# Create waterfall plot for first sample
print("\\n📊 Creating SHAP waterfall plot for sample #0...")
plt.figure (figsize=(10, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_class1[0],
        base_values=explainer.expected_value[1],
        data=X_test.iloc[0],
        feature_names=X.columns.tolist()
    ),
    show=False
)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
print("   Saved: shap_waterfall.png")

print("\\n💡 Waterfall plot shows:")
print("   • Base value: Average model prediction")
print("   • Each feature pushes prediction up (red) or down (blue)")
print("   • Final prediction: sum of all contributions")
\`\`\`

## Comparing Methods

\`\`\`python
print("\\n" + "="*70)
print("Comparing Feature Importance Methods")
print("="*70)

# Combine all methods
comparison_full = pd.DataFrame({
    'feature': X.columns,
    'builtin': feature_importance_df.set_index('feature')['importance'],
    'permutation': perm_importance_df.set_index('feature')['importance'],
    'shap': shap_importance.set_index('feature')['importance']
})

# Normalize to 0-1 for comparison
for col in ['builtin', 'permutation', 'shap']:
    comparison_full[f'{col}_norm'] = comparison_full[col] / comparison_full[col].max()

# Show top features by each method
print("\\nTop 5 Features by Each Method:")
print("\\nBuilt-in Importance:")
for feat in comparison_full.nlargest(5, 'builtin')['feature']:
    print(f"  • {feat}")

print("\\nPermutation Importance:")
for feat in comparison_full.nlargest(5, 'permutation')['feature']:
    print(f"  • {feat}")

print("\\nSHAP Importance:")
for feat in comparison_full.nlargest(5, 'shap')['feature']:
    print(f"  • {feat}")

# Calculate agreement
print("\\n📊 Agreement between methods:")
from scipy.stats import spearmanr
methods = ['builtin', 'permutation', 'shap']
for i, method1 in enumerate (methods):
    for method2 in methods[i+1:]:
        corr, _ = spearmanr (comparison_full[method1], comparison_full[method2])
        print(f"   {method1:12s} vs {method2:12s}: {corr:.3f}")

print("\\n💡 High correlation (>0.8) indicates methods agree on important features")
\`\`\`

## Partial Dependence Plots

\`\`\`python
print("\\n" + "="*70)
print("Partial Dependence Plots (Feature Effects)")
print("="*70)

from sklearn.inspection import PartialDependenceDisplay

# Select top 4 features by SHAP importance
top_4_features = shap_importance.head(4)['feature'].tolist()
feature_indices = [X.columns.get_loc (f) for f in top_4_features]

print(f"\\nCreating partial dependence plots for: {top_4_features}")

fig, ax = plt.subplots (figsize=(14, 10))
PartialDependenceDisplay.from_estimator(
    rf_model, X_train, feature_indices,
    feature_names=X.columns.tolist(),
    n_cols=2, n_jobs=-1, ax=ax
)
plt.suptitle('Partial Dependence Plots (Top 4 Features)', fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig('partial_dependence.png', dpi=150, bbox_inches='tight')
print("📊 Saved: partial_dependence.png")

print("\\n💡 Interpretation:")
print("   • X-axis: Feature value")
print("   • Y-axis: Partial dependence (effect on prediction)")
print("   • Upward slope: Higher values increase prediction")
print("   • Flat line: Feature has no effect in that range")
\`\`\`

## Best Practices

\`\`\`python
print("\\n" + "="*70)
print("Feature Importance Best Practices")
print("="*70)

practices = {
    "Use Multiple Methods": "Different methods capture different aspects",
    "Check Correlations": "Correlated features can split importance",
    "Consider Domain Knowledge": "Does it make sense? Or is it data leakage?",
    "Analyze Individual Predictions": "Global importance != local importance",
    "Test on Holdout Set": "Use test set for permutation importance",
    "Communicate Uncertainty": "Report confidence intervals for importance",
    "Watch for Bias": "Importance can be biased toward high-cardinality features",
}

for practice, explanation in practices.items():
    print(f"\\n✅ {practice}")
    print(f"   → {explanation}")

print("\\n" + "="*70)
print("When to Use Each Method")
print("="*70)

methods_guide = {
    "Built-in Importance": {
        "Best for": "Quick analysis, tree-based models",
        "Use when": "Fast approximation needed",
        "Avoid when": "Using linear models or need precise estimates",
    },
    "Permutation Importance": {
        "Best for": "Model-agnostic importance, any model type",
        "Use when": "Want importance based on actual predictions",
        "Avoid when": "Features are highly correlated (can underestimate)",
    },
    "SHAP Values": {
        "Best for": "Individual predictions, fairness analysis",
        "Use when": "Need to explain specific predictions to stakeholders",
        "Avoid when": "Very large datasets (computationally expensive)",
    },
    "Partial Dependence": {
        "Best for": "Understanding feature effects and relationships",
        "Use when": "Want to visualize how features affect predictions",
        "Avoid when": "Features have complex interactions (use SHAP instead)",
    },
}

for method, details in methods_guide.items():
    print(f"\\n{method}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
\`\`\`

## Key Takeaways

1. **Feature importance** tells you which features drive predictions
2. **Multiple methods** provide different perspectives:
   - Built-in: Fast, training-based
   - Permutation: Model-agnostic, prediction-based
   - SHAP: Individual explanations, theoretically sound
3. **SHAP values** explain individual predictions (not just global importance)
4. **Use domain knowledge** to validate that importance makes sense
5. **Watch for data leakage**: High importance for unexpected features is suspicious
6. **Communication**: Visualizations help stakeholders understand model decisions
7. **Interpretability vs Accuracy**: Sometimes you must trade accuracy for interpretability

**Recommended Workflow:**
1. Start with built-in importance (if available) for quick overview
2. Validate with permutation importance
3. Use SHAP for detailed analysis and individual explanations
4. Create visualizations for stakeholder communication
5. Check that important features align with domain knowledge

Feature importance is not just for understanding models—it's essential for debugging, building trust, and meeting regulatory requirements!
`,
};
