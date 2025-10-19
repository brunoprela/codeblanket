/**
 * Feature Selection Section
 */

export const featureselectionSection = {
  id: 'feature-selection',
  title: 'Feature Selection',
  content: `# Feature Selection

## Introduction

Feature selection identifies the most relevant features for modeling, removing irrelevant or redundant features. This improves model performance, reduces overfitting, speeds up training, and enhances interpretability.

**Benefits**:
- Reduced overfitting (fewer features → lower variance)
- Improved accuracy (remove noisy features)
- Faster training and prediction
- Better interpretability
- Reduced data collection costs

**Curse of Dimensionality**: High-dimensional data becomes sparse, making patterns harder to detect.

**Applications**:
- Gene selection in bioinformatics
- Text classification (feature is a word)
- Image recognition (pixel/feature reduction)
- Customer analytics (select key behavioral features)

## Types of Feature Selection

**Three Main Approaches**:

1. **Filter Methods**: Rank features by statistical scores (correlation, mutual information)
   - Fast, model-agnostic
   - Don't consider feature interactions
   
2. **Wrapper Methods**: Use model performance to evaluate feature subsets
   - Consider feature interactions
   - Computationally expensive
   
3. **Embedded Methods**: Feature selection during model training (regularization)
   - Balance between filter and wrapper
   - Model-specific

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.shape[1]}")
\`\`\`

## Filter Methods

### 1. Correlation

Remove highly correlated features (keep one from each correlated pair).

\`\`\`python
# Correlation matrix
correlation_matrix = X_train.corr().abs()

# Select upper triangle
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation > 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print(f"Features to drop due to high correlation: {len(to_drop)}")
print(to_drop[:5])

# Drop highly correlated features
X_train_filtered = X_train.drop(columns=to_drop)
X_test_filtered = X_test.drop(columns=to_drop)

print(f"Reduced features: {X_train_filtered.shape[1]}")

# Compare performance
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_filtered = RandomForestClassifier(n_estimators=100, random_state=42)

rf_full.fit(X_train, y_train)
rf_filtered.fit(X_train_filtered, y_train)

print(f"\\nFull features test accuracy: {rf_full.score(X_test, y_test):.4f}")
print(f"Filtered features test accuracy: {rf_filtered.score(X_test_filtered, y_test):.4f}")
\`\`\`

### 2. Variance Threshold

Remove features with low variance (near-constant features).

\`\`\`python
from sklearn.feature_selection import VarianceThreshold

# Remove features with variance < threshold
selector = VarianceThreshold(threshold=0.1)
X_train_var = selector.fit_transform(X_train)

print(f"Features after variance threshold: {X_train_var.shape[1]}")
print(f"Removed {X_train.shape[1] - X_train_var.shape[1]} features")
\`\`\`

### 3. Univariate Statistical Tests

Select features based on statistical tests (chi-squared, ANOVA F-value).

\`\`\`python
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# ANOVA F-statistic
selector_f = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = selector_f.fit_transform(X_train, y_train)

# Get selected feature names
selected_features = X_train.columns[selector_f.get_support()].tolist()

print("Top 10 features by ANOVA F-statistic:")
print(selected_features)

# Visualize feature scores
scores = pd.DataFrame({
    'feature': X_train.columns,
    'score': selector_f.scores_
}).sort_values('score', ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(range(len(scores.head(15))), scores.head(15)['score'])
plt.yticks(range(len(scores.head(15))), scores.head(15)['feature'])
plt.xlabel('F-Statistic Score')
plt.title('Top 15 Features by ANOVA F-Statistic')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
\`\`\`

### 4. Mutual Information

Measure dependency between features and target (captures non-linear relationships).

\`\`\`python
from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

mi_df = pd.DataFrame({
    'feature': X_train.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("Top 10 features by Mutual Information:")
print(mi_df.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(12, 6))
plt.barh(range(len(mi_df.head(15))), mi_df.head(15)['mi_score'])
plt.yticks(range(len(mi_df.head(15))), mi_df.head(15)['feature'])
plt.xlabel('Mutual Information Score')
plt.title('Top 15 Features by Mutual Information')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
\`\`\`

## Wrapper Methods

### Recursive Feature Elimination (RFE)

Recursively removes least important features.

\`\`\`python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# RFE with Logistic Regression
estimator = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)

rfe.fit(X_train, y_train)

# Selected features
rfe_features = X_train.columns[rfe.support_].tolist()
print("RFE Selected Features:")
print(rfe_features)

# Feature ranking (1 = selected, higher = eliminated earlier)
ranking_df = pd.DataFrame({
    'feature': X_train.columns,
    'ranking': rfe.ranking_
}).sort_values('ranking')

print("\\nFeature Rankings (1 = selected):")
print(ranking_df.head(15).to_string(index=False))

# Visualize performance vs number of features
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Accuracy')
plt.title('RFECV: Performance vs Number of Features')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--', label=f'Optimal: {rfecv.n_features_} features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nOptimal number of features: {rfecv.n_features_}")
\`\`\`

### Forward/Backward Selection

**Forward Selection**: Start with 0 features, add best one iteratively
**Backward Selection**: Start with all features, remove worst one iteratively

\`\`\`python
from mlxtend.feature_selection import SequentialFeatureSelector

# Forward selection
sfs_forward = SequentialFeatureSelector(
    RandomForestClassifier(n_estimators=50, random_state=42),
    k_features=10,
    forward=True,
    scoring='accuracy',
    cv=5
)

sfs_forward.fit(X_train.values, y_train)

forward_features = [X_train.columns[i] for i in sfs_forward.k_feature_idx_]
print("Forward Selection - Selected Features:")
print(forward_features)
\`\`\`

## Embedded Methods

### L1 Regularization (Lasso)

L1 penalty drives coefficients to exactly zero, performing feature selection.

\`\`\`python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

# Standardize features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal alpha via cross-validation
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")

# Train Lasso with optimal alpha
lasso = Lasso(alpha=lasso_cv.alpha_, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Selected features (non-zero coefficients)
lasso_coef = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lasso.coef_
})

selected_lasso = lasso_coef[lasso_coef['coefficient'] != 0].sort_values('coefficient', key=abs, ascending=False)

print(f"\\nFeatures selected by Lasso: {len(selected_lasso)}")
print("\\nTop 10 features by absolute coefficient:")
print(selected_lasso.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(12, 6))
sorted_coef = lasso_coef.sort_values('coefficient', key=abs, ascending=False).head(20)
colors = ['red' if c < 0 else 'blue' for c in sorted_coef['coefficient']]
plt.barh(range(len(sorted_coef)), sorted_coef['coefficient'], color=colors)
plt.yticks(range(len(sorted_coef)), sorted_coef['feature'])
plt.xlabel('Lasso Coefficient')
plt.title('Top 20 Features by Lasso Coefficient (Red=Negative, Blue=Positive)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
\`\`\`

### Tree-Based Feature Importance

Trees naturally rank feature importance.

\`\`\`python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest importance
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top K features
K = 10
top_k_features = rf_importance.head(K)['feature'].tolist()

print(f"Top {K} features by Random Forest importance:")
print(rf_importance.head(K).to_string(index=False))

# Train model with selected features only
X_train_selected = X_train[top_k_features]
X_test_selected = X_test[top_k_features]

rf_selected = RandomForestClassifier(n_estimators=200, random_state=42)
rf_selected.fit(X_train_selected, y_train)

print(f"\\nFull features ({X_train.shape[1]}): {rf.score(X_test, y_test):.4f}")
print(f"Top {K} features: {rf_selected.score(X_test_selected, y_test):.4f}")
\`\`\`

## Permutation Importance

Shuffle feature values and measure performance drop - more robust than built-in importance.

\`\`\`python
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Calculate permutation importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

perm_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Top 10 features by Permutation Importance:")
print(perm_df.head(10).to_string(index=False))

# Visualize with error bars
plt.figure(figsize=(12, 6))
top_10 = perm_df.head(10)
plt.barh(range(len(top_10)), top_10['importance_mean'], xerr=top_10['importance_std'])
plt.yticks(range(len(top_10)), top_10['feature'])
plt.xlabel('Permutation Importance')
plt.title('Top 10 Features by Permutation Importance (with std)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
\`\`\`

## Comparing Methods

\`\`\`python
# Compare different selection methods

methods = {
    'Correlation Filter': X_train.drop(columns=to_drop).columns.tolist(),
    'ANOVA F-statistic (top 10)': selected_features,
    'Mutual Information (top 10)': mi_df.head(10)['feature'].tolist(),
    'RFE (10 features)': rfe_features,
    'Lasso': selected_lasso['feature'].tolist(),
    'RF Importance (top 10)': top_k_features
}

# Test each method
results = []

for name, features in methods.items():
    if len(features) == 0:
        continue
    
    # Ensure features exist in both train and test
    features = [f for f in features if f in X_train.columns]
    
    rf_test = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_test.fit(X_train[features], y_train)
    
    train_acc = rf_test.score(X_train[features], y_train)
    test_acc = rf_test.score(X_test[features], y_test)
    
    results.append({
        'Method': name,
        'Num Features': len(features),
        'Train Acc': train_acc,
        'Test Acc': test_acc
    })

# Add baseline (all features)
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
rf_baseline.fit(X_train, y_train)
results.append({
    'Method': 'All Features (Baseline)',
    'Num Features': X_train.shape[1],
    'Train Acc': rf_baseline.score(X_train, y_train),
    'Test Acc': rf_baseline.score(X_test, y_test)
})

results_df = pd.DataFrame(results).sort_values('Test Acc', ascending=False)

print("\\n" + "="*60)
print("FEATURE SELECTION METHOD COMPARISON")
print("="*60)
print(results_df.to_string(index=False))
\`\`\`

## Practical Guidelines

**When to Use Each Method**:

1. **Filter Methods** (Correlation, MI, ANOVA):
   - Fast, good for high-dimensional data
   - Initial feature screening
   - When model-agnostic selection needed

2. **Wrapper Methods** (RFE):
   - Best accuracy
   - Small-medium datasets (computationally expensive)
   - When feature interactions matter

3. **Embedded Methods** (Lasso, Tree Importance):
   - Balance speed and accuracy
   - Integrated with model training
   - Most practical for production

**Best Practices**:
- Start with filter methods to remove obviously bad features
- Use embedded methods (tree importance, Lasso) for refined selection
- Always validate with cross-validation
- Consider domain knowledge (don't blindly trust algorithms)
- Monitor for data leakage (select features only on training data!)

\`\`\`python
# Production Pipeline with Feature Selection

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Pipeline ensures no data leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=15)),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline.fit(X_train, y_train)

print(f"\\nPipeline Test Accuracy: {pipeline.score(X_test, y_test):.4f}")

# Get selected features
selector = pipeline.named_steps['feature_selection']
selected_mask = selector.get_support()
selected_pipeline_features = X_train.columns[selected_mask].tolist()

print(f"Features selected by pipeline: {selected_pipeline_features}")
\`\`\`

## Summary

Feature selection improves models by removing irrelevant/redundant features:

**Methods**:
- **Filter**: Statistical scores (fast, model-agnostic)
- **Wrapper**: Model performance (accurate, expensive)
- **Embedded**: During training (practical balance)

**Key Techniques**:
- Correlation/variance thresholds
- Univariate tests (ANOVA, MI)
- RFE (recursive elimination)
- L1 regularization (Lasso)
- Tree/permutation importance

**Best Practices**:
- Select on training data only (avoid leakage)
- Use pipelines for safety
- Validate with CV
- Consider domain knowledge
- Embedded methods often best for production

Next: Handling imbalanced datasets!
`,
  codeExample: `# Complete Feature Selection Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Safe pipeline (no data leakage)
feature_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=15)),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

feature_pipeline.fit(X_train, y_train)
print(f"Test accuracy: {feature_pipeline.score(X_test, y_test):.4f}")

# Get selected features
selected = X_train.columns[feature_pipeline.named_steps['selector'].get_support()]
print(f"Selected features: {list(selected)}")
`,
};
