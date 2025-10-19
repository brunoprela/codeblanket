/**
 * Ensemble Methods Section
 */

export const ensemblemethodsSection = {
  id: 'ensemble-methods',
  title: 'Ensemble Methods',
  content: `# Ensemble Methods

## Introduction

Ensemble methods combine multiple models to achieve better performance than any single model. The key insight: diverse models make different errors, and combining their predictions reduces overall error.

**"Wisdom of Crowds" Principle**: Many weak models together can be stronger than one strong model.

**Types of Ensembles**:
1. **Bagging**: Train same algorithm on different data subsets (Random Forest)
2. **Boosting**: Train models sequentially, each correcting previous errors (Gradient Boosting)
3. **Stacking**: Train meta-model to combine predictions of different base models
4. **Voting**: Combine predictions from different algorithms

**Applications**:
- Kaggle competitions (most winners use ensembles)
- Production systems (improved reliability)
- Model selection (when unsure which algorithm is best)
- Risk-critical applications (medical diagnosis, fraud detection)

## Why Ensembles Work

**Bias-Variance Decomposition**:

\\[ \\text{Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Error} \\]

- **Bagging**: Reduces variance (averaging diverse predictions)
- **Boosting**: Reduces bias (sequential error correction)
- **Stacking**: Leverages strengths of different model types

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest (Bagging)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    results.append({'Model': name, 'Train': train_acc, 'Test': test_acc})

import pandas as pd
df_results = pd.DataFrame(results)
print("Single Models vs Ensembles:")
print(df_results.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
x = range(len(results))
width = 0.35
plt.bar([i - width/2 for i in x], df_results['Train'], width, label='Train', alpha=0.8)
plt.bar([i + width/2 for i in x], df_results['Test'], width, label='Test', alpha=0.8)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Single Models vs Ensemble Methods')
plt.xticks(x, df_results['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

print("\\nNotice: Ensembles (RF, GB) typically perform best!")
\`\`\`

## Voting Classifiers

Combine predictions from multiple different algorithms.

**Hard Voting**: Majority vote (classification)
**Soft Voting**: Average predicted probabilities (usually better)

\`\`\`python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Define base models
log_clf = LogisticRegression(random_state=42, max_iter=1000)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(probability=True, random_state=42)  # probability=True for soft voting
nb_clf = GaussianNB()

# Hard voting
voting_hard = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('svm', svm_clf), ('nb', nb_clf)],
    voting='hard'
)

# Soft voting
voting_soft = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('svm', svm_clf), ('nb', nb_clf)],
    voting='soft'
)

# Train and compare
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

print("="*60)
print("VOTING CLASSIFIERS")
print("="*60)

for name, clf in [('Logistic Regression', log_clf), ('Random Forest', rf_clf), 
                  ('SVM', svm_clf), ('Naive Bayes', nb_clf), 
                  ('Voting (Hard)', voting_hard), ('Voting (Soft)', voting_soft)]:
    if name not in ['Voting (Hard)', 'Voting (Soft)']:
        clf.fit(X_train, y_train)
    print(f"{name:25s} Test Accuracy: {clf.score(X_test, y_test):.4f}")

print("\\nSoft voting typically outperforms individual models!")
\`\`\`

## Weighted Voting

Give more weight to better models.

\`\`\`python
# Weighted soft voting
voting_weighted = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('svm', svm_clf)],
    voting='soft',
    weights=[1, 2, 1]  # Give RF double weight
)

voting_weighted.fit(X_train, y_train)

print(f"\\nWeighted Voting Test Accuracy: {voting_weighted.score(X_test, y_test):.4f}")
\`\`\`

## Stacking (Stacked Generalization)

Train a meta-model to combine predictions of base models.

**Architecture**:
- **Level 0**: Base models (diverse algorithms)
- **Level 1**: Meta-model (learns how to combine base predictions)

\`\`\`python
from sklearn.ensemble import StackingClassifier

# Define base models (level 0)
base_models = [
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Define meta-model (level 1)
meta_model = LogisticRegression(random_state=42)

# Create stacking ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Use cross-validation to generate meta-features
)

stacking_clf.fit(X_train, y_train)

print("="*60)
print("STACKING ENSEMBLE")
print("="*60)
print(f"Stacking Test Accuracy: {stacking_clf.score(X_test, y_test):.4f}")

# Compare with individual base models
for name, model in base_models:
    model.fit(X_train, y_train)
    print(f"{name.upper():5s} Test Accuracy: {model.score(X_test, y_test):.4f}")

print("\\nStacking often achieves best performance!")
\`\`\`

## How Stacking Works

\`\`\`python
# Manual stacking to understand the process

# Step 1: Train base models and get out-of-fold predictions
from sklearn.model_selection import cross_val_predict

base_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
base_model_2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
base_model_3 = SVC(probability=True, random_state=42)

# Get out-of-fold predictions for training data
oof_pred_1 = cross_val_predict(base_model_1, X_train, y_train, cv=5, method='predict_proba')
oof_pred_2 = cross_val_predict(base_model_2, X_train, y_train, cv=5, method='predict_proba')
oof_pred_3 = cross_val_predict(base_model_3, X_train, y_train, cv=5, method='predict_proba')

# Create meta-features
X_meta_train = np.column_stack([
    oof_pred_1[:, 1],  # Probability of class 1 from model 1
    oof_pred_2[:, 1],  # Probability of class 1 from model 2
    oof_pred_3[:, 1]   # Probability of class 1 from model 3
])

# Step 2: Train base models on full training data
base_model_1.fit(X_train, y_train)
base_model_2.fit(X_train, y_train)
base_model_3.fit(X_train, y_train)

# Step 3: Get base model predictions for test data
test_pred_1 = base_model_1.predict_proba(X_test)[:, 1]
test_pred_2 = base_model_2.predict_proba(X_test)[:, 1]
test_pred_3 = base_model_3.predict_proba(X_test)[:, 1]

X_meta_test = np.column_stack([test_pred_1, test_pred_2, test_pred_3])

# Step 4: Train meta-model
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta_train, y_train)

# Step 5: Make final predictions
y_pred_stacking = meta_model.predict(X_meta_test)

from sklearn.metrics import accuracy_score
print(f"\\nManual Stacking Test Accuracy: {accuracy_score(y_test, y_pred_stacking):.4f}")

# Meta-model coefficients show how much each base model contributes
print("\\nMeta-model coefficients (importance of each base model):")
print(f"Random Forest:      {meta_model.coef_[0][0]:.4f}")
print(f"Gradient Boosting:  {meta_model.coef_[0][1]:.4f}")
print(f"SVM:                {meta_model.coef_[0][2]:.4f}")
\`\`\`

## Blending

Similar to stacking but uses holdout validation set instead of cross-validation.

\`\`\`python
# Split training data
X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train base models on base training set
base_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
base_model_2 = GradientBoostingClassifier(n_estimators=100, random_state=42)

base_model_1.fit(X_train_base, y_train_base)
base_model_2.fit(X_train_base, y_train_base)

# Generate meta-features for meta training set
meta_train_1 = base_model_1.predict_proba(X_train_meta)[:, 1]
meta_train_2 = base_model_2.predict_proba(X_train_meta)[:, 1]
X_blend_train = np.column_stack([meta_train_1, meta_train_2])

# Generate meta-features for test set
meta_test_1 = base_model_1.predict_proba(X_test)[:, 1]
meta_test_2 = base_model_2.predict_proba(X_test)[:, 1]
X_blend_test = np.column_stack([meta_test_1, meta_test_2])

# Train meta-model
blend_meta = LogisticRegression(random_state=42)
blend_meta.fit(X_blend_train, y_train_meta)

# Predict
y_pred_blend = blend_meta.predict(X_blend_test)

print(f"Blending Test Accuracy: {accuracy_score(y_test, y_pred_blend):.4f}")
print("\\nBlending is simpler than stacking but uses less training data.")
\`\`\`

## Multi-Layer Stacking

Stack multiple levels of models.

\`\`\`python
# Level 0: Base models
level0_models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    SVC(probability=True, random_state=42),
    LogisticRegression(random_state=42, max_iter=1000)
]

# Level 1: Stack on level 0
level1_clf = StackingClassifier(
    estimators=[('rf', level0_models[0]), ('gb', level0_models[1]), 
                ('svm', level0_models[2]), ('lr', level0_models[3])],
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    cv=5
)

# Level 2: Final meta-model
level2_clf = StackingClassifier(
    estimators=[('level1', level1_clf)],
    final_estimator=LogisticRegression(random_state=42),
    cv=3
)

# Note: In practice, this is rarely needed and can overfit
# level2_clf.fit(X_train, y_train)
# print(f"Multi-layer stacking: {level2_clf.score(X_test, y_test):.4f}")

print("Multi-layer stacking is possible but usually unnecessary.")
print("Single-layer stacking is typically sufficient.")
\`\`\`

## Feature-Weighted Linear Stacking

Use base model predictions as features in meta-model.

\`\`\`python
from sklearn.preprocessing import StandardScaler

# Train base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Combine original features + base model predictions
rf_pred = rf.predict_proba(X_train)
gb_pred = gb.predict_proba(X_train)

X_train_extended = np.column_stack([X_train, rf_pred, gb_pred])

# Train meta-model on extended features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_extended)

meta_clf = LogisticRegression(random_state=42, max_iter=1000)
meta_clf.fit(X_train_scaled, y_train)

# Test
rf_pred_test = rf.predict_proba(X_test)
gb_pred_test = gb.predict_proba(X_test)
X_test_extended = np.column_stack([X_test, rf_pred_test, gb_pred_test])
X_test_scaled = scaler.transform(X_test_extended)

print(f"Feature-weighted stacking: {meta_clf.score(X_test_scaled, y_test):.4f}")
\`\`\`

## Ensemble Best Practices

**Diversity is Key**:
- Use different algorithm types (tree, linear, SVM, etc.)
- Use different feature subsets
- Use different hyperparameters

**Practical Tips**:
1. **Start simple**: Voting classifier with 3-5 diverse models
2. **Use stacking**: When you need maximum performance
3. **CV carefully**: Use out-of-fold predictions to avoid overfitting
4. **Don't overstack**: One level usually sufficient
5. **Computational cost**: Ensembles are slower (train multiple models)

\`\`\`python
# Example: Building production ensemble

from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Diverse base models
production_ensemble = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='soft',
    weights=[2, 2, 1, 1]  # Weight gradient boosting models more
)

production_ensemble.fit(X_train, y_train)

print("="*60)
print("PRODUCTION ENSEMBLE")
print("="*60)
print(f"Test Accuracy: {production_ensemble.score(X_test, y_test):.4f}")
\`\`\`

## Summary

Ensemble methods combine multiple models for better performance:

**Types**:
- **Voting**: Simple averaging or voting
- **Bagging**: Train on bootstraps (Random Forest)
- **Boosting**: Sequential error correction (Gradient Boosting)
- **Stacking**: Meta-model learns to combine base models

**Key Principles**:
- Diversity → better ensemble
- Soft voting > hard voting
- Stacking often achieves best performance
- One level usually sufficient

**Practical Advice**:
- Start with Voting (simplest)
- Use Stacking for competitions
- Don't forget computational cost
- Diversity matters more than number of models

Next: Feature Selection to improve models before ensembling!
`,
  codeExample: `# Production-Ready Ensemble
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Diverse base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf', random_state=42))
]

# Stacking with logistic regression meta-model
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
print(f"Ensemble Test Accuracy: {stacking_clf.score(X_test, y_test):.4f}")
`,
};
