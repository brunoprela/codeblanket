export const modelSelection = {
  title: 'Model Selection',
  content: `
# Model Selection

## Introduction

Model selection is the process of choosing the best model from a set of candidates. This involves comparing different algorithms, architectures, or configurations to find the one that best balances performance, complexity, and practical constraints.

**Key Questions:**
- Which algorithm is best for my problem?
- How do I fairly compare different models?
- What if models perform similarly?
- How do I balance accuracy with other factors?

## Comparing Multiple Models

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Model Selection")
print("="*70)

# Define models
models = {
    'Logistic Regression': LogisticRegression (max_iter=10000),
    'Decision Tree': DecisionTreeClassifier (random_state=42),
    'Random Forest': RandomForestClassifier (n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier (n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier (hidden_layers=(100,), max_iter=1000, random_state=42)
}

# Create pipelines with scaling
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Compare models with cross-validation
print("\\nComparing models with 5-fold cross-validation...")
results = []

for name, pipeline in pipelines.items():
    print(f"\\nEvaluating {name}...")
    
    start_time = time.time()
    
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=True,
        n_jobs=-1
    )
    
    fit_time = time.time() - start_time
    
    # Calculate mean and std for each metric
    result = {
        'Model': name,
        'CV Accuracy': cv_results['test_accuracy'].mean(),
        'CV Accuracy Std': cv_results['test_accuracy'].std(),
        'CV Precision': cv_results['test_precision'].mean(),
        'CV Recall': cv_results['test_recall'].mean(),
        'CV F1': cv_results['test_f1'].mean(),
        'CV ROC AUC': cv_results['test_roc_auc'].mean(),
        'Train Accuracy': cv_results['train_accuracy'].mean(),
        'Fit Time (s)': fit_time,
        'Overfitting': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
    }
    
    results.append (result)

# Display results
df_results = pd.DataFrame (results).sort_values('CV Accuracy', ascending=False)

print("\\n" + "="*70)
print("Model Comparison Results")
print("="*70)
print(df_results.to_string (index=False))

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy comparison
ax1 = axes[0, 0]
models_sorted = df_results.sort_values('CV Accuracy', ascending=True)
y_pos = np.arange (len (models_sorted))

bars = ax1.barh (y_pos, models_sorted['CV Accuracy'], xerr=models_sorted['CV Accuracy Std'])
ax1.set_yticks (y_pos)
ax1.set_yticklabels (models_sorted['Model'])
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Accuracy Comparison', fontweight='bold')
ax1.grid (axis='x', alpha=0.3)

# Color code
colors = ['green' if x > 0.95 else 'orange' if x > 0.90 else 'red' 
          for x in models_sorted['CV Accuracy']]
for bar, color in zip (bars, colors):
    bar.set_color (color)
    bar.set_alpha(0.7)

# 2. Multiple metrics heatmap
ax2 = axes[0, 1]
metrics_to_show = ['CV Accuracy', 'CV Precision', 'CV Recall', 'CV F1', 'CV ROC AUC']
heatmap_data = df_results[['Model'] + metrics_to_show].set_index('Model')
sns.heatmap (heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2, 
           vmin=0.85, vmax=1.0, cbar_kws={'label': 'Score'})
ax2.set_title('Model Performance Heatmap', fontweight='bold')

# 3. Training time vs accuracy
ax3 = axes[1, 0]
ax3.scatter (df_results['Fit Time (s)'], df_results['CV Accuracy'], 
           s=200, alpha=0.6, edgecolors='black', linewidth=2)

for idx, row in df_results.iterrows():
    ax3.annotate (row['Model'], (row['Fit Time (s)'], row['CV Accuracy']),
                fontsize=8, ha='center')

ax3.set_xlabel('Training Time (seconds)')
ax3.set_ylabel('CV Accuracy')
ax3.set_title('Accuracy vs Training Time', fontweight='bold')
ax3.grid (alpha=0.3)

# 4. Overfitting analysis
ax4 = axes[1, 1]
models_sorted_overfit = df_results.sort_values('Overfitting', ascending=True)
y_pos = np.arange (len (models_sorted_overfit))

bars = ax4.barh (y_pos, models_sorted_overfit['Overfitting'])
ax4.set_yticks (y_pos)
ax4.set_yticklabels (models_sorted_overfit['Model'])
ax4.set_xlabel('Train - Test Accuracy Gap')
ax4.set_title('Overfitting Analysis', fontweight='bold')
ax4.axvline (x=0, color='red', linestyle='--', alpha=0.5)
ax4.grid (axis='x', alpha=0.3)

# Color code
colors = ['green' if abs (x) < 0.05 else 'orange' if abs (x) < 0.10 else 'red' 
          for x in models_sorted_overfit['Overfitting']]
for bar, color in zip (bars, colors):
    bar.set_color (color)
    bar.set_alpha(0.7)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("\\nComparison plots saved to 'model_comparison.png'")
\`\`\`

## Statistical Comparison

\`\`\`python
from scipy import stats

print("\\n" + "="*70)
print("Statistical Significance Testing")
print("="*70)

# Get detailed CV scores for top 3 models
top_3_models = df_results.nsmallest(3, 'CV Accuracy')['Model'].values

detailed_scores = {}
for name in top_3_models:
    pipeline = pipelines[name]
    scores = cross_val_score (pipeline, X_train, y_train, cv=10, scoring='accuracy')
    detailed_scores[name] = scores

# Pairwise t-tests
print("\\nPairwise t-tests (paired, 10-fold CV):")
print("-"*70)

model_names = list (detailed_scores.keys())
for i in range (len (model_names)):
    for j in range (i+1, len (model_names)):
        model1 = model_names[i]
        model2 = model_names[j]
        
        scores1 = detailed_scores[model1]
        scores2 = detailed_scores[model2]
        
        t_stat, p_value = stats.ttest_rel (scores1, scores2)
        
        mean_diff = scores1.mean() - scores2.mean()
        
        print(f"\\n{model1} vs {model2}:")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            winner = model1 if mean_diff > 0 else model2
            print(f"  → {winner} is significantly better (p < 0.05)")
        else:
            print(f"  → No significant difference (p >= 0.05)")

print("\\n💡 Interpretation:")
print("  p < 0.05: Statistically significant difference")
print("  p >= 0.05: Difference could be due to chance")
print("  Always consider practical significance too!")
\`\`\`

## No Free Lunch Theorem

\`\`\`python
print("\\n" + "="*70)
print("No Free Lunch Theorem")
print("="*70)

print("""
The No Free Lunch (NFL) Theorem states:

"Averaged over ALL possible problems, every algorithm performs equally well."

Key Implications:
1. No single algorithm is best for ALL problems
2. Algorithm performance depends on the specific problem
3. Domain knowledge and experimentation are essential
4. Must try multiple approaches

What this means for practitioners:
• Don't assume your favorite algorithm will work best
• Always benchmark against multiple baselines
• Understand your data and problem domain
• Consider algorithm assumptions vs data characteristics

Example: Linear models excel when relationships are linear,
but fail on highly non-linear data where tree-based methods shine.
""")

# Demonstrate with synthetic datasets
from sklearn.datasets import make_classification, make_moons, make_circles

datasets = {
    'Linearly Separable': make_classification (n_samples=500, n_features=2, n_informative=2,
                                             n_redundant=0, n_clusters_per_class=1, 
                                             class_sep=2.0, random_state=42),
    'Moons': make_moons (n_samples=500, noise=0.1, random_state=42),
    'Circles': make_circles (n_samples=500, noise=0.1, factor=0.5, random_state=42)
}

simple_models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier (max_depth=5, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42)
}

nfl_results = []

for dataset_name, (X_data, y_data) in datasets.items():
    for model_name, model in simple_models.items():
        scores = cross_val_score (model, X_data, y_data, cv=5)
        nfl_results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Accuracy': scores.mean(),
            'Std': scores.std()
        })

df_nfl = pd.DataFrame (nfl_results)
pivot_nfl = df_nfl.pivot (index='Model', columns='Dataset', values='Accuracy')

print("\\nPerformance across different problem types:")
print(pivot_nfl.to_string())

print("\\nObservations:")
print("  • Logistic Regression: Best on linearly separable")
print("  • Decision Tree & SVM: Better on non-linear (moons, circles)")
print("  • No single winner across all datasets")
\`\`\`

## Practical Model Selection Criteria

\`\`\`python
print("\\n" + "="*70)
print("Practical Model Selection Considerations")
print("="*70)

selection_criteria = {
    'Performance': {
        'Weight': '★★★★★',
        'Considerations': [
            'Accuracy/F1/AUC on validation set',
            'Consistency across CV folds',
            'Performance on class imbalance',
            'Robustness to outliers'
        ]
    },
    'Training Time': {
        'Weight': '★★★',
        'Considerations': [
            'How often will model be retrained?',
            'Available computational resources',
            'Real-time vs batch training',
            'Cost of cloud compute'
        ]
    },
    'Prediction Time': {
        'Weight': '★★★★',
        'Considerations': [
            'Latency requirements (<10ms, <100ms, <1s?)',
            'Batch vs real-time predictions',
            'Number of predictions per second',
            'Mobile/edge deployment constraints'
        ]
    },
    'Interpretability': {
        'Weight': '★★★★',
        'Considerations': [
            'Regulatory requirements (GDPR, fair lending)',
            'Stakeholder understanding',
            'Debugging and trust',
            'Feature importance needs'
        ]
    },
    'Maintenance': {
        'Weight': '★★★',
        'Considerations': [
            'Model drift monitoring',
            'Retraining frequency',
            'Version control',
            'Team expertise'
        ]
    },
    'Scalability': {
        'Weight': '★★★',
        'Considerations': [
            'Data volume growth',
            'Feature dimensionality',
            'Distributed training needs',
            'Memory constraints'
        ]
    }
}

for criterion, details in selection_criteria.items():
    print(f"\\n{criterion} {details['Weight']}:")
    for consideration in details['Considerations']:
        print(f"  • {consideration}")
\`\`\`

## Decision Framework

\`\`\`python
print("\\n" + "="*70)
print("Model Selection Decision Framework")
print("="*70)

decision_tree_text = """
START
│
├─ Need Interpretability?
│  ├─ YES → Logistic Regression, Decision Tree, Linear Models
│  └─ NO → Continue
│
├─ Have Lots of Data (>100K samples)?
│  ├─ YES → Deep Learning, Gradient Boosting, Random Forest
│  └─ NO → Continue
│
├─ High Dimensional Data (>1000 features)?
│  ├─ YES → Linear Models with Regularization, Tree Ensembles
│  └─ NO → Continue
│
├─ Need Fast Predictions (<1ms)?
│  ├─ YES → Linear Models, Small Decision Trees
│  └─ NO → Continue
│
├─ Non-linear Relationships?
│  ├─ YES → SVM (RBF), Random Forest, Neural Networks
│  └─ NO → Logistic Regression, Linear SVM
│
└─ RESULT: Try top 3-5 candidates, compare with CV
"""

print(decision_tree_text)

# Recommendation function
def recommend_models (problem_characteristics):
    """Recommend models based on problem characteristics."""
    
    recommendations = []
    
    # Always recommend these as baselines
    recommendations.append(('Logistic Regression', 'Baseline - always try first'))
    recommendations.append(('Random Forest', 'Robust default choice'))
    
    # Based on characteristics
    if problem_characteristics.get('interpretability_required'):
        recommendations.append(('Decision Tree', 'Highly interpretable'))
        recommendations.append(('Linear Models', 'Coefficients are interpretable'))
    
    if problem_characteristics.get('large_dataset', False):
        recommendations.append(('Gradient Boosting', 'Excellent for large datasets'))
        recommendations.append(('Neural Network', 'Can learn complex patterns'))
    
    if problem_characteristics.get('high_dimensional', False):
        recommendations.append(('Lasso/Ridge', 'Handles high dimensions well'))
        recommendations.append(('Random Forest', 'Implicit feature selection'))
    
    if problem_characteristics.get('fast_prediction', False):
        recommendations.append(('Logistic Regression', 'Fastest predictions'))
        recommendations.append(('Small Decision Tree', 'Fast and simple'))
    
    if problem_characteristics.get('non_linear', False):
        recommendations.append(('SVM (RBF)', 'Good for non-linear boundaries'))
        recommendations.append(('Gradient Boosting', 'Captures non-linearity'))
    
    if problem_characteristics.get('categorical_features', False):
        recommendations.append(('CatBoost', 'Optimized for categorical features'))
        recommendations.append(('LightGBM', 'Good categorical handling'))
    
    return recommendations

# Example
example_problem = {
    'interpretability_required': False,
    'large_dataset': True,
    'high_dimensional': False,
    'fast_prediction': False,
    'non_linear': True,
    'categorical_features': False
}

print("\\nExample: Large dataset, non-linear, no interpretability requirement")
recommended = recommend_models (example_problem)
print("\\nRecommended models to try:")
for i, (model, reason) in enumerate (recommended, 1):
    print(f"  {i}. {model}: {reason}")
\`\`\`

## Ensemble Model Selection

\`\`\`python
from sklearn.ensemble import VotingClassifier, StackingClassifier

print("\\n" + "="*70)
print("Ensemble: Combining Multiple Models")
print("="*70)

# Instead of choosing one model, combine them!

# Voting Classifier
print("\\n1. Voting Classifier (combine predictions)")

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression (max_iter=10000)),
        ('rf', RandomForestClassifier (n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier (n_estimators=100, random_state=42))
    ],
    voting='soft'
)

# Scale features
from sklearn.preprocessing import StandardScaler
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().transform(X_test)

voting_clf.fit(X_train_scaled, y_train)
voting_score = voting_clf.score(X_test_scaled, y_test)

print(f"\\nVoting Classifier Test Accuracy: {voting_score:.4f}")

# Compare with individual models
individual_scores = {}
for name, model in voting_clf.named_estimators_.items():
    score = model.score(X_test_scaled, y_test)
    individual_scores[name] = score
    print(f"  {name}: {score:.4f}")

if voting_score > max (individual_scores.values()):
    print(f"\\n✓ Ensemble outperforms all individual models!")
else:
    print(f"\\n→ Best individual model still competitive")

# Stacking Classifier
print("\\n2. Stacking Classifier (meta-learning)")

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression (max_iter=10000)),
        ('rf', RandomForestClassifier (n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier (n_estimators=50, random_state=42))
    ],
    final_estimator=LogisticRegression (max_iter=10000),
    cv=5
)

stacking_clf.fit(X_train_scaled, y_train)
stacking_score = stacking_clf.score(X_test_scaled, y_test)

print(f"\\nStacking Classifier Test Accuracy: {stacking_score:.4f}")
print("\\n✓ Stacking often achieves best performance by learning how to combine models")
\`\`\`

## Trading Application

\`\`\`python
print("\\n" + "="*70)
print("Trading Application: Model Selection for Price Prediction")
print("="*70)

# Generate trading data
np.random.seed(42)
n_days = 1000
returns = np.random.randn (n_days) * 0.015
prices = 100 * np.exp (np.cumsum (returns))

# Create features
features_df = pd.DataFrame({'price': prices})
features_df['returns'] = features_df['price'].pct_change()
features_df['sma_10'] = features_df['price'].rolling(10).mean()
features_df['sma_50'] = features_df['price'].rolling(50).mean()
features_df['volatility'] = features_df['returns'].rolling(20).std()
features_df['momentum'] = features_df['price'].pct_change(20)

# Target: next day direction
features_df['target'] = (features_df['price'].shift(-1) > features_df['price']).astype (int)
features_df = features_df.dropna()

X_trading = features_df[['returns', 'sma_10', 'sma_50', 'volatility', 'momentum']].values
y_trading = features_df['target'].values

# Time-series split
split_point = int(0.7 * len(X_trading))
X_train_t = X_trading[:split_point]
X_test_t = X_trading[split_point:]
y_train_t = y_trading[:split_point]
y_test_t = y_trading[split_point:]

# Compare models for trading
trading_models = {
    'Logistic Regression': LogisticRegression (max_iter=10000),
    'Random Forest': RandomForestClassifier (n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier (n_estimators=100, random_state=42)
}

print("\\nComparing models for trading strategy:")
print("-"*70)

from sklearn.metrics import accuracy_score, precision_score, recall_score

for name, model in trading_models.items():
    model.fit(X_train_t, y_train_t)
    y_pred_t = model.predict(X_test_t)
    
    accuracy = accuracy_score (y_test_t, y_pred_t)
    precision = precision_score (y_test_t, y_pred_t)
    recall = recall_score (y_test_t, y_pred_t)
    
    print(f"\\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (of predicted UPs, how many correct?)")
    print(f"  Recall:    {recall:.4f} (of actual UPs, how many caught?)")

print("\\n💡 Trading Model Selection Considerations:")
print("  • Precision: Avoid false buy signals (cost of wrong trades)")
print("  • Recall: Catch profitable opportunities (opportunity cost)")
print("  • Stability: Consistent performance across market regimes")
print("  • Simplicity: Easier to understand and trust")
print("  • Transaction costs: More complex models may overtrade")
\`\`\`

## Key Takeaways

1. **Compare Multiple Models**: Never settle on first try
2. **Use Proper Validation**: Cross-validation or time-series split
3. **Statistical Testing**: Verify differences are significant
4. **No Free Lunch**: Best model depends on problem
5. **Beyond Accuracy**: Consider speed, interpretability, maintenance
6. **Ensemble Methods**: Often outperform single models
7. **Document Everything**: Track all experiments
8. **Business Context**: Technical metrics aren't everything

**Practical Workflow:**
1. Define success criteria (not just accuracy)
2. Select 5-7 candidate models
3. Compare with cross-validation
4. Statistical significance testing
5. Consider practical constraints
6. Final evaluation on test set
7. Monitor in production

## Further Reading

- Wolpert, D. H., & Macready, W. G. (1997). "No free lunch theorems for optimization"
- Caruana, R., & Niculescu-Mizil, A. (2006). "An empirical comparison of supervised learning algorithms"
- Fernández-Delgado, M., et al. (2014). "Do we need hundreds of classifiers to solve real world classification problems?"
`,
  exercises: [
    {
      prompt:
        'Build a comprehensive model selection framework that automates the comparison of multiple models, performs statistical testing, generates detailed reports, and provides recommendations based on both performance and practical criteria.',
      solution: `# Complete framework in evaluation tool`,
    },
  ],
  quizId: 'model-evaluation-optimization-model-selection',
  multipleChoiceId: 'model-evaluation-optimization-model-selection-mc',
};
