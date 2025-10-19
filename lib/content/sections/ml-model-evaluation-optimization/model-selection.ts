/**
 * Section: Model Selection
 * Module: Model Evaluation & Optimization
 *
 * Covers comparing models, statistical testing, and selecting the best model for deployment
 */

export const modelSelection = {
  id: 'model-selection',
  title: 'Model Selection',
  content: `
# Model Selection

## Introduction

You've trained multiple models—logistic regression, random forest, gradient boosting, neural networks. Now what? How do you choose which model to deploy?

**The Challenge**: Model selection isn't just about the highest accuracy. You must consider performance, interpretability, training time, inference latency, maintenance cost, and business constraints.

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Model Selection: Comprehensive Comparison")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.bincount(y)} (imbalance ratio: {y.sum()/len(y):.2f})")
\`\`\`

## Comparing Multiple Models

\`\`\`python
print("\\n" + "="*70)
print("Step 1: Training Multiple Models")
print("="*70)

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layers=(100, 50), max_iter=1000, random_state=42),
}

# Store results
results = []

for name, model in models.items():
    print(f"\\nTraining {name}...")
    
    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    predict_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc,
        'CV AUC (mean)': cv_scores.mean(),
        'CV AUC (std)': cv_scores.std(),
        'Train Time (s)': train_time,
        'Predict Time (s)': predict_time,
        'Predict Time (ms/sample)': (predict_time / len(X_test)) * 1000,
    })
    
    print(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(f"  Train: {train_time:.3f}s, Predict: {predict_time:.4f}s")

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\\n" + "="*70)
print("Model Comparison Results")
print("="*70)
print(results_df.to_string(index=False))
\`\`\`

## Ranking Models by Different Criteria

\`\`\`python
print("\\n" + "="*70)
print("Ranking Models by Different Criteria")
print("="*70)

criteria = ['Accuracy', 'AUC', 'F1', 'Train Time (s)', 'Predict Time (ms/sample)']

for criterion in criteria:
    ascending = 'Time' in criterion  # Lower is better for time
    ranked = results_df.sort_values(criterion, ascending=ascending)
    
    print(f"\\n{criterion}:")
    for idx, row in ranked.iterrows():
        indicator = "🥇" if idx == ranked.index[0] else "🥈" if idx == ranked.index[1] else "🥉" if idx == ranked.index[2] else "  "
        value = row[criterion]
        if 'Time' in criterion:
            print(f"  {indicator} {row['Model']:20s} {value:.4f}")
        else:
            print(f"  {indicator} {row['Model']:20s} {value:.4f}")

print("\\n💡 Insight: Different models excel at different criteria!")
\`\`\`

## Statistical Significance Testing

\`\`\`python
print("\\n" + "="*70)
print("Statistical Significance Testing")
print("="*70)

# Compare top 2 models with paired t-test
from scipy.stats import ttest_rel

# Get top 2 models by AUC
top_2_models = results_df.nlargest(2, 'AUC')['Model'].tolist()
model_1_name = top_2_models[0]
model_2_name = top_2_models[1]

model_1 = models[model_1_name]
model_2 = models[model_2_name]

print(f"\\nComparing: {model_1_name} vs {model_2_name}")

# Perform cross-validation to get paired scores
from sklearn.model_selection import cross_validate

cv_results_1 = cross_validate(model_1, X_train, y_train, cv=10, scoring='roc_auc')
cv_results_2 = cross_validate(model_2, X_train, y_train, cv=10, scoring='roc_auc')

scores_1 = cv_results_1['test_score']
scores_2 = cv_results_2['test_score']

# Paired t-test
t_stat, p_value = ttest_rel(scores_1, scores_2)

print(f"\\n{model_1_name}:")
print(f"  Mean AUC: {scores_1.mean():.4f} (+/- {scores_1.std():.4f})")
print(f"\\n{model_2_name}:")
print(f"  Mean AUC: {scores_2.mean():.4f} (+/- {scores_2.std():.4f})")

print(f"\\nPaired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    better_model = model_1_name if scores_1.mean() > scores_2.mean() else model_2_name
    print(f"  ✅ {better_model} is statistically significantly better (p < {alpha})")
else:
    print(f"  ⚠️  No statistically significant difference (p >= {alpha})")
    print(f"     → Consider simpler model or other criteria (speed, interpretability)")
\`\`\`

## Model Selection Matrix

\`\`\`python
print("\\n" + "="*70)
print("Model Selection Decision Matrix")
print("="*70)

# Create a scoring system
def score_model(row):
    """Score model based on multiple weighted criteria"""
    # Normalize metrics to 0-1 scale
    perf_score = (row['AUC'] - results_df['AUC'].min()) / (results_df['AUC'].max() - results_df['AUC'].min())
    
    # Inverse for time (lower is better)
    time_score = 1 - (row['Train Time (s)'] - results_df['Train Time (s)'].min()) / (results_df['Train Time (s)'].max() - results_df['Train Time (s)'].min())
    
    latency_score = 1 - (row['Predict Time (ms/sample)'] - results_df['Predict Time (ms/sample)'].min()) / (results_df['Predict Time (ms/sample)'].max() - results_df['Predict Time (ms/sample)'].min())
    
    # Weighted combination (adjust weights based on priorities)
    weights = {
        'performance': 0.5,  # 50% weight on predictive performance
        'training_speed': 0.2,  # 20% weight on training speed
        'inference_speed': 0.3,  # 30% weight on inference speed
    }
    
    total_score = (
        weights['performance'] * perf_score +
        weights['training_speed'] * time_score +
        weights['inference_speed'] * latency_score
    )
    
    return total_score

results_df['Composite Score'] = results_df.apply(score_model, axis=1)

# Rank by composite score
ranked = results_df.sort_values('Composite Score', ascending=False)

print("\\nComposite Ranking (Performance: 50%, Training: 20%, Inference: 30%):")
for idx, row in ranked.iterrows():
    print(f"  {row['Model']:20s} Score: {row['Composite Score']:.3f}")
    print(f"    Performance: {row['AUC']:.3f}, Train: {row['Train Time (s)']:.2f}s, Inference: {row['Predict Time (ms/sample)']:.3f}ms")

print("\\n💡 Adjust weights based on your use case:")
print("  • Real-time inference: Increase inference speed weight")
print("  • Batch predictions: Decrease inference speed weight")
print("  • Critical decisions: Increase performance weight")
\`\`\`

## Model Selection Considerations

\`\`\`python
print("\\n" + "="*70)
print("Model Selection Decision Criteria")
print("="*70)

considerations = {
    "Performance": {
        "metric": "AUC, F1, Precision/Recall",
        "priority": "High for critical applications",
        "question": "Does the model meet minimum performance requirements?",
    },
    "Interpretability": {
        "metric": "Feature importance, SHAP values",
        "priority": "Critical for regulated industries (finance, healthcare)",
        "question": "Can you explain predictions to stakeholders?",
    },
    "Training Time": {
        "metric": "Seconds/minutes to train",
        "priority": "Important for frequent retraining",
        "question": "How often will you retrain? Is real-time retraining needed?",
    },
    "Inference Latency": {
        "metric": "Milliseconds per prediction",
        "priority": "Critical for real-time applications",
        "question": "What's your latency budget? (< 100ms? < 1s?)",
    },
    "Memory Footprint": {
        "metric": "Model size (MB)",
        "priority": "Important for mobile/edge deployment",
        "question": "Deploying to edge devices or cloud?",
    },
    "Robustness": {
        "metric": "Performance on OOD data",
        "priority": "High for production systems",
        "question": "How well does it generalize to new data?",
    },
    "Maintenance": {
        "metric": "Complexity of pipeline",
        "priority": "Important for long-term deployment",
        "question": "How complex is the model? Will it be easy to maintain?",
    },
}

for criterion, details in considerations.items():
    print(f"\\n{criterion}:")
    for key, value in details.items():
        print(f"  {key.capitalize()}: {value}")

print("\\n" + "="*70)
print("Model Selection by Use Case")
print("="*70)

use_cases = {
    "Real-time Fraud Detection": {
        "priority": "Latency < 50ms, high precision",
        "recommendation": "Logistic Regression or Gradient Boosting",
        "avoid": "Deep neural networks (too slow)",
    },
    "Medical Diagnosis": {
        "priority": "High recall, interpretability",
        "recommendation": "Random Forest with feature importance",
        "avoid": "Black-box models without explanations",
    },
    "Image Classification": {
        "priority": "High accuracy, can accept latency",
        "recommendation": "Deep neural networks (CNNs)",
        "avoid": "Traditional ML (insufficient capacity)",
    },
    "Batch Predictions": {
        "priority": "High throughput, performance",
        "recommendation": "Gradient Boosting or Neural Networks",
        "avoid": "Expensive models if budget-constrained",
    },
    "Embedded/Mobile": {
        "priority": "Small model size, low latency",
        "recommendation": "Logistic Regression or small decision trees",
        "avoid": "Large ensembles or deep networks",
    },
}

for use_case, details in use_cases.items():
    print(f"\\n{use_case}:")
    for key, value in details.items():
        print(f"  {key.capitalize()}: {value}")
\`\`\`

## Practical Decision Framework

\`\`\`python
print("\\n" + "="*70)
print("Practical Model Selection Framework")
print("="*70)

framework = """
STEP 1: Define Success Criteria
  • Minimum performance threshold (e.g., AUC > 0.85)
  • Maximum latency (e.g., < 100ms per prediction)
  • Interpretability requirements (yes/no)
  • Business constraints (cost, maintenance)

STEP 2: Filter Models
  • Remove models that don't meet minimum requirements
  • E.g., if latency < 100ms required, remove slow models

STEP 3: Compare Remaining Models
  • Use cross-validation for robust comparison
  • Statistical testing if performance is close
  • Consider performance variance (not just mean)

STEP 4: Consider Non-Performance Factors
  • Training time (how often will you retrain?)
  • Inference latency (batch or real-time?)
  • Model size (cloud or edge deployment?)
  • Interpretability (regulated industry?)
  • Maintenance cost (team expertise?)

STEP 5: Pilot Testing
  • Deploy top 2-3 candidates to staging
  • A/B test with real traffic if possible
  • Monitor performance, latency, errors
  • Measure business impact (not just metrics)

STEP 6: Make Decision
  • Select model that best balances all factors
  • Document decision rationale
  • Plan for monitoring and retraining

When in doubt: Start simple (logistic regression), add complexity only if needed.
"""

print(framework)

print("\\n" + "="*70)
print("Common Mistakes in Model Selection")
print("="*70)

mistakes = {
    "Chasing small improvements": "0.01 AUC improvement may not matter in practice",
    "Ignoring inference time": "Slow models can't be deployed to production",
    "Not testing statistical significance": "Performance difference may be due to chance",
    "Overfitting to test set": "Evaluating too many models on same test set",
    "Forgetting about maintenance": "Complex ensembles are hard to maintain long-term",
    "Not considering business context": "Metrics don't always align with business value",
}

for mistake, explanation in mistakes.items():
    print(f"\\n❌ {mistake}")
    print(f"   → {explanation}")
\`\`\`

## Key Takeaways

1. **Model selection is multi-faceted**: Performance, speed, interpretability, maintenance
2. **Use cross-validation**: Get robust performance estimates with uncertainty
3. **Statistical testing**: Determine if performance differences are significant
4. **Context matters**: Best model depends on your specific use case
5. **Start simple**: Logistic regression baseline, add complexity as needed
6. **Consider total cost**: Training time, inference latency, maintenance
7. **A/B test when possible**: Real-world performance beats offline metrics
8. **Document decisions**: Record why you chose a particular model

**Decision Framework:**
1. Define minimum requirements (performance, latency, interpretability)
2. Filter models that don't meet requirements
3. Compare remaining models with cross-validation
4. Use statistical testing if performance is close
5. Consider non-performance factors (speed, maintenance, cost)
6. Pilot test top candidates
7. Choose model that best balances all factors

Remember: The "best" model is the one that best serves your specific use case and constraints!
`,
};
