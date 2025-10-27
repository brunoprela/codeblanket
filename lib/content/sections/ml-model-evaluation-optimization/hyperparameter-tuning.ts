/**
 * Section: Hyperparameter Tuning
 * Module: Model Evaluation & Optimization
 *
 * Covers grid search, random search, Bayesian optimization, and hyperparameter tuning strategies
 */

export const hyperparameterTuning = {
  id: 'hyperparameter-tuning',
  title: 'Hyperparameter Tuning',
  content: `
# Hyperparameter Tuning

## Introduction

Model parameters are learned from data (e.g., neural network weights), but **hyperparameters** are settings we choose before training (e.g., learning rate, tree depth). Choosing the right hyperparameters can mean the difference between a mediocre model and state-of-the-art performance.

**The Challenge**: The hyperparameter space is vast, and evaluating each configuration is expensive. How do we find the best settings efficiently?

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from scipy.stats import uniform, randint
import time

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Hyperparameter Tuning Overview")
print("="*60)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
\`\`\`

## Manual Tuning (Baseline)

\`\`\`python
# Try a few configurations manually
print("\\n" + "="*60)
print("Manual Tuning (Inefficient but educational)")
print("="*60)

configs = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 20},
]

results = []
for config in configs:
    start = time.time()
    model = RandomForestRegressor (random_state=42, **config)
    scores = cross_val_score (model, X_train, y_train, cv=5, scoring='r2')
    duration = time.time() - start
    
    results.append({
        'config': config,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'time': duration
    })
    
    print(f"Config: {config}")
    print(f"  CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  Time: {duration:.2f}s")

print("\\n⚠️ Problem: Manual tuning is tedious and misses optimal combinations")
\`\`\`

## Grid Search: Exhaustive Search

Grid search tries every combination of hyperparameters in a predefined grid.

\`\`\`python
print("\\n" + "="*60)
print("Grid Search: Exhaustive Search")
print("="*60)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Calculate total combinations
total_combinations = np.prod([len (v) for v in param_grid.values()])
print(f"\\nSearching {total_combinations} combinations:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

# Perform grid search
start_time = time.time()
grid_search = GridSearchCV(
    RandomForestRegressor (random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

grid_search.fit(X_train, y_train)
duration = time.time() - start_time

print(f"\\nGrid Search Results:")
print(f"  Best params: {grid_search.best_params_}")
print(f"  Best CV score: {grid_search.best_score_:.4f}")
print(f"  Total time: {duration:.2f}s")
print(f"  Time per config: {duration/total_combinations:.2f}s")

# Evaluate on test set
test_score = grid_search.score(X_test, y_test)
print(f"  Test R²: {test_score:.4f}")

# Analyze results
results_df = pd.DataFrame (grid_search.cv_results_)
print(f"\\nTop 5 configurations:")
top_5 = results_df.nsmallest(5, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
for idx, row in top_5.iterrows():
    print(f"  Rank {int (row['rank_test_score'])}: {row['mean_test_score']:.4f} - {row['params']}")

# Visualize importance of each hyperparameter
print("\\n✅ Grid search complete")
\`\`\`

**Pros of Grid Search:**
- ✅ Guaranteed to find best combination in the grid
- ✅ Exhaustive - won't miss any combination
- ✅ Parallelizable

**Cons of Grid Search:**
- ❌ Exponentially expensive (curse of dimensionality)
- ❌ Wastes computation on unpromising regions
- ❌ Can't explore continuous spaces efficiently

## Random Search: Efficient Alternative

Random search samples random combinations from hyperparameter distributions.

\`\`\`python
print("\\n" + "="*60)
print("Random Search: Efficient Sampling")
print("="*60)

# Define hyperparameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),  # Discrete: 50-299
    'max_depth': randint(5, 30),       # Discrete: 5-29
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)  # Continuous: 0.1-1.0
}

n_iter = 50  # Number of random samples
print(f"\\nSampling {n_iter} random configurations from:")
for param, dist in param_distributions.items():
    if hasattr (dist, 'a') and hasattr (dist, 'b'):
        print(f"  {param}: Uniform({dist.a:.2f}, {dist.a + dist.b:.2f})")
    else:
        print(f"  {param}: Randint({dist.a}, {dist.b})")

# Perform random search
start_time = time.time()
random_search = RandomizedSearchCV(
    RandomForestRegressor (random_state=42),
    param_distributions,
    n_iter=n_iter,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
duration = time.time() - start_time

print(f"\\nRandom Search Results:")
print(f"  Best params: {random_search.best_params_}")
print(f"  Best CV score: {random_search.best_score_:.4f}")
print(f"  Total time: {duration:.2f}s")
print(f"  Time per config: {duration/n_iter:.2f}s")

# Compare with grid search
test_score_random = random_search.score(X_test, y_test)
print(f"  Test R²: {test_score_random:.4f}")

print(f"\\nComparison:")
print(f"  Grid Search:   {grid_search.best_score_:.4f} CV, {test_score:.4f} test ({total_combinations} configs)")
print(f"  Random Search: {random_search.best_score_:.4f} CV, {test_score_random:.4f} test ({n_iter} configs)")
print(f"  Random search is {total_combinations/n_iter:.1f}x faster!")

print("\\n✅ Random search complete")
\`\`\`

**Why Random Search Works:**

Research shows random search often finds comparable solutions to grid search with far fewer iterations, especially when:
- Some hyperparameters are more important than others
- The hyperparameter space is high-dimensional
- You have a limited computation budget

## Hyperparameter Importance

\`\`\`python
print("\\n" + "="*60)
print("Analyzing Hyperparameter Importance")
print("="*60)

# Extract results
results_df = pd.DataFrame (random_search.cv_results_)

# Analyze each hyperparameter
for param in ['n_estimators', 'max_depth', 'min_samples_split']:
    values = [params[param] for params in results_df['params']]
    scores = results_df['mean_test_score']
    
    # Calculate correlation
    correlation = np.corrcoef (values, scores)[0, 1]
    
    print(f"\\n{param}:")
    print(f"  Correlation with score: {correlation:.3f}")
    
    # Bin and show average scores
    if param == 'n_estimators':
        bins = [0, 100, 200, 300]
        labels = ['50-100', '100-200', '200-300']
    elif param == 'max_depth':
        bins = [0, 10, 20, 30]
        labels = ['5-10', '10-20', '20-30']
    else:
        continue
    
    values_binned = pd.cut (values, bins=bins, labels=labels)
    df_binned = pd.DataFrame({'bin': values_binned, 'score': scores})
    avg_scores = df_binned.groupby('bin')['score'].mean()
    
    print(f"  Average scores by range:")
    for bin_name, avg_score in avg_scores.items():
        print(f"    {bin_name}: {avg_score:.4f}")
\`\`\`

## Best Practices

\`\`\`python
print("\\n" + "="*60)
print("Hyperparameter Tuning Best Practices")
print("="*60)

practices = {
    "1. Start with defaults": "Baseline performance before tuning",
    "2. Use random search first": "Efficient exploration of space",
    "3. Use nested CV": "Unbiased performance estimates",
    "4. Set time/iteration budgets": "Don't over-optimize",
    "5. Log scale for learning rate": "Try [0.001, 0.01, 0.1, 1.0]",
    "6. Early stopping": "Stop unpromising configurations early",
    "7. Warm start": "Start from previous best",
    "8. Parallelize": "Use n_jobs=-1",
}

for practice, reason in practices.items():
    print(f"  {practice}")
    print(f"    → {reason}")

print("\\n" + "="*60)
print("Common Hyperparameters by Model Type")
print("="*60)

hyperparams = {
    "Random Forest": [
        "n_estimators: Number of trees (100-500)",
        "max_depth: Tree depth (5-30 or None)",
        "min_samples_split: Min samples to split (2-20)",
        "max_features: Features per split (sqrt, log2, 0.3-0.9)",
    ],
    "Gradient Boosting": [
        "n_estimators: Number of trees (100-1000)",
        "learning_rate: Step size (0.01-0.3)",
        "max_depth: Tree depth (3-10)",
        "subsample: Row sampling (0.5-1.0)",
    ],
    "Neural Networks": [
        "learning_rate: Step size (0.0001-0.1, log scale)",
        "batch_size: Samples per batch (32, 64, 128, 256)",
        "hidden_layers: Network architecture",
        "dropout: Regularization (0.2-0.5)",
    ],
    "SVM": [
        "C: Regularization (0.1-1000, log scale)",
        "gamma: Kernel coefficient (0.001-1, log scale)",
        "kernel: Kernel type (rbf, linear, poly)",
    ],
}

for model, params in hyperparams.items():
    print(f"\\n{model}:")
    for param in params:
        print(f"  • {param}")
\`\`\`

## Practical Example: Complete Tuning Pipeline

\`\`\`python
print("\\n" + "="*60)
print("Complete Hyperparameter Tuning Pipeline")
print("="*60)

# Step 1: Baseline
print("\\nStep 1: Baseline (default hyperparameters)")
baseline_model = RandomForestRegressor (random_state=42)
baseline_score = cross_val_score (baseline_model, X_train, y_train, cv=5, scoring='r2').mean()
print(f"  Baseline CV R²: {baseline_score:.4f}")

# Step 2: Quick random search (coarse)
print("\\nStep 2: Coarse random search (20 iterations)")
param_dist_coarse = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 30),
}
random_coarse = RandomizedSearchCV(
    RandomForestRegressor (random_state=42),
    param_dist_coarse,
    n_iter=20,
    cv=3,
    random_state=42,
    n_jobs=-1
)
random_coarse.fit(X_train, y_train)
print(f"  Best CV R²: {random_coarse.best_score_:.4f}")
print(f"  Best params: {random_coarse.best_params_}")

# Step 3: Fine-grained search around best
print("\\nStep 3: Fine grid search around best")
best_n = random_coarse.best_params_['n_estimators']
best_d = random_coarse.best_params_['max_depth']

param_grid_fine = {
    'n_estimators': [best_n - 50, best_n, best_n + 50],
    'max_depth': [best_d - 2, best_d, best_d + 2],
    'min_samples_split': [2, 5],
}
grid_fine = GridSearchCV(
    RandomForestRegressor (random_state=42),
    param_grid_fine,
    cv=5,
    n_jobs=-1
)
grid_fine.fit(X_train, y_train)
print(f"  Best CV R²: {grid_fine.best_score_:.4f}")
print(f"  Best params: {grid_fine.best_params_}")

# Step 4: Final evaluation
print("\\nStep 4: Final test set evaluation")
final_model = grid_fine.best_estimator_
test_score_final = final_model.score(X_test, y_test)
print(f"  Test R²: {test_score_final:.4f}")

# Summary
print("\\n" + "="*60)
print("Summary:")
print(f"  Baseline:     {baseline_score:.4f} CV")
print(f"  After tuning: {grid_fine.best_score_:.4f} CV, {test_score_final:.4f} test")
print(f"  Improvement:  {(grid_fine.best_score_ - baseline_score)*100:.1f}%")
\`\`\`

## Key Takeaways

1. **Hyperparameters** control model behavior (vs parameters learned from data)
2. **Grid search** exhaustively tries all combinations (expensive)
3. **Random search** samples randomly (often equally good, much faster)
4. **Start simple**: Baseline → coarse search → fine search
5. **Use cross-validation** to avoid overfitting to validation set
6. **Log scale** for learning rates and regularization
7. **Budget time**: Diminishing returns after a point
8. **Parallelize**: Use all available cores (n_jobs=-1)

**Recommended Strategy:**1. Establish baseline with defaults
2. Random search for exploration (50-100 iterations)
3. Grid search for exploitation (around best random search result)
4. Final evaluation on held-out test set

Hyperparameter tuning can improve model performance by 10-30%, making it one of the highest-ROI activities in ML!
`,
};
