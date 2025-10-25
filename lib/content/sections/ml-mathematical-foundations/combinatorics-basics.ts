/**
 * Combinatorics Basics Section
 */

export const combinatoricsbasicsSection = {
  id: 'combinatorics-basics',
  title: 'Combinatorics Basics',
  content: `
# Combinatorics Basics

## Introduction

Combinatorics is the mathematics of counting. In machine learning and data science, we constantly count: possible model configurations, dataset splits, feature combinations, and more. Understanding combinatorics is essential for probability theory, analyzing algorithm complexity, and understanding model capacity.

## Fundamental Counting Principles

### Addition Principle

**If there are n ways to do task A and m ways to do task B, and these tasks cannot be done simultaneously, there are n + m ways to do either task A or task B.**

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, comb, perm

# Example: Classification models
linear_models = 3  # Linear regression, logistic regression, perceptron
tree_models = 4    # Decision tree, random forest, gradient boosting, XGBoost

total_model_choices = linear_models + tree_models
print(f"Linear models: {linear_models}")
print(f"Tree models: {tree_models}")
print(f"Total model choices: {total_model_choices}")

# ML Application: Feature selection methods
correlation_methods = 2  # Pearson, Spearman
mutual_info_methods = 1
tree_based_methods = 3   # RF importance, XGBoost gain, permutation

total_selection_methods = correlation_methods + mutual_info_methods + tree_based_methods
print(f"\\nTotal feature selection methods: {total_selection_methods}")
\`\`\`

### Multiplication Principle

**If there are n ways to do task A and m ways to do task B, there are n × m ways to do both tasks.**

\`\`\`python
# Example: Hyperparameter grid search
learning_rates = [0.001, 0.01, 0.1]         # 3 options
batch_sizes = [16, 32, 64, 128]             # 4 options
optimizers = ['SGD', 'Adam', 'RMSprop']     # 3 options

total_combinations = len (learning_rates) * len (batch_sizes) * len (optimizers)

print("Hyperparameter Grid:")
print(f"Learning rates: {len (learning_rates)}")
print(f"Batch sizes: {len (batch_sizes)}")
print(f"Optimizers: {len (optimizers)}")
print(f"Total combinations: {total_combinations}")

# Generate all combinations
from itertools import product

configs = list (product (learning_rates, batch_sizes, optimizers))
print(f"\\nFirst 5 configurations:")
for i, (lr, bs, opt) in enumerate (configs[:5]):
    print(f"{i+1}. LR={lr}, Batch={bs}, Optimizer={opt}")
\`\`\`

## Permutations

### Definition

**Permutation**: Arrangement of objects where **order matters**

**Formula**: P(n, r) = n!/(n-r)! = n × (n-1) × ... × (n-r+1)

Number of ways to arrange r objects from n total objects.

\`\`\`python
from math import perm

def permutations_formula (n, r):
    """Calculate P(n, r) = n!/(n-r)!"""
    return perm (n, r)

# Example: Feature ordering for sequential model
features = ['age', 'income', 'credit_score', 'debt_ratio', 'employment']
n_features = len (features)

# How many ways to order 3 features?
r = 3
p = permutations_formula (n_features, r)

print(f"Total features: {n_features}")
print(f"Selecting: {r} features")
print(f"Permutations P({n_features}, {r}) = {p}")

# Generate actual permutations
from itertools import permutations as perm_iter

feature_perms = list (perm_iter (features, r))
print(f"\\nFirst 10 orderings:")
for i, perm in enumerate (feature_perms[:10]):
    print(f"{i+1}. {' → '.join (perm)}")
\`\`\`

### Special Case: All Objects

P(n, n) = n! (all n objects arranged)

\`\`\`python
# Example: Order of applying data augmentations
augmentations = ['rotate', 'flip', 'crop', 'color_jitter']
n = len (augmentations)

total_orderings = factorial (n)
print(f"Augmentations: {augmentations}")
print(f"Total possible orderings: {total_orderings}")

# Show some orderings
orderings = list (perm_iter (augmentations))
print(f"\\nSome orderings:")
for i, ordering in enumerate (orderings[:6]):
    print(f"{i+1}. {' → '.join (ordering)}")
\`\`\`

## Combinations

### Definition

**Combination**: Selection of objects where **order doesn't matter**

**Formula**: C(n, r) = n!/(r!(n-r)!) = "n choose r"

Number of ways to select r objects from n total objects.

\`\`\`python
from math import comb

def combinations_formula (n, r):
    """Calculate C(n, r) = n!/(r!(n-r)!)"""
    return comb (n, r)

# Example: Selecting features for a model
all_features = ['age', 'income', 'education', 'credit_score', 'debt_ratio', 
                'employment', 'location', 'marital_status']
n_features = len (all_features)
select_k = 3

c = combinations_formula (n_features, select_k)

print(f"Total features available: {n_features}")
print(f"Selecting: {select_k} features")
print(f"Combinations C({n_features}, {select_k}) = {c}")

# Generate actual combinations
from itertools import combinations as comb_iter

feature_combs = list (comb_iter (all_features, select_k))
print(f"\\nFirst 10 feature combinations:")
for i, combo in enumerate (feature_combs[:10]):
    print(f"{i+1}. {combo}")
\`\`\`

### Permutations vs Combinations

\`\`\`python
def compare_perm_comb (n, r):
    """Compare permutations vs combinations"""
    p = permutations_formula (n, r)
    c = combinations_formula (n, r)
    
    print(f"n={n}, r={r}:")
    print(f"  Permutations (order matters): {p}")
    print(f"  Combinations (order doesn't matter): {c}")
    print(f"  Ratio P/C = {p/c:.1f} = {r}!")
    print(f"  For each combination, there are {r}! = {factorial (r)} permutations")

compare_perm_comb(5, 3)
print()
compare_perm_comb(10, 4)
\`\`\`

## Pascal\'s Triangle and Binomial Coefficients

C(n, r) are called **binomial coefficients** and appear in Pascal's triangle:

\`\`\`python
def pascals_triangle (rows):
    """Generate Pascal's triangle"""
    triangle = []
    for n in range (rows):
        row = [comb (n, r) for r in range (n + 1)]
        triangle.append (row)
    return triangle

# Generate and display
triangle = pascals_triangle(8)

print("Pascal\'s Triangle:")
for n, row in enumerate (triangle):
    spaces = ' ' * (len (triangle) - n - 1) * 2
    print(spaces + '  '.join (f'{x:3}' for x in row))

# Properties
print(f"\\nProperties:")
print(f"1. Symmetry: C(n, r) = C(n, n-r)")
print(f"   C(6, 2) = {comb(6, 2)}, C(6, 4) = {comb(6, 4)}")

print(f"\\n2. Sum of row n = 2^n:")
for n in range(6):
    row_sum = sum (triangle[n])
    print(f"   Row {n}: sum = {row_sum} = 2^{n} = {2**n}")

print(f"\\n3. Pascal\'s identity: C(n, r) = C(n-1, r-1) + C(n-1, r)")
n, r = 5, 2
print(f"   C(5, 2) = {comb(5, 2)}")
print(f"   C(4, 1) + C(4, 2) = {comb(4, 1)} + {comb(4, 2)} = {comb(4, 1) + comb(4, 2)}")
\`\`\`

## Applications in Machine Learning

### k-Fold Cross-Validation

\`\`\`python
def count_cv_orderings (n_samples, k_folds):
    """
    Count possible ways to partition n samples into k folds
    This is the multinomial coefficient
    """
    fold_size = n_samples // k_folds
    # Simplified: C(n, fold_size) for first fold, C(n-fold_size, fold_size) for second, etc.
    # Actual formula is multinomial coefficient
    
    # For equal-sized folds: n! / (fold_size!)^k * k!
    # Divided by k! if folds are unordered
    
    return comb (n_samples, fold_size)

n_samples = 100
k = 5
fold_size = n_samples // k

print(f"Cross-validation combinations:")
print(f"Samples: {n_samples}, Folds: {k}, Fold size: {fold_size}")
print(f"Ways to choose first fold: C({n_samples}, {fold_size}) = {comb (n_samples, fold_size)}")
print(f"\\nThis number is astronomical! Good thing we use random splitting.")
\`\`\`

### Feature Selection

\`\`\`python
def count_feature_subsets (n_features, min_k=1, max_k=None):
    """Count all possible feature subsets of size k to max_k"""
    if max_k is None:
        max_k = n_features
    
    total = 0
    counts = {}
    
    for k in range (min_k, max_k + 1):
        count = comb (n_features, k)
        counts[k] = count
        total += count
    
    return counts, total

n_features = 20
counts, total = count_feature_subsets (n_features, min_k=1, max_k=10)

print(f"Feature subset counts (n={n_features}):")
for k, count in counts.items():
    print(f"  {k} features: {count:,} combinations")

print(f"\\nTotal: {total:,} possible feature sets")
print(f"All possible subsets (2^n - 1): {2**n_features - 1:,}")  # -1 to exclude empty set

# Visualize
plt.figure (figsize=(10, 6))
k_values = list (counts.keys())
count_values = list (counts.values())
plt.bar (k_values, count_values)
plt.xlabel('Number of features selected (k)')
plt.ylabel('Number of combinations C(n, k)')
plt.title (f'Feature Subset Counts (n={n_features})')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

### Model Ensemble Combinations

\`\`\`python
def ensemble_combinations (n_models, min_ensemble_size=2):
    """Count ways to form ensembles from n models"""
    combinations = {}
    
    for k in range (min_ensemble_size, n_models + 1):
        combinations[k] = comb (n_models, k)
    
    return combinations

n_models = 10
ensembles = ensemble_combinations (n_models)

print(f"Ensemble combinations from {n_models} models:")
for size, count in ensembles.items():
    print(f"  Ensemble of {size} models: {count:,} ways")

total = sum (ensembles.values())
print(f"\\nTotal possible ensembles: {total:,}")
\`\`\`

### Hyperparameter Search Space

\`\`\`python
def hyperparameter_search_space (param_grid):
    """Calculate size of hyperparameter search space"""
    # Grid search: multiply all options
    grid_size = 1
    for param, values in param_grid.items():
        grid_size *= len (values)
    
    return grid_size

# Example search space
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128],
    'num_layers': [2, 3, 4, 5],
    'hidden_size': [64, 128, 256, 512],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
    'optimizer': ['sgd', 'adam', 'rmsprop']
}

space_size = hyperparameter_search_space (param_grid)

print("Hyperparameter Search Space:")
for param, values in param_grid.items():
    print(f"  {param}: {len (values)} options")

print(f"\\nTotal configurations: {space_size:,}")
print(f"\\nIf each config takes 10 minutes to train:")
print(f"  Total time: {space_size * 10 / 60:.1f} hours = {space_size * 10 / 60 / 24:.1f} days")
\`\`\`

## Trading Applications

### Portfolio Combinations

\`\`\`python
def portfolio_combinations (n_assets, portfolio_size):
    """
    Count ways to select portfolio of given size from n assets
    """
    return comb (n_assets, portfolio_size)

# Example: Selecting stocks for portfolio
available_stocks = 100
portfolio_size = 10

combinations = portfolio_combinations (available_stocks, portfolio_size)

print(f"Portfolio Selection:")
print(f"Available stocks: {available_stocks}")
print(f"Portfolio size: {portfolio_size}")
print(f"Possible portfolios: {combinations:,}")

# Different portfolio sizes
print(f"\\nPortfolio size vs combinations:")
for size in [5, 10, 15, 20]:
    count = portfolio_combinations (available_stocks, size)
    print(f"  {size} stocks: {count:,} portfolios")
\`\`\`

### Trading Strategy Combinations

\`\`\`python
# Example: Combining technical indicators
indicators = ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger', 'Stochastic', 'ADX', 'OBV']
n_indicators = len (indicators)

print("Trading Strategy Combinations:")
print(f"Available indicators: {n_indicators}")

for k in range(2, min(6, n_indicators + 1)):
    count = comb (n_indicators, k)
    print(f"  Using {k} indicators: {count} combinations")

# Show some specific combinations
print(f"\\nExample 3-indicator strategies:")
strategies = list (comb_iter (indicators, 3))
for i, strategy in enumerate (strategies[:10]):
    print(f"  {i+1}. {', '.join (strategy)}")
\`\`\`

## Summary

- **Counting principles**: Addition (OR), Multiplication (AND)
- **Permutations**: P(n, r) = n!/(n-r)! (order matters)
- **Combinations**: C(n, r) = n!/(r!(n-r)!) (order doesn't matter)
- **Pascal's triangle**: Contains binomial coefficients
- **Key insight**: P(n, r) = C(n, r) × r!

**ML Applications**:
- Cross-validation fold partitions
- Feature subset selection (2^n possible subsets!)
- Hyperparameter grid search space size
- Model ensemble combinations
- Dataset split possibilities

**Trading Applications**:
- Portfolio asset selection
- Indicator combinations for strategies
- Backtesting scenario enumeration
`,
};
