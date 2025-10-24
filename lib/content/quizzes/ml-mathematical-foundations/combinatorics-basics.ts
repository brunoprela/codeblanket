/**
 * Quiz questions for Combinatorics Basics section
 */

export const combinatoricsbasicsQuiz = [
  {
    id: 'dq1-feature-selection-complexity',
    question:
      'Explain why exhaustive feature selection becomes computationally infeasible as the number of features grows. If you have n features, how many total possible feature subsets exist? Compare brute-force search with intelligent search strategies (forward selection, backward elimination, genetic algorithms). Provide complexity analysis and practical examples.',
    sampleAnswer: `The computational complexity of exhaustive feature selection grows exponentially with the number of features, making it infeasible for even moderately-sized feature sets.

**Total Number of Feature Subsets**:

For n features, the total number of non-empty subsets is: **2^n - 1**

**Why 2^n?**

For each feature, we have 2 choices: include it or exclude it.
- Feature 1: 2 choices
- Feature 2: 2 choices
- ...
- Feature n: 2 choices

Total: 2 × 2 × ... × 2 = 2^n subsets (including empty set)

We subtract 1 to exclude the empty set (no features selected).

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from math import comb
import time

def count_all_subsets(n):
    """Total non-empty subsets of n features"""
    return 2**n - 1

# Demonstrate exponential growth
n_values = range(1, 31)
subset_counts = [count_all_subsets(n) for n in n_values]

print("Feature subsets (exhaustive search):")
for n in [5, 10, 15, 20, 25, 30]:
    count = count_all_subsets(n)
    print(f"  {n} features: {count:,} subsets")

# Visualize
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_values, subset_counts, 'b-', linewidth=2)
plt.xlabel('Number of features (n)')
plt.ylabel('Number of subsets (2^n - 1)')
plt.title('Exhaustive Feature Selection: Exponential Growth')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(n_values, subset_counts, 'r-', linewidth=2)
plt.xlabel('Number of features (n)')
plt.ylabel('Number of subsets (log scale)')
plt.title('Log Scale View')
plt.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**Practical Infeasibility**:

\`\`\`python
# Estimate time for exhaustive search
def estimate_search_time(n_features, time_per_model_seconds):
    """Estimate time for exhaustive feature selection"""
    n_subsets = count_all_subsets(n_features)
    total_seconds = n_subsets * time_per_model_seconds
    
    hours = total_seconds / 3600
    days = hours / 24
    years = days / 365
    
    return n_subsets, total_seconds, hours, days, years

print("\\nTime estimates (assuming 10 seconds per model evaluation):\\n")

for n in [10, 15, 20, 25, 30]:
    subsets, secs, hrs, days, yrs = estimate_search_time(n, 10)
    print(f"{n} features: {subsets:,} subsets")
    
    if yrs >= 1:
        print(f"  Time: {yrs:,.1f} years")
    elif days >= 1:
        print(f"  Time: {days:,.1f} days")
    elif hrs >= 1:
        print(f"  Time: {hrs:,.1f} hours")
    else:
        print(f"  Time: {secs:,.1f} seconds")
    print()
\`\`\`

**Intelligent Search Strategies**:

**1. Forward Selection (Greedy)**:

Start with no features, add one feature at a time (the one that improves performance most).

Complexity: O(n²) evaluations

\`\`\`python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           random_state=42)

def forward_selection(X, y, max_features=None):
    """
    Forward feature selection
    Greedy algorithm: add one feature at a time
    """
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    
    selected = []
    remaining = list(range(n_features))
    scores = []
    evaluations = 0
    
    for iteration in range(max_features):
        best_score = -np.inf
        best_feature = None
        
        # Try adding each remaining feature
        for feature in remaining:
            candidate = selected + [feature]
            X_subset = X[:, candidate]
            
            # Evaluate
            model = LogisticRegression(max_iter=1000)
            score = cross_val_score(model, X_subset, y, cv=3).mean()
            evaluations += 1
            
            if score > best_score:
                best_score = score
                best_feature = feature
        
        # No improvement possible
        if best_feature is None:
            break
        
        # Add best feature
        selected.append(best_feature)
        remaining.remove(best_feature)
        scores.append(best_score)
        
        print(f"Iteration {iteration + 1}: Added feature {best_feature}, "
              f"Score: {best_score:.4f}")
    
    print(f"\\nTotal evaluations: {evaluations}")
    print(f"Selected features: {selected}")
    
    return selected, scores, evaluations

print("Forward Selection:")
selected, scores, evals = forward_selection(X, y, max_features=5)

# Compare to exhaustive
n = X.shape[1]
exhaustive_evals = count_all_subsets(n)
print(f"\\nComparison:")
print(f"  Forward selection: {evals} evaluations")
print(f"  Exhaustive search: {exhaustive_evals:,} evaluations")
print(f"  Speedup: {exhaustive_evals / evals:,.0f}x")
\`\`\`

**2. Backward Elimination**:

Start with all features, remove one at a time (the least important).

Complexity: O(n²) evaluations

**3. Genetic Algorithm**:

Evolutionary approach: maintain population of feature sets, evolve through mutation and crossover.

Complexity: O(p × g × n) where p=population size, g=generations

\`\`\`python
def genetic_algorithm_feature_selection(X, y, population_size=20, generations=10):
    """
    Simplified genetic algorithm for feature selection
    """
    n_features = X.shape[1]
    evaluations = 0
    
    # Initialize random population
    population = [np.random.rand(n_features) > 0.5 for _ in range(population_size)]
    
    for gen in range(generations):
        # Evaluate fitness
        fitness = []
        for individual in population:
            if not any(individual):  # At least one feature
                individual[np.random.randint(n_features)] = True
            
            X_subset = X[:, individual]
            model = LogisticRegression(max_iter=1000)
            score = cross_val_score(model, X_subset, y, cv=3).mean()
            evaluations += 1
            fitness.append(score)
        
        # Select best half
        sorted_indices = np.argsort(fitness)[::-1]
        population = [population[i] for i in sorted_indices[:population_size // 2]]
        
        # Reproduce (crossover + mutation)
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = np.random.choice(len(population), 2, replace=False)
            child = np.array([population[parent1][i] if np.random.rand() > 0.5 
                            else population[parent2][i] for i in range(n_features)])
            
            # Mutation
            if np.random.rand() < 0.1:
                flip_idx = np.random.randint(n_features)
                child[flip_idx] = not child[flip_idx]
            
            offspring.append(child)
        
        population.extend(offspring)
        
        best_score = max(fitness)
        print(f"Generation {gen + 1}: Best score = {best_score:.4f}")
    
    # Return best individual
    final_fitness = [cross_val_score(LogisticRegression(max_iter=1000), 
                                     X[:, ind], y, cv=3).mean() 
                    for ind in population]
    evaluations += len(population)
    
    best_idx = np.argmax(final_fitness)
    best_features = np.where(population[best_idx])[0]
    
    print(f"\\nTotal evaluations: {evaluations}")
    print(f"Selected features: {list(best_features)}")
    
    return best_features, evaluations

print("\\n" + "="*50)
print("Genetic Algorithm:")
best_features, evals_ga = genetic_algorithm_feature_selection(X, y)

print(f"\\nComparison:")
print(f"  Genetic algorithm: {evals_ga} evaluations")
print(f"  Forward selection: {evals} evaluations")
print(f"  Exhaustive search: {exhaustive_evals:,} evaluations")
\`\`\`

**Comparison Table**:

\`\`\`python
import pandas as pd

comparison = pd.DataFrame({
    'Method': ['Exhaustive', 'Forward Selection', 'Backward Elimination', 
               'Genetic Algorithm', 'Random Search',],
    'Complexity': ['O(2^n)', 'O(n²)', 'O(n²)', 'O(p×g×n)', 'O(k)',],
    'Guarantees Optimal': ['Yes', 'No', 'No', 'No', 'No',],
    'Feasible for n=20': ['No', 'Yes', 'Yes', 'Yes', 'Yes',],
    'Feasible for n=100': ['No', 'Maybe', 'Maybe', 'Yes', 'Yes',]
})

print("\\nFeature Selection Methods Comparison:")
print(comparison.to_string(index=False))
\`\`\`

**Key Insights**:

1. **Exponential explosion**: 2^n grows impossibly fast
   - 20 features: 1 million subsets
   - 30 features: 1 billion subsets
   - 40 features: 1 trillion subsets

2. **Greedy methods** (forward/backward): Polynomial time, but may miss optimal

3. **Evolutionary methods** (genetic algorithms): Balance exploration and exploitation

4. **Random search**: Simple baseline, surprisingly effective

5. **Modern approaches**: Regularization (L1/L2), tree-based importance

**Trading Application**:

\`\`\`python
# Stock portfolio optimization
# With 100 stocks, choosing 10 for portfolio:

n_stocks = 100
portfolio_size = 10

portfolios = comb(n_stocks, portfolio_size)
print(f"\\nPortfolio Selection:")
print(f"Stocks: {n_stocks}, Portfolio size: {portfolio_size}")
print(f"Possible portfolios: {portfolios:,}")

# If testing each portfolio takes 1 minute of backtesting:
hours = portfolios / 60
days = hours / 24
print(f"Time for exhaustive search: {days:,.0f} days")

print(f"\\nIntelligent approaches:")
print(f"  - Genetic algorithm: Evolve good portfolios")
print(f"  - Greedy: Add stocks one by one (maximize Sharpe ratio)")
print(f"  - Random search: Sample random portfolios")
print(f"  - Domain knowledge: Pre-filter to top 20 stocks, then combine")
\`\`\`

**Summary**:
- Exhaustive search: 2^n subsets (exponentially infeasible)
- Forward/Backward: n² evaluations (polynomial, practical)
- Genetic algorithms: Flexible, good for large spaces
- Trade-off: Optimality vs computational feasibility
- Real-world: Use intelligent search + domain knowledge`,
    keyPoints: [
      'Total feature subsets = 2^n - 1 (exponential growth)',
      'Exhaustive search becomes infeasible around n=20-25 features',
      'Forward selection: O(n²) greedy algorithm, adds best feature iteratively',
      'Genetic algorithms: O(p×g×n) evolutionary approach, balances exploration/exploitation',
      'Trade-off between optimality (exhaustive) and feasibility (heuristics)',
    ],
  },
  {
    id: 'dq2-permutations-augmentation',
    question:
      'In data augmentation for computer vision, if we have 5 different transformations (flip, rotate 90°, rotate 180°, rotate 270°, no transformation) that we can apply to an image, how many different augmented versions can we create if we apply exactly one transformation? What if we apply a sequence of 2 transformations? Explain how permutations relate to data augmentation strategies and discuss the trade-off between augmentation diversity and computational cost.',
    sampleAnswer: `Data augmentation uses permutations and combinations to create diverse training samples from limited data. Understanding combinatorics helps design effective augmentation strategies.

**Single Transformation**:

With 5 transformations, applying exactly one gives us **5 augmented versions** (including the original with "no transformation").

This is simply n = 5 choices.

**Sequence of 2 Transformations**:

If we apply 2 transformations in sequence WITH replacement (can repeat):
- Total: **n² = 5² = 25** different sequences

If WITHOUT replacement (no repeats):
- Total: **P(5,2) = 5!/(5-2)! = 5×4 = 20** permutations

\`\`\`python
import numpy as np
from math import factorial, perm

# Transformations
transformations = ['flip', 'rotate_90', 'rotate_180', 'rotate_270', 'no_op',]
n = len(transformations)

print("Single transformation:")
print(f"  Options: {n}")
print(f"  Augmented versions: {n}")

print("\\nSequence of 2 transformations:")
print(f"  WITH replacement (can repeat): {n**2}")
print(f"  WITHOUT replacement (no repeats): {perm(n, 2)}")

# Generate all 2-transformation sequences (with replacement)
sequences = []
for t1 in transformations:
    for t2 in transformations:
        sequences.append(f"{t1} → {t2}")

print(f"\\nTotal sequences: {len(sequences)}")
print("\\nExample sequences:")
for seq in sequences[:10]:
    print(f"  {seq}")
\`\`\`

**Data Augmentation Strategy**:

**1. Simple Augmentation (Single Transformation)**:

\`\`\`python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define transformations
simple_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(90),
    transforms.RandomRotation(180),
    transforms.RandomRotation(270),
    transforms.Lambda(lambda x: x),  # no-op
]

def augment_single(image, num_augmentations=5):
    """Apply each transformation once"""
    augmented = []
    for transform in simple_transforms[:num_augmentations]:
        aug_img = transform(image)
        augmented.append(aug_img)
    return augmented

# With 1,000 original images and 5 transformations
original_images = 1000
augmented_per_image = 5
total_samples = original_images * augmented_per_image

print(f"Simple augmentation:")
print(f"  Original images: {original_images:,}")
print(f"  Transformations per image: {augmented_per_image}")
print(f"  Total training samples: {total_samples:,}")
\`\`\`

**2. Compositional Augmentation (Sequence of Transformations)**:

\`\`\`python
import itertools

def augment_compositional(image, transformations, seq_length=2):
    """Apply sequences of transformations"""
    augmented = []
    
    # Generate all permutations of length seq_length
    for perm in itertools.product(transformations, repeat=seq_length):
        aug_img = image.copy()
        for transform in perm:
            aug_img = transform(aug_img)
        augmented.append(aug_img)
    
    return augmented

# With sequences of length 2
seq_length = 2
augmented_per_image = len(transformations) ** seq_length

print(f"\\nCompositional augmentation (length {seq_length}):")
print(f"  Transformations available: {len(transformations)}")
print(f"  Sequences per image: {augmented_per_image}")
print(f"  Total training samples: {original_images * augmented_per_image:,}")

# Impact on training time
print(f"\\nTraining time impact:")
print(f"  Simple: ~{augmented_per_image}x more epochs")
print(f"  Compositional (len=2): ~{len(transformations)**2}x more epochs")
\`\`\`

**Trade-offs**:

**Diversity vs Computational Cost**:

\`\`\`python
import matplotlib.pyplot as plt

# Calculate augmentation options for different sequence lengths
n_transforms = 5
seq_lengths = range(1, 6)
with_replacement = [n_transforms**k for k in seq_lengths]
without_replacement = [perm(n_transforms, k) if k <= n_transforms else 0 
                       for k in seq_lengths]

print("Augmentation diversity growth:\\n")
print("Seq Length | With Replacement | Without Replacement | Compute Cost")
print("-" * 70)
for k in seq_lengths:
    wr = n_transforms**k
    wor = perm(n_transforms, k) if k <= n_transforms else 0
    cost = f"{wr}x"
    print(f"    {k}      |      {wr:>6}       |       {wor:>6}        |    {cost}")

# Practical considerations
print("\\n**Practical Guidelines**:")
print("- Small dataset (< 1K images): Use compositional augmentation (length 2-3)")
print("- Medium dataset (1K-10K): Use simple augmentation + random combinations")
print("- Large dataset (> 10K): Use online random augmentation (not exhaustive)")
print("- Trade-off: More augmentation → better generalization but slower training")
\`\`\`

**Real-World Example**:

In computer vision (e.g., CIFAR-10 with 50,000 training images):

1. **Exhaustive augmentation**: All 25 sequences → 1.25M samples → impractical
2. **Smart augmentation**: Randomly sample 5 sequences per image → 250K samples → manageable
3. **Online augmentation**: Generate random augmentation on-the-fly during training → no storage cost

**Key Insights**:

- Permutations determine the diversity of augmented data
- Compositional augmentations grow exponentially (n^k)
- Balance augmentation diversity with computational budget
- Random sampling from augmentation space often better than exhaustive application
- Online augmentation (generate during training) saves storage while maintaining diversity`,
    keyPoints: [
      'Single transformation: n options; Sequence of k transformations: n^k (with replacement) or P(n,k) (without)',
      'Data augmentation uses permutations to create diverse training samples',
      'Compositional augmentation: Apply multiple transformations in sequence',
      'Trade-off: More augmentation diversity vs higher computational and storage cost',
      'Strategy: Small datasets benefit from exhaustive augmentation, large datasets use random sampling',
    ],
  },
  {
    id: 'dq3-combinations-hyperparameter-tuning',
    question:
      'In hyperparameter tuning, suppose you have 4 hyperparameters to optimize, and each can take 5 different values. How many total configurations exist? If testing each configuration takes 30 minutes, how long would exhaustive grid search take? Compare this with random search (testing 100 random configurations) and explain why random search is often more effective despite testing fewer configurations. Discuss the role of combinatorics in AutoML.',
    sampleAnswer: `Hyperparameter tuning involves exploring a combinatorial space of configurations. Understanding combinatorics reveals why exhaustive search is often infeasible and why random search can be surprisingly effective.

**Total Configurations (Grid Search)**:

With 4 hyperparameters, each with 5 possible values:
- **Total configurations = 5⁴ = 625**

This is because we choose one value for each of 4 hyperparameters independently:
- Hyperparameter 1: 5 choices
- Hyperparameter 2: 5 choices  
- Hyperparameter 3: 5 choices
- Hyperparameter 4: 5 choices
- Total: 5 × 5 × 5 × 5 = 625

\`\`\`python
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import time

# Hyperparameter configuration
n_hyperparameters = 4
values_per_hyperparameter = 5
time_per_config_minutes = 30

# Total configurations
total_configs = values_per_hyperparameter ** n_hyperparameters
total_time_hours = (total_configs * time_per_config_minutes) / 60
total_time_days = total_time_hours / 24

print("Hyperparameter Tuning Complexity:")
print(f"  Number of hyperparameters: {n_hyperparameters}")
print(f"  Values per hyperparameter: {values_per_hyperparameter}")
print(f"  Total configurations: {total_configs:,}")
print(f"  Time per configuration: {time_per_config_minutes} minutes")
print(f"\\nExhaustive Grid Search:")
print(f"  Total time: {total_time_hours:,.1f} hours ({total_time_days:,.1f} days)")

# Example hyperparameters
hyperparameters = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'batch_size': [16, 32, 64, 128, 256],
    'num_layers': [2, 3, 4, 5, 6],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
}

# Generate all combinations
all_configs = list(product(*hyperparameters.values()))
print(f"\\nGenerated {len(all_configs)} configurations")
print("\\nFirst 10 configurations:")
keys = list(hyperparameters.keys())
for i, config in enumerate(all_configs[:10]):
    config_dict = dict(zip(keys, config))
    print(f"  {i+1}. {config_dict}")
\`\`\`

**Random Search (100 Configurations)**:

\`\`\`python
import random

n_random_samples = 100
random_time_hours = (n_random_samples * time_per_config_minutes) / 60

print(f"\\nRandom Search:")
print(f"  Configurations tested: {n_random_samples}")
print(f"  Total time: {random_time_hours:,.1f} hours")
print(f"  Speedup: {total_time_hours / random_time_hours:.1f}x faster")
print(f"  Coverage: {100 * n_random_samples / total_configs:.1f}% of grid")

# Generate random configurations
def random_config(hyperparameters):
    """Sample one random configuration"""
    return {key: random.choice(values) for key, values in hyperparameters.items()}

random_configs = [random_config(hyperparameters) for _ in range(n_random_samples)]

print("\\nFirst 10 random configurations:")
for i, config in enumerate(random_configs[:10]):
    print(f"  {i+1}. {config}")
\`\`\`

**Why Random Search is More Effective**:

**1. Not All Hyperparameters Are Equally Important**:

\`\`\`python
# Simulate a model where only 2 out of 4 hyperparameters matter
def model_performance(lr, batch_size, num_layers, dropout):
    """
    Simulated model performance
    Only learning_rate and num_layers significantly affect performance
    batch_size and dropout have minimal impact
    """
    # Important hyperparameters
    lr_score = -abs(np.log10(lr) + 2.5)  # Optimal around 0.003
    layers_score = -abs(num_layers - 4)   # Optimal at 4 layers
    
    # Less important hyperparameters (noise)
    batch_score = -0.1 * np.random.rand()
    dropout_score = -0.1 * np.random.rand()
    
    return lr_score + layers_score + batch_score + dropout_score

# Grid search: Fixed grid spacing
lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
layers_values = [2, 3, 4, 5, 6]

# Random search: Continuous sampling
random_lr = [10**np.random.uniform(-3, -1) for _ in range(100)]
random_layers = [np.random.randint(2, 7) for _ in range(100)]

print("\\nGrid Search Samples (important hyperparameters only):")
print(f"  Learning rates tested: {lr_values}")
print(f"  Num layers tested: {layers_values}")
print(f"  Total combinations: {len(lr_values) * len(layers_values)} = 25")

print("\\nRandom Search (first 10 samples):")
for i in range(10):
    print(f"  lr={random_lr[i]:.4f}, layers={random_layers[i]}")
print(f"  Total: 100 diverse samples")
\`\`\`

**2. Visualizing Coverage**:

\`\`\`python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Grid search: only tests 5x5=25 combinations of important parameters
grid_lr = np.tile(lr_values, len(layers_values))
grid_layers = np.repeat(layers_values, len(lr_values))

ax1.scatter(grid_lr, grid_layers, s=100, c='blue', marker='s', alpha=0.7, label='Grid Search')
ax1.set_xscale('log')
ax1.set_xlabel('Learning Rate (log scale)')
ax1.set_ylabel('Number of Layers')
ax1.set_title(f'Grid Search: {len(grid_lr)} configurations')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Random search: 100 samples with better coverage
ax2.scatter(random_lr, random_layers, s=100, c='red', marker='o', alpha=0.7, label='Random Search')
ax2.set_xscale('log')
ax2.set_xlabel('Learning Rate (log scale)')
ax2.set_ylabel('Number of Layers')
ax2.set_title(f'Random Search: 100 configurations')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print("\\nKey Insight: Random search explores more values per important hyperparameter")
print("  Grid: 5 learning rate values")
print("  Random: ~100 unique learning rate values (continuous sampling)")
\`\`\`

**Combinatorics Growth**:

\`\`\`python
# Show exponential growth
hp_range = range(1, 8)
values = 5

configs = [values**n for n in hp_range]
time_days = [(c * 30 / 60 / 24) for c in configs]

print("\\nCombinatorial Explosion:")
print("Hyperparameters | Configurations | Time (Grid Search)")
print("-" * 55)
for n, c, t in zip(hp_range, configs, time_days):
    if t < 1:
        time_str = f"{t*24:.1f} hours"
    else:
        time_str = f"{t:.1f} days"
    print(f"       {n}        |     {c:>6,}     | {time_str:>15}")
\`\`\`

**AutoML and Combinatorics**:

\`\`\`python
# Modern AutoML: Bayesian Optimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

print("\\nAutoML Strategies:")
print("1. Grid Search: Exhaustive, O(nᵏ), guarantees finding optimum in grid")
print("2. Random Search: O(m) where m=budget, better than grid when k>2")
print("3. Bayesian Optimization: O(m), uses past results to guide search")
print("4. Hyperband: O(m log m), adaptive resource allocation")
print("\\nCombinatorics lessons:")
print("- Exhaustive search infeasible for >3-4 hyperparameters")
print("- Random search effective when few hyperparameters are important")
print("- Smart search (Bayesian) balances exploration and exploitation")
print("- Trade-off: Computational budget vs finding optimal configuration")
\`\`\`

**Key Insights**:

- Hyperparameter space grows exponentially: O(vⁿ) where v=values, n=hyperparameters
- Grid search: 5⁴ = 625 configs, 312.5 hours = 13 days
- Random search: 100 configs, 50 hours = 2 days, often finds better results
- Random search wins because:
  - Tests more unique values per important hyperparameter
  - Not all hyperparameters equally important
  - Avoids wasting time on uniform grids
- Modern AutoML uses Bayesian optimization: ~10-50 configs often sufficient`,
    keyPoints: [
      'Hyperparameter space size: v^n where v=values per param, n=number of params',
      'Grid search: Exhaustive but exponentially expensive (625 configs = 13 days)',
      'Random search: Tests 100 configs in 2 days, often outperforms grid search',
      'Random search advantage: Better coverage of important hyperparameters',
      'AutoML uses Bayesian optimization to guide search based on past results',
    ],
  },
];
