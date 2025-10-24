/**
 * Combinatorics & Counting Section
 */

export const combinatoricscountingSection = {
  id: 'combinatorics-counting',
  title: 'Combinatorics & Counting',
  content: `# Combinatorics & Counting

## Introduction

Combinatorics is the mathematics of counting. In machine learning and probability, we often need to count:
- **Number of possible outcomes** (sample space size)
- **Ways to select training samples** (sampling strategies)
- **Arrangements in neural networks** (architecture search space)
- **Feature combinations** (interaction terms)
- **Hypothesis space size** (model complexity)

Understanding counting principles is essential for calculating probabilities and analyzing algorithm complexity.

## Fundamental Counting Principle

**Rule**: If there are \\( n_1 \\) ways to do task 1, and \\( n_2 \\) ways to do task 2, then there are \\( n_1 \\times n_2 \\) ways to do both tasks.

**Extended**: For \\( k \\) tasks: \\( n_1 \\times n_2 \\times \\cdots \\times n_k \\)

**Example**: Password with 8 characters (26 letters, 10 digits each position)
\\[ 36^8 = 2,821,109,907,456 \\text{ possible passwords} \\]

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, comb

# Example: Creating a neural network architecture
def count_architectures():
    """Count possible neural network architectures"""
    
    # Choices for each layer
    layer_sizes = [64, 128, 256, 512]  # 4 choices
    activations = ['relu', 'tanh', 'sigmoid', 'elu']  # 4 choices
    num_layers = 3  # We'll design 3 hidden layers
    
    # Each layer: choose size AND activation
    choices_per_layer = len(layer_sizes) * len(activations)
    
    # Total architectures using fundamental counting principle
    total = choices_per_layer ** num_layers
    
    print("=== Neural Network Architecture Search Space ===")
    print(f"Layer size options: {len(layer_sizes)}")
    print(f"Activation options: {len(activations)}")
    print(f"Choices per layer: {choices_per_layer}")
    print(f"Number of layers: {num_layers}")
    print(f"Total architectures: {total:,}")
    print(f"\\nIf testing 1 architecture per second: {total / 3600:.1f} hours")
    
count_architectures()

# Output:
# === Neural Network Architecture Search Space ===
# Layer size options: 4
# Activation options: 4
# Choices per layer: 16
# Number of layers: 3
# Total architectures: 4,096
#
# If testing 1 architecture per second: 1.1 hours
\`\`\`

## Permutations

**Definition**: Arrangements where **order matters**.

### Permutations of n items (all distinct)

Number of ways to arrange \\( n \\) distinct items:

\\[ P(n) = n! = n \\times (n-1) \\times (n-2) \\times \\cdots \\times 1 \\]

**Example**: Arranging 5 books on a shelf = 5! = 120 ways

### Permutations of n items taken r at a time

Number of ways to arrange \\( r \\) items selected from \\( n \\) items:

\\[ P(n, r) = \\frac{n!}{(n-r)!} \\]

**Example**: Top 3 finishers in a race of 10 runners
\\[ P(10, 3) = \\frac{10!}{7!} = 10 \\times 9 \\times 8 = 720 \\]

\`\`\`python
def permutations_demo():
    """Demonstrate permutation calculations"""
    
    # Example 1: Arranging all items
    n = 5
    all_perms = factorial(n)
    print(f"Ways to arrange {n} distinct items: {all_perms}")
    
    # Example 2: Arranging r items from n
    n, r = 10, 3
    perms = factorial(n) // factorial(n - r)
    print(f"\\nWays to arrange {r} items from {n}: {perms}")
    
    # ML Example: Feature ordering for sequential models
    n_features = 8
    select_top_k = 3
    feature_orderings = factorial(n_features) // factorial(n_features - select_top_k)
    
    print(f"\\n=== ML Application: Feature Selection ===")
    print(f"Total features: {n_features}")
    print(f"Selecting top-{select_top_k} features (order matters)")
    print(f"Possible orderings: {feature_orderings}")
    
    # Generate actual permutations for small example
    from itertools import permutations
    items = ['A', 'B', 'C']
    perms_list = list(permutations(items, 2))
    print(f"\\n2-permutations of {items}:")
    for p in perms_list:
        print(f"  {p}")
    print(f"Count: {len(perms_list)} = P(3,2) = 3!/(3-2)! = 6")

permutations_demo()

# Output:
# Ways to arrange 5 distinct items: 120
#
# Ways to arrange 3 items from 10: 720
#
# === ML Application: Feature Selection ===
# Total features: 8
# Selecting top-3 features (order matters)
# Possible orderings: 336
#
# 2-permutations of ['A', 'B', 'C']:
#   ('A', 'B')
#   ('A', 'C')
#   ('B', 'A')
#   ('B', 'C')
#   ('C', 'A')
#   ('C', 'B')
# Count: 6 = P(3,2) = 3!/(3-2)! = 6
\`\`\`

## Combinations

**Definition**: Selections where **order does NOT matter**.

### Combinations of n items taken r at a time

Number of ways to select \\( r \\) items from \\( n \\) items (order doesn't matter):

\\[ C(n, r) = \\binom{n}{r} = \\frac{n!}{r!(n-r)!} \\]

Also written as "n choose r".

**Key Difference from Permutations**:
- Permutation: {A, B} ≠ {B, A} (order matters)
- Combination: {A, B} = {B, A} (order doesn't matter)

\\[ C(n, r) = \\frac{P(n, r)}{r!} \\]

**Example**: Selecting 3 students from 10 for a committee
\\[ C(10, 3) = \\frac{10!}{3!7!} = \\frac{10 \\times 9 \\times 8}{3 \\times 2 \\times 1} = 120 \\]

\`\`\`python
def combinations_demo():
    """Demonstrate combination calculations"""
    
    # Example 1: Committee selection
    n, r = 10, 3
    combs = comb(n, r)
    print(f"Ways to select {r} people from {n} (order doesn't matter): {combs}")
    
    # Compare with permutations
    perms = factorial(n) // factorial(n - r)
    print(f"If order mattered (permutations): {perms}")
    print(f"Ratio: {perms / combs} = {r}!")
    
    # ML Example: Feature subset selection
    n_features = 20
    select_k = 5
    feature_subsets = comb(n_features, select_k)
    
    print(f"\\n=== ML Application: Feature Subset Selection ===")
    print(f"Total features: {n_features}")
    print(f"Selecting {select_k} features (order doesn't matter)")
    print(f"Possible subsets: {feature_subsets:,}")
    print(f"\\nExhaustive search would need {feature_subsets:,} model trainings!")
    
    # Generate actual combinations for small example
    from itertools import combinations
    items = ['A', 'B', 'C', 'D']
    combs_list = list(combinations(items, 2))
    print(f"\\n2-combinations of {items}:")
    for c in combs_list:
        print(f"  {set(c)}")
    print(f"Count: {len(combs_list)} = C(4,2) = 6")

combinations_demo()

# Output:
# Ways to select 3 people from 10 (order doesn't matter): 120
# If order mattered (permutations): 720
# Ratio: 6.0 = 3!
#
# === ML Application: Feature Subset Selection ===
# Total features: 20
# Selecting 5 features (order doesn't matter)
# Possible subsets: 15,504
#
# Exhaustive search would need 15,504 model trainings!
#
# 2-combinations of ['A', 'B', 'C', 'D']:
#   {'A', 'B'}
#   {'A', 'C'}
#   {'A', 'D'}
#   {'B', 'C'}
#   {'B', 'D'}
#   {'C', 'D'}
# Count: 6 = C(4,2) = 6
\`\`\`

## Binomial Coefficients

The binomial coefficient \\( \\binom{n}{k} \\) appears in many contexts:

### Properties

1. **Symmetry**: \\( \\binom{n}{k} = \\binom{n}{n-k} \\)

2. **Sum**: \\( \\sum_{k=0}^{n} \\binom{n}{k} = 2^n \\) (total number of subsets)

3. **Pascal's Identity**: \\( \\binom{n}{k} = \\binom{n-1}{k-1} + \\binom{n-1}{k} \\)

### Pascal's Triangle

\`\`\`
                1
              1   1
            1   2   1
          1   3   3   1
        1   4   6   4   1
      1   5  10  10   5   1
\`\`\`

Each entry is the sum of the two entries above it.

\`\`\`python
def pascals_triangle(n_rows):
    """Generate Pascal's Triangle"""
    
    triangle = []
    for n in range(n_rows):
        row = [comb(n, k) for k in range(n + 1)]
        triangle.append(row)
    
    # Print triangle
    print("Pascal's Triangle:")
    for i, row in enumerate(triangle):
        spaces = ' ' * (n_rows - i) * 2
        print(spaces + ' '.join(f'{x:3d}' for x in row))
    
    # Verify property: sum of row n = 2^n
    print(f"\\nVerifying: Sum of row {n_rows-1} = {sum(triangle[-1])} = 2^{n_rows-1} = {2**(n_rows-1)}")
    
    return triangle

triangle = pascals_triangle(7)

# Binomial Theorem application
print("\\n=== Binomial Theorem ===")
print("(a + b)^n = Σ C(n,k) * a^(n-k) * b^k")
print("\\nExample: (x + y)^3 =")
n = 3
terms = []
for k in range(n + 1):
    coeff = comb(n, k)
    power_x = n - k
    power_y = k
    term = f"{coeff}x^{power_x}y^{power_y}" if power_x > 0 and power_y > 0 else \
           f"{coeff}x^{power_x}" if power_y == 0 else \
           f"{coeff}y^{power_y}" if power_x == 0 else f"{coeff}"
    terms.append(term.replace('^1', ').replace('^0', '))
print(' + '.join(terms))

# Output:
# Pascal's Triangle:
#              1
#            1   1
#          1   2   1
#        1   3   3   1
#      1   4   6   4   1
#    1   5  10  10   5   1
#  1   6  15  20  15   6   1
#
# Verifying: Sum of row 6 = 64 = 2^6 = 64
#
# === Binomial Theorem ===
# (a + b)^n = Σ C(n,k) * a^(n-k) * b^k
#
# Example: (x + y)^3 =
# x^3 + 3x^2y + 3xy^2 + y^3
\`\`\`

## Permutations with Repetition

When items can repeat:

\\[ n^r \\]

where \\( n \\) is the number of choices and \\( r \\) is the number of positions.

**Example**: 4-digit PIN with digits 0-9
\\[ 10^4 = 10,000 \\text{ possible PINs} \\]

## Combinations with Repetition

Choosing \\( r \\) items from \\( n \\) types with replacement:

\\[ C(n+r-1, r) = \\binom{n+r-1}{r} \\]

**Example**: Choosing 3 scoops of ice cream from 5 flavors (can repeat)
\\[ C(5+3-1, 3) = C(7, 3) = 35 \\]

\`\`\`python
def repetition_examples():
    """Examples of counting with repetition"""
    
    # Permutations with repetition
    print("=== Permutations with Repetition ===")
    n_choices = 10  # digits 0-9
    length = 4  # 4-digit PIN
    total_pins = n_choices ** length
    print(f"{length}-digit PIN with {n_choices} digits: {total_pins:,} possibilities")
    
    # ML: Grid search hyperparameters
    param_values = [3, 5, 7, 4]  # values for 4 hyperparameters
    total_configs = 1
    for n_values in param_values:
        total_configs *= n_values
    print(f"\\nGrid search with {param_values} values per param: {total_configs} configs")
    
    # Combinations with repetition
    print("\\n=== Combinations with Repetition ===")
    n_flavors = 5
    n_scoops = 3
    combinations_rep = comb(n_flavors + n_scoops - 1, n_scoops)
    print(f"Choosing {n_scoops} scoops from {n_flavors} flavors (with repetition):")
    print(f"C({n_flavors}+{n_scoops}-1, {n_scoops}) = C({n_flavors+n_scoops-1}, {n_scoops}) = {combinations_rep}")

repetition_examples()

# Output:
# === Permutations with Repetition ===
# 4-digit PIN with 10 digits: 10,000 possibilities
#
# Grid search with [3, 5, 7, 4] values per param: 420 configs
#
# === Combinations with Repetition ===
# Choosing 3 scoops from 5 flavors (with repetition):
# C(5+3-1, 3) = C(7, 3) = 35
\`\`\`

## ML Applications

### 1. Feature Engineering: Interaction Terms

Creating all pairwise feature interactions:

\`\`\`python
from sklearn.preprocessing import PolynomialFeatures

# Original features
n_features = 10
print(f"Original features: {n_features}")

# Add all pairwise interactions (degree 2)
pairwise_interactions = comb(n_features, 2)
print(f"Pairwise interactions (choose 2 from {n_features}): {pairwise_interactions}")
print(f"Total features with interactions: {n_features + pairwise_interactions}")

# Using sklearn
X = np.random.randn(5, n_features)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X)
print(f"\\nActual sklearn output shape: {X_interactions.shape[1]} features")

# Output:
# Original features: 10
# Pairwise interactions (choose 2 from 10): 45
# Total features with interactions: 55
#
# Actual sklearn output shape: 55 features
\`\`\`

### 2. Cross-Validation Splits

k-fold cross-validation:

\`\`\`python
def analyze_cv_splits(n_samples, k_folds):
    """Analyze combinations in k-fold cross-validation"""
    
    fold_size = n_samples // k_folds
    
    print(f"=== {k_folds}-Fold Cross-Validation ===")
    print(f"Total samples: {n_samples}")
    print(f"Samples per fold: ~{fold_size}")
    print(f"Training size per fold: {n_samples - fold_size}")
    print(f"Test size per fold: {fold_size}")
    
    # How many ways to partition into k folds?
    # This is a multinomial coefficient (complex), but we can estimate
    print(f"\\nNumber of ways to assign {n_samples} samples to {k_folds} folds:")
    print(f"(Very large number - typically use fixed splits)")

analyze_cv_splits(1000, 5)

# Output:
# === 5-Fold Cross-Validation ===
# Total samples: 1000
# Samples per fold: ~200
# Training size per fold: 800
# Test size per fold: 200
#
# Number of ways to assign 1000 samples to 5 folds:
# (Very large number - typically use fixed splits)
\`\`\`

### 3. Neural Architecture Search

\`\`\`python
def architecture_search_space():
    """Calculate neural architecture search space size"""
    
    # Constraints
    max_layers = 5
    layer_sizes = [32, 64, 128, 256, 512]
    activations = ['relu', 'tanh', 'elu']
    
    total = 0
    
    # For each possible number of layers
    for n_layers in range(1, max_layers + 1):
        # Choose layer size AND activation for each layer
        configs_this_depth = (len(layer_sizes) * len(activations)) ** n_layers
        total += configs_this_depth
        print(f"{n_layers} layers: {configs_this_depth:,} architectures")
    
    print(f"\\nTotal architectures: {total:,}")
    print(f"If testing 100 per day: {total / 100:.0f} days = {total / 36500:.1f} years")

architecture_search_space()

# Output:
# 1 layers: 15 architectures
# 2 layers: 225 architectures
# 3 layers: 3,375 architectures
# 4 layers: 50,625 architectures
# 5 layers: 759,375 architectures
#
# Total architectures: 813,615
# If testing 100 per day: 8136 days = 22.3 years
\`\`\`

## Summary

| Concept | Formula | Order Matters? | Repetition? |
|---------|---------|----------------|-------------|
| **Permutations** | \\( \\frac{n!}{(n-r)!} \\) | Yes | No |
| **Combinations** | \\( \\frac{n!}{r!(n-r)!} \\) | No | No |
| **Permutations (rep)** | \\( n^r \\) | Yes | Yes |
| **Combinations (rep)** | \\( \\binom{n+r-1}{r} \\) | No | Yes |

## Key Takeaways

1. **Fundamental Counting Principle**: Multiply choices for independent tasks
2. **Permutations**: Order matters (\\( P(n,r) = n!/(n-r)! \\))
3. **Combinations**: Order doesn't matter (\\( C(n,r) = n!/[r!(n-r)!] \\))
4. **Binomial coefficients**: Appear in Pascal's Triangle and binomial theorem
5. **ML applications**: Feature engineering, architecture search, sampling strategies
6. **Complexity**: Combinatorial explosion makes exhaustive search impractical
7. **Smart search**: Use heuristics, random search, or Bayesian optimization instead

Understanding combinatorics helps you estimate search space sizes and choose appropriate optimization strategies in ML!
`,
};
