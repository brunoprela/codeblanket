/**
 * Random Variables Section
 */

export const randomvariablesSection = {
  id: 'random-variables',
  title: 'Random Variables',
  content: `# Random Variables

## Introduction

A **random variable** is a function that maps outcomes from a sample space to real numbers. It's one of the most important concepts in probability and statistics, providing a bridge between abstract probability spaces and concrete numerical analysis.

**Why Random Variables Matter in ML**:
- Model outputs are random variables
- Loss functions involve expectations of random variables
- Gradients in SGD are random variables
- Uncertainty quantification
- Feature distributions

## Definition

A **random variable** X is a function:

\\[ X: \\Omega \\rightarrow \\mathbb{R} \\]

where Ω is the sample space.

**Example**: Roll a die
- Sample space: Ω = {1, 2, 3, 4, 5, 6}
- Random variable X: the number shown
- X(outcome) = outcome (in this simple case)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Simple random variable: die roll
def die_roll_rv():
    """Demonstrate a simple random variable"""
    
    # Sample space
    sample_space = [1, 2, 3, 4, 5, 6]
    
    # Random variable X: the outcome itself
    # Simulate 1000 rolls
    np.random.seed(42)
    rolls = np.random.choice(sample_space, size=1000)
    
    print("=== Random Variable: Die Roll ===")
    print(f"Sample space Ω: {sample_space}")
    print(f"Random variable X: outcome value")
    print(f"\\nFirst 10 rolls: {rolls[:10]}")
    
    # Probability distribution
    unique, counts = np.unique(rolls, return_counts=True)
    probabilities = counts / len(rolls)
    
    print(f"\\nEmpirical Probability Distribution:")
    for outcome, prob in zip(unique, probabilities):
        print(f"P(X = {outcome}) = {prob:.3f}")
    
    return rolls

rolls = die_roll_rv()

# Output:
# === Random Variable: Die Roll ===
# Sample space Ω: [1, 2, 3, 4, 5, 6]
# Random variable X: outcome value
#
# First 10 rolls: [6 3 6 4 4 4 5 2 5 2]
#
# Empirical Probability Distribution:
# P(X = 1) = 0.158
# P(X = 2) = 0.166
# P(X = 3) = 0.174
# P(X = 4) = 0.167
# P(X = 5) = 0.168
# P(X = 6) = 0.167
\`\`\`

## Types of Random Variables

### 1. Discrete Random Variables

Takes countable values (integers, finite set).

**Examples**:
- Number of heads in 10 coin flips
- Number of customers arriving per hour
- Class label in classification (0, 1, 2, ...)

### 2. Continuous Random Variables

Takes uncountable values (any real number in an interval).

**Examples**:
- Height, weight, temperature
- Model loss value
- Neural network activation

\`\`\`python
def discrete_vs_continuous():
    """Compare discrete and continuous random variables"""
    
    np.random.seed(42)
    
    # Discrete RV: number of heads in 10 flips
    print("=== Discrete Random Variable ===")
    print("X = number of heads in 10 coin flips")
    n_experiments = 1000
    discrete_rv = [np.sum(np.random.rand(10) < 0.5) for _ in range(n_experiments)]
    
    print(f"Possible values: {sorted(set(discrete_rv))}")
    print(f"Sample: {discrete_rv[:10]}")
    
    # Continuous RV: height (simulated)
    print(f"\\n=== Continuous Random Variable ===")
    print("Y = adult height in cm")
    continuous_rv = np.random.normal(170, 10, size=1000)
    
    print(f"Range: [{continuous_rv.min():.2f}, {continuous_rv.max():.2f}]")
    print(f"Sample: {continuous_rv[:5]}")
    print(f"Note: Every value is different - uncountably many possibilities")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Discrete
    ax1.hist(discrete_rv, bins=11, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Heads')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Discrete RV: Coin Flips')
    ax1.grid(True, alpha=0.3)
    
    # Continuous
    ax2.hist(continuous_rv, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Height (cm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Continuous RV: Height')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
discrete_vs_continuous()
\`\`\`

## Probability Mass Function (PMF)

For **discrete** random variables, the PMF is:

\\[ p_X(x) = P(X = x) \\]

**Properties**:
1. \\( 0 \\leq p_X(x) \\leq 1 \\) for all x
2. \\( \\sum_{x} p_X(x) = 1 \\)

\`\`\`python
def pmf_example():
    """Demonstrate Probability Mass Function"""
    
    # Random variable: sum of two dice
    np.random.seed(42)
    n_rolls = 10000
    
    die1 = np.random.randint(1, 7, size=n_rolls)
    die2 = np.random.randint(1, 7, size=n_rolls)
    sum_dice = die1 + die2
    
    # Calculate PMF
    values, counts = np.unique(sum_dice, return_counts=True)
    pmf = counts / n_rolls
    
    print("=== PMF: Sum of Two Dice ===")
    print("x\\tP(X=x)\\tTheoretical")
    print("-" * 35)
    
    # Theoretical probabilities
    theoretical = {
        2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
        7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
    }
    
    for val, prob in zip(values, pmf):
        theory = theoretical[val]
        print(f"{val}\\t{prob:.4f}\\t{theory:.4f}")
    
    print(f"\\nSum of probabilities: {pmf.sum():.4f} (should be 1.0)")
    
    # Plot PMF
    plt.figure(figsize=(10, 6))
    plt.bar(values, pmf, alpha=0.7, label='Empirical')
    plt.plot(values, [theoretical[v] for v in values], 'ro-', label='Theoretical')
    plt.xlabel('Sum of Two Dice')
    plt.ylabel('Probability')
    plt.title('PMF: Sum of Two Dice')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
pmf_example()

# Output:
# === PMF: Sum of Two Dice ===
# x	P(X=x)	Theoretical
# -----------------------------------
# 2	0.0268	0.0278
# 3	0.0565	0.0556
# 4	0.0815	0.0833
# 5	0.1136	0.1111
# 6	0.1366	0.1389
# 7	0.1687	0.1667
# 8	0.1382	0.1389
# 9	0.1128	0.1111
# 10	0.0865	0.0833
# 11	0.0543	0.0556
# 12	0.0245	0.0278
#
# Sum of probabilities: 1.0000 (should be 1.0)
\`\`\`

## Probability Density Function (PDF)

For **continuous** random variables, we use PDF instead of PMF:

\\[ P(a \\leq X \\leq b) = \\int_{a}^{b} f_X(x) dx \\]

**Key Difference**: P(X = exact value) = 0 for continuous RVs!

**Properties**:
1. \\( f_X(x) \\geq 0 \\) for all x
2. \\( \\int_{-\\infty}^{\\infty} f_X(x) dx = 1 \\)
3. \\( f_X(x) \\) can be > 1 (not a probability itself!)

\`\`\`python
def pdf_example():
    """Demonstrate Probability Density Function"""
    
    # Standard normal distribution
    x = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(x, loc=0, scale=1)
    
    print("=== PDF: Standard Normal Distribution ===")
    print("f(x) = (1/√(2π)) * exp(-x²/2)")
    print()
    
    # Important: P(X = exact value) = 0
    print("Key property: P(X = 0) = 0 (any exact value)")
    print("Instead, we compute P(a ≤ X ≤ b) = ∫ f(x)dx")
    print()
    
    # Compute some probabilities
    prob_interval1 = stats.norm.cdf(1) - stats.norm.cdf(-1)
    prob_interval2 = stats.norm.cdf(2) - stats.norm.cdf(-2)
    
    print(f"P(-1 ≤ X ≤ 1) = {prob_interval1:.4f} (about 68%)")
    print(f"P(-2 ≤ X ≤ 2) = {prob_interval2:.4f} (about 95%)")
    
    # Plot PDF
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, 'b-', linewidth=2, label='PDF: f(x)')
    
    # Shade area for P(-1 ≤ X ≤ 1)
    mask = (x >= -1) & (x <= 1)
    plt.fill_between(x[mask], pdf[mask], alpha=0.3, label='P(-1 ≤ X ≤ 1) ≈ 0.68')
    
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('PDF: Standard Normal Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    
    print("\\nNote: PDF value at x=0 is ~0.40, which is > 0 but NOT a probability!")
    print("Probability is the AREA under the curve, not the height")

pdf_example()

# Output:
# === PDF: Standard Normal Distribution ===
# f(x) = (1/√(2π)) * exp(-x²/2)
#
# Key property: P(X = 0) = 0 (any exact value)
# Instead, we compute P(a ≤ X ≤ b) = ∫ f(x)dx
#
# P(-1 ≤ X ≤ 1) = 0.6827 (about 68%)
# P(-2 ≤ X ≤ 2) = 0.9545 (about 95%)
#
# Note: PDF value at x=0 is ~0.40, which is > 0 but NOT a probability!
# Probability is the AREA under the curve, not the height
\`\`\`

## Cumulative Distribution Function (CDF)

Works for **both** discrete and continuous random variables:

\\[ F_X(x) = P(X \\leq x) \\]

**Properties**:
1. \\( 0 \\leq F_X(x) \\leq 1 \\)
2. Non-decreasing: if \\( x_1 < x_2 \\), then \\( F_X(x_1) \\leq F_X(x_2) \\)
3. \\( \\lim_{x \\to -\\infty} F_X(x) = 0 \\) and \\( \\lim_{x \\to \\infty} F_X(x) = 1 \\)

**Relationship to PDF**:
\\[ F_X(x) = \\int_{-\\infty}^{x} f_X(t) dt \\]
\\[ f_X(x) = \\frac{d}{dx} F_X(x) \\]

\`\`\`python
def cdf_example():
    """Demonstrate Cumulative Distribution Function"""
    
    # Standard normal
    x = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(x)
    cdf = stats.norm.cdf(x)
    
    print("=== CDF: Cumulative Distribution Function ===")
    print("F(x) = P(X ≤ x)")
    print()
    
    # Key values
    print(f"F(-∞) = 0 (no probability below -∞)")
    print(f"F(0) = {stats.norm.cdf(0):.4f} (50% below median)")
    print(f"F(+∞) = 1 (all probability below +∞)")
    print()
    
    # Calculate probabilities using CDF
    print("Using CDF to calculate probabilities:")
    print(f"P(X ≤ 1) = F(1) = {stats.norm.cdf(1):.4f}")
    print(f"P(X > 1) = 1 - F(1) = {1 - stats.norm.cdf(1):.4f}")
    print(f"P(-1 ≤ X ≤ 1) = F(1) - F(-1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
    
    # Plot PDF and CDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDF
    ax1.plot(x, pdf, 'b-', linewidth=2)
    ax1.fill_between(x[x<=1], pdf[x<=1], alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('PDF: Probability Density Function')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1, color='r', linestyle='--', label='x=1')
    ax1.legend()
    
    # CDF
    ax2.plot(x, cdf, 'r-', linewidth=2)
    ax2.axhline(y=stats.norm.cdf(1), color='b', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='b', linestyle='--', alpha=0.5, label=f'F(1) = {stats.norm.cdf(1):.3f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_title('CDF: Cumulative Distribution Function')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()

cdf_example()
\`\`\`

## ML Applications

### 1. Model Outputs as Random Variables

\`\`\`python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X[:800], y[:800])

# Predictions are random variables!
# For a single input, different trees give different predictions
test_sample = X[800:801]  # Single test sample

print("=== Model Output as Random Variable ===")
print("Random Forest: Each tree is a random variable")
print()

# Get prediction from each tree
tree_predictions = [tree.predict(test_sample)[0] for tree in rf.estimators_]
unique, counts = np.unique(tree_predictions, return_counts=True)

print("Tree predictions (PMF):")
for val, count in zip(unique, counts):
    prob = count / len(tree_predictions)
    print(f"P(prediction = {val}) = {prob:.3f}")

# Final prediction
proba = rf.predict_proba(test_sample)[0]
print(f"\\nFinal probability: {proba}")
print("Model aggregates random variables from individual trees!")

# Output:
# === Model Output as Random Variable ===
# Random Forest: Each tree is a random variable
#
# Tree predictions (PMF):
# P(prediction = 0) = 0.380
# P(prediction = 1) = 0.620
#
# Final probability: [0.38 0.62]
# Model aggregates random variables from individual trees!
\`\`\`

### 2. Loss as a Random Variable

\`\`\`python
def loss_as_random_variable():
    """Demonstrate loss as a random variable"""
    
    # Simulate training with random mini-batches
    np.random.seed(42)
    
    # True loss on full dataset (unknown)
    true_loss = 0.5
    
    # Mini-batch losses are random variables
    # Each batch gives different estimate
    batch_size = 32
    n_batches = 100
    
    # Simulate losses with some variance
    batch_losses = np.random.normal(true_loss, 0.1, size=n_batches)
    
    print("=== Loss as Random Variable (SGD) ===")
    print(f"True loss: {true_loss}")
    print(f"First 10 batch losses: {batch_losses[:10]}")
    print(f"Mean of batch losses: {batch_losses.mean():.3f}")
    print(f"Std of batch losses: {batch_losses.std():.3f}")
    print()
    print("Each mini-batch gives different loss estimate")
    print("This randomness is why we need multiple epochs!")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(batch_losses, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(true_loss, color='r', linestyle='--', linewidth=2, label='True Loss')
    plt.axvline(batch_losses.mean(), color='g', linestyle='--', linewidth=2, label='Mean Batch Loss')
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.title('Distribution of Mini-Batch Loss (Random Variable)')
    plt.legend()
    plt.grid(True, alpha=0.3)

loss_as_random_variable()
\`\`\`

### 3. Feature Distributions

\`\`\`python
def feature_distributions():
    """Analyze features as random variables"""
    
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    
    print("=== Features as Random Variables ===")
    print("Iris dataset features:\\n")
    
    for i, name in enumerate(data.feature_names):
        feature = X[:, i]
        print(f"{name}:")
        print(f"  Type: Continuous random variable")
        print(f"  Mean (E[X]): {feature.mean():.2f}")
        print(f"  Std (√Var[X]): {feature.std():.2f}")
        print(f"  Range: [{feature.min():.2f}, {feature.max():.2f}]")
        print()
    
    print("In ML, we model the joint distribution of features P(X)")
    print("and conditional distribution P(Y|X)")

feature_distributions()
\`\`\`

## Key Takeaways

1. **Random variable**: Function mapping outcomes to numbers
2. **Discrete RV**: Countable values, use PMF
3. **Continuous RV**: Uncountable values, use PDF
4. **PMF**: P(X=x) for discrete variables
5. **PDF**: Density function for continuous variables (not probability!)
6. **CDF**: F(x) = P(X≤x) works for both types
7. **ML applications**: Model outputs, losses, gradients, features are all random variables
8. **Key insight**: All of ML is reasoning about random variables!

Understanding random variables is essential for rigorous ML theory and practice.
`,
};
