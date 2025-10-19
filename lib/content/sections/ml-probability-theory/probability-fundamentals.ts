/**
 * Probability Fundamentals Section
 */

export const probabilityfundamentalsSection = {
  id: 'probability-fundamentals',
  title: 'Probability Fundamentals',
  content: `# Probability Fundamentals

## Introduction

Probability theory is the mathematical framework for reasoning about uncertainty. In machine learning, nearly every algorithm involves probabilistic reasoning - from the uncertainty in predictions to the randomness in training algorithms. Understanding probability is essential for:

- **Model Uncertainty**: Quantifying confidence in predictions
- **Bayesian Inference**: Learning from data with prior beliefs
- **Stochastic Optimization**: Training with random mini-batches
- **Generative Models**: Creating probabilistic models of data
- **Risk Assessment**: Evaluating decision-making under uncertainty

## Sample Spaces and Events

### Sample Space (Ω)

The **sample space** is the set of all possible outcomes of an experiment.

**Examples**:
- Coin flip: Ω = {H, T}
- Die roll: Ω = {1, 2, 3, 4, 5, 6}
- ML prediction: Ω = {correct, incorrect}

### Events

An **event** is a subset of the sample space - a collection of outcomes.

**Examples**:
- Rolling an even number: E = {2, 4, 6}
- Getting heads: E = {H}
- Model accuracy > 90%: E = all runs with accuracy > 0.9

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Simulate coin flips
np.random.seed(42)
n_flips = 1000
flips = np.random.choice(['H', 'T'], size=n_flips)

# Count outcomes
heads_count = np.sum(flips == 'H')
tails_count = np.sum(flips == 'T')

print(f"Heads: {heads_count}, Tails: {tails_count}")
print(f"Probability of Heads: {heads_count / n_flips:.3f}")
print(f"Probability of Tails: {tails_count / n_flips:.3f}")

# Output:
# Heads: 496, Tails: 504
# Probability of Heads: 0.496
# Probability of Tails: 0.504
\`\`\`

## Probability Axioms (Kolmogorov Axioms)

Probability must satisfy three fundamental axioms:

### Axiom 1: Non-negativity
\\[ P(E) \\geq 0 \\text{ for any event } E \\]

Probabilities cannot be negative.

### Axiom 2: Normalization
\\[ P(\\Omega) = 1 \\]

The probability of the entire sample space is 1 (something must happen).

### Axiom 3: Additivity
\\[ P(A \\cup B) = P(A) + P(B) \\text{ if } A \\cap B = \\emptyset \\]

For mutually exclusive events (disjoint), probabilities add.

**Important Consequences**:
- \\( 0 \\leq P(E) \\leq 1 \\) for any event
- \\( P(\\emptyset) = 0 \\) (impossible event has probability 0)
- \\( P(A^c) = 1 - P(A) \\) (complement rule)

\`\`\`python
# Demonstrating axioms
def check_probability_axioms():
    # Simulate die rolls
    outcomes = np.random.randint(1, 7, size=10000)
    
    # Event A: rolling 1, 2, or 3
    event_a = np.sum(outcomes <= 3) / len(outcomes)
    
    # Event B: rolling 4, 5, or 6
    event_b = np.sum(outcomes > 3) / len(outcomes)
    
    # Axiom 1: Non-negativity
    print(f"P(A) = {event_a:.3f} >= 0: {event_a >= 0}")
    print(f"P(B) = {event_b:.3f} >= 0: {event_b >= 0}")
    
    # Axiom 2: Normalization (A and B partition the space)
    print(f"P(A) + P(B) = {event_a + event_b:.3f} ≈ 1")
    
    # Axiom 3: Additivity (A and B are disjoint)
    print(f"Events A and B are disjoint (mutually exclusive)")
    
check_probability_axioms()

# Output:
# P(A) = 0.500 >= 0: True
# P(B) = 0.500 >= 0: True
# P(A) + P(B) = 1.000 ≈ 1
# Events A and B are disjoint (mutually exclusive)
\`\`\`

## Types of Probability

### 1. Classical Probability (Theoretical)

Based on symmetry and equally likely outcomes:

\\[ P(E) = \\frac{\\text{Number of favorable outcomes}}{\\text{Total number of outcomes}} \\]

**Example**: Probability of rolling a 4 on a fair die = 1/6

**Limitations**: Requires equally likely outcomes (often unrealistic)

### 2. Empirical Probability (Frequentist)

Based on observed data:

\\[ P(E) \\approx \\frac{\\text{Number of times E occurred}}{\\text{Total number of trials}} \\]

**Example**: If a model is correct 85 out of 100 times, P(correct) ≈ 0.85

**Foundation of ML**: Most probability estimates in ML are empirical.

### 3. Subjective Probability (Bayesian)

Based on personal belief or prior information:

**Example**: "I believe there's a 70% chance this stock will go up"

**In ML**: Prior distributions in Bayesian methods

\`\`\`python
# Comparing classical vs empirical probability
def compare_probabilities(n_trials):
    """Compare theoretical and empirical probability of rolling a 6"""
    
    # Classical (theoretical) probability
    classical_prob = 1/6
    
    # Empirical probability from simulation
    rolls = np.random.randint(1, 7, size=n_trials)
    empirical_prob = np.sum(rolls == 6) / n_trials
    
    # Law of Large Numbers: empirical approaches theoretical
    error = abs(empirical_prob - classical_prob)
    
    print(f"Trials: {n_trials}")
    print(f"Classical P(6) = {classical_prob:.4f}")
    print(f"Empirical P(6) = {empirical_prob:.4f}")
    print(f"Error: {error:.4f}\\n")
    
    return empirical_prob

# Run with increasing sample sizes
probs = []
sample_sizes = [10, 100, 1000, 10000, 100000]

for n in sample_sizes:
    prob = compare_probabilities(n)
    probs.append(prob)

# Output shows convergence (Law of Large Numbers)
# As n increases, empirical → classical
\`\`\`

## Probability Rules

### Complement Rule

\\[ P(A^c) = 1 - P(A) \\]

Where \\( A^c \\) is "not A".

**Example**: If P(rain) = 0.3, then P(no rain) = 0.7

### Addition Rule (Union)

For any two events A and B:

\\[ P(A \\cup B) = P(A) + P(B) - P(A \\cap B) \\]

**Special case**: If A and B are mutually exclusive (disjoint):

\\[ P(A \\cup B) = P(A) + P(B) \\]

**In ML**: Probability that model is correct on sample A OR sample B

### Multiplication Rule

For independent events A and B:

\\[ P(A \\cap B) = P(A) \\times P(B) \\]

**Example**: P(two heads in a row) = 0.5 × 0.5 = 0.25

\`\`\`python
# Demonstrating probability rules
def demonstrate_rules():
    np.random.seed(42)
    n = 10000
    
    # Two events: rolling even (A) and rolling > 3 (B)
    rolls = np.random.randint(1, 7, size=n)
    
    # Event A: rolling even (2, 4, 6)
    event_a = rolls % 2 == 0
    p_a = np.mean(event_a)
    
    # Event B: rolling > 3 (4, 5, 6)
    event_b = rolls > 3
    p_b = np.mean(event_b)
    
    # Event A ∩ B: rolling even AND > 3 (4, 6)
    event_a_and_b = event_a & event_b
    p_a_and_b = np.mean(event_a_and_b)
    
    # Event A ∪ B: rolling even OR > 3 (2, 4, 5, 6)
    event_a_or_b = event_a | event_b
    p_a_or_b = np.mean(event_a_or_b)
    
    # Complement rule: P(A^c) = 1 - P(A)
    print("=== Complement Rule ===")
    print(f"P(even) = {p_a:.3f}")
    print(f"P(odd) = {1 - p_a:.3f}")
    print(f"Verified: {np.mean(~event_a):.3f}\\n")
    
    # Addition rule: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    print("=== Addition Rule ===")
    print(f"P(A ∪ B) empirical = {p_a_or_b:.3f}")
    print(f"P(A) + P(B) - P(A ∩ B) = {p_a + p_b - p_a_and_b:.3f}\\n")
    
    # Check independence (spoiler: they're not independent!)
    print("=== Independence Test ===")
    print(f"P(A ∩ B) = {p_a_and_b:.3f}")
    print(f"P(A) × P(B) = {p_a * p_b:.3f}")
    print(f"Independent? {abs(p_a_and_b - p_a * p_b) < 0.01}")

demonstrate_rules()

# Output:
# === Complement Rule ===
# P(even) = 0.500
# P(odd) = 0.500
# Verified: 0.500
#
# === Addition Rule ===
# P(A ∪ B) empirical = 0.667
# P(A) + P(B) - P(A ∩ B) = 0.667
#
# === Independence Test ===
# P(A ∩ B) = 0.333
# P(A) × P(B) = 0.250
# Independent? False
\`\`\`

## Conditional Probability

The probability of event A given that event B has occurred:

\\[ P(A|B) = \\frac{P(A \\cap B)}{P(B)} \\text{ for } P(B) > 0 \\]

**Intuition**: We update our probability when we gain information.

**Example**: 
- P(spam | contains "free") is different from P(spam)
- Observing "free" gives us information

\`\`\`python
# ML Application: Spam classification
def spam_classifier_simulation():
    """Simulate conditional probabilities in spam detection"""
    
    np.random.seed(42)
    n_emails = 1000
    
    # Generate synthetic email data
    # 30% are spam
    is_spam = np.random.rand(n_emails) < 0.3
    
    # "free" appears in 80% of spam, 10% of legitimate emails
    contains_free = np.where(
        is_spam,
        np.random.rand(n_emails) < 0.8,  # P("free"|spam) = 0.8
        np.random.rand(n_emails) < 0.1   # P("free"|legit) = 0.1
    )
    
    # Calculate probabilities
    p_spam = np.mean(is_spam)
    p_free = np.mean(contains_free)
    p_free_and_spam = np.mean(contains_free & is_spam)
    p_free_given_spam = p_free_and_spam / p_spam if p_spam > 0 else 0
    
    print("=== Email Spam Detection ===")
    print(f"P(spam) = {p_spam:.3f}")
    print(f'P(contains "free") = {p_free:.3f}')
    print(f'P(spam AND "free") = {p_free_and_spam:.3f}')
    print(f'P("free"|spam) = {p_free_given_spam:.3f}')
    print()
    
    # Now the key question: P(spam|"free")
    p_spam_given_free = p_free_and_spam / p_free if p_free > 0 else 0
    print(f'P(spam|"free") = {p_spam_given_free:.3f}')
    print(f'This is {p_spam_given_free / p_spam:.1f}x higher than P(spam)!')
    print()
    print(f'Conclusion: Seeing "free" strongly increases spam probability')

spam_classifier_simulation()

# Output:
# === Email Spam Detection ===
# P(spam) = 0.300
# P(contains "free") = 0.308
# P(spam AND "free") = 0.242
# P("free"|spam) = 0.807
#
# P(spam|"free") = 0.786
# This is 2.6x higher than P(spam)!
#
# Conclusion: Seeing "free" strongly increases spam probability
\`\`\`

## Uncertainty in ML Predictions

Every ML model deals with uncertainty:

\`\`\`python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate classification data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Get probabilistic predictions
proba = model.predict_proba(X_test)

print("=== ML Prediction Uncertainty ===")
print("First 5 predictions:")
print("P(class=0)  P(class=1)  Prediction")
print("-" * 40)
for i in range(5):
    pred = 1 if proba[i, 1] > 0.5 else 0
    print(f"{proba[i, 0]:.3f}      {proba[i, 1]:.3f}      Class {pred}")

print(f"\\nHigh confidence predictions (>90%): {np.sum(np.max(proba, axis=1) > 0.9)}")
print(f"Uncertain predictions (50-60%): {np.sum((np.max(proba, axis=1) > 0.5) & (np.max(proba, axis=1) < 0.6))}")

# Output:
# === ML Prediction Uncertainty ===
# First 5 predictions:
# P(class=0)  P(class=1)  Prediction
# ----------------------------------------
# 0.957      0.043      Class 0
# 0.069      0.931      Class 1
# 0.046      0.954      Class 1
# 0.941      0.059      Class 0
# 0.958      0.042      Class 0
#
# High confidence predictions (>90%): 159
# Uncertain predictions (50-60%): 9
\`\`\`

## Key Takeaways

1. **Probability quantifies uncertainty**: Essential for ML where predictions are rarely certain
2. **Three axioms**: Non-negativity, normalization, additivity
3. **Complement rule**: P(not A) = 1 - P(A)
4. **Addition rule**: P(A or B) = P(A) + P(B) - P(A and B)
5. **Conditional probability**: P(A|B) updates beliefs given information
6. **Empirical probability**: Most ML probabilities are estimated from data
7. **Law of Large Numbers**: Frequencies converge to true probabilities with more data

## Connection to Machine Learning

- **Classification**: Models output P(class|features)
- **Regularization**: Equivalent to Bayesian priors
- **Dropout**: Randomly zeroing neurons introduces uncertainty
- **Data Augmentation**: Creates new samples from probability distributions
- **Ensemble Methods**: Average predictions as probabilistic mixture
- **Confidence Intervals**: Quantify prediction uncertainty
- **A/B Testing**: Statistical hypothesis testing uses probability theory

Understanding probability is not optional in ML - it's the foundation of the field.
`,
};
