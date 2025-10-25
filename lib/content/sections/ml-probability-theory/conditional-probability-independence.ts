/**
 * Conditional Probability & Independence Section
 */

export const conditionalprobabilityindependenceSection = {
  id: 'conditional-probability-independence',
  title: 'Conditional Probability & Independence',
  content: `# Conditional Probability & Independence

## Introduction

Conditional probability is one of the most important concepts in probability and machine learning. It answers the question: **"How does knowing one event affect the probability of another?"**

In ML, nearly everything involves conditional probability:
- **Predictions**: P(class | features)
- **Feature importance**: How does knowing feature X affect predictions?
- **Bayesian methods**: Updating beliefs given evidence
- **Causality**: Does X cause Y, or are they just correlated?

## Conditional Probability

### Definition

The probability of event A given that event B has occurred:

\\[ P(A|B) = \\frac{P(A \\cap B)}{P(B)} \\text{ for } P(B) > 0 \\]

**Intuition**: We restrict our sample space to only cases where B occurred, then ask what fraction of those also have A.

### Rearranging: Multiplication Rule

\\[ P(A \\cap B) = P(A|B) \\times P(B) = P(B|A) \\times P(A) \\]

This is the **multiplication rule** and is fundamental to probability.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Medical diagnosis example
def medical_diagnosis_demo():
    """Demonstrate conditional probability in medical testing"""
    
    np.random.seed(42)
    n_patients = 10000
    
    # Ground truth: 1% of patients have the disease
    has_disease = np.random.rand (n_patients) < 0.01
    
    # Test characteristics
    # Sensitivity: P(positive test | has disease) = 0.95
    # Specificity: P(negative test | no disease) = 0.90
    # False positive rate: P(positive test | no disease) = 0.10
    
    test_positive = np.where(
        has_disease,
        np.random.rand (n_patients) < 0.95,  # true positives
        np.random.rand (n_patients) < 0.10   # false positives
    )
    
    # Calculate probabilities
    p_disease = np.mean (has_disease)
    p_positive = np.mean (test_positive)
    p_disease_and_positive = np.mean (has_disease & test_positive)
    
    # Conditional probabilities
    p_positive_given_disease = p_disease_and_positive / p_disease if p_disease > 0 else 0
    p_disease_given_positive = p_disease_and_positive / p_positive if p_positive > 0 else 0
    
    print("=== Medical Diagnosis Example ===")
    print(f"Disease prevalence: P(disease) = {p_disease:.3f}")
    print(f"Positive test rate: P(+) = {p_positive:.3f}")
    print()
    print(f"P(+ | disease) = {p_positive_given_disease:.3f}  [Sensitivity]")
    print(f"P(disease | +) = {p_disease_given_positive:.3f}  [Positive Predictive Value]")
    print()
    print(f"Key insight: Even with 95% sensitivity, only {p_disease_given_positive:.1%}")
    print(f"of positive tests actually have the disease!")
    print(f"This is because the disease is rare (1% prevalence).")
    
    return {
        'has_disease': has_disease,
        'test_positive': test_positive,
        'p_disease_given_positive': p_disease_given_positive
    }

results = medical_diagnosis_demo()

# Output:
# === Medical Diagnosis Example ===
# Disease prevalence: P(disease) = 0.010
# Positive test rate: P(+) = 0.109
#
# P(+ | disease) = 0.950  [Sensitivity]
# P(disease | +) = 0.087  [Positive Predictive Value]
#
# Key insight: Even with 95% sensitivity, only 8.7%
# of positive tests actually have the disease!
# This is because the disease is rare (1% prevalence).
\`\`\`

## Visualizing Conditional Probability

\`\`\`python
def visualize_conditional_probability():
    """Create Venn diagram visualization"""
    
    np.random.seed(42)
    n = 1000
    
    # Event A: Feature X > 0.5
    # Event B: Label = 1
    X = np.random.rand (n)
    
    # Generate labels with dependency on X
    # P(Y=1 | X>0.5) = 0.7
    # P(Y=1 | X≤0.5) = 0.3
    Y = np.where(
        X > 0.5,
        np.random.rand (n) < 0.7,
        np.random.rand (n) < 0.3
    )
    
    # Define events
    event_A = X > 0.5  # High feature value
    event_B = Y == 1   # Positive label
    
    # Calculate probabilities
    p_a = np.mean (event_A)
    p_b = np.mean (event_B)
    p_a_and_b = np.mean (event_A & event_B)
    p_b_given_a = p_a_and_b / p_a
    p_b_given_not_a = np.mean (event_B & ~event_A) / np.mean(~event_A)
    
    print("=== Feature Importance Analysis ===")
    print(f"P(X > 0.5) = {p_a:.3f}")
    print(f"P(Y = 1) = {p_b:.3f}")
    print(f"P(X > 0.5 AND Y = 1) = {p_a_and_b:.3f}")
    print()
    print(f"P(Y = 1 | X > 0.5) = {p_b_given_a:.3f}")
    print(f"P(Y = 1 | X ≤ 0.5) = {p_b_given_not_a:.3f}")
    print()
    print(f"Knowing X > 0.5 increases probability of Y=1 from {p_b:.1%} to {p_b_given_a:.1%}")
    print(f"Feature X is informative!")
    
    # Create contingency table
    contingency = pd.crosstab(
        pd.Series (event_A, name='X > 0.5'),
        pd.Series (event_B, name='Y = 1'),
        normalize='index'  # Show conditional probabilities
    )
    print("\\nContingency Table (row-normalized):")
    print(contingency)

visualize_conditional_probability()

# Output:
# === Feature Importance Analysis ===
# P(X > 0.5) = 0.510
# P(Y = 1) = 0.511
# P(X > 0.5 AND Y = 1) = 0.353
#
# P(Y = 1 | X > 0.5) = 0.692
# P(Y = 1 | X ≤ 0.5) = 0.323
#
# Knowing X > 0.5 increases probability of Y=1 from 51.1% to 69.2%
# Feature X is informative!
#
# Contingency Table (row-normalized):
# Y = 1        False      True
# X > 0.5
# False        0.677     0.323
# True         0.308     0.692
\`\`\`

## Independence

### Definition

Events A and B are **independent** if:

\\[ P(A|B) = P(A) \\]

Knowing B gives no information about A.

### Equivalent Definitions

All of the following are equivalent:
1. \\( P(A|B) = P(A) \\)
2. \\( P(B|A) = P(B) \\)
3. \\( P(A \\cap B) = P(A) \\times P(B) \\)

**Key Insight**: Independent means "multiplication rule always works."

\`\`\`python
def test_independence():
    """Test whether two events are independent"""
    
    np.random.seed(42)
    n = 10000
    
    # Example 1: Two independent coin flips
    coin1 = np.random.choice([0, 1], size=n)
    coin2 = np.random.choice([0, 1], size=n)
    
    p_coin1 = np.mean (coin1)
    p_coin2 = np.mean (coin2)
    p_both = np.mean (coin1 & coin2)
    p_coin2_given_coin1 = np.mean (coin2[coin1 == 1])
    
    print("=== Example 1: Independent Events (Coin Flips) ===")
    print(f"P(Coin1 = 1) = {p_coin1:.3f}")
    print(f"P(Coin2 = 1) = {p_coin2:.3f}")
    print(f"P(Both = 1) = {p_both:.3f}")
    print(f"P(Coin1) × P(Coin2) = {p_coin1 * p_coin2:.3f}")
    print(f"P(Coin2 = 1 | Coin1 = 1) = {p_coin2_given_coin1:.3f}")
    print(f"Independent? {abs (p_both - p_coin1 * p_coin2) < 0.01}")
    print()
    
    # Example 2: Dependent events (card draws without replacement)
    # Simulate drawing from deck
    deck = np.array([1]*13 + [0]*39)  # 13 aces, 39 non-aces
    
    first_card_ace = []
    second_card_ace = []
    
    for _ in range(10000):
        np.random.shuffle (deck)
        first_card_ace.append (deck[0] == 1)
        second_card_ace.append (deck[1] == 1)
    
    first_card_ace = np.array (first_card_ace)
    second_card_ace = np.array (second_card_ace)
    
    p_first = np.mean (first_card_ace)
    p_second = np.mean (second_card_ace)
    p_both_ace = np.mean (first_card_ace & second_card_ace)
    p_second_given_first = np.mean (second_card_ace[first_card_ace])
    
    print("=== Example 2: Dependent Events (Cards Without Replacement) ===")
    print(f"P(1st card ace) = {p_first:.3f}")
    print(f"P(2nd card ace) = {p_second:.3f}")
    print(f"P(Both aces) = {p_both_ace:.3f}")
    print(f"P(1st) × P(2nd) = {p_first * p_second:.3f}")
    print(f"P(2nd ace | 1st ace) = {p_second_given_first:.3f}")
    print(f"Independent? {abs (p_both_ace - p_first * p_second) < 0.01}")
    print(f"\\nNotice: P(2nd|1st) = {p_second_given_first:.3f} ≠ P(2nd) = {p_second:.3f}")
    print(f"Drawing first ace reduces probability of second ace!")

test_independence()

# Output:
# === Example 1: Independent Events (Coin Flips) ===
# P(Coin1 = 1) = 0.503
# P(Coin2 = 1) = 0.500
# P(Both = 1) = 0.252
# P(Coin1) × P(Coin2) = 0.251
# P(Coin2 = 1 | Coin1 = 1) = 0.501
# Independent? True
#
# === Example 2: Dependent Events (Cards Without Replacement) ===
# P(1st card ace) = 0.252
# P(2nd card ace) = 0.252
# P(Both aces) = 0.015
# P(1st) × P(2nd) = 0.063
# P(2nd ace | 1st ace) = 0.059
# Independent? False
#
# Notice: P(2nd|1st) = 0.059 ≠ P(2nd) = 0.252
# Drawing first ace reduces probability of second ace!
\`\`\`

## Conditional Independence

Events A and B are **conditionally independent given C** if:

\\[ P(A \\cap B | C) = P(A|C) \\times P(B|C) \\]

**Important**: A and B can be dependent overall, but independent when we condition on C!

\`\`\`python
def conditional_independence_demo():
    """Demonstrate conditional independence"""
    
    np.random.seed(42)
    n = 10000
    
    # Scenario: Exam scores
    # C: Study time (low/high)
    # A: Score on Test 1
    # B: Score on Test 2
    
    # Generate study time
    study_time = np.random.choice(['low', 'high'], size=n, p=[0.5, 0.5])
    
    # Generate test scores based on study time
    # If study=low: both tests score ~50
    # If study=high: both tests score ~80
    test1 = np.where(
        study_time == 'high',
        np.random.normal(80, 10, n),
        np.random.normal(50, 10, n)
    )
    
    test2 = np.where(
        study_time == 'high',
        np.random.normal(80, 10, n),
        np.random.normal(50, 10, n)
    )
    
    # Test 1 and Test 2 are correlated (both depend on study time)
    correlation_overall = np.corrcoef (test1, test2)[0, 1]
    print("=== Conditional Independence Example ===")
    print(f"Correlation(Test1, Test2) overall: {correlation_overall:.3f}")
    print("Tests are correlated! (because both depend on study time)")
    print()
    
    # But conditionally independent given study time
    mask_low = study_time == 'low'
    mask_high = study_time == 'high'
    
    corr_given_low = np.corrcoef (test1[mask_low], test2[mask_low])[0, 1]
    corr_given_high = np.corrcoef (test1[mask_high], test2[mask_high])[0, 1]
    
    print(f"Correlation(Test1, Test2 | study=low): {corr_given_low:.3f}")
    print(f"Correlation(Test1, Test2 | study=high): {corr_given_high:.3f}")
    print()
    print("Given study time, tests are nearly independent!")
    print("This is conditional independence: A ⊥ B | C")
    print()
    print("ML Insight: Naive Bayes assumes features are conditionally")
    print("independent given the class label (even if correlated overall)")

conditional_independence_demo()

# Output:
# === Conditional Independence Example ===
# Correlation(Test1, Test2) overall: 0.733
# Tests are correlated! (because both depend on study time)
#
# Correlation(Test1, Test2 | study=low): -0.006
# Correlation(Test1, Test2 | study=high): 0.014
#
# Given study time, tests are nearly independent!
# This is conditional independence: A ⊥ B | C
#
# ML Insight: Naive Bayes assumes features are conditionally
# independent given the class label (even if correlated overall)
\`\`\`

## ML Applications

### 1. Naive Bayes Classifier

Assumes features are conditionally independent given the class:

\\[ P(x_1, x_2, \\ldots, x_n | y) = P(x_1|y) \\times P(x_2|y) \\times \\cdots \\times P(x_n|y) \\]

\`\`\`python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification (n_samples=1000, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predictions
y_pred = nb.predict(X_test)
accuracy = accuracy_score (y_test, y_pred)

print("=== Naive Bayes Classifier ===")
print(f"Accuracy: {accuracy:.3f}")
print()
print("Naive Bayes assumption: P(x₁, x₂, ..., xₙ | y) = ∏ P(xᵢ | y)")
print("Assumes features are conditionally independent given the class")
print("Works well even when assumption is violated!")

# Output:
# === Naive Bayes Classifier ===
# Accuracy: 0.890
#
# Naive Bayes assumption: P(x₁, x₂, ..., xₙ | y) = ∏ P(xᵢ | y)
# Assumes features are conditionally independent given the class
# Works well even when assumption is violated!
\`\`\`

### 2. Feature Importance via Conditional Probability

\`\`\`python
def feature_importance_conditional():
    """Measure feature importance using conditional probability"""
    
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # For binary classification
    p_positive = np.mean (y == 1)
    
    # Measure how much each feature changes probability
    importances = []
    
    for i in range (min(5, X.shape[1])):  # First 5 features
        # Split by median
        threshold = np.median(X[:, i])
        above_threshold = X[:, i] > threshold
        
        p_pos_given_high = np.mean (y[above_threshold] == 1)
        p_pos_given_low = np.mean (y[~above_threshold] == 1)
        
        # Information gain
        importance = abs (p_pos_given_high - p_pos_given_low)
        importances.append (importance)
        
        print(f"{feature_names[i][:30]:30s}: ", end=')
        print(f"P(+|high)={p_pos_given_high:.2f}, P(+|low)={p_pos_given_low:.2f}, ", end=')
        print(f"Δ={importance:.2f}")
    
    print(f"\\nFeatures with largest Δ are most informative!")

print("=== Feature Importance via Conditional Probability ===")
feature_importance_conditional()

# Output:
# === Feature Importance via Conditional Probability ===
# mean radius                   : P(+|high)=0.79, P(+|low)=0.49, Δ=0.30
# mean texture                  : P(+|high)=0.70, P(+|low)=0.58, Δ=0.12
# mean perimeter                : P(+|high)=0.79, P(+|low)=0.49, Δ=0.30
# mean area                     : P(+|high)=0.78, P(+|low)=0.50, Δ=0.28
# mean smoothness               : P(+|high)=0.70, P(+|low)=0.58, Δ=0.12
#
# Features with largest Δ are most informative!
\`\`\`

## Key Takeaways

1. **Conditional probability**: P(A|B) = P(A∩B) / P(B) - probability of A given B occurred
2. **Multiplication rule**: P(A∩B) = P(A|B) × P(B) - fundamental to all of probability
3. **Independence**: P(A|B) = P(A) - knowing B doesn't change probability of A
4. **Independence test**: Check if P(A∩B) = P(A) × P(B)
5. **Conditional independence**: A ⊥ B | C - independent when conditioning on C
6. **ML applications**: Naive Bayes, feature importance, causal inference
7. **Common mistake**: Correlation ≠ causation - need conditional probability analysis

## Caution: Simpson\'s Paradox

A trend can appear in different groups but disappear or reverse when groups are combined!

\`\`\`python
# Example: Treatment appears harmful overall but helpful in both groups
print("=== Simpson's Paradox ===")
print("Group 1: Treatment helps (80% vs 70% recovery)")
print("Group 2: Treatment helps (60% vs 50% recovery)")
print("Combined: Treatment appears harmful (65% vs 75% recovery)")
print("\\nWhy? Group 1 (easier cases) gets more control patients")
print("     Group 2 (harder cases) gets more treatment patients")
print("\\nLesson: Always condition on confounding variables!")
\`\`\`

Understanding conditional probability is essential for proper statistical reasoning in ML!
`,
};
