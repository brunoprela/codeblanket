/**
 * Set Theory & Logic Section
 */

export const settheorylogicSection = {
  id: 'set-theory-logic',
  title: 'Set Theory & Logic',
  content: `
# Set Theory & Logic

## Introduction

Set theory provides the foundation for organizing data, understanding probability, and reasoning about machine learning algorithms. Logic enables us to make precise statements, build correct algorithms, and understand Boolean operations in neural networks and decision trees. These concepts are fundamental to data science, feature engineering, and algorithmic reasoning.

## Sets

### Definition

A **set** is a collection of distinct objects, called **elements** or **members**.

**Notation**:
- A = {1, 2, 3, 4, 5}
- x ∈ A (x is an element of A)
- x ∉ A (x is not an element of A)

**Ways to Define Sets**:
1. **Roster notation**: List all elements: A = {1, 2, 3}
2. **Set-builder notation**: A = {x | x is an integer, 1 ≤ x ≤ 3}
3. **Rule-based**: A = {x | x² < 10}

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Python sets
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

print(f"Set A: {A}")
print(f"Set B: {B}")

# Membership testing
print(f"\\n3 in A: {3 in A}")
print(f"10 in A: {10 in A}")

# Set from list (removes duplicates)
numbers = [1, 2, 2, 3, 3, 3, 4]
unique_numbers = set (numbers)
print(f"\\nOriginal list: {numbers}")
print(f"Set (unique): {unique_numbers}")

# Set comprehension (like set-builder notation)
squares = {x**2 for x in range(1, 6)}
print(f"\\nSquares: {squares}")

# ML Application: Unique classes in dataset
labels = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
unique_classes = set (labels)
print(f"\\nLabels: {labels}")
print(f"Unique classes: {unique_classes}")
print(f"Number of classes: {len (unique_classes)}")
\`\`\`

### Special Sets

**Empty Set**: ∅ or {} (contains no elements)
**Universal Set**: U (contains all elements under consideration)
**Natural Numbers**: ℕ = {1, 2, 3, ...}
**Integers**: ℤ = {..., -2, -1, 0, 1, 2, ...}
**Real Numbers**: ℝ

\`\`\`python
# Empty set
empty = set()  # Not {} (that's empty dict)
print(f"Empty set: {empty}")
print(f"Size: {len (empty)}")

# Infinite sets (represented by rules)
def is_natural (x):
    return isinstance (x, int) and x > 0

def is_integer (x):
    return isinstance (x, int)

def is_even (x):
    return isinstance (x, int) and x % 2 == 0

# Test membership
print(f"\\n5 is natural: {is_natural(5)}")
print(f"-3 is natural: {is_natural(-3)}")
print(f"6 is even: {is_even(6)}")
\`\`\`

## Set Operations

### Union (∪)

**A ∪ B**: All elements in A or B (or both)

\`\`\`python
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

union = A | B  # or A.union(B)
print(f"A ∪ B = {union}")

# ML Application: Combining feature sets
features_model1 = {'age', 'income', 'education'}
features_model2 = {'income', 'location', 'job_title'}
all_features = features_model1 | features_model2
print(f"\\nModel 1 features: {features_model1}")
print(f"Model 2 features: {features_model2}")
print(f"Combined features: {all_features}")
\`\`\`

### Intersection (∩)

**A ∩ B**: Elements in both A and B

\`\`\`python
intersection = A & B  # or A.intersection(B)
print(f"A ∩ B = {intersection}")

# ML Application: Common features
common_features = features_model1 & features_model2
print(f"\\nCommon features: {common_features}")

# Empty intersection (disjoint sets)
C = {1, 2, 3}
D = {4, 5, 6}
print(f"\\nC ∩ D = {C & D}")  # Empty set - disjoint
\`\`\`

### Difference (\\)

**A \\ B**: Elements in A but not in B

\`\`\`python
difference = A - B  # or A.difference(B)
print(f"A \\ B = {difference}")
print(f"B \\ A = {B - A}")

# ML Application: Features unique to one model
unique_to_model1 = features_model1 - features_model2
print(f"\\nUnique to Model 1: {unique_to_model1}")
\`\`\`

### Symmetric Difference (△)

**A △ B**: Elements in A or B but not both

\`\`\`python
sym_diff = A ^ B  # or A.symmetric_difference(B)
print(f"A △ B = {sym_diff}")

# Equivalent to (A ∪ B) \\ (A ∩ B)
equivalent = (A | B) - (A & B)
print(f"Equivalent: {equivalent}")
print(f"Match: {sym_diff == equivalent}")
\`\`\`

### Complement

**A'** or **Aᶜ**: Elements in universal set U but not in A

\`\`\`python
# Need to define universal set
U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
A = {1, 2, 3, 4, 5}

complement = U - A
print(f"U = {U}")
print(f"A = {A}")
print(f"A' (complement) = {complement}")

# ML Application: Negative class
all_classes = {'cat', 'dog', 'bird', 'fish'}
positive_class = {'cat'}
negative_classes = all_classes - positive_class
print(f"\\nAll classes: {all_classes}")
print(f"Positive: {positive_class}")
print(f"Negative (complement): {negative_classes}")
\`\`\`

## Venn Diagrams

Visualizing set relationships:

\`\`\`python
from matplotlib_venn import venn2, venn3

# Two sets
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

plt.figure (figsize=(8, 6))
venn2([A, B], set_labels=('A', 'B'))
plt.title('Venn Diagram: A and B')
plt.show()

# Three sets
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
C = {4, 5, 6, 7}

plt.figure (figsize=(8, 6))
venn3([A, B, C], set_labels=('A', 'B', 'C'))
plt.title('Venn Diagram: A, B, and C')
plt.show()

print("Venn diagrams displayed!")
\`\`\`

### ML Application: Data Filtering

\`\`\`python
# Example: Customer segmentation
import pandas as pd

# Sample customer data
customers = pd.DataFrame({
    'customer_id': range(1, 11),
    'age': [25, 35, 45, 22, 55, 30, 40, 28, 50, 33],
    'income': [40000, 60000, 80000, 35000, 90000, 55000, 75000, 45000, 85000, 62000],
    'purchased': [True, True, False, True, False, True, False, True, True, False]
})

# Define customer segments using sets
young = set (customers[customers['age'] < 35]['customer_id'])
high_income = set (customers[customers['income'] > 60000]['customer_id'])
purchasers = set (customers[customers['purchased']]['customer_id'])

print("Customer Segments:")
print(f"Young (< 35): {young}")
print(f"High income (> 60k): {high_income}")
print(f"Purchasers: {purchasers}")

# Set operations for targeting
young_high_income = young & high_income
print(f"\\nYoung AND high income: {young_high_income}")

young_or_high_income = young | high_income
print(f"Young OR high income: {young_or_high_income}")

young_non_purchasers = young - purchasers
print(f"Young non-purchasers (to target): {young_non_purchasers}")

# Complex query
target_segment = (young | high_income) & purchasers
print(f"\\n(Young OR high income) AND purchasers: {target_segment}")
\`\`\`

## Subsets and Supersets

**A ⊆ B** (A is subset of B): Every element of A is in B
**A ⊂ B** (A is proper subset): A ⊆ B and A ≠ B
**A ⊇ B** (A is superset of B): B ⊆ A

\`\`\`python
A = {1, 2, 3}
B = {1, 2, 3, 4, 5}
C = {1, 2, 3}

# Subset
print(f"A ⊆ B: {A <= B}")  # A.issubset(B)
print(f"B ⊆ A: {B <= A}")

# Proper subset
print(f"\\nA ⊂ B (proper): {A < B}")
print(f"A ⊂ C (proper): {A < C}")  # False, they're equal

# Superset
print(f"\\nB ⊇ A: {B >= A}")  # B.issuperset(A)

# ML Application: Feature hierarchy
basic_features = {'age', 'gender'}
extended_features = {'age', 'gender', 'income', 'education'}
premium_features = {'age', 'gender', 'income', 'education', 'credit_score', 'employment_history'}

print(f"\\nBasic ⊆ Extended: {basic_features <= extended_features}")
print(f"Extended ⊆ Premium: {extended_features <= premium_features}")
print(f"Basic ⊆ Premium: {basic_features <= premium_features}")  # Transitivity
\`\`\`

## Cardinality

**|A|**: Number of elements in set A

\`\`\`python
A = {1, 2, 3, 4, 5}
print(f"|A| = {len(A)}")

# Properties
# |A ∪ B| = |A| + |B| - |A ∩ B| (Inclusion-Exclusion)
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

size_union = len(A | B)
size_formula = len(A) + len(B) - len(A & B)

print(f"\\n|A| = {len(A)}")
print(f"|B| = {len(B)}")
print(f"|A ∩ B| = {len(A & B)}")
print(f"|A ∪ B| = {size_union}")
print(f"|A| + |B| - |A ∩ B| = {size_formula}")
print(f"Formula matches: {size_union == size_formula}")
\`\`\`

## Propositional Logic

### Basic Propositions

A **proposition** is a statement that is either true or false.

\`\`\`python
# Propositions in Python (Boolean values)
p = True   # "It is raining"
q = False  # "It is cold"

print(f"p (It is raining): {p}")
print(f"q (It is cold): {q}")

# ML Application: Conditions
threshold = 0.5
prediction = 0.75

is_positive = prediction > threshold
is_confident = prediction > 0.9

print(f"\\nPrediction: {prediction}")
print(f"Is positive class: {is_positive}")
print(f"Is confident: {is_confident}")
\`\`\`

### Logical Operators

**NOT (¬)**: Negation
**AND (∧)**: Conjunction
**OR (∨)**: Disjunction
**XOR (⊕)**: Exclusive or
**IMPLIES (→)**: Implication
**IFF (↔)**: If and only if

\`\`\`python
def logical_not (p):
    """NOT: ¬p"""
    return not p

def logical_and (p, q):
    """AND: p ∧ q"""
    return p and q

def logical_or (p, q):
    """OR: p ∨ q"""
    return p or q

def logical_xor (p, q):
    """XOR: p ⊕ q (exclusive or)"""
    return p != q

def logical_implies (p, q):
    """IMPLIES: p → q (if p then q)"""
    return (not p) or q

def logical_iff (p, q):
    """IFF: p ↔ q (if and only if)"""
    return p == q

# Test all combinations
p_values = [True, False]
q_values = [True, False]

print("Truth Table:")
print(f"{'p':<6} {'q':<6} {'¬p':<6} {'p∧q':<6} {'p∨q':<6} {'p⊕q':<6} {'p→q':<6} {'p↔q':<6}")
print("-" * 50)

for p in p_values:
    for q in q_values:
        print(f"{p!s:<6} {q!s:<6} {logical_not (p)!s:<6} {logical_and (p, q)!s:<6} "
              f"{logical_or (p, q)!s:<6} {logical_xor (p, q)!s:<6} "
              f"{logical_implies (p, q)!s:<6} {logical_iff (p, q)!s:<6}")
\`\`\`

### De Morgan\'s Laws

**¬(p ∧ q) = ¬p ∨ ¬q**
**¬(p ∨ q) = ¬p ∧ ¬q**

\`\`\`python
# Verify De Morgan's Laws
def verify_demorgan():
    """Verify De Morgan's laws for all truth values"""
    for p in [True, False]:
        for q in [True, False]:
            # Law 1: ¬(p ∧ q) = ¬p ∨ ¬q
            left1 = not (p and q)
            right1 = (not p) or (not q)

            # Law 2: ¬(p ∨ q) = ¬p ∧ ¬q
            left2 = not (p or q)
            right2 = (not p) and (not q)

            print(f"p={p}, q={q}:")
            print(f"  ¬(p∧q)={left1}, ¬p∨¬q={right1}, Equal: {left1 == right1}")
            print(f"  ¬(p∨q)={left2}, ¬p∧¬q={right2}, Equal: {left2 == right2}")

verify_demorgan()
\`\`\`

**ML Application**: Feature filtering logic

\`\`\`python
# Example: Filter data with complex conditions
age = 25
income = 70000
has_degree = True

# Condition: (age >= 25 AND income > 60000) OR has_degree
condition1 = (age >= 25 and income > 60000) or has_degree
print(f"\\nCondition 1 (original): {condition1}")

# Apply De Morgan\'s law to negate
# NOT[(age >= 25 AND income > 60000) OR has_degree]
# = NOT(age >= 25 AND income > 60000) AND NOT(has_degree)
# = (NOT(age >= 25) OR NOT(income > 60000)) AND NOT(has_degree)
condition2_negated = ((age < 25) or (income <= 60000)) and (not has_degree)
condition2 = not condition2_negated

print(f"Condition 2 (De Morgan): {condition2}")
print(f"Match: {condition1 == condition2}")
\`\`\`

## Truth Tables

Complete enumeration of logical outcomes:

\`\`\`python
import pandas as pd

def generate_truth_table (n_variables):
    """Generate truth table for n Boolean variables"""
    from itertools import product

    # Generate all combinations
    combinations = list (product([False, True], repeat=n_variables))

    # Create column names
    var_names = [f'p{i+1}' for i in range (n_variables)]

    # Create DataFrame
    df = pd.DataFrame (combinations, columns=var_names)

    return df

# Example: 3 variables
truth_table = generate_truth_table(3)
print("Truth Table for 3 variables:")
print(truth_table)

# Add derived columns
truth_table['p1 ∧ p2'] = truth_table['p1'] & truth_table['p2']
truth_table['p1 ∨ p2'] = truth_table['p1'] | truth_table['p2']
truth_table['p1 → p2'] = ~truth_table['p1'] | truth_table['p2']

print("\\nWith logical operations:")
print(truth_table)
\`\`\`

## Applications in Machine Learning

### Boolean Features

\`\`\`python
# Binary features in ML
data = pd.DataFrame({
    'has_account': [True, False, True, True, False],
    'is_premium': [False, False, True, False, True],
    'purchased': [False, False, True, False, True]
})

print("Dataset:")
print(data)

# Logical feature engineering
data['premium_no_purchase'] = data['is_premium'] & ~data['purchased']
data['account_or_premium'] = data['has_account'] | data['is_premium']

print("\\nWith engineered features:")
print(data)
\`\`\`

### Decision Tree Logic

Decision trees use logical operations:

\`\`\`python
def decision_tree_logic (age, income, credit_score):
    """
    Simple decision tree as logical expressions
    Approve loan if:
      (age >= 25 AND income > 50000) OR credit_score > 700
    """
    condition1 = age >= 25 and income > 50000
    condition2 = credit_score > 700

    approve = condition1 or condition2

    return approve, condition1, condition2

# Test cases
test_cases = [
    (30, 60000, 650),  # Satisfies age and income
    (22, 40000, 750),  # Satisfies credit score
    (28, 70000, 720),  # Satisfies both
    (20, 30000, 600),  # Satisfies neither
]

print("Decision Tree Logic:")
print(f"{'Age':<5} {'Income':<8} {'Credit':<7} {'Cond1':<7} {'Cond2':<7} {'Approve':<8}")
print("-" * 50)

for age, income, credit in test_cases:
    approve, cond1, cond2 = decision_tree_logic (age, income, credit)
    print(f"{age:<5} {income:<8} {credit:<7} {cond1!s:<7} {cond2!s:<7} {approve!s:<8}")
\`\`\`

### Neural Network Activations

Boolean logic can be implemented with neural networks:

\`\`\`python
# Perceptron implementing logic gates

def perceptron (inputs, weights, bias):
    """Simple perceptron"""
    activation = np.dot (inputs, weights) + bias
    return 1 if activation > 0 else 0

# AND gate
def neural_AND(x1, x2):
    weights = np.array([1, 1])
    bias = -1.5
    return perceptron([x1, x2], weights, bias)

# OR gate
def neural_OR(x1, x2):
    weights = np.array([1, 1])
    bias = -0.5
    return perceptron([x1, x2], weights, bias)

# NOT gate
def neural_NOT(x):
    weights = np.array([-1])
    bias = 0.5
    return perceptron([x], weights, bias)

# Test logic gates
print("\\nNeural Logic Gates:")
print(f"{'x1':<5} {'x2':<5} {'AND':<5} {'OR':<5} {'NOT x1':<8}")
print("-" * 30)

for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"{x1:<5} {x2:<5} {neural_AND(x1, x2):<5} {neural_OR(x1, x2):<5} {neural_NOT(x1):<8}")
\`\`\`

### Set Operations on Data

\`\`\`python
# Train/test split using sets
all_indices = set (range(100))
train_indices = set (np.random.choice(100, 70, replace=False))
test_indices = all_indices - train_indices

print(f"Total samples: {len (all_indices)}")
print(f"Training samples: {len (train_indices)}")
print(f"Test samples: {len (test_indices)}")
print(f"No overlap: {len (train_indices & test_indices) == 0}")

# Feature selection using sets
available_features = {'age', 'income', 'education', 'credit_score',
                     'employment_history', 'debt_ratio'}
selected_by_correlation = {'income', 'credit_score', 'debt_ratio'}
selected_by_importance = {'age', 'income', 'credit_score'}

# Intersection: features selected by both methods
robust_features = selected_by_correlation & selected_by_importance
print(f"\\nRobust features (selected by both): {robust_features}")

# Union: all selected features
all_selected = selected_by_correlation | selected_by_importance
print(f"All selected features: {all_selected}")
\`\`\`

## Summary

- **Sets**: Collections of distinct objects
- **Operations**: Union (∪), Intersection (∩), Difference (\\), Complement (')
- **Subsets**: A ⊆ B means all elements of A are in B
- **Cardinality**: |A| is the number of elements
- **Logic**: Propositions with TRUE/FALSE values
- **Operators**: NOT (¬), AND (∧), OR (∨), XOR (⊕), IMPLIES (→), IFF (↔)
- **De Morgan's Laws**: Transform negations of AND/OR
- **Truth Tables**: Systematic enumeration of logical outcomes

**ML Applications**:
- Data filtering and segmentation
- Feature engineering with Boolean operations
- Decision trees (logical conditions)
- Train/test splits
- Feature selection
- Neural network logic gates
- Boolean features in models
`,
};
