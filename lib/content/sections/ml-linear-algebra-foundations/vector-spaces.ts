/**
 * Vector Spaces Section
 */

export const vectorspacesSection = {
  id: 'vector-spaces',
  title: 'Vector Spaces',
  content: `
# Vector Spaces

## Introduction

A **vector space** is a mathematical structure formed by a collection of vectors that can be added together and multiplied by scalars. Understanding vector spaces is essential for grasping the foundations of machine learning, where data lives in high-dimensional vector spaces.

## Vector Space Definition

A set **V** is a **vector space** over a field **F** (usually ℝ) if it satisfies these axioms for all **u**, **v**, **w** ∈ **V** and scalars a, b ∈ **F**:

### Closure Axioms
1. **Closure under addition**: **u** + **v** ∈ **V**
2. **Closure under scalar multiplication**: a**v** ∈ **V**

### Addition Axioms
3. **Commutative**: **u** + **v** = **v** + **u**
4. **Associative**: (**u** + **v**) + **w** = **u** + (**v** + **w**)
5. **Identity**: ∃ **0** ∈ **V** such that **v** + **0** = **v**
6. **Inverse**: ∃ -**v** ∈ **V** such that **v** + (-**v**) = **0**

### Scalar Multiplication Axioms
7. **Distributive** (vector): a(**u** + **v**) = a**u** + a**v**
8. **Distributive** (scalar): (a + b)**v** = a**v** + b**v**
9. **Associative**: (ab)**v** = a (b**v**)
10. **Identity**: 1**v** = **v**

\`\`\`python
import numpy as np

print("=== Vector Space Examples ===")

# Example 1: ℝⁿ (Euclidean space)
print("\\n1. ℝ² (2D Euclidean space)")
u = np.array([1, 2])
v = np.array([3, 4])
a = 2

print(f"u = {u}")
print(f"v = {v}")
print(f"u + v = {u + v} (still in ℝ²)")
print(f"{a}*u = {a*u} (still in ℝ²)")
print("✓ ℝ² is a vector space")

# Example 2: Space of 2×2 matrices
print("\\n2. Space of 2×2 matrices")
M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[5, 6], [7, 8]])
b = 3

print(f"M1 + M2 =\\n{M1 + M2}")
print(f"{b}*M1 =\\n{b*M1}")
print("✓ 2×2 matrices form a vector space")

# Example 3: Space of polynomials of degree ≤ 2
print("\\n3. Space of polynomials of degree ≤ 2")
# Represent polynomial a₀ + a₁x + a₂x² as [a₀, a₁, a₂]
p1 = np.array([1, 2, 3])  # 1 + 2x + 3x²
p2 = np.array([4, 5, 6])  # 4 + 5x + 6x²
c = 2

print(f"p1 + p2 = {p1 + p2} → {p1[0]+p2[0]} + {p1[1]+p2[1]}x + {p1[2]+p2[2]}x²")
print(f"{c}*p1 = {c*p1} → {c*p1[0]} + {c*p1[1]}x + {c*p1[2]}x²")
print("✓ Polynomials of degree ≤ 2 form a vector space")
\`\`\`

## Subspaces

A **subspace** is a subset of a vector space that is itself a vector space.

**Requirements** for **W** ⊆ **V** to be a subspace:
1. **0** ∈ **W** (contains zero vector)
2. Closed under addition: **u**, **v** ∈ **W** ⇒ **u** + **v** ∈ **W**
3. Closed under scalar multiplication: **v** ∈ **W**, a ∈ ℝ ⇒ a**v** ∈ **W**

\`\`\`python
print("\\n=== Subspaces ===")

# Example 1: Lines through origin in ℝ²
print("\\n1. Line through origin: y = 2x")
# Subspace: {[x, 2x] : x ∈ ℝ}

# Check properties
v1 = np.array([1, 2])
v2 = np.array([2, 4])
v_sum = v1 + v2
v_scaled = 3 * v1

print(f"v1 = {v1} is on line (y/x = 2)")
print(f"v2 = {v2} is on line (y/x = 2)")
print(f"v1 + v2 = {v_sum}, ratio = {v_sum[1]/v_sum[0]} ✓")
print(f"3*v1 = {v_scaled}, ratio = {v_scaled[1]/v_scaled[0]} ✓")
print("→ Line through origin is a subspace")

# Example 2: NOT a subspace (line not through origin)
print("\\n2. Line NOT through origin: y = 2x + 1")
w1 = np.array([0, 1])  # On line
w2 = np.array([1, 3])  # On line
w_sum = w1 + w2

print(f"w1 = {w1}, y = {w1[1]} = 2*{w1[0]} + 1 ✓")
print(f"w2 = {w2}, y = {w2[1]} = 2*{w2[0]} + 1 ✓")
print(f"w1 + w2 = {w_sum}, y = {w_sum[1]} ≠ 2*{w_sum[0]} + 1 ✗")
print("→ NOT a subspace (not closed under addition)")
\`\`\`

## Linear Independence

Vectors **v₁**, **v₂**, ..., **vₖ** are **linearly independent** if:

c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** = **0** ⇒ c₁ = c₂ = ... = cₖ = 0

**Otherwise**, they are **linearly dependent** (one can be written as a combination of the others).

\`\`\`python
print("\\n=== Linear Independence ===")

# Example 1: Linearly independent vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

# Stack as columns
A_indep = np.column_stack([v1, v2, v3])
rank_indep = np.linalg.matrix_rank(A_indep)

print("\\nLinearly independent vectors:")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v3 = {v3}")
print(f"Rank = {rank_indep}, # vectors = {A_indep.shape[1]}")
print(f"Linearly independent: {rank_indep == A_indep.shape[1]}")

# Example 2: Linearly dependent vectors
w1 = np.array([1, 2, 3])
w2 = np.array([2, 4, 6])  # w2 = 2*w1
w3 = np.array([3, 6, 9])  # w3 = 3*w1

A_dep = np.column_stack([w1, w2, w3])
rank_dep = np.linalg.matrix_rank(A_dep)

print("\\nLinearly dependent vectors:")
print(f"w1 = {w1}")
print(f"w2 = {w2} = 2*w1")
print(f"w3 = {w3} = 3*w1")
print(f"Rank = {rank_dep}, # vectors = {A_dep.shape[1]}")
print(f"Linearly independent: {rank_dep == A_dep.shape[1]}")
\`\`\`

## Span

The **span** of vectors is the set of all linear combinations:

span(**v₁**, ..., **vₖ**) = {c₁**v₁** + ... + cₖ**vₖ** : c₁, ..., cₖ ∈ ℝ}

\`\`\`python
print("\\n=== Span ===")

# Span of standard basis in ℝ²
e1 = np.array([1, 0])
e2 = np.array([0, 1])

print(f"e1 = {e1}")
print(f"e2 = {e2}")
print()

# Any vector in ℝ² can be written as linear combination
target = np.array([3, 5])
c1, c2 = target[0], target[1]

combination = c1 * e1 + c2 * e2
print(f"Target: {target}")
print(f"As combination: {c1}*e1 + {c2}*e2 = {combination}")
print(f"Equal: {np.allclose (target, combination)}")
print("\\nspan (e1, e2) = ℝ²")

# Span of single vector (line)
v = np.array([1, 2])
print(f"\\nspan({v}) = all vectors of form c*{v}")
print(f"Examples: {0.5*v}, {v}, {2*v}, {-1*v}")
\`\`\`

## Basis and Dimension

A **basis** for vector space **V** is a set of vectors that:
1. **Spans** **V** (every vector can be written as their linear combination)
2. **Linearly independent**

The **dimension** of **V** is the number of vectors in a basis.

\`\`\`python
print("\\n=== Basis and Dimension ===")

# Standard basis for ℝ³
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

print("Standard basis for ℝ³:")
print(f"e1 = {e1}")
print(f"e2 = {e2}")
print(f"e3 = {e3}")
print("Dimension = 3")
print()

# Any vector can be expressed uniquely
v = np.array([5, 7, 9])
coords = v  # Coordinates in standard basis
reconstructed = coords[0]*e1 + coords[1]*e2 + coords[2]*e3

print(f"v = {v}")
print(f"Coordinates: [{coords[0]}, {coords[1]}, {coords[2]}]")
print(f"Reconstructed: {coords[0]}*e1 + {coords[1]}*e2 + {coords[2]}*e3 = {reconstructed}")

# Alternative basis for ℝ²
b1 = np.array([1, 1])
b2 = np.array([1, -1])

B = np.column_stack([b1, b2])
rank_B = np.linalg.matrix_rank(B)

print(f"\\nAlternative basis for ℝ²:")
print(f"b1 = {b1}")
print(f"b2 = {b2}")
print(f"Rank = {rank_B} (linearly independent)")
print(f"These also form a basis for ℝ²")
\`\`\`

## Column Space and Null Space

### Column Space

The **column space** of matrix **A** is span of its columns.

col(**A**) = {**Ax** : **x** ∈ ℝⁿ}

**Dimension**: rank(**A**)

### Null Space

The **null space** of **A** is the set of solutions to **Ax** = **0**.

null(**A**) = {**x** : **Ax** = **0**}

**Dimension**: n - rank(**A**) (nullity)

\`\`\`python
from scipy.linalg import null_space

print("\\n=== Column Space and Null Space ===")

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Matrix A:")
print(A)
print()

# Column space
rank_A = np.linalg.matrix_rank(A)
print(f"Rank(A) = {rank_A}")
print(f"Column space dimension = {rank_A}")
print(f"Column space is spanned by {rank_A} independent columns")
print()

# Null space
null_A = null_space(A)
print(f"Null space dimension (nullity) = {null_A.shape[1]}")
print(f"n - rank = {A.shape[1]} - {rank_A} = {null_A.shape[1]} ✓")
print()

print("Null space basis:")
print(null_A)
print()

# Verify: A * null_vector ≈ 0
if null_A.shape[1] > 0:
    null_vector = null_A[:, 0]
    result = A @ null_vector
    print(f"A * null_vector = {result}")
    print(f"Approximately zero: {np.allclose (result, 0)}")
\`\`\`

## Applications in Machine Learning

### 1. Feature Space

\`\`\`python
print("\\n=== ML Application: Feature Space ===")

# Dataset as vectors in feature space
# Each sample is a vector in ℝⁿ where n = number of features

# Example: House data
# Features: [size, bedrooms, bathrooms, age]
houses = np.array([
    [2000, 3, 2, 10],
    [1500, 2, 1, 15],
    [2500, 4, 3, 5],
    [1800, 3, 2, 8]
])

print(f"Number of samples: {houses.shape[0]}")
print(f"Feature space dimension: {houses.shape[1]}")
print(f"Each house is a point in ℝ⁴")
print()

# The set of all possible house vectors forms a vector space
# We can add houses, scale them, etc. (though may not have physical meaning)
\`\`\`

### 2. Checking Linear Independence of Features

\`\`\`python
print("\\n=== Checking Feature Independence ===")

# Sometimes features are linearly dependent (redundant)
X = np.array([
    [1, 2, 3],    # Feature 3 = Feature 1 + Feature 2
    [2, 3, 5],
    [3, 4, 7],
    [4, 5, 9]
])

print("Feature matrix X:")
print(X)
print()

rank_X = np.linalg.matrix_rank(X)
print(f"Number of features: {X.shape[1]}")
print(f"Rank: {rank_X}")
print(f"Linearly independent: {rank_X == X.shape[1]}")
print()

if rank_X < X.shape[1]:
    print(f"Only {rank_X} truly independent features!")
    print("Consider removing redundant features or using PCA")
\`\`\`

### 3. Range and Null Space in Regression

\`\`\`python
print("\\n=== Regression: Range and Null Space ===")

# In linear regression: y = Xw + b
# Column space of X determines which target vectors are achievable

X_reg = np.array([[1, 2], [2, 4], [3, 6]])  # Rank 1 (columns dependent)
print("Design matrix X:")
print(X_reg)
print()

rank_reg = np.linalg.matrix_rank(X_reg)
print(f"Rank: {rank_reg}")
print()

# This means only certain y vectors can be perfectly fit
# If y is not in column space, we can only approximate (least squares)

y_in_col = np.array([2, 4, 6])  # = 2 * first column
y_not_in_col = np.array([1, 2, 4])  # Not in column space

print(f"y1 = {y_in_col} is in column space")
print(f"y2 = {y_not_in_col} is NOT in column space")
print("→ For y2, least squares finds best approximation")
\`\`\`

## Summary

**Vector Space**: Set closed under addition and scalar multiplication with 10 axioms

**Subspace**: Subset that is itself a vector space
- Contains zero vector
- Closed under addition and scalar multiplication

**Linear Independence**: No vector is a combination of others
- Check: rank = number of vectors

**Span**: All linear combinations of vectors

**Basis**: Linearly independent spanning set
- Every vector has unique coordinates

**Dimension**: Number of vectors in basis

**Column Space**: Span of columns, dimension = rank

**Null Space**: Solutions to Ax = 0, dimension = n - rank

**ML Connections**:
- Data lives in feature space (vector space)
- Linear independence → non-redundant features
- Dimension = intrinsic dimensionality of data
- Column space = outputs achievable by linear model
- Null space = unidentifiable parameter directions

Understanding vector spaces provides the mathematical foundation for dimensionality reduction, feature engineering, and many ML algorithms!
`,
};
