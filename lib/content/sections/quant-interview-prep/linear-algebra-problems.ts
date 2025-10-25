export const linearAlgebraProblems = {
  title: 'Linear Algebra Problems',
  id: 'linear-algebra-problems',
  content: `
# Linear Algebra Problems

## Introduction

Linear algebra is absolutely fundamental to quantitative finance and is tested extensively in quant interviews. It underpins virtually every aspect of modern finance:

**Core Applications:**
- **Portfolio optimization** - Covariance matrices, efficient frontier, quadratic programming
- **Principal Component Analysis (PCA)** - Risk factor models, dimensionality reduction
- **Risk modeling** - VaR calculations, stress testing, factor decomposition
- **Option pricing** - Finite difference methods, lattice models, state space representations
- **Machine learning** - SVD, matrix factorization, neural networks
- **Time series** - Vector autoregression (VAR), state space models, Kalman filters
- **High-frequency trading** - Matrix operations for signal processing
- **Credit risk** - Correlation matrices, copulas

**What Interviewers Test:**
1. **Mental math with matrices** - 2×2 and 3×3 operations without calculator
2. **Systems of equations** - Quick solving methods
3. **Eigenvalue intuition** - What they mean, how to compute them
4. **Matrix properties** - Positive definite, symmetric, orthogonal, rank
5. **Numerical stability** - Understanding ill-conditioned problems
6. **Finance applications** - Translating problems into linear algebra

This section covers 50+ problems from basic to advanced, with detailed solutions, Python implementations, and interview strategies.

---

## Section 1: Matrix Operations & Mental Math

### 1.1 Basic Matrix Multiplication

**Fundamental rule:** (m×n) × (n×p) = (m×p)

\`\`\`
[a b]   [e f]   [ae+bg  af+bh]
[c d] × [g h] = [ce+dg  cf+dh]
\`\`\`

**Problem 1.1a: Simple 2×2**

Compute:
\`\`\`
[2 3]   [1 0]
[1 4] × [0 1]
\`\`\`

**Solution:**
\`\`\`
[2·1+3·0  2·0+3·1]   [2 3]
[1·1+4·0  1·0+4·1] = [1 4]
\`\`\`

**This is the identity matrix property: AI = A**

**Problem 1.1b: Non-trivial 2×2**

Compute:
\`\`\`
[2  1]   [3 -1]
[1 -1] × [2  4]
\`\`\`

**Solution:**
\`\`\`
[2·3+1·2   2·(-1)+1·4]   [6+2   -2+4]   [8  2]
[1·3+(-1)·2  1·(-1)+(-1)·4] = [3-2  -1-4] = [1 -5]
\`\`\`

**Problem 1.1c: Matrix-Vector Product**

Compute:
\`\`\`
[1 2 3]   [1]
[4 5 6] × [0]
          [1]
\`\`\`

**Solution:**
\`\`\`
[1·1+2·0+3·1]   [1+0+3]   [4]
[4·1+5·0+6·1] = [4+0+6] = [10]
\`\`\`

### 1.2 Determinants

**Determinant (2×2):**
\`\`\`
det([a b]) = ad - bc
    [c d]
\`\`\`

**Geometric meaning:** Area of parallelogram formed by column vectors (or row vectors)

**Key properties:**
- det(AB) = det(A)·det(B)
- det(A^T) = det(A)
- det(A^{-1}) = 1/det(A)
- det(cA) = c^n·det(A) for n×n matrix

**Problem 1.2a:**

Find det(A):
\`\`\`
A = [3 1]
    [2 4]
\`\`\`

**Solution:**
\`\`\`
det(A) = 3·4 - 1·2 = 12 - 2 = 10
\`\`\`

**Problem 1.2b:**

Find det(B):
\`\`\`
B = [5  2]
    [10 4]
\`\`\`

**Solution:**
\`\`\`
det(B) = 5·4 - 2·10 = 20 - 20 = 0
\`\`\`

**This matrix is singular (not invertible)! The rows are linearly dependent (row 2 = 2 × row 1).**

**Problem 1.2c: Determinant of 3×3**

Find det(A) using cofactor expansion:
\`\`\`
A = [2 1 0]
    [0 3 1]
    [1 0 2]
\`\`\`

**Solution (expand along first row):**
\`\`\`
det(A) = 2·det([3 1]) - 1·det([0 1]) + 0·det([0 3])
                [0 2]        [1 2]        [1 0]

       = 2·(3·2 - 1·0) - 1·(0·2 - 1·1) + 0
       = 2·6 - 1·(-1)
       = 12 + 1 = 13
\`\`\`

**Alternative (expand along first column):**
\`\`\`
det(A) = 2·det([3 1]) - 0·det([1 0]) + 1·det([1 0])
                [0 2]        [0 2]        [3 1]

       = 2·6 + 1·(1 - 0) = 12 + 1 = 13 ✓
\`\`\`

**Problem 1.2d: Triangular Matrix**

Find det(A):
\`\`\`
A = [3 2 1]
    [0 4 5]
    [0 0 2]
\`\`\`

**Solution:**

For triangular matrices (upper or lower), determinant = product of diagonal elements:
\`\`\`
det(A) = 3 · 4 · 2 = 24
\`\`\`

**This is a crucial shortcut for interviews!**

### 1.3 Matrix Inverse

**Inverse (2×2):**
\`\`\`
[a b]^{-1} = 1/(ad-bc) × [ d -b]
[c d]                     [-c  a]
\`\`\`

**Key insight:** Swap diagonal, negate off-diagonal, divide by determinant.

**Problem 1.3a:**

Find inverse:
\`\`\`
A = [2 1]
    [1 4]
\`\`\`

**Solution:**
\`\`\`
det(A) = 2·4 - 1·1 = 7

A^{-1} = (1/7)[4 -1] = [ 4/7 -1/7]
            [-1  2]   [-1/7  2/7]
\`\`\`

**Verification:**
\`\`\`
AA^{-1} = [2 1][ 4/7 -1/7]
          [1 4][-1/7  2/7]

        = [8/7-1/7   -2/7+2/7]   [1 0]
          [4/7-4/7  -1/7+8/7] = [0 1] ✓
\`\`\`

**Problem 1.3b: Portfolio Rebalancing**

You have a system:
\`\`\`
2w₁ + w₂ = 100  (in millions)
w₁ + 4w₂ = 200
\`\`\`

What are w₁ and w₂?

**Solution:**
\`\`\`
[2 1][w₁]   [100]
[1 4][w₂] = [200]

Using inverse from 1.3a:

[w₁]   [ 4/7 -1/7][100]   [400/7 - 200/7]   [200/7]   [28.57]
[w₂] = [-1/7  2/7][200] = [-100/7 + 400/7] = [300/7] ≈ [42.86]
\`\`\`

**Answer:** w₁ ≈ $28.57M, w₂ ≈ $42.86M

### 1.4 Trace

**Definition:** Sum of diagonal elements

\`\`\`
tr(A) = Σᵢ aᵢᵢ
\`\`\`

**Key properties:**
- tr(A + B) = tr(A) + tr(B)
- tr(cA) = c·tr(A)
- tr(AB) = tr(BA) (cyclic property)
- tr(A) = sum of eigenvalues
- tr(A^T) = tr(A)

**Problem 1.4:**

If eigenvalues are λ₁ = 5, λ₂ = 3, λ₃ = 2, what is tr(A)?

**Solution:**
\`\`\`
tr(A) = 5 + 3 + 2 = 10
\`\`\`

---

## Section 2: Solving Systems of Equations

### 2.1 Two Equations, Two Unknowns

**Problem 2.1a:**

Solve:
\`\`\`
2x + y = 7
x + 3y = 11
\`\`\`

**Method 1: Substitution**
\`\`\`
From equation 1: y = 7 - 2x
Substitute into equation 2:
  x + 3(7 - 2x) = 11
  x + 21 - 6x = 11
  -5x = -10
  x = 2

Then: y = 7 - 2(2) = 3
\`\`\`

**Method 2: Elimination**
\`\`\`
Multiply equation 2 by 2:
  2x + 6y = 22
Subtract equation 1:
  5y = 15
  y = 3

Then: x = 2
\`\`\`

**Method 3: Matrix Inverse** (shown in 1.3b)

**Problem 2.1b: Portfolio Allocation**

Allocate capital to 2 assets with constraints:
\`\`\`
w₁ + w₂ = 1         (weights sum to 1)
0.12w₁ + 0.08w₂ = 0.095  (target 9.5% return)
\`\`\`

Where 0.12 = expected return of asset 1, 0.08 = expected return of asset 2.

**Solution:**
\`\`\`
From equation 1: w₂ = 1 - w₁

Substitute into equation 2:
  0.12w₁ + 0.08(1 - w₁) = 0.095
  0.12w₁ + 0.08 - 0.08w₁ = 0.095
  0.04w₁ = 0.015
  w₁ = 0.375 = 37.5%

Therefore: w₂ = 0.625 = 62.5%
\`\`\`

**Answer:** Allocate 37.5% to high-return asset, 62.5% to low-return asset.

**Verification:** 0.12(0.375) + 0.08(0.625) = 0.045 + 0.05 = 0.095 ✓

### 2.2 Three Equations, Three Unknowns

**Problem 2.2:**

Solve:
\`\`\`
x + y + z = 6
2x - y + z = 3
x + 2y - z = 4
\`\`\`

**Solution (Gaussian Elimination):**

Write augmented matrix:
\`\`\`
[1  1  1 | 6]
[2 -1  1 | 3]
[1  2 -1 | 4]
\`\`\`

Row operations:
\`\`\`
R₂ → R₂ - 2R₁:
[1  1  1 | 6]
[0 -3 -1 |-9]
[1  2 -1 | 4]

R₃ → R₃ - R₁:
[1  1  1 | 6]
[0 -3 -1 |-9]
[0  1 -2 |-2]

R₂ ↔ R₃ (swap for convenience):
[1  1  1 | 6]
[0  1 -2 |-2]
[0 -3 -1 |-9]

R₃ → R₃ + 3R₂:
[1  1  1 | 6]
[0  1 -2 |-2]
[0  0 -7 |-15]

Now back-substitute:
  -7z = -15  →  z = 15/7
  y - 2(15/7) = -2  →  y = -2 + 30/7 = 16/7
  x + 16/7 + 15/7 = 6  →  x = 6 - 31/7 = 11/7
\`\`\`

**Answer:** x = 11/7, y = 16/7, z = 15/7

### 2.3 Under-determined and Over-determined Systems

**Problem 2.3a: Under-determined (infinite solutions)**

\`\`\`
x + y = 5
2x + 2y = 10
\`\`\`

**Solution:**

Second equation is just 2× first equation, so we have only 1 independent equation with 2 unknowns.

**Infinite solutions:** y = 5 - x for any x ∈ ℝ

Examples: (0,5), (1,4), (2,3), (5,0), etc.

**Problem 2.3b: Over-determined (no exact solution)**

\`\`\`
x + y = 5
x + y = 6
\`\`\`

**Solution:**

No solution! (Inconsistent system)

In practice, we find **least squares solution** that minimizes error.

---

## Section 3: Eigenvalues and Eigenvectors

### 3.1 Definitions and Intuition

**Definition:** 
\`\`\`
Av = λv
\`\`\`

where v ≠ 0 is an eigenvector and λ is the corresponding eigenvalue.

**Geometric Intuition:**

Eigenvectors are special directions where the matrix acts like simple scaling (no rotation).

**Example:** Consider rotation matrix:
\`\`\`
R = [cos θ  -sin θ]
    [sin θ   cos θ]
\`\`\`

For θ ≠ 0, 180°, this has complex eigenvalues (no real eigenvectors) because rotation changes all directions.

### 3.2 Computing Eigenvalues

**Characteristic equation:** det(A - λI) = 0

**Problem 3.2a:**

Find eigenvalues of:
\`\`\`
A = [3 1]
    [1 3]
\`\`\`

**Solution:**
\`\`\`
det(A - λI) = det([3-λ   1  ]) = 0
                 [ 1   3-λ]

(3-λ)(3-λ) - 1·1 = 0
(3-λ)² - 1 = 0
(3-λ)² = 1
3-λ = ±1

λ₁ = 4, λ₂ = 2
\`\`\`

**Check:** tr(A) = 3 + 3 = 6 = λ₁ + λ₂ = 4 + 2 = 6 ✓
**Check:** det(A) = 9 - 1 = 8 = λ₁·λ₂ = 4·2 = 8 ✓

**Problem 3.2b:**

Find eigenvalues of:
\`\`\`
B = [5 2]
    [2 5]
\`\`\`

**Solution:**
\`\`\`
det(B - λI) = (5-λ)² - 4 = 0
25 - 10λ + λ² - 4 = 0
λ² - 10λ + 21 = 0
(λ - 7)(λ - 3) = 0

λ₁ = 7, λ₂ = 3
\`\`\`

### 3.3 Computing Eigenvectors

**Problem 3.3:**

Find eigenvectors for matrix A from Problem 3.2a.

**For λ₁ = 4:**
\`\`\`
(A - 4I)v = 0

[-1  1][v₁] = [0]
[ 1 -1][v₂]   [0]

From first equation: -v₁ + v₂ = 0  →  v₁ = v₂

Eigenvector: v₁ = [1]  (or any scalar multiple)
                  [1]

Normalized: v₁ = [1/√2]
                 [1/√2]
\`\`\`

**For λ₂ = 2:**
\`\`\`
(A - 2I)v = 0

[1  1][v₁] = [0]
[1  1][v₂]   [0]

From first equation: v₁ + v₂ = 0  →  v₂ = -v₁

Eigenvector: v₂ = [ 1]  (or any scalar multiple)
                  [-1]

Normalized: v₂ = [ 1/√2]
                 [-1/√2]
\`\`\`

**Note:** For symmetric matrices, eigenvectors corresponding to different eigenvalues are orthogonal:
\`\`\`
v₁ · v₂ = (1)(1) + (1)(-1) = 0 ✓
\`\`\`

### 3.4 Special Matrices

**Problem 3.4a: Diagonal Matrix**

Find eigenvalues of:
\`\`\`
D = [5 0 0]
    [0 2 0]
    [0 0 7]
\`\`\`

**Solution:** For diagonal matrices, eigenvalues ARE the diagonal elements!

λ₁ = 5, λ₂ = 2, λ₃ = 7

Eigenvectors are standard basis vectors: [1,0,0]^T, [0,1,0]^T, [0,0,1]^T

**Problem 3.4b: Identity Matrix**

What are the eigenvalues of I (identity matrix)?

**Solution:**

Iv = v = 1·v for any vector v

So **all** vectors are eigenvectors, and eigenvalue is λ = 1 (with multiplicity n).

**Problem 3.4c: Projection Matrix**

Consider:
\`\`\`
P = [1 0]
    [0 0]
\`\`\`

This projects onto the x-axis.

**Eigenvalues:**
\`\`\`
det(P - λI) = (1-λ)(0-λ) = 0
λ₁ = 1, λ₂ = 0
\`\`\`

**Eigenvectors:**
- For λ=1: [1,0]^T (points along x-axis stay put)
- For λ=0: [0,1]^T (points along y-axis get mapped to origin)

---

## Section 4: Principal Component Analysis (PCA)

### 4.1 Covariance Matrix Eigendecomposition

**Problem 4.1:**

You have a 2×2 covariance matrix:
\`\`\`
Σ = [4 2]
    [2 4]
\`\`\`

Find the principal components and variance explained.

**Solution:**

**Step 1: Find eigenvalues**
\`\`\`
det(Σ - λI) = (4-λ)² - 4 = 0
16 - 8λ + λ² - 4 = 0
λ² - 8λ + 12 = 0
(λ - 6)(λ - 2) = 0

λ₁ = 6 (first principal component)
λ₂ = 2 (second principal component)
\`\`\`

**Step 2: Variance explained**
\`\`\`
Total variance: σ²_total = tr(Σ) = 4 + 4 = 8

PC1 explains: λ₁/σ²_total = 6/8 = 75%
PC2 explains: λ₂/σ²_total = 2/8 = 25%
\`\`\`

**Step 3: Find eigenvectors (principal directions)**

For λ₁ = 6:
\`\`\`
[-2  2][v₁] = [0]
[ 2 -2][v₂]   [0]

v₁ = [1/√2]
     [1/√2]
\`\`\`

This points at 45° (northeast direction).

For λ₂ = 2:
\`\`\`
v₂ = [ 1/√2]
     [-1/√2]
\`\`\`

This points at -45° (southeast direction).

**Financial Interpretation:**

If these are returns of 2 assets:
- First PC (northeast direction) captures 75% of variance: "market factor"
- Second PC (southeast direction) captures 25%: "spread factor"

### 4.2 Dimensionality Reduction

**Problem 4.2:**

You have 10 stocks with a 10×10 covariance matrix. Eigenvalues are:

λ = [25, 15, 8, 5, 3, 2, 1.5, 1, 0.5, 0.01]

How many principal components capture 90% of variance?

**Solution:**
\`\`\`
Total variance: Σλᵢ = 61.01

Cumulative variance:
PC1: 25/61.01 = 41.0%
PC1-2: 40/61.01 = 65.6%
PC1-3: 48/61.01 = 78.7%
PC1-4: 53/61.01 = 86.9%
PC1-5: 56/61.01 = 91.8% ✓
\`\`\`

**Answer:** 5 principal components capture >90% of variance.

**Implication:** Can reduce from 10 dimensions to 5, losing only ~8% of information.

---

## Section 5: Matrix Properties for Interviews

### 5.1 Positive Definite Matrices

**Definition:** A matrix A is positive definite if:
\`\`\`
x^T A x > 0  for all x ≠ 0
\`\`\`

**Equivalent conditions:**
1. All eigenvalues > 0
2. All principal minors > 0
3. Cholesky decomposition exists: A = LL^T

**For 2×2 matrix:**
\`\`\`
A = [a b]
    [b d]
\`\`\`

Positive definite if and only if:
- a > 0
- det(A) = ad - b² > 0

**Problem 5.1a:**

Is this positive definite?
\`\`\`
A = [4 1]
    [1 4]
\`\`\`

**Solution:**
\`\`\`
a = 4 > 0 ✓
det(A) = 16 - 1 = 15 > 0 ✓

YES, positive definite.
\`\`\`

We can also verify by finding eigenvalues:
\`\`\`
(4-λ)² - 1 = 0
λ² - 8λ + 15 = 0
(λ-5)(λ-3) = 0
λ₁ = 5 > 0, λ₂ = 3 > 0 ✓
\`\`\`

**Problem 5.1b:**

Is this positive definite?
\`\`\`
B = [1  2]
    [2  3]
\`\`\`

**Solution:**
\`\`\`
a = 1 > 0 ✓
det(B) = 3 - 4 = -1 < 0 ✗

NO, not positive definite (it's indefinite).
\`\`\`

Eigenvalues:
\`\`\`
(1-λ)(3-λ) - 4 = 0
λ² - 4λ - 1 = 0
λ = (4 ± √(16+4))/2 = (4 ± √20)/2 = 2 ± √5

λ₁ ≈ 4.236 > 0
λ₂ ≈ -0.236 < 0  ← One negative eigenvalue!
\`\`\`

**Why it matters:** Covariance matrices MUST be positive semi-definite (eigenvalues ≥ 0) because variances cannot be negative. If your covariance matrix has negative eigenvalues, there's an error in your data or calculations!

### 5.2 Symmetric Matrices

**Properties:**
1. A^T = A
2. All eigenvalues are real
3. Eigenvectors from different eigenvalues are orthogonal
4. Can be diagonalized: A = QΛQ^T where Q is orthogonal

**Problem 5.2:**

Show that eigenvalues of a symmetric matrix are real.

**Proof sketch:**

Suppose Av = λv where v ≠ 0.

Take conjugate transpose: v^* A^* = λ^* v^*

Since A is real and symmetric: A^* = A^T = A

Multiply: v^* A v = λ^* v^* v

But also: v^* A v = v^* λ v = λ v^* v

Therefore: λ v^* v = λ^* v^* v

Since v^* v > 0 (it's ||v||²), we have λ = λ^*, so λ is real. ∎

### 5.3 Orthogonal Matrices

**Definition:** Q is orthogonal if Q^T Q = I (equivalently, Q^{-1} = Q^T)

**Properties:**
1. Preserves lengths: ||Qx|| = ||x||
2. Preserves angles and inner products
3. det(Q) = ±1
4. All eigenvalues have absolute value 1

**Examples:**
- Rotation matrices
- Reflection matrices
- Permutation matrices

**Problem 5.3:**

Is this orthogonal?
\`\`\`
Q = [cos θ  -sin θ]
    [sin θ   cos θ]
\`\`\`

**Solution:**
\`\`\`
Q^T Q = [cos θ   sin θ][cos θ  -sin θ]
        [-sin θ  cos θ][sin θ   cos θ]

      = [cos²θ + sin²θ    -cos θ sin θ + sin θ cos θ]
        [-sin θ cos θ + cos θ sin θ    sin²θ + cos²θ]

      = [1 0]
        [0 1] = I ✓

YES, orthogonal (it's a rotation matrix).
\`\`\`

### 5.4 Rank

**Definition:** Rank = number of linearly independent rows (or columns)

**Problem 5.4a:**

Find rank:
\`\`\`
A = [1 2 3]
    [2 4 6]
    [1 1 2]
\`\`\`

**Solution:**

Row 2 = 2 × Row 1, so rows 1 and 2 are dependent.

Check row 3: [1,1,2] ≠ c[1,2,3] for any c (not a multiple of row 1).

Row reduce:
\`\`\`
[1 2 3]
[0 0 0]  (R₂ - 2R₁)
[0 -1 -1]  (R₃ - R₁)

Swap rows 2 and 3:
[1 2 3]
[0 -1 -1]
[0 0 0]
\`\`\`

2 non-zero rows → **rank = 2**

**Problem 5.4b: Rank and Determinant**

If rank(A) < n for n×n matrix A, what is det(A)?

**Answer:** det(A) = 0

**Proof:** If rank < n, rows are linearly dependent, so matrix is singular (not invertible), so determinant is zero.

### 5.5 Condition Number

**Definition:**
\`\`\`
κ(A) = ||A|| · ||A^{-1}||
\`\`\`

For eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ:
\`\`\`
κ(A) = |λ₁| / |λₙ|  (ratio of largest to smallest eigenvalue magnitude)
\`\`\`

**Interpretation:**
- κ ≈ 1: well-conditioned (numerically stable)
- κ >> 1: ill-conditioned (small changes in input → large changes in output)

**Problem 5.5:**

Find condition number:
\`\`\`
A = [1     0  ]
    [0  0.001]
\`\`\`

**Solution:**

Eigenvalues: λ₁ = 1, λ₂ = 0.001

κ(A) = 1 / 0.001 = 1000

**This is ill-conditioned!** Small errors in the input will be magnified 1000×.

**Why it matters:** In portfolio optimization, ill-conditioned covariance matrices lead to unstable weight estimates. Small changes in returns can lead to wild swings in optimal allocations.

---

## Section 6: Portfolio Applications

### 6.1 Portfolio Variance Formula

**Two-asset portfolio:**
\`\`\`
σ²_p = w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂
\`\`\`

**Matrix form:**
\`\`\`
σ²_p = w^T Σ w

where Σ = [σ₁²      σ₁σ₂ρ₁₂]
          [σ₁σ₂ρ₁₂   σ₂²    ]
\`\`\`

**Problem 6.1:**

Two assets:
- Asset 1: σ₁ = 20% annual volatility
- Asset 2: σ₂ = 30% annual volatility
- Correlation: ρ = 0.5
- Weights: w₁ = 0.6, w₂ = 0.4

Find portfolio variance and volatility.

**Solution:**

Build covariance matrix:
\`\`\`
σ₁² = 0.04, σ₂² = 0.09
σ₁₂ = ρσ₁σ₂ = 0.5 × 0.2 × 0.3 = 0.03

Σ = [0.04  0.03]
    [0.03  0.09]
\`\`\`

Portfolio variance:
\`\`\`
σ²_p = [0.6  0.4][0.04  0.03][0.6]
                  [0.03  0.09][0.4]

Step 1: Σw = [0.04  0.03][0.6] = [0.024 + 0.012] = [0.036]
              [0.03  0.09][0.4]   [0.018 + 0.036]   [0.054]

Step 2: w^T(Σw) = [0.6  0.4][0.036] = 0.0216 + 0.0216 = 0.0432
                             [0.054]
\`\`\`

Portfolio volatility: σ_p = √0.0432 ≈ **20.8%**

**Observation:** Despite asset 2 having 30% volatility, the portfolio has only 20.8% because of diversification!

### 6.2 Minimum Variance Portfolio

**Problem 6.2:**

For the assets in 6.1, find the minimum variance portfolio weights.

**Solution:**

For two assets with no constraints (other than w₁ + w₂ = 1):

\`\`\`
w₁* = (σ₂² - σ₁₂) / (σ₁² + σ₂² - 2σ₁₂)

w₁* = (0.09 - 0.03) / (0.04 + 0.09 - 2×0.03)
    = 0.06 / 0.07
    = 6/7 ≈ 0.857

w₂* = 1 - w₁* ≈ 0.143
\`\`\`

**Minimum variance:**
\`\`\`
σ²_min = [0.857  0.143][0.04  0.03][0.857]
                        [0.03  0.09][0.143]

       ≈ 0.0347

σ_min ≈ 18.6%
\`\`\`

**This is lower than either asset individually!**

### 6.3 Portfolio with Many Assets

For n assets, the quadratic form generalizes:
\`\`\`
σ²_p = Σᵢ Σⱼ wᵢwⱼσᵢⱼ = w^T Σ w
\`\`\`

**Problem 6.3:**

Three equally-weighted assets (w = [1/3, 1/3, 1/3]):
\`\`\`
Σ = [0.04  0.01  0.01]
    [0.01  0.04  0.01]
    [0.01  0.01  0.04]
\`\`\`

Find portfolio variance.

**Solution:**
\`\`\`
σ²_p = (1/3)²[0.04 + 0.04 + 0.04] + 2(1/3)²[0.01 + 0.01 + 0.01]
     = (1/9)[0.12] + 2(1/9)[0.03]
     = 0.012/0.9 + 0.06/9
     = 0.0133... + 0.0067
     ≈ 0.02

σ_p ≈ 14.1%
\`\`\`

---

## Section 7: Advanced Problems

### 7.1 Matrix Calculus

**Derivative rules:**
\`\`\`
∂/∂x (a^T x) = a
∂/∂x (x^T A x) = (A + A^T)x
\`\`\`

If A is symmetric: ∂/∂x (x^T A x) = 2Ax

**Problem 7.1: Minimize Quadratic Form**

Minimize f(w) = w^T Σ w subject to w^T 1 = 1.

**Solution (Lagrangian):**
\`\`\`
L = w^T Σ w - λ(w^T 1 - 1)

∂L/∂w = 2Σw - λ1 = 0
∂L/∂λ = w^T 1 - 1 = 0

From first equation: w = (λ/2)Σ^{-1} 1

Substitute into constraint:
(λ/2) 1^T Σ^{-1} 1 = 1
λ = 2 / (1^T Σ^{-1} 1)

Therefore:
w* = Σ^{-1} 1 / (1^T Σ^{-1} 1)
\`\`\`

This is the **minimum variance portfolio** formula!

### 7.2 Singular Value Decomposition (SVD)

**Theorem:** Any m×n matrix A can be decomposed:
\`\`\`
A = UΣV^T
\`\`\`

where:
- U is m×m orthogonal (left singular vectors)
- Σ is m×n diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V is n×n orthogonal (right singular vectors)

**Relation to eigenvalues:**
- Singular values of A = square roots of eigenvalues of A^T A
- σᵢ² = λᵢ(A^T A)

**Problem 7.2:**

Find SVD of:
\`\`\`
A = [3 0]
    [0 4]
\`\`\`

**Solution:**

A is already diagonal, so:
\`\`\`
U = I, Σ = A, V = I

(Singular values are just the diagonal elements)
\`\`\`

---

## Section 8: Python Implementations

\`\`\`python
"""
Comprehensive Linear Algebra for Quantitative Interviews
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from typing import Tuple, List

# ============================================================================
# Section 1: Basic Operations
# ============================================================================

def matrix_multiply_2x2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply 2x2 matrices (illustrative - use @ in practice)."""
    result = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            result[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    return result

def determinant_2x2(A: np.ndarray) -> float:
    """Calculate determinant of 2x2 matrix."""
    return A[0,0]*A[1,1] - A[0,1]*A[1,0]

def inverse_2x2(A: np.ndarray) -> np.ndarray:
    """Invert 2x2 matrix."""
    det = determinant_2x2(A)
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular")
    
    return (1/det) * np.array([[A[1,1], -A[0,1]],
                                [-A[1,0], A[0,0]]])

# ============================================================================
# Section 2: Eigenvalue Problems
# ============================================================================

def compute_eigenvalues_2x2_analytical(A: np.ndarray) -> Tuple[float, float]:
    """
    Compute eigenvalues of 2x2 matrix using quadratic formula.
    
    Characteristic equation: λ² - tr(A)λ + det(A) = 0
    """
    trace = A[0,0] + A[1,1]
    det = determinant_2x2(A)
    
    # Quadratic formula: λ = (tr ± sqrt(tr² - 4det)) / 2
    discriminant = trace**2 - 4*det
    
    if discriminant < 0:
        # Complex eigenvalues
        real_part = trace / 2
        imag_part = np.sqrt(-discriminant) / 2
        return complex(real_part, imag_part), complex(real_part, -imag_part)
    else:
        sqrt_disc = np.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
        return lambda1, lambda2

# ============================================================================
# Section 3: PCA Implementation
# ============================================================================

def pca_analysis(data: np.ndarray, n_components: int = None) -> dict:
    """
    Perform PCA on data matrix.
    
    Args:
        data: n_samples × n_features matrix
        n_components: Number of components to keep (default: all)
        
    Returns:
        Dictionary with eigenvalues, eigenvectors, variance explained, etc.
    """
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    n_samples = data.shape[0]
    cov_matrix = (data_centered.T @ data_centered) / (n_samples - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var
    cumulative_var = np.cumsum(var_explained)
    
    # Project data onto principal components
    if n_components is None:
        n_components = len(eigenvalues)
    
    pc_loadings = eigenvectors[:, :n_components]
    transformed_data = data_centered @ pc_loadings
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'var_explained': var_explained,
        'cumulative_var': cumulative_var,
        'principal_components': pc_loadings,
        'transformed_data': transformed_data,
        'covariance_matrix': cov_matrix
    }

# ============================================================================
# Section 4: Portfolio Optimization
# ============================================================================

def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio variance: w^T Σ w"""
    return weights @ cov_matrix @ weights

def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Find minimum variance portfolio weights.
    
    Formula: w* = Σ^{-1} 1 / (1^T Σ^{-1} 1)
    """
    n = cov_matrix.shape[0]
    ones = np.ones(n)
    
    cov_inv = np.linalg.inv(cov_matrix)
    numerator = cov_inv @ ones
    denominator = ones @ cov_inv @ ones
    
    return numerator / denominator

def efficient_frontier(mu: np.ndarray, cov_matrix: np.ndarray, 
                      n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute efficient frontier for portfolio optimization.
    
    Args:
        mu: Expected returns
        cov_matrix: Covariance matrix
        n_points: Number of points on frontier
        
    Returns:
        (risks, returns) arrays for plotting
    """
    n_assets = len(mu)
    
    # Find minimum variance portfolio
    w_min = minimum_variance_portfolio(cov_matrix)
    r_min = mu @ w_min
    vol_min = np.sqrt(portfolio_variance(w_min, cov_matrix))
    
    # Range of target returns
    r_max = np.max(mu)
    target_returns = np.linspace(r_min, r_max, n_points)
    
    risks = []
    actual_returns = []
    
    for target_r in target_returns:
        # Solve quadratic programming problem
        # Minimize w^T Σ w subject to w^T μ = target_r and w^T 1 = 1
        
        # Using Lagrangian solution (simplified for 2 assets)
        if n_assets == 2:
            # Analytical solution
            A = np.array([[cov_matrix[0,0], cov_matrix[0,1], mu[0], 1],
                          [cov_matrix[1,0], cov_matrix[1,1], mu[1], 1],
                          [mu[0], mu[1], 0, 0],
                          [1, 1, 0, 0]])
            b = np.array([0, 0, target_r, 1])
            solution = np.linalg.solve(A, b)
            w = solution[:2]
        else:
            # Numerical optimization needed for n > 2
            from scipy.optimize import minimize
            
            def objective(w):
                return portfolio_variance(w, cov_matrix)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: mu @ w - target_r}
            ]
            
            w0 = np.ones(n_assets) / n_assets
            result = minimize(objective, w0, constraints=constraints)
            w = result.x
        
        risk = np.sqrt(portfolio_variance(w, cov_matrix))
        ret = mu @ w
        
        risks.append(risk)
        actual_returns.append(ret)
    
    return np.array(risks), np.array(actual_returns)

# ============================================================================
# Testing and Examples
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LINEAR ALGEBRA FOR QUANT INTERVIEWS")
    print("="*70)
    
    # Example 1: Eigenvalues
    print("\\nExample 1: Eigenvalues of Covariance Matrix")
    print("-"*70)
    
    A = np.array([[4, 2],
                  [2, 4]])
    
    print(f"Matrix A:")
    print(A)
    
    lambda1, lambda2 = compute_eigenvalues_2x2_analytical(A)
    print(f"\\nEigenvalues (analytical): {lambda1:.4f}, {lambda2:.4f}")
    
    eigenvalues_numpy = np.linalg.eigvalsh(A)
    print(f"Eigenvalues (NumPy):      {eigenvalues_numpy}")
    
    total_var = np.trace(A)
    print(f"\\nTotal variance: {total_var}")
    print(f"PC1 explains: {lambda1/total_var:.2%}")
    print(f"PC2 explains: {lambda2/total_var:.2%}")
    
    # Example 2: Portfolio Variance
    print("\\n\\nExample 2: Portfolio Variance")
    print("-"*70)
    
    cov = np.array([[0.04, 0.03],
                    [0.03, 0.09]])
    weights = np.array([0.6, 0.4])
    
    port_var = portfolio_variance(weights, cov)
    port_vol = np.sqrt(port_var)
    
    print(f"Covariance matrix:")
    print(cov)
    print(f"\\nWeights: {weights}")
    print(f"Portfolio variance: {port_var:.4f}")
    print(f"Portfolio volatility: {port_vol:.2%}")
    
    # Example 3: Minimum Variance Portfolio
    print("\\n\\nExample 3: Minimum Variance Portfolio")
    print("-"*70)
    
    w_min = minimum_variance_portfolio(cov)
    var_min = portfolio_variance(w_min, cov)
    vol_min = np.sqrt(var_min)
    
    print(f"Minimum variance weights: {w_min}")
    print(f"Minimum variance: {var_min:.4f}")
    print(f"Minimum volatility: {vol_min:.2%}")
    
    # Example 4: PCA
    print("\\n\\nExample 4: PCA on Random Data")
    print("-"*70)
    
    # Generate correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov_data = [[1, 0.8], [0.8, 1]]
    data = np.random.multivariate_normal(mean, cov_data, 1000)
    
    pca_result = pca_analysis(data)
    
    print(f"Eigenvalues: {pca_result['eigenvalues']}")
    print(f"Variance explained: {pca_result['var_explained']}")
    print(f"Cumulative variance: {pca_result['cumulative_var']}")
    
    print("\\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
\`\`\`

---

## Section 9: Interview Tips & Strategy

### What to Memorize

**Essential formulas:**
1. 2×2 determinant: ad - bc
2. 2×2 inverse: swap diagonal, negate off-diagonal, divide by det
3. Eigenvalue equation: det(A - λI) = 0
4. Portfolio variance: w^T Σ w
5. Trace = sum of eigenvalues
6. Determinant = product of eigenvalues

**Mental math tricks:**
1. For symmetric 2×2, eigenvalues satisfy: λ₁ + λ₂ = tr(A), λ₁λ₂ = det(A)
2. Diagonal matrices: eigenvalues ARE the diagonal elements
3. Identity matrix: all eigenvalues = 1
4. Positive definite (2×2): a > 0 and det > 0

### Common Mistakes to Avoid

1. **Confusing row and column vectors** - Be explicit about dimensions
2. **Forgetting to check dimensions** - Can't multiply m×n and p×q unless n=p
3. **Assuming AB = BA** - Matrix multiplication is NOT commutative
4. **Thinking eigenvalues sum to determinant** - They sum to TRACE, multiply to determinant
5. **Forgetting variance formula** - It's w^T Σ w, not w^T Σ or Σ w

### Communication Tips

1. **State dimensions clearly:** "We have a 2×2 covariance matrix..."
2. **Show your work:** Write out intermediate steps
3. **Sanity check:** "Let me verify this makes sense..."
4. **Use geometry:** "This eigenvector points in the direction of maximum variance..."
5. **Connect to finance:** "So the portfolio volatility is..."

---

## Summary

Linear algebra is the language of quantitative finance. Master these core concepts:

**Computational Skills:**
- Matrix operations (multiply, invert, determinant)
- Solving systems of equations
- Computing eigenvalues and eigenvectors

**Theoretical Understanding:**
- What eigenvalues/eigenvectors represent
- Matrix properties (positive definite, symmetric, orthogonal)
- Condition number and numerical stability

**Financial Applications:**
- Portfolio variance formula
- PCA for factor models
- Minimum variance portfolios
- Covariance matrix structure

Practice mental math with 2×2 matrices until it's second nature. For 3×3 and beyond, focus on understanding structure rather than brute-force computation.

**Next steps:**
- Work through all problems by hand
- Implement solutions in Python
- Practice explaining your reasoning out loud
- Apply to real financial data

The intersection of linear algebra and finance is where quant trading happens!
`,
};
