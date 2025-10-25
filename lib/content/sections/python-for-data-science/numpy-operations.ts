/**
 * Section: NumPy Operations
 * Module: Python for Data Science
 *
 * Covers element-wise operations, aggregations, boolean indexing, concatenation, linear algebra, and random numbers
 */

export const numpyOperations = {
  id: 'numpy-operations',
  title: 'NumPy Operations',
  content: `
# NumPy Operations

## Introduction

NumPy\'s true power lies in its rich set of operations that work efficiently on entire arrays without explicit loops. This section covers the essential operations you'll use daily in data science and machine learning: element-wise arithmetic, aggregations, boolean operations, array manipulation, linear algebra, and random number generation.

**Key Concept: Vectorization**

Instead of writing Python loops, NumPy operations apply to entire arrays at once, leveraging optimized C code:

\`\`\`python
import numpy as np

# Slow: Python loop
arr = list (range(1000000))
result = []
for x in arr:
    result.append (x ** 2)

# Fast: NumPy vectorization
arr = np.arange(1000000)
result = arr ** 2  # 50-100x faster!
\`\`\`

## Element-wise Operations

NumPy operations are element-wise by default, meaning they apply to each element independently:

### Arithmetic Operations

\`\`\`python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

# Element-wise addition
print(f"Addition: {arr1 + arr2}")  # [11 22 33 44 55]

# Element-wise subtraction
print(f"Subtraction: {arr2 - arr1}")  # [9 18 27 36 45]

# Element-wise multiplication
print(f"Multiplication: {arr1 * arr2}")  # [10 40 90 160 250]

# Element-wise division
print(f"Division: {arr2 / arr1}")  # [10. 10. 10. 10. 10.]

# Integer division (floor division)
print(f"Floor division: {arr2 // arr1}")  # [10 10 10 10 10]

# Modulo
print(f"Modulo: {arr2 % 3}")  # [1 2 0 1 2]

# Power
print(f"Power: {arr1 ** 2}")  # [1 4 9 16 25]
\`\`\`

### Operations with Scalars

\`\`\`python
arr = np.array([1, 2, 3, 4, 5])

# Scalar operations broadcast to all elements
print(f"Add 10: {arr + 10}")  # [11 12 13 14 15]
print(f"Multiply by 2: {arr * 2}")  # [2 4 6 8 10]
print(f"Square: {arr ** 2}")  # [1 4 9 16 25]
print(f"Reciprocal: {1 / arr}")  # [1.0 0.5 0.333... 0.25 0.2]
\`\`\`

### Mathematical Functions

\`\`\`python
arr = np.array([1, 4, 9, 16, 25])

# Square root
print(f"Square root: {np.sqrt (arr)}")  # [1. 2. 3. 4. 5.]

# Exponential and logarithm
arr = np.array([1, 2, 3])
print(f"Exponential: {np.exp (arr)}")  # [2.718... 7.389... 20.085...]
print(f"Natural log: {np.log (arr)}")  # [0. 0.693... 1.098...]
print(f"Log base 10: {np.log10(arr)}")  # [0. 0.301... 0.477...]
print(f"Log base 2: {np.log2(arr)}")  # [0. 1. 1.584...]

# Trigonometric functions
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"Sine: {np.sin (angles)}")
print(f"Cosine: {np.cos (angles)}")
print(f"Tangent: {np.tan (angles)}")

# Rounding functions
arr = np.array([1.2, 2.5, 3.7, 4.9])
print(f"Round: {np.round (arr)}")  # [1. 2. 4. 5.]
print(f"Floor: {np.floor (arr)}")  # [1. 2. 3. 4.]
print(f"Ceil: {np.ceil (arr)}")  # [2. 3. 4. 5.]

# Absolute value
arr = np.array([-1, -2, 3, -4, 5])
print(f"Absolute: {np.abs (arr)}")  # [1 2 3 4 5]

# Sign function
print(f"Sign: {np.sign (arr)}")  # [-1 -1 1 -1 1]
\`\`\`

### Comparison Operations

\`\`\`python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

# Element-wise comparisons return boolean arrays
print(f"Equal: {arr1 == arr2}")  # [False False True False False]
print(f"Not equal: {arr1 != arr2}")  # [True True False True True]
print(f"Greater: {arr1 > arr2}")  # [False False False True True]
print(f"Greater or equal: {arr1 >= 3}")  # [False False True True True]

# Compare with scalar
print(f"Greater than 3: {arr1 > 3}")  # [False False False True True]
\`\`\`

## Aggregation Functions

Aggregation functions reduce array dimensions by computing summary statistics:

### Basic Aggregations

\`\`\`python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Sum
print(f"Sum: {arr.sum()}")  # 55
print(f"Sum (function): {np.sum (arr)}")  # 55

# Mean
print(f"Mean: {arr.mean()}")  # 5.5

# Median
print(f"Median: {np.median (arr)}")  # 5.5

# Standard deviation and variance
print(f"Std dev: {arr.std()}")  # 2.872...
print(f"Variance: {arr.var()}")  # 8.25

# Min and max
print(f"Min: {arr.min()}")  # 1
print(f"Max: {arr.max()}")  # 10

# Min and max index
print(f"Argmin: {arr.argmin()}")  # 0
print(f"Argmax: {arr.argmax()}")  # 9

# Cumulative sum
print(f"Cumsum: {arr.cumsum()}")  # [1 3 6 10 15 21 28 36 45 55]

# Cumulative product
arr_small = np.array([1, 2, 3, 4, 5])
print(f"Cumprod: {arr_small.cumprod()}")  # [1 2 6 24 120]
\`\`\`

### Aggregations Along Axes

For multi-dimensional arrays, you can aggregate along specific axes:

\`\`\`python
# 2D array: 3 rows, 4 columns
arr2d = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print(f"Array:\\n{arr2d}")

# Aggregate over all elements
print(f"\\nTotal sum: {arr2d.sum()}")  # 78

# Aggregate along axis 0 (down columns, across rows)
print(f"Sum axis 0: {arr2d.sum (axis=0)}")  # [15 18 21 24]
print(f"Mean axis 0: {arr2d.mean (axis=0)}")  # [5. 6. 7. 8.]

# Aggregate along axis 1 (across columns, within rows)
print(f"Sum axis 1: {arr2d.sum (axis=1)}")  # [10 26 42]
print(f"Mean axis 1: {arr2d.mean (axis=1)}")  # [2.5 6.5 10.5]

# Practical example: row and column statistics
print(f"\\nMax in each row: {arr2d.max (axis=1)}")  # [4 8 12]
print(f"Min in each column: {arr2d.min (axis=0)}")  # [1 2 3 4]
\`\`\`

### Statistical Functions

\`\`\`python
data = np.random.randn(1000)  # 1000 samples from standard normal

# Percentiles
print(f"25th percentile: {np.percentile (data, 25):.3f}")
print(f"50th percentile (median): {np.percentile (data, 50):.3f}")
print(f"75th percentile: {np.percentile (data, 75):.3f}")

# Quantiles (equivalent to percentiles/100)
print(f"Quantiles [0.25, 0.5, 0.75]: {np.quantile (data, [0.25, 0.5, 0.75])}")

# Range
print(f"Range (max - min): {np.ptp (data):.3f}")  # Peak to peak

# Correlation coefficient
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
correlation = np.corrcoef (x, y)
print(f"\\nCorrelation matrix:\\n{correlation}")
\`\`\`

## Boolean Indexing and Masking

Boolean arrays enable powerful filtering and conditional operations:

### Creating Boolean Masks

\`\`\`python
arr = np.array([1, 5, 3, 8, 2, 9, 7, 4, 6, 10])

# Create mask
mask = arr > 5
print(f"Mask: {mask}")  # [False False False True False True True False True True]
print(f"Type: {type (mask)}")  # <class 'numpy.ndarray'>
print(f"Dtype: {mask.dtype}")  # bool

# Use mask to filter
filtered = arr[mask]
print(f"Values > 5: {filtered}")  # [8 9 7 6 10]

# Count True values
print(f"Count > 5: {mask.sum()}")  # 5 (True=1, False=0)
\`\`\`

### Compound Conditions

\`\`\`python
arr = np.array([1, 5, 3, 8, 2, 9, 7, 4, 6, 10])

# AND condition (&, not 'and')
mask = (arr > 3) & (arr < 8)
print(f"3 < x < 8: {arr[mask]}")  # [5 7 4 6]

# OR condition (|, not 'or')
mask = (arr < 3) | (arr > 8)
print(f"x < 3 OR x > 8: {arr[mask]}")  # [1 2 9 10]

# NOT condition (~, not 'not')
mask = ~(arr > 5)  # Same as arr <= 5
print(f"NOT x > 5: {arr[mask]}")  # [1 5 3 2 4]

# Complex condition
mask = ((arr > 3) & (arr < 8)) | (arr == 10)
print(f"Complex: {arr[mask]}")  # [5 7 4 6 10]
\`\`\`

### Conditional Operations

\`\`\`python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# np.where: ternary operator for arrays
# np.where (condition, value_if_true, value_if_false)
result = np.where (arr > 5, "high", "low")
print(f"Where result: {result}")

# Use with numbers
result = np.where (arr % 2 == 0, arr, -arr)  # Keep evens, negate odds
print(f"Conditional: {result}")  # [-1 2 -3 4 -5 6 -7 8 -9 10]

# Replace values matching condition
arr_copy = arr.copy()
arr_copy[arr_copy > 5] = 999
print(f"Replaced > 5: {arr_copy}")  # [1 2 3 4 5 999 999 999 999 999]

# Clip values to range
clipped = np.clip (arr, 3, 7)
print(f"Clipped [3, 7]: {clipped}")  # [3 3 3 4 5 6 7 7 7 7]
\`\`\`

### Practical Example: Outlier Detection

\`\`\`python
# Generate data with outliers
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 10, 95),  # Normal data
    np.array([150, -50, 200, -100, 180])  # Outliers
])

# Method 1: Z-score
mean = data.mean()
std = data.std()
z_scores = np.abs((data - mean) / std)
outliers_zscore = data[z_scores > 3]
print(f"Outliers (Z-score > 3): {len (outliers_zscore)} found")

# Method 2: IQR (Interquartile Range)
q1 = np.percentile (data, 25)
q3 = np.percentile (data, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
print(f"Outliers (IQR method): {len (outliers_iqr)} found")

# Remove outliers
clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
print(f"Original size: {len (data)}, Clean size: {len (clean_data)}")
\`\`\`

## Array Concatenation and Splitting

### Concatenation

\`\`\`python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Concatenate 1D arrays
result = np.concatenate([arr1, arr2, arr3])
print(f"Concatenated: {result}")  # [1 2 3 4 5 6 7 8 9]

# Concatenate 2D arrays
arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6], [7, 8]])

# Vertical stack (along rows, axis=0)
vstack = np.vstack([arr1_2d, arr2_2d])
# or: np.concatenate([arr1_2d, arr2_2d], axis=0)
print(f"Vertical stack:\\n{vstack}")
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Horizontal stack (along columns, axis=1)
hstack = np.hstack([arr1_2d, arr2_2d])
# or: np.concatenate([arr1_2d, arr2_2d], axis=1)
print(f"Horizontal stack:\\n{hstack}")
# [[1 2 5 6]
#  [3 4 7 8]]

# Depth stack (creates 3D array)
dstack = np.dstack([arr1_2d, arr2_2d])
print(f"Depth stack shape: {dstack.shape}")  # (2, 2, 2)
\`\`\`

### Splitting

\`\`\`python
arr = np.arange(12)

# Split into equal parts
split_arr = np.split (arr, 3)  # Split into 3 arrays
print(f"Split into 3: {split_arr}")  # [array([0, 1, 2, 3]), ...]

# Split at specific indices
split_arr = np.split (arr, [3, 7])  # Split at indices 3 and 7
print(f"Split at [3, 7]: {split_arr}")
# [array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9, 10, 11])]

# 2D array splitting
arr2d = np.arange(16).reshape(4, 4)
print(f"Array:\\n{arr2d}")

# Horizontal split (split columns)
hsplit = np.hsplit (arr2d, 2)
print(f"Horizontal split into 2:\\n{hsplit[0]}\\n{hsplit[1]}")

# Vertical split (split rows)
vsplit = np.vsplit (arr2d, 2)
print(f"Vertical split into 2:\\n{vsplit[0]}\\n{vsplit[1]}")
\`\`\`

### Repeating and Tiling

\`\`\`python
arr = np.array([1, 2, 3])

# Repeat each element
repeated = np.repeat (arr, 3)
print(f"Repeated: {repeated}")  # [1 1 1 2 2 2 3 3 3]

# Repeat with different counts
repeated = np.repeat (arr, [2, 3, 1])
print(f"Repeated custom: {repeated}")  # [1 1 2 2 2 3]

# Tile (repeat entire array)
tiled = np.tile (arr, 3)
print(f"Tiled: {tiled}")  # [1 2 3 1 2 3 1 2 3]

# Tile in 2D
tiled_2d = np.tile (arr, (2, 3))  # 2 rows, 3 repetitions per row
print(f"Tiled 2D:\\n{tiled_2d}")
# [[1 2 3 1 2 3 1 2 3]
#  [1 2 3 1 2 3 1 2 3]]
\`\`\`

## Linear Algebra Operations

NumPy provides comprehensive linear algebra functionality:

### Matrix Multiplication

\`\`\`python
# Matrix multiplication (dot product)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Method 1: np.dot()
C = np.dot(A, B)
print(f"A @ B (dot):\\n{C}")

# Method 2: @ operator (Python 3.5+)
C = A @ B
print(f"A @ B (@):\\n{C}")

# Method 3: .dot() method
C = A.dot(B)
print(f"A @ B (.dot):\\n{C}")

# Result:
# [[19 22]
#  [43 50]]

# Element-wise multiplication (different!)
element_wise = A * B
print(f"Element-wise A * B:\\n{element_wise}")
# [[5 12]
#  [21 32]]
\`\`\`

### Vector Operations

\`\`\`python
# Dot product of vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot (v1, v2)
print(f"Dot product: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

# Cross product (3D vectors)
cross_product = np.cross (v1, v2)
print(f"Cross product: {cross_product}")  # [-3 6 -3]

# Norm (magnitude)
norm = np.linalg.norm (v1)
print(f"L2 norm: {norm:.3f}")  # sqrt(1^2 + 2^2 + 3^2) = 3.742

# Different norms
l1_norm = np.linalg.norm (v1, ord=1)  # Sum of absolute values
l2_norm = np.linalg.norm (v1, ord=2)  # Euclidean (default)
linf_norm = np.linalg.norm (v1, ord=np.inf)  # Max absolute value
print(f"L1: {l1_norm}, L2: {l2_norm:.3f}, L-inf: {linf_norm}")
\`\`\`

### Matrix Operations

\`\`\`python
# Matrix transpose
A = np.array([[1, 2, 3], [4, 5, 6]])
A_T = A.T
print(f"Transpose:\\n{A_T}")

# Matrix inverse (square matrices only)
A = np.array([[1, 2], [3, 4]])
try:
    A_inv = np.linalg.inv(A)
    print(f"Inverse:\\n{A_inv}")
    # Verify: A @ A_inv should be identity
    identity = A @ A_inv
    print(f"A @ A_inv:\\n{identity}")
except np.linalg.LinAlgError:
    print("Matrix is singular (non-invertible)")

# Determinant
det = np.linalg.det(A)
print(f"Determinant: {det}")  # -2.0

# Trace (sum of diagonal elements)
trace = np.trace(A)
print(f"Trace: {trace}")  # 1 + 4 = 5

# Rank
rank = np.linalg.matrix_rank(A)
print(f"Rank: {rank}")  # 2

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\\n{eigenvectors}")

# Solve linear system Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print(f"Solution to Ax = b: {x}")  # [2. 3.]
# Verify
print(f"Verification A @ x: {A @ x}")  # [9. 8.]
\`\`\`

## Random Number Generation

NumPy provides extensive random number generation capabilities:

### Basic Random Generation

\`\`\`python
# Set seed for reproducibility
np.random.seed(42)

# Random floats from uniform distribution [0, 1)
rand_uniform = np.random.rand(5)
print(f"Uniform [0, 1): {rand_uniform}")

# Random floats from standard normal (mean=0, std=1)
rand_normal = np.random.randn(5)
print(f"Standard normal: {rand_normal}")

# Random integers
rand_int = np.random.randint(1, 100, size=10)  # [low, high), size
print(f"Random integers: {rand_int}")

# Random floats in a range
rand_range = np.random.uniform(10, 20, size=5)  # [low, high), size
print(f"Uniform [10, 20): {rand_range}")
\`\`\`

### Distributions

\`\`\`python
# Normal (Gaussian) distribution
normal = np.random.normal (loc=100, scale=15, size=1000)  # mean=100, std=15
print(f"Normal mean: {normal.mean():.2f}, std: {normal.std():.2f}")

# Binomial distribution
binomial = np.random.binomial (n=10, p=0.5, size=1000)  # 10 trials, p=0.5
print(f"Binomial mean: {binomial.mean():.2f}")

# Poisson distribution
poisson = np.random.poisson (lam=5, size=1000)  # lambda=5
print(f"Poisson mean: {poisson.mean():.2f}")

# Exponential distribution
exponential = np.random.exponential (scale=2, size=1000)  # scale=1/lambda
print(f"Exponential mean: {exponential.mean():.2f}")

# Beta distribution
beta = np.random.beta (a=2, b=5, size=1000)
print(f"Beta mean: {beta.mean():.2f}")
\`\`\`

### Sampling Operations

\`\`\`python
# Random choice from array
arr = np.array([10, 20, 30, 40, 50])
choice = np.random.choice (arr, size=3, replace=False)  # Without replacement
print(f"Random choice: {choice}")

# Shuffle array in place
arr_shuffle = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
np.random.shuffle (arr_shuffle)
print(f"Shuffled: {arr_shuffle}")

# Permutation (returns shuffled copy)
arr = np.array([1, 2, 3, 4, 5])
perm = np.random.permutation (arr)
print(f"Permutation: {perm}")
print(f"Original unchanged: {arr}")

# Random permutation of indices
indices = np.random.permutation(10)  # Shuffled [0, 1, 2, ..., 9]
print(f"Random indices: {indices}")
\`\`\`

### Modern Random Generation (numpy.random.Generator)

NumPy 1.17+ introduced a new random generation API:

\`\`\`python
# Create generator
rng = np.random.default_rng (seed=42)

# Generate random numbers
rand = rng.random(5)  # Uniform [0, 1)
print(f"Random: {rand}")

integers = rng.integers(0, 100, size=10)  # Note: 'integers' not 'randint'
print(f"Integers: {integers}")

normal = rng.normal(0, 1, size=5)
print(f"Normal: {normal}")

# Advantages of Generator:
# - Better statistical properties
# - Parallel random number generation
# - More consistent API
\`\`\`

## Practical Examples

### Example 1: Data Normalization

\`\`\`python
# Normalize data to [0, 1] range (Min-Max scaling)
data = np.random.randint(50, 150, size=100)
normalized = (data - data.min()) / (data.max() - data.min())
print(f"Original range: [{data.min()}, {data.max()}]")
print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

# Standardize data (Z-score normalization)
standardized = (data - data.mean()) / data.std()
print(f"Standardized mean: {standardized.mean():.3f}, std: {standardized.std():.3f}")
\`\`\`

### Example 2: Moving Average

\`\`\`python
# Calculate simple moving average
prices = np.random.uniform(90, 110, size=100)

window = 5
moving_avg = np.convolve (prices, np.ones (window)/window, mode='valid')
print(f"Original shape: {prices.shape}, MA shape: {moving_avg.shape}")
\`\`\`

### Example 3: Portfolio Returns

\`\`\`python
# Calculate portfolio returns
np.random.seed(42)
n_assets = 4
n_days = 252

# Asset returns
returns = np.random.normal(0.0005, 0.02, size=(n_days, n_assets))

# Portfolio weights
weights = np.array([0.25, 0.30, 0.20, 0.25])
assert weights.sum() == 1.0

# Portfolio returns (dot product for each day)
portfolio_returns = returns @ weights
cumulative_return = (1 + portfolio_returns).cumprod()[-1] - 1

print(f"Portfolio cumulative return: {cumulative_return:.2%}")
print(f"Portfolio Sharpe ratio: {portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252):.2f}")
\`\`\`

## Key Takeaways

1. **Vectorization**: Avoid Python loops, use NumPy operations on entire arrays
2. **Element-wise operations**: Arithmetic, math functions, comparisons work element-wise
3. **Aggregations**: sum(), mean(), std(), min(), max() with axis parameter
4. **Boolean indexing**: Powerful filtering with masks and conditions
5. **Array manipulation**: concatenate, split, stack for reshaping data
6. **Linear algebra**: Comprehensive matrix operations via np.linalg
7. **Random generation**: Multiple distributions and sampling operations
8. **Performance**: NumPy operations are 50-100x faster than Python loops

Master these operations and you'll be able to express complex data transformations concisely and efficiently!
`,
};
