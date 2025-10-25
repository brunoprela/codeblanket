import { QuizQuestion } from '../../../types';

export const numpyoperationsQuiz: QuizQuestion[] = [
  {
    id: 'numpy-operations-dq-1',
    question:
      'Explain the concept of "vectorization" in NumPy and why it provides such significant performance improvements over Python loops. Provide a concrete example demonstrating the performance difference.',
    sampleAnswer: `Vectorization is the process of applying operations to entire arrays at once, rather than using explicit Python loops. This is NumPy\'s superpower and the foundation of efficient numerical computing in Python.

**Why Vectorization is Fast:**

1. **Compiled C Code**: NumPy operations are implemented in highly optimized C/Fortran, not interpreted Python
2. **Avoid Python Overhead**: Each Python loop iteration has significant overhead (type checking, reference counting, etc.)
3. **SIMD Instructions**: Modern CPUs can perform the same operation on multiple data points simultaneously
4. **Cache Efficiency**: Contiguous memory access patterns maximize CPU cache hits
5. **No GIL Contention**: NumPy releases Python's Global Interpreter Lock during operations

**Performance Comparison:**

\`\`\`python
import numpy as np
import time

# Create large array
n = 1_000_000
arr = np.arange (n, dtype=float)

# Method 1: Python loop
start = time.time()
result_loop = []
for x in arr:
    result_loop.append (x ** 2 + 2 * x + 1)
result_loop = np.array (result_loop)
time_loop = time.time() - start

# Method 2: NumPy vectorization
start = time.time()
result_vectorized = arr ** 2 + 2 * arr + 1
time_vectorized = time.time() - start

print(f"Python loop: {time_loop:.4f} seconds")
print(f"NumPy vectorized: {time_vectorized:.4f} seconds")
print(f"Speedup: {time_loop / time_vectorized:.1f}x faster")

# Typical results:
# Python loop: 0.4821 seconds
# NumPy vectorized: 0.0045 seconds
# Speedup: 107.1x faster
\`\`\`

**Why Such a Large Difference?**

The Python loop version:
- Interprets each line of Python code
- Checks types for every operation
- Manages Python objects for every number
- Executes ~3 million Python bytecode instructions

The vectorized version:
- Single call to compiled C code
- Types known at compile time
- Operations on raw memory blocks
- CPU can use SIMD to process multiple numbers per instruction

**Real-World Impact:**

Consider a neural network forward pass with 1 million parameters:
- Python loops: ~5 seconds per batch
- Vectorized NumPy: ~50 milliseconds per batch
- **100x speedup** = train model in 1 hour instead of 4 days

**Best Practices:**

1. **Always vectorize**: If you write a Python loop over array elements, you're likely doing it wrong
2. **Use NumPy operations**: +, -, *, /, @, np.sum(), np.mean(), etc.
3. **Broadcasting**: Leverage NumPy\'s ability to work with arrays of different shapes
4. **Built-in functions**: np.where(), np.clip(), np.maximum() instead of if-else loops

**When Loops Are Acceptable:**

- Iterating over small number of items (< 100)
- Operations that can't be vectorized (complex conditional logic)
- Readability is more important than speed for prototyping

Vectorization isn't just a performance optimization—it's the difference between feasible and infeasible for large-scale data science.`,
    keyPoints: [
      'Vectorization applies operations to entire arrays using compiled C code',
      'Eliminates slow Python loops - 50-1000x speedup for numerical operations',
      'Broadcasting enables operations on different shapes without explicit loops',
      'UFuncs (universal functions) are vectorized C implementations',
      'Profile with %%timeit to measure actual speedup in your use case',
    ],
  },
  {
    id: 'numpy-operations-dq-2',
    question:
      'Explain the difference between axis=0 and axis=1 in NumPy aggregation operations. Why does this often confuse beginners, and what mental model helps understand it?',
    sampleAnswer: `Understanding axis parameters in NumPy is crucial but notoriously confusing. The key insight is that the axis parameter specifies *which axis to collapse* during aggregation.

**The Confusion:**

For a 2D array (matrix), axis=0 and axis=1 seem backwards to many beginners:
- axis=0 operates *across rows* (down columns)
- axis=1 operates *across columns* (along rows)

**Example to Illustrate:**

\`\`\`python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# axis=0: collapse axis 0 (rows), leaving columns
print(arr.sum (axis=0))  # [12, 15, 18]
# Sums: column 0: 1+4+7=12, column 1: 2+5+8=15, column 2: 3+6+9=18

# axis=1: collapse axis 1 (columns), leaving rows
print(arr.sum (axis=1))  # [6, 15, 24]
# Sums: row 0: 1+2+3=6, row 1: 4+5+6=15, row 2: 7+8+9=24

# No axis: collapse everything
print(arr.sum())  # 45
\`\`\`

**Mental Models:**

**Model 1: "Collapse the Axis"**
- axis=0 means "collapse axis 0" → removes the row dimension
- Result shape: (3, 3) → (3,)
- You're aggregating along the 0th dimension

**Model 2: "The Axis You Sum Over"**
- axis=0 means "sum over all values in axis 0" (all rows in each column)
- axis=1 means "sum over all values in axis 1" (all columns in each row)

**Model 3: "Direction of Movement"** (my preferred)
- axis=0: imagine moving DOWN (across rows) → produces column results
- axis=1: imagine moving RIGHT (across columns) → produces row results

**Model 4: "What Remains"** (most intuitive for many)
- axis=0: row dimension disappears, column dimension remains → column statistics
- axis=1: column dimension disappears, row dimension remains → row statistics

**Higher Dimensions:**

For 3D arrays (e.g., images: height × width × channels):

\`\`\`python
image_batch = np.random.rand(32, 224, 224, 3)  # 32 images, 224×224, RGB
# Shape: (batch, height, width, channels)

# axis=0: aggregate across batch → (224, 224, 3)
mean_image = image_batch.mean (axis=0)  # Average image across batch

# axis=3: aggregate across channels → (32, 224, 224)
grayscale = image_batch.mean (axis=3)  # Convert to grayscale

# axis=(1,2): aggregate across height and width → (32, 3)
mean_color_per_image = image_batch.mean (axis=(1, 2))  # Mean RGB per image
\`\`\`

**Practical Applications:**

1. **Feature-wise statistics** (axis=0):
\`\`\`python
X = np.random.randn(1000, 20)  # 1000 samples, 20 features
feature_means = X.mean (axis=0)  # (20,) - mean of each feature
feature_stds = X.std (axis=0)    # (20,) - std of each feature
X_normalized = (X - feature_means) / feature_stds  # Standardization
\`\`\`

2. **Sample-wise statistics** (axis=1):
\`\`\`python
X = np.random.randn(1000, 20)  # 1000 samples, 20 features
sample_means = X.mean (axis=1)  # (1000,) - mean of each sample
\`\`\`

3. **No axis** (aggregate all):
\`\`\`python
overall_mean = X.mean()  # Scalar - mean of entire dataset
\`\`\`

**Why It Matters:**

Incorrect axis usage is a common source of bugs:
- Wrong axis → wrong statistics → poor model performance
- Shape mismatches → cryptic error messages
- Silent errors where code runs but produces nonsense

**Pro Tip:**

Always check shapes:
\`\`\`python
print(f"Original shape: {arr.shape}")
print(f"After axis=0: {arr.sum (axis=0).shape}")
print(f"After axis=1: {arr.sum (axis=1).shape}")
\`\`\`

Once you internalize "axis specifies which dimension to collapse," NumPy operations become much more intuitive!`,
    keyPoints: [
      'Broadcasting aligns arrays of different shapes following specific rules',
      'Dimensions compared from right to left - must be equal or 1',
      'Smaller array stretched (virtually) to match larger shape',
      'Enables efficient operations without copying data',
      'Common use: subtract mean (broadcasting scalar), normalize (broadcasting vector)',
    ],
  },
  {
    id: 'numpy-operations-dq-3',
    question:
      'Compare and contrast boolean indexing with np.where() for conditional operations in NumPy. When would you use each approach, and what are the performance implications?',
    sampleAnswer: `Boolean indexing and np.where() are both powerful tools for conditional operations, but they serve different purposes and have different performance characteristics.

**Boolean Indexing:**

Boolean indexing uses a boolean array (mask) to select elements:

\`\`\`python
arr = np.array([1, 5, 3, 8, 2, 9, 7, 4, 6, 10])

# Create mask
mask = arr > 5
filtered = arr[mask]  # [8, 9, 7, 6, 10]

# Modify selected elements
arr[mask] = 999  # Replace all values > 5 with 999
\`\`\`

**Characteristics:**
- Returns filtered array (potentially different size)
- Can be used for assignment
- Unknown output size until runtime
- Creates a copy (unless used for in-place modification)

**np.where():**

np.where() has two modes:

\`\`\`python
# Mode 1: Three arguments (ternary operator)
result = np.where (arr > 5, 100, -100)  # Same size as input
# Returns 100 where True, -100 where False

# Mode 2: One argument (index finding)
indices = np.where (arr > 5)  # Returns tuple of indices
# indices[0] contains the positions
\`\`\`

**Characteristics:**
- Always returns same size as input (mode 1) or indices (mode 2)
- Cannot modify original array directly
- Predictable output size
- Allows element-wise conditionals

**When to Use Each:**

**Use Boolean Indexing when:**

1. **Filtering data:**
\`\`\`python
# Remove outliers
clean_data = data[np.abs (data - data.mean()) < 2 * data.std()]

# Select specific samples
high_returns = stock_returns[stock_returns > 0.02]
\`\`\`

2. **In-place modification:**
\`\`\`python
# Clip values
arr[arr > 100] = 100
arr[arr < 0] = 0

# Replace outliers with median
median = np.median (data)
data[np.abs (data - data.mean()) > 3 * data.std()] = median
\`\`\`

3. **Multiple conditions:**
\`\`\`python
# Complex filtering
mask = (age > 18) & (income > 50000) & (credit_score > 700)
approved_applicants = applicants[mask]
\`\`\`

**Use np.where() when:**

1. **Maintaining array shape:**
\`\`\`python
# Replace negatives with zero, keep positives
arr = np.where (arr < 0, 0, arr)  # Same size as input

# Categorize values
labels = np.where (scores > 90, "A",
         np.where (scores > 80, "B",
         np.where (scores > 70, "C", "F")))
\`\`\`

2. **Element-wise conditionals:**
\`\`\`python
# Apply different transformations
result = np.where (arr % 2 == 0, arr * 2, arr * 3)

# Piecewise functions
y = np.where (x < 0, 0, np.where (x > 1, 1, x))  # Clip to [0, 1]
\`\`\`

3. **Finding indices:**
\`\`\`python
# Locate outliers
outlier_indices = np.where (arr > threshold)[0]

# Find peaks
peaks = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1
\`\`\`

**Performance Comparison:**

\`\`\`python
import numpy as np
import time

n = 10_000_000
arr = np.random.randn (n)

# Boolean indexing (creates copy)
start = time.time()
result1 = arr[arr > 0]
time1 = time.time() - start

# np.where (maintains shape)
start = time.time()
result2 = np.where (arr > 0, arr, 0)
time2 = time.time() - start

# In-place modification (fastest)
start = time.time()
arr_copy = arr.copy()
arr_copy[arr_copy < 0] = 0
time3 = time.time() - start

print(f"Boolean indexing: {time1:.4f}s, size: {result1.shape}")
print(f"np.where: {time2:.4f}s, size: {result2.shape}")
print(f"In-place: {time3:.4f}s, size: {arr_copy.shape}")
\`\`\`

**Typical Results:**
- Boolean indexing: ~0.15s, size: (5000000,) - variable size
- np.where: ~0.25s, size: (10000000,) - maintains size
- In-place: ~0.10s, size: (10000000,) - fastest

**Performance Insights:**

1. **Boolean indexing** is fast but creates a copy of selected elements
2. **np.where()** is slightly slower due to evaluating both branches
3. **In-place modification** is fastest (no new array allocation)

**Alternatives:**

For very simple conditions, consider:

\`\`\`python
# Instead of np.where for clipping
clipped = np.clip (arr, min_val, max_val)  # Faster

# Instead of boolean indexing for simple replacement
arr[arr < 0] = 0  # In-place

# For max/min comparisons
result = np.maximum (arr, 0)  # Element-wise max with 0
result = np.minimum (arr, 100)  # Element-wise min with 100
\`\`\`

**Best Practices:**

1. **Boolean indexing**: When you need to filter/select a subset
2. **np.where()**: When you need to keep array shape
3. **In-place**: When you can modify the original array
4. **Check shape**: Always verify output shape matches expectations

**Common Mistake:**

\`\`\`python
# WRONG: Boolean indexing changes size!
arr[arr > 0] = arr[arr > 0] * 2  # Size mismatch error!

# CORRECT: In-place modification
mask = arr > 0
arr[mask] *= 2

# OR: np.where maintains shape
arr = np.where (arr > 0, arr * 2, arr)
\`\`\`

Understanding these tools and their trade-offs is essential for writing efficient, correct NumPy code!`,
    keyPoints: [
      'Boolean indexing uses True/False array to filter elements',
      'Fancy indexing uses integer arrays/lists to select specific indices',
      'Boolean indexing always creates copy, returns 1D array',
      'Fancy indexing creates copy, preserves shape if indices are same shape',
      'Use boolean for conditional filtering, fancy for specific element selection',
    ],
  },
];
