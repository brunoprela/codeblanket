/**
 * Section: Performance Optimization
 * Module: Python for Data Science
 *
 * Covers vectorization, memory optimization, efficient pandas operations, and NumPy performance
 */

export const performanceOptimization = {
  id: 'performance-optimization',
  title: 'Performance Optimization',
  content: `
# Performance Optimization

## Introduction

Performance matters when working with large datasets. Understanding how NumPy and Pandas work internally enables you to write code that's orders of magnitude faster. The key principles are vectorization, memory efficiency, and choosing the right operations.

**Key Concepts:**
- Vectorization (avoid loops)
- Memory-efficient data types
- Efficient indexing and filtering
- Avoiding copies
- Using built-in optimized functions

\`\`\`python
import pandas as pd
import numpy as np
import time

# Timing utility
def time_it (func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__}: {end - start:.4f} seconds")
    return result
\`\`\`

## Vectorization: The Golden Rule

### Loop vs Vectorization

\`\`\`python
# Create large dataset
n = 1_000_000
data = np.random.randn (n)

# BAD: Python loop
def sum_with_loop (arr):
    total = 0
    for x in arr:
        total += x
    return total

# GOOD: NumPy vectorized
def sum_vectorized (arr):
    return np.sum (arr)

# Compare
start = time.time()
result1 = sum_with_loop (data)
time1 = time.time() - start
print(f"Loop: {time1:.4f} seconds")

start = time.time()
result2 = sum_vectorized (data)
time2 = time.time() - start
print(f"Vectorized: {time2:.4f} seconds")

print(f"Speedup: {time1/time2:.1f}x faster")
# Loop: 0.1234 seconds
# Vectorized: 0.0012 seconds
# Speedup: 100x faster!
\`\`\`

### Vectorizing Complex Operations

\`\`\`python
# Example: Calculate distance from origin
n = 1_000_000
x = np.random.randn (n)
y = np.random.randn (n)

# BAD: Loop
def distance_loop (x, y):
    distances = []
    for i in range (len (x)):
        distances.append (np.sqrt (x[i]**2 + y[i]**2))
    return np.array (distances)

# GOOD: Vectorized
def distance_vectorized (x, y):
    return np.sqrt (x**2 + y**2)

# Compare
start = time.time()
result1 = distance_loop (x, y)
time1 = time.time() - start
print(f"Loop: {time1:.4f} seconds")

start = time.time()
result2 = distance_vectorized (x, y)
time2 = time.time() - start
print(f"Vectorized: {time2:.4f} seconds")

print(f"Speedup: {time1/time2:.1f}x faster")
# Speedup: 50-100x faster

# Verify they match
print(f"Results match: {np.allclose (result1, result2)}")
\`\`\`

### When Vectorization Seems Impossible

\`\`\`python
# Example: Conditional logic
# Task: Apply different calculations based on value

data = np.random.randn(1_000_000)

# BAD: Loop with conditionals
def conditional_loop (arr):
    result = np.zeros_like (arr)
    for i, x in enumerate (arr):
        if x > 0:
            result[i] = x ** 2
        elif x < -1:
            result[i] = x ** 3
        else:
            result[i] = x
    return result

# GOOD: Use np.where or np.select
def conditional_vectorized (arr):
    # Method 1: Nested np.where
    result = np.where (arr > 0, arr ** 2,
                     np.where (arr < -1, arr ** 3, arr))
    return result

# Method 2: np.select (clearer for many conditions)
def conditional_vectorized_select (arr):
    conditions = [
        arr > 0,
        arr < -1
    ]
    choices = [
        arr ** 2,
        arr ** 3
    ]
    return np.select (conditions, choices, default=arr)

# Compare
start = time.time()
result1 = conditional_loop (data)
time1 = time.time() - start
print(f"Loop: {time1:.4f} seconds")

start = time.time()
result2 = conditional_vectorized (data)
time2 = time.time() - start
print(f"Vectorized (where): {time2:.4f} seconds")

start = time.time()
result3 = conditional_vectorized_select (data)
time3 = time.time() - start
print(f"Vectorized (select): {time3:.4f} seconds")

print(f"Speedup: {time1/time2:.1f}x faster")
\`\`\`

## Memory Optimization

### Efficient Data Types

\`\`\`python
# Example: Large DataFrame with inefficient types
n = 1_000_000
df = pd.DataFrame({
    'id': range (n),
    'category': np.random.choice(['A', 'B', 'C'], n),
    'value': np.random.rand (n),
    'flag': np.random.choice([True, False], n)
})

print("Original memory usage:")
print(df.memory_usage (deep=True))
print(f"Total: {df.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

# Optimize data types
df_optimized = df.copy()

# Convert category to categorical (huge savings)
df_optimized['category'] = df_optimized['category'].astype('category')

# Downcast numeric types
df_optimized['id'] = pd.to_numeric (df_optimized['id'], downcast='integer')
df_optimized['value'] = pd.to_numeric (df_optimized['value'], downcast='float')

print("\\nOptimized memory usage:")
print(df_optimized.memory_usage (deep=True))
print(f"Total: {df_optimized.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

reduction = (1 - df_optimized.memory_usage (deep=True).sum() /
             df.memory_usage (deep=True).sum()) * 100
print(f"\\nMemory reduction: {reduction:.1f}%")
\`\`\`

### Choosing Right Data Types

\`\`\`python
# Integer types
print("Integer types:")
print(f"int8: {np.iinfo (np.int8).min} to {np.iinfo (np.int8).max}")
print(f"int16: {np.iinfo (np.int16).min} to {np.iinfo (np.int16).max}")
print(f"int32: {np.iinfo (np.int32).min} to {np.iinfo (np.int32).max}")
print(f"int64: {np.iinfo (np.int64).min} to {np.iinfo (np.int64).max}")

# Float types
print("\\nFloat types:")
print(f"float16: ~{np.finfo (np.float16).min:.2e} to {np.finfo (np.float16).max:.2e}")
print(f"float32: ~{np.finfo (np.float32).min:.2e} to {np.finfo (np.float32).max:.2e}")
print(f"float64: ~{np.finfo (np.float64).min:.2e} to {np.finfo (np.float64).max:.2e}")

# Example: Age data (0-120)
ages = np.random.randint(0, 120, 1_000_000)

# Default: int64 (8 bytes per value)
ages_int64 = ages.astype (np.int64)
print(f"\\nint64: {ages_int64.nbytes / 1024**2:.2f} MB")

# Optimized: int8 (1 byte per value)
ages_int8 = ages.astype (np.int8)
print(f"int8: {ages_int8.nbytes / 1024**2:.2f} MB")
print(f"Memory saved: {(1 - ages_int8.nbytes / ages_int64.nbytes) * 100:.1f}%")
\`\`\`

### Categorical Data

\`\`\`python
# Example: Product categories
n = 1_000_000
products = np.random.choice(['Widget', 'Gadget', 'Gizmo', 'Tool'], n)

# As object dtype (stores strings)
df_object = pd.DataFrame({'product': products})
print(f"Object dtype: {df_object.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

# As categorical
df_categorical = pd.DataFrame({'product': pd.Categorical (products)})
print(f"Categorical: {df_categorical.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

reduction = (1 - df_categorical.memory_usage (deep=True).sum() /
             df_object.memory_usage (deep=True).sum()) * 100
print(f"Memory reduction: {reduction:.1f}%")

# Categorical is also faster for operations
print("\\nSpeed comparison:")

start = time.time()
_ = df_object['product'] == 'Widget'
time_object = time.time() - start

start = time.time()
_ = df_categorical['product'] == 'Widget'
time_categorical = time.time() - start

print(f"Object: {time_object:.4f} seconds")
print(f"Categorical: {time_categorical:.4f} seconds")
print(f"Speedup: {time_object/time_categorical:.1f}x faster")
\`\`\`

## Efficient Pandas Operations

### Avoid Iteration

\`\`\`python
# Create sample DataFrame
df = pd.DataFrame({
    'A': np.random.randn(100_000),
    'B': np.random.randn(100_000),
    'C': np.random.randn(100_000)
})

# BAD: iterrows (very slow)
def sum_with_iterrows (df):
    total = 0
    for index, row in df.iterrows():
        total += row['A'] + row['B'] + row['C']
    return total

# BETTER: apply (still uses Python loop internally)
def sum_with_apply (df):
    return df.apply (lambda row: row['A'] + row['B'] + row['C'], axis=1).sum()

# BEST: Vectorized operations
def sum_vectorized (df):
    return (df['A'] + df['B'] + df['C']).sum()

# Compare
times = {}

start = time.time()
result1 = sum_with_iterrows (df)
times['iterrows'] = time.time() - start

start = time.time()
result2 = sum_with_apply (df)
times['apply'] = time.time() - start

start = time.time()
result3 = sum_vectorized (df)
times['vectorized'] = time.time() - start

print("Timing results:")
for method, t in times.items():
    print(f"{method}: {t:.4f} seconds")

print(f"\\nVectorized is {times['iterrows']/times['vectorized']:.1f}x faster than iterrows")
print(f"Vectorized is {times['apply']/times['vectorized']:.1f}x faster than apply")
\`\`\`

### Efficient Filtering

\`\`\`python
# Create sample data
df = pd.DataFrame({
    'A': np.random.randn(1_000_000),
    'B': np.random.choice(['X', 'Y', 'Z'], 1_000_000),
    'C': np.random.randint(0, 100, 1_000_000)
})

# BAD: Multiple separate filters (creates intermediate DataFrames)
def filter_separate (df):
    filtered = df[df['A'] > 0]
    filtered = filtered[filtered['B'] == 'X']
    filtered = filtered[filtered['C'] < 50]
    return filtered

# GOOD: Combined boolean indexing
def filter_combined (df):
    mask = (df['A'] > 0) & (df['B'] == 'X') & (df['C'] < 50)
    return df[mask]

# BETTER: query() for complex conditions (more readable)
def filter_query (df):
    return df.query('A > 0 and B == "X" and C < 50')

# Compare
start = time.time()
result1 = filter_separate (df)
time1 = time.time() - start

start = time.time()
result2 = filter_combined (df)
time2 = time.time() - start

start = time.time()
result3 = filter_query (df)
time3 = time.time() - start

print(f"Separate: {time1:.4f} seconds")
print(f"Combined: {time2:.4f} seconds")
print(f"Query: {time3:.4f} seconds")
\`\`\`

### Avoid Chained Indexing

\`\`\`python
# Create sample DataFrame
df = pd.DataFrame({
    'A': range(1000),
    'B': range(1000)
})

# BAD: Chained indexing (ambiguous, can cause SettingWithCopyWarning)
# df[df['A'] > 500]['B'] = 0  # Don't do this!

# GOOD: Use loc
df.loc[df['A'] > 500, 'B'] = 0

# GOOD: Single indexing operation
mask = df['A'] > 500
df.loc[mask, 'B'] = 0
\`\`\`

## Views vs Copies

\`\`\`python
# Understanding views and copies
arr = np.array([1, 2, 3, 4, 5])

# Slicing creates a view (no copy)
view = arr[1:4]
view[0] = 999
print(f"Original array: {arr}")  # [1, 999, 3, 4, 5] - modified!

# To create a copy
arr = np.array([1, 2, 3, 4, 5])
copy = arr[1:4].copy()
copy[0] = 999
print(f"Original array: {arr}")  # [1, 2, 3, 4, 5] - unchanged

# Check if array is a view
print(f"Is view? {view.base is not None}")
print(f"Is copy? {copy.base is None}")

# In pandas
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Some operations return views, some return copies
# To be safe, use .copy() when you want a copy
df_copy = df.copy()
df_copy.loc[0, 'A'] = 999
print(f"Original: {df['A'].iloc[0]}")  # Still 1
\`\`\`

## Built-in Optimizations

### Use Built-in Functions

\`\`\`python
# Create large dataset
arr = np.random.randn(1_000_000)

# BAD: Manual calculation
def variance_manual (arr):
    mean = sum (arr) / len (arr)
    squared_diffs = [(x - mean)**2 for x in arr]
    return sum (squared_diffs) / len (arr)

# GOOD: NumPy built-in (highly optimized)
def variance_numpy (arr):
    return np.var (arr)

# Compare
start = time.time()
result1 = variance_manual (arr)
time1 = time.time() - start

start = time.time()
result2 = variance_numpy (arr)
time2 = time.time() - start

print(f"Manual: {time1:.4f} seconds")
print(f"NumPy: {time2:.4f} seconds")
print(f"Speedup: {time1/time2:.1f}x faster")
\`\`\`

### Efficient String Operations

\`\`\`python
# Create DataFrame with strings
df = pd.DataFrame({
    'text': ['hello world', 'foo bar', 'test string'] * 100_000
})

# BAD: apply with Python string methods
def lowercase_apply (df):
    return df['text'].apply (lambda x: x.lower())

# GOOD: Vectorized str accessor
def lowercase_vectorized (df):
    return df['text'].str.lower()

# Compare
start = time.time()
result1 = lowercase_apply (df)
time1 = time.time() - start

start = time.time()
result2 = lowercase_vectorized (df)
time2 = time.time() - start

print(f"Apply: {time1:.4f} seconds")
print(f"Vectorized: {time2:.4f} seconds")
print(f"Speedup: {time1/time2:.1f}x faster")
\`\`\`

## NumPy Broadcasting

\`\`\`python
# Broadcasting eliminates need for loops

# Example: Normalize each column
data = np.random.randn(1000, 5)

# BAD: Loop over columns
def normalize_loop (arr):
    result = np.zeros_like (arr)
    for i in range (arr.shape[1]):
        mean = arr[:, i].mean()
        std = arr[:, i].std()
        result[:, i] = (arr[:, i] - mean) / std
    return result

# GOOD: Broadcasting
def normalize_broadcast (arr):
    means = arr.mean (axis=0)  # Shape: (5,)
    stds = arr.std (axis=0)     # Shape: (5,)
    return (arr - means) / stds  # Broadcasting!

# Compare
start = time.time()
result1 = normalize_loop (data)
time1 = time.time() - start

start = time.time()
result2 = normalize_broadcast (data)
time2 = time.time() - start

print(f"Loop: {time1:.4f} seconds")
print(f"Broadcast: {time2:.4f} seconds")
print(f"Speedup: {time1/time2:.1f}x faster")

print(f"\\nResults match: {np.allclose (result1, result2)}")

# Broadcasting rules
a = np.array([1, 2, 3])  # Shape: (3,)
b = np.array([[1], [2], [3], [4]])  # Shape: (4, 1)

result = a + b  # Broadcasting to (4, 3)
print(f"\\nBroadcasting example:")
print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print(f"result shape: {result.shape}")
print(result)
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]]
\`\`\`

## Practical Examples

### Example 1: Optimizing Data Pipeline

\`\`\`python
# Simulate large dataset
n = 1_000_000
df = pd.DataFrame({
    'user_id': np.random.randint(1, 10000, n),
    'product_id': np.random.randint(1, 1000, n),
    'price': np.random.uniform(10, 1000, n),
    'quantity': np.random.randint(1, 10, n),
    'date': pd.date_range('2024-01-01', periods=n, freq='1min'),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n)
})

print(f"Original size: {df.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

# Optimization pipeline
def optimize_dataframe (df):
    df_opt = df.copy()

    # 1. Optimize numeric types
    df_opt['user_id'] = pd.to_numeric (df_opt['user_id'], downcast='integer')
    df_opt['product_id'] = pd.to_numeric (df_opt['product_id'], downcast='integer')
    df_opt['price'] = pd.to_numeric (df_opt['price'], downcast='float')
    df_opt['quantity'] = pd.to_numeric (df_opt['quantity'], downcast='integer')

    # 2. Convert to categorical
    df_opt['category'] = df_opt['category'].astype('category')

    # 3. No need to store full timestamp if only date matters
    # (But keeping for demonstration)

    return df_opt

df_optimized = optimize_dataframe (df)
print(f"Optimized size: {df_optimized.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

reduction = (1 - df_optimized.memory_usage (deep=True).sum() /
             df.memory_usage (deep=True).sum()) * 100
print(f"Memory reduction: {reduction:.1f}%")

# Performance comparison
print("\\nPerformance comparison:")

# Task: Calculate revenue by category
start = time.time()
revenue_original = df.groupby('category')['price'].sum()
time_original = time.time() - start

start = time.time()
revenue_optimized = df_optimized.groupby('category')['price'].sum()
time_optimized = time.time() - start

print(f"Original: {time_original:.4f} seconds")
print(f"Optimized: {time_optimized:.4f} seconds")
print(f"Speedup: {time_original/time_optimized:.1f}x faster")
\`\`\`

### Example 2: Vectorizing Complex Calculation

\`\`\`python
# Task: Calculate moving Sharpe ratio

# Generate returns data
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
returns = pd.Series (np.random.randn(1000) * 0.02, index=dates)

# BAD: Loop-based calculation
def sharpe_loop (returns, window=252):
    sharpe_ratios = []
    for i in range (len (returns)):
        if i < window:
            sharpe_ratios.append (np.nan)
        else:
            window_returns = returns.iloc[i-window:i]
            mean_return = window_returns.mean()
            std_return = window_returns.std()
            sharpe = np.sqrt(252) * mean_return / std_return if std_return > 0 else 0
            sharpe_ratios.append (sharpe)
    return pd.Series (sharpe_ratios, index=returns.index)

# GOOD: Vectorized with rolling
def sharpe_vectorized (returns, window=252):
    rolling_mean = returns.rolling (window).mean()
    rolling_std = returns.rolling (window).std()
    sharpe = np.sqrt(252) * rolling_mean / rolling_std
    return sharpe

# Compare
start = time.time()
result1 = sharpe_loop (returns)
time1 = time.time() - start

start = time.time()
result2 = sharpe_vectorized (returns)
time2 = time.time() - start

print(f"Loop: {time1:.4f} seconds")
print(f"Vectorized: {time2:.4f} seconds")
print(f"Speedup: {time1/time2:.1f}x faster")

print(f"\\nResults match: {np.allclose (result1.dropna(), result2.dropna())}")
\`\`\`

### Example 3: Efficient Merging

\`\`\`python
# Create large DataFrames for merging
n_transactions = 1_000_000
n_users = 10_000

transactions = pd.DataFrame({
    'transaction_id': range (n_transactions),
    'user_id': np.random.randint(1, n_users, n_transactions),
    'amount': np.random.uniform(10, 1000, n_transactions)
})

users = pd.DataFrame({
    'user_id': range(1, n_users + 1),
    'user_name': [f'User_{i}' for i in range(1, n_users + 1)],
    'country': np.random.choice(['US', 'UK', 'CA', 'AU'], n_users)
})

# Optimize types before merge
transactions['user_id'] = transactions['user_id'].astype (np.int32)
users['user_id'] = users['user_id'].astype (np.int32)
users['country'] = users['country'].astype('category')

# Merge
start = time.time()
merged = pd.merge (transactions, users, on='user_id', how='left')
merge_time = time.time() - start
print(f"Merge time: {merge_time:.4f} seconds")
print(f"Merged DataFrame: {merged.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

# Using categorical saves memory in merged result too
print(f"\\nMerged country column: {merged['country'].memory_usage (deep=True) / 1024**2:.2f} MB")
\`\`\`

## Performance Profiling

\`\`\`python
# Profile your code to find bottlenecks
import cProfile
import pstats

def example_function():
    # Some data processing
    df = pd.DataFrame (np.random.randn(10000, 5))
    result = df.apply (lambda x: x.sum(), axis=1)
    return result

# Profile it
profiler = cProfile.Profile()
profiler.enable()
result = example_function()
profiler.disable()

# Print stats
stats = pstats.Stats (profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions by time
\`\`\`

## Key Takeaways

1. **Vectorize Everything**: Avoid Python loops, use NumPy/Pandas operations
2. **Right Data Types**: Use smallest types that fit your data
3. **Categorical for Strings**: Huge memory and speed gains
4. **Avoid Iteration**: iterrows/apply are slow, vectorize instead
5. **Built-in Functions**: Use optimized library functions
6. **Broadcasting**: Eliminate loops with NumPy broadcasting
7. **Views vs Copies**: Understand to avoid unnecessary copying
8. **Profile First**: Measure before optimizing

**Performance Hierarchy (Slowest to Fastest):**1. ❌ iterrows() - Very slow
2. ❌ apply() with Python function - Slow
3. ✅ apply() with NumPy function - Better
4. ✅ Vectorized operations - Fast
5. ✅✅ NumPy built-in functions - Very fast

Remember: Premature optimization is the root of all evil. Profile first, optimize bottlenecks!
`,
};
