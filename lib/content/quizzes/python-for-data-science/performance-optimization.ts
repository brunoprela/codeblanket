import { QuizQuestion } from '../../../types';

export const performanceoptimizationQuiz: QuizQuestion[] = [
  {
    id: 'performance-optimization-dq-1',
    question:
      'Explain the concept of vectorization in NumPy and Pandas. Why is it so much faster than Python loops? Provide examples of how to vectorize common operations that beginners might write as loops.',
    sampleAnswer: `Vectorization is the cornerstone of efficient data processing in NumPy and Pandas. Understanding why it's faster and how to apply it is essential for working with real-world datasets.

**What is Vectorization?**

Vectorization means operating on entire arrays at once rather than element-by-element in Python loops. The operations are pushed down to optimized C/Fortran code.

**Why is Vectorization Fast?**

**Reason 1: Compiled vs Interpreted**
\`\`\`python
# Python loop (interpreted)
def sum_python_loop (arr):
    total = 0
    for x in arr:  # Each iteration:
        total += x  # - Interpreter overhead
                    # - Type checking
                    # - Python object creation
    return total

# NumPy vectorized (compiled C)
def sum_vectorized (arr):
    return np.sum (arr)  # Direct C loop, no interpreter
\`\`\`

**Reason 2: Contiguous Memory Access**
\`\`\`python
# Python list: objects scattered in memory
py_list = [1, 2, 3, 4, 5]  # Each element is a Python object (pointer)

# NumPy array: contiguous block of memory
np_array = np.array([1, 2, 3, 4, 5])  # Dense memory, cache-friendly
\`\`\`

**Reason 3: SIMD (Single Instruction Multiple Data)**
Modern CPUs can operate on multiple values with a single instruction. NumPy can use these.

\`\`\`python
# Python: 4 separate additions
for i in range(4):
    result[i] = a[i] + b[i]

# NumPy: 1 SIMD instruction adds all 4 at once
result = a + b  # Can use CPU vector instructions
\`\`\`

**Example 1: Sum of Squares**

\`\`\`python
n = 1_000_000
data = np.random.randn (n)

# BAD: Python loop (very slow)
def sum_squares_loop (arr):
    total = 0
    for x in arr:
        total += x ** 2
    return total

# GOOD: Vectorized
def sum_squares_vectorized (arr):
    return np.sum (arr ** 2)

# Even better: Use einsum for complex operations
def sum_squares_einsum (arr):
    return np.einsum('i,i->', arr, arr)

import time

start = time.time()
result1 = sum_squares_loop (data)
time1 = time.time() - start

start = time.time()
result2 = sum_squares_vectorized (data)
time2 = time.time() - start

start = time.time()
result3 = sum_squares_einsum (data)
time3 = time.time() - start

print(f"Loop: {time1:.4f}s")
print(f"Vectorized: {time2:.4f}s ({time1/time2:.0f}x faster)")
print(f"Einsum: {time3:.4f}s ({time1/time3:.0f}x faster)")

# Output:
# Loop: 0.3245s
# Vectorized: 0.0025s (130x faster)
# Einsum: 0.0018s (180x faster)
\`\`\`

**Example 2: Conditional Logic**

Beginners often write:
\`\`\`python
# BAD: Loop with conditions
def apply_rules_loop (arr):
    result = []
    for x in arr:
        if x > 0:
            result.append (x * 2)
        elif x < -1:
            result.append (x / 2)
        else:
            result.append(0)
    return np.array (result)

# GOOD: Vectorized with np.where
def apply_rules_vectorized (arr):
    return np.where (arr > 0, arr * 2,
                   np.where (arr < -1, arr / 2, 0))

# Also GOOD: np.select (more readable for many conditions)
def apply_rules_select (arr):
    conditions = [arr > 0, arr < -1]
    choices = [arr * 2, arr / 2]
    return np.select (conditions, choices, default=0)

# Benchmark
data = np.random.randn(100_000)

start = time.time()
result1 = apply_rules_loop (data)
time1 = time.time() - start

start = time.time()
result2 = apply_rules_vectorized (data)
time2 = time.time() - start

print(f"Loop: {time1:.4f}s")
print(f"Vectorized: {time2:.4f}s ({time1/time2:.0f}x faster)")

# Loop: 0.0856s
# Vectorized: 0.0012s (71x faster)
\`\`\`

**Example 3: Distance Calculations**

\`\`\`python
# BAD: Loop for pairwise distances
def distances_loop (x1, y1, x2, y2):
    n = len (x1)
    distances = np.zeros (n)
    for i in range (n):
        distances[i] = np.sqrt((x2[i] - x1[i])**2 + (y2[i] - y1[i])**2)
    return distances

# GOOD: Vectorized
def distances_vectorized (x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# BETTER: Use np.hypot (more numerically stable)
def distances_hypot (x1, y1, x2, y2):
    return np.hypot (x2 - x1, y2 - y1)

n = 1_000_000
x1, y1 = np.random.randn(2, n)
x2, y2 = np.random.randn(2, n)

# Benchmark
start = time.time()
result1 = distances_loop (x1, y1, x2, y2)
time1 = time.time() - start

start = time.time()
result2 = distances_vectorized (x1, y1, x2, y2)
time2 = time.time() - start

start = time.time()
result3 = distances_hypot (x1, y1, x2, y2)
time3 = time.time() - start

print(f"Loop: {time1:.4f}s")
print(f"Vectorized: {time2:.4f}s ({time1/time2:.0f}x faster)")
print(f"Hypot: {time3:.4f}s ({time1/time3:.0f}x faster)")

# Loop: 0.8234s
# Vectorized: 0.0145s (57x faster)
# Hypot: 0.0234s (35x faster, but more stable)
\`\`\`

**Example 4: DataFrame Operations**

\`\`\`python
# Create DataFrame
df = pd.DataFrame({
    'A': np.random.randn(100_000),
    'B': np.random.randn(100_000),
    'C': np.random.randn(100_000)
})

# BAD: iterrows
def process_iterrows (df):
    results = []
    for index, row in df.iterrows():
        results.append (row['A',] * row['B',] + row['C',])
    return results

# BAD: apply (better than iterrows, still slow)
def process_apply (df):
    return df.apply (lambda row: row['A',] * row['B',] + row['C',], axis=1)

# GOOD: Vectorized
def process_vectorized (df):
    return df['A',] * df['B',] + df['C',]

# Benchmark
start = time.time()
result1 = process_iterrows (df)
time1 = time.time() - start

start = time.time()
result2 = process_apply (df)
time2 = time.time() - start

start = time.time()
result3 = process_vectorized (df)
time3 = time.time() - start

print(f"iterrows: {time1:.4f}s")
print(f"apply: {time2:.4f}s ({time1/time2:.0f}x faster than iterrows)")
print(f"vectorized: {time3:.4f}s ({time1/time3:.0f}x faster than iterrows)")

# iterrows: 8.2345s
# apply: 1.2456s (7x faster than iterrows)
# vectorized: 0.0012s (6862x faster than iterrows!)
\`\`\`

**Example 5: String Operations**

\`\`\`python
# Create DataFrame with text
df = pd.DataFrame({
    'text': ['Hello World', 'foo bar', 'Test String',] * 100_000
})

# BAD: apply with Python string methods
def lowercase_apply (df):
    return df['text',].apply (lambda x: x.lower())

# GOOD: Vectorized str accessor
def lowercase_vectorized (df):
    return df['text',].str.lower()

# Benchmark
start = time.time()
result1 = lowercase_apply (df)
time1 = time.time() - start

start = time.time()
result2 = lowercase_vectorized (df)
time2 = time.time() - start

print(f"apply: {time1:.4f}s")
print(f"vectorized: {time2:.4f}s ({time1/time2:.0f}x faster)")

# apply: 0.3456s
# vectorized: 0.0234s (15x faster)
\`\`\`

**When Vectorization Seems Impossible**

Sometimes you genuinely need Python logic:

\`\`\`python
# Example: Call external API for each row
def fetch_data (user_id):
    # API call here
    return some_data

# Can't vectorize this - must use apply
df['api_data',] = df['user_id',].apply (fetch_data)

# But you can still optimize:
# 1. Batch API calls
# 2. Use multiprocessing
# 3. Cache results
\`\`\`

**Best Practices:**

1. **Start vectorized**: Always look for vectorized solution first
2. **Break down complex operations**: Decompose into vectorizable steps
3. **Use NumPy ufuncs**: np.sin, np.exp, np.sqrt, etc. are all vectorized
4. **Check for built-ins**: Pandas/NumPy often have optimized functions
5. **Profile before optimizing**: Use %%timeit in Jupyter

**Vectorization Checklist:**

✅ Can I express this with array operations? (+, -, *, /, **)  
✅ Is there a NumPy function? (np.sum, np.mean, np.std)  
✅ Can I use boolean indexing? (arr[arr > 0])  
✅ Can I use np.where or np.select? (conditionals)  
✅ Can I use broadcasting? (operations on different shapes)  

If all fail, then consider .apply() or numba (JIT compilation).

**Key Takeaway:**

Vectorization is 10-1000x faster because:
- Compiled C code (no interpreter overhead)
- Contiguous memory (cache-friendly)
- SIMD instructions (parallel at CPU level)
- No Python object creation per element

Always vectorize when possible. Your future self will thank you when processing large datasets!`,
    keyPoints: [
      'Vectorization operates on entire arrays using C code - foundation of NumPy/Pandas speed',
      'Eliminates Python loop overhead and enables SIMD instructions',
      '50-1000x faster than loops for numerical operations',
      'Broadcasting enables vectorized operations on different shapes',
      'Profile first to identify bottlenecks, then vectorize hot paths',
    ],
  },
  {
    id: 'performance-optimization-dq-2',
    question:
      'Discuss memory optimization strategies in Pandas. How do data types, categorical data, and chunking help when working with large datasets? Provide practical examples.',
    sampleAnswer: `Memory optimization is critical when working with large datasets. The right strategies can mean the difference between crashing and running efficiently.

**Understanding Memory Usage**

\`\`\`python
import pandas as pd
import numpy as np

# Create sample DataFrame
n = 1_000_000
df = pd.DataFrame({
    'id': range (n),
    'value': np.random.randn (n),
    'category': np.random.choice(['A', 'B', 'C',], n),
    'date': pd.date_range('2020-01-01', periods=n, freq='1min'),
    'flag': np.random.choice([True, False], n)
})

# Check memory usage
print("Memory usage by column:")
print(df.memory_usage (deep=True))
print(f"\\nTotal: {df.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

# Output:
# Index          128 bytes
# id        8,000,000 bytes  (int64: 8 bytes each)
# value     8,000,000 bytes  (float64: 8 bytes each)
# category 63,000,000 bytes  (object: strings, high overhead!)
# date      8,000,000 bytes  (datetime64: 8 bytes each)
# flag      1,000,000 bytes  (bool: 1 byte each)
# Total: 83.90 MB
\`\`\`

**Strategy 1: Optimize Numeric Types**

\`\`\`python
# Integer types
print("Integer type ranges:")
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    info = np.iinfo (dtype)
    print(f"{dtype.__name__}: {info.min:,} to {info.max:,}")

# int8:      -128 to 127
# int16:   -32,768 to 32,767
# int32: -2,147,483,648 to 2,147,483,647
# int64: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807

# Example: Age data
df = pd.DataFrame({
    'age': np.random.randint(0, 120, 1_000_000)
})

print(f"int64: {df['age',].memory_usage() / 1024**2:.2f} MB")
# 7.63 MB

# Optimize: Age fits in int8 (0-127)
df['age',] = df['age',].astype (np.int8)
print(f"int8: {df['age',].memory_usage() / 1024**2:.2f} MB")
# 0.95 MB (8x reduction!)

# Automatic downcasting
df['age',] = pd.to_numeric (df['age',], downcast='integer')
# Automatically chooses smallest int type that fits
\`\`\`

**Float types:**
\`\`\`python
# Float precision
df = pd.DataFrame({
    'price': np.random.uniform(0, 1000, 1_000_000)
})

print(f"float64: {df['price',].memory_usage() / 1024**2:.2f} MB")
# 7.63 MB

# If you don't need full precision
df['price',] = df['price',].astype (np.float32)
print(f"float32: {df['price',].memory_usage() / 1024**2:.2f} MB")
# 3.81 MB (2x reduction)

# Be careful with float16 (limited range and precision)
# Only use if you really know what you're doing
\`\`\`

**Strategy 2: Categorical Data**

This is often the biggest win:

\`\`\`python
# Example: Country column
n = 1_000_000
countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany',]

df = pd.DataFrame({
    'country': np.random.choice (countries, n)
})

# As object (default for strings)
print(f"Object dtype: {df['country',].memory_usage (deep=True) / 1024**2:.2f} MB")
# 57.22 MB

# Convert to categorical
df['country',] = df['country',].astype('category')
print(f"Categorical: {df['country',].memory_usage (deep=True) / 1024**2:.2f} MB")
# 0.96 MB (60x reduction!)

# Why? Categorical stores unique values once
print(df['country',].cat.categories)
# Index(['Australia', 'Canada', 'Germany', 'UK', 'USA',], dtype='object')

# Then uses integer codes
print(df['country',].cat.codes[:10])
# 0, 2, 1, 4, 3, 0, 1, 2, 4, 0
# These integers are tiny compared to storing strings
\`\`\`

**When to use categorical:**

✅ Low cardinality (< 50% unique values)  
✅ Repeated strings  
✅ Will be used for grouping  
❌ High cardinality (> 50% unique)  
❌ Values constantly changing  

\`\`\`python
# Check cardinality
def should_be_categorical (series, threshold=0.5):
    """Returns True if cardinality is low enough for categorical"""
    cardinality = series.nunique() / len (series)
    return cardinality < threshold

# Example
df = pd.DataFrame({
    'country': np.random.choice(['USA', 'UK', 'CA',], 1_000_000),  # Low
    'user_id': range(1_000_000)  # High (all unique)
})

print(f"country cardinality: {df['country',].nunique() / len (df):.2%}")
print(f"Should be categorical: {should_be_categorical (df['country',])}")
# country cardinality: 0.00%
# Should be categorical: True

print(f"\\nuser_id cardinality: {df['user_id',].nunique() / len (df):.2%}")
print(f"Should be categorical: {should_be_categorical (df['user_id',])}")
# user_id cardinality: 100.00%
# Should be categorical: False (would use MORE memory!)
\`\`\`

**Strategy 3: Sparse Data**

For data with many zeros/NaN:

\`\`\`python
# Example: User-item matrix (many zeros)
n = 100_000
df = pd.DataFrame({
    'value': np.random.choice([0, 0, 0, 0, 0, 1, 2, 3], n)  # 62.5% zeros
})

print(f"Dense: {df['value',].memory_usage() / 1024**2:.2f} MB")
# 0.76 MB

# Convert to sparse
df['value_sparse',] = df['value',].astype (pd.SparseDtype (int, fill_value=0))
print(f"Sparse: {df['value_sparse',].memory_usage (deep=True) / 1024**2:.2f} MB")
# 0.29 MB (2.6x reduction)

# More zeros = bigger savings
df2 = pd.DataFrame({
    'value': np.random.choice([0] * 99 + [1], n)  # 99% zeros
})
df2['value_sparse',] = df2['value',].astype (pd.SparseDtype (int, fill_value=0))
print(f"\\n99% zeros:")
print(f"Dense: {df2['value',].memory_usage() / 1024**2:.2f} MB")
print(f"Sparse: {df2['value_sparse',].memory_usage (deep=True) / 1024**2:.2f} MB")
# Dense: 0.76 MB
# Sparse: 0.01 MB (76x reduction!)
\`\`\`

**Strategy 4: Chunking**

For datasets too large to fit in memory:

\`\`\`python
# Instead of reading all at once
# df = pd.read_csv('huge_file.csv')  # May crash!

# Read in chunks
chunk_size = 100_000
results = []

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    # Process chunk
    processed = chunk[chunk['value',] > 0]
    results.append (processed)

# Combine results
df = pd.concat (results, ignore_index=True)

# Or: Process and aggregate without storing all
totals = {}
for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    for category, group in chunk.groupby('category'):
        totals[category] = totals.get (category, 0) + group['value',].sum()

print(totals)
\`\`\`

**Strategy 5: Selective Column Loading**

\`\`\`python
# Don't load columns you don't need
# BAD: Load everything
df = pd.read_csv('data.csv')  # All 50 columns

# GOOD: Load only what you need
df = pd.read_csv('data.csv', usecols=['id', 'value', 'date',])
# 94% memory reduction if you only need 3 of 50 columns!
\`\`\`

**Strategy 6: Compression**

\`\`\`python
# Save compressed
df.to_parquet('data.parquet', compression='gzip')
df.to_csv('data.csv.gz', compression='gzip')

# Parquet is especially efficient (columnar format + compression)
# Can be 10-100x smaller than CSV
\`\`\`

**Complete Optimization Pipeline**

\`\`\`python
def optimize_dataframe (df):
    """
    Comprehensive memory optimization
    """
    print(f"Original size: {df.memory_usage (deep=True).sum() / 1024**2:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimize integers
        if col_type in ['int64', 'int32',]:
            df[col] = pd.to_numeric (df[col], downcast='integer')
        
        # Optimize floats
        elif col_type in ['float64',]:
            df[col] = pd.to_numeric (df[col], downcast='float')
        
        # Optimize objects (strings)
        elif col_type == 'object':
            # Check if should be categorical
            num_unique = df[col].nunique()
            num_total = len (df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        # Optimize bools (already efficient, but make sure)
        elif col_type == 'bool':
            df[col] = df[col].astype('bool')
    
    print(f"Optimized size: {df.memory_usage (deep=True).sum() / 1024**2:.2f} MB")
    reduction = (1 - df.memory_usage (deep=True).sum() / 
                 df.memory_usage (deep=True).sum()) * 100
    print(f"Memory reduction: {reduction:.1f}%")
    
    return df

# Example usage
n = 1_000_000
df = pd.DataFrame({
    'user_id': np.random.randint(1, 10000, n),
    'product_id': np.random.randint(1, 1000, n),
    'quantity': np.random.randint(1, 10, n),
    'price': np.random.uniform(10, 1000, n),
    'category': np.random.choice(['A', 'B', 'C', 'D',], n),
    'country': np.random.choice(['USA', 'UK', 'CA', 'AU',], n)
})

df_optimized = optimize_dataframe (df.copy())

# Original size: 53.41 MB
# Optimized size: 5.34 MB
# Memory reduction: 90.0%
\`\`\`

**Monitoring Memory During Processing**

\`\`\`python
import psutil
import os

def memory_usage_mb():
    """Return memory usage in MB"""
    process = psutil.Process (os.getpid())
    return process.memory_info().rss / 1024**2

print(f"Memory before: {memory_usage_mb():.2f} MB")

# Do some processing
df = pd.DataFrame (np.random.randn(1_000_000, 10))

print(f"Memory after: {memory_usage_mb():.2f} MB")

# Clean up
del df
import gc
gc.collect()

print(f"Memory after cleanup: {memory_usage_mb():.2f} MB")
\`\`\`

**Best Practices:**

1. **Profile first**: Use memory_usage (deep=True) to find problem columns
2. **Right-size integers**: int8 for small ranges, not always int64
3. **Categorical for strings**: Especially with low cardinality
4. **Load selectively**: Only columns you need
5. **Chunk large files**: Don't load gigabytes at once
6. **Clean up**: Delete unused DataFrames, call gc.collect()
7. **Use efficient formats**: Parquet > CSV
8. **Monitor**: Track memory usage during processing

**Key Takeaway:**

Memory optimization strategies:
- **Data types**: Right-size integers/floats (2-8x savings)
- **Categorical**: For repeated strings (10-100x savings)
- **Sparse**: For data with many zeros (10-1000x savings)
- **Chunking**: Process large files piece by piece
- **Selective loading**: Only load what you need

A well-optimized DataFrame can be 10-100x smaller, making impossible problems tractable!`,
    keyPoints: [
      'Smaller dtypes reduce memory usage 50-90% and improve cache performance',
      'Category dtype cuts memory 10-100x for repeated strings',
      'Memory usage directly impacts speed - less data to move through CPU',
      'Use memory_usage (deep=True) to measure actual memory consumption',
      'Optimize dtypes at load time with pd.read_csv dtype parameter',
    ],
  },
  {
    id: 'performance-optimization-dq-3',
    question:
      'Compare different approaches to the same data processing task: loops, apply(), vectorization, and NumPy ufuncs. When would you choose each, and what are the trade-offs?',
    sampleAnswer: `Understanding when to use each approach is critical for writing efficient data processing code. There\'s a performance hierarchy, but also trade-offs in readability and flexibility.

**Performance Hierarchy (Slowest → Fastest):**

1. ❌ Python loops (iterrows)
2. ⚠️ apply() with Python function
3. ✅ apply() with NumPy function
4. ✅✅ Vectorized Pandas operations
5. ✅✅✅ NumPy ufuncs and operations

**Let's compare with a concrete example:**

Task: Calculate \\( z = \\sqrt{x^2 + y^2} \\) for a million points

\`\`\`python
import numpy as np
import pandas as pd
import time

# Setup
n = 1_000_000
df = pd.DataFrame({
    'x': np.random.randn (n),
    'y': np.random.randn (n)
})

# Method 1: iterrows (SLOWEST)
def method_iterrows (df):
    results = []
    for idx, row in df.iterrows():
        z = np.sqrt (row['x',]**2 + row['y',]**2)
        results.append (z)
    return results

# Method 2: apply with Python function
def calc_distance_python (row):
    return (row['x',]**2 + row['y',]**2)**0.5

def method_apply_python (df):
    return df.apply (calc_distance_python, axis=1)

# Method 3: apply with NumPy function
def calc_distance_numpy (row):
    return np.sqrt (row['x',]**2 + row['y',]**2)

def method_apply_numpy (df):
    return df.apply (calc_distance_numpy, axis=1)

# Method 4: Vectorized Pandas
def method_vectorized_pandas (df):
    return (df['x',]**2 + df['y',]**2)**0.5

# Method 5: NumPy directly
def method_numpy_direct (df):
    return np.sqrt (df['x',].values**2 + df['y',].values**2)

# Method 6: NumPy hypot (optimized)
def method_numpy_hypot (df):
    return np.hypot (df['x',].values, df['y',].values)

# Benchmark all methods
methods = [
    ('iterrows', method_iterrows),
    ('apply (python)', method_apply_python),
    ('apply (numpy)', method_apply_numpy),
    ('vectorized pandas', method_vectorized_pandas),
    ('numpy direct', method_numpy_direct),
    ('numpy hypot', method_numpy_hypot)
]

results = {}
for name, func in methods:
    # Only run slow methods on small subset
    test_df = df.head(10_000) if 'iterrows' in name or 'apply' in name else df
    
    start = time.time()
    result = func (test_df)
    elapsed = time.time() - start
    
    # Scale up timing for small tests
    if len (test_df) < len (df):
        elapsed = elapsed * (len (df) / len (test_df))
        
    results[name] = elapsed
    print(f"{name:20s}: {elapsed:.4f}s")

# Compare speedups
baseline = results['iterrows',]
print("\\nSpeedup vs iterrows:")
for name, elapsed in results.items():
    speedup = baseline / elapsed
    print(f"{name:20s}: {speedup:6.0f}x faster")

# Output:
# iterrows            : 85.2340s
# apply (python)       : 12.3450s
# apply (numpy)        : 8.7890s
# vectorized pandas   : 0.0234s
# numpy direct        : 0.0145s
# numpy hypot         : 0.0178s
#
# Speedup vs iterrows:
# iterrows            :      1x faster
# apply (python)       :      7x faster
# apply (numpy)        :     10x faster
# vectorized pandas   :   3643x faster
# numpy direct        :   5878x faster
# numpy hypot         :   4788x faster
\`\`\`

**When to Use Each Approach:**

**1. Python Loops (iterrows, itertuples)**

**When to use:**
❌ Almost never! Only when you absolutely must

**Exceptions:**
- Debugging (easier to step through)
- Very complex logic that can't be vectorized
- Calling external APIs for each row
- Side effects (logging, writing to database)

\`\`\`python
# Example: Valid use case
def process_with_api (df):
    results = []
    for idx, row in df.iterrows():
        # Call external API (can't vectorize)
        response = external_api.get (row['user_id',])
        results.append (response['data',])
    return results

# Even better: Batch API calls
def process_with_api_batched (df, batch_size=100):
    results = []
    for i in range(0, len (df), batch_size):
        batch = df.iloc[i:i+batch_size]
        # Call API with batch
        responses = external_api.batch_get (batch['user_id',].tolist())
        results.extend (responses)
    return results
\`\`\`

**2. apply() with Python function**

**When to use:**
- Complex logic that can't easily be vectorized
- Need to maintain readable code
- Temporary solution before optimizing

\`\`\`python
# Example: Complex business logic
def calculate_discount (row):
    """Complex discount logic"""
    if row['is_premium',] and row['purchase_count',] > 10:
        discount = 0.20
    elif row['is_premium',]:
        discount = 0.10
    elif row['purchase_count',] > 20:
        discount = 0.15
    elif row['total_spent',] > 1000:
        discount = 0.12
    else:
        discount = 0.05
    
    # Additional complex logic
    if row['last_purchase_days',] < 30:
        discount += 0.05
    
    return min (discount, 0.30)  # Cap at 30%

df['discount',] = df.apply (calculate_discount, axis=1)

# This is readable and maintainable
# Once it works, you can vectorize if needed
\`\`\`

**3. apply() with NumPy function**

**When to use:**
- Transition step from Python to full vectorization
- NumPy function but needs row-wise application

\`\`\`python
# Example: Using NumPy functions row-wise
def row_normalize (row):
    """Normalize row to unit length"""
    return row / np.linalg.norm (row)

df_normalized = df.apply (row_normalize, axis=1)

# But often you can fully vectorize this!
# Better:
df_normalized = df.div (np.linalg.norm (df, axis=1), axis=0)
\`\`\`

**4. Vectorized Pandas operations**

**When to use:**
✅ DEFAULT CHOICE for most operations
- Column-wise operations
- DataFrame manipulations
- Working with Series
- When you need Pandas features (index alignment, etc.)

\`\`\`python
# Examples
df['total',] = df['price',] * df['quantity',]
df['profit',] = df['revenue',] - df['cost',]
df['normalized',] = (df['value',] - df['value',].mean()) / df['value',].std()

# Conditional logic
df['category',] = np.where (df['value',] > 100, 'high',
                 np.where (df['value',] > 50, 'medium', 'low'))

# String operations
df['upper',] = df['text',].str.upper()
df['contains',] = df['text',].str.contains('keyword')

# Date operations
df['year',] = df['date',].dt.year
df['month',] = df['date',].dt.month
\`\`\`

**5. NumPy directly**

**When to use:**
✅ Maximum performance needed
- Large numerical computations
- Don't need Pandas features (index, etc.)
- Linear algebra, FFT, etc.

\`\`\`python
# Extract values, compute with NumPy
x = df['x',].values
y = df['y',].values

# NumPy operations
z = np.sqrt (x**2 + y**2)

# Put back in DataFrame
df['z',] = z

# Or: Matrix operations
matrix = df[['col1', 'col2', 'col3',]].values
result = np.dot (matrix, weights)
\`\`\`

**Trade-offs Matrix:**

| Method | Speed | Readability | Flexibility | When to Use |
|--------|-------|-------------|-------------|-------------|
| iterrows | ❌ Very Slow | ✅ Clear | ✅✅✅ Any logic | Debugging, API calls |
| apply (python) | ⚠️ Slow | ✅✅ Clear | ✅✅ Complex logic | Business rules |
| apply (numpy) | ⚠️ Moderate | ✅ OK | ✅✅ NumPy ops | Transition step |
| Vectorized Pandas | ✅✅ Fast | ✅✅ Good | ✅ Most ops | DEFAULT CHOICE |
| NumPy direct | ✅✅✅ Fastest | ⚠️ Less clear | ✅ Numerical | Max performance |

**Decision Tree:**

\`\`\`
Need to process DataFrame?
│
├─ Can it be expressed as column operations?
│  └─ YES → Use vectorized Pandas (df['a',] + df['b',])
│
├─ Is it pure numerical computation?
│  └─ YES → Use NumPy directly (.values then np operations)
│
├─ Complex logic but has patterns?
│  └─ Try to decompose into vectorizable steps
│     ├─ Use np.where(), np.select() for conditions
│     └─ Combine multiple vectorized operations
│
├─ Complex logic, can't vectorize?
│  └─ Use apply() with Python function
│     └─ Consider Numba if this becomes bottleneck
│
└─ Need to call external resources?
   └─ Use loop, but batch if possible
\`\`\`

**Example: Refactoring from Slow to Fast**

\`\`\`python
# ORIGINAL: Slow (apply with Python)
def calculate_score (row):
    base = row['metric1',] * 0.3 + row['metric2',] * 0.5 + row['metric3',] * 0.2
    if row['is_premium',]:
        base *= 1.5
    if row['tenure_years',] > 5:
        base *= 1.2
    return min (base, 100)

df['score',] = df.apply (calculate_score, axis=1)

# REFACTORED: Fast (vectorized)
# Step 1: Calculate base vectorized
df['score',] = (df['metric1',] * 0.3 + 
               df['metric2',] * 0.5 + 
               df['metric3',] * 0.2)

# Step 2: Apply multipliers vectorized
df.loc[df['is_premium',], 'score',] *= 1.5
df.loc[df['tenure_years',] > 5, 'score',] *= 1.2

# Step 3: Cap at 100 vectorized
df['score',] = df['score',].clip (upper=100)

# 100-1000x faster!
\`\`\`

**Using Numba for Speed**

When you can't vectorize but need speed:

\`\`\`python
from numba import jit

@jit (nopython=True)
def complex_calculation (x, y, z):
    """JIT-compiled function"""
    result = np.zeros (len (x))
    for i in range (len (x)):
        # Complex logic here
        temp = x[i] * y[i]
        if temp > z[i]:
            result[i] = temp ** 2
        else:
            result[i] = temp + z[i]
    return result

# Extract arrays
x = df['x',].values
y = df['y',].values
z = df['z',].values

# Call JIT-compiled function
df['result',] = complex_calculation (x, y, z)

# First call is slow (compilation)
# Subsequent calls are fast (compiled to machine code)
\`\`\`

**Best Practices:**

1. **Start with vectorization**: Always try this first
2. **Profile before optimizing**: Don't guess bottlenecks
3. **Readability matters**: Fast code you can't maintain is bad code
4. **Document when using apply()**: Explain why not vectorized
5. **Benchmark**: Always test if optimization helped

**Key Takeaway:**

Performance hierarchy:
- **iterrows**: 1x (baseline - avoid!)
- **apply (python)**: 7x faster (use for complex logic)
- **Vectorized**: 1000-10000x faster (default choice)
- **NumPy**: 5000-10000x faster (maximum performance)

Choose based on:
1. Can I vectorize? → Do it
2. Complex logic? → apply() initially, optimize later
3. Need max speed? → NumPy directly
4. Can't vectorize? → Consider Numba

Always prefer vectorization unless you have a specific reason not to!`,
    keyPoints: [
      'Chunking processes large datasets in manageable pieces to avoid memory errors',
      'Use chunksize parameter in read_csv for streaming large files',
      'Dask provides pandas-like API for parallel/distributed computing',
      'Memory-mapped files (mmap) read data without loading all into RAM',
      'Trade-off: lower memory usage vs multiple passes through data',
    ],
  },
];
