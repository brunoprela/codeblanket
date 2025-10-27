import { QuizQuestion } from '../../../types';

export const datamanipulationpandasQuiz: QuizQuestion[] = [
  {
    id: 'data-manipulation-pandas-dq-1',
    question:
      'Explain why vectorized operations are dramatically faster than .apply() with lambdas in Pandas. Provide a concrete example demonstrating the performance difference and explain when apply() is still necessary.',
    sampleAnswer: `Vectorized operations in Pandas are dramatically faster than .apply() because they operate on entire arrays at once using compiled C code, while .apply() with lambdas executes Python code for each element/row.

**Performance Comparison:**

\`\`\`python
import pandas as pd
import numpy as np
import time

# Create large DataFrame
n = 1_000_000
df = pd.DataFrame({
    'A': np.random.randn (n),
    'B': np.random.randn (n)
})

# Method 1: Apply with lambda (SLOW)
start = time.time()
df['C_apply',] = df['A',].apply (lambda x: x ** 2 if x > 0 else 0)
time_apply = time.time() - start

# Method 2: Vectorized with np.where (FAST)
start = time.time()
df['C_vectorized',] = np.where (df['A',] > 0, df['A',] ** 2, 0)
time_vectorized = time.time() - start

# Method 3: Vectorized with boolean indexing (FAST)
start = time.time()
df['C_boolean',] = 0
df.loc[df['A',] > 0, 'C_boolean',] = df.loc[df['A',] > 0, 'A',] ** 2
time_boolean = time.time() - start

print(f"Apply: {time_apply:.3f}s")
print(f"np.where: {time_vectorized:.3f}s")
print(f"Boolean indexing: {time_boolean:.3f}s")
print(f"Speedup (apply vs vectorized): {time_apply/time_vectorized:.1f}x")

# Typical results:
# Apply: 2.456s
# np.where: 0.045s
# Boolean indexing: 0.062s
# Speedup: 54.6x faster!
\`\`\`

**Why This Huge Difference?**

**Apply with Lambda:**1. Python loop over 1 million elements
2. Each element calls Python lambda function
3. Type checking for every call
4. Python object overhead
5. No SIMD (Single Instruction Multiple Data)
6. No compiler optimizations

**Vectorized Operations:**1. Single call to compiled C/Cython code
2. Operations on contiguous memory
3. SIMD instructions (process multiple values at once)
4. CPU cache-friendly access patterns
5. Minimal Python interpreter involvement
6. Optimized by C compiler

**When Apply() is Necessary:**

Despite the performance penalty, .apply() is sometimes necessary:

**1. Complex Business Logic:**
\`\`\`python
def complex_calculation (row):
    if row['type',] == 'A':
        if row['value',] > 100:
            return row['value',] * 1.5 + row['bonus',]
        else:
            return row['value',] * 1.2
    elif row['type',] == 'B':
        return row['value',] * 0.8 if row['flag',] else row['value',]
    else:
        return 0

# Too complex for np.where/np.select
df['result',] = df.apply (complex_calculation, axis=1)
\`\`\`

**2. Multiple Column Dependencies:**
\`\`\`python
# Create full name from multiple columns with complex logic
df['full_info',] = df.apply(
    lambda row: f"{row['first_name',]} {row['middle_initial',]}. {row['last_name',]}" 
                if pd.notna (row['middle_initial',]) 
                else f"{row['first_name',]} {row['last_name',]}",
    axis=1
)
\`\`\`

**3. External Function Calls:**
\`\`\`python
# Call external API or complex library
import some_external_library

df['processed',] = df['text',].apply(
    lambda x: some_external_library.complex_function (x)
)
\`\`\`

**4. Non-Vectorizable Operations:**
\`\`\`python
# Operations that can't be vectorized
df['word_count',] = df['text',].apply (lambda x: len (x.split()))
# Note: This specific example can be vectorized with .str.split().str.len()
# but illustrates the concept
\`\`\`

**Optimization Strategies:**

**Strategy 1: Vectorize When Possible**
\`\`\`python
# Instead of:
df['C',] = df.apply (lambda row: row['A',] + row['B',], axis=1)

# Use:
df['C',] = df['A',] + df['B',]  # 100x faster
\`\`\`

**Strategy 2: Use np.where for Conditionals**
\`\`\`python
# Instead of:
df['category',] = df['value',].apply (lambda x: 'High' if x > 100 else 'Low')

# Use:
df['category',] = np.where (df['value',] > 100, 'High', 'Low')  # 50x faster
\`\`\`

**Strategy 3: Use np.select for Multiple Conditions**
\`\`\`python
# Instead of:
def categorize (x):
    if x < 10: return 'Low'
    elif x < 50: return 'Medium'
    elif x < 100: return 'High'
    else: return 'Very High'

df['category',] = df['value',].apply (categorize)

# Use:
conditions = [
    df['value',] < 10,
    df['value',] < 50,
    df['value',] < 100
]
choices = ['Low', 'Medium', 'High',]
df['category',] = np.select (conditions, choices, default='Very High')  # 30x faster
\`\`\`

**Strategy 4: Use Vectorized String Methods**
\`\`\`python
# Instead of:
df['upper',] = df['name',].apply (lambda x: x.upper())

# Use:
df['upper',] = df['name',].str.upper()  # 20x faster
\`\`\`

**Strategy 5: Use .transform() for Group Operations**
\`\`\`python
# Instead of:
def pct_of_group_total (row):
    group_total = df[df['group',] == row['group',]]['value',].sum()
    return row['value',] / group_total

df['pct',] = df.apply (pct_of_group_total, axis=1)

# Use:
df['pct',] = df.groupby('group')['value',].transform (lambda x: x / x.sum())  # 100x faster
\`\`\`

**Strategy 6: Cythonize or Numba for Complex Logic**
\`\`\`python
from numba import jit

@jit (nopython=True)
def complex_calculation_numba(A, B):
    result = np.zeros (len(A))
    for i in range (len(A)):
        if A[i] > 0:
            result[i] = A[i] ** 2 + B[i]
        else:
            result[i] = B[i] * 2
    return result

df['result',] = complex_calculation_numba (df['A',].values, df['B',].values)
# Faster than apply, nearly as fast as pure vectorization
\`\`\`

**Decision Tree:**

\`\`\`
Can you vectorize the operation?
├─ Yes → Use vectorized operations (fastest)
│  ├─ Single operation → df['A',] + df['B',]
│  ├─ Conditional → np.where() or np.select()
│  ├─ String operations → .str accessor
│  └─ Group operations → .transform()
│
└─ No → Can you simplify the logic?
   ├─ Yes → Refactor to vectorize
   │
   └─ No → Must use .apply()
      ├─ Large dataset (>100K rows) → Consider Numba/Cython
      ├─ Row-wise operation → axis=1
      └─ Column-wise → axis=0 (faster than axis=1)
\`\`\`

**Real-World Impact:**

For a typical data science project:
- ETL on 10M rows taking 30 minutes with apply
- Same operations with vectorization: 30 seconds
- **60x faster** = faster iteration, more experimentation

**Key Takeaway:**
Always try to vectorize first. If you can't, consider if the complex logic is worth the performance trade-off. For very complex logic on large datasets, tools like Numba can provide C-like speed while maintaining Python expressiveness.`,
    keyPoints: [
      'Vectorized operations use compiled C code on entire arrays - 50-100x faster than .apply()',
      '.apply() with lambdas executes Python code for each element - use only when vectorization impossible',
      'Use np.where() for conditional operations, boolean indexing for filtering',
      'Method chaining improves readability but can hide performance issues',
      'Profile code with %%timeit to identify bottlenecks before optimizing',
    ],
  },
  {
    id: 'data-manipulation-pandas-dq-2',
    question:
      'Explain the differences between .apply(), .map(), .applymap(), and .transform() in Pandas. When would you use each method, and what are their performance characteristics?',
    sampleAnswer: `Pandas provides multiple methods for applying functions to data, each with specific use cases, behavior, and performance characteristics. Understanding these differences is crucial for writing efficient code.

**The Four Methods:**

**1. .apply() - Most Versatile**

Works on Series or DataFrames, can operate row-wise or column-wise:

\`\`\`python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# On Series (column-wise by default)
df['A_squared',] = df['A',].apply (lambda x: x ** 2)
# Result: [1, 4, 9]

# On DataFrame, axis=0 (apply to each column)
column_sums = df.apply (sum, axis=0)
# Result: Series with sum of each column
# A: 6, B: 15, C: 24

# On DataFrame, axis=1 (apply to each row)
row_sums = df.apply (sum, axis=1)
# Result: Series with sum of each row
# 0: 12, 1: 15, 2: 18

# Complex row-wise operation
df['row_info',] = df.apply(
    lambda row: f"A:{row['A',]}, B:{row['B',]}, Sum:{row['A',]+row['B',]}", 
    axis=1
)
\`\`\`

**Characteristics:**
- **Input**: Series or DataFrame
- **Output**: Series, DataFrame, or scalar (depends on function)
- **Axis**: Can operate on rows (axis=1) or columns (axis=0)
- **Performance**: Slow (Python function calls)
- **Use case**: Complex logic requiring multiple columns

**2. .map() - Element-wise Substitution/Transformation**

Series only, primarily for substitution or element-wise transformation:

\`\`\`python
s = pd.Series(['A', 'B', 'C', 'A',])

# Mapping with dictionary
mapping = {'A': 1, 'B': 2, 'C': 3}
s_mapped = s.map (mapping)
# Result: [1, 2, 3, 1]

# Mapping with function
s_numbers = pd.Series([1, 2, 3, 4])
s_squared = s_numbers.map (lambda x: x ** 2)
# Result: [1, 4, 9, 16]

# Mapping with Series (useful for lookups)
lookup = pd.Series([100, 200, 300], index=['A', 'B', 'C',])
s_looked_up = s.map (lookup)
# Result: [100, 200, 300, 100]

# Key behavior: unmapped values become NaN
s_partial = s.map({'A': 1, 'B': 2})  # C not mapped
# Result: [1, 2, NaN, 1]
\`\`\`

**Characteristics:**
- **Input**: Series only
- **Output**: Series
- **Axis**: N/A (element-wise)
- **Performance**: Moderate (can use dict lookup, which is fast)
- **Use case**: Value substitution, element-wise transformation
- **Special**: Returns NaN for unmapped values

**3. .applymap() - Deprecated in favor of .map() on DataFrame**

Element-wise function application to DataFrame:

\`\`\`python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# .applymap (deprecated, use .map instead)
# df_squared = df.applymap (lambda x: x ** 2)  # Old way

# New way (pandas 2.1+):
df_squared = df.map (lambda x: x ** 2)
# Result: All values squared
#    A   B
# 0  1  16
# 1  4  25
# 2  9  36
\`\`\`

**Characteristics:**
- **Input**: DataFrame only
- **Output**: DataFrame (same shape)
- **Axis**: Element-wise (every cell)
- **Performance**: Slow (Python function call per element)
- **Use case**: Apply same function to every element
- **Note**: Deprecated in favor of DataFrame.map()

**4. .transform() - Returns Same Shape as Input**

Ensures output has same index as input (crucial for group operations):

\`\`\`python
df = pd.DataFrame({
    'Group': ['A', 'A', 'B', 'B',],
    'Value': [1, 2, 3, 4]
})

# .transform returns Series with same index as input
df['Value_Normalized',] = df.groupby('Group')['Value',].transform(
    lambda x: (x - x.mean()) / x.std()
)
# Result: Normalized within each group, same length as original

# Compare with .apply (aggregates by default)
group_means = df.groupby('Group')['Value',].apply (lambda x: x.mean())
# Result: Series with 2 elements (one per group)
# A: 1.5, B: 3.5

# .transform with built-in functions (faster)
df['Group_Mean',] = df.groupby('Group')['Value',].transform('mean')
df['Cumulative_Sum',] = df['Value',].transform('cumsum')
\`\`\`

**Characteristics:**
- **Input**: Series or GroupBy object
- **Output**: Same shape as input
- **Axis**: Element-wise, but aware of groups
- **Performance**: Moderate (faster with built-in functions)
- **Use case**: Add group statistics to original DataFrame

**Comparison Table:**

| Method | Works On | Output Shape | Typical Use Case | Performance |
|--------|----------|--------------|------------------|-------------|
| **.apply()** | Series, DataFrame | Variable | Complex multi-column logic | Slow (Python loops) |
| **.map()** | Series | Same as input | Value substitution | Moderate (dict lookup fast) |
| **.applymap()** (deprecated) | DataFrame | Same as input | Element-wise on all cells | Slow (Python loops) |
| **.transform()** | Series, GroupBy | Same as input | Add group stats to data | Moderate |

**Performance Comparison:**

\`\`\`python
import time

n = 100_000
df = pd.DataFrame({
    'A': np.random.randint(0, 100, n),
    'B': np.random.randint(0, 100, n)
})

# Test: Square all values

# Method 1: applymap (element-wise)
start = time.time()
result1 = df.map (lambda x: x ** 2)  # New syntax for applymap
time1 = time.time() - start

# Method 2: apply per column
start = time.time()
result2 = df.apply (lambda x: x ** 2)
time2 = time.time() - start

# Method 3: Vectorized (fastest)
start = time.time()
result3 = df ** 2
time3 = time.time() - start

print(f"applymap (map): {time1:.3f}s")
print(f"apply: {time2:.3f}s")
print(f"Vectorized: {time3:.3f}s")

# Typical results:
# applymap: 1.234s
# apply: 0.567s
# Vectorized: 0.003s
\`\`\`

**When to Use Each:**

**Use .apply() when:**
- Need to access multiple columns in a row
- Complex conditional logic
- Return value varies in type/size
- No vectorized alternative exists

\`\`\`python
df['complex',] = df.apply(
    lambda row: row['A',] * 2 if row['B',] > 50 else row['A',] + row['B',],
    axis=1
)
\`\`\`

**Use .map() when:**
- Substituting values from dictionary
- Element-wise transformation on Series
- Looking up values from another Series
- Want NaN for unmapped values

\`\`\`python
df['category',] = df['code',].map({1: 'Low', 2: 'Medium', 3: 'High'})
\`\`\`

**Use DataFrame.map() (was applymap) when:**
- Apply same simple function to every element
- No vectorized alternative
- Working with small DataFrames

\`\`\`python
df_formatted = df.map (lambda x: f"\${x:,.2f}")  # Format all values as currency
\`\`\`

**Use .transform() when:**
- Adding group-level statistics to individual rows
- Need output with same shape as input
- Working with groupby operations
- Want to preserve index alignment

\`\`\`python
df['value_pct_of_group',] = df.groupby('category')['value',].transform(
    lambda x: x / x.sum()
)
\`\`\`

**Best Practices:**

**1. Always Try Vectorization First:**
\`\`\`python
# Don't: df['C',] = df.apply (lambda row: row['A',] + row['B',], axis=1)
# Do: df['C',] = df['A',] + df['B',]
\`\`\`

**2. Use Built-in Functions When Available:**
\`\`\`python
# Don't: df.groupby('group')['value',].transform (lambda x: x.mean())
# Do: df.groupby('group')['value',].transform('mean')  # Much faster
\`\`\`

**3. Avoid .map() on DataFrames (use vectorization):**
\`\`\`python
# Don't: df.map (lambda x: x ** 2)
# Do: df ** 2
\`\`\`

**4. Use .map() for Dictionaries:**
\`\`\`python
# Best use case for .map()
df['state_code',] = df['state',].map({
    'California': 'CA',
    'Texas': 'TX',
    'New York': 'NY'
})
\`\`\`

**5. Remember .transform() Maintains Shape:**
\`\`\`python
# transform: Returns same length as input (good for adding column)
df['group_mean',] = df.groupby('group')['value',].transform('mean')

# apply: Returns aggregated result (good for summary)
group_means = df.groupby('group')['value',].apply('mean')
\`\`\`

**Memory Considerations:**

- \`.apply()\` can be memory-intensive for row-wise operations (axis=1)
- \`.transform()\` creates full Series even if many duplicates
- Vectorized operations are most memory-efficient

**Key Takeaway:**
Use the right tool for the job: vectorize when possible, .map() for substitution, .apply() for complex logic, .transform() for group statistics. Understanding these differences can make your code 10-100x faster!`,
    keyPoints: [
      'Sorting by multiple columns with different orders uses list of bools for ascending parameter',
      'sort_values() has stable sorting - equal elements maintain original order',
      'nlargest()/nsmallest() faster than sort + head for top-N queries',
      'rank() provides various tying methods (average, min, max, first, dense)',
      'For large datasets, consider partial sorting algorithms for better performance',
    ],
  },
  {
    id: 'data-manipulation-pandas-dq-3',
    question:
      'Discuss the .str and .dt accessors in Pandas. How do they work under the hood, why are they necessary, and what are their performance implications compared to standard Python string/datetime operations?',
    sampleAnswer: `The .str and .dt accessors in Pandas provide vectorized operations for strings and datetimes respectively. They're critical for efficient data manipulation and understanding how they work reveals important design principles in Pandas.

**What Are Accessors?**

Accessors are special attributes that provide domain-specific functionality for particular data types:

\`\`\`python
import pandas as pd

# .str accessor for string operations
s_str = pd.Series(['hello', 'world', 'PANDAS',])
print(s_str.str.upper())  # Vectorized uppercase

# .dt accessor for datetime operations
s_dt = pd.Series (pd.date_range('2024-01-01', periods=3))
print(s_dt.dt.month)  # Extract month

# .cat accessor for categorical data
s_cat = pd.Series(['A', 'B', 'C',], dtype='category')
print(s_cat.cat.categories)  # Access categories
\`\`\`

**How They Work Under the Hood:**

**1. .str Accessor:**

\`\`\`python
# When you write:
s.str.upper()

# Pandas essentially does:
# 1. Checks that Series contains strings (object dtype)
# 2. Converts to StringArray if needed (optimization)
# 3. Applies operation in vectorized manner
# 4. Returns new Series with results

# Pseudocode:
def str_upper (series):
    # Check dtype
    if series.dtype != 'object':
        raise AttributeError("Can only use .str accessor with string values")
    
    # Vectorized operation (implemented in Cython)
    result = []
    for value in series:
        if pd.isna (value):
            result.append (value)  # Preserve NaN
        else:
            result.append (value.upper())  # Python str.upper()
    
    return pd.Series (result, index=series.index)
\`\`\`

**2. .dt Accessor:**

\`\`\`python
# When you write:
s.dt.month

# Pandas does:
# 1. Checks that Series is datetime64[ns] dtype
# 2. Extracts underlying NumPy datetime64 array
# 3. Uses vectorized NumPy datetime operations
# 4. Returns Series with integer results

# Pseudocode:
def dt_month (series):
    if series.dtype != 'datetime64[ns]':
        raise AttributeError("Can only use .dt accessor with datetime values")
    
    # Vectorized extraction from underlying numpy array
    # Much faster than Python datetime.month
    return pd.Series(
        series.values.astype('datetime64[M]').astype (int) % 12 + 1,
        index=series.index
    )
\`\`\`

**Why Are Accessors Necessary?**

**Problem: Name Collision**

Without accessors, operations would collide with Series/DataFrame methods:

\`\`\`python
# Imagine if .str didn't exist:
s.upper()  # Conflicts with potential Series.upper() method
s.contains('hello')  # Conflicts with DataFrame.contains() if it existed

# With accessor, clear namespace:
s.str.upper()  # Clearly string operation
s.str.contains('hello')  # No ambiguity
\`\`\`

**Problem: Type Safety**

Accessors provide clear error messages:

\`\`\`python
s_numbers = pd.Series([1, 2, 3])

# Without accessor:
# s_numbers.upper()  # Confusing error

# With accessor:
s_numbers.str.upper()  # AttributeError: Can only use .str accessor with string values
\`\`\`

**Problem: Vectorization**

Manual loops are slow; accessors provide vectorized operations:

\`\`\`python
s = pd.Series(['hello', 'world',] * 50000)

# Slow: Manual loop
import time
start = time.time()
result_slow = s.apply (lambda x: x.upper())
time_slow = time.time() - start

# Fast: Vectorized with .str
start = time.time()
result_fast = s.str.upper()
time_fast = time.time() - start

print(f"Manual: {time_slow:.3f}s")
print(f"Vectorized: {time_fast:.3f}s")
print(f"Speedup: {time_slow/time_fast:.1f}x")

# Typical results:
# Manual: 0.245s
# Vectorized: 0.089s
# Speedup: 2.8x
\`\`\`

**Performance Comparison:**

**String Operations:**

\`\`\`python
import pandas as pd
import time

n = 100_000
s = pd.Series(['test_string_' + str (i) for i in range (n)])

# Method 1: Python loop
start = time.time()
result1 = [x.upper() for x in s]
time1 = time.time() - start

# Method 2: .apply()
start = time.time()
result2 = s.apply (lambda x: x.upper())
time2 = time.time() - start

# Method 3: .str accessor
start = time.time()
result3 = s.str.upper()
time3 = time.time() - start

print(f"Python loop: {time1:.3f}s")
print(f"apply(): {time2:.3f}s")
print(f".str accessor: {time3:.3f}s")

# Typical results:
# Python loop: 0.015s
# apply(): 0.098s
# .str accessor: 0.035s

# Interesting: List comprehension is fastest for simple operations!
# But .str is more flexible and handles NaN automatically
\`\`\`

**Datetime Operations:**

\`\`\`python
n = 100_000
dates = pd.Series (pd.date_range('2020-01-01', periods=n, freq='h'))

# Method 1: .apply() with Python datetime
start = time.time()
result1 = dates.apply (lambda x: x.month)
time1 = time.time() - start

# Method 2: .dt accessor (vectorized NumPy)
start = time.time()
result2 = dates.dt.month
time2 = time.time() - start

print(f"apply(): {time1:.3f}s")
print(f".dt accessor: {time2:.3f}s")
print(f"Speedup: {time1/time2:.1f}x")

# Typical results:
# apply(): 0.856s
# .dt accessor: 0.003s
# Speedup: 285x faster!
\`\`\`

**Common .str Operations:**

\`\`\`python
s = pd.Series(['  Hello World  ', 'PANDAS', 'data science', None])

# Case conversion
print(s.str.lower())      # ['  hello world  ', 'pandas', 'data science', NaN]
print(s.str.upper())      # ['  HELLO WORLD  ', 'PANDAS', 'DATA SCIENCE', NaN]
print(s.str.title())      # ['  Hello World  ', 'Pandas', 'Data Science', NaN]

# Whitespace
print(s.str.strip())      # ['Hello World', 'PANDAS', 'data science', NaN]

# Contains/matches
print(s.str.contains('World'))  # [True, False, False, NaN]
print(s.str.startswith('Hello'))  # [False, False, False, NaN]

# Splitting
print(s.str.split().str[0])  # ['Hello', 'PANDAS', 'data', NaN]

# Length
print(s.str.len())  # [18.0, 6.0, 12.0, NaN]

# Replace
print(s.str.replace('World', 'Earth'))

# Regex operations
s_num = pd.Series(['abc123', 'def456', 'ghi789',])
print(s_num.str.extract (r'(\\d+)'))  # Extract digits

# Padding
print(s_num.str.pad(10, fillchar='0'))  # Pad to length 10
\`\`\`

**Common .dt Operations:**

\`\`\`python
dates = pd.Series (pd.date_range('2024-01-15', periods=5, freq='D'))

# Extract components
print(dates.dt.year)           # [2024, 2024, ...]
print(dates.dt.month)          # [1, 1, ...]
print(dates.dt.day)            # [15, 16, 17, 18, 19]
print(dates.dt.dayofweek)      # [0, 1, 2, 3, 4] (Monday=0)
print(dates.dt.day_name())     # ['Monday', 'Tuesday', ...]
print(dates.dt.month_name())   # ['January', 'January', ...]

# Is checks
print(dates.dt.is_month_start)  # [False, False, ...]
print(dates.dt.is_month_end)    # [False, False, ...]
print(dates.dt.is_quarter_start)

# Date arithmetic components
print(dates.dt.quarter)         # [1, 1, 1, 1, 1]
print(dates.dt.week)           # ISO week number

# Time components (if datetime has time)
times = pd.Series (pd.date_range('2024-01-01 10:30:45', periods=3, freq='h'))
print(times.dt.hour)           # [10, 11, 12]
print(times.dt.minute)         # [30, 30, 30]
print(times.dt.second)         # [45, 45, 45]

# Formatting
print(dates.dt.strftime('%Y-%m-%d'))  # ['2024-01-15', ...]

# Timezone operations
dates_tz = dates.dt.tz_localize('UTC')
dates_tz_ny = dates_tz.dt.tz_convert('America/New_York')
\`\`\`

**Handling Missing Values:**

Key advantage: accessors handle NaN automatically:

\`\`\`python
s = pd.Series(['hello', None, 'world', np.nan, 'pandas',])

# .str handles NaN gracefully
result = s.str.upper()
# ['HELLO', NaN, 'WORLD', NaN, 'PANDAS',]

# Manual operations require explicit NaN handling
def manual_upper (x):
    if pd.isna (x):
        return x
    return x.upper()

result_manual = s.apply (manual_upper)
# Same result, but more verbose
\`\`\`

**When .str/.dt Are Slower:**

For very simple operations on small data, list comprehensions can be faster:

\`\`\`python
s_small = pd.Series(['a', 'b', 'c',] * 10)

# List comprehension (faster for small data)
result = pd.Series([x.upper() for x in s_small])

# .str accessor (overhead not worth it for small data)
result = s_small.str.upper()

# Break-even point: ~1000 elements
# Beyond that, .str becomes faster
\`\`\`

**Best Practices:**

**1. Use Accessors for Consistency:**
\`\`\`python
# Good: Clear and handles NaN
df['month',] = df['date',].dt.month

# Bad: Might fail on NaN
df['month',] = df['date',].apply (lambda x: x.month)
\`\`\`

**2. Chain Operations:**
\`\`\`python
# Clean text data
df['clean_text',] = (df['text',]
    .str.lower()
    .str.strip()
    .str.replace (r'[^a-z\\s]', ', regex=True)
)
\`\`\`

**3. Use Built-in Methods:**
\`\`\`python
# Good: Vectorized
df['contains_python',] = df['text',].str.contains('python')

# Bad: Slower
df['contains_python',] = df['text',].apply (lambda x: 'python' in x)
\`\`\`

**4. Batch Date Conversions:**
\`\`\`python
# Good: Convert once
df['date',] = pd.to_datetime (df['date_string',])
df['year',] = df['date',].dt.year
df['month',] = df['date',].dt.month

# Bad: Multiple conversions
df['year',] = pd.to_datetime (df['date_string',]).dt.year
df['month',] = pd.to_datetime (df['date_string',]).dt.month  # Redundant conversion!
\`\`\`

**Key Takeaways:**1. **Accessors provide vectorized operations** for domain-specific data types
2. **.dt is dramatically faster** than .apply() for datetime (100-1000x)
3. **.str is moderately faster** than .apply() for strings (2-10x)
4. **Automatic NaN handling** is a major convenience
5. **Clear namespace** avoids method name collisions
6. **Type safety** provides better error messages

The accessor pattern is a clever design that balances performance, clarity, and safety—a good example of thoughtful API design in a data library!`,
    keyPoints: [
      'Query method enables SQL-like string syntax for filtering DataFrames',
      'Query is more readable for complex conditions and can reference variables with @',
      'Boolean indexing with bitwise operators (&, |, ~) is standard Pandas approach',
      'Query can be faster for large DataFrames due to numexpr optimization',
      'Choose query for readability with simple conditions, boolean indexing for complex logic',
    ],
  },
];
