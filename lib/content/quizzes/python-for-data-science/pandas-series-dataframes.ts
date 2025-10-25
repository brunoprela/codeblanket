import { QuizQuestion } from '../../../types';

export const pandasseriesdataframesQuiz: QuizQuestion[] = [
  {
    id: 'pandas-series-dataframes-dq-1',
    question:
      'Explain the relationship between Pandas and NumPy. Why does Pandas build on NumPy, and when would you use each library?',
    sampleAnswer: `Pandas and NumPy have a symbiotic relationship—Pandas is built on top of NumPy, extending its capabilities for practical data analysis while maintaining performance.

**Technical Relationship:**

1. **Under the Hood**: Every Pandas Series and DataFrame column is backed by a NumPy array
\`\`\`python
import pandas as pd
import numpy as np

s = pd.Series([1, 2, 3, 4, 5])
print(type (s.values))  # <class 'numpy.ndarray'>
print(s.values.dtype)  # int64

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(type (df['A',].values))  # <class 'numpy.ndarray'>
\`\`\`

2. **Memory Efficiency**: Pandas inherits NumPy\'s contiguous memory layout and efficient operations

3. **Interoperability**: Easy conversion between types
\`\`\`python
# Pandas to NumPy
arr = df.values  # or df.to_numpy()

# NumPy to Pandas
df = pd.DataFrame (arr, columns=['A', 'B',])
\`\`\`

**Why Pandas Builds on NumPy:**

**NumPy provides:**
- Fast C/Fortran implementation
- Efficient memory management
- Vectorized operations
- Mathematical foundations

**Pandas adds:**
- **Labels**: Named rows and columns (not just positions)
- **Mixed types**: Different columns with different dtypes
- **Missing data**: Native NaN handling
- **Alignment**: Automatic alignment by index
- **Rich API**: groupby, merge, pivot, time series functions

**Example Demonstrating the Difference:**

\`\`\`python
# NumPy: Positional, homogeneous
arr = np.array([[1, 2], [3, 4]])
value = arr[0, 1]  # Must remember positions

# Pandas: Labeled, heterogeneous
df = pd.DataFrame({
    'Name': ['Alice', 'Bob',],
    'Age': [25, 30],
    'Salary': [50000, 60000]
})
value = df.loc[0, 'Salary',]  # Self-documenting
\`\`\`

**When to Use NumPy:**

1. **Numerical computing**: Pure numerical operations, linear algebra
\`\`\`python
# Matrix multiplication, eigenvalues, etc.
A = np.random.randn(1000, 1000)
eigenvalues = np.linalg.eig(A)[0]
\`\`\`

2. **Homogeneous data**: All elements same type
\`\`\`python
# Image processing (all pixels are numbers)
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
\`\`\`

3. **Performance-critical code**: When every microsecond counts
\`\`\`python
# NumPy operations are slightly faster (less overhead)
arr = np.arange(1_000_000)
result = arr ** 2  # Faster than Pandas equivalent
\`\`\`

4. **Neural networks**: Input to PyTorch/TensorFlow
\`\`\`python
import torch
X = np.random.randn(32, 10)  # Batch of 32, 10 features
X_torch = torch.from_numpy(X)
\`\`\`

**When to Use Pandas:**

1. **Tabular data**: Datasets with rows and columns
\`\`\`python
# Employee data with mixed types
df = pd.DataFrame({
    'name': ['Alice', 'Bob',],
    'age': [25, 30],
    'salary': [50000, 60000],
    'department': ['IT', 'HR',]
})
\`\`\`

2. **Data cleaning**: Missing values, duplicates, type conversion
\`\`\`python
df = df.dropna()  # Remove missing values
df = df.drop_duplicates()  # Remove duplicates
df['age',] = pd.to_numeric (df['age',], errors='coerce')
\`\`\`

3. **Exploratory analysis**: Quick statistics, grouping, aggregation
\`\`\`python
# Group by department, calculate average salary
avg_salary = df.groupby('department')['salary',].mean()
\`\`\`

4. **Time series**: Date/time indexing and operations
\`\`\`python
df['date',] = pd.to_datetime (df['date',])
df = df.set_index('date')
monthly_avg = df.resample('M').mean()
\`\`\`

5. **Data integration**: Merging, joining multiple datasets
\`\`\`python
merged = pd.merge (df1, df2, on='customer_id', how='left')
\`\`\`

6. **Reading/writing files**: CSV, Excel, SQL, JSON
\`\`\`python
df = pd.read_csv('data.csv')
df.to_excel('output.xlsx')
\`\`\`

**Typical Data Science Workflow:**

1. **Load data**: Pandas (read_csv, read_sql)
2. **Clean data**: Pandas (dropna, fillna, replace)
3. **Feature engineering**: Pandas (groupby, merge, transform)
4. **Convert to NumPy**: For ML model input
\`\`\`python
X = df[features].values  # NumPy array
y = df[target].values
\`\`\`
5. **Train model**: NumPy arrays to sklearn/PyTorch
6. **Results analysis**: Back to Pandas for visualization

**Performance Considerations:**

- **NumPy**: Faster for pure numerical operations (10-20% faster)
- **Pandas**: Faster for data manipulation (groupby, merge, etc.)
- **Memory**: NumPy uses less memory for homogeneous data

**Rule of Thumb:**

- Working with labeled, heterogeneous, real-world data → **Pandas**
- Numerical computing, arrays, tensors → **NumPy**
- Both in same project → **Common and recommended**

Pandas makes data analysis intuitive and expressive, while NumPy provides the computational muscle. Together, they form the foundation of the Python data science stack.`,
    keyPoints: [
      'Pandas is built on NumPy - every Series/DataFrame column is backed by NumPy array',
      'NumPy provides speed and memory efficiency, Pandas adds labels, mixed types, and rich API',
      'Use NumPy for numerical computing, homogeneous data, and performance-critical code',
      'Use Pandas for tabular data, data cleaning, exploratory analysis, and time series',
      'Typical workflow: load/clean with Pandas → convert to NumPy for ML → results back to Pandas',
    ],
  },
  {
    id: 'pandas-series-dataframes-dq-2',
    question:
      'Compare and contrast .loc, .iloc, and direct bracket indexing in Pandas. When would you use each, and what are common pitfalls?',
    sampleAnswer: `Pandas offers multiple indexing methods, each with specific use cases and potential pitfalls. Understanding when to use each is crucial for writing correct, maintainable code.

**The Three Indexing Methods:**

**1. Direct Bracket Indexing: df[...]**

\`\`\`python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie',],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 75000]
})

# Column selection
ages = df['Age',]  # Returns Series
subset = df[['Name', 'Age',]]  # Returns DataFrame

# Row selection (only with slices or boolean mask)
first_two = df[0:2]  # Rows 0-1
adults = df[df['Age',] >= 30]  # Boolean indexing
\`\`\`

**Use cases:**
- Quick column access
- Boolean filtering
- Slicing rows

**Limitations:**
- Cannot select both rows and columns simultaneously
- Cannot select single row by position (df[0] doesn't work)
- Ambiguous: df['Age',] is column, df[0:2] is rows!

**2. .loc: Label-Based Indexing**

\`\`\`python
# Set custom index
df = df.set_index('Name')

# Single row by label
alice = df.loc['Alice',]

# Multiple rows by label
subset = df.loc[['Alice', 'Charlie',]]

# Rows and columns by label
value = df.loc['Alice', 'Salary',]  # Single value
subset = df.loc['Alice':'Charlie', 'Age':'Salary',]  # Range

# Boolean indexing
high_earners = df.loc[df['Salary',] > 60000]

# Modify values
df.loc['Alice', 'Age',] = 26  # Change Alice\'s age
df.loc[df['Salary',] < 60000, 'Salary',] *= 1.1  # 10% raise
\`\`\`

**Key characteristic: Uses labels, includes endpoint in slices**
\`\`\`python
# 'Alice':'Charlie' includes both Alice AND Charlie
df.loc['Alice':'Charlie',]  # Both included!
\`\`\`

**Use cases:**
- Working with custom indices (dates, names, etc.)
- Clear, self-documenting code
- Conditional value assignment
- When you know the label names

**3. .iloc: Integer Position-Based Indexing**

\`\`\`python
# Single row by position
first = df.iloc[0]

# Multiple rows by position
first_two = df.iloc[0:2]  # Rows 0-1

# Rows and columns by position
value = df.iloc[0, 1]  # Row 0, Column 1
subset = df.iloc[0:2, 1:3]  # Rows 0-1, Columns 1-2

# Negative indexing
last_row = df.iloc[-1]
last_col = df.iloc[:, -1]
\`\`\`

**Key characteristic: Uses positions (integers), excludes endpoint in slices**
\`\`\`python
# df.iloc[0:2] includes positions 0 and 1 (NOT 2)
\`\`\`

**Use cases:**
- Working with positional indices
- Iterating through DataFrame
- When index labels don't matter
- Reproducible selection regardless of index

**Comparison Table:**

| Aspect | df[...] | .loc | .iloc |
|--------|---------|------|-------|
| **Syntax** | df['col',] or df[0:2] | df.loc[row, col] | df.iloc[pos, pos] |
| **Row selection** | Slice or boolean only | By label | By position |
| **Column selection** | By label | By label | By position |
| **Both rows & cols** | ❌ No | ✅ Yes | ✅ Yes |
| **Slice endpoint** | N/A (rows: exclusive) | Inclusive | Exclusive |
| **Boolean indexing** | ✅ Yes | ✅ Yes | ❌ No (not typical) |
| **Assignment** | ⚠️ Possible (warning) | ✅ Yes | ✅ Yes |

**Common Pitfalls:**

**Pitfall 1: SettingWithCopyWarning**
\`\`\`python
# BAD: Chained indexing
df[df['Age',] > 30]['Salary',] = 100000  # Warning!

# GOOD: Use .loc
df.loc[df['Age',] > 30, 'Salary',] = 100000  # No warning
\`\`\`

**Pitfall 2: Confusing .loc and .iloc**
\`\`\`python
df = pd.DataFrame({'A': [1, 2, 3]}, index=[10, 20, 30])

df.loc[10]   # Row with label 10 → [1]
df.iloc[10]  # Row at position 10 → IndexError!
\`\`\`

**Pitfall 3: Slice endpoint behavior**
\`\`\`python
df.loc['Alice':'Charlie',]  # Includes Charlie
df.iloc[0:2]               # Excludes position 2
\`\`\`

**Pitfall 4: Ambiguous bracket indexing**
\`\`\`python
df[0]       # Error! Can't select single row by position
df[0:1]     # Works! Returns DataFrame with row 0
df.iloc[0]  # Better: explicit and returns Series
\`\`\`

**Pitfall 5: Forgetting to copy**
\`\`\`python
subset = df[df['Age',] > 30]  # View or copy? Unclear!
subset.loc[:, 'Salary',] = 100000  # Might modify original df

# Better: explicit copy
subset = df[df['Age',] > 30].copy()
\`\`\`

**When to Use Each:**

**Use df[...] when:**
- Quick column access for exploration
- Boolean filtering for subset
- Simple, one-dimensional operations

**Use .loc when:**
- Working with meaningful indices (dates, IDs)
- Need to select both rows and columns
- Assigning values
- Want self-documenting code
- **Default choice for most operations**

**Use .iloc when:**
- Index labels don't matter
- Need position-based access (first N rows)
- Iterating through DataFrame
- Working with numerical positions

**Best Practices:**

1. **Prefer .loc for clarity**
\`\`\`python
# Instead of: value = df[df['Age',] > 30]['Salary',].iloc[0]
# Write: value = df.loc[df['Age',] > 30, 'Salary',].iloc[0]
\`\`\`

2. **Use .copy() to avoid warnings**
\`\`\`python
subset = df[df['Age',] > 30].copy()
\`\`\`

3. **Avoid chained indexing**
\`\`\`python
# BAD: df['A',][0] = 100
# GOOD: df.loc[0, 'A',] = 100
\`\`\`

4. **Use .at and .iat for single values** (faster)
\`\`\`python
value = df.at['Alice', 'Salary',]   # Faster than .loc
value = df.iat[0, 2]               # Faster than .iloc
\`\`\`

**Real-World Example:**

\`\`\`python
# Stock data with date index
df = pd.read_csv('stocks.csv', index_col='Date', parse_dates=True)

# .loc: Select by date label
jan_data = df.loc['2024-01-01':'2024-01-31',]

# .iloc: Get first and last day
first_day = df.iloc[0]
last_day = df.iloc[-1]

# Bracket: Quick column access
closing_prices = df['Close',]

# .loc: Conditional update
df.loc[df['Volume',] > 1_000_000, 'High_Volume',] = True
\`\`\`

Understanding these indexing methods and their nuances is essential for writing efficient, correct Pandas code!`,
    keyPoints: [
      'df[] for quick column access and boolean filtering, but limited functionality',
      '.loc uses label-based indexing, includes endpoints in slices, best for most operations',
      '.iloc uses integer position-based indexing, excludes endpoints like Python slices',
      'SettingWithCopyWarning occurs with chained indexing - always use .loc for assignment',
      'Use .loc for clarity and self-documenting code, .iloc when positions matter',
      'Use .at and .iat for single-value access (faster than .loc/.iloc)',
    ],
  },
  {
    id: 'pandas-series-dataframes-dq-3',
    question:
      'Explain the memory and performance implications of different data types (dtypes) in Pandas. How can choosing the right dtype significantly reduce memory usage?',
    sampleAnswer: `Data types (dtypes) in Pandas have profound impacts on memory usage and performance. Choosing appropriate dtypes can reduce memory by 50-90% for large datasets, making the difference between fitting in RAM or not.

**Default Dtype Behavior:**

Pandas often chooses conservative (memory-heavy) defaults:

\`\`\`python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve',],
    'department': ['IT', 'IT', 'HR', 'IT', 'Sales',],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 68000]
})

print(df.dtypes)
# id             int64     # 8 bytes per value
# name          object     # Variable, ~50+ bytes per string
# department    object     # Wasteful for repeated values
# age            int64     # 8 bytes (overkill for age 0-120)
# salary         int64     # 8 bytes

print(f"Memory: {df.memory_usage (deep=True).sum():,} bytes")
# Memory: 1,389 bytes
\`\`\`

**Dtype Sizes:**

| Dtype | Bytes | Range/Capacity |
|-------|-------|----------------|
| int8 | 1 | -128 to 127 |
| int16 | 2 | -32,768 to 32,767 |
| int32 | 4 | -2B to 2B |
| int64 | 8 | -9E18 to 9E18 |
| uint8 | 1 | 0 to 255 |
| uint16 | 2 | 0 to 65,535 |
| uint32 | 4 | 0 to 4B |
| float32 | 4 | ±3.4E38 (7 decimal digits) |
| float64 | 8 | ±1.7E308 (15 decimal digits) |
| bool | 1 | True/False |
| category | 1-4 | Limited unique values |
| object | Variable | Strings, mixed types |

**Optimization Strategy:**

\`\`\`python
# Optimize dtypes
df_optimized = df.copy()

# ID: values 1-5, use uint8 (0-255)
df_optimized['id',] = df_optimized['id',].astype (np.uint8)

# Age: values 25-35, use uint8 (0-255)
df_optimized['age',] = df_optimized['age',].astype (np.uint8)

# Salary: values < 2B, use uint32
df_optimized['salary',] = df_optimized['salary',].astype (np.uint32)

# Department: repeated values, use category
df_optimized['department',] = df_optimized['department',].astype('category')

# Name: keep as object (unique strings)

print(f"Original: {df.memory_usage (deep=True).sum():,} bytes")
print(f"Optimized: {df_optimized.memory_usage (deep=True).sum():,} bytes")
print(f"Reduction: {(1 - df_optimized.memory_usage (deep=True).sum() / df.memory_usage (deep=True).sum()) * 100:.1f}%")

# Typical results:
# Original: 1,389 bytes
# Optimized: 742 bytes
# Reduction: 46.6%
\`\`\`

**Category Dtype Deep Dive:**

Category is powerful for columns with repeated values:

\`\`\`python
# Large dataset with repeated departments
n = 1_000_000
df_large = pd.DataFrame({
    'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing',], n)
})

print(f"Object dtype: {df_large.memory_usage (deep=True).sum() / 1e6:.2f} MB")
# Object dtype: ~50-60 MB

df_large['department',] = df_large['department',].astype('category')
print(f"Category dtype: {df_large.memory_usage (deep=True).sum() / 1e6:.2f} MB")
# Category dtype: ~1-2 MB

# 95%+ memory reduction!
\`\`\`

**How Category Works:**

\`\`\`python
s = pd.Series(['IT', 'HR', 'IT', 'Sales', 'IT',], dtype='category')

# Under the hood:
# Categories: ['HR', 'IT', 'Sales',] (stored once)
# Codes: [1, 0, 1, 2, 1] (integers referencing categories)

print(s.cat.categories)  # Index(['HR', 'IT', 'Sales',], dtype='object')
print(s.cat.codes)       # [1, 0, 1, 2, 1]

# Memory: 5 integers + 3 strings
# vs. object: 5 full strings
\`\`\`

**Automatic Dtype Optimization:**

\`\`\`python
def optimize_dtypes (df):
    """Automatically optimize DataFrame dtypes"""
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        # Optimize integers
        if col_type == 'int64':
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            if c_min >= 0:  # Unsigned
                if c_max < 255:
                    optimized_df[col] = optimized_df[col].astype (np.uint8)
                elif c_max < 65535:
                    optimized_df[col] = optimized_df[col].astype (np.uint16)
                elif c_max < 4294967295:
                    optimized_df[col] = optimized_df[col].astype (np.uint32)
            else:  # Signed
                if c_min > -128 and c_max < 127:
                    optimized_df[col] = optimized_df[col].astype (np.int8)
                elif c_min > -32768 and c_max < 32767:
                    optimized_df[col] = optimized_df[col].astype (np.int16)
                elif c_min > -2147483648 and c_max < 2147483647:
                    optimized_df[col] = optimized_df[col].astype (np.int32)
        
        # Optimize floats
        elif col_type == 'float64':
            optimized_df[col] = optimized_df[col].astype (np.float32)
        
        # Optimize objects (potential categories)
        elif col_type == 'object':
            num_unique = optimized_df[col].nunique()
            num_total = len (optimized_df[col])
            
            # If less than 50% unique values, use category
            if num_unique / num_total < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df

# Usage
df_opt = optimize_dtypes (df)
\`\`\`

**Performance Implications:**

1. **Smaller dtypes = faster operations**
\`\`\`python
# int8 vs int64: 8x less data to move through CPU cache
arr_int8 = np.random.randint(0, 100, 10_000_000, dtype=np.int8)
arr_int64 = np.random.randint(0, 100, 10_000_000, dtype=np.int64)

%timeit arr_int8.sum()   # ~2ms
%timeit arr_int64.sum()  # ~5ms (2.5x slower)
\`\`\`

2. **Category enables faster groupby**
\`\`\`python
df['dept_obj',] = df['department',].astype('object')
df['dept_cat',] = df['department',].astype('category')

%timeit df.groupby('dept_obj')['salary',].mean()  # ~10ms
%timeit df.groupby('dept_cat')['salary',].mean()  # ~2ms (5x faster)
\`\`\`

3. **Float32 vs Float64**
\`\`\`python
# For ML, float32 is often sufficient
# Neural networks typically use float32
X = df[features].astype (np.float32).values  # 50% memory savings
\`\`\`

**When to Use Each Dtype:**

**Integers:**
- **uint8**: IDs, counts, ages (0-255)
- **uint16**: Prices in cents, small counts (0-65K)
- **uint32**: Large IDs, populations (0-4B)
- **int8/16/32**: When negative values possible

**Floats:**
- **float32**: ML features, most scientific computing
- **float64**: High-precision calculations, financial (pennies matter)

**Category:**
- Low cardinality strings (< 50% unique)
- Ordered data (S, M, L, XL)
- Repeated values (departments, countries)

**Object:**
- Unique strings (names, descriptions)
- Mixed types (last resort)
- Free text

**Datetime:**
- **datetime64[ns]**: Timestamps
- **timedelta64[ns]**: Time differences

**Bool:**
- True/False flags

**Real-World Impact:**

\`\`\`python
# 10 million row dataset
n = 10_000_000

df_raw = pd.DataFrame({
    'customer_id': np.arange (n),                    # int64: 76 MB
    'age': np.random.randint(18, 80, n),            # int64: 76 MB
    'country': np.random.choice(['USA', 'UK', 'CA', 'AU',], n),  # object: ~400 MB
    'amount': np.random.uniform(0, 1000, n)         # float64: 76 MB
})
print(f"Raw memory: {df_raw.memory_usage (deep=True).sum() / 1e6:.0f} MB")
# ~628 MB

df_opt = df_raw.copy()
df_opt['customer_id',] = df_opt['customer_id',].astype (np.uint32)  # 38 MB
df_opt['age',] = df_opt['age',].astype (np.uint8)                   # 10 MB
df_opt['country',] = df_opt['country',].astype('category')         # 10 MB
df_opt['amount',] = df_opt['amount',].astype (np.float32)           # 38 MB
print(f"Optimized memory: {df_opt.memory_usage (deep=True).sum() / 1e6:.0f} MB")
# ~96 MB

# 85% memory reduction! (628 MB → 96 MB)
# Difference between fitting in RAM or not
\`\`\`

**Best Practices:**

1. **Check memory usage regularly**
\`\`\`python
df.memory_usage (deep=True)
df.info (memory_usage='deep')
\`\`\`

2. **Optimize dtypes when loading data**
\`\`\`python
dtypes = {'customer_id': np.uint32, 'age': np.uint8}
df = pd.read_csv('data.csv', dtype=dtypes)
\`\`\`

3. **Use category for low cardinality**
- Rule of thumb: < 50% unique values

4. **Consider nullable integer dtypes** (pandas 1.0+)
\`\`\`python
df['age',] = df['age',].astype('Int8')  # Capital I = nullable
\`\`\`

5. **Profile before and after**

Understanding dtypes is crucial for working with large datasets. The right dtype choices can make your code 5-10x faster and use 80-90% less memory!`,
    keyPoints: [
      'Default int64/float64 dtypes use 8 bytes - often overkill for actual data ranges',
      'Downcast to smaller dtypes (int8, uint8, uint16, float32) for 50-90% memory reduction',
      'Category dtype reduces memory 10-100x for repeated strings by storing as integers + mapping',
      'Smaller dtypes = faster operations (less cache misses, less data movement)',
      'Optimize dtypes at load time or use automated optimization functions',
      'Use category when < 50% unique values; uint8 for 0-255; float32 for ML',
    ],
  },
];
