import { QuizQuestion } from '../../../types';

export const mergingjoiningdataQuiz: QuizQuestion[] = [
  {
    id: 'merging-joining-data-dq-1',
    question:
      'Explain the four types of joins (inner, left, right, outer) in Pandas merge operations. For each type, provide a realistic scenario where it would be the appropriate choice, and discuss the implications of choosing the wrong join type.',
    sampleAnswer: `Understanding join types is crucial for data integration. Each join type keeps different subsets of data, and choosing incorrectly can lead to lost information or misleading analysis.

**The Four Join Types:**

**1. Inner Join (how='inner')**

**What it does:**
- Keeps only rows with matching keys in BOTH DataFrames
- Rows without matches are discarded from both sides
- Result size ≤ min(len(left), len(right))

**SQL equivalent:**
\`\`\`sql
SELECT * FROM left INNER JOIN right ON left.key = right.key
\`\`\`

**Pandas example:**
\`\`\`python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David',],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 40],
    'dept_name': ['IT', 'HR', 'Marketing',]
})

inner = pd.merge(employees, departments, on='dept_id', how='inner')
print(inner)
#    emp_id     name  dept_id dept_name
# 0       1    Alice       10        IT
# 1       3  Charlie       10        IT
# 2       2      Bob       20        HR
# David (dept 30) is excluded - no matching department
# Marketing (dept 40) is excluded - no employees
\`\`\`

**When to use:**
- **Transaction matching**: Match orders to customers where both exist
- **Data quality**: Only process records with complete information
- **Performance analysis**: Compare actual vs target where both are recorded
- **A/B test analysis**: Only users who completed both pre and post surveys

**Real-world scenario:**
\`\`\`python
# E-commerce: Analyze only completed transactions
orders = pd.DataFrame({...})  # All orders
payments = pd.DataFrame({...})  # Payment records

# Only analyze orders that have been paid
completed = pd.merge(orders, payments, on='order_id', how='inner')
# Excludes pending/cancelled orders
\`\`\`

**Wrong choice consequences:**
- If you need to track unpaid orders, inner join loses that data
- Missing data might indicate problems (high cart abandonment)

**2. Left Join (how='left')**

**What it does:**
- Keeps ALL rows from left DataFrame
- Matches from right where possible
- Fills with NaN where no match in right
- Result size = len(left)

**SQL equivalent:**
\`\`\`sql
SELECT * FROM left LEFT JOIN right ON left.key = right.key
\`\`\`

**Pandas example:**
\`\`\`python
left = pd.merge(employees, departments, on='dept_id', how='left')
print(left)
#    emp_id     name  dept_id dept_name
# 0       1    Alice       10        IT
# 1       2      Bob       20        HR
# 2       3  Charlie       10        IT
# 3       4    David       30       NaN  # No matching department
# All employees kept, department name missing for David
\`\`\`

**When to use:**
- **Primary dataset must be complete**: Customer list must stay complete
- **Optional enrichment**: Add extra info where available
- **Missing data analysis**: Identify which records lack supplementary data
- **Audit trails**: Keep all transactions, add payment info where exists

**Real-world scenario:**
\`\`\`python
# Customer segmentation: All customers, add purchase history
customers = pd.DataFrame({...})  # All registered customers
purchases = pd.DataFrame({...})  # Purchase history

# Keep all customers, show who has purchased
enriched = pd.merge(customers, purchases, on='customer_id', how='left')
# Can identify customers who haven't purchased (NaN in purchase columns)
# Then target them with promotions
\`\`\`

**Wrong choice consequences:**
- Using inner join would lose customers without purchases
- Can't calculate conversion rate (registered → purchased)
- Miss opportunity to re-engage inactive customers

**3. Right Join (how='right')**

**What it does:**
- Keeps ALL rows from right DataFrame
- Matches from left where possible
- Fills with NaN where no match in left
- Result size = len(right)

**Note:** Right join is just left join with DataFrames swapped. Rarely used because you can reorder and use left join instead.

**Pandas example:**
\`\`\`python
right = pd.merge(employees, departments, on='dept_id', how='right')
print(right)
#    emp_id     name  dept_id   dept_name
# 0       1    Alice       10          IT
# 1       3  Charlie       10          IT
# 2       2      Bob       20          HR
# 3     NaN      NaN       40   Marketing  # No employees
# All departments kept, even without employees
\`\`\`

**When to use:**
- **Reference data must be complete**: All products must appear
- **Compliance**: Show all required categories even if no data
- **Gap analysis**: Identify unused or empty categories

**Real-world scenario:**
\`\`\`python
# Product catalog: Show all products, even with zero sales
products = pd.DataFrame({...})  # Full product catalog
sales = pd.DataFrame({...})  # Sales records

# Show all products, identify which haven't sold
catalog_performance = pd.merge(sales, products, on='product_id', how='right')
# NaN in sales columns = product hasn't sold
# Identify products to discontinue or promote
\`\`\`

**Practical tip:** Most people use left join instead:
\`\`\`python
# These are equivalent:
right = pd.merge(df1, df2, on='key', how='right')
left = pd.merge(df2, df1, on='key', how='left')
# Prefer left join for consistency
\`\`\`

**4. Outer Join (how='outer') - Full Outer Join**

**What it does:**
- Keeps ALL rows from BOTH DataFrames
- Matches where possible
- Fills with NaN where no match on either side
- Result size ≥ max(len(left), len(right))

**SQL equivalent:**
\`\`\`sql
SELECT * FROM left FULL OUTER JOIN right ON left.key = right.key
\`\`\`

**Pandas example:**
\`\`\`python
outer = pd.merge(employees, departments, on='dept_id', how='outer')
print(outer)
#    emp_id     name  dept_id   dept_name
# 0       1    Alice       10          IT
# 1       2      Bob       20          HR
# 2       3  Charlie       10          IT
# 3       4    David       30         NaN  # Employee without department
# 4     NaN      NaN       40   Marketing  # Department without employees
# Both orphaned records kept
\`\`\`

**When to use:**
- **Data reconciliation**: Find mismatches between systems
- **Audit and compliance**: Identify orphaned records
- **Data quality**: Detect referential integrity issues
- **Migration validation**: Ensure all data transferred

**Real-world scenario:**
\`\`\`python
# System migration: Validate data transfer
old_system = pd.DataFrame({...})  # Records from old system
new_system = pd.DataFrame({...})  # Records from new system

# Find records only in old (not migrated) or only in new (unexpected)
reconciliation = pd.merge(
    old_system,
    new_system,
    on='record_id',
    how='outer',
    indicator=True,
    suffixes=('_old', '_new')
)

# Analyze _merge column
missing_from_new = reconciliation[reconciliation['_merge',] == 'left_only',]
unexpected_in_new = reconciliation[reconciliation['_merge',] == 'right_only',]
successfully_migrated = reconciliation[reconciliation['_merge',] == 'both',]

print(f"Not migrated: {len(missing_from_new)}")
print(f"Unexpected: {len(unexpected_in_new)}")
print(f"Success rate: {len(successfully_migrated) / len(old_system) * 100:.1f}%")
\`\`\`

**Wrong choice consequences:**
- Using inner join would hide migration failures
- Can't identify which records were lost
- Can't detect unexpected records in new system

**Comparison Table:**

| Join Type | Left Rows | Right Rows | Use Case | Result Size |
|-----------|-----------|------------|----------|-------------|
| **Inner** | Only matched | Only matched | Complete pairs only | ≤ min(L, R) |
| **Left** | All | Only matched | Keep all left | = L |
| **Right** | Only matched | All | Keep all right | = R |
| **Outer** | All | All | Keep everything | ≥ max(L, R) |

**Decision Framework:**

\`\`\`
Question: Which dataset is the "source of truth"?
│
├─ Left DataFrame is primary
│  └─ Use LEFT join
│
├─ Right DataFrame is primary
│  └─ Use RIGHT join (or swap and use LEFT)
│
├─ Both are equally important
│  ├─ Need only complete pairs → Use INNER join
│  └─ Need to keep all records → Use OUTER join
│
└─ Validation/Audit scenario
   └─ Use OUTER join with indicator=True
\`\`\`

**Common Mistakes:**

**Mistake 1: Using inner join by default**
\`\`\`python
# BAD: Loses customers without orders
result = pd.merge(customers, orders, on='customer_id')  # Default is inner

# GOOD: Keeps all customers
result = pd.merge(customers, orders, on='customer_id', how='left')
\`\`\`

**Mistake 2: Not checking the result size**
\`\`\`python
left = pd.DataFrame({'key': [1, 2, 3], 'val': [10, 20, 30]})
right = pd.DataFrame({'key': [1, 1, 2, 2], 'val': [100, 200, 300, 400]})

result = pd.merge(left, right, on='key')
print(len(result))  # 4 rows! (2 matches for key=1, 2 for key=2)
# Many-to-many join created more rows than either input
\`\`\`

**Mistake 3: Ignoring NaN values**
\`\`\`python
# After left join
result = pd.merge(df1, df2, on='key', how='left')

# BAD: Treating NaN as 0 without checking
result['value',] = result['value',].fillna(0)

# GOOD: Understand why NaN exists
print(f"Unmatched records: {result['value',].isna().sum()}")
# Investigate why these records don't match
\`\`\`

**Best Practices:**

1. **Always specify how parameter explicitly**
   \`\`\`python
   # Good: Clear intent
   result = pd.merge(df1, df2, on='key', how='left')
   
   # Bad: Implicit inner join
   result = pd.merge(df1, df2, on='key')
   \`\`\`

2. **Use indicator for validation**
   \`\`\`python
   result = pd.merge(df1, df2, on='key', how='outer', indicator=True)
   print(result['_merge',].value_counts())
   \`\`\`

3. **Check result size**
   \`\`\`python
   print(f"Left: {len(df1)}, Right: {len(df2)}, Result: {len(result)}")
   \`\`\`

4. **Document your choice**
   \`\`\`python
   # Left join: Keep all customers even without purchases
   # to calculate conversion rate
   result = pd.merge(customers, purchases, on='id', how='left')
   \`\`\`

**Key Takeaway:**

Join type determines which records survive the merge:
- **Inner**: Only matches (strictest)
- **Left/Right**: One side complete (most common)
- **Outer**: Everything (audit/validation)

Choose based on business logic, not convenience. The wrong join type can silently drop important data or create misleading results!`,
    keyPoints: [
      'Inner join: only matching keys (intersection) - safest default',
      'Left join: all left rows + matches from right - preserve main dataset',
      'Right join: all right rows + matches from left - rarely used',
      'Outer join: all rows from both (union) - risk of large result',
      'Wrong join type causes data loss or duplication - verify row counts',
    ],
  },
  {
    id: 'merging-joining-data-dq-2',
    question:
      'Discuss the performance implications of different merge strategies in Pandas. When would you use merge() vs join() vs concat()? How do indexing, sorting, and data size affect merge performance?',
    sampleAnswer: `Merge performance can vary dramatically based on the strategy used, data characteristics, and implementation details. Understanding these factors is crucial for handling large datasets efficiently.

**Three Main Merging Methods:**

**1. pd.merge() - Most Flexible**

**How it works:**
- Hash-based join for unsorted data
- Sort-merge join for sorted data
- Can merge on any columns

\`\`\`python
result = pd.merge(df1, df2, on='key', how='inner')
\`\`\`

**Performance characteristics:**
- **Time complexity**: O(n + m) average for hash join
- **Memory**: Builds hash table, memory-intensive
- **Best for**: Arbitrary column joins, small to medium data

**2. DataFrame.join() - Index-Based**

**How it works:**
- Optimized for index-based joining
- Uses index lookups (O(1) average)
- Left join by default

\`\`\`python
df1 = df1.set_index('key')
df2 = df2.set_index('key')
result = df1.join(df2)
\`\`\`

**Performance characteristics:**
- **Time complexity**: O(n) when indices are unique
- **Memory**: More efficient than merge
- **Best for**: Index-based joins, when indices are already set

**3. pd.concat() - Stacking**

**How it works:**
- Simple concatenation, no key matching
- Vertical (axis=0) or horizontal (axis=1)

\`\`\`python
result = pd.concat([df1, df2], axis=0, ignore_index=True)
\`\`\`

**Performance characteristics:**
- **Time complexity**: O(n + m) simple stacking
- **Memory**: Most efficient (no key lookup)
- **Best for**: Stacking DataFrames with same structure

**Performance Comparison:**

\`\`\`python
import pandas as pd
import numpy as np
import time

# Create test data
n = 1_000_000
df1 = pd.DataFrame({
    'key': range(n),
    'value1': np.random.randn(n)
})

df2 = pd.DataFrame({
    'key': range(n),
    'value2': np.random.randn(n)
})

# Method 1: merge() on column
start = time.time()
result1 = pd.merge(df1, df2, on='key')
time1 = time.time() - start
print(f"merge() on column: {time1:.3f}s")

# Method 2: merge() with indexed data
df1_idx = df1.set_index('key')
df2_idx = df2.set_index('key')
start = time.time()
result2 = pd.merge(df1_idx, df2_idx, left_index=True, right_index=True)
time2 = time.time() - start
print(f"merge() on index: {time2:.3f}s")

# Method 3: join() with indexed data
start = time.time()
result3 = df1_idx.join(df2_idx)
time3 = time.time() - start
print(f"join() on index: {time3:.3f}s")

# Method 4: concat() horizontal
start = time.time()
result4 = pd.concat([df1_idx, df2_idx], axis=1)
time4 = time.time() - start
print(f"concat() horizontal: {time4:.3f}s")

# Typical results:
# merge() on column: 0.542s
# merge() on index: 0.198s (2.7x faster)
# join() on index: 0.156s (3.5x faster)
# concat() horizontal: 0.089s (6.1x faster) - but doesn't match keys!
\`\`\`

**Key Insight:** Indexing dramatically improves performance!

**Indexing Impact:**

**Without index:**
\`\`\`python
# Slow: Must build hash table on the fly
df1 = pd.DataFrame({'key': [...], 'val1': [...]})
df2 = pd.DataFrame({'key': [...], 'val2': [...]})
result = pd.merge(df1, df2, on='key')
# O(n + m) with hash table overhead
\`\`\`

**With index:**
\`\`\`python
# Fast: Direct index lookup
df1 = df1.set_index('key')
df2 = df2.set_index('key')
result = df1.join(df2)
# O(n) with O(1) lookups
\`\`\`

**Why indexing matters:**
- Hash table construction is expensive
- Index already provides O(1) lookups
- Memory locality benefits

**Sorting Impact:**

Pandas can use sort-merge join for sorted data:

\`\`\`python
# Unsorted data
df1 = pd.DataFrame({
    'key': np.random.randint(0, 1000, 1_000_000),
    'val': np.random.randn(1_000_000)
})
df2 = pd.DataFrame({
    'key': np.random.randint(0, 1000, 1_000_000),
    'val': np.random.randn(1_000_000)
})

# Method 1: Merge unsorted (hash join)
start = time.time()
result1 = pd.merge(df1, df2, on='key')
time1 = time.time() - start

# Method 2: Sort then merge (sort-merge join)
df1_sorted = df1.sort_values('key')
df2_sorted = df2.sort_values('key')
start = time.time()
result2 = pd.merge(df1_sorted, df2_sorted, on='key')
time2 = time.time() - start

print(f"Unsorted: {time1:.3f}s")
print(f"Sorted: {time2:.3f}s")

# Results vary:
# - If already sorted: sort-merge is faster
# - If needs sorting: hash join usually faster (sorting overhead)
\`\`\`

**merge_asof() for Time Series:**

For time series with inexact matches:

\`\`\`python
# Regular merge (exact match required)
start = time.time()
result1 = pd.merge(trades, quotes, on='timestamp')
time1 = time.time() - start

# merge_asof (nearest match)
trades_sorted = trades.sort_values('timestamp')
quotes_sorted = quotes.sort_values('timestamp')
start = time.time()
result2 = pd.merge_asof(trades_sorted, quotes_sorted, on='timestamp')
time2 = time.time() - start

print(f"merge(): {time1:.3f}s")
print(f"merge_asof(): {time2:.3f}s")

# merge_asof typically 10-100x faster for time series
# Assumes sorted data, uses binary search
\`\`\`

**Memory Considerations:**

\`\`\`python
# Memory-intensive: Creates hash table
result = pd.merge(large_df1, large_df2, on='key')

# More memory efficient: Index-based
df1 = df1.set_index('key')
df2 = df2.set_index('key')
result = df1.join(df2)

# Most memory efficient: Chunked processing
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=100000):
    merged_chunk = pd.merge(chunk, reference_df, on='key')
    chunks.append(merged_chunk)
result = pd.concat(chunks, ignore_index=True)
\`\`\`

**Categorical Data:**

Using categorical for repeated keys saves memory and speeds up merges:

\`\`\`python
# Without categorical
df1 = pd.DataFrame({
    'category': ['A', 'B', 'C',] * 1_000_000,  # 3M repeated strings
    'value': np.random.randn(3_000_000)
})

# With categorical
df1['category',] = df1['category',].astype('category')

# Memory savings: 80-90% for repeated values
# Merge speedup: 2-3x faster

print(f"Memory without categorical: {df1.memory_usage(deep=True).sum() / 1e6:.1f} MB")
df1['category',] = df1['category',].astype('category')
print(f"Memory with categorical: {df1.memory_usage(deep=True).sum() / 1e6:.1f} MB")
\`\`\`

**Optimization Strategies:**

**1. Pre-index your data**
\`\`\`python
# Good: Set index once, reuse
df1 = df1.set_index('key')
df2 = df2.set_index('key')
result1 = df1.join(df2)
result2 = df1.join(df3)  # Reuse index

# Bad: Set index repeatedly
result1 = pd.merge(df1.set_index('key'), df2.set_index('key'), left_index=True, right_index=True)
result2 = pd.merge(df1.set_index('key'), df3.set_index('key'), left_index=True, right_index=True)
\`\`\`

**2. Filter before merging**
\`\`\`python
# Bad: Merge then filter
result = pd.merge(large_df1, large_df2, on='key')
result = result[result['date',] > '2024-01-01',]

# Good: Filter then merge
df1_filtered = large_df1[large_df1['date',] > '2024-01-01',]
result = pd.merge(df1_filtered, large_df2, on='key')
\`\`\`

**3. Use appropriate join type**
\`\`\`python
# Bad: Outer join when you need inner
result = pd.merge(df1, df2, on='key', how='outer')  # More data to process

# Good: Inner join if you only need matches
result = pd.merge(df1, df2, on='key', how='inner')  # Less data
\`\`\`

**4. Vectorize multiple merges**
\`\`\`python
# Bad: Sequential merges
result = df1.copy()
for df in [df2, df3, df4]:
    result = pd.merge(result, df, on='key')

# Good: Reduce pattern (if possible)
from functools import reduce
dfs = [df1, df2, df3, df4]
result = reduce(lambda left, right: pd.merge(left, right, on='key'), dfs)
\`\`\`

**5. Consider alternatives for large data**
\`\`\`python
# For very large data, consider:
# - Dask for distributed computing
# - SQL database with indexed tables
# - Apache Spark
# - Chunked processing

# Example: Dask
import dask.dataframe as dd
ddf1 = dd.from_pandas(df1, npartitions=10)
ddf2 = dd.from_pandas(df2, npartitions=10)
result = ddf1.merge(ddf2, on='key').compute()
\`\`\`

**Decision Tree:**

\`\`\`
Need to combine DataFrames?
│
├─ Same structure, no key matching?
│  └─ Use concat() (fastest)
│
├─ Joining on index?
│  ├─ Index already set? → Use join()
│  └─ Need to set index? → Consider cost vs benefit
│
├─ Time series with inexact matches?
│  └─ Use merge_asof()
│
└─ General key-based merge?
   ├─ Small data (< 1M rows) → merge() is fine
   ├─ Large data + can index → Set index, use join()
   ├─ Large data + can't index → Consider chunking
   └─ Very large data → Consider Dask/Spark
\`\`\`

**Real-World Example:**

\`\`\`python
# Financial data: Join trades with market data

# Scenario: 10M trades, 100K market data points
trades = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=10_000_000, freq='s'),
    'ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT',], 10_000_000),
    'quantity': np.random.randint(1, 1000, 10_000_000)
})

market_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100_000, freq='min'),
    'ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT',], 100_000),
    'price': np.random.uniform(100, 200, 100_000)
})

# SLOW: Regular merge (requires exact timestamp match)
# Most trades won't have exact market data timestamp
start = time.time()
result_slow = pd.merge(trades, market_data, on=['timestamp', 'ticker',])
time_slow = time.time() - start
print(f"Regular merge: {time_slow:.3f}s, {len(result_slow)} rows")

# FAST: merge_asof (match to nearest prior market data)
trades_sorted = trades.sort_values(['ticker', 'timestamp',])
market_sorted = market_data.sort_values(['ticker', 'timestamp',])
start = time.time()
result_fast = pd.merge_asof(
    trades_sorted, 
    market_sorted, 
    on='timestamp',
    by='ticker',  # Match within same ticker
    direction='backward'  # Use most recent market price
)
time_fast = time.time() - start
print(f"merge_asof: {time_fast:.3f}s, {len(result_fast)} rows")

# Typical result:
# Regular merge: 125.3s, 127 rows (very few exact matches!)
# merge_asof: 8.7s, 10,000,000 rows (all trades matched!)
# 14x faster + correct semantics!
\`\`\`

**Key Takeaways:**

1. **Index before joining** when possible (2-5x speedup)
2. **Use join() for index-based** merges (faster than merge())
3. **Use concat() for simple stacking** (no key lookup needed)
4. **Use merge_asof() for time series** (10-100x faster than regular merge)
5. **Filter before merging** to reduce data size
6. **Use categorical** for repeated values
7. **Consider chunking** for very large datasets
8. **Profile your code** - actual performance depends on data characteristics

The right choice can mean the difference between minutes and seconds!`,
    keyPoints: [
      'Merge combines on column values (SQL-like joins)',
      'Join combines on index values - faster for index-aligned data',
      'Concat stacks DataFrames vertically (rows) or horizontally (columns)',
      'validate parameter detects many-to-many joins and cardinality issues',
      'indicator=True adds _merge column showing source of each row',
    ],
  },
  {
    id: 'merging-joining-data-dq-3',
    question:
      'Describe common data quality issues that arise during merges (duplicate keys, missing values, type mismatches) and provide a comprehensive strategy for validating merge operations before, during, and after merging.',
    sampleAnswer: `Data quality issues during merges can silently corrupt your analysis. A comprehensive validation strategy is essential for reliable data integration.

**Common Data Quality Issues:**

**1. Duplicate Keys**

**Problem:**
- One-to-many or many-to-many relationships
- Result has more rows than expected
- Duplicated information

\`\`\`python
# Example problem
customers = pd.DataFrame({
    'customer_id': [1, 1, 2, 3],  # Duplicate key!
    'name': ['Alice', 'Alice Smith', 'Bob', 'Charlie',],
    'city': ['NYC', 'NYC', 'LA', 'Chicago',]
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'customer_id': [1, 2, 3],
    'amount': [100, 200, 150]
})

# Merge creates duplicate rows
result = pd.merge(orders, customers, on='customer_id')
print(len(result))  # 4 rows instead of 3!
# Order 101 matched both Alice entries
\`\`\`

**2. Missing Values in Keys**

**Problem:**
- NaN in merge keys
- Unexpected joins or exclusions

\`\`\`python
df1 = pd.DataFrame({
    'key': [1, 2, None, 4],
    'value1': [10, 20, 30, 40]
})

df2 = pd.DataFrame({
    'key': [1, 2, 3, None],
    'value2': [100, 200, 300, 400]
})

result = pd.merge(df1, df2, on='key', how='inner')
print(len(result))  # 2 rows - NaN keys don't match!
\`\`\`

**3. Type Mismatches**

**Problem:**
- Same logical key, different types
- No matches due to type incompatibility

\`\`\`python
df1 = pd.DataFrame({
    'id': ['1', '2', '3',],  # String
    'value1': [10, 20, 30]
})

df2 = pd.DataFrame({
    'id': [1, 2, 3],  # Integer
    'value2': [100, 200, 300]
})

result = pd.merge(df1, df2, on='id')
print(len(result))  # 0 rows! Types don't match
\`\`\`

**4. Case Sensitivity**

**Problem:**
- String keys with different cases
- Unexpected non-matches

\`\`\`python
df1 = pd.DataFrame({
    'name': ['Alice', 'Bob', 'CHARLIE',],
    'value1': [10, 20, 30]
})

df2 = pd.DataFrame({
    'name': ['alice', 'BOB', 'Charlie',],
    'value2': [100, 200, 300]
})

result = pd.merge(df1, df2, on='name')
print(len(result))  # 0 rows! Case mismatch
\`\`\`

**Comprehensive Validation Strategy:**

**Phase 1: Pre-Merge Validation**

\`\`\`python
def validate_before_merge(df1, df2, merge_keys, merge_type='inner'):
    """
    Comprehensive pre-merge validation
    Returns: dict of validation results and warnings
    """
    issues = {
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # 1. Check if merge keys exist
    for key in merge_keys:
        if key not in df1.columns:
            issues['errors',].append(f"Key '{key}' not found in left DataFrame")
        if key not in df2.columns:
            issues['errors',].append(f"Key '{key}' not found in right DataFrame")
    
    if issues['errors',]:
        return issues
    
    # 2. Check for missing values in keys
    for key in merge_keys:
        null_left = df1[key].isnull().sum()
        null_right = df2[key].isnull().sum()
        if null_left > 0:
            issues['warnings',].append(
                f"Left DataFrame has {null_left} null values in '{key}' "
                f"({null_left/len(df1)*100:.1f}%)"
            )
        if null_right > 0:
            issues['warnings',].append(
                f"Right DataFrame has {null_right} null values in '{key}' "
                f"({null_right/len(df2)*100:.1f}%)"
            )
    
    # 3. Check data types
    for key in merge_keys:
        type_left = df1[key].dtype
        type_right = df2[key].dtype
        if type_left != type_right:
            issues['warnings',].append(
                f"Type mismatch in '{key}': left={type_left}, right={type_right}"
            )
    
    # 4. Check for duplicates
    for key in merge_keys:
        dup_left = df1[key].duplicated().sum()
        dup_right = df2[key].duplicated().sum()
        if dup_left > 0:
            issues['warnings',].append(
                f"Left DataFrame has {dup_left} duplicate keys in '{key}'"
            )
        if dup_right > 0:
            issues['warnings',].append(
                f"Right DataFrame has {dup_right} duplicate keys in '{key}'"
            )
    
    # 5. Check key overlap
    if len(merge_keys) == 1:
        key = merge_keys[0]
        left_values = set(df1[key].dropna().unique())
        right_values = set(df2[key].dropna().unique())
        
        overlap = left_values & right_values
        left_only = left_values - right_values
        right_only = right_values - left_values
        
        issues['info',].append(f"Key overlap: {len(overlap)} values")
        issues['info',].append(f"Left only: {len(left_only)} values")
        issues['info',].append(f"Right only: {len(right_only)} values")
        
        if merge_type == 'inner' and len(overlap) == 0:
            issues['errors',].append("Inner join will return 0 rows (no overlap)")
    
    # 6. Estimate result size
    if len(merge_keys) == 1:
        key = merge_keys[0]
        # Rough estimate for many-to-many
        left_dups = df1[key].value_counts().max()
        right_dups = df2[key].value_counts().max()
        max_result_size = len(df1) * right_dups  # Worst case
        
        if max_result_size > len(df1) * 1.5:
            issues['warnings',].append(
                f"Potential many-to-many join: result could have up to {max_result_size} rows "
                f"(input: {len(df1)} rows)"
            )
    
    # 7. Check for string case issues
    for key in merge_keys:
        if df1[key].dtype == 'object' and df2[key].dtype == 'object':
            # Sample check
            left_sample = df1[key].dropna().head(100)
            right_sample = df2[key].dropna().head(100)
            if any(s != s.lower() for s in left_sample) or \
               any(s != s.lower() for s in right_sample):
                issues['warnings',].append(
                    f"Mixed case detected in '{key}' - consider normalizing"
                )
    
    return issues

# Usage
issues = validate_before_merge(customers, orders, ['customer_id',], 'left')

if issues['errors',]:
    print("ERRORS (must fix):")
    for error in issues['errors',]:
        print(f"  ❌ {error}")

if issues['warnings',]:
    print("\\nWARNINGS (review):")
    for warning in issues['warnings',]:
        print(f"  ⚠️  {warning}")

if issues['info',]:
    print("\\nINFO:")
    for info in issues['info',]:
        print(f"  ℹ️  {info}")
\`\`\`

**Phase 2: During Merge - Use Indicator**

\`\`\`python
# Always use indicator for validation
result = pd.merge(
    df1,
    df2,
    on='key',
    how='outer',  # Use outer to see all records
    indicator=True,
    validate='one_to_one'  # Add validation
)

# Check merge distribution
print("\\nMerge distribution:")
print(result['_merge',].value_counts())
print(f"Match rate: {(result['_merge',] == 'both').sum() / len(result) * 100:.1f}%")
\`\`\`

**Phase 3: Post-Merge Validation**

\`\`\`python
def validate_after_merge(result, df1, df2, merge_keys, expected_type='inner'):
    """
    Validate merge results
    """
    issues = []
    
    # 1. Check result size
    print(f"Input sizes: left={len(df1)}, right={len(df2)}")
    print(f"Result size: {len(result)}")
    
    size_change = len(result) / len(df1) if len(df1) > 0 else 0
    if size_change > 1.5:
        issues.append(f"⚠️  Result {size_change:.1f}x larger than left input")
    
    # 2. Check for unexpected nulls
    for col in result.columns:
        if col not in df1.columns and col not in df2.columns:
            continue  # Skip indicator column
        null_count = result[col].isnull().sum()
        if null_count > 0:
            null_pct = null_count / len(result) * 100
            if null_pct > 10:
                issues.append(
                    f"⚠️  Column '{col}' has {null_pct:.1f}% null values"
                )
    
    # 3. Check merge completeness (if indicator exists)
    if '_merge' in result.columns:
        left_only = (result['_merge',] == 'left_only').sum()
        right_only = (result['_merge',] == 'right_only').sum()
        both = (result['_merge',] == 'both').sum()
        
        print(f"\\nMerge breakdown:")
        print(f"  Both: {both} ({both/len(result)*100:.1f}%)")
        print(f"  Left only: {left_only} ({left_only/len(result)*100:.1f}%)")
        print(f"  Right only: {right_only} ({right_only/len(result)*100:.1f}%)")
        
        if expected_type == 'inner' and (left_only > 0 or right_only > 0):
            issues.append("⚠️  Inner join expected but unmatched records found")
    
    # 4. Check for duplicate keys in result
    dup_count = result.duplicated(subset=merge_keys).sum()
    if dup_count > 0:
        issues.append(f"⚠️  {dup_count} duplicate keys in result")
    
    # 5. Sample validation
    print("\\nSample result:")
    print(result.head())
    
    return issues

# Usage
issues = validate_after_merge(result, df1, df2, ['key',], 'left')
if issues:
    print("\\nIssues found:")
    for issue in issues:
        print(issue)
\`\`\`

**Complete Validation Workflow:**

\`\`\`python
def safe_merge(df1, df2, merge_keys, how='inner', validate='one_to_one'):
    """
    Merge with comprehensive validation
    """
    print("="*60)
    print("MERGE VALIDATION REPORT")
    print("="*60)
    
    # Phase 1: Pre-merge validation
    print("\\n1. PRE-MERGE VALIDATION")
    print("-"*60)
    pre_issues = validate_before_merge(df1, df2, merge_keys, how)
    
    if pre_issues['errors',]:
        print("\\nERRORS found - cannot proceed:")
        for error in pre_issues['errors',]:
            print(f"  ❌ {error}")
        return None
    
    if pre_issues['warnings',]:
        print("\\nWARNINGS:")
        for warning in pre_issues['warnings',]:
            print(f"  ⚠️  {warning}")
    
    # Phase 2: Perform merge with indicator
    print("\\n2. PERFORMING MERGE")
    print("-"*60)
    try:
        result = pd.merge(
            df1,
            df2,
            on=merge_keys if isinstance(merge_keys, list) else [merge_keys],
            how=how,
            indicator=True,
            validate=validate  # Validates cardinality
        )
        print("✅ Merge completed successfully")
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        return None
    
    # Phase 3: Post-merge validation
    print("\\n3. POST-MERGE VALIDATION")
    print("-"*60)
    post_issues = validate_after_merge(result, df1, df2, merge_keys, how)
    
    if post_issues:
        print("\\nPost-merge issues:")
        for issue in post_issues:
            print(issue)
    else:
        print("✅ No issues detected")
    
    print("\\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    return result

# Usage
result = safe_merge(customers, orders, 'customer_id', how='left', validate='one_to_many')
\`\`\`

**Fixing Common Issues:**

**1. Fix duplicate keys:**
\`\`\`python
# Identify duplicates
duplicates = df[df.duplicated(subset=['key',], keep=False)]
print(f"Duplicate keys:\\n{duplicates}")

# Strategy A: Keep first occurrence
df_dedup = df.drop_duplicates(subset=['key',], keep='first')

# Strategy B: Aggregate duplicates
df_agg = df.groupby('key').agg({
    'value': 'sum',
    'count': 'size'
}).reset_index()

# Strategy C: Investigate and fix data source
\`\`\`

**2. Fix type mismatches:**
\`\`\`python
# Convert types before merge
df1['id',] = df1['id',].astype(str)
df2['id',] = df2['id',].astype(str)
result = pd.merge(df1, df2, on='id')
\`\`\`

**3. Fix case sensitivity:**
\`\`\`python
# Normalize strings before merge
df1['name',] = df1['name',].str.lower().str.strip()
df2['name',] = df2['name',].str.lower().str.strip()
result = pd.merge(df1, df2, on='name')
\`\`\`

**4. Handle missing values:**
\`\`\`python
# Option A: Drop nulls before merge
df1_clean = df1.dropna(subset=['key',])
df2_clean = df2.dropna(subset=['key',])
result = pd.merge(df1_clean, df2_clean, on='key')

# Option B: Fill nulls with placeholder
df1['key',].fillna('UNKNOWN', inplace=True)
df2['key',].fillna('UNKNOWN', inplace=True)
result = pd.merge(df1, df2, on='key')
\`\`\`

**Best Practices:**

1. **Always validate before merging** on production data
2. **Use indicator=True** for outer joins
3. **Use validate parameter** to catch cardinality issues
4. **Check result size** - unexpected growth indicates problems
5. **Sample inspect** results after merging
6. **Document expected behavior** in code
7. **Create unit tests** for merge operations
8. **Monitor merge quality** in production pipelines

**Key Takeaway:**

Data quality issues during merges are common and can silently corrupt analysis. A systematic validation strategy—before, during, and after merging—is essential for reliable data integration. Invest time in validation; it pays off in correct results and fewer debugging sessions!`,
    keyPoints: [
      'Many-to-many joins multiply rows - can create unexpectedly large results',
      'Use validate="one_to_many" or "many_to_one" to catch accidental m:m joins',
      'Check for duplicate keys before merging with duplicated()',
      'Prevent m:m by deduplicating or grouping before merge',
      'Monitor memory usage and row counts when joining large datasets',
    ],
  },
];
