import { QuizQuestion } from '../../../types';

export const dataaggregationgroupingQuiz: QuizQuestion[] = [
  {
    id: 'data-aggregation-grouping-dq-1',
    question:
      'Explain the split-apply-combine pattern in detail. Provide examples of when you would use .agg(), .transform(), .filter(), and .apply() in GroupBy operations, and discuss the performance implications of each.',
    sampleAnswer: `The split-apply-combine pattern is the conceptual framework underlying GroupBy operations in Pandas. Understanding when to use each method is crucial for efficient data analysis.

**The Split-Apply-Combine Pattern:**

**1. Split**: Divide data into groups based on one or more keys
**2. Apply**: Perform operations on each group independently  
**3. Combine**: Merge results back into a data structure

\`\`\`python
# Visual representation
Original DataFrame
├── Split by "Category"
│   ├── Group A: [rows where Category == 'A',]
│   ├── Group B: [rows where Category == 'B',]
│   └── Group C: [rows where Category == 'C',]
├── Apply function to each group
│   ├── Group A → result_A
│   ├── Group B → result_B
│   └── Group C → result_C
└── Combine results → Final output
\`\`\`

**Method 1: .agg() - Aggregation (Reduction)**

**Purpose**: Reduce each group to summary statistics

\`\`\`python
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'Department': ['Sales', 'Sales', 'IT', 'IT', 'HR', 'HR',] * 100,
    'Employee': range(600),
    'Salary': np.random.randint(40000, 120000, 600),
    'Performance': np.random.uniform(2.5, 5.0, 600)
})

# Single aggregation
dept_avg_salary = df.groupby('Department')['Salary',].agg('mean')
# Output: One value per department
# Department
# HR       75234.5
# IT       81092.3
# Sales    78456.1

# Multiple aggregations
dept_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'median', 'std', 'min', 'max',],
    'Performance': ['mean', 'count',]
})
# Output: Statistical summary per department

# Named aggregations (readable)
result = df.groupby('Department').agg(
    avg_salary=('Salary', 'mean'),
    median_salary=('Salary', 'median'),
    salary_std=('Salary', 'std'),
    employee_count=('Employee', 'count'),
    avg_performance=('Performance', 'mean')
)
\`\`\`

**When to use .agg():**
- Creating summary reports
- Computing statistics by group
- Comparing groups
- Generating dashboards
- Output size much smaller than input

**Performance**: ⚡⚡⚡ Fast (optimized C code for built-in functions)

**Method 2: .transform() - Maintain Shape**

**Purpose**: Apply function but return value for each original row

\`\`\`python
# Add group statistics to each row
df['dept_avg_salary',] = df.groupby('Department')['Salary',].transform('mean')

# Now each row has its department's average
print(df[['Employee', 'Department', 'Salary', 'dept_avg_salary',]].head())
#    Employee Department  Salary  dept_avg_salary
# 0         0      Sales   65000          78456.1
# 1         1      Sales   82000          78456.1
# 2         2         IT   95000          81092.3
# 3         3         IT   71000          81092.3
# 4         4         HR   68000          75234.5

# Compare individual to group
df['salary_vs_dept_avg',] = df['Salary',] - df['dept_avg_salary',]
df['salary_pct_of_dept_avg',] = df['Salary',] / df['dept_avg_salary',] * 100

# Normalize within group
df['normalized_salary',] = df.groupby('Department')['Salary',].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Rank within group
df['salary_rank_in_dept',] = df.groupby('Department')['Salary',].transform(
    lambda x: x.rank(ascending=False)
)

# Cumulative sum within group
df['cumsum_in_dept',] = df.groupby('Department')['Salary',].transform('cumsum')

# Fill missing with group mean
df['Salary_filled',] = df.groupby('Department')['Salary',].transform(
    lambda x: x.fillna(x.mean())
)
\`\`\`

**When to use .transform():**
- Adding group statistics to original data
- Normalizing within groups
- Computing differences from group mean
- Filling missing values with group statistics
- Ranking within groups
- Output same size as input

**Performance**: ⚡⚡ Moderate (depends on function)
- Built-in: Fast
- Lambda: Slower

**Method 3: .filter() - Remove Entire Groups**

**Purpose**: Keep or remove entire groups based on condition

\`\`\`python
# Keep only large departments (>150 employees)
large_depts = df.groupby('Department').filter(lambda x: len(x) > 150)
print(f"Original: {len(df)} rows, After filter: {len(large_depts)} rows")

# Keep departments with average salary > 80000
high_paying = df.groupby('Department').filter(
    lambda x: x['Salary',].mean() > 80000
)

# Keep departments with low performance variance (consistent)
consistent_depts = df.groupby('Department').filter(
    lambda x: x['Performance',].std() < 0.5
)

# Multiple conditions
quality_depts = df.groupby('Department').filter(
    lambda x: (len(x) > 100) and 
              (x['Salary',].mean() > 70000) and 
              (x['Performance',].mean() > 4.0)
)

# Note: filter() removes entire groups, not individual rows!
# If IT department doesn't meet criteria, ALL IT employees are removed
\`\`\`

**When to use .filter():**
- Removing underperforming groups
- Keeping only significant groups
- Data quality filtering (remove small sample sizes)
- Focus analysis on specific group types
- All-or-nothing group selection

**Performance**: ⚡⚡ Moderate (evaluates condition for each group)

**Method 4: .apply() - Maximum Flexibility**

**Purpose**: Apply arbitrary function to each group

\`\`\`python
# Return custom aggregation
def group_report(group):
    return pd.Series({
        'count': len(group),
        'avg_salary': group['Salary',].mean(),
        'top_performer': group.loc[group['Performance',].idxmax(), 'Employee',],
        'salary_range': group['Salary',].max() - group['Salary',].min(),
        'high_performers_pct': (group['Performance',] > 4.5).sum() / len(group) * 100
    })

dept_reports = df.groupby('Department').apply(group_report)
print(dept_reports)

# Return modified DataFrame
def add_rankings(group):
    group = group.copy()
    group['salary_rank',] = group['Salary',].rank(ascending=False)
    group['perf_rank',] = group['Performance',].rank(ascending=False)
    group['combined_rank',] = (group['salary_rank',] + group['perf_rank',]) / 2
    return group

df_ranked = df.groupby('Department').apply(add_rankings)

# Complex analysis
def department_analysis(group):
    # Compute correlations, percentiles, custom metrics
    return {
        'size': len(group),
        'salary_perf_corr': group['Salary',].corr(group['Performance',]),
        'p90_salary': group['Salary',].quantile(0.9),
        'top_talent_ratio': (
            (group['Salary',] > group['Salary',].quantile(0.75)) &
            (group['Performance',] > 4.5)
        ).sum() / len(group)
    }

analysis = df.groupby('Department').apply(
    lambda x: pd.Series(department_analysis(x))
)
print(analysis)
\`\`\`

**When to use .apply():**
- Complex custom logic
- Need to return different structure
- Multiple operations on group
- When .agg(), .transform(), .filter() insufficient
- Last resort (slowest option)

**Performance**: ⚡ Slow (Python function overhead)

**Performance Comparison:**

\`\`\`python
import time

# Large dataset
np.random.seed(42)
large_df = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C', 'D',], 1000000),
    'Value': np.random.randn(1000000)
})

# Method 1: .agg() with built-in
start = time.time()
result1 = large_df.groupby('Category')['Value',].agg('mean')
time1 = time.time() - start

# Method 2: .agg() with lambda
start = time.time()
result2 = large_df.groupby('Category')['Value',].agg(lambda x: x.mean())
time2 = time.time() - start

# Method 3: .transform() with built-in
start = time.time()
result3 = large_df.groupby('Category')['Value',].transform('mean')
time3 = time.time() - start

# Method 4: .apply()
start = time.time()
result4 = large_df.groupby('Category').apply(lambda x: x['Value',].mean())
time4 = time.time() - start

print(f".agg('mean'): {time1:.4f}s (baseline)")
print(f".agg(lambda): {time2:.4f}s ({time2/time1:.1f}x slower)")
print(f".transform('mean'): {time3:.4f}s ({time3/time1:.1f}x slower)")
print(f".apply(): {time4:.4f}s ({time4/time1:.1f}x slower)")

# Typical results:
# .agg('mean'): 0.0143s (baseline)
# .agg(lambda): 0.1892s (13.2x slower)
# .transform('mean'): 0.0201s (1.4x slower)
# .apply(): 0.5671s (39.7x slower)
\`\`\`

**Decision Tree:**

\`\`\`
Need to operate on groups?
│
├─ Want summary statistics? → Use .agg()
│  ├─ Single statistic → .agg('mean')
│  ├─ Multiple statistics → .agg(['mean', 'std', 'count',])
│  └─ Different stats per column → .agg({'col1': 'mean', 'col2': 'sum'})
│
├─ Want to add group info to each row? → Use .transform()
│  ├─ Simple operation → .transform('mean')
│  ├─ Normalize → .transform(lambda x: (x - x.mean()) / x.std())
│  └─ Fill missing → .transform(lambda x: x.fillna(x.mean()))
│
├─ Want to remove entire groups? → Use .filter()
│  └─ .filter(lambda x: len(x) > 100)
│
└─ Need custom complex logic? → Use .apply()
   └─ But try to use .agg()/.transform() if possible!
\`\`\`

**Best Practices:**

1. **Start with .agg() or .transform()**
   - Faster and clearer intent

2. **Use built-in functions** when possible
   \`\`\`python
   # Good
   df.groupby('Cat')['Val',].transform('mean')
   
   # Bad
   df.groupby('Cat')['Val',].transform(lambda x: x.mean())
   \`\`\`

3. **Avoid .apply() unless necessary**
   - Use .agg() with custom function instead if possible

4. **Chain operations** for clarity
   \`\`\`python
   result = (df
       .groupby('Department')
       .agg({'Salary': ['mean', 'std',], 'Count': 'size'})
       .round(2)
       .sort_values(('Salary', 'mean'), ascending=False)
   )
   \`\`\`

5. **Document complex groupby** operations
   \`\`\`python
   # Compute department statistics for benchmarking
   dept_stats = df.groupby('Department').agg(
       avg_salary=('Salary', 'mean'),
       p50_salary=('Salary', 'median'),
       headcount=('Employee', 'count')
   )
   \`\`\`

**Key Takeaway:**

Choose the right tool:
- **Summary** → .agg()
- **Add to rows** → .transform()
- **Remove groups** → .filter()
- **Custom logic** → .apply() (last resort)

Master these four methods and you can answer any group-based question about your data efficiently!`,
    keyPoints: [
      'Split-apply-combine: split data into groups, apply function, combine results',
      '.agg() applies aggregation functions (sum, mean) - reduces to summary statistics',
      '.transform() applies function and returns same shape - useful for normalization',
      '.filter() keeps/removes entire groups based on condition',
      '.apply() most flexible but slowest - use specific methods when possible',
    ],
  },
  {
    id: 'data-aggregation-grouping-dq-2',
    question:
      'Discuss pivot tables vs. groupby operations in Pandas. When would you use each, how do they differ in output format, and how can you convert between them?',
    sampleAnswer: `Pivot tables and GroupBy operations both aggregate data, but they serve different purposes and produce different output formats. Understanding when to use each is crucial for effective data analysis and presentation.

**Core Differences:**

**GroupBy**: Split-apply-combine with Series/DataFrame output
**Pivot Table**: Spreadsheet-style table with index and columns

**1. Basic Comparison:**

\`\`\`python
import pandas as pd
import numpy as np

# Sample sales data
df = pd.DataFrame({
    'Region': ['East', 'East', 'West', 'West', 'East', 'West',] * 10,
    'Product': ['A', 'B', 'A', 'B', 'A', 'B',] * 10,
    'Quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2',] * 10,
    'Sales': np.random.randint(1000, 5000, 60),
    'Quantity': np.random.randint(10, 100, 60)
})

# GroupBy approach
grouped = df.groupby(['Region', 'Product',])['Sales',].mean()
print(grouped)
# Output: Series with MultiIndex
# Region  Product
# East    A          2543.5
#         B          2987.2
# West    A          3124.8
#         B          2756.1

# Pivot table approach
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='mean'
)
print(pivot)
# Output: DataFrame (spreadsheet-style)
# Product      A        B
# Region                
# East     2543.5  2987.2
# West     3124.8  2756.1
\`\`\`

**2. When to Use GroupBy:**

**Advantages:**
- More flexible (any number of grouping columns)
- Can apply complex custom functions
- Better for programmatic operations
- Native to Pandas workflow
- Efficient for single aggregation

**Use GroupBy when:**

**a) Single dimension grouping:**
\`\`\`python
# Department-level analysis
dept_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'median', 'std',],
    'Experience': 'mean',
    'Employee': 'count'
})
# Clear, hierarchical output
\`\`\`

**b) Many grouping levels (3+):**
\`\`\`python
# Multi-level analysis
complex_group = df.groupby(['Region', 'Product', 'Quarter', 'Salesperson',])['Sales',].sum()
# Pivot table becomes unwieldy with 4+ dimensions
\`\`\`

**c) Need custom aggregations:**
\`\`\`python
def custom_stat(x):
    return (x > x.median()).sum() / len(x) * 100

result = df.groupby('Region')['Sales',].agg([
    'mean',
    ('above_median_pct', custom_stat)
])
\`\`\`

**d) Programmatic access to groups:**
\`\`\`python
# Iterate through groups
for region, group in df.groupby('Region'):
    print(f"{region}: {group['Sales',].sum()}")
    # Perform region-specific analysis
\`\`\`

**e) Chaining operations:**
\`\`\`python
result = (df
    .groupby('Region')['Sales',]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
# Natural Pandas workflow
\`\`\`

**3. When to Use Pivot Tables:**

**Advantages:**
- Spreadsheet-style output (easier to read)
- Natural for 2D comparisons
- Better for presentations/reports
- Easy column/row comparisons
- Visual clarity

**Use Pivot Tables when:**

**a) Two-dimensional comparison:**
\`\`\`python
# Compare regions (rows) vs products (columns)
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum'
)
# Product      A      B      C
# Region                      
# East     50000  45000  38000
# West     62000  58000  41000

# Easy to see: West sells more of Product A than East
\`\`\`

**b) Creating reports/dashboards:**
\`\`\`python
# Executive summary
summary = df.pivot_table(
    values=['Sales', 'Quantity',],
    index='Region',
    columns='Quarter',
    aggfunc={'Sales': 'sum', 'Quantity': 'mean'},
    margins=True,  # Adds totals
    margins_name='Total'
)
# Professional-looking summary table
\`\`\`

**c) Need Excel-like functionality:**
\`\`\`python
# With subtotals
pivot = df.pivot_table(
    values='Sales',
    index=['Region', 'Product',],
    columns='Quarter',
    aggfunc='sum',
    fill_value=0,
    margins=True
)
# Similar to Excel pivot table
\`\`\`

**d) Visual comparison across categories:**
\`\`\`python
# Heat map-friendly format
pivot = df.pivot_table(
    values='Sales',
    index='Product',
    columns='Region',
    aggfunc='mean'
)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Average Sales by Product and Region')
plt.show()
\`\`\`

**4. Converting Between Formats:**

**GroupBy → Pivot Table (unstack):**
\`\`\`python
# GroupBy result
grouped = df.groupby(['Region', 'Product',])['Sales',].mean()
# Region  Product
# East    A          2543.5
#         B          2987.2
# West    A          3124.8
#         B          2756.1

# Convert to pivot format
pivot = grouped.unstack()
# or: pivot = grouped.unstack(level='Product')
print(pivot)
# Product      A        B
# Region                
# East     2543.5  2987.2
# West     3124.8  2756.1

# Multiple levels
grouped = df.groupby(['Region', 'Product', 'Quarter',])['Sales',].mean()
pivot = grouped.unstack(level=['Product', 'Quarter',])
\`\`\`

**Pivot Table → GroupBy (stack):**
\`\`\`python
# Pivot table result
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='mean'
)

# Convert to long format (GroupBy-style)
stacked = pivot.stack()
print(stacked)
# Region  Product
# East    A          2543.5
#         B          2987.2
# West    A          3124.8
#         B          2756.1
\`\`\`

**5. Performance Comparison:**

\`\`\`python
import time

# Large dataset
n = 1_000_000
large_df = pd.DataFrame({
    'Category1': np.random.choice(['A', 'B', 'C',], n),
    'Category2': np.random.choice(['X', 'Y', 'Z',], n),
    'Value': np.random.randn(n)
})

# GroupBy
start = time.time()
gb_result = large_df.groupby(['Category1', 'Category2',])['Value',].mean()
gb_time = time.time() - start

# Pivot Table
start = time.time()
pv_result = large_df.pivot_table(
    values='Value',
    index='Category1',
    columns='Category2',
    aggfunc='mean'
)
pv_time = time.time() - start

print(f"GroupBy: {gb_time:.4f}s")
print(f"Pivot Table: {pv_time:.4f}s")
print(f"Difference: {abs(pv_time - gb_time) / min(pv_time, gb_time) * 100:.1f}%")

# Results typically show:
# GroupBy: 0.1234s
# Pivot Table: 0.1456s
# Difference: ~15-20% (similar performance)
\`\`\`

**6. Complex Example: Sales Analysis:**

\`\`\`python
# Comprehensive sales data
np.random.seed(42)
sales = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=1000, freq='D'),
    'Region': np.random.choice(['North', 'South', 'East', 'West',], 1000),
    'Product': np.random.choice(['A', 'B', 'C',], 1000),
    'Salesperson': np.random.choice(['SP1', 'SP2', 'SP3', 'SP4',], 1000),
    'Sales': np.random.randint(1000, 10000, 1000),
    'Quantity': np.random.randint(1, 50, 1000)
})
sales['Month',] = sales['Date',].dt.to_period('M')

# GroupBy approach: Complex multi-metric analysis
groupby_analysis = sales.groupby(['Region', 'Product',]).agg({
    'Sales': ['sum', 'mean', 'std', 'count',],
    'Quantity': ['sum', 'mean',],
    'Salesperson': lambda x: x.nunique()  # Unique salespeople
}).round(2)
groupby_analysis.columns = ['_'.join(col) for col in groupby_analysis.columns]
print("GroupBy Analysis:")
print(groupby_analysis.head())

# Pivot table approach: Monthly trends by region
pivot_monthly = sales.pivot_table(
    values='Sales',
    index='Month',
    columns='Region',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)
print("\\nPivot Table - Monthly Trends:")
print(pivot_monthly.head())

# Pivot table: Product mix by region
pivot_mix = sales.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum'
)
# Add percentage columns
pivot_mix['Total',] = pivot_mix.sum(axis=1)
for col in ['A', 'B', 'C',]:
    pivot_mix[f'{col}_pct',] = pivot_mix[col] / pivot_mix['Total',] * 100
print("\\nProduct Mix by Region:")
print(pivot_mix)

# Best of both: GroupBy then unstack
combo = (sales
    .groupby(['Region', 'Product',])
    .agg({'Sales': 'sum', 'Quantity': 'mean'})
    .unstack(level='Product')
)
print("\\nCombination Approach:")
print(combo)
\`\`\`

**7. Advanced Techniques:**

**Multiple values in pivot:**
\`\`\`python
# Show both sales and quantity
multi_pivot = df.pivot_table(
    values=['Sales', 'Quantity',],
    index='Region',
    columns='Product',
    aggfunc={'Sales': 'sum', 'Quantity': 'mean'}
)
print(multi_pivot)
\`\`\`

**Custom aggregation in pivot:**
\`\`\`python
def cv(x):  # Coefficient of variation
    return x.std() / x.mean() if x.mean() != 0 else 0

pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc=[np.mean, cv]
)
\`\`\`

**Conditional pivot:**
\`\`\`python
# Only high-value transactions
high_value = df[df['Sales',] > df['Sales',].median()]
pivot = high_value.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='count'  # Count of high-value sales
)
\`\`\`

**Decision Matrix:**

| Scenario | Use GroupBy | Use Pivot Table |
|----------|-------------|-----------------|
| 3+ grouping levels | ✅ | ❌ |
| 2D comparison | ❌ | ✅ |
| Custom aggregations | ✅ | ⚠️ Limited |
| Report/presentation | ❌ | ✅ |
| Further processing | ✅ | ❌ |
| Visualization | ⚠️ Need unstack | ✅ Ready |
| Multiple metrics | ✅ | ✅ |
| Subtotals/margins | ❌ | ✅ |

**Best Practices:**

1. **Start with GroupBy**, unstack if needed for presentation
2. **Use Pivot for reports**, GroupBy for analysis
3. **Know unstack/stack** for format conversion
4. **Add margins** to pivot tables for totals
5. **Document complex pivots** for maintainability

**Key Takeaway:**

- **GroupBy**: Flexible, programmatic, good for complex analysis
- **Pivot**: Visual, spreadsheet-like, good for reports
- **Often use both**: GroupBy for analysis → unstack for presentation

Choose based on your audience and purpose—data scientists prefer GroupBy, business stakeholders prefer pivot tables!`,
    keyPoints: [
      'GroupBy with multiple columns creates hierarchical MultiIndex',
      'Access group levels with .xs() cross-section or boolean indexing',
      'reset_index() flattens MultiIndex back to regular columns',
      'Use level parameter in aggregations to specify which index level to operate on',
      'MultiIndex enables powerful hierarchical aggregations and slicing',
    ],
  },
  {
    id: 'data-aggregation-grouping-dq-3',
    question:
      "Explain how to properly handle hierarchical/multi-level grouping in Pandas. Discuss MultiIndex, when it's useful, how to navigate it, and common pitfalls when working with grouped data.",
    sampleAnswer: `Multi-level grouping and MultiIndex (hierarchical indexing) are powerful features in Pandas, but they can be confusing. Understanding how to work with them effectively is crucial for complex data analysis.

**What is MultiIndex?**

MultiIndex allows multiple index levels, enabling representation of higher-dimensional data in a 2D DataFrame.

\`\`\`python
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'Region': ['East', 'East', 'West', 'West',] * 4,
    'Product': ['A', 'B', 'A', 'B',] * 4,
    'Quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2',] * 2,
    'Sales': np.random.randint(1000, 5000, 16),
    'Quantity': np.random.randint(10, 100, 16)
})

# Group by multiple columns creates MultiIndex
grouped = df.groupby(['Region', 'Product', 'Quarter',])['Sales',].sum()
print(grouped)
# Region  Product  Quarter
# East    A        Q1        3245
#                  Q2        4123
#         B        Q1        2987
#                  Q2        3456
# West    A        Q1        3789
#                  Q2        4234
#         B        Q1        3123
#                  Q2        2876

print(type(grouped.index))  # <class 'pandas.core.indexes.multi.MultiIndex'>
print(f"Index levels: {grouped.index.nlevels}")  # 3
print(f"Index names: {grouped.index.names}")  # ['Region', 'Product', 'Quarter',]
\`\`\`

**Creating MultiIndex:**

\`\`\`python
# Method 1: From groupby
multi_idx = df.groupby(['Region', 'Product',])['Sales',].mean()

# Method 2: set_index with multiple columns
df_multi = df.set_index(['Region', 'Product', 'Quarter',])

# Method 3: From arrays
arrays = [
    ['East', 'East', 'West', 'West',],
    ['A', 'B', 'A', 'B',]
]
index = pd.MultiIndex.from_arrays(arrays, names=['Region', 'Product',])
df_multi = pd.DataFrame({'Sales': [100, 200, 300, 400]}, index=index)

# Method 4: From tuples
tuples = [('East', 'A'), ('East', 'B'), ('West', 'A'), ('West', 'B')]
index = pd.MultiIndex.from_tuples(tuples, names=['Region', 'Product',])

# Method 5: From product (cartesian)
index = pd.MultiIndex.from_product([
    ['East', 'West',],
    ['A', 'B', 'C',],
    ['Q1', 'Q2',]
], names=['Region', 'Product', 'Quarter',])
\`\`\`

**Selecting from MultiIndex:**

\`\`\`python
# Select by tuple
value = grouped[('East', 'A', 'Q1')]
print(f"East, Product A, Q1: {value}")

# Select with .loc
# Single level
east_sales = grouped.loc['East',]
print(east_sales)
# Product  Quarter
# A        Q1        3245
#          Q2        4123
# B        Q1        2987
#          Q2        3456

# Multiple levels
east_a = grouped.loc[('East', 'A')]
print(east_a)
# Quarter
# Q1    3245
# Q2    4123

# Specific value
value = grouped.loc[('East', 'A', 'Q1')]

# Slicing (requires sorted index)
grouped_sorted = grouped.sort_index()
slice_result = grouped_sorted.loc[('East', 'A'):('East', 'B')]

# Cross-section (xs) - select one level
q1_sales = grouped.xs('Q1', level='Quarter')
print(q1_sales)
# Region  Product
# East    A          3245
#         B          2987
# West    A          3789
#         B          3123

# Multiple level selection
east_q1 = grouped.xs(('East', 'Q1'), level=('Region', 'Quarter'))
\`\`\`

**Navigating MultiIndex:**

\`\`\`python
# Get specific level
regions = grouped.index.get_level_values('Region')
products = grouped.index.get_level_values('Product')
quarters = grouped.index.get_level_values('Quarter')

# Unique values in level
unique_regions = grouped.index.get_level_values('Region').unique()
print(f"Regions: {unique_regions}")

# Swap levels
swapped = grouped.swaplevel('Region', 'Product')
print(swapped.head())
# Product  Region  Quarter
# A        East    Q1        3245
# B        East    Q1        2987
# ...

# Reorder levels
reordered = grouped.reorder_levels(['Quarter', 'Region', 'Product',])

# Remove level (aggregate)
by_region_product = grouped.groupby(level=['Region', 'Product',]).sum()
print(by_region_product)
# Region  Product
# East    A          7368
#         B          6443
# West    A          8023
#         B          5999

# Drop level
dropped = grouped.droplevel('Quarter')  # Keeps first occurrence
\`\`\`

**Reshaping MultiIndex:**

\`\`\`python
# Unstack (MultiIndex → columns)
unstacked = grouped.unstack()
print(unstacked)
# Quarter     Q1    Q2
# Region Product          
# East   A      3245  4123
#        B      2987  3456
# West   A      3789  4234
#        B      3123  2876

# Unstack specific level
unstacked_region = grouped.unstack(level='Region')
print(unstacked_region)
# Region        East  West
# Product Quarter          
# A       Q1     3245  3789
#         Q2     4123  4234
# B       Q1     2987  3123
#         Q2     3456  2876

# Stack (columns → MultiIndex)
stacked = unstacked.stack()
print(stacked)  # Back to original

# Reset index (MultiIndex → columns)
flat = grouped.reset_index()
print(flat)
#   Region Product Quarter  Sales
# 0   East       A      Q1   3245
# 1   East       A      Q2   4123
# ...

# Partial reset
partially_flat = grouped.reset_index(level='Quarter')
print(partially_flat)
#                Quarter  Sales
# Region Product              
# East   A            Q1   3245
#        A            Q2   4123
# ...
\`\`\`

**Aggregating MultiIndex Data:**

\`\`\`python
# Aggregate across specific level
# Sum across quarters (collapse Quarter level)
by_region_product = grouped.groupby(level=['Region', 'Product',]).sum()

# Sum across products (collapse Product level)
by_region_quarter = grouped.groupby(level=['Region', 'Quarter',]).sum()

# Multiple aggregations
result = grouped.groupby(level='Region').agg(['sum', 'mean', 'count',])
print(result)

# Complex aggregation
df_multi = grouped.reset_index()
complex_agg = df_multi.groupby(['Region', 'Product',]).agg({
    'Sales': ['sum', 'mean', 'std',],
    'Quarter': 'count'
})
print(complex_agg)
\`\`\`

**Common Pitfalls and Solutions:**

**Pitfall 1: Forgetting to sort index**
\`\`\`python
# Problem: Slicing unsorted MultiIndex
try:
    result = grouped.loc['East':'West',]  # Error!
except KeyError as e:
    print("Error: MultiIndex not sorted")

# Solution: Sort first
grouped_sorted = grouped.sort_index()
result = grouped_sorted.loc['East':'West',]  # Works
\`\`\`

**Pitfall 2: Confusing loc with tuple indexing**
\`\`\`python
# Wrong: Missing comma (interpreted as list)
# grouped.loc[['East' 'A',]]  # Error

# Correct: Tuple for multiple levels
result = grouped.loc[('East', 'A')]

# Correct: List of tuples for multiple selections
result = grouped.loc[[('East', 'A'), ('West', 'B')]]
\`\`\`

**Pitfall 3: Unexpected broadcast behavior**
\`\`\`python
# Be careful with operations on MultiIndex
df_multi = grouped.reset_index()
df_multi = df_multi.set_index(['Region', 'Product',])

# Adding scalar works as expected
result = df_multi + 100

# But operations between MultiIndex DataFrames require alignment
df1 = df_multi.loc['East',]
df2 = df_multi.loc['West',]
# result = df1 + df2  # Alignment issues

# Solution: Reset index or use explicit joins
\`\`\`

**Pitfall 4: Column names after groupby.agg()**
\`\`\`python
# Problem: Hierarchical column names
result = df.groupby(['Region', 'Product',]).agg({
    'Sales': ['sum', 'mean',],
    'Quantity': ['sum', 'mean',]
})
print(result.columns)
# MultiIndex([('Sales', 'sum'), ('Sales', 'mean'), ...])

# Solution: Flatten column names
result.columns = ['_'.join(col).strip() for col in result.columns.values]
print(result.columns)
# Index(['Sales_sum', 'Sales_mean', 'Quantity_sum', 'Quantity_mean',])

# Or use named aggregations (Pandas 0.25+)
result = df.groupby(['Region', 'Product',]).agg(
    sales_sum=('Sales', 'sum'),
    sales_mean=('Sales', 'mean'),
    qty_sum=('Quantity', 'sum'),
    qty_mean=('Quantity', 'mean')
)
\`\`\`

**Pitfall 5: Memory with large MultiIndex**
\`\`\`python
# Problem: MultiIndex with many levels can be memory-intensive
# Each index level stores all values

# Solution: Use categorical for repeated values
df['Region',] = df['Region',].astype('category')
df['Product',] = df['Product',].astype('category')
grouped = df.groupby(['Region', 'Product',])['Sales',].sum()
# Reduced memory usage
\`\`\`

**When to Use MultiIndex:**

**Use MultiIndex when:**

1. **Hierarchical data structure**
\`\`\`python
# Organizational hierarchy
company_data = df.groupby(['Division', 'Department', 'Team',])
\`\`\`

2. **Time series with multiple dimensions**
\`\`\`python
# Regional sales over time
time_series = df.groupby(['Region', pd.Grouper(key='Date', freq='M')])
\`\`\`

3. **Statistical tables**
\`\`\`python
# Demographic analysis
demographics = df.groupby(['Age_Group', 'Gender', 'Income_Bracket',])
\`\`\`

4. **Pivot-like operations**
\`\`\`python
# Cross-tabulation
cross_tab = df.groupby(['Category1', 'Category2', 'Category3',])
\`\`\`

**Avoid MultiIndex when:**

1. **Simple one-level grouping** (use Series index)
2. **Need frequent filtering** (flat structure is easier)
3. **Working with inexperienced users** (confusing)
4. **Exporting to CSV** (loses structure, flatten first)

**Best Practices:**

1. **Sort MultiIndex** for slicing
\`\`\`python
grouped = grouped.sort_index()
\`\`\`

2. **Name your index levels**
\`\`\`python
grouped.index.names = ['Region', 'Product', 'Quarter',]
\`\`\`

3. **Flatten for exports**
\`\`\`python
df_flat = grouped.reset_index()
df_flat.to_csv('output.csv')
\`\`\`

4. **Use named aggregations**
\`\`\`python
result = df.groupby(['A', 'B',]).agg(
    sum_sales=('Sales', 'sum'),
    avg_sales=('Sales', 'mean')
)
\`\`\`

5. **Document index structure**
\`\`\`python
# Clear documentation
def analyze_sales(df):
    """
    Groups sales data by Region and Product.
    Returns MultiIndex Series with levels:
      0: Region (str)
      1: Product (str)
    Values: Total sales (float)
    """
    return df.groupby(['Region', 'Product',])['Sales',].sum()
\`\`\`

**Key Takeaway:**

MultiIndex is powerful for hierarchical data but adds complexity. Use it when structure genuinely requires multiple levels. For presentation, flatten with \`reset_index()\` or \`unstack()\`. Master \`.loc\`, \`.xs()\`, and level operations to navigate effectively. When in doubt, start with flat structure—you can always add levels later!`,
    keyPoints: [
      'Pivot creates wide-format table with unique values as columns',
      'Melt converts wide format back to long format (inverse of pivot)',
      'pivot_table adds aggregation function for handling duplicate entries',
      'Wide format better for human reading, long format better for analysis/plotting',
      'Use pd.crosstab() for frequency tables with categorical data',
    ],
  },
];
