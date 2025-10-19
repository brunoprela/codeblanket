/**
 * Section: Data Aggregation & Grouping
 * Module: Python for Data Science
 *
 * Covers GroupBy operations, aggregation functions, pivot tables, cross-tabulations, transformations within groups
 */

export const dataAggregationGrouping = {
  id: 'data-aggregation-grouping',
  title: 'Data Aggregation & Grouping',
  content: `
# Data Aggregation & Grouping

## Introduction

Data aggregation and grouping are fundamental operations in data analysis, allowing you to summarize data by categories, compute statistics within groups, and transform data based on group membership. Pandas' GroupBy functionality is one of its most powerful features, implementing the split-apply-combine pattern efficiently.

**Split-Apply-Combine Pattern:**
1. **Split**: Divide data into groups based on criteria
2. **Apply**: Compute some function within each group  
3. **Combine**: Merge results back together

\`\`\`python
import pandas as pd
import numpy as np

# Sample dataset
df = pd.DataFrame({
    'Department': ['Sales', 'Sales', 'IT', 'IT', 'HR', 'HR'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Salary': [50000, 60000, 75000, 65000, 55000, 52000],
    'Experience': [2, 5, 8, 4, 3, 6]
})
\`\`\`

## GroupBy Fundamentals

### Creating GroupBy Objects

\`\`\`python
# Group by single column
grouped = df.groupby('Department')
print(type(grouped))  # <class 'pandas.core.groupby.generic.DataFrameGroupBy'>

# Group by multiple columns
grouped = df.groupby(['Department', 'Experience'])

# View groups
print(grouped.groups)
# {'HR': [4, 5], 'IT': [2, 3], 'Sales': [0, 1]}

# Number of groups
print(f"Number of groups: {grouped.ngroups}")

# Size of each group
print(grouped.size())
# Department
# HR       2
# IT       2
# Sales    2

# Iterate through groups
for name, group in df.groupby('Department'):
    print(f"\\n{name}:")
    print(group)
\`\`\`

### Selecting Groups

\`\`\`python
# Get specific group
it_group = grouped.get_group('IT')
print(it_group)

# Filter groups (groups meeting condition)
large_depts = grouped.filter(lambda x: len(x) > 1)
print(large_depts)

# Select specific columns after grouping
salary_by_dept = df.groupby('Department')['Salary']
print(salary_by_dept.mean())
\`\`\`

## Aggregation Functions

### Single Aggregation

\`\`\`python
# Mean salary by department
print(df.groupby('Department')['Salary'].mean())
# Department
# HR       53500.0
# IT       70000.0
# Sales    55000.0

# Multiple statistics
print(df.groupby('Department')['Salary'].sum())
print(df.groupby('Department')['Salary'].min())
print(df.groupby('Department')['Salary'].max())
print(df.groupby('Department')['Salary'].std())
print(df.groupby('Department')['Salary'].count())

# Median (robust to outliers)
print(df.groupby('Department')['Salary'].median())

# Quantiles
print(df.groupby('Department')['Salary'].quantile(0.75))
\`\`\`

### Multiple Aggregations

\`\`\`python
# Apply multiple aggregations
result = df.groupby('Department')['Salary'].agg(['mean', 'median', 'std', 'min', 'max'])
print(result)
#               mean  median          std    min    max
# Department                                           
# HR          53500.0  53500.0  2121.320344  52000  55000
# IT          70000.0  70000.0  7071.067812  65000  75000
# Sales       55000.0  55000.0  7071.067812  50000  60000

# Custom names for aggregations
result = df.groupby('Department')['Salary'].agg([
    ('Average', 'mean'),
    ('Median', 'median'),
    ('Range', lambda x: x.max() - x.min())
])
print(result)

# Different aggregations for different columns
result = df.groupby('Department').agg({
    'Salary': ['mean', 'median', 'std'],
    'Experience': ['mean', 'max']
})
print(result)
#               Salary                     Experience      
#                 mean  median          std       mean  max
# Department                                              
# HR          53500.0  53500.0  2121.320344       4.5    6
# IT          70000.0  70000.0  7071.067812       6.0    8
# Sales       55000.0  55000.0  7071.067812       3.5    5
\`\`\`

### Custom Aggregation Functions

\`\`\`python
# Define custom aggregation
def salary_range(x):
    return x.max() - x.min()

def cv(x):  # Coefficient of variation
    return x.std() / x.mean()

result = df.groupby('Department')['Salary'].agg([
    'mean',
    salary_range,
    cv
])
print(result)

# Lambda functions
result = df.groupby('Department')['Salary'].agg([
    ('mean', 'mean'),
    ('range', lambda x: x.max() - x.min()),
    ('cv', lambda x: x.std() / x.mean() if x.mean() != 0 else 0)
])
print(result)
\`\`\`

### Named Aggregations (Pandas 0.25+)

\`\`\`python
# More readable syntax
result = df.groupby('Department').agg(
    mean_salary=('Salary', 'mean'),
    median_salary=('Salary', 'median'),
    salary_range=('Salary', lambda x: x.max() - x.min()),
    avg_experience=('Experience', 'mean'),
    num_employees=('Employee', 'count')
)
print(result)
\`\`\`

## Transformation and Filtering

### Transform (Returns Same Shape)

\`\`\`python
# Add group statistics to each row
df['dept_avg_salary'] = df.groupby('Department')['Salary'].transform('mean')
print(df)
#   Department Employee  Salary  Experience  dept_avg_salary
# 0      Sales    Alice   50000           2          55000.0
# 1      Sales      Bob   60000           5          55000.0
# 2         IT  Charlie   75000           8          70000.0
# 3         IT    David   65000           4          70000.0
# 4         HR      Eve   55000           3          53500.0
# 5         HR    Frank   52000           6          53500.0

# Normalize within group
df['salary_vs_dept_avg'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Rank within group
df['rank_in_dept'] = df.groupby('Department')['Salary'].rank(ascending=False)

# Cumulative sum within group
df['cumsum_salary'] = df.groupby('Department')['Salary'].cumsum()

# Group-specific scaling
df['pct_of_dept_total'] = df.groupby('Department')['Salary'].transform(
    lambda x: x / x.sum()
)

print(df[['Department', 'Salary', 'dept_avg_salary', 'pct_of_dept_total']])
\`\`\`

### Filter (Remove Entire Groups)

\`\`\`python
# Keep only departments with average salary > 60000
high_paying_depts = df.groupby('Department').filter(
    lambda x: x['Salary'].mean() > 60000
)
print(high_paying_depts)  # Only IT department

# Keep groups with more than 2 employees
large_depts = df.groupby('Department').filter(lambda x: len(x) > 2)

# Keep groups where max experience > 5
experienced_depts = df.groupby('Department').filter(
    lambda x: x['Experience'].max() > 5
)
print(experienced_depts)
\`\`\`

## Pivot Tables and Cross-Tabulation

### Pivot Tables

\`\`\`python
# Create more complex dataset
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=12, freq='M'),
    'Region': ['East', 'West', 'East', 'West'] * 3,
    'Product': ['A', 'A', 'B', 'B'] * 3,
    'Sales': np.random.randint(100, 500, 12),
    'Quantity': np.random.randint(10, 50, 12)
})

# Basic pivot table
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='mean'
)
print(pivot)
# Product    A    B
# Region           
# East     ...  ...
# West     ...  ...

# Multiple values
pivot = df.pivot_table(
    values=['Sales', 'Quantity'],
    index='Region',
    columns='Product',
    aggfunc='mean'
)
print(pivot)

# Multiple aggregation functions
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc=['mean', 'sum', 'count']
)
print(pivot)

# Add margins (totals)
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    margins=True,  # Adds 'All' row and column
    margins_name='Total'
)
print(pivot)

# Fill missing values
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0  # Replace NaN with 0
)
print(pivot)
\`\`\`

### Cross-Tabulation

\`\`\`python
# Frequency tables
crosstab = pd.crosstab(df['Region'], df['Product'])
print(crosstab)
# Product  A  B
# Region       
# East     3  3
# West     3  3

# With values (like pivot table)
crosstab = pd.crosstab(
    df['Region'],
    df['Product'],
    values=df['Sales'],
    aggfunc='mean'
)
print(crosstab)

# Add margins
crosstab = pd.crosstab(
    df['Region'],
    df['Product'],
    margins=True
)
print(crosstab)

# Normalize (percentages)
crosstab_pct = pd.crosstab(
    df['Region'],
    df['Product'],
    normalize='all'  # 'index', 'columns', or 'all'
)
print(crosstab_pct)
\`\`\`

## Multi-Level Indexing with GroupBy

### Hierarchical Indexing

\`\`\`python
# Group by multiple columns
grouped = df.groupby(['Region', 'Product'])['Sales'].mean()
print(grouped)
# Region  Product
# East    A          ...
#         B          ...
# West    A          ...
#         B          ...

# Access multi-index
print(grouped.loc['East', 'A'])

# Unstack to convert to columns
unstacked = grouped.unstack()
print(unstacked)
# Product    A    B
# Region           
# East     ...  ...
# West     ...  ...

# Stack back
stacked = unstacked.stack()
print(stacked)

# Reset index to flat DataFrame
flat = grouped.reset_index()
print(flat)
#   Region Product  Sales
# 0   East       A    ...
# 1   East       B    ...
# 2   West       A    ...
# 3   West       B    ...
\`\`\`

### Multiple Aggregations with MultiIndex

\`\`\`python
# Complex aggregation
result = df.groupby(['Region', 'Product']).agg({
    'Sales': ['mean', 'sum', 'count'],
    'Quantity': ['mean', 'sum']
})
print(result)
#                Sales                Quantity      
#                 mean   sum count     mean  sum
# Region Product                                  
# East   A         ...   ...   ...      ...  ...
#        B         ...   ...   ...      ...  ...
# West   A         ...   ...   ...      ...  ...
#        B         ...   ...   ...      ...  ...

# Flatten column names
result.columns = ['_'.join(col).strip() for col in result.columns.values]
result = result.reset_index()
print(result)
\`\`\`

## Practical Examples

### Example 1: Sales Analysis

\`\`\`python
# Create comprehensive sales dataset
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')
sales_data = pd.DataFrame({
    'date': np.random.choice(dates, 1000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'product': np.random.choice(['A', 'B', 'C'], 1000),
    'salesperson': np.random.choice([f'SP{i}' for i in range(1, 11)], 1000),
    'sales': np.random.randint(100, 5000, 1000),
    'quantity': np.random.randint(1, 20, 1000)
})

# 1. Total sales by region
region_sales = sales_data.groupby('region')['sales'].sum().sort_values(ascending=False)
print("Total sales by region:")
print(region_sales)

# 2. Average sales per product per region
product_region = sales_data.pivot_table(
    values='sales',
    index='product',
    columns='region',
    aggfunc='mean'
)
print("\\nAverage sales per product per region:")
print(product_region)

# 3. Top 5 salespeople
top_sellers = sales_data.groupby('salesperson').agg({
    'sales': ['sum', 'mean', 'count']
}).sort_values(('sales', 'sum'), ascending=False).head()
print("\\nTop 5 salespeople:")
print(top_sellers)

# 4. Monthly trends
sales_data['month'] = pd.to_datetime(sales_data['date']).dt.to_period('M')
monthly_sales = sales_data.groupby('month')['sales'].agg(['sum', 'mean', 'count'])
print("\\nMonthly sales trends:")
print(monthly_sales.head())

# 5. Product mix by region
product_mix = pd.crosstab(
    sales_data['region'],
    sales_data['product'],
    values=sales_data['sales'],
    aggfunc='sum',
    normalize='index'  # Percentage within region
)
print("\\nProduct mix by region (%):")
print(product_mix * 100)
\`\`\`

### Example 2: Employee Performance Analysis

\`\`\`python
# Employee dataset
employees = pd.DataFrame({
    'employee_id': range(1, 101),
    'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], 100),
    'level': np.random.choice(['Junior', 'Mid', 'Senior', 'Lead'], 100),
    'salary': np.random.randint(40000, 150000, 100),
    'years_exp': np.random.randint(0, 20, 100),
    'performance_score': np.random.uniform(2.5, 5.0, 100)
})

# 1. Department statistics
dept_stats = employees.groupby('department').agg({
    'salary': ['mean', 'median', 'std'],
    'years_exp': 'mean',
    'performance_score': 'mean',
    'employee_id': 'count'
}).round(2)
dept_stats.columns = ['_'.join(col) for col in dept_stats.columns]
dept_stats = dept_stats.rename(columns={'employee_id_count': 'headcount'})
print("Department statistics:")
print(dept_stats)

# 2. Salary by level and department
salary_matrix = employees.pivot_table(
    values='salary',
    index='level',
    columns='department',
    aggfunc='mean'
).round(0)
print("\\nAverage salary by level and department:")
print(salary_matrix)

# 3. High performers (top 25% in each department)
employees['performance_quartile'] = employees.groupby('department')['performance_score'].transform(
    lambda x: pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
)
high_performers = employees[employees['performance_quartile'] == 'Q4']
print(f"\\nHigh performers: {len(high_performers)} employees")

# 4. Salary vs performance correlation by department
for dept in employees['department'].unique():
    dept_data = employees[employees['department'] == dept]
    corr = dept_data['salary'].corr(dept_data['performance_score'])
    print(f"{dept}: Salary-Performance correlation = {corr:.2f}")

# 5. Experience distribution
exp_bins = [0, 2, 5, 10, 20]
exp_labels = ['0-2', '3-5', '6-10', '11+']
employees['exp_bracket'] = pd.cut(employees['years_exp'], bins=exp_bins, labels=exp_labels)
exp_dist = pd.crosstab(employees['department'], employees['exp_bracket'])
print("\\nExperience distribution by department:")
print(exp_dist)
\`\`\`

### Example 3: Time Series Aggregation

\`\`\`python
# Stock price data
dates = pd.date_range('2024-01-01', periods=252, freq='B')  # Business days
stock_data = pd.DataFrame({
    'date': dates,
    'ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 252),
    'price': np.random.uniform(100, 200, 252),
    'volume': np.random.randint(1000000, 10000000, 252)
})
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data = stock_data.sort_values('date')

# 1. Weekly aggregation
stock_data['week'] = stock_data['date'].dt.to_period('W')
weekly = stock_data.groupby(['ticker', 'week']).agg({
    'price': ['first', 'last', 'min', 'max', 'mean'],
    'volume': 'sum'
})
print("Weekly OHLC:")
print(weekly.head(10))

# 2. Monthly returns
monthly = stock_data.groupby(['ticker', stock_data['date'].dt.to_period('M')])['price'].agg(['first', 'last'])
monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
print("\\nMonthly returns:")
print(monthly)

# 3. Rolling statistics within groups
stock_data['ma_20'] = stock_data.groupby('ticker')['price'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)
stock_data['volatility_20'] = stock_data.groupby('ticker')['price'].transform(
    lambda x: x.rolling(window=20, min_periods=1).std()
)
print("\\nRolling statistics:")
print(stock_data[['date', 'ticker', 'price', 'ma_20', 'volatility_20']].tail(10))
\`\`\`

## Advanced GroupBy Techniques

### Apply with Custom Functions

\`\`\`python
# Apply custom function to each group
def group_summary(group):
    return pd.Series({
        'count': len(group),
        'mean_salary': group['Salary'].mean(),
        'top_earner': group.loc[group['Salary'].idxmax(), 'Employee'],
        'salary_range': group['Salary'].max() - group['Salary'].min()
    })

result = df.groupby('Department').apply(group_summary)
print(result)

# Return modified DataFrame
def normalize_salaries(group):
    group['normalized_salary'] = (group['Salary'] - group['Salary'].mean()) / group['Salary'].std()
    return group

df_normalized = df.groupby('Department').apply(normalize_salaries)
print(df_normalized)
\`\`\`

### Grouping by Multiple Criteria

\`\`\`python
# Group by computed column
df['salary_bracket'] = pd.cut(df['Salary'], bins=[0, 55000, 70000, 100000], labels=['Low', 'Mid', 'High'])
by_bracket = df.groupby('salary_bracket')['Employee'].count()
print(by_bracket)

# Group by multiple conditions
df['high_exp'] = df['Experience'] > 5
by_dept_exp = df.groupby(['Department', 'high_exp'])['Salary'].mean()
print(by_dept_exp)

# Group by time periods
df['hire_date'] = pd.date_range('2020-01-01', periods=len(df), freq='ME')
df['hire_year'] = df['hire_date'].dt.year
by_year = df.groupby('hire_year')['Salary'].mean()
print(by_year)
\`\`\`

## Performance Optimization

\`\`\`python
# Use built-in aggregations (faster than custom functions)
# Slow
result = df.groupby('Department')['Salary'].agg(lambda x: x.mean())

# Fast
result = df.groupby('Department')['Salary'].mean()

# Use transform for element-wise operations
# Slow
df['dept_avg'] = df.apply(
    lambda row: df[df['Department'] == row['Department']]['Salary'].mean(),
    axis=1
)

# Fast
df['dept_avg'] = df.groupby('Department')['Salary'].transform('mean')

# Cache groupby object if reusing
grouped = df.groupby('Department')
result1 = grouped['Salary'].mean()
result2 = grouped['Experience'].mean()
# Better than calling groupby twice
\`\`\`

## Key Takeaways

1. **GroupBy** implements split-apply-combine pattern efficiently
2. **Aggregation** summarizes groups (reduces size)
3. **Transform** applies group-wise operations (maintains size)
4. **Filter** removes entire groups based on conditions
5. **Pivot tables** reshape data for analysis
6. **Multi-level indexing** handles complex groupings
7. **Named aggregations** improve code readability
8. **Built-in functions** are faster than custom lambdas

Master groupby and you can answer almost any "by category" question about your data!
`,
};
