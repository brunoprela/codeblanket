/**
 * Section: Merging & Joining Data
 * Module: Python for Data Science
 *
 * Covers concatenation, merge types, join operations, handling key conflicts, and complex data integration
 */

export const mergingJoiningData = {
  id: 'merging-joining-data',
  title: 'Merging & Joining Data',
  content: `
# Merging & Joining Data

## Introduction

In real-world data analysis, data often comes from multiple sources and needs to be combined. Pandas provides powerful tools for merging, joining, and concatenating datasets. Understanding these operations is essential for data integration and preparation.

**Key Operations:**
- **Concatenate**: Stack DataFrames vertically or horizontally
- **Merge**: SQL-style joins based on key columns
- **Join**: Merge using index
- **Combine**: Element-wise combination with custom logic

\`\`\`python
import pandas as pd
import numpy as np
\`\`\`

## Concatenation

### Vertical Concatenation (Stack Rows)

\`\`\`python
# Create sample DataFrames
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2'],
    'C': ['C0', 'C1', 'C2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5'],
    'C': ['C3', 'C4', 'C5']
})

# Simple concatenation (vertical)
result = pd.concat([df1, df2])
print(result)
#     A   B   C
# 0  A0  B0  C0
# 1  A1  B1  C1
# 2  A2  B2  C2
# 0  A3  B3  C3  # Note: Index repeats!
# 1  A4  B4  C4
# 2  A5  B5  C5

# Reset index
result = pd.concat([df1, df2], ignore_index=True)
print(result)
#     A   B   C
# 0  A0  B0  C0
# 1  A1  B1  C1
# 2  A2  B2  C2
# 3  A3  B3  C3
# 4  A4  B4  C4
# 5  A5  B5  C5

# Add keys to identify source
result = pd.concat([df1, df2], keys=['df1', 'df2'])
print(result)
#         A   B   C
# df1 0  A0  B0  C0
#     1  A1  B1  C1
#     2  A2  B2  C2
# df2 0  A3  B3  C3
#     1  A4  B4  C4
#     2  A5  B5  C5
\`\`\`

### Horizontal Concatenation (Stack Columns)

\`\`\`python
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})

df2 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
})

# Concatenate horizontally
result = pd.concat([df1, df2], axis=1)
print(result)
#     A   B   C   D
# 0  A0  B0  C0  D0
# 1  A1  B1  C1  D1
# 2  A2  B2  C2  D2
\`\`\`

### Handling Missing Columns

\`\`\`python
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})

df2 = pd.DataFrame({
    'B': ['B3', 'B4', 'B5'],
    'C': ['C3', 'C4', 'C5']
})

# Outer join (keep all columns)
result = pd.concat([df1, df2], ignore_index=True)
print(result)
#      A   B    C
# 0   A0  B0  NaN
# 1   A1  B1  NaN
# 2   A2  B2  NaN
# 3  NaN  B3   C3
# 4  NaN  B4   C4
# 5  NaN  B5   C5

# Inner join (keep only common columns)
result = pd.concat([df1, df2], ignore_index=True, join='inner')
print(result)
#     B
# 0  B0
# 1  B1
# 2  B2
# 3  B3
# 4  B4
# 5  B5
\`\`\`

## Merge Operations (SQL-style Joins)

### Inner Join

\`\`\`python
# Create sample DataFrames
employees = pd.DataFrame({
    'employee_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 30, 40],
    'dept_name': ['IT', 'HR', 'Sales', 'Marketing']
})

# Inner join (only matching keys)
result = pd.merge(employees, departments, on='dept_id')
print(result)
#    employee_id     name  dept_id dept_name
# 0            1    Alice       10        IT
# 1            3  Charlie       10        IT
# 2            2      Bob       20        HR
# 3            4    David       30     Sales
# Note: Marketing department (40) is excluded (no employees)
\`\`\`

### Left Join

\`\`\`python
# Left join (keep all left records)
result = pd.merge(employees, departments, on='dept_id', how='left')
print(result)
#    employee_id     name  dept_id dept_name
# 0            1    Alice       10        IT
# 1            2      Bob       20        HR
# 2            3  Charlie       10        IT
# 3            4    David       30     Sales
# All employees included, even if no matching department
\`\`\`

### Right Join

\`\`\`python
# Right join (keep all right records)
result = pd.merge(employees, departments, on='dept_id', how='right')
print(result)
#    employee_id     name  dept_id   dept_name
# 0          1.0    Alice       10          IT
# 1          3.0  Charlie       10          IT
# 2          2.0      Bob       20          HR
# 3          4.0    David       30       Sales
# 4          NaN      NaN       40   Marketing
# Marketing included with NaN for employee fields
\`\`\`

### Outer Join (Full Join)

\`\`\`python
# Outer join (keep all records from both)
result = pd.merge(employees, departments, on='dept_id', how='outer')
print(result)
#    employee_id     name  dept_id   dept_name
# 0          1.0    Alice       10          IT
# 1          3.0  Charlie       10          IT
# 2          2.0      Bob       20          HR
# 3          4.0    David       30       Sales
# 4          NaN      NaN       40   Marketing
# All employees and all departments
\`\`\`

### Merge on Multiple Keys

\`\`\`python
# Create DataFrames with composite keys
sales = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250]
})

targets = pd.DataFrame({
    'region': ['East', 'East', 'West'],
    'product': ['A', 'B', 'A'],
    'target': [120, 180, 140]
})

# Merge on multiple columns
result = pd.merge(sales, targets, on=['region', 'product'], how='left')
print(result)
#   region product  sales  target
# 0   East       A    100   120.0
# 1   East       B    200   180.0
# 2   West       A    150   140.0
# 3   West       B    250     NaN  # No target for West-B

# Compare performance to target
result['vs_target'] = result['sales'] - result['target']
result['pct_of_target'] = result['sales'] / result['target'] * 100
print(result)
\`\`\`

### Merge with Different Column Names

\`\`\`python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'manager_id': [None, 1, 1]
})

# Self-join to get manager names
result = pd.merge(
    employees,
    employees,
    left_on='manager_id',
    right_on='emp_id',
    how='left',
    suffixes=('', '_manager')
)
print(result[['emp_id', 'name', 'name_manager']])
#    emp_id     name name_manager
# 0       1    Alice          NaN
# 1       2      Bob        Alice
# 2       3  Charlie        Alice
\`\`\`

## Joining (Merge on Index)

\`\`\`python
# Create DataFrames with meaningful indices
left = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index=['K0', 'K1', 'K2'])

right = pd.DataFrame({
    'C': ['C0', 'C2', 'C3'],
    'D': ['D0', 'D2', 'D3']
}, index=['K0', 'K2', 'K3'])

# Join on index
result = left.join(right)
print(result)
#      A   B    C    D
# K0  A0  B0   C0   D0
# K1  A1  B1  NaN  NaN
# K2  A2  B2   C2   D2
# Default is left join

# Inner join
result = left.join(right, how='inner')
print(result)
#      A   B   C   D
# K0  A0  B0  C0  D0
# K2  A2  B2  C2  D2

# Join on column vs index
left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2'],
    'A': ['A0', 'A1', 'A2']
})

right = pd.DataFrame({
    'C': ['C0', 'C2', 'C3'],
    'D': ['D0', 'D2', 'D3']
}, index=['K0', 'K2', 'K3'])

result = left.join(right, on='key')
print(result)
#   key   A    C    D
# 0  K0  A0   C0   D0
# 1  K1  A1  NaN  NaN
# 2  K2  A2   C2   D2
\`\`\`

## Handling Duplicate Keys

### One-to-Many Merge

\`\`\`python
# One department, many employees
departments = pd.DataFrame({
    'dept_id': [10, 20, 30],
    'dept_name': ['IT', 'HR', 'Sales']
})

employees = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'dept_id': [10, 10, 20, 10, 30]
})

result = pd.merge(employees, departments, on='dept_id')
print(result)
#    employee_id     name  dept_id dept_name
# 0            1    Alice       10        IT
# 1            2      Bob       10        IT
# 2            4    David       10        IT
# 3            3  Charlie       20        HR
# 4            5      Eve       30     Sales
# Department info repeated for each employee
\`\`\`

### Many-to-Many Merge

\`\`\`python
# Students can take multiple courses, courses have multiple students
students = pd.DataFrame({
    'course_id': ['CS101', 'CS101', 'CS102', 'CS102'],
    'student_id': [1, 2, 2, 3],
    'student_name': ['Alice', 'Bob', 'Bob', 'Charlie']
})

courses = pd.DataFrame({
    'course_id': ['CS101', 'CS101', 'CS102'],
    'instructor': ['Prof. Smith', 'Prof. Jones', 'Prof. Smith']
})

# Many-to-many creates all combinations
result = pd.merge(students, courses, on='course_id')
print(result)
# Creates row for each student-instructor pair
\`\`\`

### Indicator Column

\`\`\`python
# Track which DataFrame each row came from
result = pd.merge(employees, departments, on='dept_id', how='outer', indicator=True)
print(result)
#    employee_id     name  dept_id   dept_name      _merge
# 0          1.0    Alice       10          IT        both
# 1          2.0      Bob       20          HR        both
# 2          3.0  Charlie       10          IT        both
# 3          4.0    David       30       Sales        both
# 4          NaN      NaN       40   Marketing  right_only

# Identify unmatched records
unmatched = result[result['_merge'] != 'both']
print(f"Unmatched records: {len(unmatched)}")
\`\`\`

## Practical Examples

### Example 1: Customer Orders Analysis

\`\`\`python
# Customer information
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Boston'],
    'signup_date': pd.to_datetime(['2024-01-15', '2024-01-20', '2024-02-01', 
                                     '2024-02-10', '2024-03-05'])
})

# Orders
orders = pd.DataFrame({
    'order_id': range(1, 11),
    'customer_id': [1, 2, 1, 3, 2, 4, 1, 5, 3, 2],
    'order_date': pd.to_datetime(['2024-01-20', '2024-01-25', '2024-02-01',
                                   '2024-02-05', '2024-02-10', '2024-02-15',
                                   '2024-03-01', '2024-03-10', '2024-03-15', '2024-03-20']),
    'amount': [100, 250, 75, 300, 150, 200, 125, 175, 225, 100]
})

# Order items
order_items = pd.DataFrame({
    'order_id': [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'product_id': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A'],
    'quantity': [2, 1, 3, 1, 1, 2, 2, 1, 3, 1, 2, 1]
})

# 1. Merge customers with orders
customer_orders = pd.merge(orders, customers, on='customer_id', how='left')
print("Customer Orders:")
print(customer_orders.head())

# 2. Calculate customer statistics
customer_stats = customer_orders.groupby('customer_id').agg({
    'order_id': 'count',
    'amount': ['sum', 'mean']
})
customer_stats.columns = ['order_count', 'total_spent', 'avg_order']
customer_stats = customer_stats.reset_index()

# Merge back with customer info
customer_summary = pd.merge(customers, customer_stats, on='customer_id', how='left')
customer_summary['total_spent'].fillna(0, inplace=True)
customer_summary['order_count'].fillna(0, inplace=True)
print("\\nCustomer Summary:")
print(customer_summary)

# 3. Product analysis - which products in which orders
order_details = pd.merge(
    pd.merge(orders, order_items, on='order_id'),
    customers[['customer_id', 'name', 'city']],
    on='customer_id'
)
print("\\nOrder Details:")
print(order_details.head())

# 4. Product popularity by city
product_by_city = order_details.groupby(['city', 'product_id'])['quantity'].sum().unstack(fill_value=0)
print("\\nProduct Popularity by City:")
print(product_by_city)
\`\`\`

### Example 2: Time Series Merge with Tolerance

\`\`\`python
# Stock prices (frequent)
prices = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01 09:00', periods=100, freq='min'),
    'ticker': 'AAPL',
    'price': np.random.uniform(150, 160, 100)
})

# News events (sporadic)
news = pd.DataFrame({
    'timestamp': pd.to_datetime(['2024-01-01 09:15', '2024-01-01 10:30', '2024-01-01 11:45']),
    'headline': ['Product Launch', 'Earnings Beat', 'CEO Statement']
})

# Merge with nearest timestamp (within 5 minutes)
result = pd.merge_asof(
    prices.sort_values('timestamp'),
    news.sort_values('timestamp'),
    on='timestamp',
    direction='backward',  # Use most recent news
    tolerance=pd.Timedelta('5min')
)

# Show prices around news events
news_impact = result[result['headline'].notna()]
print("Prices around news events:")
print(news_impact[['timestamp', 'price', 'headline']])
\`\`\`

### Example 3: Handling Conflicting Column Names

\`\`\`python
# Two tables with overlapping column names
sales_q1 = pd.DataFrame({
    'product_id': ['A', 'B', 'C'],
    'sales': [1000, 1500, 1200],
    'units': [100, 150, 120]
})

sales_q2 = pd.DataFrame({
    'product_id': ['A', 'B', 'C'],
    'sales': [1100, 1400, 1300],
    'units': [110, 140, 130]
})

# Merge with suffixes
result = pd.merge(
    sales_q1,
    sales_q2,
    on='product_id',
    suffixes=('_q1', '_q2')
)
print(result)
#   product_id  sales_q1  units_q1  sales_q2  units_q2
# 0          A      1000       100      1100       110
# 1          B      1500       150      1400       140
# 2          C      1200       120      1300       130

# Calculate growth
result['sales_growth'] = (result['sales_q2'] - result['sales_q1']) / result['sales_q1'] * 100
result['units_growth'] = (result['units_q2'] - result['units_q1']) / result['units_q1'] * 100
print("\\nGrowth Analysis:")
print(result[['product_id', 'sales_growth', 'units_growth']])
\`\`\`

## Validation and Quality Checks

\`\`\`python
# Check merge results
def validate_merge(left, right, result, merge_keys):
    """Validate merge operation results"""
    print(f"Left DataFrame: {len(left)} rows")
    print(f"Right DataFrame: {len(right)} rows")
    print(f"Merged DataFrame: {len(result)} rows")
    
    # Check for unexpected duplicates
    if len(result) > len(left) and len(result) > len(right):
        print("⚠️  Warning: Result has more rows than both inputs (possible many-to-many)")
    
    # Check for missing values in merge keys
    for key in merge_keys:
        if result[key].isnull().any():
            print(f"⚠️  Warning: Missing values in merge key '{key}'")
    
    # Check merge quality
    if '_merge' in result.columns:
        merge_counts = result['_merge'].value_counts()
        print("\\nMerge distribution:")
        print(merge_counts)

# Example usage
result = pd.merge(employees, departments, on='dept_id', how='outer', indicator=True)
validate_merge(employees, departments, result, ['dept_id'])
\`\`\`

## Performance Considerations

\`\`\`python
# For large DataFrames

# 1. Ensure merge keys are indexed
df1 = df1.set_index('key')
df2 = df2.set_index('key')
result = df1.join(df2)  # Faster than merge on unindexed columns

# 2. Use categorical for repeated values
df['category'] = df['category'].astype('category')

# 3. Filter before merging
df1_filtered = df1[df1['date'] > '2024-01-01']
result = pd.merge(df1_filtered, df2, on='key')

# 4. Use merge_asof for sorted time series
# Much faster than merge for time-based joins
result = pd.merge_asof(df1.sort_values('time'), df2.sort_values('time'), on='time')

# 5. Consider chunking for very large datasets
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    merged_chunk = pd.merge(chunk, reference_df, on='key')
    chunks.append(merged_chunk)
result = pd.concat(chunks, ignore_index=True)
\`\`\`

## Common Pitfalls

\`\`\`python
# Pitfall 1: Forgetting to check for duplicates
# Always check for duplicate keys before merging
print(df1['key'].duplicated().sum())  # Should be 0 for one-to-one
print(df2['key'].duplicated().sum())

# Pitfall 2: Wrong join type
# Left join when you need inner join
result_left = pd.merge(df1, df2, on='key', how='left')  # Keeps all df1
result_inner = pd.merge(df1, df2, on='key', how='inner')  # Only matching

# Pitfall 3: Not using indicator
# Always use indicator for outer joins to track sources
result = pd.merge(df1, df2, on='key', how='outer', indicator=True)

# Pitfall 4: Ignoring suffixes with conflicting names
# Specify meaningful suffixes
result = pd.merge(df1, df2, on='key', suffixes=('_current', '_target'))

# Pitfall 5: Merging unsorted data with merge_asof
# merge_asof requires sorted data
df1 = df1.sort_values('timestamp')
df2 = df2.sort_values('timestamp')
result = pd.merge_asof(df1, df2, on='timestamp')
\`\`\`

## Key Takeaways

1. **concat()**: Stack DataFrames vertically (axis=0) or horizontally (axis=1)
2. **merge()**: SQL-style joins with flexible key matching
3. **join()**: Quick merge on index
4. **Join types**: inner, left, right, outer - choose based on which records to keep
5. **Indicator**: Use indicator=True to track merge sources
6. **Suffixes**: Handle overlapping column names
7. **merge_asof**: Efficient for time series and nearest-match scenarios
8. **Validation**: Always check merge results for unexpected duplicates or missing data
9. **Performance**: Index merge keys, filter before merging, use categorical types

Master merging and you can integrate data from any source!
`,
};
