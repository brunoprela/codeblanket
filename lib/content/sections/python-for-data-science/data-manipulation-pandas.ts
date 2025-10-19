/**
 * Section: Data Manipulation with Pandas
 * Module: Python for Data Science
 *
 * Covers filtering, sorting, adding/removing columns, apply/map/transform, string and datetime operations
 */

export const dataManipulationPandas = {
  id: 'data-manipulation-pandas',
  title: 'Data Manipulation with Pandas',
  content: `
# Data Manipulation with Pandas

## Introduction

Data manipulation is the heart of data analysis—transforming raw data into insights. Pandas provides a rich set of methods for filtering, sorting, transforming, and manipulating data efficiently. This section covers the essential techniques you'll use daily.

## Filtering and Querying

### Boolean Filtering

\`\`\`python
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Age': [25, 35, 30, 28, 42, 33],
    'Salary': [50000, 75000, 60000, 55000, 90000, 65000],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR']
})

# Single condition
high_earners = df[df['Salary'] > 60000]
print(high_earners)

# Multiple conditions (AND)
it_high_earners = df[(df['Department'] == 'IT') & (df['Salary'] > 55000)]
print(it_high_earners)

# Multiple conditions (OR)
it_or_senior = df[(df['Department'] == 'IT') | (df['Age'] > 35)]
print(it_or_senior)

# NOT condition
not_it = df[~(df['Department'] == 'IT')]
print(not_it)

# Between values
mid_career = df[df['Age'].between(28, 35)]
print(mid_career)
\`\`\`

### Using .query() Method

The \`.query()\` method provides a more readable syntax for complex filters:

\`\`\`python
# Equivalent to: df[(df['Age'] > 30) & (df['Salary'] > 60000)]
result = df.query('Age > 30 and Salary > 60000')
print(result)

# Using variables
min_salary = 60000
result = df.query('Salary > @min_salary')  # @ references Python variables
print(result)

# String matching
result = df.query('Department == "IT"')
print(result)

# Complex queries
result = df.query('(Age > 30 and Department == "IT") or Salary > 80000')
print(result)

# Query with method chaining
result = (df
    .query('Department == "IT"')
    .query('Salary > 55000')
    .query('Age < 40')
)
print(result)
\`\`\`

### Using .isin() for Multiple Values

\`\`\`python
# Filter for multiple departments
tech_depts = df[df['Department'].isin(['IT', 'Sales'])]
print(tech_depts)

# Exclude multiple values (NOT IN)
non_tech = df[~df['Department'].isin(['IT', 'Sales'])]
print(non_tech)

# Multiple columns
specific_people = df[df['Name'].isin(['Alice', 'Charlie', 'Eve'])]
print(specific_people)
\`\`\`

## Sorting and Ranking

### Sorting by Values

\`\`\`python
# Sort by single column
sorted_by_age = df.sort_values('Age')
print(sorted_by_age)

# Sort descending
sorted_salary_desc = df.sort_values('Salary', ascending=False)
print(sorted_salary_desc)

# Sort by multiple columns
sorted_multi = df.sort_values(['Department', 'Salary'], 
                               ascending=[True, False])
print(sorted_multi)

# Sort in place
df.sort_values('Age', inplace=True)

# Reset index after sorting
df = df.sort_values('Salary').reset_index(drop=True)
print(df)
\`\`\`

### Sorting by Index

\`\`\`python
# Sort by index
df_sorted = df.sort_index()

# Reverse order
df_sorted = df.sort_index(ascending=False)

# Sort columns (axis=1)
df_sorted = df.sort_index(axis=1)  # Alphabetical column order
print(df_sorted)
\`\`\`

### Ranking

\`\`\`python
# Add rank column
df['Salary_Rank'] = df['Salary'].rank(method='dense', ascending=False)
print(df[['Name', 'Salary', 'Salary_Rank']])

# Different ranking methods
df['rank_average'] = df['Salary'].rank(method='average')  # Average of tied ranks
df['rank_min'] = df['Salary'].rank(method='min')          # Minimum of tied ranks
df['rank_max'] = df['Salary'].rank(method='max')          # Maximum of tied ranks
df['rank_first'] = df['Salary'].rank(method='first')      # Order they appear
df['rank_dense'] = df['Salary'].rank(method='dense')      # Like min but no gaps

# Percentile ranking
df['Salary_Percentile'] = df['Salary'].rank(pct=True) * 100
print(df[['Name', 'Salary', 'Salary_Percentile']])
\`\`\`

## Adding and Removing Columns

### Adding Columns

\`\`\`python
# Add constant value
df['Country'] = 'USA'

# Add from list
df['Bonus'] = [5000, 7500, 6000, 5500, 9000, 6500]

# Calculate from existing columns
df['Total_Comp'] = df['Salary'] + df['Bonus']

# Conditional column
df['Senior'] = df['Age'] >= 35

# Multiple conditions with np.where
df['Level'] = np.where(df['Age'] < 30, 'Junior',
               np.where(df['Age'] < 40, 'Mid', 'Senior'))

# Multiple conditions with np.select
conditions = [
    df['Salary'] < 60000,
    df['Salary'] < 75000,
    df['Salary'] >= 75000
]
choices = ['Low', 'Medium', 'High']
df['Salary_Bracket'] = np.select(conditions, choices)

print(df)
\`\`\`

### Inserting Columns at Specific Position

\`\`\`python
# Insert at position 1
df.insert(1, 'Employee_ID', range(1, len(df) + 1))
print(df)
\`\`\`

### Removing Columns

\`\`\`python
# Drop single column
df_dropped = df.drop('Bonus', axis=1)
# or
df_dropped = df.drop(columns=['Bonus'])

# Drop multiple columns
df_dropped = df.drop(['Bonus', 'Country'], axis=1)

# Drop in place
df.drop('Bonus', axis=1, inplace=True)

# Drop columns matching pattern
cols_to_drop = [col for col in df.columns if 'rank_' in col]
df = df.drop(columns=cols_to_drop)
\`\`\`

### Renaming Columns

\`\`\`python
# Rename specific columns
df = df.rename(columns={'Salary': 'Annual_Salary', 'Age': 'Years'})

# Rename all columns
df.columns = ['name', 'years', 'annual_salary', 'dept']

# String manipulation on column names
df.columns = df.columns.str.lower()  # Lowercase
df.columns = df.columns.str.replace(' ', '_')  # Replace spaces
df.columns = df.columns.str.strip()  # Remove whitespace

# Rename index
df = df.rename(index={0: 'first', 1: 'second'})
\`\`\`

## Apply, Map, and Transform

### Using .apply()

Apply a function along an axis (column or row):

\`\`\`python
# Apply to single column (returns Series)
df['Name_Length'] = df['Name'].apply(len)
df['Name_Upper'] = df['Name'].apply(str.upper)

# Lambda functions
df['Salary_K'] = df['Salary'].apply(lambda x: x / 1000)

# Custom function
def classify_salary(salary):
    if salary < 60000:
        return 'Low'
    elif salary < 75000:
        return 'Medium'
    else:
        return 'High'

df['Salary_Class'] = df['Salary'].apply(classify_salary)

# Apply with multiple arguments
def adjust_salary(salary, factor=1.1):
    return salary * factor

df['Adjusted_Salary'] = df['Salary'].apply(adjust_salary, factor=1.15)

# Apply to multiple columns
df['Full_Info'] = df.apply(
    lambda row: f"{row['Name']} ({row['Age']}): \${row['Salary']:,}", 
    axis=1
)
print(df['Full_Info'])
\`\`\`

### Using .map()

Map values in a Series to other values:

\`\`\`python
# Map with dictionary
dept_map = {'IT': 'Technology', 'HR': 'Human Resources', 'Sales': 'Sales & Marketing'}
df['Department_Full'] = df['Department'].map(dept_map)

# Map with function
df['Age_Squared'] = df['Age'].map(lambda x: x ** 2)

# Map with Series
avg_salaries = df.groupby('Department')['Salary'].mean()
df['Dept_Avg_Salary'] = df['Department'].map(avg_salaries)
print(df[['Name', 'Department', 'Salary', 'Dept_Avg_Salary']])
\`\`\`

### Using .replace()

\`\`\`python
# Replace single value
df['Department'] = df['Department'].replace('IT', 'Information Technology')

# Replace multiple values
df['Department'] = df['Department'].replace({
    'IT': 'Information Technology',
    'HR': 'Human Resources'
})

# Replace with regex
df['Name'] = df['Name'].replace(r'[aeiou]', 'X', regex=True)

# Replace in entire DataFrame
df = df.replace({'IT': 'Tech', 'HR': 'People'})
\`\`\`

### Using .transform()

Transform returns a Series with the same index as the original:

\`\`\`python
# Normalize within group
df['Salary_Normalized'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Running calculations
df['Cumulative_Salary'] = df['Salary'].transform('cumsum')

# Fill missing with group mean
df['Salary_Filled'] = df.groupby('Department')['Salary'].transform(
    lambda x: x.fillna(x.mean())
)
\`\`\`

### Performance: apply() vs vectorization

\`\`\`python
# Slow: apply with lambda
df['Salary_Double_Slow'] = df['Salary'].apply(lambda x: x * 2)

# Fast: vectorized operation
df['Salary_Double_Fast'] = df['Salary'] * 2

# For complex logic, use np.where or np.select instead of apply
# Slow
df['Tax_Slow'] = df['Salary'].apply(lambda x: x * 0.25 if x > 60000 else x * 0.20)

# Fast
df['Tax_Fast'] = np.where(df['Salary'] > 60000, df['Salary'] * 0.25, df['Salary'] * 0.20)
\`\`\`

## String Operations

Pandas provides vectorized string operations via \`.str\`:

\`\`\`python
# Sample data
df = pd.DataFrame({
    'Name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
    'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com'],
    'Phone': ['555-1234', '555-5678', '555-9012']
})

# Case conversion
df['Name_Upper'] = df['Name'].str.upper()
df['Name_Lower'] = df['Name'].str.lower()
df['Name_Title'] = df['Name'].str.title()

# String methods
df['First_Name'] = df['Name'].str.split().str[0]
df['Last_Name'] = df['Name'].str.split().str[-1]

# Contains
df['Has_Brown'] = df['Name'].str.contains('Brown')

# Extract with regex
df['Username'] = df['Email'].str.extract(r'([^@]+)@')

# Replace
df['Phone_Clean'] = df['Phone'].str.replace('-', '')

# String length
df['Name_Length'] = df['Name'].str.len()

# Strip whitespace
df['Name_Strip'] = df['Name'].str.strip()

# Substring
df['Name_First3'] = df['Name'].str[:3]

# Startswith/Endswith
df['Starts_Alice'] = df['Name'].str.startswith('Alice')
df['Ends_Com'] = df['Email'].str.endswith('.com')

# Pad strings
df['Phone_Padded'] = df['Phone'].str.pad(15, fillchar='0')

# Join strings
df['Full_Contact'] = df['Name'].str.cat([df['Email'], df['Phone']], sep=' | ')

print(df)
\`\`\`

### String Pattern Matching

\`\`\`python
# Extract all matches
text = pd.Series(['My phone: 555-1234', 'Call 555-5678 or 555-9012'])
phones = text.str.extractall(r'(\\d{3}-\\d{4})')
print(phones)

# Find all occurrences
text = pd.Series(['apple banana apple', 'cherry apple'])
count = text.str.count('apple')
print(count)  # [2, 1]

# Replace with regex
df['Email_Masked'] = df['Email'].str.replace(r'@.*', '@xxx.com', regex=True)
\`\`\`

## Datetime Operations

Working with dates and times:

\`\`\`python
# Create sample data
df = pd.DataFrame({
    'Date': ['2024-01-15', '2024-02-20', '2024-03-10'],
    'Sales': [1000, 1500, 1200]
})

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'].dtype)  # datetime64[ns]

# Extract components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
df['Day_Name'] = df['Date'].dt.day_name()
df['Month_Name'] = df['Date'].dt.month_name()
df['Quarter'] = df['Date'].dt.quarter
df['Week_of_Year'] = df['Date'].dt.isocalendar().week

# Date arithmetic
df['Next_Week'] = df['Date'] + pd.Timedelta(days=7)
df['Last_Month'] = df['Date'] - pd.DateOffset(months=1)

# Date differences
df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days

# Format dates
df['Date_Formatted'] = df['Date'].dt.strftime('%Y-%m-%d')
df['Date_Pretty'] = df['Date'].dt.strftime('%B %d, %Y')

print(df)
\`\`\`

### Date Ranges and Periods

\`\`\`python
# Generate date range
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
print(f"Days in 2024: {len(date_range)}")

# Business days only
business_days = pd.bdate_range(start='2024-01-01', end='2024-01-31')
print(f"Business days in Jan 2024: {len(business_days)}")

# Monthly periods
months = pd.period_range(start='2024-01', end='2024-12', freq='M')
print(months)

# Create DataFrame with date index
df_timeseries = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'Value': np.random.randn(100)
})
df_timeseries = df_timeseries.set_index('Date')
\`\`\`

## Practical Examples

### Example 1: Employee Data Analysis

\`\`\`python
# Load and clean employee data
employees = pd.DataFrame({
    'employee_id': range(1, 101),
    'name': [f'Employee{i}' for i in range(1, 101)],
    'hire_date': pd.date_range('2020-01-01', periods=100, freq='W'),
    'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], 100),
    'salary': np.random.randint(40000, 120000, 100)
})

# Add tenure
employees['tenure_days'] = (pd.Timestamp.now() - employees['hire_date']).dt.days
employees['tenure_years'] = employees['tenure_days'] / 365.25

# Classify employees
employees['level'] = pd.cut(employees['tenure_years'], 
                             bins=[0, 1, 3, 10], 
                             labels=['Junior', 'Mid', 'Senior'])

# Department averages
employees['dept_avg_salary'] = employees.groupby('department')['salary'].transform('mean')
employees['salary_vs_dept_avg'] = employees['salary'] - employees['dept_avg_salary']

# Performance score (example)
employees['performance'] = np.random.choice(['Low', 'Medium', 'High'], 100)

# Filter high performers in IT
high_performers = employees.query('department == "IT" and performance == "High"')

print(f"High performing IT employees: {len(high_performers)}")
print(high_performers[['name', 'salary', 'tenure_years']].head())
\`\`\`

### Example 2: Sales Data Processing

\`\`\`python
# Sales transactions
sales = pd.DataFrame({
    'transaction_id': range(1, 1001),
    'date': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'product': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'quantity': np.random.randint(1, 10, 1000),
    'price': np.random.uniform(10, 100, 1000)
})

# Calculate revenue
sales['revenue'] = sales['quantity'] * sales['price']

# Add time features
sales['hour'] = sales['date'].dt.hour
sales['day_of_week'] = sales['date'].dt.day_name()
sales['is_weekend'] = sales['date'].dt.dayofweek >= 5

# Classify by revenue
sales['revenue_category'] = pd.cut(sales['revenue'], 
                                    bins=[0, 100, 500, float('inf')],
                                    labels=['Low', 'Medium', 'High'])

# Running totals
sales = sales.sort_values('date')
sales['cumulative_revenue'] = sales['revenue'].cumsum()

# Day over day growth
sales['daily_revenue'] = sales.groupby(sales['date'].dt.date)['revenue'].transform('sum')

print(sales.head())
print(f"\\nTotal revenue: \${sales['revenue'].sum():,.2f}")
print(f"Average transaction: \${sales['revenue'].mean():.2f}")
\`\`\`

## Key Takeaways

1. **Filtering**: Boolean indexing, .query(), .isin() for flexible data selection
2. **Sorting**: sort_values() for data, sort_index() for index, rank() for rankings
3. **Columns**: Easy addition, removal, renaming with multiple methods
4. **Apply family**: apply(), map(), transform() for custom transformations
5. **Vectorization**: Prefer vectorized operations over apply() for performance
6. **Strings**: Powerful .str accessor for vectorized string operations
7. **Dates**: .dt accessor for datetime operations, date arithmetic
8. **Method chaining**: Combine operations for readable data pipelines

Master these techniques and you'll be able to transform any dataset efficiently!
`,
};
