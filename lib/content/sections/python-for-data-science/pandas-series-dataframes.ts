/**
 * Section: Pandas Series & DataFrames
 * Module: Python for Data Science
 *
 * Covers Series and DataFrame creation, indexing, selection, data types, and real-world data loading
 */

export const pandasSeriesDataFrames = {
  id: 'pandas-series-dataframes',
  title: 'Pandas Series & DataFrames',
  content: `
# Pandas Series & DataFrames

## Introduction

Pandas is the de facto standard for data manipulation and analysis in Python. Built on top of NumPy, it provides two primary data structures: **Series** (1-dimensional) and **DataFrame** (2-dimensional), along with powerful tools for data cleaning, transformation, and analysis.

**Why Pandas?**
- **Labeled data**: Named columns and indices (unlike NumPy\'s positional access)
- **Mixed types**: Different columns can have different data types
- **Missing data**: Built-in handling of NaN values
- **Rich functionality**: Grouping, merging, reshaping, time series
- **Integration**: Works seamlessly with NumPy, matplotlib, scikit-learn

\`\`\`python
import pandas as pd
import numpy as np

# Check version
print(f"Pandas version: {pd.__version__}")
\`\`\`

## Pandas Series

A Series is a one-dimensional labeled array, similar to a NumPy array but with an index.

### Creating Series

\`\`\`python
# From Python list
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# Output:
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64

# Components
print(f"Values: {s.values}")  # NumPy array
print(f"Index: {s.index}")    # RangeIndex (start=0, stop=5, step=1)

# With custom index
s = pd.Series([10, 20, 30, 40, 50],
              index=['a', 'b', 'c', 'd', 'e'])
print(s)
# a    10
# b    20
# c    30
# d    40
# e    50

# With name
s = pd.Series([10, 20, 30], name='prices')
print(s)
# 0    10
# 1    20
# 2    30
# Name: prices, dtype: int64
\`\`\`

### Creating Series from Dictionaries

\`\`\`python
# Dictionary becomes index: value pairs
data = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 310.0, 'AMZN': 3300.0}
stocks = pd.Series (data)
print(stocks)
# AAPL     150.0
# GOOGL   2800.0
# MSFT     310.0
# AMZN    3300.0
# dtype: float64

# Dictionary keys become index automatically
print(f"Index: {stocks.index.tolist()}")  # ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
\`\`\`

### Creating Series from Scalars

\`\`\`python
# Repeat scalar value
s = pd.Series(100, index=['a', 'b', 'c', 'd'])
print(s)
# a    100
# b    100
# c    100
# d    100
# dtype: int64
\`\`\`

### Series Operations

\`\`\`python
prices = pd.Series([100, 105, 102, 108, 110],
                   index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])

# Arithmetic operations
print(f"Prices + 10:\\n{prices + 10}")
print(f"Prices * 1.1:\\n{prices * 1.1}")

# Statistical operations
print(f"Mean: {prices.mean():.2f}")
print(f"Std: {prices.std():.2f}")
print(f"Min: {prices.min()}, Max: {prices.max()}")

# Boolean indexing
high_prices = prices[prices > 105]
print(f"High prices:\\n{high_prices}")

# Alignment by index
prices2 = pd.Series([102, 103, 104], index=['Tue', 'Wed', 'Sat'])
combined = prices + prices2
print(f"Combined (aligned):\\n{combined}")
# Mon    NaN  (not in prices2)
# Tue    208.0
# Wed    206.0
# Thu    NaN  (not in prices2)
# Fri    NaN  (not in prices2)
# Sat    NaN  (not in prices)
\`\`\`

## Pandas DataFrame

A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. Think of it as a spreadsheet or SQL table.

### Creating DataFrames from Dictionaries

\`\`\`python
# Dictionary of lists (columns)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 75000, 55000],
    'Department': ['IT', 'HR', 'IT', 'Sales']
}
df = pd.DataFrame (data)
print(df)
#       Name  Age  Salary Department
# 0    Alice   25   50000         IT
# 1      Bob   30   60000         HR
# 2  Charlie   35   75000         IT
# 3    David   28   55000      Sales

# Specify index
df = pd.DataFrame (data, index=['emp1', 'emp2', 'emp3', 'emp4'])
print(df)
#           Name  Age  Salary Department
# emp1     Alice   25   50000         IT
# emp2       Bob   30   60000         HR
# emp3   Charlie   35   75000         IT
# emp4     David   28   55000      Sales

# Select specific columns
df = pd.DataFrame (data, columns=['Name', 'Age'])
print(df)
#       Name  Age
# 0    Alice   25
# 1      Bob   30
# 2  Charlie   35
# 3    David   28
\`\`\`

### Creating DataFrames from NumPy Arrays

\`\`\`python
# Random data
data = np.random.randn(4, 3)
df = pd.DataFrame (data,
                  columns=['A', 'B', 'C'],
                  index=['row1', 'row2', 'row3', 'row4'])
print(df)
#              A         B         C
# row1  0.496714 -0.138264  0.647689
# row2  1.523030 -0.234153 -0.234137
# row3  1.579213  0.767435 -0.469474
# row4  0.542560 -0.463418 -0.465730
\`\`\`

### Creating DataFrames from Lists of Dictionaries

\`\`\`python
# Each dictionary is a row
data = [
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Bob', 'age': 30, 'city': 'LA'},
    {'name': 'Charlie', 'age': 35}  # Missing 'city'
]
df = pd.DataFrame (data)
print(df)
#       name  age city
# 0    Alice   25  NYC
# 1      Bob   30   LA
# 2  Charlie   35  NaN  # Automatically fills missing with NaN
\`\`\`

### Creating DataFrames from CSV Files

\`\`\`python
# Read CSV
df = pd.read_csv('data.csv')

# Common parameters
df = pd.read_csv(
    'data.csv',
    sep=',',                    # Delimiter (default: ',')
    header=0,                   # Row number for column names
    index_col=0,                # Column to use as index
    names=['A', 'B', 'C'],      # Custom column names
    usecols=['A', 'C'],         # Read only specific columns
    nrows=1000,                 # Read only first N rows
    skiprows=5,                 # Skip first N rows
    na_values=['NA', 'missing'] # Additional NaN values
)

# Example with sample data
import io

csv_data = """Date,Open,High,Low,Close,Volume
2024-01-01,100,105,99,103,1000000
2024-01-02,103,108,102,107,1200000
2024-01-03,107,110,105,108,900000"""

df = pd.read_csv (io.StringIO(csv_data))
print(df)
#          Date  Open  High  Low  Close   Volume
# 0  2024-01-01   100   105   99    103  1000000
# 1  2024-01-02   103   108  102    107  1200000
# 2  2024-01-03   107   110  105    108   900000
\`\`\`

## DataFrame Structure

### Basic Attributes

\`\`\`python
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Salary': [50000, 60000, 75000, 55000, 68000],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'IT']
}
df = pd.DataFrame (data)

# Shape (rows, columns)
print(f"Shape: {df.shape}")  # (5, 4)

# Dimensions
print(f"Dimensions: {df.ndim}")  # 2

# Size (total elements)
print(f"Size: {df.size}")  # 20

# Column names
print(f"Columns: {df.columns.tolist()}")  # ['Name', 'Age', 'Salary', 'Department']

# Index
print(f"Index: {df.index.tolist()}")  # [0, 1, 2, 3, 4]

# Data types
print(f"Dtypes:\\n{df.dtypes}")
# Name          object
# Age            int64
# Salary         int64
# Department    object
# dtype: object

# Memory usage
print(f"Memory:\\n{df.memory_usage()}")

# Quick info
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   Name        5 non-null      object
#  1   Age         5 non-null      int64
#  2   Salary      5 non-null      int64
#  3   Department  5 non-null      object
# dtypes: int64(2), object(2)
# memory usage: 288.0+ bytes
\`\`\`

### Viewing Data

\`\`\`python
# First N rows
print(df.head())  # Default: 5 rows
print(df.head(3))  # First 3 rows

# Last N rows
print(df.tail())  # Default: 5 rows
print(df.tail(2))  # Last 2 rows

# Random sample
print(df.sample(3))  # 3 random rows
print(df.sample (frac=0.5))  # 50% of rows

# Display settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
\`\`\`

## Indexing and Selection

### Selecting Columns

\`\`\`python
# Single column (returns Series)
ages = df['Age']
print(type (ages))  # <class 'pandas.core.series.Series'>
print(ages)

# Alternative syntax (only for valid identifiers)
ages = df.Age
print(ages)

# Multiple columns (returns DataFrame)
subset = df[['Name', 'Age']]
print(type (subset))  # <class 'pandas.core.frame.DataFrame'>
print(subset)

# Column reordering
reordered = df[['Salary', 'Age', 'Name']]
print(reordered)
\`\`\`

### Selecting Rows by Position (.iloc)

\`\`\`python
# Single row (returns Series)
first_row = df.iloc[0]
print(first_row)
# Name          Alice
# Age              25
# Salary        50000
# Department       IT
# Name: 0, dtype: object

# Multiple rows (returns DataFrame)
first_three = df.iloc[:3]
print(first_three)

# Specific rows
subset = df.iloc[[0, 2, 4]]
print(subset)

# Rows and columns by position
subset = df.iloc[0:3, 1:3]  # Rows 0-2, Columns 1-2
print(subset)
#    Age  Salary
# 0   25   50000
# 1   30   60000
# 2   35   75000

# Single element
value = df.iloc[0, 1]  # Row 0, Column 1
print(f"Value: {value}")  # 25
\`\`\`

### Selecting Rows by Label (.loc)

\`\`\`python
# Set custom index
df_indexed = df.set_index('Name')
print(df_indexed)
#          Age  Salary Department
# Name
# Alice     25   50000         IT
# Bob       30   60000         HR
# Charlie   35   75000         IT
# David     28   55000      Sales
# Eve       32   68000         IT

# Select by index label
alice_data = df_indexed.loc['Alice']
print(alice_data)

# Multiple rows by label
subset = df_indexed.loc[['Alice', 'Charlie']]
print(subset)

# Rows and columns by label
subset = df_indexed.loc['Alice':'Charlie', 'Age':'Salary']
print(subset)
#          Age  Salary
# Name
# Alice     25   50000
# Bob       30   60000
# Charlie   35   75000

# Boolean indexing with loc
high_salary = df_indexed.loc[df_indexed['Salary'] > 60000]
print(high_salary)
\`\`\`

### Boolean Indexing

\`\`\`python
# Simple condition
high_salary = df[df['Salary'] > 60000]
print(high_salary)

# Multiple conditions (use & and |, not 'and'/'or')
it_high_salary = df[(df['Department'] == 'IT') & (df['Salary'] > 60000)]
print(it_high_salary)

# Using .isin()
it_or_sales = df[df['Department'].isin(['IT', 'Sales'])]
print(it_or_sales)

# String methods
names_with_a = df[df['Name'].str.contains('a', case=False)]
print(names_with_a)

# Negation
not_it = df[~(df['Department'] == 'IT')]  # ~ is NOT
print(not_it)
\`\`\`

## Data Types

### Viewing Data Types

\`\`\`python
print(df.dtypes)
# Name          object
# Age            int64
# Salary         int64
# Department    object
# dtype: object

# Check specific column
print(f"Age dtype: {df['Age'].dtype}")  # int64
\`\`\`

### Converting Data Types

\`\`\`python
# Convert to numeric
df['Age'] = df['Age'].astype (np.float64)
print(df['Age'].dtype)  # float64

# Convert to string
df['Salary'] = df['Salary'].astype (str)
print(df['Salary'].dtype)  # object

# Convert back to numeric (handles errors)
df['Salary'] = pd.to_numeric (df['Salary'], errors='coerce')
# errors='coerce': invalid values become NaN
# errors='ignore': leave unchanged
# errors='raise': raise exception

# Convert to category (saves memory for repeated values)
df['Department'] = df['Department'].astype('category')
print(df['Department'].dtype)  # category
print(f"Categories: {df['Department'].cat.categories}")

# Memory savings
print(f"\\nMemory usage:\\n{df.memory_usage()}")
\`\`\`

### Category Data Type

\`\`\`python
# Categorical data (efficient for repeated strings)
df['Department'] = df['Department'].astype('category')

# Benefits:
# 1. Saves memory (stores integers + mapping)
# 2. Enables ordering
# 3. Faster groupby operations

# Ordered categories
df['Size'] = pd.Categorical(['S', 'M', 'L', 'XL', 'M'],
                             categories=['S', 'M', 'L', 'XL'],
                             ordered=True)
print(df['Size'])
# 0     S
# 1     M
# 2     L
# 3    XL
# 4     M
# dtype: category
# Categories (4, object): ['S' < 'M' < 'L' < 'XL']

# Now comparisons work
print(df['Size'] > 'M')
# 0    False
# 1    False
# 2     True
# 3     True
# 4    False
# dtype: bool
\`\`\`

## Column and Row Operations

### Adding Columns

\`\`\`python
# New column from scalar
df['Country'] = 'USA'

# New column from list
df['Bonus'] = [5000, 6000, 7500, 5500, 6800]

# New column from calculation
df['Total_Comp'] = df['Salary'] + df['Bonus']

# New column from function
df['Tax'] = df['Salary'] * 0.25

# Conditional column
df['Senior'] = df['Age'] >= 30

# Using apply
df['Name_Length'] = df['Name'].apply (len)
df['Tax_Bracket'] = df['Salary'].apply (lambda x: 'High' if x > 60000 else 'Low')

print(df)
\`\`\`

### Removing Columns

\`\`\`python
# Drop columns
df_dropped = df.drop(['Bonus', 'Tax'], axis=1)  # axis=1 for columns
# OR
df_dropped = df.drop (columns=['Bonus', 'Tax'])

# In-place
df.drop(['Bonus', 'Tax'], axis=1, inplace=True)

# Drop by index
df_dropped = df.drop([0, 2])  # Drop rows 0 and 2 (axis=0 default)
\`\`\`

### Removing Rows

\`\`\`python
# Drop rows by index
df_dropped = df.drop([0, 2])

# Drop rows by condition
df_filtered = df[df['Age'] >= 30]

# Drop duplicates
df_unique = df.drop_duplicates()

# Drop duplicates based on specific columns
df_unique = df.drop_duplicates (subset=['Department'])

# Drop rows with any NaN
df_clean = df.dropna()

# Drop rows with all NaN
df_clean = df.dropna (how='all')
\`\`\`

## Real-World Example: Stock Data

\`\`\`python
# Create sample stock data
dates = pd.date_range('2024-01-01', periods=10, freq='D')
stock_data = {
    'Date': dates,
    'Open': np.random.uniform(95, 105, 10),
    'High': np.random.uniform(100, 110, 10),
    'Low': np.random.uniform(90, 100, 10),
    'Close': np.random.uniform(95, 105, 10),
    'Volume': np.random.randint(1000000, 5000000, 10)
}
df = pd.DataFrame (stock_data)

# Set date as index
df = df.set_index('Date')
print(df.head())

# Calculate returns
df['Return'] = df['Close'].pct_change()

# Calculate moving average
df['MA_3'] = df['Close'].rolling (window=3).mean()

# Daily range
df['Range'] = df['High'] - df['Low']

# Filter profitable days
profitable_days = df[df['Return'] > 0]

print(f"\\nProfitable days:\\n{profitable_days}")
print(f"\\nAverage return: {df['Return'].mean():.2%}")
print(f"Volatility (std): {df['Return'].std():.2%}")
\`\`\`

## Key Takeaways

1. **Series**: 1D labeled array, similar to dict or NumPy array with index
2. **DataFrame**: 2D labeled data structure, the core pandas object
3. **Creation**: From dicts, lists, NumPy arrays, CSV files
4. **Indexing**: .iloc (position-based), .loc (label-based), boolean
5. **Columns**: Easy to add, remove, transform
6. **Data types**: object, int64, float64, category, datetime64
7. **Memory**: Use categories for repeated strings, appropriate dtypes
8. **Integration**: Works with NumPy, matplotlib, scikit-learn

Pandas DataFrames are the foundation of data analysis in Python. Master these basics, and you'll be ready for more advanced operations!
`,
};
