/**
 * Section: Data Cleaning
 * Module: Python for Data Science
 *
 * Covers handling missing values, duplicates, data type conversions, outlier treatment, string cleaning, and validation
 */

export const dataCleaning = {
  id: 'data-cleaning',
  title: 'Data Cleaning',
  content: `
# Data Cleaning

## Introduction

Data cleaning is the process of detecting and correcting (or removing) corrupt, inaccurate, or irrelevant data. In real-world projects, data scientists spend 60-80% of their time on data cleaning. Quality data is the foundation of reliable analysis and machine learning models.

**Common Data Quality Issues:**
- Missing values (NaN, None, empty strings)
- Duplicates (exact or fuzzy)
- Inconsistent formatting
- Invalid data types
- Outliers and anomalies
- Encoding issues
- Inconsistent naming conventions

## Handling Missing Values

### Detecting Missing Values

\`\`\`python
import pandas as pd
import numpy as np

# Sample data with missing values
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'Age': [25, np.nan, 30, 28, np.nan],
    'Salary': [50000, 60000, np.nan, 55000, 68000],
    'Department': ['IT', 'HR', 'IT', None, 'IT']
})

# Check for missing values
print(df.isnull())  # or df.isna()
# Returns boolean DataFrame

# Count missing values per column
print(df.isnull().sum())
# Name          1
# Age           2
# Salary        1
# Department    1

# Percentage of missing values
print(df.isnull().sum() / len(df) * 100)

# Total missing values
print(f"Total missing: {df.isnull().sum().sum()}")

# Check if any value is missing
print(f"Has missing: {df.isnull().any().any()}")

# Rows with any missing value
rows_with_missing = df[df.isnull().any(axis=1)]
print(f"Rows with missing: {len(rows_with_missing)}")

# Rows with all values present
complete_rows = df.dropna()
print(f"Complete rows: {len(complete_rows)}")
\`\`\`

### Visualizing Missing Data

\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap of missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Missing data summary
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': df.isnull().sum() / len(df) * 100
})
print(missing_summary)
\`\`\`

### Removing Missing Values

\`\`\`python
# Drop rows with any missing value
df_dropped = df.dropna()
print(f"Original rows: {len(df)}, After dropna: {len(df_dropped)}")

# Drop rows with all values missing
df_dropped = df.dropna(how='all')

# Drop rows with missing values in specific columns
df_dropped = df.dropna(subset=['Age', 'Salary'])

# Drop columns with any missing value
df_dropped = df.dropna(axis=1)

# Drop columns with more than 50% missing
threshold = len(df) * 0.5
df_dropped = df.dropna(thresh=threshold, axis=1)

# In-place modification
df.dropna(inplace=True)
\`\`\`

### Filling Missing Values

\`\`\`python
# Fill with scalar value
df['Age'].fillna(0, inplace=True)

# Fill with mean
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Fill with median (better for skewed data)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill with mode (most frequent)
df['Department'].fillna(df['Department'].mode()[0], inplace=True)

# Forward fill (use previous value)
df['Age'].fillna(method='ffill', inplace=True)
# or: df['Age'].ffill(inplace=True)

# Backward fill (use next value)
df['Age'].fillna(method='bfill', inplace=True)
# or: df['Age'].bfill(inplace=True)

# Fill with different values per column
fill_values = {
    'Age': df['Age'].median(),
    'Salary': df['Salary'].mean(),
    'Department': 'Unknown'
}
df.fillna(fill_values, inplace=True)

# Interpolate (for time series or ordered data)
df['Value'].interpolate(method='linear', inplace=True)
\`\`\`

### Advanced Missing Value Imputation

\`\`\`python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation with sklearn
imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent', 'constant'
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# KNN imputation (uses similar rows)
knn_imputer = KNNImputer(n_neighbors=5)
df[['Age', 'Salary']] = knn_imputer.fit_transform(df[['Age', 'Salary']])

# Group-based imputation
df['Salary'] = df.groupby('Department')['Salary'].transform(
    lambda x: x.fillna(x.mean())
)
\`\`\`

## Handling Duplicates

### Detecting Duplicates

\`\`\`python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'David', 'Bob'],
    'Age': [25, 30, 25, 28, 30],
    'City': ['NYC', 'LA', 'NYC', 'Chicago', 'LA']
})

# Check for duplicate rows
print(df.duplicated())
# [False, False, True, False, True]

# Count duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")

# Show duplicate rows
duplicates = df[df.duplicated()]
print(duplicates)

# Check duplicates based on specific columns
print(df.duplicated(subset=['Name']))
# [False, False, True, False, True]

# Keep first occurrence (default)
print(df.duplicated(keep='first'))

# Keep last occurrence
print(df.duplicated(keep='last'))

# Mark all duplicates (including first occurrence)
print(df.duplicated(keep=False))
\`\`\`

### Removing Duplicates

\`\`\`python
# Remove duplicate rows
df_unique = df.drop_duplicates()

# Remove duplicates based on specific columns
df_unique = df.drop_duplicates(subset=['Name'])

# Keep last occurrence instead of first
df_unique = df.drop_duplicates(keep='last')

# Remove duplicates in place
df.drop_duplicates(inplace=True)

# Remove duplicates but keep original index
df_unique = df.drop_duplicates(ignore_index=False)

# Reset index after removing duplicates
df_unique = df.drop_duplicates().reset_index(drop=True)
\`\`\`

### Fuzzy Duplicate Detection

\`\`\`python
# For similar but not identical strings
from difflib import SequenceMatcher

def similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

# Find similar names
df = pd.DataFrame({
    'Company': ['Apple Inc.', 'Apple Inc', 'Microsoft Corp', 'Microsoft Corporation']
})

# Compare all pairs
for i, name1 in enumerate(df['Company']):
    for j, name2 in enumerate(df['Company'][i+1:], i+1):
        sim = similarity(name1, name2)
        if sim > 0.8:  # 80% similar
            print(f"{name1} <-> {name2}: {sim:.2f}")

# Using fuzzywuzzy library (install: pip install fuzzywuzzy)
# from fuzzywuzzy import fuzz
# fuzz.ratio('Apple Inc.', 'Apple Inc')  # 96
\`\`\`

## Data Type Conversions

### Viewing and Converting Types

\`\`\`python
df = pd.DataFrame({
    'id': ['1', '2', '3', '4'],
    'value': ['100.5', '200.3', '300.7', 'invalid'],
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    'flag': ['True', 'False', 'True', 'False']
})

# Check current types
print(df.dtypes)
# All are 'object' (string)

# Convert to numeric
df['id'] = pd.to_numeric(df['id'])
print(df['id'].dtype)  # int64

# Convert to numeric with error handling
df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Invalid -> NaN
# errors='ignore': leave unchanged
# errors='raise': raise exception

# Convert to integer (requires no NaN)
df['id'] = df['id'].astype(int)

# Nullable integer (allows NaN)
df['value'] = df['value'].astype('Int64')  # Capital I

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
print(df['date'].dtype)  # datetime64[ns]

# Convert to datetime with format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Convert to boolean
df['flag'] = df['flag'].map({'True': True, 'False': False})

# Convert to category
df['category'] = df['category'].astype('category')

# Downcast for memory efficiency
df['id'] = pd.to_numeric(df['id'], downcast='integer')  # Use smallest int type
df['value'] = pd.to_numeric(df['value'], downcast='float')  # Use float32 if possible
\`\`\`

### Handling Date Parsing Errors

\`\`\`python
dates = pd.Series(['2024-01-01', '2024-02-30', 'invalid', '2024-03-15'])

# Coerce errors to NaT (Not a Time)
dates_parsed = pd.to_datetime(dates, errors='coerce')
print(dates_parsed)
# ['2024-01-01', NaT, NaT, '2024-03-15']

# Infer date format
dates_inferred = pd.to_datetime(dates, infer_datetime_format=True, errors='coerce')

# Multiple date formats
dates_mixed = pd.Series(['01/15/2024', '2024-02-20', '03-10-2024'])
dates_parsed = pd.to_datetime(dates_mixed, format='mixed', errors='coerce')
\`\`\`

## Outlier Detection and Treatment

### Statistical Methods

\`\`\`python
# Sample data with outliers
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 10, 95),  # Normal data
    np.array([150, -50, 200, -100, 180])  # Outliers
])
df = pd.DataFrame({'value': data})

# Method 1: Z-score
mean = df['value'].mean()
std = df['value'].std()
df['z_score'] = (df['value'] - mean) / std
df['is_outlier_z'] = np.abs(df['z_score']) > 3

print(f"Z-score outliers: {df['is_outlier_z'].sum()}")

# Method 2: IQR (Interquartile Range)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['is_outlier_iqr'] = (df['value'] < lower_bound) | (df['value'] > upper_bound)
print(f"IQR outliers: {df['is_outlier_iqr'].sum()}")

# Method 3: Modified Z-score (robust to outliers)
median = df['value'].median()
mad = np.median(np.abs(df['value'] - median))
df['modified_z'] = 0.6745 * (df['value'] - median) / mad
df['is_outlier_mod_z'] = np.abs(df['modified_z']) > 3.5

print(f"Modified Z-score outliers: {df['is_outlier_mod_z'].sum()}")

# Visualize outliers
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Box plot
axes[0].boxplot(df['value'])
axes[0].set_title('Box Plot')
axes[0].set_ylabel('Value')

# Histogram
axes[1].hist(df['value'], bins=30, edgecolor='black')
axes[1].set_title('Histogram')
axes[1].set_xlabel('Value')

# Scatter plot with outliers highlighted
axes[2].scatter(df.index, df['value'], c=df['is_outlier_iqr'], cmap='coolwarm')
axes[2].set_title('Scatter Plot (outliers in red)')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Value')

plt.tight_layout()
plt.show()
\`\`\`

### Treating Outliers

\`\`\`python
# Option 1: Remove outliers
df_clean = df[~df['is_outlier_iqr']]

# Option 2: Cap outliers (winsorization)
df['value_capped'] = df['value'].clip(lower=lower_bound, upper=upper_bound)

# Option 3: Replace with boundary values
df['value_winsorized'] = df['value']
df.loc[df['value'] < lower_bound, 'value_winsorized'] = lower_bound
df.loc[df['value'] > upper_bound, 'value_winsorized'] = upper_bound

# Option 4: Replace with median
df['value_median'] = df['value']
df.loc[df['is_outlier_iqr'], 'value_median'] = median

# Option 5: Transform (log for right-skewed data)
df['value_log'] = np.log1p(df['value'] - df['value'].min() + 1)

# Option 6: Use robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['value_scaled'] = scaler.fit_transform(df[['value']])
\`\`\`

## String Cleaning

### Common String Cleaning Operations

\`\`\`python
df = pd.DataFrame({
    'name': ['  Alice Smith  ', 'bob JOHNSON', 'Charlie_Brown', 'DAVID-WILSON'],
    'email': ['Alice@Email.COM', 'bob@email.com  ', 'charlie@EMAIL.COM', '  david@email.com'],
    'phone': ['(555) 123-4567', '555.123.4567', '5551234567', '+1-555-123-4567']
})

# Remove whitespace
df['name'] = df['name'].str.strip()
df['email'] = df['email'].str.strip()

# Convert case
df['name_lower'] = df['name'].str.lower()
df['name_upper'] = df['name'].str.upper()
df['name_title'] = df['name'].str.title()

# Normalize case for emails (always lowercase)
df['email'] = df['email'].str.lower()

# Replace characters
df['name_clean'] = df['name'].str.replace('_', ' ').str.replace('-', ' ')

# Remove special characters
df['name_alpha'] = df['name'].str.replace(r'[^a-zA-Z\\s]', '', regex=True)

# Standardize phone numbers
df['phone_clean'] = (df['phone']
    .str.replace(r'[^0-9]', '', regex=True)  # Keep only digits
    .str[-10:]  # Last 10 digits (remove country code)
)

# Format phone numbers
df['phone_formatted'] = (df['phone_clean']
    .str[:3] + '-' + df['phone_clean'].str[3:6] + '-' + df['phone_clean'].str[6:]
)

print(df[['name', 'name_clean', 'phone', 'phone_formatted']])
\`\`\`

### Advanced String Cleaning

\`\`\`python
# Remove extra whitespace between words
df['text'] = df['text'].str.replace(r'\\s+', ' ', regex=True)

# Remove leading/trailing punctuation
df['text'] = df['text'].str.strip('.,!?;:')

# Standardize separators
df['text'] = df['text'].str.replace(r'[\\-_/]', ' ', regex=True)

# Remove digits
df['text'] = df['text'].str.replace(r'\\d+', '', regex=True)

# Expand contractions
contractions = {
    "don't": "do not",
    "won't": "will not",
    "can't": "cannot"
}
for contraction, expansion in contractions.items():
    df['text'] = df['text'].str.replace(contraction, expansion)

# Lemmatization/Stemming (requires nltk)
# import nltk
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# df['text_lemmatized'] = df['text'].apply(
#     lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
# )
\`\`\`

## Data Validation

### Validation Rules

\`\`\`python
df = pd.DataFrame({
    'age': [25, 150, -5, 30, 200],
    'salary': [50000, 1000000, -10000, 60000, 500000],
    'email': ['valid@email.com', 'invalid-email', 'test@test.com', 'bad@', 'good@domain.org']
})

# Age validation (must be between 0 and 120)
df['age_valid'] = df['age'].between(0, 120)
invalid_ages = df[~df['age_valid']]
print(f"Invalid ages: {len(invalid_ages)}")

# Salary validation (must be positive and reasonable)
df['salary_valid'] = (df['salary'] > 0) & (df['salary'] < 1000000)

# Email validation (basic regex)
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
df['email_valid'] = df['email'].str.match(email_pattern)

# Create validation report
validation_report = {
    'Total Rows': len(df),
    'Invalid Age': (~df['age_valid']).sum(),
    'Invalid Salary': (~df['salary_valid']).sum(),
    'Invalid Email': (~df['email_valid']).sum()
}
print(pd.Series(validation_report))

# Flag all invalid rows
df['is_valid'] = df['age_valid'] & df['salary_valid'] & df['email_valid']
valid_df = df[df['is_valid']]
\`\`\`

### Schema Validation

\`\`\`python
# Define expected schema
expected_schema = {
    'name': 'object',
    'age': 'int64',
    'salary': 'float64',
    'date': 'datetime64[ns]'
}

# Validate schema
def validate_schema(df, expected):
    issues = []
    for col, dtype in expected.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif df[col].dtype != dtype:
            issues.append(f"Wrong type for {col}: expected {dtype}, got {df[col].dtype}")
    
    # Check for extra columns
    extra_cols = set(df.columns) - set(expected.keys())
    if extra_cols:
        issues.append(f"Extra columns: {extra_cols}")
    
    return issues

issues = validate_schema(df, expected_schema)
if issues:
    print("Schema validation failed:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Schema validation passed!")
\`\`\`

## Practical Example: Complete Data Cleaning Pipeline

\`\`\`python
def clean_dataset(df):
    """
    Comprehensive data cleaning pipeline
    """
    print(f"Starting with {len(df)} rows")
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)} rows")
    
    # 2. Handle missing values
    # Drop rows where critical columns are missing
    df = df.dropna(subset=['customer_id', 'transaction_date'])
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].notna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"After handling missing values: {df.isnull().sum().sum()} missing values remain")
    
    # 3. Fix data types
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # 4. Clean strings
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.lower()
    
    # 5. Remove outliers (using IQR method)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    print(f"After removing outliers: {len(df)} rows")
    
    # 6. Reset index
    df = df.reset_index(drop=True)
    
    print(f"Cleaning complete! Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df

# Usage
# df_clean = clean_dataset(df_raw)
\`\`\`

## Key Takeaways

1. **Missing values**: Understand the cause before deciding to drop or fill
2. **Duplicates**: Check both exact and fuzzy matches
3. **Data types**: Correct types enable proper operations and save memory
4. **Outliers**: Detect with multiple methods (Z-score, IQR, domain knowledge)
5. **String cleaning**: Standardize formats for consistency
6. **Validation**: Define and enforce data quality rules
7. **Documentation**: Document cleaning decisions for reproducibility
8. **Iterative process**: Data cleaning often requires multiple passes

Clean data is the foundation of reliable analysis and models. Invest time in thorough cleaning—it pays dividends!
`,
};
