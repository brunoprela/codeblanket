/**
 * EDA Framework Section
 */

export const edaframeworkSection = {
  id: 'eda-framework',
  title: 'EDA Framework',
  content: `# EDA Framework

## Introduction

Exploratory Data Analysis (EDA) is the critical first step in any machine learning project. Before building models, you must deeply understand your data - its structure, quality, distributions, relationships, and anomalies. EDA is detective work: you're uncovering the story your data tells.

**Why EDA Matters**:
- **Prevents Garbage In, Garbage Out**: Bad data leads to bad models, no matter how sophisticated
- **Informs Feature Engineering**: Discover which features matter and how to transform them
- **Identifies Data Quality Issues**: Missing values, outliers, inconsistencies
- **Guides Model Selection**: Understand problem complexity and appropriate algorithms
- **Builds Domain Intuition**: Learn the business context encoded in data

**EDA in the ML Pipeline**:
\`\`\`
Raw Data → EDA → Data Cleaning → Feature Engineering → Model Building → Evaluation
         ↑                                            ↓
         └────────── Iterate based on findings ──────┘
\`\`\`

## Understanding the Data Problem

### 1. Define the Objective

Start with clarity on what you're trying to achieve:

**Questions to Ask**:
- What business problem are we solving?
- What is the target variable (if supervised learning)?
- What constitutes success? (accuracy, precision, profit, risk reduction)
- What are the constraints? (latency, interpretability, cost)
- Who are the stakeholders and what do they need?

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Example: E-commerce customer churn prediction
# Objective: Predict which customers will stop purchasing (churn)
# Success Metric: Maximize recall (catch as many churners as possible)
# Constraint: Must explain predictions to retention team (interpretability)

# Document your objective clearly
project_objective = {
    'problem': 'Customer churn prediction',
    'target': 'will_churn (binary: 0/1)',
    'primary_metric': 'recall (minimize false negatives)',
    'secondary_metric': 'precision (keep false positives reasonable)',
    'constraints': ['interpretable model', 'predictions within 100ms'],
    'business_impact': 'Proactive retention campaigns, reduce churn by 15%'
}

print("PROJECT OBJECTIVE")
print("=" * 50)
for key, value in project_objective.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
\`\`\`

### 2. Data Collection and Sources

Understand where your data comes from and its reliability:

\`\`\`python
# Example: Multiple data sources for customer churn
data_sources = {
    'transactions': {
        'source': 'PostgreSQL database',
        'refresh': 'Daily',
        'quality': 'High - OLTP system',
        'size': '5M rows, 20 columns'
    },
    'customer_profiles': {
        'source': 'CRM system (Salesforce)',
        'refresh': 'Real-time',
        'quality': 'Medium - user input, incomplete',
        'size': '500K rows, 35 columns'
    },
    'support_tickets': {
        'source': 'Zendesk API',
        'refresh': 'Hourly',
        'quality': 'Variable - free text',
        'size': '200K rows, 15 columns'
    },
    'web_analytics': {
        'source': 'Google Analytics export',
        'refresh': 'Daily batch',
        'quality': 'High but delayed 24h',
        'size': '50M events, 10 columns'
    }
}

print("\\nDATA SOURCES")
print("=" * 50)
for name, info in data_sources.items():
    print(f"\\n{name.upper()}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
\`\`\`

## Initial Data Inspection

### 1. Load and Examine Structure

\`\`\`python
# Load sample dataset
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# For this example, we'll use California Housing dataset
housing = fetch_california_housing (as_frame=True)
df = housing.frame

print("=" * 70)
print("INITIAL DATA INSPECTION")
print("=" * 70)

# 1. Dataset shape
print(f"\\n1. SHAPE: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 2. First few rows
print("\\n2. FIRST 5 ROWS:")
print(df.head())

# 3. Column names and types
print("\\n3. COLUMN INFORMATION:")
print(df.dtypes)

# 4. Memory usage
print(f"\\n4. MEMORY USAGE: {df.memory_usage (deep=True).sum() / 1024**2:.2f} MB")

# 5. Basic statistics
print("\\n5. BASIC STATISTICS:")
print(df.describe())

# Output:
# ======================================================================
# INITIAL DATA INSPECTION
# ======================================================================
# 
# 1. SHAPE: 20,640 rows × 9 columns
# 
# 2. FIRST 5 ROWS:
#    MedInc  HouseAge  AveRooms  ...  Longitude  Latitude  MedHouseVal
# 0  8.3252      41.0  6.984127  ...    -122.23     37.88        4.526
# 1  8.3014      21.0  6.238137  ...    -122.22     37.86        3.585
# ...
\`\`\`

### 2. Identify Column Types

\`\`\`python
def categorize_columns (df):
    """Categorize columns by type for appropriate analysis"""
    
    numerical_cols = df.select_dtypes (include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes (include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes (include=['datetime64']).columns.tolist()
    
    # Further split numerical into continuous and discrete
    discrete_cols = []
    continuous_cols = []
    
    for col in numerical_cols:
        # Heuristic: if unique values < 20 and all integers, consider discrete
        n_unique = df[col].nunique()
        if n_unique < 20 and df[col].dtype in ['int64', 'int32']:
            discrete_cols.append (col)
        else:
            continuous_cols.append (col)
    
    print("COLUMN CATEGORIZATION")
    print("=" * 70)
    print(f"\\nContinuous Numerical ({len (continuous_cols)}): {continuous_cols}")
    print(f"Discrete Numerical ({len (discrete_cols)}): {discrete_cols}")
    print(f"Categorical ({len (categorical_cols)}): {categorical_cols}")
    print(f"Datetime ({len (datetime_cols)}): {datetime_cols}")
    
    return {
        'continuous': continuous_cols,
        'discrete': discrete_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

# Categorize our dataset
column_types = categorize_columns (df)
\`\`\`

## Data Quality Assessment

### 1. Missing Values Analysis

\`\`\`python
def analyze_missing_values (df):
    """Comprehensive missing value analysis"""
    
    missing_counts = df.isnull().sum()
    missing_percent = 100 * missing_counts / len (df)
    
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percent': missing_percent.values
    })
    
    # Sort by missing percentage
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    )
    
    print("\\nMISSING VALUES ANALYSIS")
    print("=" * 70)
    
    if len (missing_df) == 0:
        print("✓ No missing values found!")
    else:
        print(f"\\n{len (missing_df)} columns have missing values:\\n")
        print(missing_df.to_string (index=False))
        
        # Categorize severity
        critical = missing_df[missing_df['Missing_Percent'] > 50]
        high = missing_df[(missing_df['Missing_Percent'] > 20) & 
                         (missing_df['Missing_Percent'] <= 50)]
        medium = missing_df[(missing_df['Missing_Percent'] > 5) & 
                           (missing_df['Missing_Percent'] <= 20)]
        low = missing_df[missing_df['Missing_Percent'] <= 5]
        
        print(f"\\nSEVERITY BREAKDOWN:")
        print(f"  🔴 Critical (>50%): {len (critical)} columns")
        print(f"  🟠 High (20-50%): {len (high)} columns")
        print(f"  🟡 Medium (5-20%): {len (medium)} columns")
        print(f"  🟢 Low (<5%): {len (low)} columns")
    
    return missing_df

# Analyze missing values
missing_analysis = analyze_missing_values (df)
\`\`\`

### 2. Duplicate Detection

\`\`\`python
def analyze_duplicates (df):
    """Detect and analyze duplicate rows"""
    
    print("\\nDUPLICATE ANALYSIS")
    print("=" * 70)
    
    # Check exact duplicates
    n_duplicates = df.duplicated().sum()
    duplicate_percent = 100 * n_duplicates / len (df)
    
    print(f"\\nExact duplicates: {n_duplicates:,} ({duplicate_percent:.2f}%)")
    
    if n_duplicates > 0:
        # Show examples
        duplicated_rows = df[df.duplicated (keep=False)].sort_values(
            by=df.columns.tolist()
        )
        print(f"\\nExample duplicated rows:")
        print(duplicated_rows.head(10))
        
        print(f"\\n⚠️  ACTION NEEDED: Investigate and remove {n_duplicates} duplicates")
    else:
        print("✓ No duplicates found")
    
    return n_duplicates

# Check for duplicates
n_dups = analyze_duplicates (df)
\`\`\`

### 3. Data Type Validation

\`\`\`python
def validate_data_types (df):
    """Validate that columns have appropriate data types"""
    
    print("\\nDATA TYPE VALIDATION")
    print("=" * 70)
    
    issues = []
    
    for col in df.columns:
        dtype = df[col].dtype
        
        # Check if numeric column is stored as object/string
        if dtype == 'object':
            try:
                # Try to convert to numeric
                pd.to_numeric (df[col], errors='raise')
                issues.append (f"⚠️  {col}: stored as 'object' but could be numeric")
            except:
                pass
        
        # Check if dates are stored as strings
        if dtype == 'object' and 'date' in col.lower():
            issues.append (f"⚠️  {col}: might be a date stored as string")
        
        # Check for mixed types
        if dtype == 'object':
            type_counts = df[col].apply (type).value_counts()
            if len (type_counts) > 1:
                issues.append (f"⚠️  {col}: contains mixed types {type_counts.to_dict()}")
    
    if issues:
        print("\\nData type issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\\n✓ All data types appear appropriate")
    
    return issues

# Validate data types
type_issues = validate_data_types (df)
\`\`\`

## EDA Workflow

### Complete EDA Checklist

\`\`\`python
def eda_checklist():
    """Comprehensive EDA checklist"""
    
    checklist = {
        '1. UNDERSTAND THE PROBLEM': [
            '☐ Define clear business objective',
            '☐ Identify target variable',
            '☐ Define success metrics',
            '☐ Understand constraints',
            '☐ Document data sources'
        ],
        '2. INITIAL INSPECTION': [
            '☐ Load data and check shape',
            '☐ Examine first/last rows',
            '☐ Review column names and types',
            '☐ Check memory usage',
            '☐ Generate basic statistics'
        ],
        '3. DATA QUALITY': [
            '☐ Analyze missing values',
            '☐ Check for duplicates',
            '☐ Validate data types',
            '☐ Identify outliers',
            '☐ Check data consistency'
        ],
        '4. UNIVARIATE ANALYSIS': [
            '☐ Distribution of each feature',
            '☐ Central tendency and spread',
            '☐ Identify skewness',
            '☐ Visualize distributions',
            '☐ Check for anomalies'
        ],
        '5. BIVARIATE ANALYSIS': [
            '☐ Feature vs target relationships',
            '☐ Correlation analysis',
            '☐ Scatter plots for continuous',
            '☐ Box plots for categorical',
            '☐ Statistical tests'
        ],
        '6. MULTIVARIATE ANALYSIS': [
            '☐ Correlation matrix',
            '☐ Feature interactions',
            '☐ Dimensionality reduction viz',
            '☐ Identify multicollinearity',
            '☐ Complex patterns'
        ],
        '7. DOMAIN INSIGHTS': [
            '☐ Validate against domain knowledge',
            '☐ Identify data errors',
            '☐ Unexpected patterns',
            '☐ Business rule violations',
            '☐ Temporal patterns'
        ],
        '8. DOCUMENTATION': [
            '☐ Document key findings',
            '☐ Note data quality issues',
            '☐ List transformations needed',
            '☐ Identify feature engineering opportunities',
            '☐ Share insights with stakeholders'
        ]
    }
    
    print("\\n" + "=" * 70)
    print("COMPREHENSIVE EDA CHECKLIST")
    print("=" * 70)
    
    for category, items in checklist.items():
        print(f"\\n{category}")
        for item in items:
            print(f"  {item}")
    
    return checklist

# Display checklist
eda_checklist()
\`\`\`

## Creating an EDA Report

\`\`\`python
def generate_eda_report (df, target_col=None):
    """Generate a comprehensive EDA report"""
    
    report = {
        'dataset_name': 'California Housing',
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'shape': df.shape,
        'memory_mb': df.memory_usage (deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_total': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    
    print("\\n" + "=" * 70)
    print("EXECUTIVE EDA SUMMARY REPORT")
    print("=" * 70)
    print(f"\\nGenerated: {report['timestamp']}")
    print(f"Dataset: {report['dataset_name']}")
    
    print(f"\\n📊 DATASET OVERVIEW")
    print(f"  Rows: {report['shape'][0]:,}")
    print(f"  Columns: {report['shape'][1]}")
    print(f"  Memory: {report['memory_mb']:.2f} MB")
    
    print(f"\\n🔢 DATA TYPES")
    for dtype, count in report['dtypes'].items():
        print(f"  {dtype}: {count} columns")
    
    print(f"\\n⚠️  DATA QUALITY")
    print(f"  Missing values: {report['missing_total']:,}")
    print(f"  Duplicate rows: {report['duplicates']:,}")
    
    if target_col and target_col in df.columns:
        print(f"\\n🎯 TARGET VARIABLE: {target_col}")
        if df[target_col].dtype in ['int64', 'float64']:
            print(f"  Type: Regression (continuous)")
            print(f"  Range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
            print(f"  Mean: {df[target_col].mean():.2f}")
            print(f"  Median: {df[target_col].median():.2f}")
        else:
            print(f"  Type: Classification")
            print(f"  Classes: {df[target_col].nunique()}")
            print(f"  Distribution:")
            for label, count in df[target_col].value_counts().items():
                pct = 100 * count / len (df)
                print(f"    {label}: {count:,} ({pct:.1f}%)")
    
    print(f"\\n✅ NEXT STEPS")
    recommendations = []
    
    if report['missing_total'] > 0:
        recommendations.append("Handle missing values (imputation or removal)")
    if report['duplicates'] > 0:
        recommendations.append("Remove or investigate duplicate rows")
    
    recommendations.extend([
        "Perform univariate analysis on each feature",
        "Analyze feature-target relationships",
        "Check for multicollinearity",
        "Engineer new features based on domain knowledge",
        "Prepare data for modeling"
    ])
    
    for i, rec in enumerate (recommendations, 1):
        print(f"  {i}. {rec}")
    
    return report

# Generate report
report = generate_eda_report (df, target_col='MedHouseVal')
\`\`\`

## Best Practices for EDA

### 1. Start Simple, Then Go Deep

\`\`\`python
# Bad: Jump straight into complex analysis
# Good: Build understanding incrementally

# Level 1: Basic inspection
print(df.info())
print(df.describe())

# Level 2: Visual inspection
df.hist (figsize=(15, 10))
plt.tight_layout()
plt.show()

# Level 3: Detailed analysis
# ... deeper dives into specific features
\`\`\`

### 2. Always Visualize

\`\`\`python
# Numbers alone don't tell the whole story

# Example: Anscombe\'s Quartet - same statistics, different data!
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

# All have same mean, variance, correlation!
print(f"Mean y1: {np.mean (y1):.2f}, Mean y2: {np.mean (y2):.2f}")
print(f"Std y1: {np.std (y1):.2f}, Std y2: {np.std (y2):.2f}")

# But very different patterns when visualized!
# This is why we ALWAYS visualize data
\`\`\`

### 3. Document Everything

\`\`\`python
# Create a findings document as you go

findings = {
    'data_quality': [
        'No missing values in dataset',
        'No duplicates found',
        'All numeric columns have appropriate dtypes'
    ],
    'distributions': [
        'MedInc is right-skewed, may need log transform',
        'HouseAge appears bimodal',
        'Latitude/Longitude define geographic clusters'
    ],
    'relationships': [
        'Strong positive correlation between MedInc and MedHouseVal',
        'Proximity to ocean affects house values',
        'Older houses tend to be cheaper'
    ],
    'anomalies': [
        'Some houses have AveRooms > 20 (possible data errors)',
        'A few blocks with population = 0 (investigate)',
    ],
    'next_steps': [
        'Create location-based features (distance to city center)',
        'Log transform right-skewed features',
        'Engineer price per room feature',
        'Remove impossible values (AveRooms > threshold)'
    ]
}

# Save findings
import json
with open('eda_findings.json', 'w') as f:
    json.dump (findings, f, indent=2)

print("✓ EDA findings documented")
\`\`\`

### 4. Iterate and Refine

\`\`\`python
# EDA is not a one-time process

# Initial EDA → Data Cleaning → More EDA → Feature Engineering → More EDA → Modeling

# After each step, revisit your data:
# - Does the cleaning make sense?
# - Are engineered features behaving as expected?
# - Did I introduce any errors?

# Keep an EDA script that you can run quickly to check data state
\`\`\`

## Key Takeaways

1. **EDA is not optional**: It\'s the foundation of successful ML projects
2. **Start with objectives**: Know what you're trying to achieve
3. **Understand your data sources**: Quality and reliability vary
4. **Check data quality first**: Missing values, duplicates, type issues
5. **Build incrementally**: Start simple, go deeper as needed
6. **Always visualize**: Numbers alone can be misleading
7. **Document findings**: Track insights for future reference
8. **Iterate**: EDA continues throughout the project
9. **Use checklists**: Ensure systematic coverage
10. **Think critically**: Question assumptions and unusual patterns

## Connection to Machine Learning

- **Feature Selection**: EDA reveals which features are informative
- **Data Preprocessing**: Identifies necessary transformations
- **Model Selection**: Problem complexity guides algorithm choice
- **Evaluation Strategy**: Understanding data distribution informs validation approach
- **Error Analysis**: Post-modeling EDA on predictions reveals improvement opportunities
- **Business Value**: Clear understanding enables actionable insights

EDA is where data science becomes an art. The insights you discover here will drive every downstream decision.
`,
};
