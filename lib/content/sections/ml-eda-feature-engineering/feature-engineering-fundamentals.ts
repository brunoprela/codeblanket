/**
 * Feature Engineering Fundamentals Section
 */

export const featureengineeringfundamentalsSection = {
  id: 'feature-engineering-fundamentals',
  title: 'Feature Engineering Fundamentals',
  content: `# Feature Engineering Fundamentals

## Introduction

Feature engineering is the art and science of creating new features from existing data to improve model performance. It\'s often said that "applied machine learning" is essentially feature engineering - good features can make a simple model outperform a complex one with poor features.

**Why Feature Engineering Matters**:
- **Boost Model Performance**: Often 10-50% improvement from good features
- **Capture Domain Knowledge**: Encode expert insights into features
- **Enable Simpler Models**: Good features mean simpler, faster, more interpretable models
- **Handle Non-linearity**: Create features that make relationships linear
- **Reduce Dimensionality**: Combine multiple weak features into strong ones

## What is Feature Engineering?

### Definition and Scope

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

print("=" * 70)
print("FEATURE ENGINEERING FUNDAMENTALS")
print("=" * 70)

# Example: Simple dataset where feature engineering matters
np.random.seed(42)
n = 1000

# Generate data with non-linear relationship
X_raw = np.random.uniform(-5, 5, n)
y = 2 * X_raw**2 - 3 * X_raw + 5 + np.random.normal(0, 10, n)

df = pd.DataFrame({
    'x': X_raw,
    'y': y
})

print("\\nOriginal Data:")
print(df.head())

# Model 1: Using raw feature (poor performance)
X_raw_2d = df[['x']]
model_raw = LinearRegression()
scores_raw = cross_val_score (model_raw, X_raw_2d, y, cv=5, 
                             scoring='r2')

print(f"\\nModel 1 (Raw Feature):")
print(f"  R² Score: {scores_raw.mean():.4f} (+/- {scores_raw.std():.4f})")

# Model 2: With engineered polynomial features
df['x_squared'] = df['x'] ** 2
X_engineered = df[['x', 'x_squared']]
model_engineered = LinearRegression()
scores_engineered = cross_val_score (model_engineered, X_engineered, y, 
                                   cv=5, scoring='r2')

print(f"\\nModel 2 (Engineered Features):")
print(f"  Features: x, x²")
print(f"  R² Score: {scores_engineered.mean():.4f} (+/- {scores_engineered.std():.4f})")
print(f"\\n✓ Improvement: {(scores_engineered.mean() - scores_raw.mean()) / abs (scores_raw.mean()) * 100:.1f}%")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Raw data with linear fit
axes[0].scatter (df['x'], df['y'], alpha=0.5, s=10)
model_raw.fit(X_raw_2d, y)
x_line = np.linspace(-5, 5, 100).reshape(-1, 1)
axes[0].plot (x_line, model_raw.predict (x_line), 'r-', linewidth=2, 
            label='Linear (raw)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title (f'Raw Feature (R²={scores_raw.mean():.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: With polynomial feature
axes[1].scatter (df['x'], df['y'], alpha=0.5, s=10)
model_engineered.fit(X_engineered, y)
x_line_df = pd.DataFrame({'x': x_line.ravel()})
x_line_df['x_squared'] = x_line_df['x'] ** 2
axes[1].plot (x_line, model_engineered.predict (x_line_df), 'r-', linewidth=2,
            label='Polynomial (x, x²)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title (f'Engineered Features (R²={scores_engineered.mean():.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nKEY INSIGHT: Feature engineering captured the quadratic relationship")
\`\`\`

## Types of Feature Engineering

### 1. Domain-Driven Features

\`\`\`python
# Example: E-commerce customer data
customer_data = pd.DataFrame({
    'customer_id': range(1, 6),
    'total_purchases': [45, 12, 78, 5, 150],
    'total_revenue': [2250, 360, 4680, 150, 12000],
    'days_since_first_purchase': [730, 90, 1095, 30, 1825],
    'days_since_last_purchase': [5, 25, 10, 28, 3],
    'num_returns': [2, 0, 5, 1, 8],
    'support_tickets': [1, 0, 3, 2, 5]
})

print("\\nDOMAIN-DRIVEN FEATURE ENGINEERING")
print("=" * 70)
print("\\nOriginal Features:")
print(customer_data)

# Engineer domain-specific features
customer_data['avg_purchase_value'] = (
    customer_data['total_revenue'] / customer_data['total_purchases']
)

customer_data['purchase_frequency'] = (
    customer_data['total_purchases'] / 
    (customer_data['days_since_first_purchase'] / 30)  # per month
)

customer_data['return_rate'] = (
    customer_data['num_returns'] / customer_data['total_purchases']
)

customer_data['is_at_risk'] = (
    (customer_data['days_since_last_purchase'] > 90) & 
    (customer_data['support_tickets'] > 2)
).astype (int)

customer_data['customer_lifetime_value_estimate'] = (
    customer_data['avg_purchase_value'] * 
    customer_data['purchase_frequency'] * 
    12  # projected annual value
)

customer_data['loyalty_score'] = (
    (customer_data['days_since_first_purchase'] / 365) * 
    (customer_data['total_purchases'] / 10) * 
    (1 - customer_data['return_rate'])
)

print("\\nEngineered Features:")
for col in ['avg_purchase_value', 'purchase_frequency', 'return_rate', 
           'is_at_risk', 'customer_lifetime_value_estimate', 'loyalty_score']:
    print(f"\\n{col}:")
    print(customer_data[col].values)

print("\\n✓ Domain knowledge converted to predictive features")
\`\`\`

### 2. Mathematical Transformations

\`\`\`python
# Example: Transform skewed features
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing (as_frame=True)
df_housing = housing.frame

print("\\nMATHEMATICAL TRANSFORMATIONS")
print("=" * 70)

# Original feature (right-skewed)
print(f"\\nOriginal 'MedInc' (Median Income):")
print(f"  Skewness: {df_housing['MedInc'].skew():.3f}")
print(f"  Min: {df_housing['MedInc'].min():.3f}")
print(f"  Max: {df_housing['MedInc'].max():.3f}")

# Transformations
df_housing['MedInc_log'] = np.log1p (df_housing['MedInc'])
df_housing['MedInc_sqrt'] = np.sqrt (df_housing['MedInc'])
df_housing['MedInc_squared'] = df_housing['MedInc'] ** 2

print(f"\\nAfter Log Transform:")
print(f"  Skewness: {df_housing['MedInc_log'].skew():.3f}")

print(f"\\nAfter Sqrt Transform:")
print(f"  Skewness: {df_housing['MedInc_sqrt'].skew():.3f}")

# Visualize transformations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist (df_housing['MedInc'], bins=50, edgecolor='black')
axes[0, 0].set_title (f"Original (Skew={df_housing['MedInc'].skew():.2f})")
axes[0, 0].set_xlabel('MedInc')

axes[0, 1].hist (df_housing['MedInc_log'], bins=50, edgecolor='black')
axes[0, 1].set_title (f"Log Transform (Skew={df_housing['MedInc_log'].skew():.2f})")
axes[0, 1].set_xlabel('log(MedInc)')

axes[1, 0].hist (df_housing['MedInc_sqrt'], bins=50, edgecolor='black')
axes[1, 0].set_title (f"Sqrt Transform (Skew={df_housing['MedInc_sqrt'].skew():.2f})")
axes[1, 0].set_xlabel('sqrt(MedInc)')

axes[1, 1].hist (df_housing['MedInc_squared'], bins=50, edgecolor='black')
axes[1, 1].set_title (f"Squared (Skew={df_housing['MedInc_squared'].skew():.2f})")
axes[1, 1].set_xlabel('MedInc²')

plt.tight_layout()
plt.show()

print("\\n✓ Transformations reduce skewness for linear models")
\`\`\`

### 3. Interaction Features

\`\`\`python
# Example: Feature interactions
print("\\nINTERACTION FEATURES")
print("=" * 70)

# Original features
print(f"\\nOriginal Features:")
print(f"  MedInc: Median Income")
print(f"  HouseAge: House Age")

# Create interaction
df_housing['MedInc_x_HouseAge'] = df_housing['MedInc'] * df_housing['HouseAge']

# Test if interaction is useful
from sklearn.model_selection import train_test_split

X_without = df_housing[['MedInc', 'HouseAge']]
X_with = df_housing[['MedInc', 'HouseAge', 'MedInc_x_HouseAge']]
y = df_housing['MedHouseVal']

X_train1, X_test1, y_train, y_test = train_test_split(
    X_without, y, test_size=0.2, random_state=42
)
X_train2, X_test2, _, _ = train_test_split(
    X_with, y, test_size=0.2, random_state=42
)

model1 = LinearRegression().fit(X_train1, y_train)
model2 = LinearRegression().fit(X_train2, y_train)

score1 = model1.score(X_test1, y_test)
score2 = model2.score(X_test2, y_test)

print(f"\\nModel without interaction:")
print(f"  R² Score: {score1:.4f}")

print(f"\\nModel with interaction:")
print(f"  R² Score: {score2:.4f}")
print(f"  Improvement: {(score2-score1)/score1*100:.2f}%")

print("\\n✓ Interaction features capture combined effects")
\`\`\`

## Feature Engineering Process

### Systematic Approach

\`\`\`python
def feature_engineering_workflow():
    """Complete feature engineering workflow"""
    
    workflow = {
        '1. UNDERSTAND THE PROBLEM': [
            'What are you predicting?',
            'What data is available?',
            'What domain knowledge exists?',
            'What are the business constraints?'
        ],
        '2. EXPLORE THE DATA (EDA)': [
            'Distributions of features',
            'Relationships with target',
            'Correlations between features',
            'Missing value patterns',
            'Outliers and anomalies'
        ],
        '3. BRAINSTORM FEATURES': [
            'Domain-driven features',
            'Mathematical transformations',
            'Aggregations and statistics',
            'Interaction terms',
            'Time-based features'
        ],
        '4. CREATE FEATURES': [
            'Implement transformations',
            'Handle edge cases (zeros, nulls)',
            'Maintain train/test consistency',
            'Document feature logic',
            'Version control feature code'
        ],
        '5. SELECT FEATURES': [
            'Correlation with target',
            'Feature importance from models',
            'Remove highly correlated features',
            'Consider computational cost',
            'Use cross-validation'
        ],
        '6. VALIDATE FEATURES': [
            'Cross-validation performance',
            'Check for data leakage',
            'Interpretability assessment',
            'Production feasibility',
            'A/B test if possible'
        ],
        '7. ITERATE': [
            'Analyze model errors',
            'Create features to fix errors',
            'Test new features',
            'Remove unhelpful features',
            'Repeat until diminishing returns'
        ]
    }
    
    print("\\nFEATURE ENGINEERING WORKFLOW")
    print("=" * 70)
    
    for step, items in workflow.items():
        print(f"\\n{step}")
        for item in items:
            print(f"  • {item}")
    
    return workflow

workflow = feature_engineering_workflow()
\`\`\`

## Common Feature Engineering Mistakes

\`\`\`python
def common_mistakes():
    """Common feature engineering mistakes to avoid"""
    
    mistakes = {
        '1. DATA LEAKAGE': {
            'mistake': 'Using future information to create features',
            'example': 'Including "days_until_churn" when predicting churn',
            'fix': 'Only use information available at prediction time'
        },
        '2. TRAIN/TEST INCONSISTENCY': {
            'mistake': 'Fitting transformations on entire dataset',
            'example': 'Using overall mean for imputation instead of train mean',
            'fix': 'Fit transformations on train only, apply to test'
        },
        '3. OVERFITTING TO TRAINING DATA': {
            'mistake': 'Creating too many specific features',
            'example': '1000 features for 500 samples',
            'fix': 'Use cross-validation, regularization, feature selection'
        },
        '4. IGNORING DOMAIN KNOWLEDGE': {
            'mistake': 'Purely algorithmic feature engineering',
            'example': 'Not creating "business_days" for financial data',
            'fix': 'Talk to domain experts, read literature'
        },
        '5. NOT DOCUMENTING FEATURES': {
            'mistake': 'Creating features without documentation',
            'example': 'feature_17_v2 with no explanation',
            'fix': 'Clear naming, document logic, version control'
        },
        '6. FEATURE EXPLOSION': {
            'mistake': 'Creating every possible feature',
            'example': 'All polynomial combinations up to degree 5',
            'fix': 'Be selective, use domain knowledge, measure impact'
        }
    }
    
    print("\\nCOMMON FEATURE ENGINEERING MISTAKES")
    print("=" * 70)
    
    for category, details in mistakes.items():
        print(f"\\n{category}")
        print(f"  ❌ Mistake: {details['mistake']}")
        print(f"  Example: {details['example']}")
        print(f"  ✓ Fix: {details['fix']}")
    
    return mistakes

mistakes = common_mistakes()
\`\`\`

## Key Takeaways

1. **Feature engineering often provides more improvement than algorithm choice**
2. **Good features make complex relationships linear and separable**
3. **Domain knowledge is essential for effective feature engineering**
4. **Always maintain train/test consistency in transformations**
5. **Watch for data leakage - only use information available at prediction time**
6. **Document feature logic clearly for reproducibility**
7. **Use cross-validation to evaluate feature utility**
8. **Polynomial and interaction features capture non-linear relationships**
9. **Mathematical transformations (log, sqrt) handle skewness**
10. **Iterate: create features → test → analyze errors → create more features**

## Connection to Machine Learning

- **Linear models** benefit most from feature engineering (need linearity)
- **Tree-based models** can discover some transformations automatically
- **Neural networks** learn features but benefit from good starting features
- **Feature engineering** reduces need for complex models
- **Good features** improve interpretability of models
- **Domain features** enable transfer of human expertise to models
- **Proper feature engineering** can reduce training time significantly

Feature engineering is where domain expertise meets machine learning - it's the secret weapon of top data scientists.
`,
};
