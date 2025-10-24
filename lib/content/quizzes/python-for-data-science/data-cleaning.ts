import { QuizQuestion } from '../../../types';

export const datacleaningQuiz: QuizQuestion[] = [
  {
    id: 'data-cleaning-dq-1',
    question:
      'Compare different strategies for handling missing data: deletion (dropna), simple imputation (fillna with mean/median), and advanced imputation (KNN, MICE). When would you use each approach, and what are the trade-offs?',
    sampleAnswer: `Handling missing data is one of the most critical decisions in data preprocessing, as it can significantly impact analysis results and model performance. Different strategies have different implications:

**1. Complete Case Deletion (dropna)**

\`\`\`python
df_clean = df.dropna()
\`\`\`

**When to use:**
- Missing data is < 5% of dataset
- Data is Missing Completely At Random (MCAR)
- You have abundant data
- Missing values in critical columns

**Advantages:**
- Simple and fast
- No assumptions about missing data
- Preserves relationships in complete cases
- No imputation bias

**Disadvantages:**
- Loss of information (can be substantial)
- Reduced statistical power
- Can introduce bias if data is not MCAR
- May lose important patterns

**Example scenario:**
\`\`\`python
# Customer database where email is required
# Only 2% missing email - safe to remove
df = df.dropna(subset=['email',])
\`\`\`

**2. Simple Imputation (Mean/Median/Mode)**

\`\`\`python
# Mean imputation
df['age',].fillna(df['age',].mean(), inplace=True)

# Median imputation (better for skewed data)
df['salary',].fillna(df['salary',].median(), inplace=True)

# Mode imputation (categorical)
df['department',].fillna(df['department',].mode()[0], inplace=True)
\`\`\`

**When to use:**
- Missing data is 5-20%
- Data is Missing At Random (MAR)
- Need quick solution
- Feature is not critical for model

**Advantages:**
- Preserves sample size
- Simple to implement
- Fast computation
- Works with any data type

**Disadvantages:**
- Reduces variance
- Distorts relationships between variables
- Underestimates standard errors
- Can create artificial peaks in distribution
- Mean sensitive to outliers

**Example scenario:**
\`\`\`python
# Survey data with 10% missing age values
# Use median (robust to outliers)
df['age',].fillna(df['age',].median(), inplace=True)
\`\`\`

**3. Group-Based Imputation**

\`\`\`python
# Impute with group mean
df['salary',] = df.groupby('department')['salary',].transform(
    lambda x: x.fillna(x.mean())
)
\`\`\`

**When to use:**
- Clear grouping structure exists
- Within-group variation is smaller than between-group
- Missing data has systematic patterns

**Advantages:**
- More accurate than global imputation
- Preserves group-level patterns
- Simple to understand and implement

**Disadvantages:**
- Requires meaningful grouping variable
- Still reduces variance within groups
- May fail with small groups

**4. KNN Imputation**

\`\`\`python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df[['age', 'salary', 'experience',]] = imputer.fit_transform(
    df[['age', 'salary', 'experience',]]
)
\`\`\`

**When to use:**
- Missing data is 10-30%
- Clear similarity structure exists
- Multiple correlated features
- Sufficient computational resources

**Advantages:**
- Uses information from similar observations
- Preserves multivariate relationships
- More accurate than simple imputation
- No distribution assumptions

**Disadvantages:**
- Computationally expensive (O(n²))
- Sensitive to choice of k
- Requires feature scaling
- Doesn't work well with high-dimensional data
- Still underestimates variance

**5. Multiple Imputation (MICE)**

\`\`\`python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
\`\`\`

**When to use:**
- Missing data is > 20%
- Statistical inference is important
- Have computational resources
- Need uncertainty estimates

**Advantages:**
- Accounts for uncertainty in missing values
- Preserves relationships between variables
- Proper statistical inference
- Flexible (can use different models)

**Disadvantages:**
- Computationally intensive
- Complex to implement and interpret
- Multiple datasets to manage
- Requires appropriate convergence

**Decision Framework:**

\`\`\`
1. Assess Missing Data Pattern:
   ├─ MCAR (Missing Completely At Random)
   │  └─ < 5% missing → Deletion safe
   │
   ├─ MAR (Missing At Random)
   │  ├─ 5-20% missing → Simple/Group imputation
   │  ├─ 20-40% missing → KNN or MICE
   │  └─ > 40% missing → Consider if feature is useful
   │
   └─ MNAR (Missing Not At Random)
      └─ Model the missingness mechanism

2. Consider Analysis Goal:
   ├─ Exploratory analysis → Simple imputation OK
   ├─ Predictive modeling → KNN or MICE
   └─ Statistical inference → Multiple imputation (MICE)

3. Check Resources:
   ├─ Time-constrained → Simple imputation
   ├─ Computational limits → Mean/median
   └─ Production system → Pre-computed values or simple rules
\`\`\`

**Real-World Example:**

\`\`\`python
# Medical study dataset
df = pd.DataFrame({
    'patient_id': range(1000),
    'age': np.random.randint(20, 80, 1000),
    'blood_pressure': np.random.normal(120, 15, 1000),
    'cholesterol': np.random.normal(200, 30, 1000),
    'outcome': np.random.choice(['healthy', 'at_risk',], 1000)
})

# Introduce missing values (10%)
mask = np.random.random((1000, 4)) < 0.1
mask[:, 0] = False  # Don't remove patient_id
df = df.mask(mask)

# Strategy selection:
# 1. Patient ID: Must have (drop if missing)
df = df.dropna(subset=['patient_id',])

# 2. Age: Use median (robust to outliers)
df['age',].fillna(df['age',].median(), inplace=True)

# 3. Blood pressure & cholesterol: Correlated, use KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[['blood_pressure', 'cholesterol',]] = imputer.fit_transform(
    df[['blood_pressure', 'cholesterol',]]
)

# 4. Outcome: Categorical, use mode or model-based
df['outcome',].fillna(df['outcome',].mode()[0], inplace=True)
\`\`\`

**Best Practices:**

1. **Understand why data is missing** before choosing strategy
2. **Document your approach** for reproducibility
3. **Compare multiple strategies** on a validation set
4. **Create a missing indicator** if missingness is informative
   \`\`\`python
   df['age_was_missing',] = df['age',].isna()
   \`\`\`
5. **Check distributions** before and after imputation
6. **Use domain knowledge** to guide imputation
7. **Consider the downstream task** (prediction vs. inference)

**Common Mistakes:**

❌ **Using mean on skewed data** → Use median  
❌ **Imputing before train/test split** → Causes data leakage  
❌ **Ignoring missing data pattern** → Can introduce bias  
❌ **One-size-fits-all approach** → Different columns need different strategies  
❌ **Not checking imputation quality** → Always validate results  

**Key Takeaway:**
There's no universally best method—the choice depends on the percentage of missing data, missingness mechanism, analysis goals, and computational resources. Start simple, but be prepared to use more sophisticated methods when stakes are high.`,
    keyPoints: [
      'dropna() removes rows/columns with missing data - simple but loses information',
      'Simple imputation (mean/median/mode) easy but ignores relationships',
      'Forward/backward fill good for time series with continuity',
      'KNN imputation uses similar observations - better preserves relationships',
      'MICE (iterative) imputation models each variable - most sophisticated',
    ],
  },
  {
    id: 'data-cleaning-dq-2',
    question:
      'Explain the difference between the Z-score method and IQR method for outlier detection. In what scenarios would each method be more appropriate, and how do they handle different types of distributions?',
    sampleAnswer: `Z-score and IQR are two fundamental outlier detection methods with different assumptions and use cases. Understanding their differences is crucial for proper data cleaning.

**Z-Score Method (Standard Score)**

**Definition:**
\`\`\`python
z_score = (x - mean) / std
is_outlier = abs(z_score) > 3
\`\`\`

An observation is typically considered an outlier if |z-score| > 3 (or 2.5-3 depending on context).

**Assumptions:**
- Data follows normal (Gaussian) distribution
- Mean and standard deviation are representative
- Outliers are rare events (< 0.3% if using z > 3)

**Example:**
\`\`\`python
data = np.random.normal(100, 15, 1000)
# Add outliers
data = np.append(data, [200, 250, -50])

mean = data.mean()
std = data.std()
z_scores = (data - mean) / std
outliers_z = data[np.abs(z_scores) > 3]

print(f"Z-score outliers: {len(outliers_z)}")
print(f"Outlier values: {outliers_z}")
\`\`\`

**IQR Method (Interquartile Range)**

**Definition:**
\`\`\`python
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
is_outlier = (x < lower_bound) | (x > upper_bound)
\`\`\`

**Assumptions:**
- No distribution assumptions required
- Based on quartiles (robust to extreme values)
- Identifies values far from the central 50%

**Example:**
\`\`\`python
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = data[(data < lower) | (data > upper)]

print(f"IQR outliers: {len(outliers_iqr)}")
print(f"Outlier values: {outliers_iqr}")
\`\`\`

**Key Differences:**

| Aspect | Z-Score | IQR |
|--------|---------|-----|
| **Distribution Assumption** | Assumes normal distribution | No assumptions |
| **Robustness** | Sensitive to outliers | Robust to outliers |
| **Statistics Used** | Mean, standard deviation | Quartiles (Q1, Q3) |
| **Outlier Impact** | Outliers affect mean/std | Outliers don't affect Q1/Q3 |
| **Threshold** | Typically |z| > 3 | Q1 - 1.5*IQR, Q3 + 1.5*IQR |
| **Expected Outliers** | ~0.3% (if normal) | ~0.7% (any distribution) |

**Problem with Z-Score in Presence of Outliers:**

\`\`\`python
# Normal data
normal_data = np.random.normal(100, 15, 1000)

# Add extreme outlier
data_with_outlier = np.append(normal_data, [1000])

# Z-score method
mean_affected = data_with_outlier.mean()  # 100.9 (pulled by outlier)
std_affected = data_with_outlier.std()    # 31.8 (inflated)
z_score_outlier = (1000 - mean_affected) / std_affected  # 28.3

# The outlier affects the statistics used to detect it!
# This can mask other outliers

# IQR method (robust)
Q1 = np.percentile(data_with_outlier, 25)  # ~89
Q3 = np.percentile(data_with_outlier, 75)  # ~111
IQR = Q3 - Q1  # ~22
# Quartiles not affected by extreme outlier!
\`\`\`

**When to Use Each Method:**

**Use Z-Score When:**

1. **Data is approximately normal**
\`\`\`python
# Check normality
from scipy import stats
_, p_value = stats.normaltest(data)
if p_value > 0.05:  # Not rejecting normality
    # Use Z-score
    pass
\`\`\`

2. **You want to detect very extreme values**
- Z-score > 3: ~0.3% of data (very rare)
- More conservative than IQR

3. **Data is clean (no pre-existing outliers)**
- Z-score works well when statistics are reliable

4. **Example scenarios:**
- Quality control in manufacturing (known normal process)
- Test scores (typically normal)
- Measurement errors in calibrated instruments

**Use IQR When:**

1. **Distribution is unknown or non-normal**
\`\`\`python
# Skewed distribution
data_skewed = np.random.exponential(50, 1000)
# IQR handles skewness better
\`\`\`

2. **Data contains outliers**
- IQR is robust: existing outliers don't affect Q1, Q3

3. **Want to detect mild outliers**
- IQR detects more outliers (~0.7% vs ~0.3%)

4. **Example scenarios:**
- Income data (highly skewed)
- Real estate prices (long tail)
- Web traffic (power law distribution)
- Financial data (fat tails)

**Comparison on Different Distributions:**

\`\`\`python
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

distributions = [
    ('Normal', np.random.normal(100, 15, 1000)),
    ('Skewed (Exponential)', np.random.exponential(50, 1000)),
    ('Heavy Tails (Cauchy)', stats.cauchy.rvs(100, 25, 1000)),
]

for idx, (name, data) in enumerate(distributions):
    # Add outliers
    data = np.append(data, [data.max() * 1.5, data.min() * 1.5])
    
    # Z-score method
    mean, std = data.mean(), data.std()
    z_scores = np.abs((data - mean) / std)
    z_outliers = data[z_scores > 3]
    
    # IQR method
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_outliers = data[(data < lower) | (data > upper)]
    
    # Plot distribution
    axes[0, idx].hist(data, bins=50, edgecolor='black', alpha=0.7)
    axes[0, idx].set_title(f'{name}\\nZ-score outliers: {len(z_outliers)}')
    axes[0, idx].axvline(mean, color='red', linestyle='--', label='Mean')
    
    # Plot boxplot
    axes[1, idx].boxplot(data)
    axes[1, idx].set_title(f'IQR outliers: {len(iqr_outliers)}')
    
    print(f"\\n{name}:")
    print(f"  Z-score found: {len(z_outliers)} outliers")
    print(f"  IQR found: {len(iqr_outliers)} outliers")

plt.tight_layout()
plt.show()

# Results typically show:
# - Normal: Both methods similar
# - Skewed: IQR performs better
# - Heavy tails: IQR more appropriate
\`\`\`

**Modified Z-Score (Robust Alternative):**

For situations where you want z-score-like interpretation but with robustness:

\`\`\`python
def modified_z_score(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation
    modified_z = 0.6745 * (data - median) / mad
    return modified_z

# Use threshold of 3.5 instead of 3
outliers_mod_z = data[np.abs(modified_z_score(data)) > 3.5]
\`\`\`

**Best Practices:**

**1. Visualize First**
\`\`\`python
# Always plot before deciding
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.hist(data, bins=50)
plt.title('Histogram')

plt.subplot(132)
plt.boxplot(data)
plt.title('Boxplot (IQR)')

plt.subplot(133)
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot (normality check)')
plt.tight_layout()
plt.show()
\`\`\`

**2. Use Both Methods**
\`\`\`python
# Combine both methods
z_outliers = set(np.where(np.abs(z_scores) > 3)[0])
iqr_outliers = set(np.where((data < lower) | (data > upper))[0])

# Outliers detected by both (high confidence)
both = z_outliers & iqr_outliers

# Detected by at least one
either = z_outliers | iqr_outliers
\`\`\`

**3. Domain Knowledge**
\`\`\`python
# Example: Human age
# Z-score might flag 100 years old
# But it's possible, not an error
# Use domain-specific thresholds
valid_ages = data[(data >= 0) & (data <= 120)]
\`\`\`

**4. Iterative Outlier Detection**
\`\`\`python
# IQR can be applied iteratively
def iterative_iqr(data, max_iter=3):
    for i in range(max_iter):
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data >= lower) & (data <= upper)]
    return data
\`\`\`

**Key Takeaways:**

1. **IQR is safer default**: Works without distribution assumptions
2. **Z-score for normal data**: More conservative, fewer false positives
3. **Visualize first**: Check distribution shape before choosing
4. **Consider context**: Statistical outliers ≠ data errors
5. **Use modified Z-score**: If you need z-score but data has outliers
6. **Combine methods**: Agreement between methods gives confidence

**Remember:** Outlier detection is art + science. Statistical methods detect unusual values, but domain expertise determines if they're errors or valuable rare events!`,
    keyPoints: [
      'Identify outliers using IQR, z-score, or domain-specific rules',
      'Outliers can be errors (remove), natural extremes (keep), or interesting (investigate)',
      'Winsorization caps extreme values at percentile threshold',
      'Log transformation reduces impact of outliers while preserving order',
      'Always visualize and understand outliers before deciding treatment',
    ],
  },
  {
    id: 'data-cleaning-dq-3',
    question:
      'Discuss the concept of "data leakage" in the context of data cleaning. How can improper handling of missing values, outliers, or normalization lead to leakage, and what practices prevent it?',
    sampleAnswer: `Data leakage is one of the most insidious problems in machine learning—it leads to overly optimistic performance estimates that don't generalize to production. Many leakage issues occur during data cleaning when information from the test set "leaks" into the training process.

**What is Data Leakage?**

Data leakage occurs when information from outside the training dataset is used to create the model. This results in models that perform well in validation but fail in production.

**Types of Leakage in Data Cleaning:**

**1. Leakage from Imputation**

**Wrong (Leaks):**
\`\`\`python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create data with missing values
df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})
# Introduce 20% missing
mask = np.random.random((1000, 2)) < 0.2
df[['feature1', 'feature2',]] = df[['feature1', 'feature2',]].mask(mask)

# WRONG: Impute before split
df['feature1',].fillna(df['feature1',].mean(), inplace=True)  # Uses ALL data including test!
df['feature2',].fillna(df['feature2',].mean(), inplace=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df[['feature1', 'feature2',]], df['target',], test_size=0.2
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test set already "knows" about training set statistics!
score = model.score(X_test, y_test)  # Overly optimistic
\`\`\`

**Correct (No Leakage):**
\`\`\`python
# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    df[['feature1', 'feature2',]], df['target',], test_size=0.2, random_state=42
)

# Compute statistics on training set only
train_mean_f1 = X_train['feature1',].mean()
train_mean_f2 = X_train['feature2',].mean()

# Apply to both sets
X_train['feature1',].fillna(train_mean_f1, inplace=True)
X_train['feature2',].fillna(train_mean_f2, inplace=True)
X_test['feature1',].fillna(train_mean_f1, inplace=True)  # Use training statistics
X_test['feature2',].fillna(train_mean_f2, inplace=True)

# Now test score is realistic
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
\`\`\`

**Why This Matters:**
\`\`\`python
# Demonstration of impact
np.random.seed(42)
n = 1000

# Create data where missingness is informative
df = pd.DataFrame({
    'feature': np.random.randn(n),
    'target': np.random.randint(0, 2, n)
})

# Make missingness correlated with target
mask = (df['target',] == 1) & (np.random.random(n) < 0.5)
df.loc[mask, 'feature',] = np.nan

print(f"Missing in class 0: {df[df['target',]==0]['feature',].isna().sum()}")
print(f"Missing in class 1: {df[df['target',]==1]['feature',].isna().sum()}")
# Class 1 has more missing values!

# Wrong way (leaks)
df_wrong = df.copy()
df_wrong['feature',].fillna(df_wrong['feature',].mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    df_wrong[['feature',]], df_wrong['target',], test_size=0.2
)
model = LogisticRegression()
model.fit(X_train, y_train)
score_wrong = model.score(X_test, y_test)

# Correct way (no leakage)
X_train, X_test, y_train, y_test = train_test_split(
    df[['feature',]], df['target',], test_size=0.2, random_state=42
)
train_mean = X_train['feature',].mean()
X_train_filled = X_train.fillna(train_mean)
X_test_filled = X_test.fillna(train_mean)
model.fit(X_train_filled, y_train)
score_correct = model.score(X_test_filled, y_test)

print(f"\\nScore with leakage: {score_wrong:.3f}")
print(f"Score without leakage: {score_correct:.3f}")
# Leakage can inflate score by 5-20%!
\`\`\`

**2. Leakage from Normalization/Scaling**

**Wrong:**
\`\`\`python
from sklearn.preprocessing import StandardScaler

# WRONG: Fit scaler on all data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)  # Computes mean/std on ALL data

X_train, X_test = train_test_split(df_scaled, test_size=0.2)
# Test set influenced training set's mean/std
\`\`\`

**Correct:**
\`\`\`python
X_train, X_test = train_test_split(df, test_size=0.2)

# Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation to test (using training statistics)
X_test_scaled = scaler.transform(X_test)  # Don't use fit_transform!
\`\`\`

**3. Leakage from Outlier Removal**

**Wrong:**
\`\`\`python
# WRONG: Remove outliers from combined data
Q1 = df['feature',].quantile(0.25)
Q3 = df['feature',].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['feature',] >= Q1 - 1.5*IQR) & (df['feature',] <= Q3 + 1.5*IQR)]

# Split after outlier removal
X_train, X_test = train_test_split(df_clean, test_size=0.2)
# Test set was used to compute outlier bounds!
\`\`\`

**Correct:**
\`\`\`python
# Split first
X_train, X_test = train_test_split(df, test_size=0.2)

# Compute outlier bounds on training set only
Q1 = X_train['feature',].quantile(0.25)
Q3 = X_train['feature',].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers from training set
X_train_clean = X_train[
    (X_train['feature',] >= lower_bound) & 
    (X_train['feature',] <= upper_bound)
]

# For test set: either keep all data or cap outliers
X_test_capped = X_test.copy()
X_test_capped['feature',] = X_test_capped['feature',].clip(lower_bound, upper_bound)
\`\`\`

**4. Leakage from Feature Engineering**

**Wrong:**
\`\`\`python
# Create target encoding before split
df['category_mean_target',] = df.groupby('category')['target',].transform('mean')
# This uses future information!

X_train, X_test = train_test_split(df, test_size=0.2)
\`\`\`

**Correct:**
\`\`\`python
X_train, X_test, y_train, y_test = train_test_split(
    df[['category',]], df['target',], test_size=0.2
)

# Compute target encoding on training set
category_means = X_train.join(y_train).groupby('category')['target',].mean()

# Apply to both sets
X_train['category_mean_target',] = X_train['category',].map(category_means)
X_test['category_mean_target',] = X_test['category',].map(category_means)

# Handle unseen categories in test set
X_test['category_mean_target',].fillna(y_train.mean(), inplace=True)
\`\`\`

**5. Temporal Leakage (Time Series)**

**Wrong:**
\`\`\`python
# Random split of time series data
df = df.sort_values('date')
X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True)
# Training set contains future data!
\`\`\`

**Correct:**
\`\`\`python
# Time-based split
df = df.sort_values('date')
split_date = df['date',].quantile(0.8)
X_train = df[df['date',] < split_date]
X_test = df[df['date',] >= split_date]
# Respects temporal order
\`\`\`

**Complete Correct Pipeline:**

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Create pipeline (prevents leakage automatically)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['target',], test_size=0.2, random_state=42
)

# Fit pipeline on training data only
pipeline.fit(X_train, y_train)

# Transform and predict on test data
# Pipeline uses training statistics for imputation and scaling
score = pipeline.score(X_test, y_test)

# This is the correct way—no leakage!
\`\`\`

**Cross-Validation with Proper Handling:**

\`\`\`python
from sklearn.model_selection import cross_val_score

# Pipeline ensures each fold is processed independently
scores = cross_val_score(pipeline, X, y, cv=5)
# Each fold:
# 1. Fits imputer on training folds
# 2. Transforms training and validation folds
# 3. Fits scaler on training folds
# 4. Transforms training and validation folds
# 5. Trains model on training folds
# 6. Evaluates on validation fold

print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
\`\`\`

**Checklist to Prevent Leakage:**

✅ **Split data FIRST**, then clean  
✅ **Fit on training set** only (imputers, scalers, encoders)  
✅ **Transform both sets** using training statistics  
✅ **Use pipelines** to automate correct behavior  
✅ **Respect temporal order** in time series  
✅ **Be careful with target encoding** (use cross-validation or leave-one-out)  
✅ **Document preprocessing steps** for production  
✅ **Test on truly unseen data** before deployment  

**Common Mistakes:**

❌ Imputing before split  
❌ Scaling all data together  
❌ Removing duplicates after split  
❌ Feature selection on entire dataset  
❌ Outlier removal on combined data  
❌ Using test set for any tuning decisions  

**Real-World Impact:**

\`\`\`python
# Example: Kaggle competition

# Leaky approach
# Train AUC: 0.95, Public LB: 0.92, Private LB: 0.72 😱
# Model fails when new data arrives

# Correct approach
# Train AUC: 0.88, Public LB: 0.86, Private LB: 0.85 ✅
# Model generalizes well to production
\`\`\`

**Best Practices for Production:**

1. **Use sklearn Pipeline**
\`\`\`python
# Encapsulates all preprocessing
# Guarantees correct train/test handling
\`\`\`

2. **Save preprocessing artifacts**
\`\`\`python
import joblib

# Save fitted imputer, scaler, etc.
joblib.dump(pipeline, 'model_pipeline.pkl')

# Load in production
pipeline = joblib.load('model_pipeline.pkl')
predictions = pipeline.predict(new_data)
\`\`\`

3. **Validate on time-based holdout**
\`\`\`python
# Use most recent data as test set
# Simulates production scenario
\`\`\`

4. **Monitor for distribution shift**
\`\`\`python
# Track if new data differs from training
# May need to retrain
\`\`\`

**Key Takeaway:**

Data leakage during cleaning is subtle but has major impact. Always follow the golden rule: **Train on training data, test on test data, and never let information from test set influence training process.** Use sklearn Pipeline to automate correct behavior and prevent mistakes. The difference between a great model in development and a failed model in production is often just proper handling of the train/test split during data cleaning!`,
    keyPoints: [
      'Data validation catches issues early in pipeline before propagation',
      'Check dtypes, ranges, null counts, uniqueness, and referential integrity',
      'Use assert statements to enforce invariants and fail fast',
      'Great Expectations framework provides comprehensive validation library',
      'Document data quality rules and run them automatically on new data',
    ],
  },
];
