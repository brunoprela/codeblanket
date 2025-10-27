/**
 * Numerical Feature Engineering Section
 */

export const numericalfeatureengineeringSection = {
  id: 'numerical-feature-engineering',
  title: 'Numerical Feature Engineering',
  content: `# Numerical Feature Engineering

## Introduction

Numerical features are the most common type in machine learning. Proper engineering of numerical features - through scaling, transformation, binning, and creating derived features - can dramatically improve model performance.

**Why Numerical Feature Engineering Matters**:
- **Scale Normalization**: Prevent features with large ranges from dominating
- **Distribution Correction**: Transform skewed features for linear models
- **Relationship Linearization**: Make non-linear relationships linear
- **Outlier Handling**: Reduce sensitivity to extreme values
- **Feature Creation**: Derive new informative features from existing ones

## Scaling and Normalization

### StandardScaler (Z-score Normalization)

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing (as_frame=True)
df = housing.frame

print("=" * 70)
print("SCALING AND NORMALIZATION")
print("=" * 70)

# Original features
features = ['MedInc', 'HouseAge', 'AveRooms']
X = df[features].copy()

print("\\nOriginal Data:")
print(X.describe())

# 1. StandardScaler (mean=0, std=1)
scaler_standard = StandardScaler()
X_standard = pd.DataFrame(
    scaler_standard.fit_transform(X),
    columns=[f'{col}_standard' for col in features]
)

print("\\nStandardScaler (Z-score):")
print(X_standard.describe())

# 2. MinMaxScaler (range [0, 1])
scaler_minmax = MinMaxScaler()
X_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(X),
    columns=[f'{col}_minmax' for col in features]
)

print("\\nMinMaxScaler:")
print(X_minmax.describe())

# 3. RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
X_robust = pd.DataFrame(
    scaler_robust.fit_transform(X),
    columns=[f'{col}_robust' for col in features]
)

print("\\nRobustScaler (uses median and IQR):")
print(X_robust.describe())

# Visualize scaling effects
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for idx, col in enumerate (features):
    # Original
    axes[idx, 0].hist(X[col], bins=50, edgecolor='black')
    axes[idx, 0].set_title (f'Original: {col}')
    axes[idx, 0].set_ylabel('Frequency')
    
    # StandardScaler
    axes[idx, 1].hist(X_standard[f'{col}_standard'], bins=50, edgecolor='black')
    axes[idx, 1].set_title (f'StandardScaler')
    
    # MinMaxScaler
    axes[idx, 2].hist(X_minmax[f'{col}_minmax'], bins=50, edgecolor='black')
    axes[idx, 2].set_title (f'MinMaxScaler')
    
    # RobustScaler
    axes[idx, 3].hist(X_robust[f'{col}_robust'], bins=50, edgecolor='black')
    axes[idx, 3].set_title (f'RobustScaler')

plt.tight_layout()
plt.show()

print("\\n✓ Different scalers for different use cases")
\`\`\`

### When to Use Each Scaler

\`\`\`python
def scaler_recommendations():
    """Guide for choosing appropriate scaler"""
    
    recommendations = {
        'StandardScaler': {
            'when': 'Features approximately normally distributed',
            'use_cases': [
                'Most common choice',
                'Before PCA',
                'With linear models, SVM',
                'Neural networks'
            ],
            'formula': '(x - mean) / std',
            'pros': ['Preserves shape', 'Well-understood'],
            'cons': ['Sensitive to outliers']
        },
        'MinMaxScaler': {
            'when': 'Need specific range (0-1 or custom)',
            'use_cases': [
                'Neural networks with sigmoid/tanh',
                'Image pixel normalization',
                'When zero matters',
                'Bounded activation functions'
            ],
            'formula': '(x - min) / (max - min)',
            'pros': ['Bounded output', 'Preserves zero'],
            'cons': ['Very sensitive to outliers']
        },
        'RobustScaler': {
            'when': 'Data has outliers',
            'use_cases': [
                'Financial data',
                'Sensor data with anomalies',
                'When outliers are valid',
                'After outlier detection'
            ],
            'formula': '(x - median) / IQR',
            'pros': ['Robust to outliers', 'Stable'],
            'cons': ['Not as common', 'Less interpretable']
        },
        'No Scaling': {
            'when': 'Tree-based models',
            'use_cases': [
                'Random Forest',
                'XGBoost',
                'LightGBM',
                'Decision Trees'
            ],
            'formula': 'x (unchanged)',
            'pros': ['Faster', 'Preserves interpretability'],
            'cons': ['Can't use with distance-based models']
        }
    }
    
    print("\\nSCALER SELECTION GUIDE")
    print("=" * 70)
    
    for scaler, info in recommendations.items():
        print(f"\\n{scaler}:")
        print(f"  When: {info['when']}")
        print(f"  Formula: {info['formula']}")
        print(f"  Use Cases:")
        for uc in info['use_cases']:
            print(f"    • {uc}")
        print(f"  Pros: {', '.join (info['pros'])}")
        print(f"  Cons: {', '.join (info['cons'])}")
    
    return recommendations

recommendations = scaler_recommendations()
\`\`\`

## Mathematical Transformations

### Log, Square Root, and Box-Cox

\`\`\`python
from scipy import stats

def apply_transformations (df, feature):
    """Apply various transformations to handle skewness"""
    
    print(f"\\nTRANSFORMATION ANALYSIS: {feature}")
    print("=" * 70)
    
    df_trans = pd.DataFrame()
    df_trans['original'] = df[feature]
    
    # Log transformation (for right-skewed data)
    df_trans['log'] = np.log1p (df[feature])  # log(1+x) handles zeros
    
    # Square root (milder than log)
    df_trans['sqrt'] = np.sqrt (df[feature] - df[feature].min() + 1)
    
    # Cube root (handles negative values)
    df_trans['cbrt'] = np.cbrt (df[feature])
    
    # Box-Cox (automatically finds best power transformation)
    df_trans['boxcox'], lambda_param = stats.boxcox (df[feature] - df[feature].min() + 1)
    
    # Yeo-Johnson (handles negative values)
    df_trans['yeojohnson'], lambda_yj = stats.yeojohnson (df[feature])
    
    # Compare skewness
    print("\\nSkewness Comparison:")
    for col in df_trans.columns:
        skew = df_trans[col].skew()
        print(f"  {col:15s}: {skew:7.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate (df_trans.columns):
        axes[idx].hist (df_trans[col], bins=50, edgecolor='black')
        axes[idx].set_title (f'{col}\\nSkewness: {df_trans[col].skew():.3f}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Recommendation
    best_transform = min (df_trans.columns, 
                        key=lambda x: abs (df_trans[x].skew()))
    print(f"\\n✓ Best transformation: {best_transform} (closest to symmetric)")
    
    return df_trans

# Apply to right-skewed feature
trans_result = apply_transformations (df, 'MedInc')
\`\`\`

## Binning and Discretization

### Converting Continuous to Categorical

\`\`\`python
def create_bins (df, feature, n_bins=5, strategy='quantile'):
    """Create bins from continuous features"""
    
    print(f"\\nBINNING: {feature}")
    print("=" * 70)
    
    # Equal-width binning
    df['equal_width'] = pd.cut (df[feature], bins=n_bins, labels=False)
    
    # Equal-frequency binning (quantiles)
    df['equal_freq'] = pd.qcut (df[feature], q=n_bins, labels=False, duplicates='drop')
    
    # Custom bins based on domain knowledge
    if feature == 'MedInc':
        custom_bins = [0, 2, 4, 6, 10, 15]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['custom'] = pd.cut (df[feature], bins=custom_bins, labels=labels)
    
    # Visualize bin distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Equal width
    df['equal_width'].value_counts().sort_index().plot (kind='bar', ax=axes[0])
    axes[0].set_title('Equal-Width Bins')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Count')
    
    # Equal frequency
    df['equal_freq'].value_counts().sort_index().plot (kind='bar', ax=axes[1])
    axes[1].set_title('Equal-Frequency Bins (Quantiles)')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Count')
    
    # Custom
    if 'custom' in df.columns:
        df['custom'].value_counts().plot (kind='bar', ax=axes[2])
        axes[2].set_title('Custom Bins (Domain-Based)')
        axes[2].set_xlabel('Bin')
        axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    print("\\nBin Statistics:")
    print("\\nEqual-Width:")
    print(df.groupby('equal_width')[feature].agg(['count', 'min', 'max', 'mean']))
    
    print("\\nEqual-Frequency:")
    print(df.groupby('equal_freq')[feature].agg(['count', 'min', 'max', 'mean']))
    
    if 'custom' in df.columns:
        print("\\nCustom Bins:")
        print(df.groupby('custom')[feature].agg(['count', 'min', 'max', 'mean']))
    
    return df

# Create bins
df_binned = create_bins (df.copy(), 'MedInc', n_bins=5)
\`\`\`

## Polynomial Features

### Creating Non-linear Features

\`\`\`python
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(X, degree=2):
    """Create polynomial and interaction features"""
    
    print("\\nPOLYNOMIAL FEATURES")
    print("=" * 70)
    
    print(f"\\nOriginal Features: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)}")
    
    # Create polynomial features
    poly = PolynomialFeatures (degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
    
    print(f"\\nAfter Polynomial Transform (degree={degree}):")
    print(f"  Total features: {X_poly.shape[1]}")
    print(f"  New features created: {X_poly.shape[1] - X.shape[1]}")
    
    print(f"\\nFeature names (first 10):")
    for name in feature_names[:10]:
        print(f"  • {name}")
    
    if len (feature_names) > 10:
        print(f"  ... and {len (feature_names) - 10} more")
    
    # Example: Show correlation with target
    y = df['MedHouseVal']
    correlations = {}
    for col in X_poly_df.columns:
        corr = X_poly_df[col].corr (y)
        correlations[col] = abs (corr)
    
    # Top 10 most correlated features
    top_features = sorted (correlations.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\\nTop 10 Features by Correlation with Target:")
    for feature, corr in top_features:
        print(f"  {feature:30s}: {corr:.4f}")
    
    return X_poly_df

# Create polynomial features
X_sample = df[['MedInc', 'HouseAge']].sample(1000, random_state=42)
X_poly = create_polynomial_features(X_sample, degree=2)
\`\`\`

## Ratios and Derived Features

### Creating Business-Meaningful Features

\`\`\`python
def create_ratio_features (df):
    """Create ratio and derived features"""
    
    print("\\nRATIO AND DERIVED FEATURES")
    print("=" * 70)
    
    df_derived = df.copy()
    
    # Ratios
    df_derived['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    df_derived['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
    df_derived['population_per_household'] = df['Population'] / df['AveOccup']
    
    # Proximity/Distance features (using lat/lon)
    # Distance from a reference point (e.g., San Francisco: 37.7749, -122.4194)
    sf_lat, sf_lon = 37.7749, -122.4194
    df_derived['distance_to_sf'] = np.sqrt(
        (df['Latitude'] - sf_lat)**2 + (df['Longitude'] - sf_lon)**2
    )
    
    # Aggregations
    df_derived['total_rooms'] = df['AveRooms'] * df['AveOccup']
    df_derived['total_bedrooms'] = df['AveBedrms'] * df['AveOccup']
    
    # Flags/Indicators
    df_derived['is_new_house'] = (df['HouseAge'] < 10).astype (int)
    df_derived['is_large_household'] = (df['AveOccup'] > 3).astype (int)
    df_derived['is_high_income'] = (df['MedInc'] > df['MedInc'].median()).astype (int)
    
    # Show correlation with target
    new_features = [col for col in df_derived.columns if col not in df.columns 
                   and col != 'MedHouseVal']
    
    print("\\nNew Features and Correlation with Target:")
    for feat in new_features:
        corr = df_derived[feat].corr (df_derived['MedHouseVal'])
        print(f"  {feat:30s}: {corr:7.4f}")
    
    # Visualize top features
    top_feats = sorted([(f, abs (df_derived[f].corr (df_derived['MedHouseVal']))) 
                       for f in new_features], key=lambda x: x[1], reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (feat, corr) in enumerate (top_feats):
        sample = df_derived.sample(1000, random_state=42)
        axes[idx].scatter (sample[feat], sample['MedHouseVal'], alpha=0.5)
        axes[idx].set_xlabel (feat)
        axes[idx].set_ylabel('MedHouseVal')
        axes[idx].set_title (f'{feat}\\nCorrelation: {corr:.3f}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_derived

# Create derived features
df_with_ratios = create_ratio_features (df)
\`\`\`

## Key Takeaways

1. **Scaling is crucial for distance-based and linear models**2. **StandardScaler most common; RobustScaler for outliers**3. **Tree-based models don't require scaling**4. **Log transformation effective for right-skewed data**5. **Box-Cox automatically finds optimal transformation**6. **Binning converts continuous to categorical (useful for linear models)**7. **Polynomial features capture non-linear relationships**8. **Ratio features often highly interpretable and predictive**9. **Derived features encode domain knowledge**10. **Always maintain train/test consistency in transformations**

## Connection to Machine Learning

- **Scaling** essential for gradient descent convergence (neural networks, linear models)
- **Transformations** make relationships linear for linear models
- **Polynomial features** enable linear models to capture non-linearity
- **Ratios** often more predictive than raw features
- **Binning** can improve model interpretability and handle non-monotonic relationships
- **Proper numerical engineering** can boost performance 20-50% for linear models

Numerical feature engineering is the foundation - master these techniques for consistent improvements.
`,
};
