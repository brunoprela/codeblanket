/**
 * Advanced Feature Engineering Section
 */

export const advancedfeatureengineeringSection = {
  id: 'advanced-feature-engineering',
  title: 'Advanced Feature Engineering',
  content: `# Advanced Feature Engineering

## Introduction

Advanced feature engineering combines domain knowledge, mathematical creativity, and automated techniques to extract maximum value from data. These sophisticated approaches can provide the competitive edge in machine learning competitions and real-world applications.

**Why Advanced Techniques Matter**:
- **Automated Feature Generation**: Scale feature creation
- **Non-linear Transformations**: Capture complex relationships
- **Feature Selection**: Identify most impactful features
- **Domain-Specific Engineering**: Industry-specific patterns
- **Ensemble Features**: Combine multiple feature types

## Automated Feature Engineering

### Using Featuretools

\\\`\\\`\\\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 70)

# Sample data - e-commerce transactions
np.random.seed(42)

customers = pd.DataFrame({
    'customer_id': range(1, 101),
    'signup_date': pd.date_range('2020-01-01', periods=100, freq='3D'),
    'age': np.random.randint(18, 70, 100),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 100)
})

transactions = pd.DataFrame({
    'transaction_id': range(1, 501),
    'customer_id': np.random.choice(range(1, 101), 500),
    'transaction_date': pd.date_range('2022-01-01', periods=500, freq='6H'),
    'amount': np.random.exponential(50, 500),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 500)
})

print("\\nSample Data:")
print("\\nCustomers:")
print(customers.head())
print("\\nTransactions:")
print(transactions.head())

# Manual feature engineering from relationships
print("\\n" + "=" * 70)
print("RELATIONSHIP-BASED FEATURES")
print("=" * 70)

# Aggregate features from transactions for each customer
customer_features = transactions.groupby('customer_id').agg({
    'transaction_id': 'count',  # Number of transactions
    'amount': ['sum', 'mean', 'std', 'min', 'max'],  # Transaction statistics
    'transaction_date': ['min', 'max']  # First and last transaction
}).reset_index()

# Flatten column names
customer_features.columns = ['_'.join(col).strip('_') for col in customer_features.columns.values]
customer_features.columns = ['customer_id', 'num_transactions', 'total_spent', 
                            'avg_transaction', 'std_transaction', 'min_transaction',
                            'max_transaction', 'first_transaction', 'last_transaction']

# Merge with customer data
df_enriched = customers.merge(customer_features, on='customer_id', how='left')

# Create derived features
df_enriched['customer_lifetime_days'] = (
    df_enriched['last_transaction'] - df_enriched['first_transaction']
).dt.days

df_enriched['transaction_frequency'] = (
    df_enriched['num_transactions'] / (df_enriched['customer_lifetime_days'] + 1)
)

df_enriched['customer_value_score'] = (
    df_enriched['total_spent'] * df_enriched['transaction_frequency']
)

print("\\nEnriched Customer Features:")
print(df_enriched[['customer_id', 'num_transactions', 'total_spent', 
                   'customer_value_score']].head())

print("\\n✓ Created features from entity relationships")
\\\`\\\`\\\`

## Feature Interactions and Combinations

### Creating Complex Interactions

\\\`\\\`\\\`python
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

def create_feature_interactions(df, features, max_interaction=2):
    """Create interaction features between specified features"""
    
    print("\\nFEATURE INTERACTIONS")
    print("=" * 70)
    
    df_interactions = df[features].copy()
    original_features = features.copy()
    
    # Pairwise interactions (multiplication)
    if max_interaction >= 2:
        for feat1, feat2 in combinations(features, 2):
            interaction_name = f'{feat1}_x_{feat2}'
            df_interactions[interaction_name] = df[feat1] * df[feat2]
            print(f"Created: {interaction_name}")
    
    # Ratio features
    for feat1, feat2 in combinations(features, 2):
        if (df[feat2] != 0).all():
            ratio_name = f'{feat1}_div_{feat2}'
            df_interactions[ratio_name] = df[feat1] / (df[feat2] + 1e-8)
            print(f"Created: {ratio_name}")
    
    # Difference features
    for feat1, feat2 in combinations(features, 2):
        diff_name = f'{feat1}_minus_{feat2}'
        df_interactions[diff_name] = df[feat1] - df[feat2]
        print(f"Created: {diff_name}")
    
    print(f"\\nOriginal features: {len(original_features)}")
    print(f"Total features after interactions: {df_interactions.shape[1]}")
    
    return df_interactions

# Create interactions for numerical features
numerical_features = ['age', 'num_transactions', 'total_spent']
df_with_interactions = create_feature_interactions(
    df_enriched.fillna(0), 
    numerical_features
)

print("\\n✓ Interaction features can capture non-additive effects")
\\\`\\\`\\\`

## Feature Selection

### Multiple Selection Methods

\\\`\\\`\\\`python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def comprehensive_feature_selection(X, y, k=10):
    """Apply multiple feature selection methods"""
    
    print("\\nCOMPREHENSIVE FEATURE SELECTION")
    print("=" * 70)
    
    feature_names = X.columns
    results = pd.DataFrame({'feature': feature_names})
    
    # 1. Univariate Selection (ANOVA F-statistic)
    selector_f = SelectKBest(score_func=f_classif, k=k)
    selector_f.fit(X, y)
    results['f_score'] = selector_f.scores_
    results['f_selected'] = results['f_score'].rank(ascending=False) <= k
    
    # 2. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    results['mi_score'] = mi_scores
    results['mi_selected'] = results['mi_score'].rank(ascending=False) <= k
    
    # 3. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    results['rf_importance'] = rf.feature_importances_
    results['rf_selected'] = results['rf_importance'].rank(ascending=False) <= k
    
    # 4. L1-based Selection (Lasso)
    selector_l1 = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        max_features=k
    )
    selector_l1.fit(X, y)
    results['l1_selected'] = selector_l1.get_support()
    
    # 5. Recursive Feature Elimination
    rfe = RFE(LogisticRegression(random_state=42), n_features_to_select=k)
    rfe.fit(X, y)
    results['rfe_selected'] = rfe.support_
    
    # Consensus: features selected by multiple methods
    selection_columns = ['f_selected', 'mi_selected', 'rf_selected', 
                         'l1_selected', 'rfe_selected']
    results['selection_count'] = results[selection_columns].sum(axis=1)
    results['consensus'] = results['selection_count'] >= 3
    
    # Sort by consensus
    results = results.sort_values('selection_count', ascending=False)
    
    print(f"\\nTop {k} Features by Consensus:")
    print(results[['feature', 'selection_count', 'consensus']].head(k))
    
    # Visualize feature selection
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # F-scores
    top_f = results.nlargest(k, 'f_score')
    axes[0, 0].barh(range(len(top_f)), top_f['f_score'])
    axes[0, 0].set_yticks(range(len(top_f)))
    axes[0, 0].set_yticklabels(top_f['feature'])
    axes[0, 0].set_xlabel('F-Score')
    axes[0, 0].set_title('Top Features by ANOVA F-test')
    
    # Mutual Information
    top_mi = results.nlargest(k, 'mi_score')
    axes[0, 1].barh(range(len(top_mi)), top_mi['mi_score'])
    axes[0, 1].set_yticks(range(len(top_mi)))
    axes[0, 1].set_yticklabels(top_mi['feature'])
    axes[0, 1].set_xlabel('MI Score')
    axes[0, 1].set_title('Top Features by Mutual Information')
    
    # Random Forest Importance
    top_rf = results.nlargest(k, 'rf_importance')
    axes[1, 0].barh(range(len(top_rf)), top_rf['rf_importance'])
    axes[1, 0].set_yticks(range(len(top_rf)))
    axes[1, 0].set_yticklabels(top_rf['feature'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top Features by Random Forest')
    
    # Consensus
    top_consensus = results.nlargest(k, 'selection_count')
    axes[1, 1].barh(range(len(top_consensus)), top_consensus['selection_count'])
    axes[1, 1].set_yticks(range(len(top_consensus)))
    axes[1, 1].set_yticklabels(top_consensus['feature'])
    axes[1, 1].set_xlabel('Selection Count')
    axes[1, 1].set_title('Consensus (selected by N methods)')
    axes[1, 1].axvline(3, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Generate synthetic target for demonstration
df_enriched['high_value'] = (df_enriched['customer_value_score'] > 
                             df_enriched['customer_value_score'].median()).astype(int)

# Prepare features
X = df_enriched[numerical_features].fillna(0)
y = df_enriched['high_value']

# Perform feature selection
selection_results = comprehensive_feature_selection(X, y, k=min(10, X.shape[1]))
\\\`\\\`\\\`

## Domain-Specific Feature Engineering

### Trading/Finance Features

\\\`\\\`\\\`python
def create_financial_features(df, price_col='price', volume_col='volume'):
    """Create domain-specific financial trading features"""
    
    print("\\nFINANCIAL/TRADING FEATURES")
    print("=" * 70)
    
    # Technical indicators
    
    # 1. Moving Averages
    df['sma_20'] = df[price_col].rolling(window=20, min_periods=1).mean()
    df['sma_50'] = df[price_col].rolling(window=50, min_periods=1).mean()
    df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
    
    # 2. MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # 3. RSI (Relative Strength Index)
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Bands
    df['bb_middle'] = df[price_col].rolling(window=20, min_periods=1).mean()
    bb_std = df[price_col].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # 5. Volatility
    df['volatility'] = df[price_col].pct_change().rolling(window=20, min_periods=1).std()
    
    # 6. Price momentum
    df['momentum'] = df[price_col] - df[price_col].shift(10)
    df['rate_of_change'] = df[price_col].pct_change(periods=10)
    
    # 7. Volume features (if available)
    if volume_col in df.columns:
        df['volume_sma'] = df[volume_col].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df[volume_col] / (df['volume_sma'] + 1e-8)
    
    print("\\nCreated Financial Features:")
    print("  • Moving Averages (SMA, EMA)")
    print("  • MACD Indicator")
    print("  • RSI (Relative Strength Index)")
    print("  • Bollinger Bands")
    print("  • Volatility Measures")
    print("  • Momentum Indicators")
    
    return df

# Example with synthetic stock data
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
df_stock = pd.DataFrame({
    'date': dates,
    'price': 100 + np.cumsum(np.random.randn(len(dates))) + 
             np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10,
    'volume': np.random.exponential(1000000, len(dates))
})

df_stock = create_financial_features(df_stock, 'price', 'volume')

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Price with Moving Averages
axes[0].plot(df_stock['date'], df_stock['price'], label='Price', alpha=0.7)
axes[0].plot(df_stock['date'], df_stock['sma_20'], label='SMA 20', linewidth=2)
axes[0].plot(df_stock['date'], df_stock['sma_50'], label='SMA 50', linewidth=2)
axes[0].set_ylabel('Price')
axes[0].set_title('Price with Moving Averages')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MACD
axes[1].plot(df_stock['date'], df_stock['macd'], label='MACD')
axes[1].plot(df_stock['date'], df_stock['macd_signal'], label='Signal')
axes[1].bar(df_stock['date'], df_stock['macd_histogram'], label='Histogram', alpha=0.3)
axes[1].set_ylabel('MACD')
axes[1].set_title('MACD Indicator')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# RSI
axes[2].plot(df_stock['date'], df_stock['rsi'])
axes[2].axhline(70, color='r', linestyle='--', alpha=0.5, label='Overbought')
axes[2].axhline(30, color='g', linestyle='--', alpha=0.5, label='Oversold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('RSI')
axes[2].set_title('Relative Strength Index (RSI)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✓ Domain-specific features encode expert knowledge")
\\\`\\\`\\\`

## Feature Engineering Pipeline

### Creating Reproducible Pipelines

\\\`\\\`\\\`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def create_feature_engineering_pipeline():
    """Create sklearn pipeline for feature engineering"""
    
    print("\\nFEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    # Define transformations for different feature types
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Example pipeline structure
    pipeline_structure = {
        'Step 1: Handle Missing': [
            'Imputation strategy (mean, median, mode)',
            'Forward/backward fill for time series',
            'Flag missing values as separate feature'
        ],
        'Step 2: Encode Categorical': [
            'One-hot encoding for nominal',
            'Label encoding for ordinal',
            'Target encoding for high cardinality'
        ],
        'Step 3: Create Features': [
            'Polynomial features',
            'Interaction terms',
            'Domain-specific features',
            'Time-based features'
        ],
        'Step 4: Scale Numerical': [
            'StandardScaler for normal distributions',
            'RobustScaler for outliers',
            'MinMaxScaler for bounded ranges'
        ],
        'Step 5: Select Features': [
            'Remove low variance',
            'Remove highly correlated',
            'Select top K by importance'
        ]
    }
    
    print("\\nRecommended Pipeline Structure:\\n")
    for step, actions in pipeline_structure.items():
        print(f"{step}:")
        for action in actions:
            print(f"  • {action}")
        print()
    
    print("✓ Use sklearn Pipeline for reproducibility")
    print("✓ Fit on training data, transform train and test")
    print("✓ Save pipeline for production deployment")
    
    return pipeline_structure

pipeline_info = create_feature_engineering_pipeline()
\\\`\\\`\\\`

## Key Takeaways

1. **Automated feature engineering** scales creation to hundreds of features
2. **Feature interactions** capture non-additive relationships
3. **Multiple feature selection methods** provide robust identification
4. **Domain-specific features** encode expert knowledge
5. **Financial indicators** (MACD, RSI, Bollinger Bands) for trading
6. **Use pipelines** for reproducibility and production deployment
7. **Feature selection crucial** to avoid curse of dimensionality
8. **Test multiple approaches** - no single method always best
9. **Balance automation with domain knowledge** for best results
10. **Document feature engineering logic** for maintenance

## Connection to Machine Learning

- **Automated feature engineering** enables rapid experimentation
- **Feature selection** prevents overfitting and improves generalization
- **Domain features** often outperform pure automated approaches
- **Pipelines** ensure train/test consistency and production readiness
- **Feature engineering** can provide 2-10x performance improvement
- **Interpretability** suffers with too many automated features
- **Balance**: Start with domain features, augment with automated features

Advanced feature engineering separates good data scientists from great ones - it's where creativity meets rigor.
`,
};
