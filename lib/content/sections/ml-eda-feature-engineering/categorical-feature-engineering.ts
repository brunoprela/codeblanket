/**
 * Categorical Feature Engineering Section
 */

export const categoricalfeatureengineeringSection = {
  id: 'categorical-feature-engineering',
  title: 'Categorical Feature Engineering',
  content: `# Categorical Feature Engineering

## Introduction

Categorical features (gender, country, product category) require special encoding techniques since most ML algorithms work with numbers. Proper categorical encoding can dramatically impact model performance.

**Why Categorical Engineering Matters**:
- **Enable ML algorithms**: Convert categories to numerical representations
- **Preserve information**: Maintain categorical relationships
- **Handle high cardinality**: Deal with features with many categories
- **Capture ordinal relationships**: Encode natural ordering
- **Prevent data leakage**: Use appropriate encoding strategies

## Label Encoding

### Simple Ordinal Encoding

\\\`\\\`\\\`python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("LABEL ENCODING")
print("=" * 70)

# Example data
df = pd.DataFrame({
    'size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium'],
    'color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'quality': ['Low', 'Medium', 'High', 'Medium', 'High', 'Low'],
    'price': [10, 20, 30, 15, 35, 25]
})

print("\\nOriginal Data:")
print(df)

# Label Encoding for ordinal features (has natural order)
quality_order = ['Low', 'Medium', 'High']
size_order = ['Small', 'Medium', 'Large']

df['quality_encoded'] = df['quality'].map({
    'Low': 0, 'Medium': 1, 'High': 2
})

df['size_encoded'] = df['size'].map({
    'Small': 0, 'Medium': 1, 'Large': 2  
})

print("\\nAfter Label Encoding:")
print(df[['quality', 'quality_encoded', 'size', 'size_encoded']])

print("\\n✓ Use for ordinal features with natural ordering")
print("⚠️  DON'T use for nominal features (model assumes order)")
\\\`\\\`\\\`

## One-Hot Encoding

### Creating Binary Columns

\\\`\\\`\\\`python
print("\\nONE-HOT ENCODING")
print("=" * 70)

# One-hot encoding for nominal features (no natural order)
df_onehot = pd.get_dummies (df, columns=['color'], prefix='color')

print("\\nAfter One-Hot Encoding 'color':")
print(df_onehot)

print("\\nOriginal shape:", df.shape)
print("After one-hot shape:", df_onehot.shape)

# Drop first column to avoid multicollinearity (dummy variable trap)
df_onehot_dropped = pd.get_dummies (df, columns=['color'], 
                                   prefix='color', drop_first=True)

print("\\nWith drop_first=True (avoid multicollinearity):")
print(df_onehot_dropped[['color_Green', 'color_Red']])

print("\\n✓ Use for nominal features (no order)")
print("✓ Drop first column for linear models")
print("⚠️  Creates many columns with high cardinality")
\\\`\\\`\\\`

## Target Encoding

### Mean/Frequency-Based Encoding

\\\`\\\`\\\`python
from category_encoders import TargetEncoder

print("\\nTARGET ENCODING")
print("=" * 70)

# Generate larger dataset for target encoding
np.random.seed(42)
n = 1000

categories = ['A', 'B', 'C', 'D', 'E']
df_large = pd.DataFrame({
    'category': np.random.choice (categories, n),
    'value': np.random.randn (n)
})

# Category 'A' tends to have higher target values
df_large.loc[df_large['category'] == 'A', 'value'] += 2
df_large.loc[df_large['category'] == 'E', 'value'] -= 1

# Target encoding: replace category with mean target value
target_means = df_large.groupby('category')['value'].mean()

print("\\nTarget Encoding (Mean per Category):")
print(target_means)

df_large['category_encoded'] = df_large['category'].map (target_means)

print("\\nOriginal vs Encoded:")
print(df_large[['category', 'category_encoded', 'value']].head(10))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df_large.boxplot (column='value', by='category', ax=axes[0])
axes[0].set_title('Target Value by Category')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Target Value')

axes[1].scatter (df_large['category_encoded'], df_large['value'], alpha=0.5)
axes[1].set_xlabel('Category Encoded (Mean Target)')
axes[1].set_ylabel('Target Value')
axes[1].set_title('Target Encoding Relationship')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✓ Powerful for high-cardinality features")
print("⚠️  Risk of target leakage - use with cross-validation!")
\\\`\\\`\\\`

## Frequency Encoding

\\\`\\\`\\\`python
print("\\nFREQUENCY ENCODING")
print("=" * 70)

# Count frequency of each category
freq_map = df_large['category'].value_counts().to_dict()

df_large['category_freq'] = df_large['category'].map (freq_map)

print("\\nFrequency Encoding:")
print(df_large.groupby('category')['category_freq'].first().sort_values (ascending=False))

print("\\nSample data:")
print(df_large[['category', 'category_freq']].head(10))

print("\\n✓ Simple and effective")
print("✓ No risk of target leakage")
print("✓ Good for tree-based models")
\\\`\\\`\\\`

## Handling High Cardinality

\\\`\\\`\\\`python
def handle_high_cardinality (df, column, target, top_n=10, method='target'):
    """Handle categorical features with many unique values"""
    
    print(f"\\nHANDLING HIGH CARDINALITY: {column}")
    print("=" * 70)
    
    n_unique = df[column].nunique()
    print(f"\\nUnique values: {n_unique}")
    
    if n_unique <= top_n:
        print(f"✓ Low cardinality - use one-hot encoding")
        return pd.get_dummies (df, columns=[column], prefix=column)
    
    print(f"⚠️  High cardinality - using grouping strategy")
    
    # Strategy 1: Keep top N, group rest as "Other"
    top_categories = df[column].value_counts().head (top_n).index
    df[f'{column}_grouped'] = df[column].apply(
        lambda x: x if x in top_categories else 'Other'
    )
    
    # Strategy 2: Target encoding for high cardinality
    if method == 'target' and target in df.columns:
        target_means = df.groupby (column)[target].mean()
        df[f'{column}_target_encoded'] = df[column].map (target_means)
    
    # Strategy 3: Frequency encoding
    freq_map = df[column].value_counts().to_dict()
    df[f'{column}_freq'] = df[column].map (freq_map)
    
    print(f"\\n✓ Created grouped version (top {top_n} + Other)")
    print(f"✓ Created target encoding")
    print(f"✓ Created frequency encoding")
    
    return df

# Simulate high-cardinality feature
df_high_card = pd.DataFrame({
    'user_id': [f'user_{i}' for i in range(1000)],
    'country': np.random.choice([f'country_{i}' for i in range(50)], 1000),
    'purchase_amount': np.random.exponential(50, 1000)
})

df_encoded = handle_high_cardinality(
    df_high_card.copy(), 
    'country', 
    'purchase_amount', 
    top_n=10
)

print("\\nResult columns:")
print([col for col in df_encoded.columns if 'country' in col])
\\\`\\\`\\\`

## Binary Encoding

\\\`\\\`\\\`python
print("\\nBINARY ENCODING")
print("=" * 70)

# Binary encoding for high cardinality
# Converts category to binary representation
from category_encoders import BinaryEncoder

categories_list = [f'cat_{i}' for i in range(20)]
df_binary = pd.DataFrame({
    'category': np.random.choice (categories_list, 100)
})

encoder = BinaryEncoder (cols=['category'])
df_binary_encoded = encoder.fit_transform (df_binary)

print(f"\\nOriginal unique values: {df_binary['category'].nunique()}")
print(f"Binary encoded columns: {len (df_binary_encoded.columns)}")
print(f"\\nBinary encoding uses log2(n) columns instead of n")
print(f"For 20 categories: log2(20) ≈ 5 columns instead of 20")

print("\\nSample encoded:")
print(df_binary_encoded.head(10))

print("\\n✓ Reduces dimensionality for high cardinality")
print("✓ Maintains some ordinal information")
\\\`\\\`\\\`

## Embeddings for Categorical Features

\\\`\\\`\\\`python
print("\\nEMBEDDINGS (Entity Embeddings)")
print("=" * 70)

print("""
Entity embeddings learn dense vector representations of categories.

CONCEPT:
- Similar to word embeddings (Word2Vec)
- Neural network learns representation during training
- Similar categories get similar vectors

ADVANTAGES:
✓ Captures complex relationships between categories
✓ Reduces dimensionality (e.g., 1000 categories → 10D embedding)
✓ Similar categories grouped in embedding space
✓ Works excellently for high cardinality

IMPLEMENTATION:
- Use in neural networks (Keras Embedding layer)
- Or pre-train and use as features in other models

EXAMPLE USE CASES:
- User IDs (millions of users)
- Product IDs (thousands of products)  
- Location codes
- Any high-cardinality nominal feature

CODE EXAMPLE:
\\\`\\\`\\\`python
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.models import Model

# Input for categorical feature
cat_input = Input (shape=(1,))

# Embedding layer: vocab_size categories → embedding_dim dimensions
embedding = Embedding (input_dim=1000, output_dim=10)(cat_input)
flat = Flatten()(embedding)

# Use in larger model...
\\\`\\\`\\\`
""")

print("\\n✓ Most powerful for high-cardinality features in deep learning")
\\\`\\\`\\\`

## Key Takeaways

1. **Label encoding** for ordinal features (natural order exists)
2. **One-hot encoding** for nominal features (no order, low cardinality)
3. **Target encoding** powerful but risks leakage (use CV!)
4. **Frequency encoding** simple and safe for tree models
5. **For high cardinality**: Group top-N + "Other", target encode, or embeddings
6. **Binary encoding** reduces dimensionality (log2(n) columns)
7. **Embeddings** best for high-cardinality in neural networks
8. **Drop first** in one-hot for linear models (multicollinearity)
9. **Maintain train/test consistency** - fit encoder on train only
10. **Tree models** handle categorical features well natively

## Connection to Machine Learning

- **One-hot encoding** essential for linear models with nominal features
- **Target encoding** often best single encoding for tree models
- **Embeddings** enable neural networks to learn from categorical data
- **High cardinality** features can dominate models if not handled properly
- **Proper encoding** can improve model performance 20-40%
- **Wrong encoding** (e.g., label encoding nominal features) can destroy performance

Categorical feature engineering is critical - many real-world datasets are primarily categorical!
`,
};
