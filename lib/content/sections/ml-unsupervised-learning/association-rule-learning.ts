/**
 * Section: Association Rule Learning
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive coverage of market basket analysis, Apriori, and association rules
 */

export const associationRuleLearning = {
  id: 'association-rule-learning',
  title: 'Association Rule Learning',
  content: `
# Association Rule Learning

## Introduction

Association Rule Learning is an unsupervised technique for discovering interesting relationships, patterns, and associations among items in large datasets. It's most famous for **Market Basket Analysis** but has applications across many domains.

**Classic Example**: "Customers who buy diapers also tend to buy beer"

**Applications**:
- **Retail**: Product recommendations, store layout optimization
- **E-commerce**: "Customers who bought this also bought..."
- **Healthcare**: Identifying co-occurring symptoms or conditions
- **Web Mining**: Page navigation patterns
- **Bioinformatics**: Gene co-expression patterns

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example: Grocery store transactions
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter', 'eggs'],
    ['bread', 'butter', 'eggs'],
    ['milk', 'bread', 'butter', 'eggs'],
    ['bread', 'eggs'],
    ['milk', 'bread', 'eggs'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter']
]

print(f"Total transactions: {len(transactions)}")
print("\\nSample transactions:")
for i, t in enumerate(transactions[:5], 1):
    print(f"{i}. {t}")

# Question: What items are frequently bought together?
\`\`\`

## Key Concepts

### Itemset

A collection of one or more items
- **1-itemset**: {milk}
- **2-itemset**: {milk, bread}
- **3-itemset**: {milk, bread, butter}

### Support

How frequently an itemset appears in the dataset

$$\\text{Support}(X) = \\frac{\\text{Transactions containing } X}{\\text{Total transactions}}$$

\`\`\`python
# Calculate support for individual items
from collections import Counter

# Flatten all transactions
all_items = [item for transaction in transactions for item in transaction]
item_counts = Counter(all_items)

print("Item Support:")
for item, count in sorted(item_counts.items(), key=lambda x: x[1], reverse=True):
    support = count / len(transactions)
    print(f"{item:10s}: {count:2d} transactions ({support:5.1%} support)")

# milk appears in 7/10 = 70% of transactions
\`\`\`

### Association Rule

An implication: If X, then Y (written as X → Y)

**Example**: {milk, bread} → {butter}
- "If customer buys milk and bread, they also buy butter"

### Confidence

How often rule is true

$$\\text{Confidence}(X \\rightarrow Y) = \\frac{\\text{Support}(X \\cup Y)}{\\text{Support}(X)}$$

**Interpretation**: Probability of Y given X

\`\`\`python
# Example: {milk} → {bread}
milk_count = sum(1 for t in transactions if 'milk' in t)
milk_and_bread_count = sum(1 for t in transactions if 'milk' in t and 'bread' in t)

confidence = milk_and_bread_count / milk_count
print(f"Rule: {{milk}} → {{bread}}")
print(f"Support(milk): {milk_count}/{len(transactions)} = {milk_count/len(transactions):.1%}")
print(f"Support(milk, bread): {milk_and_bread_count}/{len(transactions)} = {milk_and_bread_count/len(transactions):.1%}")
print(f"Confidence: {milk_and_bread_count}/{milk_count} = {confidence:.1%}")
print(f"\\nInterpretation: {confidence:.0%} of customers who buy milk also buy bread")
\`\`\`

### Lift

How much more likely Y is purchased when X is purchased, compared to just purchasing Y

$$\\text{Lift}(X \\rightarrow Y) = \\frac{\\text{Confidence}(X \\rightarrow Y)}{\\text{Support}(Y)}$$

**Interpretation**:
- Lift > 1: X and Y appear together more than expected (positive correlation)
- Lift = 1: X and Y are independent
- Lift < 1: X and Y appear together less than expected (negative correlation)

\`\`\`python
# Calculate lift for {milk} → {bread}
bread_support = sum(1 for t in transactions if 'bread' in t) / len(transactions)
lift = confidence / bread_support

print(f"\\nLift calculation:")
print(f"Support(bread): {bread_support:.1%}")
print(f"Lift: {confidence:.1%} / {bread_support:.1%} = {lift:.2f}")

if lift > 1:
    print(f"Interpretation: Buying milk makes you {lift:.2f}x more likely to buy bread")
elif lift < 1:
    print(f"Interpretation: Buying milk makes you less likely to buy bread")
else:
    print(f"Interpretation: Milk and bread are independent")
\`\`\`

## The Apriori Algorithm

The most famous algorithm for association rule mining

**Key Insight (Apriori Property)**: 
- If an itemset is frequent, all its subsets must also be frequent
- If an itemset is infrequent, all its supersets must also be infrequent

**Algorithm**:
1. Find all frequent 1-itemsets (single items)
2. Use frequent k-itemsets to generate candidate (k+1)-itemsets
3. Prune candidates using Apriori property
4. Repeat until no more frequent itemsets
5. Generate rules from frequent itemsets

\`\`\`python
# Manual Apriori implementation (simplified)

def get_support(itemset, transactions):
    '''Calculate support for an itemset'''
    count = sum(1 for t in transactions if set(itemset).issubset(set(t)))
    return count / len(transactions)

def find_frequent_itemsets(transactions, min_support):
    '''Find all frequent itemsets using Apriori'''
    from itertools import combinations
    
    # Get all unique items
    unique_items = set(item for transaction in transactions for item in transaction)
    
    # Level 1: frequent 1-itemsets
    frequent_itemsets = []
    for item in unique_items:
        support = get_support([item], transactions)
        if support >= min_support:
            frequent_itemsets.append(([item], support))
    
    print(f"Frequent 1-itemsets: {len([f for f in frequent_itemsets])}")
    
    # Level 2+: generate larger itemsets
    k = 2
    current_frequent = [[item] for item, _ in frequent_itemsets]
    
    while current_frequent:
        # Generate candidates
        candidates = []
        for i in range(len(current_frequent)):
            for j in range(i+1, len(current_frequent)):
                # Combine two itemsets
                union = sorted(set(current_frequent[i]) | set(current_frequent[j]))
                if len(union) == k and union not in candidates:
                    candidates.append(union)
        
        # Check support
        new_frequent = []
        for candidate in candidates:
            support = get_support(candidate, transactions)
            if support >= min_support:
                frequent_itemsets.append((candidate, support))
                new_frequent.append(candidate)
        
        print(f"Frequent {k}-itemsets: {len(new_frequent)}")
        
        current_frequent = new_frequent
        k += 1
        
        if k > 5:  # Safety limit
            break
    
    return frequent_itemsets

# Find frequent itemsets
min_support = 0.3  # 30%
frequent_itemsets = find_frequent_itemsets(transactions, min_support)

print(f"\\nFrequent Itemsets (support >= {min_support:.0%}):")
for itemset, support in sorted(frequent_itemsets, key=lambda x: x[1], reverse=True):
    print(f"{str(itemset):30s} support: {support:.1%}")
\`\`\`

## Using mlxtend Library

\`\`\`python
# Note: Requires mlxtend library
# pip install mlxtend

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    
    # Convert transactions to one-hot encoded format
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    
    print("One-hot encoded transactions:")
    print(df_encoded.head())
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)
    
    print(f"\\nFrequent Itemsets (using mlxtend):")
    print(frequent_itemsets.sort_values('support', ascending=False))
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
    
    print(f"\\nAssociation Rules (confidence >= 60%):")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))
    
    # Visualize rules
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(rules['support'], rules['confidence'], 
                         s=rules['lift']*100, c=rules['lift'], 
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidths=1)
    plt.xlabel('Support', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)
    plt.title('Association Rules\\n(bubble size = lift)', fontsize=14)
    plt.colorbar(scatter, label='Lift')
    plt.grid(True, alpha=0.3)
    
    # Annotate some rules
    for idx, row in rules.iterrows():
        if row['lift'] > 1.2:  # Only annotate interesting rules
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            plt.annotate(f"{antecedents} → {consequents}", 
                        (row['support'], row['confidence']),
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("mlxtend not installed. Install with: pip install mlxtend")
    print("Continuing with manual implementation...")
\`\`\`

## Real-World Example: Larger Dataset

\`\`\`python
# Generate a more realistic dataset
np.random.seed(42)
n_transactions = 1000

# Define products with different probabilities
products = {
    'milk': 0.4,
    'bread': 0.5,
    'eggs': 0.3,
    'butter': 0.25,
    'cheese': 0.2,
    'yogurt': 0.15,
    'coffee': 0.35,
    'tea': 0.2,
    'sugar': 0.15,
    'flour': 0.1
}

# Generate transactions with some correlations
transactions_large = []
for _ in range(n_transactions):
    transaction = []
    
    # Add items based on probabilities
    for product, prob in products.items():
        if np.random.rand() < prob:
            transaction.append(product)
    
    # Add correlations
    if 'coffee' in transaction and np.random.rand() < 0.6:
        transaction.append('sugar')
    if 'bread' in transaction and np.random.rand() < 0.5:
        transaction.append('butter')
    if 'milk' in transaction and np.random.rand() < 0.4:
        transaction.append('eggs')
    
    if transaction:  # Only add non-empty transactions
        transactions_large.append(transaction)

print(f"Generated {len(transactions_large)} transactions")
print(f"Average items per transaction: {np.mean([len(t) for t in transactions_large]):.1f}")

# Sample transactions
print("\\nSample transactions:")
for i in range(5):
    print(f"{i+1}. {transactions_large[i]}")
\`\`\`

### Mining the Large Dataset

\`\`\`python
try:
    # Encode transactions
    te_large = TransactionEncoder()
    te_array_large = te_large.fit(transactions_large).transform(transactions_large)
    df_large = pd.DataFrame(te_array_large, columns=te_large.columns_)
    
    # Apply Apriori
    frequent_large = apriori(df_large, min_support=0.05, use_colnames=True)
    print(f"Found {len(frequent_large)} frequent itemsets")
    
    # Generate rules
    rules_large = association_rules(frequent_large, metric='confidence', min_threshold=0.3)
    print(f"Found {len(rules_large)} association rules")
    
    # Top rules by lift
    print("\\nTop 10 Rules by Lift:")
    top_rules = rules_large.nlargest(10, 'lift')
    for idx, row in top_rules.iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        print(f"{ant:20s} → {con:15s}  "
              f"(support: {row['support']:.2%}, "
              f"confidence: {row['confidence']:.1%}, "
              f"lift: {row['lift']:.2f})")
    
    # Visualize top rules
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Support vs Confidence
    axes[0].scatter(rules_large['support'], rules_large['confidence'], 
                   alpha=0.5, s=30)
    axes[0].set_xlabel('Support')
    axes[0].set_ylabel('Confidence')
    axes[0].set_title('Support vs Confidence')
    axes[0].grid(True, alpha=0.3)
    
    # Support vs Lift
    axes[1].scatter(rules_large['support'], rules_large['lift'], 
                   alpha=0.5, s=30)
    axes[1].set_xlabel('Support')
    axes[1].set_ylabel('Lift')
    axes[1].set_title('Support vs Lift')
    axes[1].axhline(y=1, color='r', linestyle='--', label='Lift=1 (independence)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Confidence vs Lift
    axes[2].scatter(rules_large['confidence'], rules_large['lift'], 
                   alpha=0.5, s=30)
    axes[2].set_xlabel('Confidence')
    axes[2].set_ylabel('Lift')
    axes[2].set_title('Confidence vs Lift')
    axes[2].axhline(y=1, color='r', linestyle='--', label='Lift=1 (independence)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except NameError:
    print("mlxtend not available, skipping large dataset analysis")
\`\`\`

## Filtering and Selecting Rules

### By Metric Thresholds

\`\`\`python
try:
    # High confidence rules
    high_confidence = rules_large[rules_large['confidence'] >= 0.7]
    print(f"Rules with confidence >= 70%: {len(high_confidence)}")
    
    # High lift rules (strong associations)
    high_lift = rules_large[rules_large['lift'] >= 1.5]
    print(f"Rules with lift >= 1.5: {len(high_lift)}")
    
    # Balanced rules (good support, confidence, and lift)
    balanced = rules_large[
        (rules_large['support'] >= 0.1) &
        (rules_large['confidence'] >= 0.6) &
        (rules_large['lift'] >= 1.2)
    ]
    
    print(f"\\nBalanced Rules (support>=10%, confidence>=60%, lift>=1.2):")
    for idx, row in balanced.iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        print(f"{ant} → {con}")
        print(f"  Support: {row['support']:.1%}, "
              f"Confidence: {row['confidence']:.1%}, "
              f"Lift: {row['lift']:.2f}\\n")

except NameError:
    pass
\`\`\`

### By Specific Items

\`\`\`python
try:
    # Rules involving 'coffee'
    coffee_rules = rules_large[
        rules_large['antecedents'].apply(lambda x: 'coffee' in x) |
        rules_large['consequents'].apply(lambda x: 'coffee' in x)
    ]
    
    print(f"Rules involving 'coffee': {len(coffee_rules)}")
    print("\\nTop coffee rules by lift:")
    for idx, row in coffee_rules.nlargest(5, 'lift').iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        print(f"{ant} → {con} (lift: {row['lift']:.2f})")

except NameError:
    pass
\`\`\`

## Evaluation Metrics Summary

| Metric | Formula | Interpretation | Typical Threshold |
|--------|---------|----------------|-------------------|
| **Support** | P(X) | How common is the pattern? | > 0.01 (1%) |
| **Confidence** | P(Y\\|X) | How reliable is the rule? | > 0.6 (60%) |
| **Lift** | P(Y\\|X) / P(Y) | How much better than random? | > 1.0 |
| **Conviction** | [1-P(Y)] / [1-P(Y\\|X)] | How much more likely Y without X? | > 1.0 |
| **Leverage** | P(X,Y) - P(X)P(Y) | Difference from independence | > 0 |

\`\`\`python
try:
    # Calculate additional metrics
    rules_metrics = rules_large.copy()
    
    # Conviction
    rules_metrics['conviction'] = (
        (1 - rules_metrics['consequent support']) /
        (1 - rules_metrics['confidence'])
    )
    
    # Leverage
    rules_metrics['leverage'] = (
        rules_metrics['support'] - 
        (rules_metrics['antecedent support'] * rules_metrics['consequent support'])
    )
    
    print("Sample rules with all metrics:")
    cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conviction', 'leverage']
    print(rules_metrics[cols].head(10).to_string(index=False))

except NameError:
    pass
\`\`\`

## Practical Considerations

### Computational Complexity

Apriori can be slow for large datasets or low support thresholds

\`\`\`python
import time

# Compare different min_support values
try:
    support_values = [0.01, 0.05, 0.1, 0.2]
    results = []
    
    for min_sup in support_values:
        start_time = time.time()
        freq = apriori(df_large, min_support=min_sup, use_colnames=True)
        elapsed_time = time.time() - start_time
        
        results.append({
            'min_support': min_sup,
            'n_itemsets': len(freq),
            'time': elapsed_time
        })
        print(f"min_support={min_sup:.0%}: {len(freq)} itemsets in {elapsed_time:.2f}s")
    
    results_df = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(results_df['min_support']*100, results_df['n_itemsets'], 'go-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Min Support (%)')
    axes[0].set_ylabel('Number of Frequent Itemsets')
    axes[0].set_title('Frequent Itemsets vs Min Support')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results_df['min_support']*100, results_df['time'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Min Support (%)')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Computation Time vs Min Support')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\\nLower min_support → More itemsets → Longer computation time")

except NameError:
    pass
\`\`\`

### Alternative Algorithms

**FP-Growth**: Faster than Apriori for large datasets
- Uses a compressed data structure (FP-tree)
- Avoids candidate generation
- More memory efficient

\`\`\`python
try:
    from mlxtend.frequent_patterns import fpgrowth
    
    # Compare Apriori vs FP-Growth
    start_time = time.time()
    freq_apriori = apriori(df_large, min_support=0.05, use_colnames=True)
    apriori_time = time.time() - start_time
    
    start_time = time.time()
    freq_fpgrowth = fpgrowth(df_large, min_support=0.05, use_colnames=True)
    fpgrowth_time = time.time() - start_time
    
    print(f"Apriori:    {len(freq_apriori)} itemsets in {apriori_time:.3f}s")
    print(f"FP-Growth:  {len(freq_fpgrowth)} itemsets in {fpgrowth_time:.3f}s")
    print(f"FP-Growth is {apriori_time/fpgrowth_time:.1f}x faster")

except (ImportError, NameError):
    print("FP-Growth comparison not available")
\`\`\`

## Real-World Applications

### Application 1: Product Recommendations

\`\`\`python
def recommend_products(cart_items, rules_df, n_recommendations=3):
    '''Recommend products based on current cart'''
    # Find rules where antecedents match cart items
    recommendations = []
    
    for idx, row in rules_df.iterrows():
        if set(row['antecedents']).issubset(set(cart_items)):
            for item in row['consequents']:
                if item not in cart_items:
                    recommendations.append({
                        'item': item,
                        'confidence': row['confidence'],
                        'lift': row['lift']
                    })
    
    # Sort by confidence * lift
    recommendations_df = pd.DataFrame(recommendations)
    if len(recommendations_df) > 0:
        recommendations_df['score'] = recommendations_df['confidence'] * recommendations_df['lift']
        recommendations_df = recommendations_df.drop_duplicates('item')
        recommendations_df = recommendations_df.nlargest(n_recommendations, 'score')
        return recommendations_df['item'].tolist()
    return []

try:
    # Example: Customer has milk and bread in cart
    cart = ['milk', 'bread']
    recommended = recommend_products(cart, rules_large, n_recommendations=5)
    
    print(f"Items in cart: {cart}")
    print(f"Recommended products: {recommended}")
    print("\\nThis is how e-commerce 'Frequently Bought Together' works!")

except NameError:
    print("Recommendation system demo not available")
\`\`\`

### Application 2: Store Layout Optimization

\`\`\`python
# Products frequently bought together should be placed apart
# to increase customer journey through store

try:
    strong_associations = rules_large[rules_large['lift'] > 1.5].nlargest(10, 'confidence')
    
    print("Products to place APART (frequently bought together):")
    for idx, row in strong_associations.iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        print(f"  {ant} <---> {con} (lift: {row['lift']:.2f})")

except NameError:
    pass
\`\`\`

### Application 3: Healthcare - Comorbidity Analysis

\`\`\`python
# Example: Medical conditions that co-occur
medical_records = [
    ['diabetes', 'hypertension', 'obesity'],
    ['hypertension', 'heart_disease'],
    ['diabetes', 'obesity'],
    ['asthma', 'allergies'],
    ['diabetes', 'hypertension'],
    ['obesity', 'heart_disease'],
    ['diabetes', 'obesity', 'hypertension'],
    ['asthma'],
    ['hypertension', 'obesity'],
    ['diabetes', 'heart_disease']
]

try:
    # Encode
    te_medical = TransactionEncoder()
    te_array_medical = te_medical.fit(medical_records).transform(medical_records)
    df_medical = pd.DataFrame(te_array_medical, columns=te_medical.columns_)
    
    # Find associations
    freq_medical = apriori(df_medical, min_support=0.3, use_colnames=True)
    rules_medical = association_rules(freq_medical, metric='confidence', min_threshold=0.5)
    
    print("Medical Comorbidity Patterns:")
    for idx, row in rules_medical.nlargest(5, 'lift').iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        print(f"{ant:25s} → {con:20s} "
              f"(confidence: {row['confidence']:.1%}, lift: {row['lift']:.2f})")
    
    print("\\nThese patterns help doctors:")
    print("- Screen for comorbid conditions")
    print("- Understand disease relationships")
    print("- Develop treatment protocols")

except NameError:
    pass
\`\`\`

## Best Practices

1. **Set Appropriate Thresholds**:
   - min_support: Start with 1%, adjust based on dataset
   - min_confidence: Typically 60-80%
   - min_lift: > 1.0 (preferably > 1.2)

2. **Interpret Carefully**:
   - Lift > 1 doesn't mean causation!
   - Consider domain knowledge
   - Validate with experts
   - Check for spurious correlations

3. **Handle Large Itemsets**:
   - Use FP-Growth for speed
   - Prune trivial rules
   - Focus on actionable rules

4. **Temporal Aspects**:
   - Update rules periodically
   - Consider seasonal patterns
   - Monitor rule stability

5. **Avoid Common Pitfalls**:
   - Don't mine with very low support (exponential explosion)
   - Don't ignore lift (high confidence ≠ interesting)
   - Don't assume causality
   - Don't forget domain context

## Limitations

❌ **Not Causal**: Association ≠ causation  
❌ **Computational Cost**: Exponential in worst case  
❌ **Many Rules**: Difficult to interpret thousands of rules  
❌ **Static**: Doesn't capture temporal or sequential patterns  
❌ **Binary**: Doesn't handle quantities or ordinal data well  

## Summary

**Key Takeaways**:

- **Association Rule Learning** discovers patterns in transactional data
- **Apriori Algorithm** efficiently mines frequent itemsets
- **Key Metrics**: Support, Confidence, Lift
- **Applications**: Retail, healthcare, web mining, recommendations

**Metrics**:
- **Support**: How common is the pattern?
- **Confidence**: How reliable is the rule?
- **Lift**: How much better than random?

**Best Practices**:
1. Use appropriate thresholds (domain-dependent)
2. Focus on high-lift rules (interesting associations)
3. Validate with domain experts
4. Consider FP-Growth for large datasets
5. Don't confuse association with causation

**When to Use**:
- Transactional data (market baskets, clickstreams)
- Looking for co-occurring patterns
- Recommendation systems
- Cross-selling opportunities
- Pattern discovery

**Alternatives**:
- **Sequential Pattern Mining**: For ordered patterns
- **Collaborative Filtering**: For personalized recommendations
- **Graph-Based Methods**: For complex relationships

This completes our tour of unsupervised learning! From clustering (K-Means, Hierarchical, DBSCAN) to dimensionality reduction (PCA, t-SNE, UMAP) to anomaly detection and association rules, you now have a comprehensive toolkit for discovering patterns in unlabeled data.
`,
};
