/**
 * Quiz questions for Set Theory & Logic section
 */

export const settheorylogicQuiz = [
  {
    id: 'dq1-feature-selection-sets',
    question:
      'Explain how set theory is used in feature selection for machine learning. Discuss multiple feature selection methods (correlation-based, importance-based, recursive elimination) and how set operations (union, intersection, difference) help combine or compare their results. Provide concrete examples with code showing how to identify robust features.',
    sampleAnswer: `Set theory provides an elegant framework for feature selection, allowing us to combine insights from multiple selection methods and reason about feature relationships.

**Feature Selection as Set Operations**:

Each feature selection method produces a set of selected features. Set operations let us:
- Find features selected by all methods (intersection = robust)
- Combine features from multiple methods (union = comprehensive)
- Find method-specific features (difference = unique insights)
- Compare method agreements (symmetric difference = disagreements)

**Implementation**:

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, n_repeated=0, random_state=42)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Method 1: Correlation-based (ANOVA F-statistic)
selector_corr = SelectKBest(f_classif, k=10)
selector_corr.fit(X, y)
features_correlation = set([feature_names[i] for i in selector_corr.get_support(indices=True)])

print(f"\\nCorrelation-based: {len(features_correlation)} features")
print(f"  {features_correlation}")

# Method 2: Tree-based importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
# Select top 10 by importance
top_indices = np.argsort(importances)[-10:]
features_importance = set([feature_names[i] for i in top_indices])

print(f"\\nImportance-based: {len(features_importance)} features")
print(f"  {features_importance}")

# Method 3: Recursive Feature Elimination
estimator = LogisticRegression(max_iter=1000)
selector_rfe = RFE(estimator, n_features_to_select=10, step=1)
selector_rfe.fit(X, y)
features_rfe = set([feature_names[i] for i in selector_rfe.get_support(indices=True)])

print(f"\\nRFE-based: {len(features_rfe)} features")
print(f"  {features_rfe}")
\`\`\`

**Set Operations for Analysis**:

\`\`\`python
# Intersection: Features selected by ALL methods (most robust)
robust_features = features_correlation & features_importance & features_rfe
print(f"\\nRobust features (all 3 methods): {len(robust_features)}")
print(f"  {robust_features}")

# Union: Features selected by ANY method (comprehensive)
all_selected = features_correlation | features_importance | features_rfe
print(f"\\nAll selected features (any method): {len(all_selected)}")
print(f"  {all_selected}")

# Pairwise intersections
corr_imp = features_correlation & features_importance
corr_rfe = features_correlation & features_rfe
imp_rfe = features_importance & features_rfe

print(f"\\nPairwise agreements:")
print(f"  Correlation ∩ Importance: {len(corr_imp)} features")
print(f"  Correlation ∩ RFE: {len(corr_rfe)} features")
print(f"  Importance ∩ RFE: {len(imp_rfe)} features")

# Features unique to each method
unique_to_corr = features_correlation - features_importance - features_rfe
unique_to_imp = features_importance - features_correlation - features_rfe
unique_to_rfe = features_rfe - features_correlation - features_importance

print(f"\\nUnique selections:")
print(f"  Only Correlation: {unique_to_corr}")
print(f"  Only Importance: {unique_to_imp}")
print(f"  Only RFE: {unique_to_rfe}")

# Symmetric difference: features with disagreement
disagreement_corr_imp = features_correlation ^ features_importance
print(f"\\nDisagreement (Correlation △ Importance): {disagreement_corr_imp}")
\`\`\`

**Visualizing Set Relationships**:

\`\`\`python
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
venn3([features_correlation, features_importance, features_rfe],
      set_labels=('Correlation', 'Importance', 'RFE'))
plt.title('Feature Selection Methods: Set Relationships')
plt.show()
\`\`\`

**Decision Strategy Using Sets**:

**Strategy 1: Conservative (Intersection)**
- Use features selected by all methods
- High confidence, may miss some useful features
- Best for: high-stakes applications, limited compute

\`\`\`python
conservative_features = features_correlation & features_importance & features_rfe
print(f"\\nConservative strategy: {len(conservative_features)} features")
\`\`\`

**Strategy 2: Majority Vote**
- Use features selected by at least 2 methods
- Balanced approach

\`\`\`python
# Count votes for each feature
all_features = features_correlation | features_importance | features_rfe
feature_votes = {}

for feature in all_features:
    votes = 0
    if feature in features_correlation:
        votes += 1
    if feature in features_importance:
        votes += 1
    if feature in features_rfe:
        votes += 1
    feature_votes[feature] = votes

majority_features = {f for f, v in feature_votes.items() if v >= 2}
print(f"\\nMajority vote (≥2 methods): {len(majority_features)} features")
print(f"  {majority_features}")
\`\`\`

**Strategy 3: Aggressive (Union)**
- Use features from any method
- Maximum coverage, may include noise
- Best for: exploration, ensemble models

\`\`\`python
aggressive_features = features_correlation | features_importance | features_rfe
print(f"\\nAggressive strategy: {len(aggressive_features)} features")
\`\`\`

**Analyzing Method Agreement**:

\`\`\`python
def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

# Compute pairwise similarities
sim_corr_imp = jaccard_similarity(features_correlation, features_importance)
sim_corr_rfe = jaccard_similarity(features_correlation, features_rfe)
sim_imp_rfe = jaccard_similarity(features_importance, features_rfe)

print(f"\\nMethod Agreement (Jaccard Similarity):")
print(f"  Correlation ↔ Importance: {sim_corr_imp:.3f}")
print(f"  Correlation ↔ RFE: {sim_corr_rfe:.3f}")
print(f"  Importance ↔ RFE: {sim_imp_rfe:.3f}")

# Overall agreement
overall_agreement = len(robust_features) / len(all_selected)
print(f"\\nOverall agreement: {overall_agreement:.3f}")
print(f"({len(robust_features)} robust / {len(all_selected)} total)")
\`\`\`

**Trading Application**:

\`\`\`python
# Example: Selecting features for stock prediction model

# Different feature sets from domain knowledge
technical_indicators = {'RSI', 'MACD', 'SMA_20', 'SMA_50', 'volume', 'volatility'}
fundamental_features = {'PE_ratio', 'earnings', 'revenue', 'debt_ratio', 'ROE'}
sentiment_features = {'news_sentiment', 'social_sentiment', 'analyst_rating'}

# Statistical selection from backtesting
selected_by_sharpe = {'RSI', 'MACD', 'volume', 'earnings', 'news_sentiment'}
selected_by_sortino = {'SMA_20', 'SMA_50', 'PE_ratio', 'analyst_rating', 'volatility'}

# Combine domain knowledge with statistical selection
# Strategy: Use technical features that are statistically validated
validated_technical = technical_indicators & (selected_by_sharpe | selected_by_sortino)
print(f"\\nValidated technical indicators: {validated_technical}")

# Add fundamental features selected by either metric
validated_fundamental = fundamental_features & (selected_by_sharpe | selected_by_sortino)
print(f"Validated fundamental features: {validated_fundamental}")

# Final feature set
final_features = validated_technical | validated_fundamental
print(f"\\nFinal feature set: {final_features}")
print(f"Total features: {len(final_features)}")
\`\`\`

**Key Insights**:

1. **Intersection gives confidence**: Features selected by multiple methods are likely truly important

2. **Union enables exploration**: Including all candidates helps discover unexpected relationships

3. **Difference reveals specialization**: Each method captures different aspects

4. **Set size trades off**: Larger sets = more information but also more noise/compute

5. **Jaccard similarity quantifies agreement**: Low similarity suggests methods capture complementary information

**Summary**:
Set operations provide a principled way to combine feature selection methods, moving beyond arbitrary choices to systematic analysis of feature importance across multiple perspectives.`,
    keyPoints: [
      'Each selection method produces a set of features',
      'Intersection (∩) finds robust features selected by all methods',
      'Union (∪) combines features from all methods for comprehensive coverage',
      'Difference (\\) identifies method-specific selections',
      'Jaccard similarity quantifies agreement between methods',
      'Strategy choice (conservative/majority/aggressive) depends on application requirements',
    ],
  },
  {
    id: 'dq2-logic-decision-trees',
    question:
      "Decision trees use logical conditions to make predictions. Explain how Boolean logic (AND, OR, NOT) maps to decision tree structure. How can complex logical expressions be represented as trees? Discuss De Morgan's laws in the context of decision tree simplification and provide examples showing equivalent tree representations.",
    sampleAnswer: `Decision trees are essentially visual representations of logical expressions, where each path from root to leaf represents a conjunction (AND) of conditions, and the tree as a whole represents a disjunction (OR) of these paths.

**Decision Trees as Logic**:

**Basic Structure**:
- Each node tests a condition (Boolean expression)
- Left/right branches represent TRUE/FALSE
- Leaf nodes give predictions
- Path from root to leaf = AND of conditions
- Multiple paths to same prediction = OR of conditions

**Example Tree**:

\`\`\`
         age >= 25?
          /      \\
        YES       NO
        /          \\
   income > 50k?   REJECT
      /      \\
    YES      NO
    /         \\
  APPROVE   credit > 700?
              /        \\
            YES        NO
            /           \\
         APPROVE      REJECT
\`\`\`

**As Logical Expression**:
\`\`\`
APPROVE = (age >= 25 ∧ income > 50k) 
          ∨ (age >= 25 ∧ income ≤ 50k ∧ credit > 700)
\`\`\`

**Implementation**:

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({
    'age': [22, 30, 35, 28, 45, 25, 32, 27, 40, 23],
    'income': [35000, 60000, 75000, 45000, 90000, 55000, 48000, 40000, 80000, 38000],
    'credit_score': [620, 680, 720, 650, 750, 700, 640, 710, 730, 630],
    'approved': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
})

X = data[['age', 'income', 'credit_score']]
y = data['approved']

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=['age', 'income', 'credit_score'],
          class_names=['Reject', 'Approve'], filled=True, fontsize=10)
plt.show()

# Extract rules as text
tree_rules = export_text(tree, feature_names=['age', 'income', 'credit_score'])
print("Decision Tree Rules:")
print(tree_rules)
\`\`\`

**Manual Logic Implementation**:

\`\`\`python
def decision_tree_as_logic(age, income, credit_score):
    """
    Manually implement tree logic
    Shows explicit AND/OR structure
    """
    # Path 1: age >= 25 AND income > 50000
    path1 = (age >= 25) and (income > 50000)
    
    # Path 2: age >= 25 AND income <= 50000 AND credit_score > 700
    path2 = (age >= 25) and (income <= 50000) and (credit_score > 700)
    
    # Path 3: age < 25 AND credit_score > 720
    path3 = (age < 25) and (credit_score > 720)
    
    # Final decision: OR of all approval paths
    approve = path1 or path2 or path3
    
    return approve, (path1, path2, path3)

# Test
test_cases = [
    (30, 60000, 650, "Path 1: mature + high income"),
    (28, 45000, 750, "Path 2: mature + low income but great credit"),
    (23, 40000, 730, "Path 3: young but excellent credit"),
    (35, 75000, 720, "Path 1: mature + high income"),
    (22, 35000, 650, "No path: young, low income, ok credit"),
]

print("\\nTesting logical paths:")
for age, income, credit, desc in test_cases:
    approve, paths = decision_tree_as_logic(age, income, credit)
    active_paths = [i+1 for i, p in enumerate(paths) if p]
    print(f"{desc}")
    print(f"  Input: age={age}, income={income}, credit={credit}")
    print(f"  Decision: {'APPROVE' if approve else 'REJECT'}")
    print(f"  Active paths: {active_paths if active_paths else 'None'}")
\`\`\`

**De Morgan's Laws in Trees**:

De Morgan's laws allow us to transform decision trees:
- ¬(A ∧ B) = ¬A ∨ ¬B
- ¬(A ∨ B) = ¬A ∧ ¬B

**Example - Rejection Condition**:

Instead of defining approval conditions, we can define rejection:

\`\`\`python
def approval_positive_logic(age, income, credit):
    """Define APPROVAL conditions (positive logic)"""
    return ((age >= 25 and income > 50000) or
            (age >= 25 and income <= 50000 and credit > 700) or
            (age < 25 and credit > 720))

def approval_negative_logic(age, income, credit):
    """
    Define REJECTION conditions (negative logic)
    Then negate to get approval
    
    Using De Morgan's laws to transform
    """
    # REJECT if:
    # NOT[(age >= 25 ∧ income > 50000) ∨ ...]
    # = [¬(age >= 25 ∧ income > 50000)] ∧ [¬(...)] ∧ [¬(...)]
    
    not_path1 = not ((age >= 25) and (income > 50000))
    not_path2 = not ((age >= 25) and (income <= 50000) and (credit > 700))
    not_path3 = not ((age < 25) and (credit > 720))
    
    reject = not_path1 and not_path2 and not_path3
    approve = not reject
    
    return approve

# Verify equivalence
for age, income, credit, _ in test_cases:
    pos = approval_positive_logic(age, income, credit)
    neg = approval_negative_logic(age, income, credit)
    print(f"age={age}, income={income}, credit={credit}: "
          f"Positive={pos}, Negative={neg}, Match={pos == neg}")
\`\`\`

**Tree Simplification with De Morgan's**:

\`\`\`python
# Original condition (complex)
def original_condition(x1, x2, x3):
    return not ((x1 and x2) or x3)

# Apply De Morgan's: ¬(A ∨ B) = ¬A ∧ ¬B
# ¬[(x1 ∧ x2) ∨ x3] = ¬(x1 ∧ x2) ∧ ¬x3

# Apply again: ¬(A ∧ B) = ¬A ∨ ¬B
# ¬(x1 ∧ x2) = ¬x1 ∨ ¬x2

# Final simplified:
def simplified_condition(x1, x2, x3):
    return ((not x1) or (not x2)) and (not x3)

# Verify equivalence
print("\\nDe Morgan's Simplification:")
for x1 in [True, False]:
    for x2 in [True, False]:
        for x3 in [True, False]:
            orig = original_condition(x1, x2, x3)
            simp = simplified_condition(x1, x2, x3)
            match = "✓" if orig == simp else "✗"
            print(f"x1={x1}, x2={x2}, x3={x3}: "
                  f"Original={orig}, Simplified={simp} {match}")
\`\`\`

**Equivalent Tree Representations**:

The same logical expression can be represented by different tree structures:

\`\`\`python
# Representation 1: (A ∧ B) ∨ (C ∧ D)
def tree_representation_1(A, B, C, D):
    """
    Tree splits on A first:
           A?
          / \\
        B?   C?
        /\\   /\\
       T F  D? F
           /\\
          T  F
    """
    if A:
        return B  # If A, result depends on B
    else:
        return C and D  # If not A, need both C and D

# Representation 2: Equivalent but splits on C first
def tree_representation_2(A, B, C, D):
    """
    Tree splits on C first:
           C?
          / \\
        D?   A?
        /\\   /\\
       T F  B? F
           /\\
          T  F
    """
    if C:
        return D  # If C, result depends on D
    else:
        return A and B  # If not C, need both A and B

# Both represent: (A ∧ B) ∨ (C ∧ D)
# Verify equivalence
print("\\nEquivalent Tree Representations:")
for A in [True, False]:
    for B in [True, False]:
        for C in [True, False]:
            for D in [True, False]:
                r1 = tree_representation_1(A, B, C, D)
                r2 = tree_representation_2(A, B, C, D)
                if r1 != r2:
                    print(f"MISMATCH: A={A}, B={B}, C={C}, D={D}")
                    
print("All cases match! Trees are equivalent.")
\`\`\`

**Trading Application**:

\`\`\`python
# Trading signal decision tree

def trading_signal_positive(price_above_sma, rsi_oversold, volume_high, trend_up):
    """
    BUY signal if:
    (price below SMA AND RSI oversold) OR (volume high AND trend up)
    
    Positive logic formulation
    """
    signal_1 = (not price_above_sma) and rsi_oversold
    signal_2 = volume_high and trend_up
    
    return signal_1 or signal_2

def trading_signal_negative(price_above_sma, rsi_oversold, volume_high, trend_up):
    """
    Same logic using De Morgan's transformation
    Negative logic: when NOT to buy
    
    ¬BUY = ¬[(¬price_above_sma ∧ rsi_oversold) ∨ (volume_high ∧ trend_up)]
         = [¬(¬price_above_sma ∧ rsi_oversold)] ∧ [¬(volume_high ∧ trend_up)]
         = [(price_above_sma ∨ ¬rsi_oversold)] ∧ [(¬volume_high ∨ ¬trend_up)]
    """
    not_signal_1 = price_above_sma or (not rsi_oversold)
    not_signal_2 = (not volume_high) or (not trend_up)
    
    no_buy = not_signal_1 and not_signal_2
    return not no_buy

# Test
test_scenarios = [
    (False, True, False, False, "Below SMA + oversold"),
    (True, False, True, True, "Above SMA but high volume + uptrend"),
    (False, False, False, False, "No clear signal"),
    (True, True, True, True, "All positive indicators"),
]

print("\\nTrading Signal Logic:")
for price_above, rsi_os, vol_high, trend, desc in test_scenarios:
    pos = trading_signal_positive(price_above, rsi_os, vol_high, trend)
    neg = trading_signal_negative(price_above, rsi_os, vol_high, trend)
    print(f"{desc}: {'BUY' if pos else 'HOLD'} (Match: {pos == neg})")
\`\`\`

**Summary**:
- Decision trees = visual logic (AND for paths, OR for multiple paths)
- De Morgan's laws enable tree transformations and simplifications
- Same logic can have multiple equivalent tree structures
- Understanding logical equivalences helps optimize decision trees
- Negative logic (rejection rules) often simpler than positive logic`,
    keyPoints: [
      'Decision tree paths are AND operations (conjunction of conditions)',
      'Multiple paths to same class are OR operations (disjunction)',
      "De Morgan's laws: ¬(A∧B)=¬A∨¬B and ¬(A∨B)=¬A∧¬B enable transformations",
      'Same logical expression can be represented by different tree structures',
      'Negative logic (rejection conditions) sometimes simpler than positive',
      'Tree simplification using logical equivalences reduces complexity',
    ],
  },
  {
    id: 'dq3-set-operations-data-splitting',
    question:
      'In machine learning, train/validation/test splits must be disjoint sets with no overlap. Explain using set theory: (1) Why splits must be disjoint, (2) How to verify no data leakage using set operations, (3) Stratification as preserving set proportions, (4) Cross-validation as set partitioning, (5) Time-series splits and ordered sets. Provide code examples.',
    sampleAnswer: `Set theory provides a rigorous framework for understanding data splitting in machine learning, ensuring proper evaluation and preventing data leakage.

**1. Why Splits Must Be Disjoint**:

**Definition**: Sets A and B are **disjoint** if A ∩ B = ∅ (empty set)

**Requirement**: Train ∩ Val ∩ Test = ∅

**Reason**: To accurately estimate generalization performance
- Training set: Learn patterns
- Validation set: Tune hyperparameters  
- Test set: Final evaluation

If sets overlap → model has seen test data → overoptimistic performance estimate

**Mathematical Formulation**:

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create dataset
n_samples = 1000
all_indices = set(range(n_samples))

# Split into train/val/test
train_size = 0.6
val_size = 0.2
test_size = 0.2

# First split: train vs temp
train_indices, temp_indices = train_test_split(
    list(all_indices), train_size=train_size, random_state=42
)
train_indices = set(train_indices)
temp_indices = set(temp_indices)

# Second split: val vs test
val_indices, test_indices = train_test_split(
    list(temp_indices), train_size=val_size/(val_size + test_size), random_state=42
)
val_indices = set(val_indices)
test_indices = set(test_indices)

print("Set Sizes:")
print(f"Total: {len(all_indices)}")
print(f"Train: {len(train_indices)} ({len(train_indices)/n_samples:.1%})")
print(f"Val: {len(val_indices)} ({len(val_indices)/n_samples:.1%})")
print(f"Test: {len(test_indices)} ({len(test_indices)/n_samples:.1%})")
\`\`\`

**2. Verifying No Data Leakage with Set Operations**:

\`\`\`python
def verify_data_splits(train, val, test, all_data):
    """
    Comprehensive verification of data splits using set theory
    """
    print("\\n=== Data Split Verification ===\\n")
    
    # Check 1: Pairwise disjoint (no overlap)
    train_val_overlap = train & val
    train_test_overlap = train & test
    val_test_overlap = val & test
    
    print("1. Disjoint Sets (must be empty):")
    print(f"   Train ∩ Val = {train_val_overlap} (size: {len(train_val_overlap)})")
    print(f"   Train ∩ Test = {train_test_overlap} (size: {len(train_test_overlap)})")
    print(f"   Val ∩ Test = {val_test_overlap} (size: {len(val_test_overlap)})")
    
    all_disjoint = (len(train_val_overlap) == 0 and 
                    len(train_test_overlap) == 0 and 
                    len(val_test_overlap) == 0)
    print(f"   ✓ All disjoint: {all_disjoint}")
    
    # Check 2: Union equals original (no missing data)
    union = train | val | test
    print(f"\\n2. Complete Coverage:")
    print(f"   Train ∪ Val ∪ Test = {len(union)} samples")
    print(f"   Original data = {len(all_data)} samples")
    print(f"   ✓ Complete: {union == all_data}")
    
    # Check 3: No missing indices
    missing = all_data - union
    print(f"\\n3. Missing Data:")
    print(f"   All - Union = {missing} (size: {len(missing)})")
    print(f"   ✓ No missing: {len(missing) == 0}")
    
    # Check 4: No duplicate indices within sets
    print(f"\\n4. No Duplicates (set property automatically enforced):")
    print(f"   ✓ Sets inherently have no duplicates")
    
    # Summary
    is_valid = all_disjoint and (union == all_data) and (len(missing) == 0)
    print(f"\\n{'='*40}")
    print(f"Overall: {'✓ VALID SPLIT' if is_valid else '✗ INVALID SPLIT'}")
    print(f"{'='*40}")
    
    return is_valid

# Verify our splits
is_valid = verify_data_splits(train_indices, val_indices, test_indices, all_indices)
\`\`\`

**3. Stratification as Preserving Set Proportions**:

**Goal**: Maintain class distribution across splits

\`\`\`python
from sklearn.model_split import train_test_split
from collections import Counter

# Create imbalanced dataset
y = np.array([0]*700 + [1]*250 + [2]*50)  # Imbalanced classes
X = np.arange(len(y)).reshape(-1, 1)

print("Original class distribution:")
original_dist = Counter(y)
for class_label, count in sorted(original_dist.items()):
    print(f"  Class {class_label}: {count} ({count/len(y):.1%})")

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.6, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42
)

# Analyze stratification using sets
train_by_class = {c: set(X_train[y_train == c].flatten()) for c in [0, 1, 2]}
val_by_class = {c: set(X_val[y_val == c].flatten()) for c in [0, 1, 2]}
test_by_class = {c: set(X_test[y_test == c].flatten()) for c in [0, 1, 2]}

print("\\nStratified split class distributions:")
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"\\n{split_name}:")
    split_dist = Counter(y_split)
    for class_label, count in sorted(split_dist.items()):
        original_prop = original_dist[class_label] / len(y)
        split_prop = count / len(y_split)
        print(f"  Class {class_label}: {count} ({split_prop:.1%}) "
              f"[Original: {original_prop:.1%}]")

# Verify class-wise disjointness
print("\\nClass-wise disjoint verification:")
for c in [0, 1, 2]:
    overlap_train_val = train_by_class[c] & val_by_class[c]
    overlap_train_test = train_by_class[c] & test_by_class[c]
    overlap_val_test = val_by_class[c] & test_by_class[c]
    
    all_disjoint = (len(overlap_train_val) == 0 and 
                    len(overlap_train_test) == 0 and 
                    len(overlap_val_test) == 0)
    print(f"  Class {c}: {'✓ Disjoint' if all_disjoint else '✗ Overlap detected'}")
\`\`\`

**4. Cross-Validation as Set Partitioning**:

**Definition**: Partition dataset into k disjoint subsets (folds)

**Properties**:
- Fold₁ ∪ Fold₂ ∪ ... ∪ Foldₖ = All Data
- Foldᵢ ∩ Foldⱼ = ∅ for i ≠ j

\`\`\`python
from sklearn.model_selection import KFold

# K-Fold Cross-Validation
n_samples = 100
k_folds = 5

indices = np.arange(n_samples)
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Create folds as sets
folds = []
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
    fold = set(val_idx)
    folds.append(fold)
    print(f"Fold {fold_idx + 1}: {len(fold)} samples")

# Verify partition properties
print("\\n=== Partition Verification ===")

# 1. Union equals all data
union_folds = set().union(*folds)
all_data = set(indices)
print(f"\\n1. Union = All Data: {union_folds == all_data}")

# 2. Pairwise disjoint
print(f"\\n2. Pairwise Disjoint:")
all_disjoint = True
for i in range(len(folds)):
    for j in range(i + 1, len(folds)):
        overlap = folds[i] & folds[j]
        if len(overlap) > 0:
            print(f"   Fold {i+1} ∩ Fold {j+1}: {len(overlap)} (PROBLEM!)")
            all_disjoint = False

if all_disjoint:
    print(f"   ✓ All folds are pairwise disjoint")

# 3. Equal sizes (approximately)
sizes = [len(fold) for fold in folds]
print(f"\\n3. Fold Sizes: {sizes}")
print(f"   Min: {min(sizes)}, Max: {max(sizes)}, Difference: {max(sizes) - min(sizes)}")

# Visualize CV folds
print("\\n=== Cross-Validation Iterations ===")
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
    train_set = set(train_idx)
    val_set = set(val_idx)
    
    print(f"\\nIteration {fold_idx + 1}:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val: {len(val_set)} samples")
    print(f"  Disjoint: {len(train_set & val_set) == 0}")
    print(f"  Union = All: {(train_set | val_set) == all_data}")
\`\`\`

**5. Time-Series Splits and Ordered Sets**:

**Key Difference**: Time series requires preserving temporal order

**Time-Based Partitioning**:
- Train: {t₁, t₂, ..., tₙ}
- Test: {tₙ₊₁, tₙ₊₂, ..., tₘ}
- Constraint: max(Train) < min(Test)

\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

# Simulate time series data
n_samples = 100
dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
data = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(n_samples),
    'index': range(n_samples)
})

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

print("=== Time Series Cross-Validation ===\\n")

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data)):
    train_set = set(train_idx)
    test_set = set(test_idx)
    
    print(f"Fold {fold_idx + 1}:")
    print(f"  Train: indices {min(train_idx)} to {max(train_idx)} ({len(train_set)} samples)")
    print(f"  Test: indices {min(test_idx)} to {max(test_idx)} ({len(test_set)} samples)")
    
    # Verify temporal ordering
    max_train = max(train_idx)
    min_test = min(test_idx)
    temporal_order_preserved = max_train < min_test
    
    # Verify disjoint
    disjoint = len(train_set & test_set) == 0
    
    print(f"  Temporal order preserved: {temporal_order_preserved}")
    print(f"  Disjoint: {disjoint}")
    print()

# Visualize temporal splits
plt.figure(figsize=(14, 8))

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data)):
    plt.subplot(5, 1, fold_idx + 1)
    
    # Plot train as blue, test as red
    train_mask = np.zeros(n_samples)
    train_mask[train_idx] = 1
    test_mask = np.zeros(n_samples)
    test_mask[test_idx] = 2
    
    combined = train_mask + test_mask
    plt.scatter(range(n_samples), [fold_idx + 1] * n_samples, 
                c=combined, cmap='RdBu', s=50, marker='|')
    plt.ylabel(f'Fold {fold_idx + 1}', rotation=0, labelpad=20)
    plt.yticks([])
    
    if fold_idx == 4:
        plt.xlabel('Time Index')

plt.suptitle('Time Series Cross-Validation Splits (Blue=Train, Red=Test)')
plt.tight_layout()
plt.show()
\`\`\`

**Trading Application - Walk-Forward Validation**:

\`\`\`python
def walk_forward_validation(data, train_window, test_window):
    """
    Walk-forward validation for trading strategies
    Maintains temporal order and disjoint sets
    """
    n = len(data)
    splits = []
    
    start = 0
    while start + train_window + test_window <= n:
        train_end = start + train_window
        test_end = train_end + test_window
        
        train_indices = set(range(start, train_end))
        test_indices = set(range(train_end, test_end))
        
        splits.append((train_indices, test_indices))
        start += test_window  # Move forward by test window
    
    return splits

# Example: 500 days of trading data
n_days = 500
train_window = 252  # 1 year
test_window = 63    # Quarter

splits = walk_forward_validation(range(n_days), train_window, test_window)

print(f"=== Walk-Forward Validation ===")
print(f"Total periods: {len(splits)}\\n")

for i, (train, test) in enumerate(splits):
    print(f"Period {i + 1}:")
    print(f"  Train: days {min(train)} to {max(train)} ({len(train)} days)")
    print(f"  Test: days {min(test)} to {max(test)} ({len(test)} days)")
    
    # Verify properties
    disjoint = len(train & test) == 0
    temporal = max(train) < min(test)
    print(f"  Disjoint: {disjoint}, Temporal order: {temporal}")
    print()
\`\`\`

**Summary**:
- Disjoint sets (Train ∩ Val ∩ Test = ∅) prevent data leakage
- Set operations verify split validity systematically
- Stratification preserves class proportions across splits
- Cross-validation partitions data into k disjoint folds
- Time series requires ordered sets with temporal constraints
- Set theory provides rigorous framework for proper evaluation`,
    keyPoints: [
      'Splits must be disjoint (no overlap) to prevent data leakage',
      'Verify splits: Train ∩ Val ∩ Test = ∅ and Train ∪ Val ∪ Test = All',
      'Stratification maintains class proportions across splits',
      'Cross-validation partitions data into k disjoint, equal-sized folds',
      'Time series splits require temporal ordering: max(Train) < min(Test)',
      'Set operations provide systematic verification of split validity',
    ],
  },
];
