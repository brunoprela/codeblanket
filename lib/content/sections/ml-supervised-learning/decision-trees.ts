/**
 * Decision Trees Section
 */

export const decisiontreesSection = {
  id: 'decision-trees',
  title: 'Decision Trees',
  content: `# Decision Trees

## Introduction

Decision trees are intuitive, interpretable machine learning models that make predictions by learning a series of if-then-else decision rules from data. They work for both classification and regression and serve as the foundation for powerful ensemble methods like Random Forests and Gradient Boosting.

**Key Characteristics:**
- Tree structure with nodes and branches
- Non-parametric (no assumptions about data distribution)
- Interpretable and visualizable
- Handles both numerical and categorical features
- Can capture non-linear relationships
- Foundation for ensemble methods

**Applications:**
- Medical diagnosis (decision protocols)
- Credit risk assessment
- Customer segmentation
- Fraud detection
- Feature importance analysis

## Tree Structure

**Components:**
- **Root node**: Top of tree, first split
- **Internal nodes**: Decision points (splits)
- **Leaf nodes**: Final predictions
- **Branches**: Paths from root to leaves

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X = iris.data[:, 2:]  # Use only petal features for visualization
y = iris.target

# Remove one class for binary classification (easier to visualize)
X_binary = X[y != 2]
y_binary = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(15, 8))
plot_tree(tree, feature_names=['petal length', 'petal width'],
         class_names=['setosa', 'versicolor'], filled=True, 
         rounded=True, fontsize=10)
plt.title('Decision Tree Structure')
plt.show()

print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")
print(f"Training accuracy: {tree.score(X_train, y_train):.4f}")
print(f"Test accuracy: {tree.score(X_test, y_test):.4f}")
\`\`\`

## Splitting Criteria

Decision trees recursively split data to create homogeneous subsets. The quality of a split is measured by:

### Gini Impurity (Classification)

\\[ Gini = 1 - \\sum_{i=1}^{C} p_i^2 \\]

Where \\( p_i \\) is the proportion of class i in the node.

**Interpretation:**
- Gini = 0: Pure node (all samples same class)
- Gini = 0.5: Maximum impurity (binary, 50-50 split)

### Entropy (Information Gain)

\\[ Entropy = -\\sum_{i=1}^{C} p_i \\log_2(p_i) \\]

**Information Gain** = Entropy(parent) - Weighted Entropy(children)

### Mean Squared Error (Regression)

\\[ MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\bar{y})^2 \\]

\`\`\`python
# Demonstrate splitting criteria
from sklearn.tree import DecisionTreeClassifier

# Generate sample data
X_sample = np.array([[1], [2], [3], [4], [5], [6]])
y_sample = np.array([0, 0, 0, 1, 1, 1])

# Compare Gini vs Entropy
tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=2)
tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=2)

tree_gini.fit(X_sample, y_sample)
tree_entropy.fit(X_sample, y_sample)

print("Splitting Criteria Comparison:")
print(f"Gini - Feature importances: {tree_gini.feature_importances_}")
print(f"Entropy - Feature importances: {tree_entropy.feature_importances_}")
print("\\nBoth usually give similar results!")
\`\`\`

## Overfitting and Tree Pruning

**Problem**: Deep trees perfectly fit training data but fail to generalize.

**Solutions**:

1. **Pre-pruning (Early Stopping)**:
   - max_depth: Limit tree depth
   - min_samples_split: Minimum samples to split node
   - min_samples_leaf: Minimum samples in leaf
   - max_leaf_nodes: Maximum number of leaves

2. **Post-pruning**:
   - Build full tree, then remove branches

\`\`\`python
from sklearn.datasets import make_classification

# Generate data with noise
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compare different tree depths
depths = [1, 3, 5, 10, 20, None]
results = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)
    n_leaves = tree.get_n_leaves()
    
    results.append({
        'depth': depth if depth else 'None',
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_leaves': n_leaves
    })
    
import pandas as pd
df_results = pd.DataFrame(results)
print("\\nTree Depth vs Performance:")
print(df_results.to_string(index=False))

# Plot bias-variance tradeoff
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_pos = range(len(depths))
plt.plot(x_pos, df_results['train_acc'], 'o-', label='Training', linewidth=2)
plt.plot(x_pos, df_results['test_acc'], 's-', label='Test', linewidth=2)
plt.xlabel('Tree Complexity →')
plt.ylabel('Accuracy')
plt.title('Overfitting: Deep Trees')
plt.legend()
plt.xticks(x_pos, [str(d) for d in depths])
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x_pos, df_results['n_leaves'], 'o-', linewidth=2, color='green')
plt.xlabel('max_depth')
plt.ylabel('Number of Leaves')
plt.title('Tree Complexity Growth')
plt.xticks(x_pos, [str(d) for d in depths])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Feature Importance

Decision trees naturally provide feature importance scores based on how much each feature reduces impurity.

\`\`\`python
from sklearn.datasets import load_breast_cancer

# Load dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Train tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
\`\`\`

## Decision Boundaries

\`\`\`python
# Visualize decision boundaries
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=200, noise=0.25, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

max_depths = [2, 5, None]

for idx, max_depth in enumerate(max_depths):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_moons, y_moons)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    axes[idx].scatter(X_moons[y_moons==0, 0], X_moons[y_moons==0, 1], 
                     c='blue', s=30, edgecolors='k')
    axes[idx].scatter(X_moons[y_moons==1, 0], X_moons[y_moons==1, 1], 
                     c='red', s=30, edgecolors='k')
    
    depth_str = max_depth if max_depth else 'Unlimited'
    axes[idx].set_title(f'max_depth={depth_str}\\nLeaves: {tree.get_n_leaves()}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Decision trees create axis-aligned (rectangular) decision boundaries!")
\`\`\`

## Regression Trees

\`\`\`python
from sklearn.tree import DecisionTreeRegressor

# Generate regression data
np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(100, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(100) * 0.1

# Train trees with different depths
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

depths = [2, 5, 20]

for idx, depth in enumerate(depths):
    tree_reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_reg.fit(X_reg, y_reg)
    
    # Predict
    X_test_reg = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_pred_reg = tree_reg.predict(X_test_reg)
    
    # Plot
    axes[idx].scatter(X_reg, y_reg, s=20, edgecolor="black", c="blue", label="Data")
    axes[idx].plot(X_test_reg, y_pred_reg, color="red", linewidth=2, label="Prediction")
    axes[idx].set_xlabel("X")
    axes[idx].set_ylabel("y")
    axes[idx].set_title(f"max_depth={depth}")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Regression trees create piecewise constant predictions!")
\`\`\`

## Advantages and Limitations

**Advantages:**
- Highly interpretable
- Visualizable
- Handles non-linear relationships
- No feature scaling needed
- Handles mixed data types
- Automatic feature selection
- Fast prediction

**Limitations:**
- Prone to overfitting
- Unstable (small data changes → different tree)
- Biased toward features with more levels
- Not optimal for linear relationships
- Can create overly complex trees

## Real-World Example

\`\`\`python
# Customer churn prediction
from sklearn.metrics import classification_report, confusion_matrix

# Generate customer data
np.random.seed(42)
n = 1000

customer_data = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'tenure_months': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(20, 150, n),
    'num_calls': np.random.poisson(3, n),
    'has_contract': np.random.binomial(1, 0.3, n)
})

# Generate churn (influenced by features)
churn_prob = 0.2 + 0.005 * customer_data['monthly_charges'] - 0.01 * customer_data['tenure_months']
customer_data['churned'] = (np.random.random(n) < np.clip(churn_prob, 0, 1)).astype(int)

X = customer_data.drop('churned', axis=1)
y = customer_data['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train optimized tree
tree = DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=20, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print("="*60)
print("CUSTOMER CHURN PREDICTION")
print("="*60)
print(f"\\nTest Accuracy: {tree.score(X_test, y_test):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
print("\\nFeature Importance:")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.to_string(index=False))
\`\`\`

## Summary

Decision trees make predictions using learned decision rules:
- Intuitive, interpretable structure
- Handles non-linear relationships
- Provides feature importance
- Prone to overfitting (use pruning)
- Foundation for ensemble methods

**Key Insight**: Trees partition feature space into rectangles!

Next: Random Forests combine multiple trees for better performance!
`,
  codeExample: `# Complete Decision Tree Implementation
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Test accuracy: {grid.best_estimator_.score(X_test, y_test):.4f}")
`,
};
