/**
 * Logistic Regression Section
 */

export const logisticregressionSection = {
  id: 'logistic-regression',
  title: 'Logistic Regression',
  content: `# Logistic Regression

## Introduction

Despite its name, **Logistic Regression is a classification algorithm**, not a regression algorithm. It predicts the probability that an instance belongs to a particular class, making it one of the most widely-used algorithms for binary and multiclass classification.

**Why Logistic Regression Matters:**
- Foundation for understanding neural networks (similar structure)
- Probabilistic predictions (not just class labels)
- Interpretable coefficients
- Fast training and prediction
- Works well as a baseline
- Extends naturally to multiclass problems

**Real-World Applications:**
- Medical diagnosis (disease present or not)
- Credit default prediction (will default or not)
- Email spam detection
- Customer churn prediction
- Click-through rate prediction
- Fraud detection

## From Linear to Logistic Regression

### The Problem with Linear Regression for Classification

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification

# Generate binary classification data
np.random.seed(42)
X, y = make_classification (n_samples=100, n_features=1, n_informative=1, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Sort for visualization
sort_idx = X.ravel().argsort()
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

# Try linear regression (wrong approach)
linear_model = LinearRegression()
linear_model.fit(X_sorted, y_sorted)
y_pred_linear = linear_model.predict(X_sorted)

# Proper logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(X_sorted, y_sorted)
y_pred_prob = logistic_model.predict_proba(X_sorted)[:, 1]

# Visualize
plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_sorted[y_sorted==0], y_sorted[y_sorted==0], label='Class 0', alpha=0.7)
plt.scatter(X_sorted[y_sorted==1], y_sorted[y_sorted==1], label='Class 1', alpha=0.7)
plt.plot(X_sorted, y_pred_linear, 'r-', linewidth=2, label='Linear Regression')
plt.axhline (y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Feature')
plt.ylabel('Target / Prediction')
plt.title('Linear Regression for Classification (WRONG)')
plt.legend()
plt.ylim(-0.5, 1.5)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_sorted[y_sorted==0], y_sorted[y_sorted==0], label='Class 0', alpha=0.7)
plt.scatter(X_sorted[y_sorted==1], y_sorted[y_sorted==1], label='Class 1', alpha=0.7)
plt.plot(X_sorted, y_pred_prob, 'g-', linewidth=2, label='Logistic Regression')
plt.axhline (y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision boundary')
plt.xlabel('Feature')
plt.ylabel('Probability of Class 1')
plt.title('Logistic Regression (CORRECT)')
plt.legend()
plt.ylim(-0.1, 1.1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Issues with Linear Regression:")
print("• Predictions can be < 0 or > 1 (not valid probabilities)")
print("• No S-curve shape to model probability transitions")
print("• Sensitive to outliers in feature space")
print("\\nLogistic Regression:")
print("• Outputs always in [0, 1] (valid probabilities)")
print("• S-shaped curve models smooth transitions")
print("• More robust to outliers")
\`\`\`

## The Sigmoid Function

The key to logistic regression is the **sigmoid (logistic) function**:

\\[ \\sigma (z) = \\frac{1}{1 + e^{-z}} \\]

**Properties:**
- Input: any real number (-∞ to +∞)
- Output: probability between 0 and 1
- S-shaped curve
- \\( \\sigma(0) = 0.5 \\) (decision boundary)
- \\( \\sigma (z) \\to 1 \\) as \\( z \\to \\infty \\)
- \\( \\sigma (z) \\to 0 \\) as \\( z \\to -\\infty \\)

\`\`\`python
# Visualize sigmoid function
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (z, sigmoid, linewidth=3, color='blue')
plt.axhline (y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision threshold')
plt.axvline (x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('z = wᵀx + b')
plt.ylabel('σ(z) = Probability')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.legend()

# Derivative of sigmoid
sigmoid_derivative = sigmoid * (1 - sigmoid)

plt.subplot(1, 2, 2)
plt.plot (z, sigmoid_derivative, linewidth=3, color='green')
plt.xlabel('z')
plt.ylabel("σ'(z)")
plt.title("Sigmoid Derivative (for gradient descent)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Sigmoid properties:")
print(f"σ(-∞) ≈ {1/(1+np.exp(10)):.6f}")
print(f"σ(0)   = {1/(1+np.exp(0)):.6f}")
print(f"σ(+∞) ≈ {1/(1+np.exp(-10)):.6f}")
print(f"\\nDerivative: σ'(z) = σ(z)(1 - σ(z))")
\`\`\`

## Binary Logistic Regression Model

### The Model

For binary classification (two classes: 0 and 1):

\\[ P(y=1|\\mathbf{x}) = \\sigma(\\mathbf{w}^T\\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T\\mathbf{x} + b)}} \\]

Where:
- \\( \\mathbf{x} \\): Feature vector
- \\( \\mathbf{w} \\): Weight vector (coefficients)
- \\( b \\): Bias term (intercept)
- \\( P(y=1|\\mathbf{x}) \\): Probability that instance belongs to class 1

**Decision Rule:**
- If \\( P(y=1|\\mathbf{x}) \\geq 0.5 \\), predict class 1
- If \\( P(y=1|\\mathbf{x}) < 0.5 \\), predict class 0

### Log-Odds (Logit)

The model can be rewritten as:

\\[ \\log\\left(\\frac{P(y=1|\\mathbf{x})}{1 - P(y=1|\\mathbf{x})}\\right) = \\mathbf{w}^T\\mathbf{x} + b \\]

The left side is the **log-odds** (logit), which is linear in the features!

## Loss Function: Binary Cross-Entropy

Unlike linear regression (MSE), logistic regression uses **binary cross-entropy loss**:

\\[ L(\\mathbf{w}, b) = -\\frac{1}{n}\\sum_{i=1}^{n}\\left[y_i \\log(\\hat{p}_i) + (1-y_i)\\log(1-\\hat{p}_i)\\right] \\]

Where \\( \\hat{p}_i = P(y=1|\\mathbf{x}_i) \\) is the predicted probability.

**Intuition:**
- If actual class is 1 (\\(y_i=1\\)): Loss is \\(-\\log(\\hat{p}_i)\\)
  - If we predict \\(\\hat{p}_i \\approx 1\\): Loss ≈ 0 (good!)
  - If we predict \\(\\hat{p}_i \\approx 0\\): Loss → ∞ (very bad!)
- If actual class is 0 (\\(y_i=0\\)): Loss is \\(-\\log(1-\\hat{p}_i)\\)
  - If we predict \\(\\hat{p}_i \\approx 0\\): Loss ≈ 0 (good!)
  - If we predict \\(\\hat{p}_i \\approx 1\\): Loss → ∞ (very bad!)

\`\`\`python
# Visualize cross-entropy loss
p_pred = np.linspace(0.001, 0.999, 200)

# Loss when true class is 1
loss_y1 = -np.log (p_pred)

# Loss when true class is 0
loss_y0 = -np.log(1 - p_pred)

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (p_pred, loss_y1, linewidth=3, label='True class = 1')
plt.xlabel('Predicted Probability P(y=1)')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss (True Class = 1)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot (p_pred, loss_y0, linewidth=3, color='orange', label='True class = 0')
plt.xlabel('Predicted Probability P(y=1)')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss (True Class = 0)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
\`\`\`

## Training: Gradient Descent

We minimize the loss using gradient descent. The gradient is:

\\[ \\frac{\\partial L}{\\partial \\mathbf{w}} = \\frac{1}{n}\\mathbf{X}^T(\\hat{\\mathbf{p}} - \\mathbf{y}) \\]

Where \\( \\hat{\\mathbf{p}} \\) is the vector of predicted probabilities.

**Update rule:**
\\[ \\mathbf{w} := \\mathbf{w} - \\alpha \\frac{\\partial L}{\\partial \\mathbf{w}} \\]

\`\`\`python
def sigmoid (z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w, b):
    """Binary cross-entropy loss"""
    n = len (y)
    z = X @ w + b
    p = sigmoid (z)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-7
    p = np.clip (p, epsilon, 1 - epsilon)
    loss = -np.mean (y * np.log (p) + (1 - y) * np.log(1 - p))
    return loss

def logistic_regression_gd(X, y, learning_rate=0.01, iterations=1000):
    """Train logistic regression using gradient descent"""
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros (n_features)
    b = 0
    
    losses = []
    
    for i in range (iterations):
        # Forward pass
        z = X @ w + b
        p = sigmoid (z)
        
        # Compute loss
        loss = compute_loss(X, y, w, b)
        losses.append (loss)
        
        # Compute gradients
        dw = (1/n_samples) * X.T @ (p - y)
        db = (1/n_samples) * np.sum (p - y)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
    
    return w, b, losses

# Example: Train from scratch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification (n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
w, b, losses = logistic_regression_gd(X_train_scaled, y_train, learning_rate=0.1, iterations=1000)

# Plot convergence
plt.figure (figsize=(10, 5))
plt.plot (losses, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training Loss Convergence')
plt.grid(True, alpha=0.3)
plt.show()

# Make predictions
def predict(X, w, b, threshold=0.5):
    probabilities = sigmoid(X @ w + b)
    predictions = (probabilities >= threshold).astype (int)
    return predictions, probabilities

y_pred_train, y_prob_train = predict(X_train_scaled, w, b)
y_pred_test, y_prob_test = predict(X_test_scaled, w, b)

# Accuracy
train_accuracy = np.mean (y_pred_train == y_train)
test_accuracy = np.mean (y_pred_test == y_test)

print(f"\\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
\`\`\`

## Using Scikit-Learn

\`\`\`python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Train model
model = LogisticRegression (random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Probabilities
y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

print("="*60)
print("LOGISTIC REGRESSION RESULTS")
print("="*60)

print(f"\\nTraining Accuracy: {accuracy_score (y_train, y_pred_train):.4f}")
print(f"Test Accuracy: {accuracy_score (y_test, y_pred_test):.4f}")

print(f"\\nTest AUC-ROC: {roc_auc_score (y_test, y_prob_test):.4f}")

print("\\nClassification Report (Test Set):")
print(classification_report (y_test, y_pred_test, target_names=['Class 0', 'Class 1']))

print("\\nConfusion Matrix (Test Set):")
cm = confusion_matrix (y_test, y_pred_test)
print(cm)

# Visualize decision boundary
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid (np.arange (x_min, x_max, h),
                         np.arange (y_min, y_max, h))
    
    Z = model.predict (np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape (xx.shape)
    
    plt.contourf (xx, yy, Z, alpha=0.3, cmap=ListedColormap(['blue', 'red']))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', edgecolors='k')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title (title)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.figure (figsize=(10, 6))
plot_decision_boundary(X_test_scaled, y_test, model, "Logistic Regression Decision Boundary")
plt.show()
\`\`\`

## Multiclass Logistic Regression

### One-vs-Rest (OvR)

Train K binary classifiers (one for each class):
- Classifier 1: class 1 vs. all others
- Classifier 2: class 2 vs. all others
- ...
- Classifier K: class K vs. all others

Prediction: Choose class with highest probability.

### Softmax (Multinomial) Logistic Regression

Generalizes sigmoid to K classes:

\\[ P(y=k|\\mathbf{x}) = \\frac{e^{\\mathbf{w}_k^T\\mathbf{x}+b_k}}{\\sum_{j=1}^{K}e^{\\mathbf{w}_j^T\\mathbf{x}+b_j}} \\]

**Cross-Entropy Loss (multiclass):**
\\[ L = -\\frac{1}{n}\\sum_{i=1}^{n}\\sum_{k=1}^{K} y_{ik} \\log(\\hat{p}_{ik}) \\]

\`\`\`python
from sklearn.datasets import make_blobs

# Generate 3-class dataset
X_multi, y_multi = make_blobs (n_samples=300, centers=3, n_features=2,
                              cluster_std=1.5, random_state=42)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42
)

# Softmax logistic regression
model_multi = LogisticRegression (multi_class='multinomial', random_state=42)
model_multi.fit(X_train_m, y_train_m)

y_pred_multi = model_multi.predict(X_test_m)
y_prob_multi = model_multi.predict_proba(X_test_m)

print("\\nMulticlass Logistic Regression:")
print(f"Test Accuracy: {accuracy_score (y_test_m, y_pred_multi):.4f}")
print("\\nClassification Report:")
print(classification_report (y_test_m, y_pred_multi))

# Visualize
plt.figure (figsize=(14, 6))

plt.subplot(1, 2, 1)
colors = ['blue', 'red', 'green']
for i in range(3):
    mask = y_test_m == i
    plt.scatter(X_test_m[mask, 0], X_test_m[mask, 1], 
                c=colors[i], label=f'Class {i}', s=100, alpha=0.7, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('True Classes')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Decision regions
h = 0.02
x_min, x_max = X_multi[:, 0].min() - 1, X_multi[:, 0].max() + 1
y_min, y_max = X_multi[:, 1].min() - 1, X_multi[:, 1].max() + 1
xx, yy = np.meshgrid (np.arange (x_min, x_max, h), np.arange (y_min, y_max, h))

Z = model_multi.predict (np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape (xx.shape)

plt.contourf (xx, yy, Z, alpha=0.3, levels=2, colors=['blue', 'red', 'green'])

for i in range(3):
    mask = y_test_m == i
    plt.scatter(X_test_m[mask, 0], X_test_m[mask, 1],
                c=colors[i], label=f'Class {i}', s=100, alpha=0.7, edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Regularization in Logistic Regression

Similar to linear regression, we add L1 or L2 penalties:

**Ridge (L2):**
\\[ L = -\\frac{1}{n}\\sum_{i=1}^{n}\\left[y_i \\log(\\hat{p}_i) + (1-y_i)\\log(1-\\hat{p}_i)\\right] + \\lambda\\sum_{j=1}^{p}w_j^2 \\]

**Lasso (L1):**
\\[ L = -\\frac{1}{n}\\sum_{i=1}^{n}\\left[y_i \\log(\\hat{p}_i) + (1-y_i)\\log(1-\\hat{p}_i)\\right] + \\lambda\\sum_{j=1}^{p}|w_j| \\]

In sklearn, use parameter \`C\` (inverse of regularization strength):
- Small C → Strong regularization
- Large C → Weak regularization

\`\`\`python
# Compare different regularization strengths
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

plt.figure (figsize=(15, 10))

for idx, C in enumerate(C_values, 1):
    model = LogisticRegression(C=C, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score (y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score (y_test, model.predict(X_test_scaled))
    
    plt.subplot(2, 3, idx)
    plot_decision_boundary(X_test_scaled, y_test, model,
                          f"C={C}\\nTrain:{train_acc:.3f} Test:{test_acc:.3f}")
    
    print(f"C={C}: Train={train_acc:.4f}, Test={test_acc:.4f}")

plt.tight_layout()
plt.show()
\`\`\`

## Interpreting Coefficients

Coefficients represent the change in **log-odds** for a unit change in the feature:

\\[ \\Delta \\log\\left(\\frac{P(y=1)}{P(y=0)}\\right) = w_j \\Delta x_j \\]

**Odds Ratio:** \\( e^{w_j} \\)
- \\( e^{w_j} > 1 \\): Feature increases probability of class 1
- \\( e^{w_j} < 1 \\): Feature decreases probability of class 1
- \\( e^{w_j} = 1 \\): Feature has no effect

\`\`\`python
# Example: Credit default prediction
from sklearn.datasets import make_classification

X_credit, y_credit = make_classification(
    n_samples=1000, n_features=5, n_informative=5, n_redundant=0,
    random_state=42
)

feature_names = ['Income', 'Debt', 'Credit_Score', 'Age', 'Existing_Loans']

# Fit model
model_credit = LogisticRegression (random_state=42)
model_credit.fit(X_credit, y_credit)

# Interpret coefficients
coefficients = model_credit.coef_[0]
odds_ratios = np.exp (coefficients)

print("="*60)
print("COEFFICIENT INTERPRETATION")
print("="*60)
print(f"\\n{'Feature':<20} {'Coefficient':<15} {'Odds Ratio':<15} {'Interpretation'}")
print("-"*80)

for feature, coef, odds in zip (feature_names, coefficients, odds_ratios):
    if odds > 1:
        interp = f"↑ by {(odds-1)*100:.1f}%"
    else:
        interp = f"↓ by {(1-odds)*100:.1f}%"
    print(f"{feature:<20} {coef:<15.4f} {odds:<15.4f} {interp}")

print("\\n* Odds ratio represents multiplicative change in odds for 1-unit increase in feature")
\`\`\`

## Real-World Example: Customer Churn Prediction

\`\`\`python
# Generate realistic customer churn data
np.random.seed(42)
n_customers = 1000

data = {
    'monthly_charges': np.random.uniform(20, 150, n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'support_calls': np.random.poisson(2, n_customers),
    'contract_monthly': np.random.binomial(1, 0.5, n_customers),
    'paperless_billing': np.random.binomial(1, 0.6, n_customers),
}

# Churn probability based on features
import pandas as pd
df = pd.DataFrame (data)

# Higher charges, shorter tenure, more support calls → higher churn
churn_prob = (
    0.05 +
    0.004 * df['monthly_charges'] +
    -0.01 * df['tenure_months'] +
    0.05 * df['support_calls'] +
    0.1 * df['contract_monthly'] +
    0.05 * df['paperless_billing']
)
churn_prob = 1 / (1 + np.exp(-churn_prob + 3))  # Apply sigmoid
df['churned'] = (np.random.random (n_customers) < churn_prob).astype (int)

print("="*60)
print("CUSTOMER CHURN PREDICTION")
print("="*60)
print(f"\\nDataset: {len (df)} customers")
print(f"Churn rate: {df['churned'].mean():.1%}")

# Split data
X = df.drop('churned', axis=1)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression (random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"\\nTest Accuracy: {accuracy_score (y_test, y_pred):.4f}")
print(f"Test AUC-ROC: {roc_auc_score (y_test, y_prob):.4f}")

print("\\nClassification Report:")
print(classification_report (y_test, y_pred, target_names=['Retained', 'Churned']))

# Feature importance
print("\\nFeature Importance (Coefficients):")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feature:<20}: {coef:>8.4f}")
\`\`\`

## Advantages and Limitations

**Advantages:**
- Probabilistic predictions
- Fast training and prediction
- Interpretable coefficients
- Works well with linearly separable classes
- No hyperparameters (except regularization)
- Naturally extends to multiclass

**Limitations:**
- Assumes linear decision boundary
- Can underfit with non-linear relationships
- Sensitive to outliers
- Requires feature scaling
- Assumes independence of features
- Poor with highly correlated features

## Summary

Logistic Regression is a fundamental classification algorithm:
- Uses sigmoid function to output probabilities
- Trained with binary cross-entropy loss
- Linear decision boundaries in feature space
- Coefficients interpretable as log-odds changes
- Extends to multiclass via softmax
- Regularization prevents overfitting

**Key Insight:** Despite the name, it's classification, not regression!

Next: k-Nearest Neighbors for non-parametric classification!
`,
  codeExample: `# Complete Logistic Regression Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

# Generate synthetic dataset: Email spam detection
np.random.seed(42)
n_emails = 2000

# Features
email_length = np.random.normal(1000, 500, n_emails)
num_links = np.random.poisson(3, n_emails)
num_capitals = np.random.exponential(50, n_emails)
has_urgency_words = np.random.binomial(1, 0.3, n_emails)
has_money_words = np.random.binomial(1, 0.25, n_emails)

# Spam probability (logistic relationship)
z = (
    -3 +
    -0.001 * email_length +
    0.3 * num_links +
    0.02 * num_capitals +
    1.5 * has_urgency_words +
    1.2 * has_money_words
)
spam_prob = 1 / (1 + np.exp(-z))
is_spam = (np.random.random (n_emails) < spam_prob).astype (int)

# Create DataFrame
df = pd.DataFrame({
    'email_length': email_length,
    'num_links': num_links,
    'num_capitals': num_capitals,
    'has_urgency_words': has_urgency_words,
    'has_money_words': has_money_words,
    'is_spam': is_spam
})

print("="*70)
print("LOGISTIC REGRESSION: EMAIL SPAM DETECTION")
print("="*70)
print(f"\\nDataset: {len (df)} emails")
print(f"Spam rate: {df['is_spam'].mean():.1%}")
print(f"\\nFeatures: {list (df.columns[:-1])}")

# Split features and target
X = df.drop('is_spam', axis=1)
y = df['is_spam']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features (important for logistic regression!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

# Train logistic regression
model = LogisticRegression (random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Get probability estimates
y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

print(f"\\nTraining Set:")
print(f"  Accuracy:  {accuracy_score (y_train, y_train_pred):.4f}")
print(f"  Precision: {precision_score (y_train, y_train_pred):.4f}")
print(f"  Recall:    {recall_score (y_train, y_train_pred):.4f}")
print(f"  F1-Score:  {f1_score (y_train, y_train_pred):.4f}")
print(f"  AUC-ROC:   {roc_auc_score (y_train, y_train_prob):.4f}")

print(f"\\nTest Set:")
print(f"  Accuracy:  {accuracy_score (y_test, y_test_pred):.4f}")
print(f"  Precision: {precision_score (y_test, y_test_pred):.4f}")
print(f"  Recall:    {recall_score (y_test, y_test_pred):.4f}")
print(f"  F1-Score:  {f1_score (y_test, y_test_pred):.4f}")
print(f"  AUC-ROC:   {roc_auc_score (y_test, y_test_prob):.4f}")

# Confusion matrix
cm = confusion_matrix (y_test, y_test_pred)
print(f"\\nConfusion Matrix (Test Set):")
print(f"              Predicted")
print(f"              Not Spam  Spam")
print(f"Actual Not Spam   {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       Spam       {cm[1,0]:4d}    {cm[1,1]:4d}")

tn, fp, fn, tp = cm.ravel()
print(f"\\nTrue Negatives:  {tn} (correctly identified as not spam)")
print(f"False Positives: {fp} (spam but marked as not spam)")
print(f"False Negatives: {fn} (not spam but marked as spam)")
print(f"True Positives:  {tp} (correctly identified as spam)")

# Classification report
print("\\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report (y_test, y_test_pred, 
                          target_names=['Not Spam', 'Spam']))

# Feature importance
print("\\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

coefficients = model.coef_[0]
odds_ratios = np.exp (coefficients)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios,
    'Abs_Coefficient': np.abs (coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\\n{'Feature':<25} {'Coefficient':<12} {'Odds Ratio':<12} {'Effect'}")
print("-"*70)
for _, row in feature_importance.iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    odds = row['Odds_Ratio']
    
    if odds > 1.1:
        effect = f"↑ Increases spam prob by {(odds-1)*100:.1f}%"
    elif odds < 0.9:
        effect = f"↓ Decreases spam prob by {(1-odds)*100:.1f}%"
    else:
        effect = "≈ Minimal effect"
    
    print(f"{feature:<25} {coef:>11.4f} {odds:>11.4f} {effect}")

print("\\nIntercept (bias):", model.intercept_[0])

# Visualizations
fig = plt.figure (figsize=(16, 12))

# 1. ROC Curve
ax1 = plt.subplot(2, 3, 1)
fpr, tpr, thresholds = roc_curve (y_test, y_test_prob)
auc = roc_auc_score (y_test, y_test_prob)

plt.plot (fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
ax2 = plt.subplot(2, 3, 2)
precision, recall, _ = precision_recall_curve (y_test, y_test_prob)
plt.plot (recall, precision, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)

# 3. Probability Distribution
ax3 = plt.subplot(2, 3, 3)
plt.hist (y_test_prob[y_test==0], bins=30, alpha=0.5, label='Not Spam', color='blue')
plt.hist (y_test_prob[y_test==1], bins=30, alpha=0.5, label='Spam', color='red')
plt.axvline (x=0.5, color='black', linestyle='--', label='Decision threshold')
plt.xlabel('Predicted Probability of Spam')
plt.ylabel('Frequency')
plt.title('Probability Distribution by True Class')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Feature Coefficients
ax4 = plt.subplot(2, 3, 4)
colors = ['red' if x > 0 else 'blue' for x in coefficients]
plt.barh(X.columns, coefficients, color=colors, alpha=0.7)
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients\\n(Red = increases spam prob)')
plt.axvline (x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

# 5. Confusion Matrix Heatmap
ax5 = plt.subplot(2, 3, 5)
import seaborn as sns
sns.heatmap (cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks([0.5, 1.5], ['Not Spam', 'Spam'])
plt.yticks([0.5, 1.5], ['Not Spam', 'Spam'])

# 6. Calibration Plot
ax6 = plt.subplot(2, 3, 6)
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve (y_test, y_test_prob, n_bins=10)
plt.plot (prob_pred, prob_true, marker='o', linewidth=2, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Example predictions
print("\\n" + "="*70)
print("EXAMPLE PREDICTIONS")
print("="*70)

# Select random samples
sample_indices = np.random.choice (len(X_test), 5, replace=False)

print(f"\\n{'Actual':<10} {'Predicted':<10} {'Probability':<12} {'Features'}")
print("-"*70)

for idx in sample_indices:
    actual = "Spam" if y_test.iloc[idx] == 1 else "Not Spam"
    predicted = "Spam" if y_test_pred[idx] == 1 else "Not Spam"
    prob = y_test_prob[idx]
    
    features_str = ", ".join([f"{col}={X_test.iloc[idx][col]:.1f}" 
                             for col in X.columns[:3]])
    
    print(f"{actual:<10} {predicted:<10} {prob:<12.3f} {features_str}...")

# Summary
print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
The logistic regression model achieves {accuracy_score (y_test, y_test_pred):.1%} accuracy
on the test set with an AUC-ROC of {roc_auc_score (y_test, y_test_prob):.3f}.

Top factors indicating spam:
1. {feature_importance.iloc[0]['Feature']}: odds ratio = {feature_importance.iloc[0]['Odds_Ratio']:.2f}
2. {feature_importance.iloc[1]['Feature']}: odds ratio = {feature_importance.iloc[1]['Odds_Ratio']:.2f}
3. {feature_importance.iloc[2]['Feature']}: odds ratio = {feature_importance.iloc[2]['Odds_Ratio']:.2f}

The model correctly identifies {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%) of actual spam emails
while maintaining a false positive rate of {fp/(fp+tn)*100:.1f}%.

Key insight: Logistic regression provides interpretable, probabilistic predictions
suitable for spam filtering where understanding why an email is classified matters.
""")
`,
};
