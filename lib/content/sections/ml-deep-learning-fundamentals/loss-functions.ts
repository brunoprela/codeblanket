/**
 * Section: Loss Functions
 * Module: Deep Learning Fundamentals
 *
 * Covers MSE, cross-entropy, loss function design, derivatives, and choosing
 * appropriate loss functions for different tasks
 */

export const lossFunctionsSection = {
  id: 'loss-functions',
  title: 'Loss Functions',
  content: `
# Loss Functions

## Introduction

A **loss function** (or cost function) quantifies how well a neural network's predictions match the actual targets. It\'s the objective function we minimize during training. The choice of loss function profoundly affects what the network learns.

**What You'll Learn:**
- Mean Squared Error (MSE) for regression
- Binary and categorical cross-entropy for classification
- Mathematical properties and derivatives
- Custom loss functions
- Choosing the right loss for your problem
- Loss functions in trading applications

## What is a Loss Function?

### Mathematical Definition

Given:
- True labels: \\(y\\) (ground truth)
- Predictions: \\(\\hat{y}\\) (model output)

A loss function \\(L(y, \\hat{y})\\) measures the discrepancy:
- \\(L = 0\\) when \\(y = \\hat{y}\\) (perfect prediction)
- \\(L > 0\\) when predictions differ from truth
- Larger error → larger loss

### Training Objective

\`\`\`
Goal: Find parameters θ that minimize average loss

θ* = argmin_θ (1/N) Σᵢ L(yᵢ, ŷᵢ)

Where N is the number of training samples
\`\`\`

## Mean Squared Error (MSE) - Regression

### Definition

MSE measures the average squared difference between predictions and targets:

\`\`\`
MSE = (1/N) Σᵢ (yᵢ - ŷᵢ)²
\`\`\`

**Properties:**
- Always non-negative (≥ 0)
- Heavily penalizes large errors (quadratic)
- Differentiable everywhere
- Units: (target units)²

### Implementation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def mse_loss (y_true, y_pred):
    """
    Mean Squared Error loss
    
    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)
    
    Returns:
        loss: Scalar MSE value
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient (y_true, y_pred):
    """
    Gradient of MSE w.r.t. predictions
    
    Returns:
        gradient: ∂MSE/∂ŷ = (2/N)(ŷ - y)
    """
    N = len (y_true)
    return (2 / N) * (y_pred - y_true)


# Example: Regression on synthetic data
np.random.seed(42)

# True function: y = 2x + 1 + noise
x = np.linspace(0, 10, 100)
y_true = 2 * x + 1 + np.random.randn(100) * 2

# Model predictions (imperfect fit)
y_pred = 1.8 * x + 1.5

# Calculate loss
loss = mse_loss (y_true, y_pred)
gradient = mse_gradient (y_true, y_pred)

print("Mean Squared Error Example:")
print(f"  MSE Loss: {loss:.4f}")
print(f"  RMSE: {np.sqrt (loss):.4f}")
print(f"  Average gradient magnitude: {np.mean (np.abs (gradient)):.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Predictions vs True
axes[0].scatter (x, y_true, alpha=0.5, label='True values')
axes[0].plot (x, y_pred, 'r-', linewidth=2, label='Predictions')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Predictions vs Ground Truth')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Errors
errors = y_pred - y_true
axes[1].scatter (x, errors, alpha=0.5, color='red')
axes[1].axhline (y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Error (ŷ - y)')
axes[1].set_title('Prediction Errors')
axes[1].grid(True, alpha=0.3)

# Plot 3: Squared errors
squared_errors = errors ** 2
axes[2].bar (range (len (squared_errors)), squared_errors, alpha=0.6, color='orange')
axes[2].axhline (y=loss, color='r', linestyle='--', linewidth=2, label=f'MSE = {loss:.2f}')
axes[2].set_xlabel('Sample')
axes[2].set_ylabel('Squared Error')
axes[2].set_title('Squared Errors (contribution to MSE)')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
\`\`\`

### Derivative of MSE

\`\`\`
∂MSE/∂ŷᵢ = ∂/∂ŷᵢ [(1/N) Σⱼ (yⱼ - ŷⱼ)²]
         = (2/N)(ŷᵢ - yᵢ)
\`\`\`

**Key properties:**
- Gradient proportional to error
- Linear gradient (not saturating)
- Larger errors → larger gradients

### Variants

**1. Mean Absolute Error (MAE):**
\`\`\`
MAE = (1/N) Σᵢ |yᵢ - ŷᵢ|
\`\`\`
- Less sensitive to outliers
- Gradient: sign(ŷ - y) (non-smooth at 0)

**2. Huber Loss (Robust):**
\`\`\`
Huber (y, ŷ) = {
  (1/2)(y - ŷ)²         if |y - ŷ| ≤ δ
  δ|y - ŷ| - (1/2)δ²    otherwise
}
\`\`\`
- Quadratic for small errors (like MSE)
- Linear for large errors (like MAE)
- Robust to outliers

\`\`\`python
def huber_loss (y_true, y_pred, delta=1.0):
    """Huber loss - robust to outliers"""
    error = y_true - y_pred
    is_small_error = np.abs (error) <= delta
    
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs (error) - 0.5 * delta ** 2
    
    return np.mean (np.where (is_small_error, squared_loss, linear_loss))

# Compare losses with outliers
y_true_outlier = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Last value is outlier
y_pred_outlier = np.array([1.1, 2.1, 3.1, 4.1, 4.5])

print("\\nComparison with Outlier:")
print(f"  MSE: {mse_loss (y_true_outlier, y_pred_outlier):.2f}")
print(f"  MAE: {np.mean (np.abs (y_true_outlier - y_pred_outlier)):.2f}")
print(f"  Huber: {huber_loss (y_true_outlier, y_pred_outlier, delta=1.0):.2f}")
print("  → Huber less affected by outlier than MSE")
\`\`\`

## Binary Cross-Entropy - Binary Classification

### Definition

For binary classification (y ∈ {0, 1}):

\`\`\`
BCE = -(1/N) Σᵢ [yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ)]
\`\`\`

Where ŷᵢ ∈ (0, 1) is the predicted probability of class 1.

**Interpretation:**
- Measures surprise: how unexpected is the true label given predicted probability
- Related to information theory (cross-entropy between distributions)
- Penalizes confident wrong predictions heavily

### Implementation

\`\`\`python
def binary_cross_entropy (y_true, y_pred, epsilon=1e-10):
    """
    Binary cross-entropy loss
    
    Args:
        y_true: True labels (0 or 1), shape (n_samples,)
        y_pred: Predicted probabilities, shape (n_samples,)
        epsilon: Small value to prevent log(0)
    
    Returns:
        loss: Scalar BCE value
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    
    # BCE formula
    loss = -np.mean(
        y_true * np.log (y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )
    return loss

def bce_gradient (y_true, y_pred, epsilon=1e-10):
    """
    Gradient of BCE w.r.t. predictions
    
    Returns:
        gradient: ∂BCE/∂ŷ = (ŷ - y) / [ŷ(1 - ŷ)]
    """
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


# Example: Binary classification
np.random.seed(42)

# True labels
y_true_binary = np.array([1, 1, 0, 1, 0, 0, 1, 0])

# Different prediction scenarios
scenarios = {
    'Perfect': np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
    'Good': np.array([0.9, 0.85, 0.1, 0.95, 0.15, 0.2, 0.8, 0.1]),
    'Fair': np.array([0.7, 0.6, 0.3, 0.8, 0.4, 0.35, 0.65, 0.25]),
    'Poor': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    'Terrible': np.array([0.1, 0.2, 0.9, 0.15, 0.95, 0.85, 0.1, 0.9]),
}

print("\\nBinary Cross-Entropy Comparison:")
print("-" * 60)
for name, y_pred in scenarios.items():
    loss = binary_cross_entropy (y_true_binary, y_pred)
    accuracy = np.mean((y_pred > 0.5) == y_true_binary)
    print(f"{name:12s}: BCE = {loss:.4f}, Accuracy = {accuracy:.1%}")

# Visualize BCE surface
y_true_viz = 1  # True label = 1
y_pred_range = np.linspace(0.01, 0.99, 100)
losses_y1 = -np.log (y_pred_range)  # When y=1: -log(ŷ)
losses_y0 = -np.log(1 - y_pred_range)  # When y=0: -log(1-ŷ)

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (y_pred_range, losses_y1, linewidth=2, label='True label = 1')
plt.plot (y_pred_range, losses_y0, linewidth=2, label='True label = 0')
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline (x=0.5, color='k', linestyle='--', alpha=0.3, label='Decision boundary')

plt.subplot(1, 2, 2)
# Show gradient
gradient_y1 = 1 / y_pred_range  # When y=1
gradient_y0 = -1 / (1 - y_pred_range)  # When y=0
plt.plot (y_pred_range, gradient_y1, linewidth=2, label='True label = 1')
plt.plot (y_pred_range, gradient_y0, linewidth=2, label='True label = 0')
plt.xlabel('Predicted Probability')
plt.ylabel('Gradient')
plt.title('BCE Gradient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-20, 20)

plt.tight_layout()
plt.show()
\`\`\`

### Why Cross-Entropy for Classification?

**Probabilistic Interpretation:**
- Neural network outputs probability distribution
- Cross-entropy measures distance between distributions
- Optimal under maximum likelihood principle

**Gradient Properties:**
- Gradient: ∂BCE/∂ŷ = (ŷ - y) / [ŷ(1 - ŷ)]
- With sigmoid output: Clean gradient ∂BCE/∂z = ŷ - y
- No vanishing gradient problem (unlike MSE for classification)

## Categorical Cross-Entropy - Multi-class Classification

### Definition

For multi-class classification with C classes:

\`\`\`
CCE = -(1/N) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)
\`\`\`

Where:
- y is one-hot encoded: yᵢⱼ = 1 if sample i belongs to class j, else 0
- ŷᵢⱼ is predicted probability of class j for sample i
- Σⱼ ŷᵢⱼ = 1 (output from softmax)

### Implementation

\`\`\`python
def categorical_cross_entropy (y_true_one_hot, y_pred, epsilon=1e-10):
    """
    Categorical cross-entropy loss
    
    Args:
        y_true_one_hot: True labels (one-hot), shape (n_samples, n_classes)
        y_pred: Predicted probabilities, shape (n_samples, n_classes)
        epsilon: Small value to prevent log(0)
    
    Returns:
        loss: Scalar CCE value
    """
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    return -np.mean (np.sum (y_true_one_hot * np.log (y_pred), axis=1))

def sparse_categorical_cross_entropy (y_true_labels, y_pred, epsilon=1e-10):
    """
    Sparse categorical cross-entropy (y_true as class indices)
    
    Args:
        y_true_labels: True class indices, shape (n_samples,)
        y_pred: Predicted probabilities, shape (n_samples, n_classes)
    
    Returns:
        loss: Scalar CCE value
    """
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    n_samples = len (y_true_labels)
    
    # Extract predicted probability for true class
    log_probs = np.log (y_pred[range (n_samples), y_true_labels])
    return -np.mean (log_probs)


# Example: Multi-class classification (3 classes)
np.random.seed(42)

# True labels (5 samples)
y_true_labels = np.array([0, 1, 2, 0, 1])  # Class indices
y_true_one_hot = np.eye(3)[y_true_labels]   # One-hot encoding

# Predicted probabilities (softmax output)
y_pred_multi = np.array([
    [0.7, 0.2, 0.1],  # True: 0, Pred: 0 ✓
    [0.1, 0.8, 0.1],  # True: 1, Pred: 1 ✓
    [0.2, 0.3, 0.5],  # True: 2, Pred: 2 ✓
    [0.6, 0.3, 0.1],  # True: 0, Pred: 0 ✓
    [0.3, 0.4, 0.3],  # True: 1, Pred: 1 ✓ (barely)
])

# Calculate loss
loss_cce = categorical_cross_entropy (y_true_one_hot, y_pred_multi)
loss_sparse = sparse_categorical_cross_entropy (y_true_labels, y_pred_multi)

print("\\nCategorical Cross-Entropy Example:")
print(f"  CCE Loss (one-hot): {loss_cce:.4f}")
print(f"  CCE Loss (sparse): {loss_sparse:.4f}")
print(f"  Accuracy: {np.mean (np.argmax (y_pred_multi, axis=1) == y_true_labels):.1%}")

# Show per-sample losses
print("\\n  Per-sample losses:")
for i, (true_label, probs) in enumerate (zip (y_true_labels, y_pred_multi)):
    sample_loss = -np.log (probs[true_label])
    pred_label = np.argmax (probs)
    correct = "✓" if pred_label == true_label else "✗"
    print(f"    Sample {i}: true={true_label}, pred={pred_label} {correct}, "
          f"loss={sample_loss:.4f}")
\`\`\`

### Gradient with Softmax

When using softmax output + categorical cross-entropy:

\`\`\`
Output: ŷ = softmax (z)
Loss: L = -Σⱼ yⱼ log(ŷⱼ)

Gradient: ∂L/∂zᵢ = ŷᵢ - yᵢ  (remarkably simple!)
\`\`\`

This clean gradient is why softmax + cross-entropy is standard for classification.

## Custom Loss Functions

### Designing Custom Losses

For specific applications, you may need custom loss functions:

**1. Weighted Loss (Imbalanced Classes):**

\`\`\`python
def weighted_binary_cross_entropy (y_true, y_pred, pos_weight=1.0):
    """
    Weighted BCE for imbalanced datasets
    
    Args:
        pos_weight: Weight for positive class (>1 if positive class rare)
    """
    epsilon = 1e-10
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    
    loss = -(pos_weight * y_true * np.log (y_pred) + 
             (1 - y_true) * np.log(1 - y_pred))
    return np.mean (loss)

# Example: Rare positive class (fraud detection)
y_true_imbalanced = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 10% positive
y_pred_imbalanced = np.array([0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.1, 0.6])

loss_normal = binary_cross_entropy (y_true_imbalanced, y_pred_imbalanced)
loss_weighted = weighted_binary_cross_entropy (y_true_imbalanced, y_pred_imbalanced, 
                                               pos_weight=9.0)  # 9:1 ratio

print("\\nImbalanced Classification:")
print(f"  Standard BCE: {loss_normal:.4f}")
print(f"  Weighted BCE (9x): {loss_weighted:.4f}")
print("  → Weighted loss emphasizes the rare positive class")
\`\`\`

**2. Focal Loss (Hard Examples):**

\`\`\`python
def focal_loss (y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss - focuses on hard examples
    
    FL(p_t) = -α(1 - p_t)^γ log (p_t)
    
    Where p_t = p if y=1, else 1-p
    """
    epsilon = 1e-10
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    
    # p_t: probability of true class
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    # Focal term: (1 - p_t)^gamma
    focal_term = (1 - p_t) ** gamma
    
    # BCE term: -log (p_t)
    ce_term = -np.log (p_t)
    
    # Combine with alpha weighting
    loss = alpha * focal_term * ce_term
    return np.mean (loss)

# Focal loss reduces weight of easy examples (high p_t)
# and focuses on hard examples (low p_t)
\`\`\`

### Trading-Specific Loss Functions

**1. Directional Accuracy Loss:**

\`\`\`python
def directional_loss (y_true_returns, y_pred_returns):
    """
    Penalize incorrect return direction predictions
    Reward correct directions, penalize wrong ones
    """
    # Sign agreement: 1 if same direction, -1 if opposite
    direction_agreement = np.sign (y_true_returns) * np.sign (y_pred_returns)
    
    # Loss: negative of agreement (want to maximize agreement)
    # Add magnitude weighting
    magnitude_weight = np.abs (y_true_returns)
    loss = -np.mean (direction_agreement * magnitude_weight)
    
    return loss

# Example
true_returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
pred_returns = np.array([0.015, -0.008, 0.025, 0.005, 0.012])

dir_loss = directional_loss (true_returns, pred_returns)
print(f"\\nDirectional Loss: {dir_loss:.4f}")
print(f"Directional Accuracy: {np.mean (np.sign (true_returns) == np.sign (pred_returns)):.1%}")
\`\`\`

**2. Sharpe Ratio Loss:**

\`\`\`python
def sharpe_loss (y_pred_returns):
    """
    Negative Sharpe ratio as loss
    Goal: Maximize Sharpe by minimizing this loss
    """
    mean_return = np.mean (y_pred_returns)
    std_return = np.std (y_pred_returns)
    sharpe = mean_return / (std_return + 1e-6)
    
    # Minimize negative Sharpe = Maximize Sharpe
    return -sharpe

# Used in reinforcement learning for trading
\`\`\`

**3. Risk-Adjusted Return Loss:**

\`\`\`python
def risk_adjusted_loss (returns, predictions, risk_aversion=0.5):
    """
    Loss that balances returns and risk
    
    L = -E[r] + λ * Var[r]
    
    Where λ is risk aversion parameter
    """
    predicted_pnl = returns * predictions
    mean_return = np.mean (predicted_pnl)
    variance = np.var (predicted_pnl)
    
    # Risk-adjusted objective
    loss = -mean_return + risk_aversion * variance
    return loss
\`\`\`

## Choosing the Right Loss Function

### Decision Guide

\`\`\`
TASK TYPE?
│
├─→ REGRESSION
│   ├─→ Standard regression: MSE or MAE
│   ├─→ Outliers present: Huber loss
│   ├─→ Quantile regression: Quantile loss
│   └─→ Heavy tails: Log-cosh loss
│
├─→ BINARY CLASSIFICATION
│   ├─→ Standard: Binary cross-entropy
│   ├─→ Imbalanced: Weighted BCE or Focal loss
│   └─→ Probability calibration important: BCE
│
├─→ MULTI-CLASS CLASSIFICATION
│   ├─→ Mutually exclusive classes: Categorical CE
│   ├─→ Multiple labels possible: Binary CE per class
│   ├─→ Imbalanced: Weighted CE
│   └─→ Hard examples: Focal loss
│
└─→ RANKING / TRADING
    ├─→ Direction matters: Directional loss
    ├─→ Risk-adjusted: Sharpe or custom
    └─→ Transaction costs: Include in loss
\`\`\`

### Common Mistakes

❌ **Using MSE for classification:**
- Saturating gradients for wrong predictions
- Poor probability calibration
- Use cross-entropy instead

❌ **Using CE for regression:**
- Cross-entropy expects probabilities
- Not suitable for continuous targets

❌ **Ignoring class imbalance:**
- Model predicts majority class
- Use weighted loss or resampling

❌ **Not matching output activation and loss:**
- Sigmoid output → Binary CE
- Softmax output → Categorical CE
- Linear output → MSE/MAE

## Key Takeaways

1. **Loss functions** define what the network optimizes
2. **MSE** for regression - penalizes squared errors
3. **Cross-entropy** for classification - measures distribution distance
4. **Gradient properties** matter - avoid vanishing gradients
5. **Match loss to task** - classification needs CE, not MSE
6. **Custom losses** for domain-specific objectives
7. **Trading requires** risk-aware loss functions
8. **Consider imbalance** with weighted losses

## What\'s Next

We've covered forward propagation (predictions) and loss functions (measuring errors). Next:
- **Backpropagation**: Computing gradients to update weights
- The chain rule in action
- Efficient gradient computation
- The algorithm that makes deep learning possible
`,
};
