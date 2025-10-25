import { QuizQuestion } from '../../../types';

export const lossFunctionsQuiz: QuizQuestion[] = [
  {
    id: 'loss-functions-dq-1',
    question:
      "Explain why Mean Squared Error (MSE) is problematic for binary classification, even though it's mathematically valid. What issues arise with gradients, and why is Binary Cross-Entropy preferred?",
    sampleAnswer: `Using MSE for binary classification leads to severe optimization problems despite being mathematically valid. The issues stem from how MSE interacts with the sigmoid activation function commonly used for binary classification:

**The Problem with MSE for Classification:**

Setup:
- Output: ŷ = sigmoid (z) where z = wx + b
- MSE: L = (1/2)(y - ŷ)²
- Goal: Classify y ∈ {0, 1}

**Issue 1: Saturating Gradients**

The gradient of MSE w.r.t. pre-activation z:
\`\`\`
∂L/∂z = ∂L/∂ŷ · ∂ŷ/∂z
      = (ŷ - y) · σ'(z)
      = (ŷ - y) · σ(z)(1 - σ(z))
              ^^^^^^^^^^^^^^^^
                 Problem here!
\`\`\`

When the prediction is very wrong:
- If y = 1 and ŷ ≈ 0: z is very negative → σ(z) ≈ 0 → σ'(z) ≈ 0
- If y = 0 and ŷ ≈ 1: z is very positive → σ(z) ≈ 1 → σ'(z) ≈ 0

Result: **Gradient vanishes when you need it most** (when prediction is wrong)!

Concrete example:
\`\`\`python
# True label: y = 1
# Very wrong prediction: ŷ = 0.01 (z ≈ -4.6)

# MSE gradient
error = 0.01 - 1.0  # -0.99 (large error)
sigmoid_deriv = 0.01 * 0.99  # ≈ 0.01 (very small)
gradient = error * sigmoid_deriv  # ≈ -0.01 (tiny!)

# Training stalls despite huge error!
\`\`\`

**Issue 2: Non-Convex Loss Surface**

MSE with sigmoid creates a complex, non-convex loss landscape:
- Multiple local minima
- Plateaus where gradients are near zero
- Slow convergence even with good initialization
- Training can get stuck easily

**Issue 3: Misaligned Objectives**

MSE measures distance in probability space, but classification cares about decision boundary:
- MSE: Penalizes ŷ = 0.51 vs ŷ = 0.99 equally if both are correct
- Classification: Only cares if ŷ > 0.5 (decision boundary)
- Wastes optimization effort on already-correct predictions

**Why Binary Cross-Entropy Works Better:**

BCE formula:
\`\`\`
BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]
\`\`\`

Gradient w.r.t. pre-activation z:
\`\`\`
∂BCE/∂z = ∂BCE/∂ŷ · ∂ŷ/∂z
        = [(ŷ - y)/[ŷ(1-ŷ)]] · [ŷ(1-ŷ)]
        = ŷ - y  (simple!)
\`\`\`

**Benefits:**

1. **No Vanishing Gradient**: Gradient = ŷ - y (independent of σ'(z))
   - Large error → Large gradient
   - Works even when prediction is very wrong

2. **Convex Loss Surface**: With sigmoid + BCE, loss is convex
   - Single global minimum
   - Guaranteed convergence with gradient descent
   - Faster, more reliable training

3. **Probabilistic Interpretation**:
   - Maximizes log-likelihood
   - Optimal under Bayesian framework
   - Well-calibrated probabilities

4. **Appropriate Penalty Structure**:
   - Infinite loss for confident wrong prediction (ŷ=0 when y=1)
   - Logarithmic scaling focuses on decision boundary

**Numerical Comparison:**

\`\`\`python
import numpy as np

# Scenario: True label y = 1
y_true = 1.0
predictions = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])

# MSE loss and gradient
mse_losses = 0.5 * (y_true - predictions) ** 2
mse_gradients = predictions - y_true
# With sigmoid: multiply by σ'(z) which is small for extreme predictions

# BCE loss and gradient
bce_losses = -np.log (predictions)
bce_gradients_preactivation = predictions - y_true  # After cancellation

print("Comparison (True label = 1):")
print("\\nPrediction | MSE Loss | MSE Grad | BCE Loss | BCE Grad (pre-z)")
print("-" * 65)
for p, mse_l, mse_g, bce_l, bce_g in zip (predictions, mse_losses, 
                                           mse_gradients, bce_losses, 
                                           bce_gradients_preactivation):
    print(f"{p:6.2f}     | {mse_l:8.4f} | {mse_g:8.4f} | {bce_l:8.4f} | {bce_g:8.4f}")

# Output shows:
# - When p = 0.01 (very wrong): MSE gradient ≈ -0.99 BUT gets multiplied by σ'(z) ≈ 0.01 → tiny!
# - When p = 0.01 (very wrong): BCE gradient = -0.99 (large, as it should be)
\`\`\`

**Real Training Example:**

\`\`\`python
# Train with both losses
def train_comparison():
    X = generate_binary_data()
    y = binary_labels()
    
    model_mse = train_with_mse(X, y, epochs=1000)
    model_bce = train_with_bce(X, y, epochs=1000)
    
    # Typical results:
    # MSE: 50 epochs to converge, gets stuck at 85% accuracy
    # BCE: 20 epochs to converge, reaches 98% accuracy
    
    return model_mse, model_bce
\`\`\`

**When MSE Might Be Acceptable:**

Only if:
- Predictions already well-calibrated (unlikely initially)
- Using linear output (no sigmoid) - but then not true classification
- Outputs are already far from saturation regions
- You don't care about probability calibration

**Conclusion:**

MSE for binary classification fails because:
1. Vanishing gradients when most needed (wrong predictions)
2. Non-convex optimization landscape
3. Misaligned with classification objectives
4. Poor probability calibration

BCE solves all these issues:
1. Gradient = ŷ - y (no vanishing)
2. Convex with sigmoid
3. Probabilistically motivated
4. Proper calibration

**Recommendation**: Always use Binary Cross-Entropy for binary classification. The mathematical elegance of BCE + sigmoid is not accidental—it's the optimal combination for this task.`,
    keyPoints: [
      'MSE with sigmoid causes vanishing gradients when predictions are most wrong',
      "Gradient ∂L/∂z includes σ'(z) which approaches 0 for extreme activations",
      "BCE gradient ∂L/∂z = ŷ - y cancels σ'(z), avoiding vanishing gradient",
      'MSE creates non-convex loss surface with sigmoid; BCE is convex',
      'BCE has probabilistic interpretation (maximum likelihood)',
      'BCE provides better probability calibration than MSE',
      'Training with BCE converges faster and more reliably',
    ],
  },
  {
    id: 'loss-functions-dq-2',
    question:
      'In trading applications, standard loss functions like MSE may not capture the true objective (profit). Design a custom loss function for a stock return prediction model that accounts for transaction costs, directional accuracy, and risk aversion. Explain your design choices.',
    sampleAnswer: `Designing a trading-specific loss function requires balancing multiple objectives that standard ML losses ignore. Here\'s a comprehensive approach:

**Standard ML vs Trading Objectives:**

Standard ML:
- MSE: Minimize squared prediction error
- Objective: Accurate point predictions
- Ignores: Trading costs, directionality, risk

Trading Reality:
- Goal: Maximize risk-adjusted returns
- Costs: Transaction fees, slippage, market impact
- Direction matters more than magnitude (often)
- Risk management is critical

**Custom Trading Loss Function Design:**

\`\`\`python
def trading_loss (y_pred_returns, y_true_returns, positions, 
                 transaction_cost=0.001, risk_aversion=0.5):
    """
    Comprehensive trading loss function
    
    Components:
    1. Directional accuracy (most important)
    2. P&L from predictions
    3. Transaction costs
    4. Risk penalty (variance)
    5. Drawdown penalty
    
    Args:
        y_pred_returns: Predicted returns, shape (n_samples,)
        y_true_returns: Actual returns, shape (n_samples,)
        positions: Position sizes [-1, 1], shape (n_samples,)
        transaction_cost: Cost per trade (e.g., 0.1%)
        risk_aversion: Risk penalty weight (higher = more conservative)
    
    Returns:
        loss: Scalar value to minimize
    """
    n_samples = len (y_true_returns)
    
    # Component 1: Directional Accuracy Loss
    # Penalize wrong direction, reward correct direction
    direction_correct = np.sign (y_pred_returns) == np.sign (y_true_returns)
    direction_loss = -np.mean (direction_correct.astype (float))
    
    # Weight by magnitude of true move
    magnitude_weights = np.abs (y_true_returns)
    weighted_direction_loss = -np.mean (direction_correct * magnitude_weights)
    
    # Component 2: P&L-based Loss
    # Positions based on predictions
    predicted_pnl = positions * y_true_returns
    pnl_loss = -np.mean (predicted_pnl)  # Minimize negative PnL = Maximize PnL
    
    # Component 3: Transaction Cost Penalty
    # Cost incurred when changing positions
    position_changes = np.abs (np.diff (positions, prepend=0))
    total_transaction_costs = np.sum (position_changes * transaction_cost)
    transaction_loss = total_transaction_costs / n_samples
    
    # Component 4: Risk Penalty (Variance)
    # Penalize high variance (unstable returns)
    pnl_variance = np.var (predicted_pnl)
    risk_loss = risk_aversion * pnl_variance
    
    # Component 5: Drawdown Penalty
    # Penalize large drawdowns
    cumulative_pnl = np.cumsum (predicted_pnl)
    running_max = np.maximum.accumulate (cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = np.max (drawdowns)
    drawdown_loss = 0.1 * max_drawdown  # Penalty for large drawdown
    
    # Component 6: Sharpe Ratio (Negative for Minimization)
    mean_return = np.mean (predicted_pnl)
    std_return = np.std (predicted_pnl) + 1e-6
    sharpe_ratio = mean_return / std_return
    sharpe_loss = -sharpe_ratio  # Minimize negative Sharpe = Maximize Sharpe
    
    # Combine all components
    total_loss = (
        0.3 * weighted_direction_loss +   # 30% weight on direction
        0.3 * pnl_loss +                   # 30% weight on P&L
        0.2 * transaction_loss +           # 20% weight on costs
        0.1 * risk_loss +                  # 10% weight on risk
        0.1 * drawdown_loss                # 10% weight on drawdown
    )
    
    # Alternative: Use Sharpe directly
    # total_loss = sharpe_loss + transaction_loss
    
    return total_loss, {
        'direction_loss': weighted_direction_loss,
        'pnl_loss': pnl_loss,
        'transaction_loss': transaction_loss,
        'risk_loss': risk_loss,
        'drawdown_loss': drawdown_loss,
        'sharpe_ratio': sharpe_ratio,
    }


# Example usage
np.random.seed(42)
n_days = 252  # One trading year

# Simulate returns
y_true_returns = np.random.randn (n_days) * 0.02
y_pred_returns = y_true_returns + np.random.randn (n_days) * 0.01

# Position sizing based on predictions (simple: sign and scale)
positions = np.clip (y_pred_returns * 10, -1, 1)  # Scale to [-1, 1]

# Calculate loss
loss, components = trading_loss (y_pred_returns, y_true_returns, positions,
                                transaction_cost=0.001, risk_aversion=0.5)

print("Trading Loss Components:")
for component, value in components.items():
    print(f"  {component}: {value:.6f}")
print(f"\\nTotal Loss: {loss:.6f}")

# Compare with standard MSE
mse = np.mean((y_pred_returns - y_true_returns) ** 2)
print(f"Standard MSE: {mse:.6f}")
print("\\n→ Trading loss captures profitability, not just prediction accuracy")
\`\`\`

**Design Rationale:**

**1. Directional Accuracy (30% weight):**
- In trading, getting the direction right is often more important than exact magnitude
- A prediction of +0.5% when truth is +2% is better than -0.5%
- Weight by magnitude: larger moves more important

**2. P&L Component (30% weight):**
- Directly optimizes what we care about: profit
- Position-weighted returns = actual P&L
- Encourages predictions that lead to profitable trades

**3. Transaction Costs (20% weight):**
- Critical in real trading
- Prevents overtrading (excessive position changes)
- Encourages holding periods > 1 day
- Typical costs: 0.05-0.2% per trade

**4. Risk Penalty (10% weight):**
- Variance of P&L
- Prevents strategies with high volatility
- Encourages stable, consistent returns
- Weight increases with risk_aversion parameter

**5. Drawdown Penalty (10% weight):**
- Maximum drawdown = largest peak-to-trough decline
- Critical for investor psychology
- Prevents catastrophic losses
- 20% drawdown worse than 2x 10% drawdowns

**Alternative Formulations:**

**A. Pure Sharpe Ratio Loss:**
\`\`\`python
def sharpe_ratio_loss (predictions, returns, positions):
    pnl = positions * returns
    return -np.mean (pnl) / (np.std (pnl) + 1e-6)

# Simplest, most direct
# Downside: Doesn't explicitly penalize transaction costs
\`\`\`

**B. Risk-Parity Loss:**
\`\`\`python
def risk_parity_loss (predictions, returns, target_vol=0.15):
    """Maintain constant volatility"""
    pnl = predictions * returns
    current_vol = np.std (pnl)
    vol_penalty = (current_vol - target_vol) ** 2
    return -np.mean (pnl) + vol_penalty
\`\`\`

**C. Kelly Criterion Loss:**
\`\`\`python
def kelly_criterion_loss (predictions, returns):
    """Optimal position sizing"""
    win_rate = np.mean (returns > 0)
    avg_win = np.mean (returns[returns > 0])
    avg_loss = np.mean (np.abs (returns[returns < 0]))
    
    kelly_fraction = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    positions = predictions * kelly_fraction
    
    return -np.mean (positions * returns)
\`\`\`

**Practical Implementation Considerations:**

**1. Differentiability:**
All components must be differentiable for backpropagation:
\`\`\`python
# AVOID: Non-differentiable operations
if prediction > 0:
    return profit
else:
    return loss

# PREFER: Smooth approximations
return torch.sigmoid (prediction * 10) * profit + (1 - torch.sigmoid (prediction * 10)) * loss
\`\`\`

**2. Numerical Stability:**
\`\`\`python
# Add epsilon to prevent division by zero
sharpe = mean / (std + 1e-6)

# Clip extreme values
positions = np.clip (positions, -1, 1)
\`\`\`

**3. Time-Series Considerations:**
\`\`\`python
# Don't shuffle time series data!
# Maintain temporal order in loss calculation

# Consider look-ahead bias
# Only use information available at time t for time t+1 prediction
\`\`\`

**4. Backtesting Alignment:**
\`\`\`python
# Loss function should match backtesting metrics
# If you backtest on Sharpe, train on Sharpe loss
# Ensures training and evaluation are aligned
\`\`\`

**Hyperparameter Tuning:**

\`\`\`python
# Grid search over loss function weights
weight_configs = [
    {'direction': 0.5, 'pnl': 0.3, 'cost': 0.2},
    {'direction': 0.3, 'pnl': 0.5, 'cost': 0.2},
    {'direction': 0.4, 'pnl': 0.4, 'cost': 0.2},
]

# Validate on out-of-sample Sharpe ratio
best_config = None
best_sharpe = -np.inf

for config in weight_configs:
    model = train_with_loss (config)
    sharpe = evaluate_sharpe (model, val_data)
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_config = config
\`\`\`

**Challenges:**

1. **Multiple Objectives**: Hard to balance direction, profit, risk
2. **Non-Stationarity**: Optimal weights change with market regime
3. **Overfitting**: Easy to overfit to training period
4. **Computational**: Complex loss slows training

**Best Practices:**

1. **Start Simple**: Begin with Sharpe ratio loss
2. **Add Complexity**: Add transaction costs next, then other components
3. **Validate Out-of-Sample**: Always test on unseen data
4. **Walk-Forward**: Retrain regularly with recent data
5. **Monitor Components**: Track individual loss components during training
6. **Domain Knowledge**: Weight components based on trading style

**Conclusion:**

Custom trading loss functions bridge the gap between ML objectives (prediction accuracy) and trading objectives (profitability). Key design principles:

1. **Directional accuracy** often more important than magnitude
2. **Transaction costs** must be explicitly modeled
3. **Risk management** through variance and drawdown penalties
4. **Differentiability** for gradient-based optimization
5. **Alignment** with backtesting metrics

The ideal loss function depends on your trading style, constraints, and objectives. Start with Sharpe ratio, add transaction costs, then customize based on your specific needs.`,
    keyPoints: [
      'Standard ML losses (MSE) optimize prediction accuracy, not profitability',
      'Trading requires multi-objective loss: direction, P&L, costs, risk, drawdown',
      'Directional accuracy often more important than magnitude accuracy',
      'Transaction costs must be explicitly modeled to prevent overtrading',
      'Risk penalties (variance, drawdown) essential for stable strategies',
      'Sharpe ratio loss directly optimizes risk-adjusted returns',
      'All components must be differentiable for backpropagation',
      'Loss function should align with backtesting metrics',
      'Weight components based on trading style and constraints',
    ],
  },
  {
    id: 'loss-functions-dq-3',
    question:
      'Explain why the combination of softmax activation and categorical cross-entropy loss produces such a clean gradient (∂L/∂z = ŷ - y). Walk through the mathematical derivation and discuss why this property is important for training neural networks.',
    sampleAnswer: `The clean gradient from softmax + categorical cross-entropy is one of the most elegant results in deep learning. Let\'s derive it step by step:

**Setup:**

Output layer:
- Pre-activation (logits): z = [z₁, z₂, ..., zₓ] for C classes
- Activation (softmax): ŷᵢ = exp (zᵢ) / Σⱼ exp (zⱼ)
- True labels (one-hot): y = [y₁, y₂, ..., yₓ] where yᵢ ∈ {0,1} and Σᵢyᵢ = 1

Loss function:
- Categorical cross-entropy: L = -Σᵢ yᵢ log(ŷᵢ)

**Goal:** Find ∂L/∂zₖ for any logit zₖ

**Step 1: Gradient of Loss w.r.t. Softmax Output**

\`\`\`
∂L/∂ŷₖ = ∂/∂ŷₖ [-Σᵢ yᵢ log(ŷᵢ)]
       = -yₖ/ŷₖ
\`\`\`

Simple application of chain rule and derivative of log.

**Step 2: Gradient of Softmax w.r.t. Logits**

This is the tricky part. Softmax for element i:
\`\`\`
ŷᵢ = exp (zᵢ) / Σⱼ exp (zⱼ)
\`\`\`

We need ∂ŷᵢ/∂zₖ. Two cases:

**Case 1: i = k (diagonal elements)**
\`\`\`
∂ŷᵢ/∂zᵢ = ∂/∂zᵢ [exp (zᵢ) / Σⱼ exp (zⱼ)]

Using quotient rule: ∂(u/v)/∂x = (u'v - uv')/v²

u = exp (zᵢ), u' = exp (zᵢ)
v = Σⱼ exp (zⱼ), v' = exp (zᵢ)  (only zᵢ term in sum depends on zᵢ)

∂ŷᵢ/∂zᵢ = [exp (zᵢ)·Σⱼexp (zⱼ) - exp (zᵢ)·exp (zᵢ)] / [Σⱼexp (zⱼ)]²
        = exp (zᵢ)/Σⱼexp (zⱼ) · [1 - exp (zᵢ)/Σⱼexp (zⱼ)]
        = ŷᵢ · (1 - ŷᵢ)
\`\`\`

**Case 2: i ≠ k (off-diagonal elements)**
\`\`\`
∂ŷᵢ/∂zₖ = ∂/∂zₖ [exp (zᵢ) / Σⱼ exp (zⱼ)]

u = exp (zᵢ), u' = 0  (doesn't depend on zₖ)
v = Σⱼ exp (zⱼ), v' = exp (zₖ)  (only zₖ term depends on zₖ)

∂ŷᵢ/∂zₖ = [0·Σⱼexp (zⱼ) - exp (zᵢ)·exp (zₖ)] / [Σⱼexp (zⱼ)]²
        = -exp (zᵢ)/Σⱼexp (zⱼ) · exp (zₖ)/Σⱼexp (zⱼ)
        = -ŷᵢ · ŷₖ
\`\`\`

**Step 3: Chain Rule to Combine**

\`\`\`
∂L/∂zₖ = Σᵢ (∂L/∂ŷᵢ) · (∂ŷᵢ/∂zₖ)
\`\`\`

Substitute our results:
\`\`\`
∂L/∂zₖ = Σᵢ (-yᵢ/ŷᵢ) · (∂ŷᵢ/∂zₖ)
\`\`\`

Split the sum into i=k and i≠k terms:

**Term 1 (i = k):**
\`\`\`
(-yₖ/ŷₖ) · ŷₖ(1 - ŷₖ) = -yₖ(1 - ŷₖ)
                       = -yₖ + yₖŷₖ
\`\`\`

**Term 2 (i ≠ k):**
\`\`\`
Σᵢ≠ₖ (-yᵢ/ŷᵢ) · (-ŷᵢŷₖ) = Σᵢ≠ₖ yᵢŷₖ
                         = ŷₖ Σᵢ≠ₖ yᵢ
\`\`\`

**Combine:**
\`\`\`
∂L/∂zₖ = (-yₖ + yₖŷₖ) + (ŷₖ Σᵢ≠ₖ yᵢ)
       = -yₖ + yₖŷₖ + ŷₖ Σᵢ≠ₖ yᵢ
       = -yₖ + ŷₖ(yₖ + Σᵢ≠ₖ yᵢ)
       = -yₖ + ŷₖ(Σᵢ yᵢ)
       = -yₖ + ŷₖ·1    (because y is one-hot, Σᵢ yᵢ = 1)
       = ŷₖ - yₖ
\`\`\`

**Result: ∂L/∂zₖ = ŷₖ - yₖ**

Beautifully simple! For all classes:
\`\`\`
∂L/∂z = ŷ - y
\`\`\`

**Why This Is Important:**

**1. Computational Efficiency:**
No need to compute and store intermediate Jacobian matrices. Just subtract!

\`\`\`python
# Instead of complex chain rule computation:
# dL_dz = (dL_dy @ dy_dz)  # Matrix multiplication

# We get:
dL_dz = y_pred - y_true  # Simple subtraction!
\`\`\`

**2. No Vanishing Gradient:**

Gradient magnitude = |ŷ - y|:
- When very wrong: |ŷ - y| ≈ 1 (large gradient)
- When correct: |ŷ - y| ≈ 0 (small gradient, as desired)
- No sigmoid derivative term that would cause vanishing gradients

**3. Intuitive Interpretation:**

Gradient = error in probability space:
- If y = [0, 1, 0] and ŷ = [0.2, 0.7, 0.1]
- Gradient = [-0.2, -0.3, +0.1]
- Interpretation: Decrease class 0 and 2, increase class 1

**4. Numerical Stability:**

The derivation shows why numerical stability matters:

\`\`\`python
# Naive implementation (numerically unstable):
def softmax_naive (z):
    return np.exp (z) / np.sum (np.exp (z))

# For large z values, exp (z) overflows!
z = np.array([1000, 1001, 1002])
# exp(1000) ≈ 10^434 → overflow!

# Stable implementation:
def softmax_stable (z):
    z_shifted = z - np.max (z)  # Shift for stability
    exp_z = np.exp (z_shifted)
    return exp_z / np.sum (exp_z)

# Now: exp(0), exp(1), exp(2) → No overflow!
\`\`\`

The clean gradient ŷ - y is preserved even after numerical stabilization.

**5. Gradient Flow in Deep Networks:**

In backpropagation:
\`\`\`
∂L/∂wₗ = ∂L/∂zL · ∂zL/∂aL-1 · ... · ∂a1/∂z1 · ∂z1/∂w1

The fact that ∂L/∂zL = ŷ - y (not involving derivative terms) means:
- Initial gradient is clear and well-scaled
- No vanishing gradient from output layer
- Makes training deep networks practical
\`\`\`

**Comparison with Other Combinations:**

**MSE + Linear:**
\`\`\`
L = (1/2)(y - ŷ)²
∂L/∂ŷ = ŷ - y
∂ŷ/∂z = 1 (linear)
∂L/∂z = ŷ - y  (also clean!)
\`\`\`
But: Linear output not suitable for probabilities

**MSE + Sigmoid:**
\`\`\`
L = (1/2)(y - ŷ)²
∂L/∂ŷ = ŷ - y
∂ŷ/∂z = σ'(z) = σ(z)(1-σ(z))
∂L/∂z = (ŷ - y) · σ(z)(1-σ(z))  (vanishing gradient!)
\`\`\`
The σ'(z) term causes problems

**BCE + Sigmoid:**
\`\`\`
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
∂L/∂ŷ = (ŷ - y)/[ŷ(1-ŷ)]
∂ŷ/∂z = σ(z)(1-σ(z))
∂L/∂z = [(ŷ - y)/[ŷ(1-ŷ)]] · [ŷ(1-ŷ)]
      = ŷ - y  (clean!)
\`\`\`
Binary version also clean!

**Mathematical Insight:**

The cleanliness comes from the fact that:
1. Cross-entropy is the conjugate prior to softmax/sigmoid
2. They "cancel" each other's nonlinearities
3. Result: Linear relationship between gradient and error

This is not a coincidence—it's a consequence of maximum likelihood estimation and exponential families.

**Practical Implementation:**

\`\`\`python
import torch

# PyTorch combines softmax + cross-entropy for numerical stability
logits = model (x)  # Raw outputs (no softmax!)

# Don't do this (numerically unstable):
# probs = torch.softmax (logits, dim=1)
# loss = -torch.mean (torch.log (probs[range (len (y)), y]))

# Do this (stable):
loss = torch.nn.CrossEntropyLoss()(logits, y)

# Backward pass automatically computes clean gradient
loss.backward()
# logits.grad = probs - y_one_hot  (simple!)
\`\`\`

**Conclusion:**

The gradient ∂L/∂z = ŷ - y emerges from the mathematical interplay between softmax and cross-entropy. This elegant result:

1. Simplifies computation (just subtraction)
2. Prevents vanishing gradients  
3. Provides intuitive interpretation (error signal)
4. Enables stable numerical implementation
5. Facilitates gradient flow in deep networks

This is why softmax + cross-entropy is the universal standard for multi-class classification. The mathematics aligns perfectly with the optimization needs.`,
    keyPoints: [
      'Gradient ∂L/∂z = ŷ - y emerges from softmax derivative canceling with cross-entropy',
      'Softmax derivative has diagonal (ŷᵢ(1-ŷᵢ)) and off-diagonal (-ŷᵢŷₖ) terms',
      'Chain rule combines these terms with -yᵢ/ŷᵢ from cross-entropy',
      'One-hot constraint Σyᵢ=1 enables final simplification to ŷ-y',
      'Clean gradient prevents vanishing gradient problem',
      'Computationally efficient - just subtraction, no matrix operations',
      'Gradient magnitude proportional to error: large when wrong, small when correct',
      'Similar clean gradient for binary case: sigmoid + BCE also gives ŷ-y',
    ],
  },
];
