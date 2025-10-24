/**
 * Quiz questions for Exponents & Logarithms section
 */

export const exponentslogarithmsQuiz = [
  {
    id: 'dq1-log-space-stability',
    question:
      'Explain why computing in log space is more numerically stable than direct computation. Provide specific examples from machine learning where this matters (softmax, likelihood computation). What are the trade-offs?',
    sampleAnswer: `Computing in log space is crucial for numerical stability in machine learning, especially when dealing with very small or very large numbers that can cause underflow or overflow.

**Why Log Space is More Stable**:

**Problem 1: Underflow**
When multiplying many small probabilities (common in ML), the product can underflow to 0:

\`\`\`python
# Example: Computing likelihood of a sequence
probs = np.array([0.1, 0.15, 0.08, 0.12, 0.09])
print(f"Direct product: {np.prod(probs)}")  # 0.00001296

# With 100 such probabilities
tiny_probs = np.full(100, 0.1)
print(f"100 probs: {np.prod(tiny_probs):.2e}")  # Underflows to 0!

# Log space is stable
log_likelihood = np.sum(np.log(tiny_probs))
print(f"Log-likelihood: {log_likelihood:.4f}")  # -230.2585 (stable!)
# Convert back if needed: np.exp(log_likelihood)
\`\`\`

**Why it works**:
- log(a · b · c) = log(a) + log(b) + log(c)
- Turns multiplication → addition
- Addition is numerically stable
- Log maps (0, 1) → (-∞, 0), spreading out small numbers

**Problem 2: Overflow in Softmax**

Naive softmax with large inputs overflows:

\`\`\`python
def softmax_naive(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Large logits (common in deep networks)
logits = np.array([1000, 1001, 1002])
try:
    result = softmax_naive(logits)
    print(result)  # RuntimeWarning: overflow
except:
    print("OVERFLOW!")

# Log-sum-exp trick
def softmax_stable(x):
    x_shifted = x - np.max(x)  # Shift by max
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

result_stable = softmax_stable(logits)
print(f"Stable result: {result_stable}")  # [0.09, 0.24, 0.67]
\`\`\`

**Derivation**:
softmax(x) = exp(xᵢ) / Σexp(xⱼ)
= exp(xᵢ - c) / Σexp(xⱼ - c)  [for any constant c]
= exp(xᵢ - max(x)) / Σexp(xⱼ - max(x))  [choose c = max(x)]

Subtracting max ensures all exponentials are ≤ 1, preventing overflow.

**Application 3: Log-Likelihood in Training**

Maximum likelihood estimation is more stable in log space:

\`\`\`python
# Likelihood of data given model parameters
def likelihood(data, model_params):
    """Product of individual probabilities"""
    probs = [model_prob(x, model_params) for x in data]
    return np.prod(probs)  # Can underflow!

def log_likelihood(data, model_params):
    """Sum of log probabilities"""
    log_probs = [np.log(model_prob(x, model_params)) for x in data]
    return np.sum(log_probs)  # Stable!

# In practice, we minimize negative log-likelihood
nll = -log_likelihood(data, params)
\`\`\`

**Why this matters**:
- Maximizing likelihood = Minimizing negative log-likelihood
- log is monotonic, so arg max doesn't change
- But computation is stable

**Application 4: Numerical Precision**

\`\`\`python
# Compare precision
a = 1e-100
b = 1e-100

# Direct multiplication
product = a * b
print(f"Direct: {product}")  # May be 0 due to underflow

# Log space
log_product = np.log(a) + np.log(b)
print(f"Log space: {log_product:.4f}")  # -460.5170 (precise!)
print(f"Recovered: {np.exp(log_product):.2e}")  # 1.00e-200
\`\`\`

**Trade-offs**:

**Advantages**:
✅ Prevents underflow/overflow
✅ Multiplication becomes addition (faster, more accurate)
✅ Essential for long sequences (RNNs, HMMs)
✅ Natural for likelihood-based methods

**Disadvantages**:
❌ log() and exp() are expensive operations
❌ Must remember to convert back (exp) when needed
❌ Can't directly compare probabilities (must exponentiate)
❌ Not intuitive (working with log-probabilities)
❌ Requires careful bookkeeping

**When to Use Log Space**:

✅ **Always use** for:
- Likelihood computation with many terms
- Softmax with potentially large logits
- Sequence modeling (RNNs, HMMs)
- Bayesian inference
- Information theory metrics

❌ **Don't need** for:
- Single probability computations
- Small datasets where underflow unlikely
- When direct probability interpretation needed

**Real Trading Example**:

\`\`\`python
# Bayesian portfolio optimization
def portfolio_log_likelihood(returns, weights, params):
    """
    Compute log-likelihood of portfolio returns
    More stable than direct likelihood
    """
    residuals = returns - np.dot(weights, params)
    # Log of Gaussian likelihood
    log_like = -0.5 * np.sum(residuals**2 / params['variance',])
    log_like -= 0.5 * len(returns) * np.log(2 * np.pi * params['variance',])
    return log_like

# Optimize in log space, interpret results in probability space
\`\`\`

**Summary**:
Log space transforms multiplication into addition, preventing numerical issues with extreme values. Essential for ML stability, especially in deep learning and probabilistic models. The computational overhead is worth it for numerical reliability.`,
    keyPoints: [
      'Log space turns multiplication into addition, preventing underflow/overflow',
      'Softmax uses log-sum-exp trick: shift by max before exponentiation',
      'Maximum likelihood = Minimum negative log-likelihood (stable optimization)',
      'Trade-off: computational cost vs numerical stability',
      'Essential for: sequence models, likelihood computation, deep networks',
    ],
  },
  {
    id: 'dq2-compound-growth',
    question:
      'Compare linear growth, exponential growth, and logarithmic growth. For each, provide the mathematical form, real-world examples, and explain when each dominates. How does compound interest relate to portfolio returns in trading?',
    sampleAnswer: `Understanding different growth patterns is fundamental to mathematics, computer science, and finance. Each has distinct characteristics and applications.

**Linear Growth: f(x) = mx + b**

**Characteristics**:
- Constant additive change per unit
- Straight line on regular plot
- Predictable, steady growth

**Examples**:
- Saving $100/month (no interest)
- Distance traveled at constant speed
- Simple interest: I = Prt

\`\`\`python
def linear_growth(initial, rate, time):
    return initial + rate * time

# $1000 + $100/month
t = np.arange(0, 60)  # 60 months
linear = linear_growth(1000, 100, t)

plt.plot(t, linear, label='Linear', linewidth=2)
plt.xlabel('Time (months)')
plt.ylabel('Value ($)')
plt.title('Linear Growth')
plt.grid(True)
\`\`\`

**Exponential Growth: f(x) = a · bˣ or f(x) = a · eʳˣ**

**Characteristics**:
- Constant multiplicative change per unit
- Growth rate proportional to current value
- Curves upward on regular plot, straight on log plot

**Examples**:
- Compound interest
- Population growth
- Viral spread
- Neural network gradient explosion
- Portfolio returns (compounded)

\`\`\`python
def exponential_growth(initial, rate, time):
    return initial * np.exp(rate * time)

# $1000 at 10% annual rate, compounded
t = np.arange(0, 60)
exponential = exponential_growth(1000, 0.10/12, t)  # Monthly

plt.plot(t, exponential, label='Exponential', linewidth=2)
\`\`\`

**Logarithmic Growth: f(x) = a · log(x) + b**

**Characteristics**:
- Growth slows down over time
- Early rapid growth, then plateaus
- Inverse of exponential

**Examples**:
- Algorithm complexity: binary search O(log n)
- Information gain from data (diminishing returns)
- Depth of balanced binary tree
- Learning curves (diminishing improvement)

\`\`\`python
def logarithmic_growth(initial, scale, time):
    return initial + scale * np.log(time + 1)  # +1 to avoid log(0)

t = np.arange(0, 60)
logarithmic = logarithmic_growth(1000, 500, t)

plt.plot(t, logarithmic, label='Logarithmic', linewidth=2)
\`\`\`

**Comparison**:

\`\`\`python
# Compare all three
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

t = np.arange(1, 61)
linear = 1000 + 100 * t
exponential = 1000 * (1.10)**(t/12)
logarithmic = 1000 + 500 * np.log(t)

# Linear scale
ax1.plot(t, linear, label='Linear', linewidth=2)
ax1.plot(t, exponential, label='Exponential', linewidth=2)
ax1.plot(t, logarithmic, label='Logarithmic', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_title('Growth Patterns (Linear Scale)')
ax1.legend()
ax1.grid(True)

# Log scale
ax2.plot(t, linear, label='Linear', linewidth=2)
ax2.plot(t, exponential, label='Exponential', linewidth=2)
ax2.plot(t, logarithmic, label='Logarithmic', linewidth=2)
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.set_yscale('log')
ax2.set_title('Growth Patterns (Log Scale)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# On log scale:
# - Exponential becomes linear
# - Linear becomes logarithmic
# - Logarithmic becomes even flatter
\`\`\`

**When Each Dominates**:

**Short term** (small x):
- All similar initially
- Logarithmic grows fastest early
- Exponential looks slow at first

**Medium term**:
- Exponential overtakes linear
- Linear overtakes logarithmic

**Long term** (large x):
- Exponential dominates everything
- Linear grows steadily
- Logarithmic barely increases

**Formal relationships**:
log(x) << x << xᵏ << eˣ << x! << xˣ

**Compound Interest & Portfolio Returns**:

**Compound Interest Formula**:
A = P(1 + r/n)ⁿᵗ

As n → ∞ (continuous): A = Peʳᵗ

\`\`\`python
# Compare simple vs compound returns
initial = 10000
annual_rate = 0.12  # 12% annual
years = np.arange(0, 30)

# Simple interest (linear)
simple = initial * (1 + annual_rate * years)

# Compound annually
compound_annual = initial * (1 + annual_rate)**years

# Compound monthly
compound_monthly = initial * (1 + annual_rate/12)**(12*years)

# Continuous compounding
continuous = initial * np.exp(annual_rate * years)

plt.figure(figsize=(12, 6))
plt.plot(years, simple, label='Simple (Linear)', linewidth=2)
plt.plot(years, compound_annual, label='Compound (Annual)', linewidth=2)
plt.plot(years, compound_monthly, label='Compound (Monthly)', linewidth=2)
plt.plot(years, continuous, label='Continuous', linewidth=2, linestyle='--')
plt.xlabel('Years')
plt.ylabel('Portfolio Value ($)')
plt.title('Simple vs Compound Returns')
plt.legend()
plt.grid(True)

print(f"After 30 years:")
print(f"Simple: \${simple[-1]:,.2f}")
print(f"Compound: \${compound_annual[-1]:,.2f}")
print(f"Continuous: \${continuous[-1]:,.2f}")
            \`\`\`

**Trading Application: Compounding Returns**:

\`\`\`python
def portfolio_simulation(initial, monthly_returns):
    """
    Simulate portfolio with compound returns
    Each month's return compounds on previous value
    """
    portfolio_values = [initial]
    current_value = initial
    
    for r in monthly_returns:
        current_value *= (1 + r)  # Compound effect
        portfolio_values.append(current_value)
    
    return np.array(portfolio_values)

# Simulate 5 years of trading
np.random.seed(42)
months = 60
# Average 1% monthly return with 5% volatility
monthly_returns = np.random.normal(0.01, 0.05, months)

portfolio = portfolio_simulation(10000, monthly_returns)

# Compare with simple (non-compounded)
simple_portfolio = 10000 * (1 + np.cumsum(np.insert(monthly_returns, 0, 0)))

plt.figure(figsize=(12, 6))
plt.plot(portfolio, label='Compound Returns', linewidth=2)
plt.plot(simple_portfolio, label='Simple Returns', linewidth=2, linestyle='--')
plt.xlabel('Month')
plt.ylabel('Portfolio Value ($)')
plt.title('Compound vs Simple Returns in Trading')
plt.legend()
plt.grid(True)

total_compound = (portfolio[-1] - 10000) / 10000 * 100
total_simple = (simple_portfolio[-1] - 10000) / 10000 * 100
print(f"Total return (compound): {total_compound:.2f}%")
print(f"Total return (simple): {total_simple:.2f}%")
print(f"Difference: {total_compound - total_simple:.2f}%")
\`\`\`

**Key Insights for Trading**:

1. **Compound returns matter**: Even small differences in return rates compound dramatically over time

2. **Drawdowns hurt more with compounding**: 
   - Lose 50% → need 100% gain to recover
   - Because you're compounding from a lower base

3. **Consistent small gains beat volatile large swings**:
   - 1% monthly (compounded) = 12.68% annually
   - 12% once per year = 12% annually
   - Compounding frequency matters!

4. **Exponential growth is powerful but rare**:
   - Can't sustain indefinitely (reversion to mean)
   - Market returns are NOT perfectly exponential
   - But long-term equity returns approximate it

**Summary**:
- Linear: constant absolute change (O(n))
- Exponential: constant relative change, dominates long-term (O(eⁿ))
- Logarithmic: diminishing returns, very efficient (O(log n))
- Compound interest = exponential growth
- In trading, compounding small consistent returns is powerful
- Understanding growth patterns helps with algorithm choice and portfolio strategy`,
    keyPoints: [
      'Linear: constant addition; Exponential: constant multiplication; Logarithmic: inverse of exponential',
      'Long-term: Exponential dominates, then linear, then logarithmic',
      'Compound interest is exponential growth: A = P(1+r)ⁿ',
      'In trading, compounding small consistent returns beats large volatile swings',
      'Drawdowns hurt more with compounding: -50% requires +100% to recover',
    ],
  },
  {
    id: 'dq3-entropy-information',
    question:
      'Explain Shannon entropy and cross-entropy. Why do we use cross-entropy as a loss function in classification? How does it relate to information theory? Provide intuition and mathematical details.',
    sampleAnswer: `Entropy and cross-entropy connect information theory to machine learning. Understanding them provides deep insight into why certain loss functions work well for classification.

**Shannon Entropy: Measuring Uncertainty**

**Definition**:
H(X) = -Σ p(x) log₂(p(x))

**Intuition**: 
Entropy measures the average amount of information (in bits) needed to describe a random variable. Higher entropy = more uncertainty = more information required.

**Examples**:

\`\`\`python
def shannon_entropy(probs):
    """Calculate Shannon entropy (base 2 for bits)"""
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

# Example 1: Fair coin (maximum uncertainty for 2 outcomes)
fair_coin = np.array([0.5, 0.5])
H_fair = shannon_entropy(fair_coin)
print(f"Fair coin: H = {H_fair:.4f} bits")  # 1.0 bit

# You need 1 bit to represent: 0=heads, 1=tails

# Example 2: Biased coin (less uncertainty)
biased_coin = np.array([0.9, 0.1])
H_biased = shannon_entropy(biased_coin)
print(f"Biased coin: H = {H_biased:.4f} bits")  # 0.469 bits

# Less information needed since outcome is more predictable

# Example 3: Certain outcome (no uncertainty)
certain = np.array([1.0, 0.0])
H_certain = shannon_entropy(certain)
print(f"Certain: H = {H_certain:.4f} bits")  # 0 bits

# No information needed - outcome is known

# Example 4: Uniform distribution (maximum uncertainty)
uniform_4 = np.array([0.25, 0.25, 0.25, 0.25])
H_uniform_4 = shannon_entropy(uniform_4)
print(f"Uniform (4 outcomes): H = {H_uniform_4:.4f} bits")  # 2.0 bits

# Need 2 bits to represent 4 equally likely outcomes
\`\`\`

**Key Property**: 
For uniform distribution over n outcomes: H = log₂(n)

**Intuition**:
- Fair coin: 2 outcomes → 1 bit
- Fair 4-sided die: 4 outcomes → 2 bits
- Fair 8-sided die: 8 outcomes → 3 bits

**Cross-Entropy: Comparing Distributions**

**Definition**:
H(p, q) = -Σ p(x) log(q(x))

Where:
- p(x): true distribution
- q(x): predicted/approximate distribution

**Intuition**: 
Average number of bits needed to encode data from true distribution p using code optimized for distribution q.

**Relationship to Entropy**:
H(p, q) ≥ H(p, p) = H(p)

Equality holds only when q = p (perfect match).

**KL Divergence** (relative entropy):
D_KL(p || q) = H(p, q) - H(p) = Σ p(x) log(p(x)/q(x))

Measures "distance" from q to p (not symmetric).

\`\`\`python
def cross_entropy(p, q, epsilon=1e-10):
    """Cross-entropy between distributions p and q"""
    q = np.clip(q, epsilon, 1)  # Avoid log(0)
    return -np.sum(p * np.log(q))

def kl_divergence(p, q, epsilon=1e-10):
    """KL divergence from q to p"""
    return cross_entropy(p, q, epsilon) - shannon_entropy(p)

# True distribution
p_true = np.array([0.6, 0.3, 0.1])

# Perfect prediction
q_perfect = np.array([0.6, 0.3, 0.1])

# Good prediction
q_good = np.array([0.5, 0.35, 0.15])

# Bad prediction
q_bad = np.array([0.1, 0.1, 0.8])

print(f"Entropy H(p): {shannon_entropy(p_true):.4f}")
print(f"\\nCross-Entropy:")
print(f"  Perfect: {cross_entropy(p_true, q_perfect):.4f}")
print(f"  Good: {cross_entropy(p_true, q_good):.4f}")
print(f"  Bad: {cross_entropy(p_true, q_bad):.4f}")

print(f"\\nKL Divergence:")
print(f"  Perfect: {kl_divergence(p_true, q_perfect):.6f}")
print(f"  Good: {kl_divergence(p_true, q_good):.4f}")
print(f"  Bad: {kl_divergence(p_true, q_bad):.4f}")
\`\`\`

**Cross-Entropy as Loss Function**

In classification, we want to minimize distance between:
- p: true distribution (one-hot encoded labels)
- q: predicted distribution (model outputs)

**Binary Cross-Entropy**:
BCE = -Σ[y log(ŷ) + (1-y) log(1-ŷ)]

**Categorical Cross-Entropy** (multi-class):
CCE = -Σ Σ y_ij log(ŷ_ij)

\`\`\`python
# Binary classification example
def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1, 0])

# Confident correct predictions
y_pred_good = np.array([0.95, 0.05, 0.90, 0.85, 0.10])
loss_good = binary_cross_entropy(y_true, y_pred_good)

# Uncertain predictions
y_pred_uncertain = np.array([0.6, 0.4, 0.6, 0.55, 0.45])
loss_uncertain = binary_cross_entropy(y_true, y_pred_uncertain)

# Confident wrong predictions
y_pred_bad = np.array([0.1, 0.9, 0.2, 0.15, 0.85])
loss_bad = binary_cross_entropy(y_true, y_pred_bad)

print(f"Good predictions: Loss = {loss_good:.4f}")
print(f"Uncertain predictions: Loss = {loss_uncertain:.4f}")
print(f"Bad predictions: Loss = {loss_bad:.4f}")

# Visualize loss landscape
pred_range = np.linspace(0.01, 0.99, 100)
loss_when_true = -np.log(pred_range)  # y=1
loss_when_false = -np.log(1 - pred_range)  # y=0

plt.figure(figsize=(10, 6))
plt.plot(pred_range, loss_when_true, label='True label = 1', linewidth=2)
plt.plot(pred_range, loss_when_false, label='True label = 0', linewidth=2)
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 5)
\`\`\`

**Why Cross-Entropy for Classification?**

**1. Probabilistic Interpretation**:
- Minimizing cross-entropy = Maximizing likelihood
- Model outputs interpreted as probabilities
- Natural fit for classification

**Proof**:
Given data D and model parameters θ:
- Likelihood: L(θ) = Π p(yᵢ|xᵢ; θ)
- Log-likelihood: log L(θ) = Σ log p(yᵢ|xᵢ; θ)
- For classification: p(y=1|x) = ŷ
- Negative log-likelihood = Cross-entropy loss!

**2. Proper Gradients with Softmax/Sigmoid**:

With softmax + cross-entropy:
∂L/∂z = ŷ - y (simple!)

With softmax + MSE:
∂L/∂z = (ŷ - y) · ŷ · (1 - ŷ) (more complex, can vanish)

\`\`\`python
# Compare gradients
y_true = 1
predictions = np.linspace(0.01, 0.99, 100)

# Cross-entropy gradient magnitude: |ŷ - y|
ce_grad = np.abs(predictions - y_true)

# MSE gradient magnitude: |ŷ - y| · ŷ · (1 - ŷ)
mse_grad = np.abs(predictions - y_true) * predictions * (1 - predictions)

plt.figure(figsize=(10, 6))
plt.plot(predictions, ce_grad, label='Cross-Entropy', linewidth=2)
plt.plot(predictions, mse_grad, label='MSE', linewidth=2)
plt.xlabel('Predicted Probability (ŷ)')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Comparison: Cross-Entropy vs MSE')
plt.legend()
plt.grid(True)

# Notice: MSE gradient vanishes at extremes!
# Cross-entropy maintains strong gradient even when very wrong
\`\`\`

**3. Penalizes Confidence on Wrong Predictions**:

\`\`\`python
# When true label is 1:
for pred in [0.01, 0.1, 0.5, 0.9, 0.99]:
    ce_loss = -np.log(pred)
    mse_loss = (1 - pred)**2
    print(f"Pred={pred:.2f}: CE={ce_loss:.4f}, MSE={mse_loss:.4f}")

# Cross-entropy heavily penalizes confident wrong predictions
# MSE penalty is more uniform
\`\`\`

**Information Theory Connection**:

**Optimal Coding**: 
If event has probability p, optimal code length = -log₂(p) bits

**Example**:
- Event with p=0.5 → -log₂(0.5) = 1 bit
- Event with p=0.25 → -log₂(0.25) = 2 bits
- Rare events need more bits!

**Cross-entropy in ML**:
Model assigns probability ŷ to true event y.
- If model is confident and correct (ŷ ≈ 1 when y=1): low loss
- If model is confident and wrong (ŷ ≈ 0 when y=1): high loss

**Trading Application**:

\`\`\`python
# Predicting market direction
def train_direction_classifier(features, directions):
    """
    Use cross-entropy loss for binary direction classification
    directions: 1 = up, 0 = down
    """
    model = ...  # Your model
    
    # Cross-entropy loss
    def loss_fn(y_true, y_pred):
        return binary_cross_entropy(y_true, y_pred)
    
    # Train to minimize cross-entropy
    # Encourages confident predictions when pattern is clear
    # Uncertain predictions when market is ambiguous
    
    return model

# Entropy can also measure strategy diversity
def strategy_entropy(position_distribution):
    """
    High entropy = diversified positions
    Low entropy = concentrated positions
    """
    return shannon_entropy(position_distribution)

positions = np.array([0.4, 0.3, 0.2, 0.1])  # 4 assets
print(f"Portfolio entropy: {strategy_entropy(positions):.4f} bits")
\`\`\`

**Summary**:
- Entropy measures uncertainty/information content
- Cross-entropy compares two distributions
- Minimizing cross-entropy = Maximizing likelihood
- Natural loss for classification due to:
  - Probabilistic interpretation
  - Clean gradients with softmax
  - Appropriate penalty structure
- Information theory provides deep theoretical foundation for ML`,
    keyPoints: [
      'Entropy H(X) measures uncertainty: higher entropy = more information needed',
      'Cross-entropy H(p,q) measures cost of encoding p using code for q',
      'Minimizing cross-entropy = Maximizing likelihood (probabilistic interpretation)',
      'Cross-entropy + softmax gives clean gradient: ∂L/∂z = ŷ - y',
      'Heavily penalizes confident wrong predictions (important for classification)',
    ],
  },
];
