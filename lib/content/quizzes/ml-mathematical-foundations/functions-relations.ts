/**
 * Quiz questions for Functions & Relations section
 */

export const functionsrelationsQuiz = [
  {
    id: 'dq1-neural-networks-composition',
    question:
      'Explain why deep neural networks are fundamentally function compositions. How does this perspective help us understand backpropagation and the chain rule? Provide specific examples from a 3-layer network.',
    sampleAnswer: `Deep neural networks are literally compositions of functions, and this mathematical perspective is crucial for understanding how they work and learn:

**Function Composition in Neural Networks**:

A 3-layer network can be written as:
y = f₃(f₂(f₁(x)))

Where each layer fᵢ is itself a composition:
fᵢ(x) = σᵢ(Wᵢx + bᵢ)
- Linear transformation: Wᵢx + bᵢ  
- Activation function: σᵢ(·)

**Explicit Example**:
\`\`\`python
# Layer 1: Input (3 features) → Hidden (4 units)
def f1(x):
    z1 = W1 @ x + b1  # Linear: 4×3 @ 3×1 = 4×1
    a1 = relu (z1)      # Activation
    return a1

# Layer 2: Hidden (4) → Hidden (3 units)
def f2(x):
    z2 = W2 @ x + b2  # Linear: 3×4 @ 4×1 = 3×1
    a2 = relu (z2)      # Activation
    return a2

# Layer 3: Hidden (3) → Output (1 unit)
def f3(x):
    z3 = W3 @ x + b3  # Linear: 1×3 @ 3×1 = 1×1
    return z3          # No activation (regression)

# Complete network: function composition
def network (x):
    return f3(f2(f1(x)))
\`\`\`

**Connection to Chain Rule**:

The chain rule from calculus states:
d/dx[f (g(x))] = f'(g (x)) · g'(x)

For multiple compositions:
d/dx[f₃(f₂(f₁(x)))] = f₃'(f₂(f₁(x))) · f₂'(f₁(x)) · f₁'(x)

This IS backpropagation!

**Backpropagation Derivation**:

Given loss L = (y - ŷ)² where ŷ = f₃(f₂(f₁(x))):

1. **Output gradient**:
   ∂L/∂ŷ = 2(ŷ - y)

2. **Layer 3 gradient** (chain rule):
   ∂L/∂W₃ = ∂L/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂W₃
   where z₃ = W₃a₂ + b₃

3. **Layer 2 gradient** (chain continues):
   ∂L/∂W₂ = ∂L/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂W₂

4. **Layer 1 gradient** (full chain):
   ∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁

Notice the pattern: we multiply gradients flowing backward through each function in the composition.

**Why This Perspective Matters**:

1. **Vanishing Gradients**: 
   - Chain rule multiplies many terms
   - If σ'(x) < 1 for many layers, product → 0
   - Deep networks: more compositions = more multiplications
   - Solution: Better activations (ReLU), normalization

2. **Exploding Gradients**:
   - If some σ'(x) > 1, product → ∞
   - Solution: Gradient clipping, careful initialization

3. **Skip Connections** (ResNets):
   - Instead of f₃(f₂(f₁(x))), use f₃(f₂(f₁(x)) + x)
   - Creates additional paths in the chain rule
   - Helps gradients flow directly backward

4. **Automatic Differentiation**:
   - PyTorch/TensorFlow build computational graphs
   - Each node is a function in the composition
   - Chain rule applied automatically by traversing graph backward

**Practical Implications**:

Understanding networks as function compositions helps you:
- Debug gradient flow issues
- Design better architectures
- Understand why certain activation functions work better
- Appreciate why depth (more compositions) can be powerful but also challenging

**Trading Context**:
When building trading models with neural networks:
- More layers = more complex feature transformations (compositions)
- But deeper networks may overfit or have unstable gradients
- Balance complexity with stability
- Monitor gradient magnitudes during training`,
    keyPoints: [
      'Neural networks are literal function compositions: f₃(f₂(f₁(x)))',
      'Backpropagation is the chain rule applied to function composition',
      "Each layer's gradient involves product of all subsequent derivatives",
      'Vanishing/exploding gradients result from multiplying many terms',
      'Skip connections provide alternative paths in the composition chain',
    ],
  },
  {
    id: 'dq2-activation-functions',
    question:
      'Why do we need activation functions in neural networks? Compare sigmoid, tanh, and ReLU - discuss their mathematical properties, advantages, disadvantages, and when to use each in practical ML applications.',
    sampleAnswer: `Activation functions are essential because they introduce non-linearity into neural networks. Without them, deep networks would be equivalent to a single linear transformation:

**Why Non-Linearity is Necessary**:

Consider a 2-layer network WITHOUT activation functions:
h = W₁x + b₁
y = W₂h + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)

This simplifies to y = W'x + b', a single linear transformation!

No matter how many layers you stack, without activation functions, the network can only learn linear relationships. Most real-world patterns (images, text, trading patterns) are highly non-linear.

**Sigmoid Function: σ(x) = 1/(1 + e⁻ˣ)**

**Properties**:
- Range: (0, 1)
- Smooth, continuously differentiable
- Derivative: σ'(x) = σ(x)(1 - σ(x))
- S-shaped curve

**Advantages**:
- Output interpretable as probability
- Smooth gradients
- Historically significant (early neural networks)
- Perfect for binary classification output layer

**Disadvantages**:
- Vanishing gradient problem: σ'(x) ≈ 0 for |x| > 4
  - Max derivative is 0.25 (at x=0)
  - Deep networks: multiplying many 0.25s → gradient → 0
- Not zero-centered (outputs always positive)
- Expensive computation (exponential)

**When to use**:
- OUTPUT layer for binary classification
- AVOID in hidden layers of deep networks

**Tanh Function: tanh (x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)**

**Properties**:
- Range: (-1, 1)
- Zero-centered
- Derivative: tanh'(x) = 1 - tanh²(x)
- Similar shape to sigmoid, but symmetric

**Advantages**:
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid (max derivative = 1)
- Symmetric around origin

**Disadvantages**:
- Still suffers from vanishing gradient for |x| > 2
- Expensive computation

**When to use**:
- Hidden layers when you need zero-centered activations
- RNNs/LSTMs (historically common)
- Better than sigmoid but worse than ReLU for deep networks

**ReLU Function: ReLU(x) = max(0, x)**

**Properties**:
- Range: [0, ∞)
- Piecewise linear
- Derivative: 1 if x > 0, else 0
- Non-differentiable at x=0 (but we use subgradient)

**Advantages**:
- Computationally cheap (just comparison and multiplication)
- No vanishing gradient for x > 0
- Sparse activation (about 50% of neurons are zero)
- Empirically works very well
- Faster convergence than sigmoid/tanh

**Disadvantages**:
- Not zero-centered
- "Dying ReLU" problem: if neuron outputs 0, gradient is 0, it never recovers
  - Happens with large learning rates
  - Neuron gets "stuck" at 0 forever
- Unbounded output (can lead to numerical issues)

**When to use**:
- DEFAULT choice for hidden layers in deep networks
- Computer vision models
- Most modern architectures

**Comparison Summary**:

| Property | Sigmoid | Tanh | ReLU |
|----------|---------|------|------|
| Range | (0,1) | (-1,1) | [0,∞) |
| Zero-centered | ❌ | ✅ | ❌ |
| Vanishing gradient | ✅ Bad | ✅ Bad | ❌ Good |
| Computation | Slow | Slow | Fast |
| Sparse activation | ❌ | ❌ | ✅ |
| Dead neurons | ❌ | ❌ | ✅ Possible |

**Modern Variants**:

1. **Leaky ReLU**: max(0.01x, x)
   - Fixes dying ReLU (small gradient for x < 0)

2. **ELU**: x if x>0, else α(eˣ-1)
   - Smooth, zero-centered mean

3. **GELU**: x·Φ(x) (Gaussian error linear unit)
   - Used in transformers (BERT, GPT)

**Practical Recommendations**:

**For hidden layers**:
- Start with ReLU (default)
- If dying ReLU occurs: try Leaky ReLU or ELU
- For transformers/NLP: consider GELU

**For output layers**:
- Binary classification: Sigmoid
- Multi-class classification: Softmax
- Regression: Linear (no activation) or ReLU (if output ≥ 0)

**Trading Application Example**:
\`\`\`python
# Predicting stock returns (can be positive or negative)
# Hidden layers: ReLU for efficiency
# Output: Linear or tanh (symmetric around 0)

model = nn.Sequential(
    nn.Linear (features, 64),
    nn.ReLU(),              # Hidden layer 1
    nn.Linear(64, 32),
    nn.ReLU(),              # Hidden layer 2
    nn.Linear(32, 1),       # Output
    nn.Tanh()               # Symmetric output for returns
)
\`\`\`

**Key Insight**: ReLU's success isn't just mathematical—it's empirical. Despite theoretical disadvantages (not zero-centered, unbounded), it works remarkably well in practice due to computational efficiency and sparse representations.`,
    keyPoints: [
      'Activation functions provide non-linearity; without them, deep networks = single linear layer',
      'Sigmoid: good for output probabilities, bad for hidden layers (vanishing gradient)',
      'Tanh: better than sigmoid (zero-centered), still has vanishing gradient',
      'ReLU: default choice, fast, no vanishing gradient, watch for dying ReLU',
      'Choose activation based on layer type and problem requirements',
    ],
  },
  {
    id: 'dq3-loss-functions',
    question:
      'Loss functions are special functions in ML that measure prediction error. Explain the mathematical properties of mean squared error (MSE) and cross-entropy loss. Why is MSE used for regression and cross-entropy for classification? How do their derivatives influence gradient descent?',
    sampleAnswer: `Loss functions quantify how wrong our model's predictions are. Their mathematical properties directly affect training dynamics and model convergence:

**Mean Squared Error (MSE) - Regression**

**Formula**:
MSE = (1/n) Σ(yᵢ - ŷᵢ)²

**Mathematical Properties**:

1. **Always non-negative**: (y - ŷ)² ≥ 0
2. **Minimum at y = ŷ**: Perfect predictions give MSE = 0
3. **Convex** for linear models: Single global minimum
4. **Differentiable** everywhere: Smooth optimization
5. **Symmetric**: Overestimation and underestimation penalized equally

**Derivative**:
∂MSE/∂ŷ = (2/n) Σ(ŷᵢ - yᵢ)

The gradient is **linear** in the error: If error is large, gradient is large (fast updates). If error is small, gradient is small (slow updates).

**Why MSE for Regression**:

1. **Gaussian assumption**: MSE assumes errors follow normal distribution
   - Maximizing likelihood under Gaussian noise = minimizing MSE
   
2. **Penalizes large errors**: Quadratic term heavily penalizes outliers
   - Error of 2 is 4x worse than error of 1 (2² vs 1²)
   
3. **Smooth gradients**: Easy to optimize with gradient descent

4. **Interpretable**: In same units as target variable squared

**Example**:
\`\`\`python
def mse_loss (y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_gradient (y_true, y_pred):
    return 2 * (y_pred - y_true) / len (y_true)

# Price prediction
y_true = np.array([100, 150, 200])
y_pred = np.array([110, 140, 210])

loss = mse_loss (y_true, y_pred)
grad = mse_gradient (y_true, y_pred)

print(f"MSE Loss: {loss}")  # 66.67
print(f"Gradient: {grad}")  # [6.67, -6.67, 6.67]
# Notice: Large errors → large gradients
\`\`\`

**Cross-Entropy Loss - Classification**

**Binary Cross-Entropy**:
BCE = -(1/n) Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]

Where y ∈ {0, 1} and ŷ ∈ (0, 1)

**Categorical Cross-Entropy** (multi-class):
CCE = -(1/n) Σ Σ yᵢⱼlog(ŷᵢⱼ)

Where yᵢⱼ is one-hot encoded

**Mathematical Properties**:

1. **Always non-negative**: -log (p) ≥ 0 for p ∈ (0,1)
2. **Asymmetric penalty**: 
   - Predicting 0.01 when truth is 1: Loss ≈ 4.6
   - Predicting 0.99 when truth is 0: Loss ≈ 4.6
   - But predicting 0.5 when truth is 1: Loss ≈ 0.69
3. **Convex** for logistic regression
4. **Unbounded**: As ŷ → 0 when y=1, loss → ∞

**Derivative** (binary case with sigmoid):
∂BCE/∂z = ŷ - y

Where z is pre-activation (logit) and ŷ = sigmoid (z)

**Remarkable property**: The gradient simplifies to just the error!

**Why Cross-Entropy for Classification**:

1. **Probabilistic interpretation**: 
   - Minimizing cross-entropy = maximizing likelihood
   - ŷ represents probability distribution
   - Measures "distance" between true and predicted distributions

2. **Handles probabilities correctly**:
   - If y=1, only log(ŷ) matters → encourages ŷ → 1
   - If y=0, only log(1-ŷ) matters → encourages ŷ → 0

3. **Better gradients for classification**:
   - MSE + sigmoid leads to flat gradients when very wrong
   - Cross-entropy + sigmoid gives gradient proportional to error

4. **Penalizes confidence on wrong predictions**:
   - Being confidently wrong (ŷ=0.99 when y=0) is heavily penalized

**Example**:
\`\`\`python
def binary_cross_entropy (y_true, y_pred, epsilon=1e-10):
    # Clip to avoid log(0)
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    return -np.mean (y_true * np.log (y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

# Classification predictions
y_true = np.array([1, 0, 1, 1, 0])
y_pred_good = np.array([0.9, 0.1, 0.8, 0.85, 0.15])
y_pred_bad = np.array([0.6, 0.4, 0.6, 0.55, 0.45])

loss_good = binary_cross_entropy (y_true, y_pred_good)
loss_bad = binary_cross_entropy (y_true, y_pred_bad)

print(f"Good predictions loss: {loss_good:.4f}")  # ~0.15
print(f"Bad predictions loss: {loss_bad:.4f}")    # ~0.62

# Notice: Confident correct predictions heavily rewarded
\`\`\`

**Why NOT MSE for Classification**:

With sigmoid + MSE:
- Derivative: σ'(z) · (ŷ - y)
- Problem: σ'(z) ≈ 0 when z is very large/small (saturated)
- If model is very wrong (z >> 0 when y=0), σ'(z) ≈ 0 → gradient ≈ 0
- Learning stalls even though error is large!

With sigmoid + cross-entropy:
- Derivative simplifies to just (ŷ - y)
- No saturation problem
- Large error → large gradient → fast learning

**Gradient Descent Dynamics**:

**MSE**:
\`\`\`python
# Update rule
θ_new = θ_old - α · (2/n) Σ(ŷ - y) · ∂ŷ/∂θ

# Linear gradient in error
# Far from optimum → large gradient → big steps
# Near optimum → small gradient → small steps (good!)
\`\`\`

**Cross-Entropy**:
\`\`\`python
# Update rule (with softmax)
θ_new = θ_old - α · (1/n) Σ(ŷ - y)

# Also linear in error, but
# Logarithmic penalty encourages extreme probabilities (0 or 1)
# Better for classification where we want confident decisions
\`\`\`

**Trading Application**:

**Regression (price prediction)**:
\`\`\`python
# Predicting stock price: Use MSE
loss = mse_loss (actual_prices, predicted_prices)
# Treats $10 error on $100 stock same as $10 error on $1000 stock
\`\`\`

**Classification (trade direction)**:
\`\`\`python
# Predicting up/down: Use cross-entropy
loss = binary_cross_entropy (actual_direction, predicted_prob)
# Heavily penalizes confident wrong predictions
# In trading, being confidently wrong is especially costly!
\`\`\`

**Advanced**: For trading, you might use custom losses:
\`\`\`python
def asymmetric_mse (y_true, y_pred):
    """Penalize underestimating risk more than overestimating"""
    error = y_pred - y_true
    return np.mean (np.where (error > 0, error**2, 2 * error**2))
\`\`\`

**Summary**:
- MSE: Regression, Gaussian assumption, quadratic penalty, symmetric
- Cross-Entropy: Classification, probabilistic, logarithmic penalty, matches sigmoid/softmax
- Derivatives determine learning speed and stability
- Choice of loss function should match problem structure`,
    keyPoints: [
      'MSE: Quadratic penalty, symmetric, convex for linear models, linear gradient',
      'Cross-Entropy: Logarithmic penalty, probabilistic interpretation, unbounded',
      'MSE + sigmoid has vanishing gradient problem for classification',
      'Cross-Entropy + sigmoid derivative simplifies to (ŷ - y)',
      'Choose loss function based on problem: regression → MSE, classification → cross-entropy',
    ],
  },
];
