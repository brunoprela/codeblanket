/**
 * Quiz questions for Numerical Optimization Methods section
 */

export const numericaloptimizationQuiz = [
  {
    id: 'numopt-disc-1',
    question:
      'Discuss the trade-offs between full-batch gradient descent and mini-batch SGD. Why is SGD preferred for deep learning despite noisier gradients?',
    hint: 'Consider computational cost per iteration, convergence properties, generalization, and implicit regularization.',
    sampleAnswer: `**Full-Batch GD vs Mini-Batch SGD:**

**1. Computational Cost:**

**Full-Batch GD:**
\`\`\`
Cost per iteration = O(n·d)
where n = dataset size, d = model complexity

For n = 1M images, d = ResNet-50 forward pass:
One iteration ≈ seconds to minutes
\`\`\`

**Mini-Batch SGD:**
\`\`\`
Cost per iteration = O(B·d)
where B = batch size (typically 32-256)

Same model, B = 64:
One iteration ≈ milliseconds
Speedup: ~15,000×!
\`\`\`

**Key insight:** SGD does many cheap updates instead of few expensive ones.

**2. Convergence Properties:**

**Full-Batch GD:**
- Smooth, deterministic convergence
- Guaranteed descent: f(x_{k+1}) < f(x_k)
- Convergence rate: O(1/k) for convex, O(exp(-k)) for strongly convex
- Finds precise minimum

**Mini-Batch SGD:**
- Noisy, stochastic convergence
- May increase loss temporarily
- Expected descent: E[f(x_{k+1})] < f(x_k)
- Oscillates around minimum
- Slower final convergence

**3. Why SGD Still Preferred:**

**A) Computational Efficiency:**

Total work to reach ε accuracy:

**Full-Batch:**
\`\`\`
Iterations needed: K ~ 1/ε
Cost: K × n = n/ε
\`\`\`

**SGD:**
\`\`\`
Iterations needed: K ~ 1/ε²
Cost: K × B = B/ε²
\`\`\`

For large n >> B, SGD wins despite slower convergence!

**Example:**
\`\`\`
n = 1M, B = 64, target ε = 0.01

Full-batch: 1M / 0.01 = 100M operations
SGD: 64 / 0.01² = 640K operations
→ SGD ~156× faster to ε-accuracy!
\`\`\`

**B) Generalization Benefits:**

**Flat Minima:**
SGD noise helps escape sharp minima and find flat minima.

\`\`\`
Sharp minimum: Small perturbation → large loss increase
Flat minimum: Perturbations don't hurt much
\`\`\`

Flat minima generalize better to test data!

**Evidence:**
\`\`\`python
# Measure sharpness (max Hessian eigenvalue)
model_gd = train_full_batch(data)
model_sgd = train_mini_batch(data)

sharpness_gd = max_eigenvalue(hessian(model_gd))
sharpness_sgd = max_eigenvalue(hessian(model_sgd))

print(f"GD sharpness: {sharpness_gd:.2f}")    # Higher
print(f"SGD sharpness: {sharpness_sgd:.2f}")  # Lower

test_acc_gd = evaluate(model_gd, test_data)
test_acc_sgd = evaluate(model_sgd, test_data)

print(f"GD test accuracy: {test_acc_gd:.2%}")   # Lower
print(f"SGD test accuracy: {test_acc_sgd:.2%}") # Higher!
\`\`\`

**C) Implicit Regularization:**

SGD noise acts as implicit regularizer:
- Prevents memorization of training data
- Encourages simpler solutions
- Similar to adding noise to weights

**D) Escaping Saddle Points:**

High-dimensional optimization: saddle points prevalent

**Full-Batch GD:**
- Can get stuck near saddle points (gradient ≈ 0)
- Slow escape (requires tiny gradient component)

**SGD:**
- Noise perturbs away from saddles
- Faster escape

**4. Practical Considerations:**

**Batch Size Effects:**

**Small batches (B=32):**
- More noise → better generalization
- Slower convergence
- Less parallelism

**Large batches (B=8192):**
- Less noise → sharper minima
- Faster convergence per epoch
- Better hardware utilization
- May need learning rate scaling

**Modern practice:** B=64-256 good balance

**Learning Rate Schedules:**

SGD benefits from decaying learning rate:
\`\`\`
α_t = α_0 / (1 + decay × t)
\`\`\`

Reduces noise as training progresses.

**5. Hybrid Approaches:**

**Variance Reduction:**
Methods like SVRG reduce SGD variance while keeping low cost:
\`\`\`
Occasionally compute full gradient
Use it to correct mini-batch gradients
\`\`\`

**Large Batch Training:**
Scale learning rate with batch size:
\`\`\`
α = α_base × (B / B_base)
\`\`\`

Enables efficient distributed training.

**6. Comparison Table:**

| Aspect | Full-Batch GD | Mini-Batch SGD |
|--------|---------------|----------------|
| **Cost/iter** | O(n) | O(B) |
| **Convergence** | Smooth | Noisy |
| **To ε-accuracy** | O(n/ε) | O(B/ε²) |
| **Generalization** | Worse (sharp) | Better (flat) |
| **Parallelism** | Limited | High |
| **Memory** | Need full data | Need only batch |
| **Use case** | Small datasets | Large datasets |

**7. When to Use Each:**

**Full-Batch GD:**
- Small datasets (n < 10K)
- Convex optimization
- High precision required
- Determinism important

**Mini-Batch SGD:**
- Large datasets (n > 100K)
- Deep learning
- Good enough solution acceptable
- Computational efficiency critical

**8. Summary:**

**Why SGD preferred for deep learning:**

1. **Computational efficiency**: ~100-1000× faster per epoch
2. **Generalization**: Noise finds flat minima
3. **Scalability**: Constant memory per iteration
4. **Practical success**: Empirically works extremely well

**Trade-off:**
- Sacrifice smooth convergence and final precision
- Gain computational efficiency and generalization

**Key Insight:**

For deep learning, getting to "good enough" solution quickly matters more than converging to exact minimum slowly. SGD's noise is a feature, not a bug - it implicitly regularizes and helps generalization.

**Practical takeaway:**
Almost all modern deep learning uses mini-batch SGD (or variants like Adam that build on SGD). Full-batch gradient descent is rarely used except for small-scale convex optimization.`,
    keyPoints: [
      'SGD: O(B) cost per iteration vs GD: O(n)',
      'SGD reaches ε-accuracy faster in wall-clock time despite slower convergence',
      'SGD noise finds flat minima → better generalization',
      'SGD implicitly regularizes, prevents overfitting',
      'Batch size trade-off: small (generalization) vs large (speed)',
      'Modern deep learning: almost exclusively mini-batch SGD',
    ],
  },
  {
    id: 'numopt-disc-2',
    question:
      'Explain how Adam optimizer works and why it has become the default choice for training deep neural networks.',
    hint: 'Consider adaptive learning rates, momentum, bias correction, and practical performance.',
    sampleAnswer: `**Adam Optimizer: Adaptive Moment Estimation**

Adam is the most widely used optimizer for deep learning, combining the best features of momentum and adaptive learning rates.

**1. Core Mechanism:**

**Update equations:**
\`\`\`
m_t = β₁·m_{t-1} + (1-β₁)·g_t         # First moment (momentum)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²        # Second moment (variance)
m̂_t = m_t / (1 - β₁^t)                # Bias-corrected first moment
v̂_t = v_t / (1 - β₂^t)                # Bias-corrected second moment
θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)   # Parameter update
\`\`\`

**Default hyperparameters:**
- α = 0.001 (learning rate)
- β₁ = 0.9 (first moment decay)
- β₂ = 0.999 (second moment decay)
- ε = 10⁻⁸ (numerical stability)

**2. Key Components:**

**A) First Moment (Momentum):**

Like momentum in physics:
\`\`\`
m_t = 0.9·m_{t-1} + 0.1·g_t
\`\`\`

**Benefits:**
- Smooths noisy gradients
- Accelerates in consistent directions
- Dampens oscillations

**Example:**
\`\`\`python
# Gradient oscillates: [1, -1, 1, -1, ...]
# Without momentum: Updates oscillate
# With momentum: Averages to ~0, reduces oscillation
\`\`\`

**B) Second Moment (Adaptive LR):**

Tracks gradient variance:
\`\`\`
v_t = 0.999·v_{t-1} + 0.001·g_t²
\`\`\`

**Intuition:**
- If parameter has large gradients → large v_t → small effective LR
- If parameter has small gradients → small v_t → large effective LR

**Effect:** Per-parameter adaptive learning rates!

**C) Bias Correction:**

**Problem:** m_t and v_t initialized at 0, biased toward zero early in training.

**Solution:**
\`\`\`
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
\`\`\`

**Example:**
\`\`\`
t=1: 1 - 0.9¹ = 0.1  → divide by 0.1 (10× correction)
t=2: 1 - 0.9² = 0.19 → divide by 0.19 (5.3× correction)
t=10: 1 - 0.9¹⁰ = 0.65 → divide by 0.65 (1.5× correction)
t→∞: 1 - 0.9^t → 1 → no correction needed
\`\`\`

Ensures unbiased estimates from the start!

**3. Why Adam is Default:**

**A) Robust to Hyperparameters:**

Works well with default values:
\`\`\`python
# Usually just tune learning rate
optimizer = Adam(lr=0.001)  # Often works!

# Compare to SGD:
optimizer = SGD(lr=???, momentum=???)  # Need to tune both
\`\`\`

**B) Handles Sparse Gradients:**

Adaptive LR helps with sparse features (NLP, recommender systems):
\`\`\`
# Word embeddings: most gradients are zero
# Adam: Large LR for rare words (small v_t)
# SGD: Same LR for all → slow learning for rare words
\`\`\`

**C) Fast Convergence:**

Combines benefits of momentum + adaptive LR:
\`\`\`
Typical training:
- Adam: 50 epochs to converge
- SGD+momentum: 150 epochs
- Plain SGD: 500+ epochs
\`\`\`

**D) Less Sensitive to Learning Rate:**

Works across wide range of α:
\`\`\`
SGD: α too large → divergence, α too small → slow
Adam: Adaptive per-parameter → more forgiving
\`\`\`

**4. Intuitive Understanding:**

**Analogy:** Driving a car with adaptive cruise control

**SGD:**
- Fixed speed (learning rate)
- Hits bumps (noisy gradients) → jerky ride

**SGD + Momentum:**
- Smooth acceleration/deceleration
- Still fixed target speed

**Adam:**
- Smooth acceleration (momentum)
- Adaptive speed per terrain (adaptive LR)
- Speed up on highway (consistent gradient)
- Slow down on bumpy roads (high variance)

**5. Practical Implementation:**

\`\`\`python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize moments
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
\`\`\`

**6. When Adam Excels:**

**Best for:**
- Deep neural networks
- Sparse data (NLP, recommendations)
- First try / prototyping
- Limited hyperparameter tuning time

**Example use cases:**
- Transformers (BERT, GPT)
- Image classification (ResNet, ViT)
- GANs
- Reinforcement learning

**7. Limitations:**

**A) Generalization:**

Sometimes SGD+momentum generalizes better:
\`\`\`
Adam: Fast convergence, potentially sharp minima
SGD: Slow convergence, flatter minima → better test accuracy
\`\`\`

**Solution:** Switch to SGD for final epochs

**B) Learning Rate Decay:**

Adam less sensitive but still benefits:
\`\`\`python
# Cosine annealing
lr_t = lr_initial * 0.5 * (1 + cos(π * t / T))
\`\`\`

**C) Weight Decay:**

Original Adam has issues with weight decay. Use AdamW:
\`\`\`python
# Adam: weight_decay applies to gradients
# AdamW: weight_decay applies to weights directly
optimizer = AdamW(lr=0.001, weight_decay=0.01)
\`\`\`

**8. Comparison Table:**

| Optimizer | Speed | Robustness | Generalization | Use Case |
|-----------|-------|------------|----------------|----------|
| **SGD** | Slow | Low | Best | Simple problems, final tuning |
| **SGD+Mom** | Medium | Medium | Good | CNNs, careful tuning |
| **Adam** | **Fast** | **High** | Good | **Default choice** |
| **AdamW** | Fast | High | **Better** | **Modern best practice** |

**9. Modern Variants:**

**AdamW:** Decoupled weight decay (current best practice)
**RAdam:** Rectified Adam (fixes early training issues)
**Lookahead:** Wraps Adam for better generalization
**LAMB:** Large batch training

**10. Summary:**

**Why Adam is default:**

1. **Fast convergence**: Momentum + adaptive LR
2. **Robust**: Works with default hyperparameters
3. **Adaptive**: Per-parameter learning rates
4. **Handles sparsity**: Great for NLP, recommendations
5. **Easy to use**: Usually just tune learning rate
6. **Empirically successful**: Powers modern deep learning

**Best practice:**
\`\`\`python
# Start with Adam/AdamW
optimizer = AdamW(lr=0.001, weight_decay=0.01)

# If performance critical, try:
# 1. Learning rate schedule
# 2. Warmup (linear increase for first N steps)
# 3. Switch to SGD for final epochs (better generalization)
\`\`\`

**Key insight:** Adam's combination of momentum and adaptive learning rates makes it remarkably robust across diverse architectures and datasets, explaining its status as the default optimizer in modern deep learning.`,
    keyPoints: [
      'Combines momentum (first moment) with adaptive LR (second moment)',
      'Bias correction ensures unbiased estimates early in training',
      'Per-parameter learning rates adapt to gradient statistics',
      'Robust to hyperparameters, works with defaults',
      'Handles sparse gradients well (NLP, recommendations)',
      'Default choice for modern deep learning',
    ],
  },
  {
    id: 'numopt-disc-3',
    question:
      "Compare and contrast first-order methods (gradient descent) with second-order methods (Newton's method) for optimization. Why aren't second-order methods used more in deep learning?",
    hint: 'Consider computational cost, convergence rate, memory requirements, and scalability.',
    sampleAnswer: `**First-Order vs Second-Order Optimization Methods**

Understanding the trade-offs between these method families is crucial for choosing the right optimization approach.

**1. Mathematical Foundation:**

**First-Order (Gradient Descent):**
\`\`\`
Uses: f(x), ∇f(x)
Update: x_{k+1} = x_k - α∇f(x_k)
Information: Direction of steepest descent
\`\`\`

**Second-Order (Newton's Method):**
\`\`\`
Uses: f(x), ∇f(x), ∇²f(x)
Update: x_{k+1} = x_k - [∇²f(x_k)]⁻¹∇f(x_k)
Information: Direction + curvature
\`\`\`

**2. Convergence Analysis:**

**Gradient Descent:**
- **Rate**: O(1/k) for convex, O(exp(-k)) for strongly convex
- **Dependency**: Condition number κ = λ_max/λ_min affects speed
- **Ill-conditioned**: Very slow (zigzagging)

**Newton's Method:**
- **Rate**: O(exp(-2^k)) (quadratic convergence)
- **Near optimum**: Doubling precision each iteration!
- **Independent**: Doesn't depend on condition number

**Example:**
\`\`\`
Target accuracy: ε = 10⁻⁸

GD: ~10⁸ iterations (if κ = 10⁴)
Newton: ~5 iterations (near optimum)
\`\`\`

**3. Computational Cost:**

**Per Iteration:**

**Gradient Descent:**
\`\`\`
- Gradient computation: O(n·d)
  where n = data size, d = parameters
- Memory: O(d)
- Total: O(n·d)
\`\`\`

**Newton's Method:**
\`\`\`
- Gradient: O(n·d)
- Hessian: O(n·d²)  ← expensive!
- Inversion: O(d³)   ← very expensive!
- Memory: O(d²)      ← huge!
- Total: O(n·d² + d³)
\`\`\`

**Deep learning example:**
\`\`\`
ResNet-50: d = 25 million parameters

Gradient: 25M memory
Hessian: 25M × 25M = 625 trillion entries!
         ≈ 5 petabytes (in float64)!
         
Inversion: (25M)³ operations ≈ impossible!
\`\`\`

**4. Practical Comparison:**

**For small problems (d < 1000):**

\`\`\`python
import numpy as np
from time import time

d = 100  # Small dimension

# Gradient descent
def gd_iteration(x, grad):
    return 0.01 * grad  # O(d)

# Newton iteration  
def newton_iteration(x, grad, hessian):
    return np.linalg.solve(hessian, grad)  # O(d³)

grad = np.random.randn(d)
hessian = np.random.randn(d, d)

# Time comparison
t0 = time()
for _ in range(1000):
    update = gd_iteration(x, grad)
gd_time = time() - t0

t0 = time()
for _ in range(1000):
    update = newton_iteration(x, grad, hessian)
newton_time = time() - t0

print(f"GD: {gd_time:.3f}s, Newton: {newton_time:.3f}s")
print(f"Newton {newton_time/gd_time:.1f}× slower per iteration")
\`\`\`

Output:
\`\`\`
GD: 0.002s, Newton: 0.050s
Newton 25× slower per iteration
\`\`\`

**5. Why Second-Order Methods Not Used in DL:**

**A) Computational Intractability:**

\`\`\`
Modern models:
- GPT-3: 175 billion parameters
- Hessian: 175B × 175B matrix
- Storage: 30 septillion entries!
- Physically impossible to store
\`\`\`

**B) Memory Requirements:**

\`\`\`
Even modest network:
- Parameters: 1M
- Gradient: 4 MB (float32)
- Hessian: 4 TB (!!)
- GPU memory: typically 12-80 GB
- → Doesn't fit!
\`\`\`

**C) Non-Convexity:**

Neural networks: highly non-convex
- Newton's method: designed for convex optimization
- Saddle points: Newton can converge to saddles
- Negative curvature: Hessian not positive definite
- Catastrophic steps possible

**D) Stochasticity:**

Deep learning uses mini-batches:
- Hessian estimation: extremely noisy
- Need large batches for accurate Hessian
- → Defeats purpose of stochasticity

**6. Quasi-Newton Methods:**

Attempt to get benefits without full cost:

**BFGS (Broyden-Fletcher-Goldfarb-Shanno):**
\`\`\`
- Approximate Hessian inverse from gradients
- O(d²) memory (still too much for DL)
- Effective for d < 10,000
\`\`\`

**L-BFGS (Limited-memory BFGS):**
\`\`\`
- Store only m recent gradient pairs (m ≈ 10)
- O(m·d) memory ← much better!
- Used for some ML applications
\`\`\`

**Why L-BFGS not standard in DL:**
- Still expensive for very large d
- Doesn't leverage GPU parallelism well
- Not compatible with mini-batch SGD
- Works better for convex problems

**7. Approximations in Deep Learning:**

**A) Diagonal Hessian (AdaGrad, RMSProp, Adam):**

Instead of full Hessian:
\`\`\`
Hessian: O(d²) entries
Diagonal: O(d) entries ← feasible!

Adam uses diagonal second moment:
v_t ≈ diag(Hessian)
\`\`\`

**Trade-off:** Fast but ignores correlations

**B) K-FAC (Kronecker-Factored Approximate Curvature):**

Exploit structure in neural networks:
\`\`\`
Hessian ≈ A ⊗ B
where A and B are much smaller matrices

Cost: O(d^{1.5}) instead of O(d³)
\`\`\`

Still not mainstream due to complexity.

**C) Natural Gradient:**

Uses Fisher information matrix:
\`\`\`
Update: θ_{t+1} = θ_t - α F⁻¹∇L
where F = E[∇log p · ∇log p^T]
\`\`\`

More practical than Hessian, but still expensive.

**8. When to Use Each:**

**Use First-Order (GD, Adam):**
- Large-scale deep learning (d > 10⁶)
- Stochastic mini-batch training
- GPU acceleration important
- Non-convex optimization
- **→ 99% of deep learning**

**Use Second-Order (Newton, L-BFGS):**
- Small to medium scale (d < 10⁴)
- Convex or nearly convex
- Deterministic (full-batch)
- High precision required
- **→ Traditional ML, scientific computing**

**9. Comparison Table:**

| Aspect | First-Order | Second-Order |
|--------|-------------|--------------|
| **Cost/iter** | O(d) | O(d³) |
| **Memory** | O(d) | O(d²) |
| **Convergence** | Linear | Quadratic |
| **Iterations** | Many | Few |
| **Total cost** | **Often lower** | Often higher |
| **Scalability** | ✓ Millions of params | ✗ Thousands max |
| **GPU-friendly** | ✓ Yes | ✗ No |
| **Stochastic** | ✓ Yes | ✗ Difficult |
| **Non-convex** | ✓ Robust | ⚠ Can fail |

**10. The Paradox:**

**Newton's method:**
- Fewer iterations (5-10)
- But each iteration extremely expensive
- Total time: Usually worse for large d

**Gradient descent:**
- Many iterations (1000s)
- But each iteration cheap
- Total time: Usually better for large d

**Example:**
\`\`\`
Problem: d = 1M parameters

Newton:
- 10 iterations × 10 hours/iter = 100 hours

GD/Adam:
- 10,000 iterations × 0.01 hours/iter = 100 hours

But Newton requires 5 PB memory → impossible!
GD requires 4 MB → feasible!
\`\`\`

**11. Future Directions:**

**Hybrid approaches:**
- Start with first-order (fast initial progress)
- Switch to second-order near optimum (fast convergence)

**Randomized methods:**
- Stochastic Hessian estimation
- Sub-sampled Newton methods

**Hardware:**
- TPUs optimized for matrix operations
- Could make second-order more viable

**12. Summary:**

**Why first-order dominates deep learning:**

1. **Scalability**: O(d) vs O(d³) - critical for millions/billions of parameters
2. **Memory**: O(d) vs O(d²) - Hessian doesn't fit in memory
3. **Stochasticity**: Compatible with mini-batch SGD
4. **GPU-friendly**: Highly parallelizable operations
5. **Non-convexity**: More robust to saddle points
6. **Empirical success**: Proven to work at scale

**When second-order wins:**
- Small, convex problems
- High-precision requirements
- Scientific computing

**Key insight:** In deep learning, the ability to take many cheap steps (first-order) outweighs the benefit of taking few expensive steps (second-order). The curse of dimensionality makes second-order methods computationally intractable for modern neural networks.`,
    keyPoints: [
      'First-order: O(d) per iteration, many iterations needed',
      'Second-order: O(d³) per iteration, few iterations needed',
      'Deep learning: d = millions → Hessian storage impossible',
      'Mini-batch SGD incompatible with accurate Hessian estimation',
      'First-order + adaptive LR (Adam) good enough in practice',
      'Second-order reserved for small-scale convex optimization',
    ],
  },
];
