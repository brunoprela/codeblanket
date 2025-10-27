/**
 * Quiz questions for Limits & Continuity section
 */

export const limitscontinuityQuiz = [
  {
    id: 'limits-disc-1',
    question:
      'Why is understanding limits crucial for training neural networks? Discuss specific scenarios where limit analysis helps explain training dynamics.',
    hint: 'Think about vanishing/exploding gradients, activation function saturation, and learning rate selection.',
    sampleAnswer: `Understanding limits is fundamental to neural network training for several reasons:

**1. Activation Function Saturation:**
When inputs to sigmoid or tanh become very large (|x| → ∞), the activations saturate (approach 0 or 1). Understanding lim_(x→±∞) σ(x) helps us:
- Predict when neurons will stop learning (gradient ≈ 0 in saturated regions)
- Choose appropriate weight initialization to avoid immediate saturation
- Design better activation functions (ReLU avoids saturation for positive values)

**2. Vanishing Gradient Problem:**
In deep networks, gradients are products of many terms. If each term is less than 1, the product approaches 0 as depth increases. This is fundamentally a limit problem: lim_(n→∞) (0.9)^n = 0. Understanding this limit helps us:
- Recognize why deep networks were historically difficult to train
- Motivate skip connections (ResNet) that provide alternative gradient paths
- Choose activation functions with better gradient properties

**3. Learning Rate Selection:**
Gradient descent convergence is a limit: we want x_t → x* as t → ∞. The learning rate determines whether:
- We converge: lim_(t→∞) |x_t - x*| = 0 (learning rate too small → slow)
- We diverge: lim_(t→∞) |x_t - x*| = ∞ (learning rate too large)
- We oscillate without converging

**4. Loss Function Behavior:**
Understanding lim_(x→x*) L(x) near optimal parameters helps us:
- Set convergence criteria (when is gradient "close enough" to 0?)
- Understand plateaus in training
- Design better optimization algorithms

In practice, limit analysis transforms abstract mathematical concepts into practical guidelines for architecture design, initialization strategies, and training procedures.`,
    keyPoints: [
      'Activation function saturation relates to limits at infinity',
      'Vanishing gradients are fundamentally a limit problem with products',
      'Learning rate affects whether gradient descent converges (limit exists)',
      'Understanding limits near optima guides convergence criteria',
    ],
  },
  {
    id: 'limits-disc-2',
    question:
      'Compare and contrast the behavior of different activation functions using limit analysis. Which properties make an activation function "good" from a limits perspective?',
    hint: 'Consider sigmoid, tanh, ReLU, and Leaky ReLU. Analyze their limits at ±∞ and continuity properties.',
    sampleAnswer: `Activation functions can be analyzed through their limit behavior, revealing important training properties:

**Sigmoid: σ(x) = 1/(1 + e^(-x))**
- lim_(x→-∞) σ(x) = 0, lim_(x→+∞) σ(x) = 1
- Continuous and differentiable everywhere
- Problems: Bounded output causes vanishing gradients (derivative → 0 at extremes)
- The limit boundaries [0, 1] help with probability interpretation but harm gradient flow

**Tanh: tanh (x) = (e^x - e^(-x))/(e^x + e^(-x))**
- lim_(x→-∞) tanh (x) = -1, lim_(x→+∞) tanh (x) = 1
- Continuous and differentiable everywhere
- Better than sigmoid due to zero-centered output, but still saturates

**ReLU: ReLU(x) = max(0, x)**
- lim_(x→-∞) ReLU(x) = 0, lim_(x→+∞) ReLU(x) = ∞
- Continuous everywhere but not differentiable at x = 0
- Advantages: No upper saturation (lim as x→∞ is unbounded), cheap computation
- Problems: "Dying ReLU" when neurons always output 0 (stuck at left limit)

**Leaky ReLU: LReLU(x) = max(αx, x) where α ≈ 0.01**
- lim_(x→-∞) LReLU(x) = -∞, lim_(x→+∞) LReLU(x) = ∞
- Continuous everywhere, not differentiable at x = 0
- Solves dying ReLU problem (limit exists but is unbounded on both sides)

**Properties of "Good" Activation Functions (from limits perspective):**1. **No upper bound:** Allows gradient flow even for large inputs
2. **Non-zero gradient regions:** Enables learning throughout the domain
3. **Continuous:** Ensures stable optimization (no jumps)
4. **Appropriate behavior at limits:** Not both limits at 0 (information loss)

ReLU and its variants succeed because lim_(x→∞) ReLU(x) = ∞ (no saturation), despite the discontinuous derivative. This shows that limits matter more than smoothness for practical deep learning.`,
    keyPoints: [
      'Bounded activation functions (sigmoid, tanh) have vanishing gradient problems due to limits',
      'Unbounded activations (ReLU) maintain gradient flow for large inputs',
      'Continuity is important but differentiability can be sacrificed',
      'Limit behavior at ±∞ determines saturation properties',
    ],
  },
  {
    id: 'limits-disc-3',
    question:
      'Explain how the Intermediate Value Theorem can be applied to root-finding in optimization algorithms. What role does continuity play?',
    hint: 'Consider finding where gradients equal zero, binary search methods, and why gradient descent works.',
    sampleAnswer: `The Intermediate Value Theorem (IVT) is fundamental to many optimization and root-finding techniques in machine learning:

**IVT in Root-Finding (Finding Critical Points):**
When minimizing a loss function L(θ), we seek points where ∇L(θ) = 0. The IVT guarantees solutions exist:
- If ∇L(θ₁) < 0 and ∇L(θ₂) > 0, and ∇L is continuous
- Then there exists θ* ∈ (θ₁, θ₂) where ∇L(θ*) = 0
- This θ* is a critical point (potential minimum)

**Binary Search for Roots:**
The IVT enables efficient root-finding algorithms:
1. Start with interval [a, b] where f (a) and f (b) have opposite signs
2. Check midpoint m = (a + b)/2
3. Replace [a, b] with either [a, m] or [m, b] based on sign of f (m)
4. Repeat until |f (m)| < tolerance

This converges logarithmically: O(log(ε)) iterations for precision ε.

**Line Search in Optimization:**
In gradient descent, we often use line search to find step size α:
- We want to minimize φ(α) = L(θ - α∇L(θ))
- IVT guarantees that if φ'(0) < 0, there exists α > 0 where φ(α) < φ(0)
- We can binary search for optimal α where φ'(α) ≈ 0

**Importance of Continuity:**
Continuity is ESSENTIAL for IVT to hold. If the function has jumps:
- We might skip over roots entirely
- Binary search could fail
- Optimization becomes unreliable

In neural networks:
- Non-continuous activation functions (step function) make optimization difficult
- Discontinuous loss functions lead to optimization challenges
- We prefer continuous loss surfaces even if not everywhere differentiable (ReLU)

**Why Gradient Descent Works:**
Gradient descent relies implicitly on IVT-like reasoning:
- Starting from θ₀, we move in direction -∇L(θ₀)
- If we move far enough (but not too far), we'll cross a level set
- Continuity ensures we don't "jump over" better solutions
- The existence of a better point in the direction is guaranteed by IVT when gradient is non-zero

**Practical Implications:**1. Choosing continuous activation functions (sigmoid, ReLU, tanh) ensures IVT applies
2. Batch normalization maintains continuity despite adding noise
3. Adversarial training can create near-discontinuities that break IVT assumptions
4. Discrete optimization (integer constraints) requires different techniques since IVT doesn't apply

The IVT transforms existence questions ("Does a solution exist?") into algorithmic questions ("How do we find it?"), making it indispensable for practical optimization.`,
    keyPoints: [
      'IVT guarantees existence of roots/critical points in continuous functions',
      'Enables efficient binary search algorithms for optimization',
      'Continuity is essential - discontinuous functions break IVT',
      'Gradient descent implicitly relies on IVT-like reasoning',
      'Line search methods use IVT to find optimal step sizes',
    ],
  },
];
