import { QuizQuestion } from '../../../types';

export const optimizationAlgorithmsQuiz: QuizQuestion[] = [
  {
    id: 'optimization-algorithms-dq-1',
    question:
      'Explain why Adam optimizer is so widely used in deep learning. What problems does it solve compared to vanilla SGD, and when might you choose SGD with momentum instead?',
    sampleAnswer: `Adam (Adaptive Moment Estimation) has become the default optimizer in deep learning because it combines the best features of momentum and adaptive learning rates while being robust to hyperparameter choices. Here's why it's so effective and when alternatives might be better:

**Why Adam is Popular:**

1. **Adaptive Learning Rates Per Parameter:**
   - Automatically adjusts learning rate for each parameter based on gradient history
   - Parameters with consistently large gradients get smaller effective learning rates
   - Parameters with small gradients get larger effective learning rates
   - No manual per-layer learning rate tuning needed

2. **Momentum for Acceleration:**
   - First moment (m_t) accumulates gradient direction
   - Smooths out noisy gradients
   - Accelerates in consistent directions
   - β₁ = 0.9 works well by default

3. **Robust to Noisy Gradients:**
   - Second moment (v_t) tracks gradient variance
   - Dampens oscillations in high-variance directions
   - β₂ = 0.999 provides stable estimates

4. **Bias Correction:**
   - Corrects initialization bias in early iterations
   - Ensures m̂_t and v̂_t are unbiased estimates
   - Critical for proper convergence from start

5. **Works Out-of-the-Box:**
   - Default hyperparameters (lr=0.001, β₁=0.9, β₂=0.999) work well for most problems
   - Minimal tuning required
   - Saves experimentation time

**When to Use SGD with Momentum Instead:**

Despite Adam's popularity, SGD+Momentum often achieves better generalization in certain scenarios:

1. **Computer Vision (CNNs):**
   - SGD+Momentum often reaches better final accuracy on ImageNet
   - Slower convergence but better test performance
   - Requires careful LR scheduling (cosine annealing, step decay)

2. **Small Datasets:**
   - Adam's adaptive rates can overfit quickly
   - SGD's simpler update provides better regularization
   - Particularly true with <10K training samples

3. **When Training Time is Not Critical:**
   - SGD+Momentum may take 2-3x longer but reaches better minima
   - For critical applications where final 0.5% accuracy matters

4. **Well-Understood Problems:**
   - When you have good LR schedules from prior work
   - Transfer learning with established protocols

**Practical Recommendations for Trading:**

For financial ML:
- **Development/Exploration:** Adam (lr=0.001) - fast iteration
- **Production Models:** AdamW (lr=0.001, weight_decay=0.01) - better regularization
- **High-Frequency:** Adam with higher LR (0.01) - faster adaptation
- **Critical Strategies:** Try both Adam and SGD+Momentum, compare Sharpe ratios

Adam's robustness makes it ideal when you need reliable results without extensive hyperparameter tuning, which is why it's become the de facto standard.`,
    keyPoints: [
      'Adam combines momentum and adaptive learning rates for robust optimization',
      'Adapts learning rate per parameter based on gradient history',
      'Default hyperparameters (lr=0.001, β₁=0.9, β₂=0.999) work well out-of-box',
      'Bias correction ensures proper convergence from start',
      'SGD+Momentum may generalize better on CV tasks and small datasets',
      'For trading: Adam for development, AdamW for production (better regularization)',
    ],
  },
  {
    id: 'optimization-algorithms-dq-2',
    question:
      'Describe learning rate scheduling and why it is important. Compare step decay, exponential decay, and cosine annealing strategies, and explain when each is most appropriate.',
    sampleAnswer: `Learning rate scheduling adjusts the learning rate during training to improve convergence and final performance. It's based on the principle that different training phases require different step sizes:

**Why LR Scheduling Matters:**

Early training: Large LR for fast exploration
Late training: Small LR for fine-tuning
Result: Faster convergence + better final performance

**Three Main Strategies:**

**1. Step Decay:**
Multiply LR by γ (e.g., 0.1) every k epochs:
- Simple and interpretable
- Works well when you know training dynamics
- Common: Reduce by 10x at epochs 30, 60, 90
- Use when: You have prior knowledge of when to reduce LR

**2. Exponential Decay:**
η_t = η_0 × e^(-kt)
- Smooth, continuous decay
- More gradual than step decay
- Hyperparameter k controls decay rate
- Use when: You want smooth reduction throughout training

**3. Cosine Annealing:**
η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2
- Smooth sinusoidal decay
- Popular in modern deep learning
- Can include warm restarts (SGDR)
- Use when: Training modern architectures (ResNets, Transformers)

For trading models, cosine annealing with Adam often works best. Start with lr=0.01, decay to 0.0001 over training.`,
    keyPoints: [
      'LR scheduling improves convergence: large LR early, small LR late',
      'Step decay: Multiply by γ every k epochs - simple, requires prior knowledge',
      'Exponential: Smooth continuous decay - gradual reduction',
      'Cosine annealing: Sinusoidal decay - popular for modern architectures',
      'For trading: Cosine annealing with Adam, start 0.01 → end 0.0001',
    ],
  },
  {
    id: 'optimization-algorithms-dq-3',
    question:
      'What is the difference between Adam with L2 regularization and AdamW? Why does AdamW provide better weight decay, and when should you use it?',
    sampleAnswer: `AdamW (Adam with decoupled Weight decay) fixes a fundamental problem with how L2 regularization interacts with adaptive learning rate methods like Adam:

**The Problem with Adam + L2:**

Standard approach (INCORRECT):
\\\`\\\`\\\`
g_t = ∇L + λθ_t  # Add L2 to gradient
Apply Adam with g_t
\\\`\\\`\\\`

Issue: Adam's adaptive learning rates scale the L2 term differently for different parameters. Parameters with large historical gradients get their L2 penalty reduced, defeating the purpose of regularization.

**AdamW Solution (CORRECT):**

\\\`\\\`\\\`
1. Compute gradient: g_t = ∇L (no L2)
2. Apply Adam update with g_t
3. Apply weight decay separately: θ = θ - η·λ·θ
\\\`\\\`\\\`

By decoupling weight decay from the gradient, every parameter gets the same proportional decay regardless of its gradient history. This provides consistent, predictable regularization.

**When to Use AdamW:**

1. **Fine-tuning Large Models:** Pre-trained models benefit from consistent regularization
2. **High-Capacity Networks:** Deep networks with many parameters need strong regularization
3. **Trading Models:** Financial data is noisy - AdamW prevents overfitting to noise
4. **When L2 Matters:** If regularization is critical to your task

**Typical Settings:**
- weight_decay = 0.01 for general use
- weight_decay = 0.1 for aggressive regularization
- weight_decay = 0.001 for light regularization

For trading, AdamW with weight_decay=0.01 is recommended over Adam to ensure models don't overfit to historical patterns that won't repeat.`,
    keyPoints: [
      'Adam+L2 incorrectly couples weight decay with adaptive learning rates',
      'AdamW decouples: Apply Adam, then apply weight decay separately',
      'AdamW provides consistent regularization across all parameters',
      'Use AdamW for fine-tuning, large models, and noisy data (like financial)',
      'Typical weight_decay: 0.01 general, 0.1 aggressive, 0.001 light',
      'For trading: AdamW prevents overfitting to non-repeating historical patterns',
    ],
  },
];
