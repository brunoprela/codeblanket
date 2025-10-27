import { QuizQuestion } from '../../../types';

export const weightInitializationQuiz: QuizQuestion[] = [
  {
    id: 'weight-init-q1',
    question:
      'Explain why initializing all weights to zero is problematic. What is the "symmetry problem" and how does it prevent a network from learning effectively?',
    sampleAnswer: `Initializing all weights to zero creates a critical symmetry problem that prevents neural networks from learning:

**The Symmetry Problem:**

When all weights are initialized to zero, all neurons in a given layer compute identical outputs:
- Given input x, every neuron computes: output = 0·x₁ + 0·x₂ + ... + 0 = 0
- During forward propagation, all neurons produce the same activation
- During backpropagation, all neurons receive identical gradients
- Weight updates are identical: Δw = -η·gradient (same gradient for all)
- After the update, all weights remain identical (just non-zero now)

**Why This Prevents Learning:**1. **No Differentiation**: All neurons remain functionally identical throughout training. If neuron 1 and neuron 2 always have the same weights, they'll always produce the same output.

2. **Wasted Capacity**: A layer with 512 neurons effectively acts like a single neuron, completely wasting the network's representational capacity.

3. **No Feature Specialization**: Different neurons should learn different features (e.g., detecting edges, textures, patterns). With identical weights, they can't specialize.

4. **Mathematical Redundancy**: If all neurons are identical, the network reduces to: output = n · (single neuron output), where n is the number of neurons. You might as well have one neuron.

**The Solution: Random Initialization**

Random initialization breaks symmetry by giving each neuron different starting weights:
- Neuron 1 might start with [0.23, -0.45, 0.12, ...]
- Neuron 2 might start with [-0.31, 0.56, -0.08, ...]
- Each neuron receives different gradients
- Weight updates differ across neurons
- Neurons specialize to learn different features

**Practical Example:**

Consider a 2-input, 3-hidden-neuron network classifying XOR:
- With zero initialization: All 3 neurons always compute the same function → Cannot solve XOR
- With random initialization: Neurons can specialize:
  - Neuron 1 might learn: "x₁ AND NOT x₂"
  - Neuron 2 might learn: "NOT x₁ AND x₂"
  - Neuron 3 might learn: "x₁ OR x₂"
  - Output combines these → Can solve XOR

**Key Takeaway:**

Zero initialization reduces any network to a trivial single-neuron-per-layer model, regardless of actual width. Random initialization is essential to unlock the full representational power of neural networks by allowing neurons to learn diverse, specialized features.`,
    keyPoints: [
      'Zero initialization causes all neurons in a layer to compute identical outputs',
      'All neurons receive identical gradients during backpropagation, leading to identical weight updates',
      'Network cannot break symmetry - neurons remain identical throughout training',
      'Effectively reduces each layer to a single neuron regardless of width',
      'Wasts network capacity - 512 neurons behave like 1 neuron',
      'Random initialization breaks symmetry by giving each neuron different starting weights',
      'Different initial weights → different gradients → different updates → feature specialization',
    ],
  },
  {
    id: 'weight-init-q2',
    question:
      'Compare Xavier (Glorot) initialization with He initialization. When should each be used, and why do they have different formulas? What problem are they each trying to solve?',
    sampleAnswer: `Xavier and He initialization are both designed to maintain stable variance of activations and gradients through deep networks, but they're optimized for different activation functions:

**Xavier (Glorot) Initialization:**

Formula: W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
Or: W ~ Normal(0, √(2/(n_in + n_out)))

**Designed for:** Sigmoid and Tanh activations

**Why this formula?** Xavier initialization maintains variance ≈ 1 for both activations and gradients. For sigmoid/tanh:
- These are symmetric functions centered at zero
- Gradient depends on both forward and backward pass
- Need to account for both n_in (forward) and n_out (backward)
- The formula √(2/(n_in + n_out)) balances both directions

**He Initialization:**

Formula: W ~ Normal(0, √(2/n_in))

**Designed for:** ReLU and its variants (Leaky ReLU, PReLU)

**Why this formula?** ReLU zeros out negative activations, effectively halving variance:
- ReLU(x) = max(0, x) → About 50% of activations become zero
- This halves the variance of activations
- To compensate, we need larger initial weights
- √(2/n_in) is larger than Xavier\'s √(2/(n_in + n_out))
- Only depends on n_in because ReLU gradient is simple (0 or 1)

**The Problem They Solve:**

Both address **vanishing and exploding gradients** in deep networks:

1. **Too Small Initialization** (e.g., W ~ N(0, 0.01)):
   - Activations: 1.0 → 0.5 → 0.25 → 0.12 → ... (vanishing)
   - Gradients shrink exponentially through backprop
   - Deep layers barely learn, training stalls

2. **Too Large Initialization** (e.g., W ~ N(0, 10)):
   - Activations: 1.0 → 10.0 → 100.0 → 1000.0 → ... (exploding)
   - Gradients explode, causing NaN/Inf
   - Training diverges

3. **Proper Initialization**:
   - Maintains variance ≈ 1 through all layers
   - Forward pass: var (a_L) ≈ var (a_0) for all layers L
   - Backward pass: var(∂L/∂a_L) ≈ var(∂L/∂a_0)
   - Enables training of very deep networks (50-200 layers)

**When to Use Which:**

- **ReLU, Leaky ReLU, ELU** → He initialization
- **Sigmoid, Tanh** → Xavier initialization
- **Modern default**: He (most networks use ReLU)

**Example Impact:**

Consider a 50-layer ReLU network:

With Xavier (wrong choice):
- Layer 10: activation std ≈ 0.3 (shrinking)
- Layer 30: activation std ≈ 0.01 (nearly vanished)
- Layer 50: activation std ≈ 0.0001 (dead)
- Training: Slow convergence or failure

With He (correct choice):
- Layer 10: activation std ≈ 1.0
- Layer 30: activation std ≈ 1.0
- Layer 50: activation std ≈ 1.0
- Training: Fast, stable convergence

**Key Insight:**

The difference between Xavier and He reflects a fundamental property of activation functions. ReLU's non-symmetric behavior (zeroing negatives) requires different initialization than symmetric functions (sigmoid/tanh). Using the wrong initialization can make deep networks untrainable.`,
    keyPoints: [
      'Xavier: √(2/(n_in + n_out)) for sigmoid/tanh activations',
      'He: √(2/n_in) for ReLU activations',
      'Both maintain variance ≈ 1 through forward and backward passes',
      'ReLU zeros half the activations, requiring larger initialization to compensate',
      'Xavier balances n_in and n_out for symmetric activations',
      'He only depends on n_in because ReLU gradient is binary (0 or 1)',
      'Wrong initialization causes vanishing/exploding gradients in deep networks',
      'Modern default: He initialization for ReLU networks',
    ],
  },
  {
    id: 'weight-init-q3',
    question:
      'In very deep networks (100+ layers), even with proper initialization, activations and gradients can still vanish or explode. What additional techniques beyond initialization are needed? How do residual connections, batch normalization, and layer normalization address these issues?',
    sampleAnswer: `While proper initialization is crucial, it's insufficient for very deep networks (100+ layers). Modern deep learning combines multiple techniques to enable stable training at extreme depths:

**Why Initialization Alone Isn't Enough:**

Even with He/Xavier initialization, errors accumulate:
- Small numerical errors compound over 100+ layers
- Floating point precision limits
- Non-linear dynamics during training change activation distributions
- Learning rate and optimizer choices interact with depth

**Residual Connections (ResNets):**

**How they work:**
Instead of learning H(x), learn residual F(x) = H(x) - x
\`\`\`
output = F(x) + x  # Skip connection adds input directly to output
\`\`\`

**Why they help:**1. **Gradient Highway**: Gradients can flow directly through skip connections
   - Without skip: gradient must flow through all layers → exponential decay
   - With skip: gradient has direct path → linear accumulation

2. **Identity Mapping**: Network can learn identity function by setting F(x) = 0
   - Easier optimization problem
   - Deep networks don't hurt performance (worst case: act like shallow network)

3. **Ensemble Effect**: Network effectively ensembles paths of different depths

**Impact**: Enabled training of 152-layer (ResNet-152) and even 1000-layer networks

**Batch Normalization:**

**How it works:**
Normalize activations to zero mean, unit variance within each mini-batch:
\`\`\`
x_normalized = (x - μ_batch) / √(σ²_batch + ε)
output = γ · x_normalized + β  # Learnable scale and shift
\`\`\`

**Why it helps:**1. **Stabilizes Distributions**: Keeps activations in healthy range regardless of depth
2. **Reduces Internal Covariate Shift**: Layer inputs stay consistent as parameters update
3. **Higher Learning Rates**: Stability allows 10-100x larger learning rates
4. **Gradient Flow**: Normalization prevents gradients from vanishing/exploding
5. **Acts as Regularizer**: Batch statistics add noise, reducing overfitting

**Impact**: Made training 50-100 layer networks routine

**Layer Normalization:**

**How it works:**
Normalize across features (not batch) for each sample independently:
\`\`\`
x_normalized = (x - μ_features) / √(σ²_features + ε)
\`\`\`

**Why it helps:**1. **Batch-Independent**: Each sample normalized independently
2. **Sequence-Friendly**: Works with variable-length sequences (RNNs)
3. **Small Batch Compatible**: No batch statistics dependency
4. **Transformer Standard**: Used in all modern Transformers (GPT, BERT)

**Differences from Batch Norm:**
- Batch Norm: Normalize across batch (same feature, different samples)
- Layer Norm: Normalize across features (same sample, different features)

**Other Essential Techniques:**

**Gradient Clipping:**
\`\`\`python
if ||gradient|| > threshold:
    gradient = gradient * threshold / ||gradient||
\`\`\`
Prevents single bad batch from exploding gradients

**Learning Rate Scheduling:**
- Warmup: Gradually increase LR for first few epochs
- Decay: Reduce LR as training progresses
- Prevents instability from large initial updates

**Architecture Choices:**
- Bottleneck layers (reduce dimensions between blocks)
- Careful depth/width balance
- Strategic placement of normalization layers

**Combining Techniques - Modern Best Practice:**

For 100+ layer networks:
1. **He initialization** → Good starting point
2. **Residual connections** → Gradient flow
3. **Batch/Layer Normalization** → Stable activations
4. **Gradient clipping** → Safety net
5. **Learning rate warmup** → Smooth start
6. **Careful architecture** → Thoughtful design

**Example: ResNet-152 Training Stack**
\`\`\`python
# 152 layers would be impossible without:
- He initialization (starting point)
- Residual blocks every 2-3 layers (gradient highway)
- Batch normalization after each convolution (stability)
- SGD with momentum (smooth updates)
- Learning rate warmup + cosine decay (careful LR schedule)
- Gradient clipping (safety)
\`\`\`

**Historical Context:**

- Pre-2012: 3-5 layers was "deep"
- 2014: 19 layers (VGG) was cutting edge
- 2015: 152 layers (ResNet) broke records
- 2024: 100-layer networks are routine, 1000+ layers possible

The key insight: depth requires a holistic approach. Initialization starts the journey, but residual connections, normalization, and careful training enable reaching the destination.`,
    keyPoints: [
      'Initialization alone insufficient for very deep (100+) networks',
      'Residual connections provide gradient highways, enabling direct gradient flow',
      'Skip connections allow learning identity function (F(x) = 0)',
      'Batch Normalization normalizes activations, stabilizes training, allows higher LR',
      'Layer Normalization normalizes within each sample, better for RNNs/Transformers',
      'Gradient clipping prevents catastrophic divergence from single bad batch',
      'Learning rate warmup prevents instability from large initial gradients',
      'Modern architectures combine multiple techniques: init + residual + norm + clipping + LR schedule',
      'This stack enabled progression from 3-layer to 1000-layer networks',
    ],
  },
];
