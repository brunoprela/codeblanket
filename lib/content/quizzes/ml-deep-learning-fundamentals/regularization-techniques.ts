import { QuizQuestion } from '../../../types';

export const regularizationTechniquesQuiz: QuizQuestion[] = [
  {
    id: 'regularization-q1',
    question:
      'Compare and contrast L1 and L2 regularization. What are the mathematical differences, and how do they affect the learned weights differently? When might you prefer one over the other?',
    sampleAnswer: `L1 and L2 regularization are both techniques to prevent overfitting by penalizing large weights, but they have fundamentally different mathematical properties and practical effects:

**Mathematical Formulation:**

L2 (Ridge) Regularization:
- Loss: L_total = L_data + (λ/2) · Σ(w²)
- Gradient: ∂L/∂w = ∂L_data/∂w + λ·w
- Penalty grows quadratically with weight magnitude

L1 (Lasso) Regularization:
- Loss: L_total = L_data + λ · Σ|w|
- Gradient: ∂L/∂w = ∂L_data/∂w + λ·sign(w)
- Penalty grows linearly with weight magnitude

**Key Differences:**

1. **Effect on Weights:**
   - L2: Shrinks all weights proportionally, smoothly toward zero
   - L1: Drives weights exactly to zero, creating sparse models

2. **Gradient Behavior:**
   - L2: Gradient is proportional to weight magnitude (λ·w)
   - L1: Gradient is constant (λ·sign(w)), independent of magnitude

3. **Sparsity:**
   - L2: Dense models (all weights small but non-zero)
   - L1: Sparse models (many weights exactly zero)

**Why L1 Creates Sparsity:**

The constant gradient of L1 means small weights get the same penalty as large weights. Consider a weight w = 0.01:
- L2 gradient: λ·0.01 = very small push toward zero
- L1 gradient: λ·sign(0.01) = λ = large push toward zero
- L1 can push small weights all the way to zero

**When to Use Each:**

Use L1 when:
- Feature selection is desired (automatic variable selection)
- Interpretability matters (sparse models easier to explain)
- Many features are irrelevant or redundant
- Want to identify most important features
- Example: Predicting stock returns with 1000 technical indicators → L1 might select the 50 most important

Use L2 when:
- All features are potentially useful
- Want smooth weight distribution
- Multicollinearity present (L2 handles better)
- General-purpose regularization
- Example: Image classification where all pixels could be relevant

Use Elastic Net (L1 + L2) when:
- Want benefits of both
- Feature selection + stable solution
- Groups of correlated features (L1 alone unstable)

**Practical Example:**

Consider 100 features predicting house prices:
- True important features: square footage, location, age (3 features)
- Noise features: 97 random variables

L2 Regularization:
- Weights: [0.5, 0.4, 0.3, 0.02, 0.01, 0.01, ...]  # All non-zero
- Uses all 100 features with varying importance
- Model complexity: 100 parameters

L1 Regularization:
- Weights: [0.6, 0.5, 0.4, 0, 0, 0, ...]  # Many exact zeros
- Uses only 3-10 features automatically
- Model complexity: 3-10 parameters
- Better interpretation: "Price depends on size, location, age"`,
    keyPoints: [
      'L2 penalty: λ/2·Σ(w²), gradient: λ·w, shrinks weights proportionally',
      'L1 penalty: λ·Σ|w|, gradient: λ·sign(w), drives weights to exactly zero',
      'L2 produces dense models with many small weights',
      'L1 produces sparse models with many zero weights (automatic feature selection)',
      'L1 gradient is constant regardless of weight magnitude',
      'Use L1 for feature selection and interpretability',
      'Use L2 for general regularization with all features',
      'Elastic Net combines both for grouped feature selection',
    ],
  },
  {
    id: 'regularization-q2',
    question:
      'Explain how Dropout acts as a regularizer despite not directly constraining weights. What is the "ensemble interpretation" of dropout, and why is it important to handle dropout differently during training vs inference?',
    sampleAnswer: `Dropout is a powerful regularization technique that prevents overfitting through a completely different mechanism than weight penalties. It works by randomly "dropping out" (setting to zero) neurons during training:

**How Dropout Works:**

During training (each forward pass):
1. For each neuron, generate random number r ~ Uniform(0, 1)
2. If r < p (dropout rate), set neuron output to 0
3. Otherwise, keep neuron active and scale by 1/(1-p)

**Why It Regularizes (Prevents Co-Adaptation):**

Without dropout, neurons can become co-dependent:
- Neuron A learns to detect "eyes"
- Neuron B learns to detect "nose"
- Neuron C learns "if both A and B fire, it's a face"
- Problem: Neurons rely on specific other neurons being present
- If weights change slightly, fragile dependencies break

With dropout:
- Any neuron might be randomly disabled
- Neurons cannot rely on specific other neurons
- Must learn robust, independent features
- Forces redundant representations
- Each neuron must be individually useful

**The Ensemble Interpretation:**

Dropout effectively trains an exponential ensemble of models:

For a network with n neurons:
- Each forward pass uses a different random subset of neurons
- Total possible subnetworks: 2^n
- Each subnetwork is trained on each batch
- At inference, using all neurons averages predictions from all subnetworks

Example with 100 neurons:
- Possible subnetworks: 2^100 ≈ 10^30 models
- Each training batch trains a different random model
- Inference combines all these models
- Ensemble of 10^30 models is extremely powerful!

**Training vs Inference Handling:**

**During Training (Inverted Dropout):**
\`\`\`python
if training:
    mask = (torch.rand(x.shape) > dropout_rate)
    x = x * mask / (1 - dropout_rate)  # Scale up active neurons
\`\`\`
- Randomly drop neurons
- Scale remaining by 1/(1-p) to maintain expected value
- Forces network to learn robust features

**During Inference:**
\`\`\`python
if not training:
    x = x  # Use all neurons, no dropout, no scaling
\`\`\`
- Use ALL neurons (no dropout)
- No scaling needed (we scaled during training)
- Approximates ensemble average of all subnetworks

**Why Different Handling Is Critical:**

If you applied dropout at inference:
- Random predictions (not deterministic)
- Lower performance (missing neurons)
- Not using full model capacity
- Defeats purpose of training ensemble

If you didn't scale during training:
- Training: average activation = (1-p) · original
- Inference: average activation = original
- Mismatch causes poor performance
- Network trained with lower activations, tested with higher

**Mathematical Justification:**

Expected activation value:
- Training with scaling: E[a] = (1-p)·(a/(1-p)) + p·0 = a
- Inference: E[a] = a
- Values match! Network sees consistent signal

**Practical Example:**

Consider a neuron with output = 10:

Training (p=0.5):
- 50% chance: output = 0 (dropped)
- 50% chance: output = 10 / (1-0.5) = 20 (scaled up)
- Expected value: 0.5·0 + 0.5·20 = 10 ✓

Inference:
- Always active: output = 10
- Expected value: 10 ✓

Consistent expectations enable effective learning.

**Why Dropout Is So Effective:**

1. **Prevents overfitting** through forced independence
2. **Reduces co-adaptation** between neurons
3. **Creates ensembles** of exponentially many models
4. **Adds noise** during training (implicit regularization)
5. **No computational cost** at inference
6. **Simple to implement** (few lines of code)
7. **Works with any architecture** (not task-specific)

**Typical Dropout Rates:**
- Input layer: 0.1-0.2 (keep most information)
- Hidden layers: 0.5 (standard choice)
- Output layer: 0 (never drop final predictions)

Dropout demonstrates that regularization doesn't require explicit weight constraints—preventing feature co-adaptation can be just as effective!`,
    keyPoints: [
      'Dropout randomly deactivates neurons with probability p during training',
      'Prevents co-adaptation: neurons cannot rely on specific other neurons',
      'Forces learning of robust, independent features',
      'Ensemble interpretation: training 2^n subnetworks exponentially',
      'Inverted dropout scales active neurons by 1/(1-p) during training',
      'Inference uses all neurons without dropout (approximates ensemble average)',
      'Different train/test behavior is intentional and necessary',
      'Training/inference mismatch would cause poor performance',
      'No direct weight constraint, but effective regularization through architectural randomness',
    ],
  },
  {
    id: 'regularization-q3',
    question:
      'Batch Normalization has become ubiquitous in deep learning, but it has drawbacks. Explain why Layer Normalization was developed as an alternative, what problems it solves, and in what scenarios each normalization method is preferred.',
    sampleAnswer: `Batch Normalization revolutionized deep learning by stabilizing training, but it has limitations that led to the development of Layer Normalization. Understanding when to use each is crucial for effective model design:

**Batch Normalization (BatchNorm):**

**How it works:**
\`\`\`python
# Normalize across batch dimension (same feature, different samples)
mean = batch.mean(dim=0)  # Shape: (features,)
var = batch.var(dim=0)
x_normalized = (x - mean) / sqrt(var + eps)
output = gamma * x_normalized + beta  # Learnable scale/shift
\`\`\`

**Benefits:**
- Stabilizes training (reduces internal covariate shift)
- Allows higher learning rates (10-100x)
- Reduces sensitivity to initialization
- Acts as regularizer (batch statistics add noise)
- Enables training of very deep networks (50-200 layers)

**Problems with BatchNorm:**

1. **Batch Coupling:**
   - Each sample's normalization depends on other samples in batch
   - Breaks independence assumption
   - Sample's gradient affected by other samples

2. **Small Batch Issues:**
   - Batch statistics unreliable with small batches (< 16)
   - Noisy estimates of mean/variance
   - Common in: distributed training, large models, limited memory

3. **Different Train/Test Behavior:**
   - Training: uses batch statistics
   - Inference: uses running average (computed during training)
   - Mismatch can cause performance degradation
   - Must track running statistics carefully

4. **Sequence Length Problems:**
   - RNNs process variable-length sequences
   - Different sequences, different lengths
   - BatchNorm across sequences doesn't make semantic sense
   - Statistics not meaningful across different sentence structures

5. **Online Learning:**
   - Cannot use batch statistics with single sample
   - Incompatible with online/continual learning scenarios

**Layer Normalization (LayerNorm):**

**How it works:**
\`\`\`python
# Normalize across feature dimension (same sample, different features)
mean = sample.mean(dim=-1, keepdim=True)  # Shape: (batch, 1)
var = sample.var(dim=-1, keepdim=True)
x_normalized = (x - mean) / sqrt(var + eps)
output = gamma * x_normalized + beta
\`\`\`

**Key Difference:**
- BatchNorm: normalize across batch (same feature, all samples)
- LayerNorm: normalize across features (same sample, all features)

**Advantages of LayerNorm:**

1. **Sample Independence:**
   - Each sample normalized independently
   - No coupling between samples
   - Batch size doesn't matter

2. **Consistent Train/Test:**
   - Same computation during training and inference
   - No running statistics needed
   - More predictable behavior

3. **RNN/Transformer Friendly:**
   - Works naturally with sequences
   - Each timestep normalized independently
   - Standard in Transformers (GPT, BERT, etc.)

4. **Small Batch Compatible:**
   - Works with batch size = 1
   - No statistical estimation issues
   - Great for large models with memory constraints

5. **Online Learning:**
   - Can process single samples in real-time
   - No batch accumulation needed

**When to Use Each:**

**Use Batch Normalization when:**
- Training CNNs on images (standard choice)
- Large batch sizes available (>= 32)
- Feedforward architectures
- Want regularization effect from batch coupling
- Examples: ResNet, VGG, EfficientNet for ImageNet

**Use Layer Normalization when:**
- Training RNNs or Transformers
- Small batch sizes (<= 16)
- Online/continual learning
- Need deterministic train/test behavior
- Variable-length sequences
- Examples: GPT-3, BERT, any modern Transformer

**Hybrid Approaches:**

Group Normalization:
- Divides channels into groups, normalizes within groups
- Middle ground between Batch and Layer
- Works with batch size = 1
- Good for object detection, segmentation

Instance Normalization:
- Normalizes each sample independently like Layer
- Used in style transfer, GANs

**Performance Comparison:**

CNNs (ImageNet classification):
- BatchNorm: ✓ Best accuracy
- LayerNorm: Works but slightly worse
- Large batches available, spatial structure benefits from batch stats

Transformers (Language modeling):
- LayerNorm: ✓ Standard choice
- BatchNorm: Problematic (variable lengths, small batches)
- Sequence processing needs sample independence

**Example: GPT-3 Architecture:**
\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(self):
        self.ln1 = LayerNorm(d_model)  # Before attention
        self.attn = MultiHeadAttention()
        self.ln2 = LayerNorm(d_model)  # Before FFN
        self.ffn = FeedForward()
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # LayerNorm + residual
        x = x + self.ffn(self.ln2(x))
        return x
\`\`\`

All modern Transformers use LayerNorm exclusively—BatchNorm would fail for variable-length text sequences.

**Key Insight:**

The choice between Batch and Layer Normalization reflects fundamental assumptions about your data:
- BatchNorm: "Samples in a batch should have similar statistics"
- LayerNorm: "Each sample should be normalized independently"

For images with consistent structure, BatchNorm's batch coupling helps. For sequences with variable structure, LayerNorm's independence is essential.`,
    keyPoints: [
      'BatchNorm normalizes across batch (same feature, different samples)',
      'LayerNorm normalizes across features (same sample, different features)',
      'BatchNorm couples samples in batch, breaks independence',
      'BatchNorm problematic for: small batches, RNNs, variable sequences, online learning',
      'LayerNorm normalizes each sample independently',
      'LayerNorm consistent train/test behavior (no running stats)',
      'Use BatchNorm for CNNs with large batches',
      'Use LayerNorm for RNNs, Transformers, small batches',
      'All modern Transformers (GPT, BERT) use LayerNorm exclusively',
    ],
  },
];
