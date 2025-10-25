import { QuizQuestion } from '../../../types';

export const activationFunctionsQuiz: QuizQuestion[] = [
  {
    id: 'activation-functions-dq-1',
    question:
      'Explain the "dying ReLU" problem in detail. Why does it occur, how can you detect it in practice, and what are the most effective strategies to prevent or mitigate it?',
    sampleAnswer: `The dying ReLU problem is a significant issue in deep learning where neurons become permanently inactive during training, outputting zero for all inputs and preventing any further learning:

**Why It Occurs:**

ReLU has the property that for all negative inputs, both the output and gradient are zero:
- ReLU(z) = max(0, z)
- ReLU'(z) = 1 if z > 0, else 0

The problem emerges when:
1. During backpropagation, a neuron receives a large negative gradient
2. The weight update is large: w_new = w_old - learning_rate * gradient
3. After the update, for ALL training examples: z = w^T x + b < 0
4. The neuron always outputs 0
5. The gradient is always 0
6. No weight updates occur → neuron is permanently "dead"

**Mathematical Example:**
\`\`\`
Before: w = [1.0, 0.5], b = 0.2
Large gradient: ∇w = [5.0, 3.0], ∇b = 4.0
Update with lr=1.0: w_new = [-4.0, -2.5], b_new = -3.8

For any reasonable input, z will be negative:
z = -4.0*x₁ - 2.5*x₂ - 3.8 < 0 for most (x₁, x₂)
→ ReLU(z) = 0 always
→ Gradient = 0 always
→ Dead neuron!
\`\`\`

**How to Detect in Practice:**

1. **Monitor Activation Statistics:**
\`\`\`python
def check_dead_neurons (activations):
    """
    Check percentage of dead neurons
    activations: shape (batch, neurons)
    """
    dead_neurons = np.sum (np.all (activations == 0, axis=0))
    total_neurons = activations.shape[1]
    return dead_neurons / total_neurons

# During training
activations = model.get_activations (validation_data)
dead_percentage = check_dead_neurons (activations)
print(f"Dead neurons: {dead_percentage:.1%}")
\`\`\`

2. **Gradient Flow Analysis:**
- Track gradients layer by layer
- If gradients are consistently zero for some neurons → dead

3. **Warning Signs:**
- Training loss stops improving
- Validation loss plateaus
- Layer activations show many zeros (>50% consistently)
- Gradients vanish in early layers

**Prevention and Mitigation Strategies:**

**1. Use Leaky ReLU Instead:**
\`\`\`python
def leaky_relu (z, alpha=0.01):
    return np.where (z > 0, z, alpha * z)
\`\`\`
- Allows small negative gradient (alpha * z when z < 0)
- Neurons can recover from negative weights
- Typical alpha: 0.01 or 0.1

**2. Use ELU (Exponential Linear Unit):**
\`\`\`python
def elu (z, alpha=1.0):
    return np.where (z > 0, z, alpha * (np.exp (z) - 1))
\`\`\`
- Smooth activation with negative values
- Better gradient flow
- Self-normalizing properties

**3. Lower Learning Rate:**
- Prevents large weight updates that kill neurons
- Use learning rate schedules
- Start with lr=0.001 instead of 0.01

**4. Proper Weight Initialization:**
- Use He initialization for ReLU: w ~ N(0, sqrt(2/n_in))
- Prevents weights from starting in bad regions
\`\`\`python
W = np.random.randn (n_in, n_out) * np.sqrt(2.0 / n_in)
\`\`\`

**5. Batch Normalization:**
- Normalizes inputs to each layer
- Prevents extreme pre-activations
- Reduces likelihood of all inputs being negative

**6. Gradient Clipping:**
\`\`\`python
def clip_gradients (gradients, max_norm=1.0):
    total_norm = np.linalg.norm (gradients)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        gradients *= clip_coef
    return gradients
\`\`\`

**7. Monitor and Reinitialize:**
- Track dead neuron percentage during training
- If >40% dead, consider:
  - Lowering learning rate
  - Switching to Leaky ReLU
  - Reinitializing dead neurons

**Comparison of Solutions:**

| Solution | Effectiveness | Computational Cost | Ease of Implementation |
|----------|---------------|-------------------|----------------------|
| Leaky ReLU | ★★★★★ | None | ★★★★★ |
| ELU | ★★★★☆ | Small (exp) | ★★★★☆ |
| Lower LR | ★★★☆☆ | None | ★★★★★ |
| He Init | ★★★★☆ | None | ★★★★★ |
| Batch Norm | ★★★★★ | Medium | ★★★☆☆ |

**Real-World Example:**

In a trading model predicting stock returns, dying ReLU can be particularly problematic:
- Market regimes change → data distribution shifts
- Dead neurons cannot adapt to new patterns
- Model performance degrades silently
- Using Leaky ReLU + proper initialization + monitoring is essential

**Conclusion:**

The dying ReLU problem is preventable with proper practices. The best strategy is to:
1. Use Leaky ReLU by default (almost no downside vs ReLU)
2. Use He initialization
3. Monitor activation statistics during training
4. Keep learning rates reasonable (< 0.01 to start)

These practices ensure neurons remain trainable throughout the learning process.`,
    keyPoints: [
      'Dying ReLU occurs when large negative gradients push weights such that all outputs become zero permanently',
      'Dead neurons have zero gradient and cannot recover through backpropagation',
      'Detect by monitoring activation statistics - look for >30-40% zero activations',
      'Leaky ReLU is the simplest fix - allows small negative gradient (alpha * z)',
      'He initialization and lower learning rates reduce likelihood of dying neurons',
      'Batch normalization helps by preventing extreme pre-activations',
      'Monitor dead neuron percentage during training as early warning system',
    ],
  },
  {
    id: 'activation-functions-dq-2',
    question:
      'Compare and contrast sigmoid, tanh, and ReLU activation functions. For a deep neural network (10+ layers) predicting next-day stock returns, which activation would you use for hidden layers and why? What problems might arise with each choice?',
    sampleAnswer: `Choosing the right activation function is critical for deep networks. Let\'s analyze sigmoid, tanh, and ReLU in the context of deep learning for financial prediction:

**Sigmoid Function:**

Properties:
- Range: (0, 1)
- Formula: σ(z) = 1 / (1 + e^(-z))
- Derivative: σ'(z) = σ(z)(1 - σ(z))
- Maximum gradient: 0.25 (at z=0)

Advantages:
- Output interpretable as probability
- Smooth and differentiable
- Historical significance

Disadvantages for Deep Networks:
- **Vanishing gradient**: For |z| > 3, gradient < 0.05
- **Not zero-centered**: All outputs positive (0, 1)
- **Saturation**: Neurons can saturate, stopping learning
- **Expensive**: Requires exp() computation

**In a 10-layer network:**
- Gradient through 10 layers: 0.25^10 ≈ 0.00000095
- Gradients vanish completely → early layers don't train
- Would fail for deep stock return prediction

**Tanh Function:**

Properties:
- Range: (-1, 1)
- Formula: tanh (z) = (e^z - e^(-z)) / (e^z + e^(-z))
- Derivative: tanh'(z) = 1 - tanh²(z)
- Maximum gradient: 1.0 (at z=0)

Advantages over Sigmoid:
- Zero-centered outputs
- Stronger gradients (max=1.0 vs 0.25)
- Better than sigmoid for hidden layers

Disadvantages for Deep Networks:
- **Still saturates**: For |z| > 2, gradient < 0.1
- **Vanishing gradient persists**: Through 10 layers: 0.5^10 ≈ 0.001
- **Computational cost**: Requires exp()

**In a 10-layer network:**
- Better than sigmoid, but still problematic
- Training would be slow
- Early layers would learn slowly

**ReLU Function:**

Properties:
- Range: [0, ∞)
- Formula: ReLU(z) = max(0, z)
- Derivative: 1 if z > 0, else 0
- Maximum gradient: 1.0 (no saturation for z > 0)

Advantages for Deep Networks:
- **No vanishing gradient** for positive activations
- **Computational efficiency**: Just comparison
- **Sparse activations**: ~50% neurons zero
- **Fast convergence**: 6x faster than sigmoid/tanh

Disadvantages:
- **Dying ReLU**: Neurons can permanently die
- **Not zero-centered**: All outputs ≥ 0
- **Unbounded**: Can lead to exploding activations

**For Stock Return Prediction (10 layers):**

**Recommended Choice: ReLU (or Leaky ReLU)**

Reasoning:
1. **Gradient Flow**: With 10 layers, gradient preservation is critical
   - ReLU: Gradient = 1.0 through all active neurons
   - Sigmoid: Gradient ≈ 0.25^10 ≈ 0 (training fails)
   - Tanh: Gradient ≈ 0.5^10 ≈ 0.001 (very slow training)

2. **Financial Data Characteristics**:
   - Non-stationary and noisy
   - Requires deep networks to learn complex patterns
   - Fast adaptation needed as markets evolve
   - ReLU enables faster training and adaptation

3. **Practical Performance**:
   - Modern financial ML models use ReLU-family activations
   - Success of deep learning in finance relies on ReLU

**Implementation Strategy:**

\`\`\`python
class StockReturnPredictor:
    def __init__(self, input_features=50):
        # 10-layer network with Leaky ReLU
        layers = [input_features, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        self.weights = []
        self.biases = []
        
        for i in range (len (layers) - 1):
            # He initialization for ReLU
            W = np.random.randn (layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros (layers[i+1])
            self.weights.append(W)
            self.biases.append (b)
    
    def leaky_relu (self, z, alpha=0.01):
        return np.where (z > 0, z, alpha * z)
    
    def forward (self, X):
        activation = X
        # Hidden layers: Leaky ReLU
        for i in range (len (self.weights) - 1):
            z = activation @ self.weights[i] + self.biases[i]
            activation = self.leaky_relu (z)
        
        # Output layer: Linear (for regression)
        output = activation @ self.weights[-1] + self.biases[-1]
        return output
\`\`\`

**Problems That Might Arise:**

1. **With Sigmoid/Tanh (if used):**
   - Vanishing gradients → training fails
   - Early layers barely update
   - Model cannot learn complex patterns
   - Stuck at poor local minimum

2. **With ReLU:**
   - Dying neurons (20-40% may die)
   - Exploding activations without batch norm
   - Not zero-centered can slow convergence
   - Solution: Use Leaky ReLU + batch norm

3. **Financial-Specific Issues:**
   - Market regime changes require retraining
   - Dead neurons cannot adapt to new patterns
   - Non-stationary data amplifies all problems

**Mitigation Strategies:**

1. **Use Leaky ReLU instead of ReLU**:
   - Prevents dying neurons
   - Minimal computational overhead
   - alpha=0.01 works well

2. **Add Batch Normalization**:
   - Normalizes activations between layers
   - Prevents exploding/vanishing activations
   - Acts as regularization

3. **Use Residual Connections** (for very deep networks):
   - Allows gradient to skip layers
   - Prevents gradient vanishing
   - Enables training of 50+ layer networks

4. **Monitor Training**:
   - Track activation statistics
   - Check for dead neurons (>30% zeros)
   - Monitor gradient norms
   - Adjust if needed

**Conclusion:**

For deep financial prediction models:
- **Use Leaky ReLU or ELU for hidden layers**
- **Never use sigmoid/tanh in deep networks** (vanishing gradient)
- **Combine with batch normalization** for stability
- **Monitor dead neurons** during training
- **For output layer**: Linear (regression) or Sigmoid (if predicting probability)

The key insight: depth requires gradient preservation, which only ReLU-family activations provide.`,
    keyPoints: [
      'Sigmoid/tanh cause vanishing gradients in deep networks (gradient^n ≈ 0)',
      'ReLU maintains gradient=1 for positive activations, enabling deep learning',
      'For 10+ layer networks, ReLU is essentially mandatory',
      'Leaky ReLU preferred over ReLU to prevent dying neurons',
      'Combine ReLU with batch normalization for stability',
      'Financial models require fast adaptation - ReLU enables this',
      'Output layer depends on task: Linear for regression, Sigmoid for probability',
    ],
  },
  {
    id: 'activation-functions-dq-3',
    question:
      'Softmax is used for multi-class classification. Explain how the temperature parameter in softmax affects the output distribution, and discuss when you would want to use temperature scaling in practice, particularly in the context of model calibration or knowledge distillation.',
    sampleAnswer: `The temperature parameter in softmax is a powerful tool for controlling the "sharpness" or "smoothness" of probability distributions, with important applications in model calibration and knowledge distillation:

**Softmax with Temperature:**

Standard softmax:
\`\`\`
softmax (zᵢ) = exp (zᵢ) / Σⱼ exp (zⱼ)
\`\`\`

Softmax with temperature T:
\`\`\`
softmax_T(zᵢ) = exp (zᵢ/T) / Σⱼ exp (zⱼ/T)
\`\`\`

**Effect of Temperature:**

1. **T = 1**: Standard softmax (default)

2. **T < 1** (e.g., T=0.5): "Sharper" distribution
   - Increases difference between probabilities
   - Winner becomes more confident
   - Lower entropy
   - More "decisive" predictions

3. **T > 1** (e.g., T=2): "Smoother" distribution
   - Decreases difference between probabilities
   - More uniform distribution
   - Higher entropy
   - More "uncertain" predictions

4. **T → 0**: Approaches one-hot (argmax)
   - All probability to highest logit
   - No uncertainty

5. **T → ∞**: Approaches uniform distribution
   - Equal probability to all classes
   - Maximum uncertainty

**Mathematical Example:**

\`\`\`python
def softmax_temperature (logits, T=1.0):
    """Softmax with temperature"""
    logits_scaled = logits / T
    exp_logits = np.exp (logits_scaled - np.max (logits_scaled))
    return exp_logits / np.sum (exp_logits)

# Example logits for 3-class problem
logits = np.array([2.0, 1.0, 0.5])

print("Effect of Temperature:")
for T in [0.5, 1.0, 2.0, 5.0]:
    probs = softmax_temperature (logits, T)
    entropy = -np.sum (probs * np.log (probs + 1e-10))
    print(f"\\nT = {T}:")
    print(f"  Probabilities: {probs}")
    print(f"  Max prob: {np.max (probs):.3f}")
    print(f"  Entropy: {entropy:.3f}")
\`\`\`

**Output:**
\`\`\`
T = 0.5:
  Probabilities: [0.756 0.211 0.033]
  Max prob: 0.756
  Entropy: 0.715
  → Sharp, confident

T = 1.0:
  Probabilities: [0.558 0.307 0.135]
  Max prob: 0.558
  Entropy: 0.965
  → Standard

T = 2.0:
  Probabilities: [0.436 0.329 0.235]
  Max prob: 0.436
  Entropy: 1.066
  → Smooth, uncertain

T = 5.0:
  Probabilities: [0.359 0.333 0.308]
  Max prob: 0.359
  Entropy: 1.095
  → Nearly uniform
\`\`\`

**Application 1: Model Calibration**

**Problem**: Neural networks are often overconfident
- Predicts 99% probability when true accuracy is 70%
- Miscalibrated probabilities mislead decision-making

**Solution**: Temperature scaling for calibration

\`\`\`python
def find_optimal_temperature (logits_val, labels_val):
    """
    Find temperature that minimizes calibration error
    using validation set
    """
    def nll_loss(T):
        probs = softmax_temperature (logits_val, T)
        return -np.mean (np.log (probs[range (len (labels_val)), labels_val] + 1e-10))
    
    # Grid search for best T
    T_range = np.linspace(0.1, 10, 100)
    losses = [nll_loss(T) for T in T_range]
    best_T = T_range[np.argmin (losses)]
    return best_T

# Example: Model is overconfident
logits_val = np.array([[3.0, 1.0, 0.5], [2.5, 2.0, 0.3], ...])
labels_val = np.array([0, 1, ...])

optimal_T = find_optimal_temperature (logits_val, labels_val)
print(f"Optimal temperature for calibration: {optimal_T:.2f}")

# If optimal_T > 1, model was overconfident
# Apply this T to test predictions for calibrated probabilities
\`\`\`

**Why This Works:**
- Overconfident models typically have optimal_T > 1
- T > 1 "smooths" probabilities, reducing overconfidence
- Calibrated probabilities better reflect true likelihood

**Trading Example:**
\`\`\`python
# Market regime classification: Bull/Neutral/Bear
logits = model.predict (market_features)

# Without calibration (overconfident)
probs_uncalibrated = softmax (logits)  # [0.95, 0.03, 0.02]
# → Model is 95% sure it's bull market, might be wrong!

# With calibration (T=2.5 found optimal)
probs_calibrated = softmax_temperature (logits, T=2.5)  # [0.65, 0.25, 0.10]
# → More realistic uncertainty, better for decision-making
\`\`\`

**Application 2: Knowledge Distillation**

**Goal**: Train small "student" model to mimic large "teacher" model

**Challenge**: One-hot labels are too "hard"
- Label: [0, 1, 0] provides no information about relative similarity of wrong classes
- Hard to learn nuanced decision boundaries

**Solution**: Use "soft" targets with high temperature

\`\`\`python
def knowledge_distillation (teacher_model, student_model, X_train, y_train, T=4.0):
    """
    Train student model using soft targets from teacher
    
    Args:
        teacher_model: Large, accurate model
        student_model: Small, fast model to train
        T: Temperature for soft targets (typically 2-10)
    """
    # Get teacher's soft predictions
    teacher_logits = teacher_model.predict(X_train)
    soft_targets = softmax_temperature (teacher_logits, T)
    
    # Train student on soft targets (higher temperature)
    student_model.train(
        X_train,
        soft_targets,  # Soft labels, not hard one-hot
        temperature=T   # Use same T for student output
    )
    
    # At inference time, use T=1 (standard softmax)
    return student_model
\`\`\`

**Why High Temperature Helps:**
- Teacher outputs: [0.90, 0.08, 0.02] at T=1
- With T=4: [0.55, 0.30, 0.15] - "softer" targets
- Soft targets encode:
  - Which class is most likely (0.55 vs 0.30)
  - Relative similarity of classes (0.30 vs 0.15)
  - Uncertainty in decision boundary
- Student learns more nuanced boundaries

**Trading Example:**
\`\`\`python
# Distill large ensemble model into fast single model for real-time trading

# Teacher: Ensemble of 10 models (slow, accurate)
teacher_logits = ensemble.predict (features)  # [2.5, 1.8, 0.3]

# Standard softmax (hard targets)
hard_targets = softmax (teacher_logits)  # [0.67, 0.30, 0.03]
# → Student learns: "Class 0 is right, others are wrong"

# Soft targets with T=5
soft_targets = softmax_temperature (teacher_logits, T=5.0)  # [0.45, 0.38, 0.17]
# → Student learns: "Class 0 slightly better than 1, both better than 2"
# → Preserves nuanced decision boundaries
# → Better generalization

# Result: Fast student model performs nearly as well as slow teacher
\`\`\`

**Application 3: Sampling and Exploration**

**Use Case**: Text generation, strategy exploration

**Low Temperature (T < 1)**: Exploitation
- More deterministic outputs
- Sticks to high-probability choices
- Good for final predictions

**High Temperature (T > 1)**: Exploration
- More random outputs
- Tries lower-probability choices
- Good for diversity, exploration

**Trading Strategy Selection Example:**
\`\`\`python
# Multi-strategy trading system
strategy_logits = [3.0, 2.5, 1.0, 0.5]  # Quality scores for 4 strategies

# Exploitation (T=0.5): Always pick best strategy
probs_exploit = softmax_temperature (strategy_logits, T=0.5)  # [0.53, 0.34, 0.09, 0.04]
# → Almost always use strategy 0

# Exploration (T=2.0): Try different strategies
probs_explore = softmax_temperature (strategy_logits, T=2.0)  # [0.35, 0.31, 0.21, 0.13]
# → More diversity, discover robust strategies
\`\`\`

**Practical Guidelines:**

**For Calibration:**
- Use validation set to find optimal T
- Typically T ∈ [1, 3] for modern networks
- Apply only at test time (don't retrain)
- Essential for risk-sensitive applications (medicine, trading)

**For Knowledge Distillation:**
- Use T ∈ [2, 10] during training
- Higher T for more complex teachers
- Revert to T=1 at inference
- Can achieve 90% of teacher performance with 10% of parameters

**For Sampling:**
- T < 1: Confident, repetitive (boring)
- T = 1: Standard
- T > 1: Creative, diverse (potentially nonsensical)
- Adjust based on application needs

**Financial ML Considerations:**

1. **Model Calibration Critical**:
   - Probabilities guide position sizing
   - Miscalibration causes over/under-betting
   - Use temperature scaling to calibrate

2. **Real-time Constraints**:
   - Distill large ensembles to fast models
   - Maintain accuracy with speed
   - T=3-5 works well for distillation

3. **Strategy Diversity**:
   - Use temperature in strategy selection
   - Balance exploitation vs exploration
   - Adapt T based on market conditions

**Conclusion:**

Temperature scaling is a simple but powerful technique:
- **T < 1**: Sharpens probabilities (confidence)
- **T = 1**: Standard softmax
- **T > 1**: Smooths probabilities (uncertainty)

Applications:
- Calibration: Correct overconfident models
- Distillation: Transfer knowledge to small models
- Sampling: Control exploration vs exploitation

In financial ML, proper calibration via temperature scaling is essential for reliable decision-making under uncertainty.`,
    keyPoints: [
      'Temperature T scales logits before softmax: softmax (z/T)',
      'T < 1 sharpens distribution (more confident), T > 1 smooths it (less confident)',
      'For calibration: Find optimal T on validation set to correct overconfident predictions',
      'For knowledge distillation: Use T=2-10 to create "soft targets" that preserve nuanced information',
      'Soft targets encode relative class similarities, improving student model generalization',
      'In trading, calibration is critical - miscalibrated probabilities lead to poor position sizing',
      'Temperature enables exploitation (T<1) vs exploration (T>1) trade-off in strategy selection',
    ],
  },
];
