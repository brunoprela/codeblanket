import { QuizQuestion } from '../../../types';

export const biasVarianceTradeoffQuiz: QuizQuestion[] = [
  {
    id: 'bias-variance-tradeoff-dq-1',
    question:
      "Explain the bias-variance tradeoff using a concrete example. Why can't we minimize both simultaneously, and how do you find the optimal balance?",
    sampleAnswer: `The bias-variance tradeoff is the fundamental tension in machine learning between models that are too simple (high bias) and too complex (high variance). Understanding this tradeoff is key to building models that generalize well.

**The Mathematical Decomposition:**

Total Error = Bias² + Variance + Irreducible Error

Where:
- Bias = Error from wrong model assumptions
- Variance = Error from sensitivity to training data
- Irreducible Error = Inherent noise (cannot reduce)

**Concrete Example: Polynomial Regression**

Task: Fit y = sin(x) + noise using polynomial regression

**High Bias (Degree 1 - Linear):**
- Model: y = ax + b
- Cannot capture sine wave pattern (linear assumption wrong)
- Training error: High (0.28)
- Test error: High (0.29)
- Problem: Model too simple, **underfits** both train and test

**Optimal Balance (Degree 3-4):**
- Model: y = ax³ + bx² + cx + d
- Can approximate sine wave reasonably
- Training error: Medium (0.08)
- Test error: Medium (0.09)
- Balance: Captures main pattern without overfitting

**High Variance (Degree 15):**
- Model: y = a₁₅x¹⁵ + a₁₄x¹⁴ + ... + a₁x + b
- Fits training data perfectly, including noise
- Training error: Near zero (0.002)
- Test error: Very high (0.42)
- Problem: Model **overfits** training data, fails on new data

**Why Can't We Minimize Both?**

1. **Fundamental Tradeoff:**
   - Increasing model complexity decreases bias (better fit)
   - But increases variance (more sensitive to data)
   - Decreasing complexity does the opposite

2. **Limited Information:**
   - Training data is finite and noisy
   - Simple models ignore some signal (high bias)
   - Complex models learn noise as signal (high variance)

3. **The Tradeoff Curve:**
   \`\`\`
   Test Error = Bias² + Variance + Noise
   
   Complexity ↑ → Bias ↓, Variance ↑
    Complexity ↓ → Bias ↑, Variance ↓

    Optimal complexity minimizes their sum
        \`\`\`

**Finding the Optimal Balance:**

**1. Use Learning Curves:**
- Plot train/validation error vs training set size
- High bias: Both errors plateau at high value
- High variance: Large gap between train and validation
- Optimal: Low errors with small gap

**2. Cross-Validation:**
- Try multiple complexity levels
- Choose complexity with best CV performance
- Example: Test polynomial degrees 1-15, pick best

**3. Regularization:**
- Add penalty for complexity
- Ridge/Lasso control complexity smoothly
- Find optimal regularization strength via CV

**4. Analyze Error Patterns:**
\`\`\`
Symptoms                        Diagnosis        Fix
─────────────────────────────────────────────────────────
Train: High, Test: High        High Bias       ↑ Complexity
Train: Low, Test: High         High Variance   ↓ Complexity or ↑ Data
Train: Low, Test: Low          Good Balance    ✓ Deploy
    \`\`\`

**Real-World Stock Prediction Example:**

**Model A: Moving Average (Simple)**
- Just averages last 5 days
- High bias: Can't capture complex patterns
- Train R²: 0.3, Test R²: 0.28 (consistent but poor)
- Underfits: Misses trends, reversals, seasonality

**Model B: Random Forest (Moderate)**
- 100 trees, max_depth=10
- Optimal balance
- Train R²: 0.85, Test R²: 0.78 (good generalization)
- Fits well: Captures main patterns

**Model C: Deep Network (Complex)**
- 10 layers, memorizes training data
- High variance: Learns market noise
- Train R²: 0.99, Test R²: 0.45 (huge gap!)
- Overfits: Perfect on history, fails on future

**Practical Strategy:**

1. Start Simple:
   - Begin with high-bias model (linear, shallow tree)
   - Establishes baseline

2. Increase Complexity Gradually:
   - Add features, increase model capacity
   - Monitor validation performance

3. Stop When:
   - Validation error starts increasing (overfitting)
   - Or validation error plateaus (optimal)

4. Use Regularization:
   - Instead of discrete complexity levels
   - Smooth tradeoff via penalty strength

**Key Insight:** The optimal model is NOT the one with zero training error—it's the one that best predicts unseen data. The bias-variance tradeoff forces us to accept some bias (systematic error) to reduce variance (error variability), or vice versa. The sweet spot depends on the data, noise level, and how much training data you have.`,
    keyPoints: [
      'Total error = Bias² + Variance + Irreducible noise',
      "Bias (underfitting): Model too simple, can't capture pattern",
      'Variance (overfitting): Model too complex, learns noise',
      'Increasing complexity: Decreases bias, increases variance',
      'Cannot minimize both simultaneously—must find optimal balance',
      'Use learning curves and cross-validation to diagnose',
      'Optimal model balances fitting signal while ignoring noise',
    ],
  },
  {
    id: 'bias-variance-tradeoff-dq-2',
    question:
      'Describe how learning curves help diagnose whether a model has high bias or high variance, and explain what actions you would take in each case.',
    sampleAnswer: `Learning curves plot training and validation errors as functions of training set size. They're powerful diagnostic tools that reveal whether your model suffers from bias or variance problems—and crucially, whether more data will help.

**What Learning Curves Show:**

X-axis: Training set size (e.g., 100, 500, 1000, 5000 samples)
Y-axis: Error (MSE, cross-entropy, etc.)
Two lines: Training error (blue) and Validation error (red)

**Pattern 1: High Bias (Underfitting)**

\`\`\`
    Error
  │
  │     ┌───────────────────  Validation
  │    ┌┘
  │   ┌┘
  │  ┌┘
  │ ┌┴──────────────────────  Training
  │┌┘
  └────────────────────────► Training Size
     Both curves plateau at HIGH error
        \`\`\`

**Characteristics:**
- Training error starts low, increases to plateau
- Validation error starts high, decreases to plateau
- **Both converge to high error** (> acceptable threshold)
- Small gap between curves
- Curves flatten—more data doesn't help!

**Interpretation:**
- Model too simple to capture pattern
- Increasing training data won't help (curves plateaued)
- Model has reached its representational limit

**Example:**
Linear model for sine wave:
- Train error: 0.28
- Val error: 0.30
- Both high, small gap
- → Model fundamentally can't fit sine wave

**Actions for High Bias:**

1. **Increase Model Complexity:**
   - Polynomial: degree 1 → 3
   - Tree: max_depth 3 → 10
   - Neural net: Add layers/neurons

2. **Add Features:**
   - Polynomial features (x² , x³)
   - Feature interactions (x₁ × x₂)
   - Domain-specific features

3. **Reduce Regularization:**
   - Lower L1/L2 penalty
   - Increase dropout rate → reduce
   - More model capacity

4. **Train Longer:**
   - More epochs (if not converged)

5. **Try Different Model:**
   - Linear → Tree-based
   - Tree → Neural network

**DON'T:**
- Add more data (won't help—curves plateaued)
- Increase regularization (makes it worse)

**Pattern 2: High Variance (Overfitting)**

\`\`\`
    Error
  │
  │              ┌──────────  Validation
  │          ┌───┘
  │       ┌──┘
  │    ┌──┘
  │ ┌──┘
  │┌┴──────────────────────  Training
  └────────────────────────► Training Size
     Large gap between curves
        \`\`\`

**Characteristics:**
- Training error very low (near zero)
- Validation error much higher
- **Large gap between curves** (main diagnostic)
- Validation error decreasing but not converged
- More data would likely help (curves not flat)

**Interpretation:**
- Model too complex, memorizing training data
- Learning noise as if it were signal
- Would benefit from more data (if available)

**Example:**
Degree-15 polynomial:
- Train error: 0.002 (nearly perfect)
- Val error: 0.42 (terrible)
- Huge gap of 0.418
- → Overfitting dramatically

**Actions for High Variance:**

1. **Get More Training Data (Primary Fix):**
   - If curves haven't converged, more data helps
   - Collect more samples if possible
   - Data augmentation (images, text)

2. **Reduce Model Complexity:**
   - Polynomial: degree 15 → 4
   - Tree: max_depth unlimited → 5
   - Neural net: Remove layers

3. **Regularization:**
   - Add L1/L2 penalty (Ridge, Lasso)
   - Dropout in neural networks
   - Early stopping

4. **Feature Selection:**
   - Remove irrelevant features
   - Use L1 regularization for automatic selection

5. **Ensemble Methods:**
   - Bagging (Random Forest)
   - Reduces variance by averaging

**DON'T:**
- Increase model complexity (makes it worse)
- Remove regularization (makes it worse)

**Pattern 3: Good Fit (Optimal)**

\`\`\`
    Error
  │
  │     ┌─────────────────  Validation
  │   ┌─┘
  │  ┌┘
  │ ┌┴──────────────────────  Training
  │┌┘
  └────────────────────────► Training Size
     Small gap, both low
        \`\`\`

**Characteristics:**
- Both errors low (< acceptable threshold)
- Small gap between curves
- Both converging
- More data provides diminishing returns

**Interpretation:**
- Model captures main pattern
- Generalizes well
- Near-optimal complexity

**Real-World Diagnostic Example:**

**Scenario: House Price Prediction**

Try 3 models, plot learning curves:

**Model A: Linear Regression**
Learning curve shows:
- Train: 0.25 MSE (stable)
- Val: 0.27 MSE (stable)
- Small gap, but both high
Diagnosis: HIGH BIAS
Action: Try polynomial features or tree-based model

**Model B: Random Forest (depth=20)**
Learning curve shows:
- Train: 0.01 MSE (decreasing)
- Val: 0.15 MSE (decreasing but gap)
- Gap: 0.14, curves haven't flattened
Diagnosis: HIGH VARIANCE (but more data would help)
Action: Reduce depth to 10 or get more data

**Model C: Random Forest (depth=5)**
Learning curve shows:
- Train: 0.05 MSE
- Val: 0.06 MSE
- Small gap (0.01), both low
Diagnosis: GOOD FIT
Action: Deploy this model!

**Practical Tips:**

1. **Always Plot Learning Curves:**
   - Don't just look at final errors
   - Curves reveal what will help

2. **Check Convergence:**
   - Flat curves → More data won't help (bias problem)
   - Decreasing curves → More data will help (variance problem)

3. **Gap is Key Diagnostic:**
   - Small gap + high error → Bias
   - Large gap → Variance
   - Small gap + low error → Good

4. **Use Early in Development:**
   - Don't wait until final model
   - Helps decide next steps

**Key Insight:** Learning curves tell you not just WHETHER your model has problems, but WHAT TYPE of problems and WHAT TO DO about them. High bias and high variance require opposite solutions—diagnosis is critical before wasting time on the wrong fix.`,
    keyPoints: [
      'Learning curves plot train/val error vs training set size',
      'High bias: Both errors plateau at high value, small gap',
      'High variance: Low train error, high val error, large gap',
      "Converged flat curves → more data won't help (bias problem)",
      'Decreasing curves → more data will help (variance problem)',
      'Fix bias: Increase complexity, add features, reduce regularization',
      'Fix variance: More data, reduce complexity, add regularization',
      'Learning curves guide whether to collect more data',
    ],
  },
  {
    id: 'bias-variance-tradeoff-dq-3',
    question:
      'A neural network achieves 99% training accuracy but only 70% test accuracy. Identify the problem and provide five specific techniques to address it, explaining how each helps.',
    sampleAnswer: `This is a classic case of **high variance (overfitting)**. The model has memorized the training data (99% accuracy) but fails to generalize to new data (70% test accuracy). The 29% gap indicates severe overfitting.

**Problem Diagnosis:**

Symptoms:
- High training accuracy: 99%
- Low test accuracy: 70%
- Large gap: 29 percentage points
- **Verdict: HIGH VARIANCE**

The network has enough capacity to memorize training examples, including noise and outliers, rather than learning general patterns.

**Five Specific Techniques:**

**1. Regularization (L2/Weight Decay)**

**How it works:**
Add penalty to loss function proportional to squared weights:
\`\`\`
    Loss_total = Loss_original + λ × Σ(weights²)
        \`\`\`

**Why it helps:**
- Forces network to use smaller weights
- Prevents any single weight from dominating
- Smoother, simpler functions (reduces complexity)
- Makes model less sensitive to individual training examples

**Implementation:**
\`\`\`python
# PyTorch
    optimizer = torch.optim.Adam(model.parameters(),
        lr = 0.001,
        weight_decay = 0.01)  # L2 regularization

# TensorFlow / Keras
    model.add(Dense(128, kernel_regularizer = l2(0.01)))
        \`\`\`

**Expected improvement:** 70% → 80% test accuracy

**2. Dropout**

**How it works:**
Randomly "drop" (zero out) neurons during training with probability p:
- During training: Randomly disable p% of neurons each batch
- During testing: Use all neurons (scaled by keep probability)

**Why it helps:**
- Prevents co-adaptation of neurons (neurons can't rely on specific others)
- Forces network to learn redundant representations
- Effectively trains ensemble of sub-networks
- Reduces sensitivity to specific training examples

**Implementation:**
\`\`\`python
# PyTorch
    model = nn.Sequential(
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Drop 50 % of neurons
    nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),  # Drop 30 % of neurons
    nn.Linear(256, num_classes)
    )

# Keras
    model.add(Dropout(0.5))
        \`\`\`

**Typical dropout rates:**
- 0.2-0.5 for hidden layers
- 0.5 is common default
- Higher dropout (0.7) for very high variance

**Expected improvement:** 70% → 78% test accuracy

**3. Early Stopping**

**How it works:**
Monitor validation error during training:
- Track validation accuracy every epoch
- Stop when validation accuracy stops improving
- Restore weights from best validation epoch

**Why it helps:**
- Network initially learns general patterns (test accuracy improves)
- Later starts memorizing training data (test accuracy degrades)
- Early stopping catches the sweet spot
- Prevents overfitting without changing architecture

**Implementation:**
\`\`\`python
# PyTorch example
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(1000):
        train(model, train_loader)
    val_acc = evaluate(model, val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
    save_checkpoint(model)  # Save best model
    patience_counter = 0
    else:
    patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
    break

# Restore best model
    load_checkpoint(model)
        \`\`\`

**Expected improvement:** 70% → 75% test accuracy
**Bonus:** Faster training (stops early)

**4. Data Augmentation**

**How it works:**
Artificially expand training set by creating modified versions:
- Images: rotation, flipping, cropping, color jittering
- Text: synonym replacement, back-translation
- Tabular: add Gaussian noise, interpolation

**Why it helps:**
- Increases effective training set size
- Model sees more variation, learns invariances
- Reduces memorization (can't memorize augmented versions)
- Forces learning of robust features

**Implementation:**
\`\`\`python
# Image augmentation(PyTorch)
from torchvision import transforms

train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
            transforms.RandomCrop(224, padding = 4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

# Text augmentation
def augment_text(text):
    # Synonym replacement
    words = text.split()
    for i in range(len(words)):
        if random.random() < 0.3:
            words[i] = get_synonym(words[i])
    return ' '.join(words)
        \`\`\`

**Expected improvement:** 70% → 82% test accuracy
**Best for:** Small datasets with natural augmentation

**5. Simplify Architecture (Reduce Capacity)**

**How it works:**
Reduce model complexity:
- Fewer layers: 10 layers → 5 layers
- Fewer neurons: 1024 units → 256 units
- Smaller kernel sizes in CNNs
- Remove unnecessary connections

**Why it helps:**
- Less capacity = less ability to memorize
- Forces model to learn most important patterns
- Occam's Razor: simplest model that fits is best
- Reduces search space for optimization

**Implementation:**
\`\`\`python
# Original(overfitting)
    model = nn.Sequential(
        nn.Linear(100, 1024),  # Too large
    nn.ReLU(),
        nn.Linear(1024, 1024),  # Unnecessary depth
    nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )

# Simplified(better generalization)
    model = nn.Sequential(
        nn.Linear(100, 256),  # Smaller
    nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10)
    )
        \`\`\`

**Expected improvement:** 70% → 76% test accuracy

**Comprehensive Strategy:**

**Phase 1: Quick Wins (Don't change architecture)**
1. Add dropout (0.5): Expected 70% → 78%
2. Add early stopping: Prevents further degradation
3. Add L2 regularization (0.01): Additional 2-3%

**Phase 2: If Still Overfitting**
4. Data augmentation (if applicable): Additional 5-8%
5. Reduce architecture if needed: Additional 2-4%

**Expected Final Result:**
Combining techniques: 70% → 85% test accuracy

**Implementation Order:**
1. Start with dropout + early stopping (easiest, fast to implement)
2. Add L2 regularization
3. If still overfitting, try data augmentation
4. As last resort, simplify architecture

**Monitoring:**
\`\`\`
                    Train Acc  Test Acc  Gap
Original            99 % 70 % 29 %  ← Problem
        + Dropout           92 % 78 % 14 %  ← Improvement
            + L2 Reg            88 % 81 % 7 %   ← Better
                + Early Stop        85 % 83 % 2 %   ← Good
                    + Data Aug          87 % 85 % 2 %   ← Excellent
                        \`\`\`

**Key Insight:** Don't just apply one technique—combine multiple. Start with non-invasive regularization (dropout, L2, early stopping), then add data or reduce complexity if needed. The goal isn't 100% training accuracy—it's closing the gap between train and test performance.`,
    keyPoints: [
      '99% train, 70% test = high variance (overfitting) with 29% gap',
      'Regularization (L2): Penalizes large weights, simpler functions',
      'Dropout: Randomly disables neurons, prevents co-adaptation',
      'Early stopping: Stops before memorization, catches sweet spot',
      'Data augmentation: Expands training set, reduces memorization',
      'Simplify architecture: Less capacity = less ability to overfit',
      'Combine multiple techniques for best results',
      'Monitor gap reduction, not just test accuracy increase',
    ],
  },
];
