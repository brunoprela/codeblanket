import { QuizQuestion } from '../../../types';

export const trainingNeuralNetworksQuiz: QuizQuestion[] = [
  {
    id: 'training-nn-q1',
    question:
      'Discuss the trade-offs between small and large batch sizes in neural network training. How does batch size affect convergence speed, generalization, memory usage, and GPU utilization? What is the "linear scaling rule" for learning rates?',
    sampleAnswer: `Batch size is a critical hyperparameter that affects training dynamics, computational efficiency, and final model performance. Understanding the trade-offs is essential for effective training:

**Small Batch Sizes (32-64):**

Advantages:
- More frequent weight updates (faster convergence in steps)
- Noisier gradients help escape sharp minima
- Better generalization (flat minima)
- Lower memory requirements
- Can train larger models

Disadvantages:
- Slower per-epoch wall-clock time
- Less efficient GPU utilization
- Noisier training curves
- May need more epochs

**Large Batch Sizes (256-512+):**

Advantages:
- Better GPU utilization (parallelism)
- Faster per-epoch wall-clock time
- More stable, smoother gradients
- Efficient for distributed training

Disadvantages:
- Fewer weight updates per epoch
- Converge to sharp minima (poor generalization)
- Higher memory requirements
- May need learning rate adjustment

**The Linear Scaling Rule:**

When increasing batch size by k, multiply learning rate by k:
\`\`\`python
# Base: batch_size=32, lr=0.001
# Scaled: batch_size=256 (8x larger)
lr_scaled = 0.001 * 8 = 0.008
\`\`\`

**Why it works:**
- Larger batches = less noisy gradient estimates
- Can take larger steps safely
- Maintains similar learning dynamics

**Modern Best Practices:**
1. Start with 32-64 (good default)
2. Increase until GPU memory full or convergence slows
3. Use linear scaling rule for LR
4. Add warmup for very large batches (>1024)`,
    keyPoints: [
      'Small batches: more updates, better generalization, lower memory',
      'Large batches: faster per-epoch, better GPU utilization, higher memory',
      'Small batches converge to flat minima (better generalization)',
      'Large batches converge to sharp minima (worse generalization)',
      'Linear scaling rule: double batch size → double learning rate',
      'Modern default: 32-64, increase until GPU full',
      'Very large batches need warmup and careful tuning',
    ],
  },
  {
    id: 'training-nn-q2',
    question:
      'Explain why learning rate scheduling is important and compare different scheduling strategies (step decay, exponential decay, cosine annealing, reduce on plateau). When might you choose one strategy over another? What is the purpose of learning rate warmup?',
    sampleAnswer: `Learning rate scheduling is crucial because a fixed learning rate either prevents convergence (too high) or causes slow training (too low). Adaptive schedules provide fast initial progress and fine-tuning:

**Why Scheduling Matters:**
- High LR: Fast progress but can't converge (oscillates around minimum)
- Low LR: Converges but extremely slow
- Solution: Start high, decay gradually

**Step Decay:**
\`\`\`python
lr = initial_lr * (0.5 ** (epoch // 10))
\`\`\`
- Simple, interpretable
- Discrete jumps
- Use when: Fixed training budget known

**Exponential Decay:**
\`\`\`python
lr = initial_lr * (0.95 ** epoch)
\`\`\`
- Smooth, continuous decay
- Gradual reduction
- Use when: Want smooth transitions

**Cosine Annealing:**
\`\`\`python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * epoch / T))
\`\`\`
- Smooth decay following cosine curve
- Popular in modern training
- Use when: Fixed epochs known, want smooth schedule

**Reduce on Plateau:**
\`\`\`python
if val_loss doesn't improve for N epochs:
    lr = lr * 0.5
\`\`\`
- Adaptive to training progress
- Reduces when stuck
- Use when: Unknown optimal training length

**Learning Rate Warmup:**
\`\`\`python
if epoch < warmup_epochs:
    lr = initial_lr * (epoch / warmup_epochs)
\`\`\`

**Why warmup helps:**
- Prevents instability from large initial gradients
- Especially important for large batch sizes
- Allows higher max learning rates
- Standard in Transformer training

**Modern Default:**
Warmup (5-10 epochs) + Cosine Annealing`,
    keyPoints: [
      'Fixed LR either too high (no convergence) or too low (too slow)',
      'Scheduling: fast initial progress + fine-tuning convergence',
      'Step decay: simple, discrete jumps every N epochs',
      'Cosine annealing: smooth decay, popular for fixed budgets',
      'Reduce on plateau: adaptive, reduces when validation plateaus',
      'Warmup: linearly increase LR first few epochs',
      'Warmup prevents instability from large initial gradients',
      'Modern standard: warmup + cosine annealing',
    ],
  },
  {
    id: 'training-nn-q3',
    question:
      'How do you diagnose training problems from training curves? What do different patterns (train/val loss diverging, both losses plateauing high, loss spiking, NaN losses) indicate, and what are appropriate solutions for each?',
    sampleAnswer: `Training curves reveal what's happening during training. Learning to interpret them is essential for debugging:

**Pattern 1: Train↓ Val↑ (Diverging)**
- Diagnosis: **Overfitting**
- Train loss decreasing, val loss increasing
- Model memorizing training data
- Solutions:
  * Add regularization (dropout, L2)
  * Early stopping
  * Get more training data
  * Reduce model capacity
  * Data augmentation

**Pattern 2: Both High and Flat**
- Diagnosis: **Underfitting**
- Both losses high and not decreasing
- Model lacks capacity
- Solutions:
  * Increase model size
  * Train longer
  * Reduce regularization
  * Check feature engineering
  * Verify data quality

**Pattern 3: Loss Spikes**
- Diagnosis: **Instability**
- Sudden jumps in loss
- Learning rate too high or bad batch
- Solutions:
  * Reduce learning rate
  * Gradient clipping
  * Check data for outliers
  * Increase batch size
  * Use batch normalization

**Pattern 4: NaN Losses**
- Diagnosis: **Numerical Instability**
- Loss becomes NaN (not a number)
- Gradient explosion or invalid operations
- Solutions:
  * Reduce learning rate significantly
  * Add gradient clipping (max_norm=5)
  * Check for division by zero
  * Verify data normalization
  * Check for extreme values

**Pattern 5: Both Decreasing (Healthy)**
- Diagnosis: **Good Training**
- Both losses decreasing
- Val loss tracks train loss
- Keep training!

**Additional Diagnostics:**
- Monitor gradient norms (>100 = explosion, <1e-6 = vanishing)
- Check activation distributions
- Verify learning rate isn't too small
- Compare train/val accuracy gap`,
    keyPoints: [
      'Train↓ Val↑: overfitting → add regularization, early stopping',
      'Both high: underfitting → increase capacity, train longer',
      'Loss spikes: instability → reduce LR, add gradient clipping',
      'NaN losses: numerical issues → lower LR, clip gradients, check data',
      'Both decreasing: healthy training → continue',
      'Monitor gradient norms to diagnose vanishing/exploding',
      'Large train/val gap indicates overfitting',
      'Use multiple metrics: loss, accuracy, gradient norms',
    ],
  },
];
