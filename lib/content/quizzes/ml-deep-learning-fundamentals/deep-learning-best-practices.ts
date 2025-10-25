import { QuizQuestion } from '../../../types';

export const deepLearningBestPracticesQuiz: QuizQuestion[] = [
  {
    id: 'best-practices-q1',
    question:
      'Explain the importance of proper data normalization/standardization in deep learning. Why must statistics be computed only on the training set? What problems can occur if you normalize train/val/test sets independently?',
    sampleAnswer: `Proper data preprocessing is crucial for neural network training. Normalization/standardization ensures features have similar scales, preventing some features from dominating others during training.

**Why Normalization Matters:**
- Neural networks sensitive to input scale
- Features with large values dominate gradients
- Different scales slow convergence
- Proper initialization assumes normalized inputs

**The Critical Rule: Train-Only Statistics**

CORRECT:
\`\`\`python
# Compute statistics on training set ONLY
mean = X_train.mean (axis=0)
std = X_train.std (axis=0)

# Apply to all sets
X_train_norm = (X_train - mean) / std
X_val_norm = (X_val - mean) / std
X_test_norm = (X_test - mean) / std
\`\`\`

WRONG:
\`\`\`python
# Computing independently causes data leakage!
X_train_norm = (X_train - X_train.mean()) / X_train.std()
X_test_norm = (X_test - X_test.mean()) / X_test.std()  # LEAKAGE!
\`\`\`

**Why Independent Normalization Is Wrong:**

1. **Data Leakage**: Test statistics influence preprocessing
2. **Overoptimistic Performance**: Test set appears easier than reality
3. **Production Failure**: Real data uses different statistics
4. **Violates ML Principles**: Test set should be unseen

**Production Implications:**
- Must save training statistics (mean, std)
- Apply same statistics to new data
- Cannot recompute statistics for each prediction

**Key Principle**: Test set must remain completely isolated until final evaluation.`,
    keyPoints: [
      'Normalization ensures features have similar scales',
      'Compute statistics (mean, std) on training set ONLY',
      'Apply training statistics to validation and test sets',
      'Independent normalization causes data leakage',
      'Leakage leads to overoptimistic performance estimates',
      'Production inference must use exact training statistics',
      'Test set isolation is fundamental ML principle',
    ],
  },
  {
    id: 'best-practices-q2',
    question:
      'Describe the learning rate finder technique. How does it work, how do you interpret the results, and why is it more reliable than manually guessing learning rates?',
    sampleAnswer: `The Learning Rate Finder is a systematic approach to finding optimal learning rates, avoiding hours of trial-and-error:

**How It Works:**

1. Start with very small LR (e.g., 1e-7)
2. Train for a few iterations
3. Exponentially increase LR after each iteration
4. Record loss at each LR
5. Plot loss vs LR on log scale
6. Stop when loss explodes

**Implementation:**
\`\`\`python
lrs = []
losses = []
lr = 1e-7

for i in range(100):
    optimizer.param_groups[0]['lr',] = lr
    loss = train_step()
    
    lrs.append (lr)
    losses.append (loss)
    
    lr *= 1.1  # Exponential increase
    
    if loss > 4 * best_loss:  # Stop if exploding
        break
\`\`\`

**Interpreting Results:**

Good LR curve:
- Flat region (LR too low, slow learning)
- Steep descent (optimal LR range)
- Explosion (LR too high)

**Choosing LR:**
- Find steepest descent point
- Choose LR at steepest gradient
- Or slightly before steepest point
- Avoid explosion region

**Why Better Than Guessing:**
- Systematic exploration
- Visual interpretation
- Finds optimal range quickly
- Avoids hours of trial-and-error
- One test reveals optimal LR

**Typical Pattern:**
- LR < 1e-5: Loss barely decreases
- LR ∈ [1e-4, 1e-3]: Steep descent → CHOOSE HERE
- LR > 1e-2: Loss explodes

**Key Insight**: The LR finder saves hours of hyperparameter search by providing visual, systematic guidance for the most important hyperparameter.`,
    keyPoints: [
      'LR finder: exponentially increase LR, plot loss vs LR',
      'Interpret: flat → descent → explosion pattern',
      'Choose LR at steepest descent point',
      'Systematic alternative to trial-and-error',
      'Saves hours of manual hyperparameter search',
      'One test reveals optimal learning rate range',
      'Visual interpretation makes choice clear',
    ],
  },
  {
    id: 'best-practices-q3',
    question:
      'Compare grid search, random search, and Bayesian optimization for hyperparameter tuning. What are the computational costs and effectiveness of each?',
    sampleAnswer: `Different hyperparameter search strategies have different trade-offs in efficiency and effectiveness:

**Grid Search:**
Exhaustively tries all combinations:
\`\`\`python
for lr in [1e-4, 1e-3, 1e-2]:
    for batch_size in [32, 64, 128]:
        for dropout in [0.3, 0.5, 0.7]:
            train (lr, batch_size, dropout)
\`\`\`

**Pros:**
- Complete coverage of grid
- Reproducible
- Simple to implement

**Cons:**
- Exponential growth (3³ = 27 trials)
- Wastes trials on unimportant parameters
- Doesn't explore continuous spaces well

**Random Search:**
Samples randomly from distributions:
\`\`\`python
for trial in range(20):
    lr = 10 ** uniform(-5, -2)
    batch_size = choice([32, 64, 128, 256])
    dropout = uniform(0.2, 0.7)
    train (lr, batch_size, dropout)
\`\`\`

**Pros:**
- Explores more values per parameter
- Better for high dimensions
- Often finds better solutions than grid
- No exponential growth

**Cons:**
- No guarantee of coverage
- Random, not systematic

**Why Random > Grid:**
Some hyperparameters matter more than others. Random search explores more values for important parameters.

**Bayesian Optimization:**
Uses previous trials to guide next ones:
\`\`\`python
for trial in range(50):
    # Model predicts promising hyperparameters
    params = optimizer.suggest()
    score = train (params)
    optimizer.update (params, score)
\`\`\`

**Pros:**
- Most sample-efficient
- Requires fewer trials
- Best for expensive training

**Cons:**
- Complex implementation
- Overhead for simple problems

**When to Use:**
- Grid: 2-3 hyperparameters, small space
- Random: 4+ hyperparameters, quick exploration
- Bayesian: expensive training, need best result

**Modern Practice:** Random search for initial exploration, Bayesian optimization for refinement.`,
    keyPoints: [
      'Grid search: exhaustive, exponential growth, wastes compute',
      'Random search: explores more values, better in high dimensions',
      'Random often outperforms grid (important parameter coverage)',
      'Bayesian optimization: most efficient, uses previous trials',
      'Grid: 2-3 params; Random: 4+ params; Bayesian: expensive training',
      'Random search is modern default for initial exploration',
      'Bayesian optimization for refinement with limited budget',
    ],
  },
];
