import { QuizQuestion } from '../../../types';

export const hyperparameterTuningQuiz: QuizQuestion[] = [
  {
    id: 'hyperparameter-tuning-dq-1',
    question:
      'Compare and contrast grid search vs random search for hyperparameter tuning. When would you choose one over the other? Explain with a concrete scenario.',
    sampleAnswer: `Grid search exhaustively tries all combinations in a predefined grid, while random search samples random combinations from distributions.

**Grid Search:**
- Pros: Guaranteed to find best combination within the grid, reproducible
- Cons: Exponentially expensive (e.g., 5 params × 5 values = 3,125 combinations), wastes computation on unpromising regions
- Best for: Small hyperparameter spaces (2-3 params), fine-tuning around a known good region

**Random Search:**
- Pros: Much faster (often 10-50x), explores continuous spaces, finds good solutions with fewer iterations
- Cons: No guarantee of finding optimal, less reproducible (though can set random_state)
- Best for: Large spaces, initial exploration, limited compute budget

**Concrete Scenario:**
Training a neural network with 5 hyperparameters (learning_rate, batch_size, dropout, hidden_layers, optimizer):
1. Start with random search (100 iterations) to explore the space
2. Identify the best configuration
3. Use grid search to fine-tune around that configuration
4. This hybrid approach is 10x faster than pure grid search while achieving similar performance

**Key Insight:** Random search works because not all hyperparameters are equally important. It gives you more coverage of important parameters than grid search with the same budget.`,
    keyPoints: [
      'Grid search: exhaustive but exponentially expensive',
      'Random search: efficient, good for high-dimensional spaces',
      'Choose grid for small spaces, random for exploration',
      'Hybrid approach: random first, then grid for fine-tuning',
      'Random search often finds comparable solutions with far fewer iterations',
    ],
  },
  {
    id: 'hyperparameter-tuning-dq-2',
    question:
      'Explain why you should use nested cross-validation when performing hyperparameter tuning. What problem does it solve, and what is the computational cost?',
    sampleAnswer: `Nested cross-validation solves the problem of overfitting to the validation set during hyperparameter tuning.

**The Problem:**
When you tune hyperparameters using CV on the entire training set, you're indirectly fitting to the validation folds. The reported CV score is optimistically biased because you've optimized to maximize it. This leads to overly optimistic performance estimates.

**Nested CV Solution:**
- Outer loop: K-fold CV for final performance evaluation (e.g., 5 folds)
- Inner loop: K-fold CV for hyperparameter selection (e.g., 5 folds)
- For each outer fold:
  1. Use inner CV to find best hyperparameters on training folds
  2. Train model with those hyperparameters
  3. Evaluate on outer test fold (never seen during tuning)
- Result: Unbiased estimate of model performance with tuning

**Computational Cost:**
- Regular CV: K × (tuning iterations) = 5 × 50 = 250 model fits
- Nested CV: K_outer × K_inner × (tuning iterations) = 5 × 5 × 50 = 1,250 model fits
- **5x more expensive** but gives honest performance estimate

**When to Use:**
- Critical applications where you need accurate performance estimates
- Research papers and model comparisons
- When you have sufficient compute budget
- Skip it for rapid prototyping or when compute is limited`,
    keyPoints: [
      'Nested CV prevents overfitting to the validation set',
      'Outer loop evaluates performance, inner loop tunes hyperparameters',
      'Provides unbiased performance estimate',
      'K_outer × K_inner more expensive than regular CV',
      'Essential for rigorous model evaluation and comparison',
    ],
  },
  {
    id: 'hyperparameter-tuning-dq-3',
    question:
      'You have a limited compute budget to tune hyperparameters for a deep learning model. Describe a practical, efficient strategy that balances exploration and computational cost.',
    sampleAnswer: `Here's an efficient hyperparameter tuning strategy for limited compute:

**Step 1: Establish Baseline (1-2 hours)**
- Train with default hyperparameters
- Measure baseline performance
- This gives you a reference point

**Step 2: Coarse Random Search (4-6 hours)**
- Use random search with 20-50 iterations
- Wide ranges for all hyperparameters
- Use 3-fold CV (instead of 5) to save time
- Early stopping: terminate bad configs early
- Focus on most impactful hyperparameters first:
  * Learning rate (log scale: 0.0001 to 0.1)
  * Batch size (32, 64, 128, 256)
  * Architecture (number of layers)

**Step 3: Focus on Important Hyperparameters (2-4 hours)**
- Analyze coarse search results
- Identify 2-3 most important hyperparameters
- Run random search with narrower ranges
- 5-fold CV now for better estimates

**Step 4: Fine-Tuning (2-3 hours)**
- Small grid search around best configuration
- Use full training set
- Monitor test set periodically

**Step 5: Final Training (2-4 hours)**
- Train best configuration from scratch
- Use full training data
- Early stopping based on validation set

**Time-Saving Techniques:**
- Learning rate finder: 1 epoch to find good range
- Reduce CV folds (3 instead of 5)
- Early stopping: kill bad configs after few epochs
- Warm start: resume from checkpoints
- Use smaller subset of data for initial search

**Total: ~15-20 hours** vs 100+ hours for exhaustive search`,
    keyPoints: [
      'Start with baseline for reference',
      'Coarse random search for exploration',
      'Focus on most impactful hyperparameters',
      'Fine-tune around best configuration',
      'Use early stopping and reduced CV folds to save time',
    ],
  },
];
