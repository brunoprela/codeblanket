/**
 * Multiple choice questions for Model Fine-Tuning Fundamentals section
 */

export const modelFineTuningFundamentalsMultipleChoice = [
  {
    id: 'fine-tune-fundamentals-mc-1',
    question:
      'You fine-tune a model with learning rate 1e-3 for 5 epochs on 5,000 examples. Training loss decreases nicely, but validation accuracy is worse than the base model. What is the MOST likely problem?',
    options: [
      'Learning rate is too low, model is underfitting',
      'Learning rate is too high, model is overfitting or diverging',
      'Not enough training data',
      'Base model was already optimal for this task',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (learning rate too high) is most likely. Signs: Training loss decreases (model is learning the training set), but validation is worse than base model (catastrophic forgetting or overfitting). Learning rate 1e-3 is typically too high for fine-tuning—fine-tuning usually needs 1e-5 to 1e-4 (10-100x smaller than pre-training rates). At 1e-3, the model aggressively overwrites pre-trained weights, destroying general knowledge and overfitting to training data. Solution: Reduce learning rate to 1e-5, use warmup, reduce epochs to 2-3. Option A is wrong—underfitting would show poor training loss. Option C is possible but 5K is usually sufficient. Option D is unlikely—base model performance should be floor, not ceiling.',
  },
  {
    id: 'fine-tune-fundamentals-mc-2',
    question:
      'You compare LoRA (r=8) vs full fine-tuning. LoRA trains 10x faster and costs 20x less, but accuracy is 89% vs 92% for full fine-tuning. What should you try FIRST to improve LoRA performance?',
    options: [
      'Switch to full fine-tuning (accept the cost for better performance)',
      'Increase LoRA rank (r=8 → r=32) and add more target modules',
      'Collect more training data',
      'Increase learning rate for LoRA',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (increase LoRA rank) is the first thing to try. LoRA rank controls adapter capacity: r=8 means each adapter matrix is low-rank (8-dimensional). Increasing to r=16 or r=32 gives adapters more expressiveness. Also, target more modules—if you only adapted attention (q_proj, v_proj), add k_proj, o_proj, and MLP layers (gate_proj, up_proj, down_proj). Expected improvement: 89% → 91% (close to full fine-tuning). Cost: 2-4x more parameters but still 10-50x cheaper than full. Option A gives up too early—LoRA hasn't been fully explored. Option C is generic. Option D is risky—could destabilize training. Sequence: Try B first (30 min test), if still <90%, try C (more data), only use A (full fine-tuning) if LoRA truly can't reach target performance.",
  },
  {
    id: 'fine-tune-fundamentals-mc-3',
    question:
      "You fine-tune on 10,000 customer support conversations from 2022. It\'s now 2024, and production performance has degraded from 91% to 78%. What is the problem and solution?",
    options: [
      'Model has forgotten its training—retrain with same 2022 data',
      'Distribution drift—collect 2024 data and fine-tune on recent examples',
      'Model needs more training epochs on original data',
      'Fine-tuning was bad—switch to prompt engineering',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (distribution drift + recent data) is correct. 2022→2024 changes: customer language evolves, new products/services, new types of questions, different topics. Model trained on 2022 doesn't understand 2024 patterns. Solution: Collect 2-5K recent conversations from 2024, fine-tune on mix of 2022 and 2024 data (or just 2024 if very different). Expected recovery: 78% → 88-91%. Option A (retrain on same data) won't help—same training data = same capabilities. Option C (more epochs) risks overfitting and still doesn't address 2024 distribution. Option D throws away working solution. Key lesson: Fine-tuned models need periodic updates as distribution drifts. Set up quarterly retraining with recent data.",
  },
  {
    id: 'fine-tune-fundamentals-mc-4',
    question:
      'You have 100 labeled examples. Which fine-tuning approach will likely work BEST?',
    options: [
      'Full fine-tuning with aggressive regularization',
      'LoRA with small rank (r=4) and high dropout',
      'Few-shot prompting instead of fine-tuning',
      'Train from scratch (no pre-training)',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (few-shot prompting) is best for only 100 examples. With 100 examples, fine-tuning risks severe overfitting—model memorizes training set, poor generalization. Few-shot prompting: Include 3-5 examples in prompt, let pre-trained model generalize, no overfitting risk, zero training cost, instant iteration. Expected performance: 70-80% accuracy. Fine-tuning with 100 examples: Even with LoRA/regularization (Options A, B), expect high variance and overfitting, 60-75% accuracy, not worth training cost. Option D (train from scratch) is completely infeasible—needs millions of examples. Rule of thumb: <500 examples → few-shot prompting, 500-5,000 → LoRA, >5,000 → consider full fine-tuning. At 100 examples, focus on data collection, not fine-tuning.',
  },
  {
    id: 'fine-tune-fundamentals-mc-5',
    question:
      'Your fine-tuned model shows: Training accuracy = 98%, Validation accuracy = 72%. This indicates:',
    options: [
      'Excellent performance, deploy immediately',
      'Underfitting, increase model capacity',
      'Overfitting, add regularization or reduce training',
      'Data quality issue, clean dataset',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (overfitting) is textbook clear. Large gap (98% vs 72%) = model memorized training data, failed to generalize. Causes: Too many epochs, too high learning rate, too large model for data size, no regularization. Solutions: Reduce epochs (5 → 2-3), Add dropout (0.1-0.2), Use weight decay (0.01), Reduce learning rate (1e-4 → 1e-5), Early stopping (stop when validation stops improving), Data augmentation (if applicable), Get more training data. Option A is dangerously wrong—deploying this will fail in production. Option B is opposite—model already has too much capacity for the data. Option D might be partial cause but overfitting is primary issue. Target: Training and validation within 5-10% (e.g., 85% train, 82% val).',
  },
];
