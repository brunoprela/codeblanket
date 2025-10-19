import { MultipleChoiceQuestion } from '../../../types';

export const modelSelectionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'model-selection-mc-1',
    question:
      'You tested 20 models on the same test set and chose the one with highest accuracy. What is the primary risk?',
    options: [
      'The model might be too complex',
      'You overfitted to the test set through multiple comparisons',
      'You need to test more models',
      'The test set is too small',
    ],
    correctAnswer: 1,
    explanation:
      "Testing many models on the same test set leads to overfitting through multiple comparisons. By chance, one model will perform better even if it's not truly better. The test set essentially becomes a validation set. Use Bonferroni correction or reserve test set for final evaluation only.",
  },
  {
    id: 'model-selection-mc-2',
    question:
      'Two models have similar performance (AUC: 0.85 vs 0.84, p=0.15). Which should you choose?',
    options: [
      'Always choose the higher AUC model',
      'Choose the simpler, more interpretable model',
      'Train more models to find a better one',
      'Increase the test set size',
    ],
    correctAnswer: 1,
    explanation:
      'When performance is not statistically significantly different (p=0.15 > 0.05), choose the simpler model. It will be easier to maintain, faster to train/deploy, more interpretable, and less prone to overfitting. The marginal 0.01 AUC improvement is likely not meaningful.',
  },
  {
    id: 'model-selection-mc-3',
    question:
      'For real-time fraud detection requiring <100ms latency, which factor should you prioritize?',
    options: [
      'Maximize AUC above all else',
      'Find a model that meets minimum AUC threshold and latency constraint',
      'Choose the most interpretable model',
      'Use the largest ensemble possible',
    ],
    correctAnswer: 1,
    explanation:
      'In production systems with hard constraints (latency, memory), you must meet those constraints first. Find models that meet minimum performance AND latency requirements, then optimize. A model with 0.95 AUC but 500ms latency is useless if you need <100ms.',
  },
  {
    id: 'model-selection-mc-4',
    question:
      'What is the purpose of nested cross-validation in model selection?',
    options: [
      'To make model training faster',
      'To get an unbiased performance estimate when comparing models',
      'To tune hyperparameters more efficiently',
      'To reduce overfitting on the training set',
    ],
    correctAnswer: 1,
    explanation:
      'Nested CV provides unbiased performance estimates when you evaluate and compare multiple models. The outer loop evaluates performance, the inner loop selects models/tunes hyperparameters. This prevents overfitting to the validation set that occurs with repeated evaluation.',
  },
  {
    id: 'model-selection-mc-5',
    question:
      'You deployed Model A (AUC=0.90 offline) to production but it performs at AUC=0.80. What is the most likely cause?',
    options: [
      'The test set was too small',
      'Data distribution shift between training and production',
      'The model is too simple',
      'You need more training data',
    ],
    correctAnswer: 1,
    explanation:
      'A large gap between offline (0.90) and online (0.80) performance usually indicates data distribution shift. Production data has different characteristics than training data (different time period, user behavior, data collection process). This emphasizes the importance of monitoring and retraining.',
  },
];
