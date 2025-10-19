import { MultipleChoiceQuestion } from '../../../types';

export const modelDebuggingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'model-debugging-mc-1',
    question:
      'Your model has 98% training accuracy and 72% test accuracy. What is the primary issue?',
    options: [
      'High bias (underfitting)',
      'High variance (overfitting)',
      'Perfect fit',
      'Data leakage',
    ],
    correctAnswer: 1,
    explanation:
      'The large gap (26%) between training (98%) and test (72%) accuracy indicates high variance (overfitting). The model has memorized the training data rather than learning generalizable patterns. Solutions: add regularization, get more data, or reduce model complexity.',
  },
  {
    id: 'model-debugging-mc-2',
    question:
      'What should you do FIRST when debugging a model that performs worse than a baseline that always predicts the majority class?',
    options: [
      'Increase model complexity',
      'Get more training data',
      'Check the class distribution and use appropriate metrics',
      'Add more features',
    ],
    correctAnswer: 2,
    explanation:
      "When a model performs worse than majority-class baseline, first check if class imbalance makes accuracy misleading. With 90% majority class, always predicting it gives 90% accuracy. Your model might actually be learning patterns but accuracy doesn't show it. Use precision, recall, F1, and AUC instead.",
  },
  {
    id: 'model-debugging-mc-3',
    question:
      'Your model has AUC=0.90 offline but AUC=0.70 in production. What is the most likely cause?',
    options: [
      'The model is perfect',
      'Data distribution shift or data leakage in training',
      'The model is underfitting',
      'You need more training data',
    ],
    correctAnswer: 1,
    explanation:
      'A large gap between offline (0.90) and online (0.70) performance indicates either: (1) data leakage in training (used future information), or (2) distribution shift (production data differs from training data). Use temporal validation and check for leakage.',
  },
  {
    id: 'model-debugging-mc-4',
    question:
      'Learning curves show both training and validation errors plateau at high values with a small gap. What does this indicate?',
    options: [
      'High variance - model is overfitting',
      'High bias - model is underfitting',
      'Perfect fit - model is optimal',
      'Need different evaluation metric',
    ],
    correctAnswer: 1,
    explanation:
      "When both curves plateau at high error with a small gap, the model has high bias (underfitting). The model has reached its representational limit—adding more data won't help. You need to increase model complexity or add better features.",
  },
  {
    id: 'model-debugging-mc-5',
    question:
      'Which visualization is MOST useful for diagnosing bias vs variance?',
    options: [
      'Confusion matrix',
      'ROC curve',
      'Learning curves (training and validation scores vs dataset size)',
      'Feature importance plot',
    ],
    correctAnswer: 2,
    explanation:
      "Learning curves are the best tool for diagnosing bias vs variance. They show: (1) if both curves plateau high = high bias, (2) if large gap = high variance, (3) if validation curve still rising = more data would help. Other visualizations are useful but don't specifically diagnose bias/variance.",
  },
];
