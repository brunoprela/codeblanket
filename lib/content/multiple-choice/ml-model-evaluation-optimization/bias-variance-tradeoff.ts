import { MultipleChoiceQuestion } from '../../../types';

export const biasVarianceTradeoffMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bias-variance-tradeoff-mc-1',
    question:
      'A model has training error of 15% and test error of 16%. What is the most likely problem?',
    options: [
      'High variance (overfitting)',
      'High bias (underfitting)',
      'Good fit (optimal complexity)',
      'Data leakage',
    ],
    correctAnswer: 1,
    explanation:
      'Both training and test errors are high (15-16%) with a small gap (1%). This indicates high bias—the model is too simple to capture the underlying pattern. With high variance, you would see low training error but high test error with a large gap.',
  },
  {
    id: 'bias-variance-tradeoff-mc-2',
    question:
      'Your neural network achieves 2% training error but 25% test error. What should you do FIRST?',
    options: [
      'Increase model complexity (add more layers)',
      'Add dropout or L2 regularization',
      'Remove features to simplify the model',
      'Get more training data if possible',
    ],
    correctAnswer: 1,
    explanation:
      'The large gap (23%) between training (2%) and test (25%) error indicates high variance (overfitting). The quickest fix is adding regularization like dropout or L2 penalty. Getting more data would help but is often not immediately available. Increasing complexity would make overfitting worse.',
  },
  {
    id: 'bias-variance-tradeoff-mc-3',
    question:
      'Learning curves show both training and validation errors plateau at high values with a small gap. What does this indicate?',
    options: [
      'Perfect model - ready for deployment',
      'High bias - model too simple, need to increase complexity',
      'High variance - model too complex, need regularization',
      'Need more training data',
    ],
    correctAnswer: 1,
    explanation:
      "When both curves plateau at high error with a small gap, the model has high bias (underfitting). The plateau indicates that adding more data won't help—the model has reached its representational limit. You need to increase model complexity or add features. High variance would show a large gap between curves.",
  },
  {
    id: 'bias-variance-tradeoff-mc-4',
    question:
      'Which technique would MOST effectively reduce variance in an overfitting model?',
    options: [
      'Add more polynomial features',
      'Increase the depth of decision trees',
      'Apply dropout regularization',
      'Remove regularization constraints',
    ],
    correctAnswer: 2,
    explanation:
      'Dropout regularization directly reduces variance by preventing neurons from co-adapting and forcing the network to learn more robust features. Adding polynomial features, increasing tree depth, or removing regularization would all INCREASE variance and worsen overfitting.',
  },
  {
    id: 'bias-variance-tradeoff-mc-5',
    question:
      'In the bias-variance tradeoff, what happens as model complexity increases?',
    options: [
      'Both bias and variance decrease',
      'Both bias and variance increase',
      'Bias decreases and variance increases',
      'Bias increases and variance decreases',
    ],
    correctAnswer: 2,
    explanation:
      "This is the fundamental tradeoff: as model complexity increases, bias decreases (better fit to training data) but variance increases (more sensitive to training data fluctuations). This is why we can't minimize both simultaneously and must find an optimal balance.",
  },
];
