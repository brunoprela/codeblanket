import { MultipleChoiceQuestion } from '../../../types';

export const lossFunctionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'loss-functions-mc-1',
    question:
      'Which loss function is most appropriate for a regression task predicting stock prices?',
    options: [
      'Binary Cross-Entropy',
      'Categorical Cross-Entropy',
      'Mean Squared Error (MSE)',
      'Softmax',
    ],
    correctAnswer: 2,
    explanation:
      'MSE is appropriate for regression tasks with continuous targets like stock prices. Cross-entropy losses are for classification (BCE for binary, categorical CE for multi-class). Softmax is an activation function, not a loss function. For regression with outliers, Huber loss might be better than MSE.',
  },
  {
    id: 'loss-functions-mc-2',
    question:
      'Why is Binary Cross-Entropy preferred over MSE for binary classification?',
    options: [
      'BCE is faster to compute than MSE',
      'BCE avoids vanishing gradients when predictions are wrong, unlike MSE with sigmoid',
      'BCE requires less memory than MSE',
      'BCE works with any activation function',
    ],
    correctAnswer: 1,
    explanation:
      "With sigmoid activation, MSE gradient includes σ'(z) which approaches 0 when predictions are very wrong, causing vanishing gradients. BCE gradient (∂L/∂z = ŷ - y) cancels the σ'(z) term, maintaining strong gradients even for wrong predictions. This enables faster, more reliable training.",
  },
  {
    id: 'loss-functions-mc-3',
    question:
      'For a 10-class classification problem, what is the output layer configuration?',
    options: [
      '10 neurons with sigmoid activation and Binary Cross-Entropy loss',
      '10 neurons with softmax activation and Categorical Cross-Entropy loss',
      '1 neuron with softmax activation and Categorical Cross-Entropy loss',
      '10 neurons with ReLU activation and MSE loss',
    ],
    correctAnswer: 1,
    explanation:
      '10 neurons with softmax + Categorical Cross-Entropy is correct for 10-class classification. Softmax ensures outputs sum to 1 (probability distribution), and categorical CE measures distance between predicted and true distributions. Sigmoid is for binary, ReLU/MSE for regression.',
  },
  {
    id: 'loss-functions-mc-4',
    question:
      'What is the gradient of Categorical Cross-Entropy with respect to the logits (∂L/∂z) when using softmax activation?',
    options: [
      '∂L/∂z = y - ŷ',
      '∂L/∂z = ŷ - y',
      '∂L/∂z = ŷ(1 - ŷ)',
      '∂L/∂z = -y/ŷ',
    ],
    correctAnswer: 1,
    explanation:
      'The gradient is ∂L/∂z = ŷ - y (predicted probabilities minus true one-hot labels). This elegant result comes from the mathematical relationship between softmax and cross-entropy, where the softmax derivative terms cancel with the cross-entropy derivative terms, leaving just the prediction error.',
  },
  {
    id: 'loss-functions-mc-5',
    question:
      'In a trading model, why might you use a custom loss function instead of standard MSE?',
    options: [
      'MSE is too computationally expensive for trading',
      'Custom loss can incorporate transaction costs, directional accuracy, and risk metrics',
      'MSE only works for daily predictions, not intraday',
      'Custom loss functions train faster than MSE',
    ],
    correctAnswer: 1,
    explanation:
      'Standard MSE optimizes prediction accuracy but ignores trading-specific concerns like transaction costs (reduce overtrading), directional accuracy (profit depends on direction, not magnitude), risk management (variance, drawdown), and risk-adjusted returns (Sharpe ratio). Custom losses can directly optimize profitability rather than prediction error.',
  },
];
