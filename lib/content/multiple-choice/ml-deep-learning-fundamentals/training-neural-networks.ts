import { MultipleChoiceQuestion } from '../../../types';

export const trainingNeuralNetworksMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'training-nn-mc1',
    question:
      'When training with mini-batch gradient descent, what is a typical recommended starting batch size for modern GPU hardware?',
    options: [
      'Batch size of 1 (pure stochastic gradient descent)',
      'Batch size between 32-64',
      'Batch size between 1024-2048',
      'Batch size equal to the full dataset size',
    ],
    correctAnswer: 1,
    explanation:
      'Starting with batch size 32-64 is a good default that balances gradient noise (helps generalization) with computational efficiency. You can then increase batch size until GPU memory is full or convergence slows. Very small batches (1) are inefficient, while very large batches (>1024) may require careful tuning and hurt generalization.',
  },
  {
    id: 'training-nn-mc2',
    question:
      'What is the purpose of gradient clipping during neural network training?',
    options: [
      'To speed up training by skipping small gradients',
      'To prevent exploding gradients by limiting the maximum gradient norm',
      'To improve generalization by adding noise to gradients',
      'To reduce memory usage during backpropagation',
    ],
    correctAnswer: 1,
    explanation:
      "Gradient clipping prevents exploding gradients by limiting the maximum norm of gradients. If the global gradient norm exceeds a threshold (typically 5.0), all gradients are scaled down proportionally. This is especially important for RNNs and very deep networks. It doesn't add noise or reduce memory, but ensures training stability.",
  },
  {
    id: 'training-nn-mc3',
    question:
      'You observe that training loss is decreasing steadily but validation loss started increasing after epoch 30. What is happening and what should you do?',
    options: [
      'The model is underfitting; increase model capacity',
      'The model is overfitting; add regularization or use early stopping',
      'The learning rate is too high; reduce it immediately',
      'The model has converged; training is complete',
    ],
    correctAnswer: 1,
    explanation:
      'Train loss decreasing while validation loss increases is the classic sign of overfitting - the model is memorizing training data rather than learning generalizable patterns. Solutions include: add regularization (dropout, L2), use early stopping to restore weights from epoch 30, get more training data, or reduce model capacity.',
  },
  {
    id: 'training-nn-mc4',
    question:
      'What is the purpose of learning rate warmup, especially when training with large batch sizes?',
    options: [
      'To heat up the GPU for optimal performance',
      'To gradually increase learning rate at the start to prevent training instability',
      'To find the optimal learning rate through experimentation',
      'To speed up the initial epochs of training',
    ],
    correctAnswer: 1,
    explanation:
      'Learning rate warmup gradually increases the learning rate from a small value to the target value over the first few epochs. This prevents instability that can occur when starting training with a high learning rate and large batch sizes, as initial gradients can be large and poorly estimated. Warmup is especially important for Transformers and large-batch training.',
  },
  {
    id: 'training-nn-mc5',
    question:
      'In the "Reduce Learning Rate on Plateau" scheduling strategy, what does the "patience" parameter control?',
    options: [
      'The factor by which to reduce the learning rate',
      'The minimum learning rate allowed',
      'The number of epochs without improvement before reducing the learning rate',
      'The number of times the learning rate can be reduced',
    ],
    correctAnswer: 2,
    explanation:
      'The patience parameter specifies how many consecutive epochs without improvement in validation loss to wait before reducing the learning rate. For example, patience=10 means "if validation loss doesn\'t improve for 10 epochs, reduce the learning rate." This prevents premature reduction due to temporary fluctuations.',
  },
];
