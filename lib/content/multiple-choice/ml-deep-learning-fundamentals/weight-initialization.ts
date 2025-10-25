import { MultipleChoiceQuestion } from '../../../types';

export const weightInitializationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'weight-init-mc1',
    question:
      'Why is initializing all weights to zero problematic in neural networks?',
    options: [
      'It causes the model to overfit to the training data',
      'All neurons compute identical outputs and receive identical gradients, preventing them from learning different features',
      'It makes the network train too slowly due to small weight values',
      'It violates the assumption that weights should be normalized',
    ],
    correctAnswer: 1,
    explanation:
      'Zero initialization causes all neurons in a layer to be identical - they compute the same outputs and receive the same gradients during backpropagation. This means all weights update identically and neurons cannot specialize. This "symmetry problem" effectively reduces each layer to a single neuron regardless of width.',
  },
  {
    id: 'weight-init-mc2',
    question:
      'Which initialization method is most appropriate for a deep network using ReLU activations?',
    options: [
      'Xavier (Glorot) initialization: W ~ N(0, √(2/(n_in + n_out)))',
      'He initialization: W ~ N(0, √(2/n_in))',
      'Uniform initialization: W ~ U(-0.5, 0.5)',
      'Small random initialization: W ~ N(0, 0.01)',
    ],
    correctAnswer: 1,
    explanation:
      'He initialization (√(2/n_in)) is designed for ReLU activations. ReLU zeros out negative values, effectively halving the variance of activations. He initialization compensates by using a larger scale factor. Xavier initialization is better for sigmoid/tanh, while uniform and small random initialization can cause vanishing/exploding gradients.',
  },
  {
    id: 'weight-init-mc3',
    question:
      'What is the primary goal of proper weight initialization methods like Xavier and He?',
    options: [
      'To minimize the initial loss function value',
      'To ensure weights are as small as possible to prevent overfitting',
      'To maintain similar variance of activations and gradients across layers',
      'To maximize the learning rate that can be used during training',
    ],
    correctAnswer: 2,
    explanation:
      'The primary goal is to maintain similar variance of activations and gradients across all layers. If variance grows (exploding) or shrinks (vanishing) through the network, training becomes difficult or impossible. Proper initialization keeps signal strength consistent, enabling effective gradient flow and faster convergence.',
  },
  {
    id: 'weight-init-mc4',
    question:
      'Consider a network with ReLU activations initialized using Xavier initialization instead of He initialization. What is most likely to happen?',
    options: [
      'The network will train normally with no issues',
      'Gradients will explode due to too large initialization',
      'Activations will gradually diminish through deeper layers (vanishing activations)',
      'The network will overfit more easily',
    ],
    correctAnswer: 2,
    explanation:
      "Xavier initialization uses √(2/(n_in + n_out)), which is smaller than He\'s √(2/n_in). Combined with ReLU zeroing negative values, this causes variance to shrink by ~50% at each layer. In deep networks, activations gradually vanish, leading to very small gradients and slow/stalled training. This is why He initialization is crucial for ReLU networks.",
  },
  {
    id: 'weight-init-mc5',
    question:
      'In modern deep learning frameworks (PyTorch, TensorFlow), what is the default initialization for linear/dense layers?',
    options: [
      'All weights initialized to zero',
      'Xavier (Glorot) uniform initialization',
      'He normal initialization',
      'Random uniform initialization between -1 and 1',
    ],
    correctAnswer: 1,
    explanation:
      "Most modern frameworks default to Xavier (Glorot) uniform initialization for compatibility with various activation functions. However, when using ReLU activations (the most common case), it's often beneficial to explicitly use He initialization for better performance. PyTorch provides init.xavier_uniform_() and init.kaiming_normal_() (He) for explicit control.",
  },
];
