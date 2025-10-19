import { MultipleChoiceQuestion } from '../../../types';

export const activationFunctionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'activation-functions-mc-1',
    question:
      'What is the maximum value of the gradient for the sigmoid activation function σ(z) = 1/(1 + e^(-z))?',
    options: ['1.0', '0.5', '0.25', '0.1'],
    correctAnswer: 2,
    explanation:
      "The derivative of sigmoid is σ'(z) = σ(z)(1 - σ(z)). This is maximized when σ(z) = 0.5 (at z=0), giving 0.5 * 0.5 = 0.25. This small maximum gradient is why sigmoid causes vanishing gradients in deep networks—through multiple layers, 0.25^n approaches zero quickly.",
  },
  {
    id: 'activation-functions-mc-2',
    question:
      'Which activation function is most appropriate for the output layer of a binary classification problem?',
    options: ['ReLU', 'Sigmoid', 'Softmax', 'tanh'],
    correctAnswer: 1,
    explanation:
      'Sigmoid is the correct choice for binary classification because it outputs values in (0, 1) which can be interpreted as probabilities. ReLU is unbounded, softmax is for multi-class classification, and tanh outputs to (-1, 1). Sigmoid output represents P(class=1).',
  },
  {
    id: 'activation-functions-mc-3',
    question: 'The "dying ReLU" problem occurs when:',
    options: [
      'The learning rate is too small',
      'Neurons permanently output zero for all inputs due to negative weights',
      'The network has too many layers',
      'The activation function is not differentiable',
    ],
    correctAnswer: 1,
    explanation:
      "Dying ReLU occurs when a neuron's weights become so negative that for all training examples, the pre-activation z = w^Tx + b < 0. Since ReLU(z) = 0 and ReLU'(z) = 0 for z < 0, the neuron outputs zero and receives zero gradients, making it impossible to recover through backpropagation. This is prevented by using Leaky ReLU or proper weight initialization.",
  },
  {
    id: 'activation-functions-mc-4',
    question:
      'For a 3-class classification problem, which activation function should be used in the output layer?',
    options: [
      'Three sigmoid activations',
      'Softmax',
      'ReLU',
      'Three tanh activations',
    ],
    correctAnswer: 1,
    explanation:
      'Softmax is correct for multi-class classification because it ensures: (1) all outputs are non-negative, and (2) all outputs sum to 1, forming a valid probability distribution over classes. Multiple sigmoids would not guarantee outputs sum to 1, and ReLU/tanh are inappropriate for probability outputs.',
  },
  {
    id: 'activation-functions-mc-5',
    question:
      'Why is ReLU preferred over sigmoid/tanh for hidden layers in deep neural networks?',
    options: [
      'ReLU is smoother and more differentiable',
      'ReLU outputs are bounded between 0 and 1',
      'ReLU does not suffer from vanishing gradients for positive inputs',
      'ReLU is zero-centered which helps optimization',
    ],
    correctAnswer: 2,
    explanation:
      "ReLU's gradient is 1 for all positive inputs (no saturation), while sigmoid/tanh saturate for large |z| with gradients approaching 0. In deep networks, gradients must propagate through many layers—ReLU maintains gradient magnitude while sigmoid/tanh cause exponential decay (vanishing gradients). This is why ReLU enables training of deep networks. Note: ReLU is NOT smoother (has a kink at 0), NOT bounded above, and NOT zero-centered.",
  },
];
