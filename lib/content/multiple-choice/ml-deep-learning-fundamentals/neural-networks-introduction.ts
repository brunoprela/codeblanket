import { MultipleChoiceQuestion } from '../../../types';

export const neuralNetworksIntroductionMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'neural-networks-introduction-mc-1',
      question:
        'What is the fundamental limitation that prevents a single perceptron from solving the XOR problem?',
      options: [
        'Perceptrons cannot handle binary inputs',
        'Perceptrons can only learn linear decision boundaries',
        'Perceptrons require too much training data',
        'Perceptrons lack a proper activation function',
      ],
      correctAnswer: 1,
      explanation:
        'A perceptron learns a linear decision boundary (a straight line in 2D, a hyperplane in higher dimensions). The XOR problem is not linearly separable—you cannot draw a single straight line to separate the two classes. This fundamental limitation requires multiple layers with non-linear activations to solve.',
    },
    {
      id: 'neural-networks-introduction-mc-2',
      question:
        'In a multi-layer perceptron (MLP), what is the primary role of the activation function?',
      options: [
        'To initialize the weights randomly',
        'To introduce non-linearity enabling the network to learn complex patterns',
        'To prevent overfitting during training',
        'To normalize the input data',
      ],
      correctAnswer: 1,
      explanation:
        'Without non-linear activation functions, multiple layers would collapse to a single linear transformation (matrix multiplication). Non-linear activations like sigmoid, tanh, or ReLU enable the network to learn non-linear decision boundaries and complex patterns. This is essential for the power of deep learning.',
    },
    {
      id: 'neural-networks-introduction-mc-3',
      question:
        'According to the Universal Approximation Theorem, what can a feedforward neural network with a single hidden layer approximate?',
      options: [
        'Only linear functions',
        'Any discontinuous function',
        'Any continuous function on compact subsets of ℝⁿ',
        'Only polynomial functions up to degree 3',
      ],
      correctAnswer: 2,
      explanation:
        "The Universal Approximation Theorem (Cybenko, 1989) states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of ℝⁿ, given appropriate activation functions. However, this doesn't mean single-layer networks are practical—deep networks are far more efficient.",
    },
    {
      id: 'neural-networks-introduction-mc-4',
      question:
        'Given the following neural network layer computation: z = W^T x + b, followed by a = σ(z), which statement is correct?',
      options: [
        'z is the final output and a is the loss',
        'z is the pre-activation and a is the post-activation output',
        'a is always between 0 and 1 regardless of activation function',
        'W^T x computes the element-wise product of weights and inputs',
      ],
      correctAnswer: 1,
      explanation:
        'In neural network terminology, z = W^T x + b is the "pre-activation" (weighted sum), and a = σ(z) is the "post-activation" (after applying the activation function σ). This distinction is important for backpropagation. Note that the range of a depends on the activation function used (sigmoid: 0-1, tanh: -1 to 1, ReLU: 0 to ∞).',
    },
    {
      id: 'neural-networks-introduction-mc-5',
      question:
        'Why are deep neural networks (many layers) preferred over shallow wide networks (one large hidden layer) despite the Universal Approximation Theorem?',
      options: [
        'Deep networks train faster in all cases',
        'Deep networks require exponentially fewer parameters for the same representational power',
        'Shallow networks cannot approximate non-linear functions',
        'Deep networks always have higher accuracy',
      ],
      correctAnswer: 1,
      explanation:
        'While the Universal Approximation Theorem guarantees that shallow networks CAN approximate any function, deep networks do it much more efficiently. For many functions, deep networks need O(n²) parameters while shallow networks need O(2^n) parameters—an exponential difference. Deep networks also learn hierarchical features that generalize better, but the parameter efficiency is the primary mathematical reason.',
    },
  ];
