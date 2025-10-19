import { MultipleChoiceQuestion } from '../../../types';

export const forwardPropagationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'forward-propagation-mc-1',
    question:
      'In a neural network layer with 100 input features and 50 neurons, processing a batch of 32 samples, what is the shape of the output?',
    options: ['(100, 50)', '(32, 100)', '(32, 50)', '(50, 32)'],
    correctAnswer: 2,
    explanation:
      'The output shape is (batch_size, num_neurons) = (32, 50). The matrix multiplication X @ W gives (32, 100) @ (100, 50) = (32, 50), where 32 samples are processed simultaneously, each producing 50 outputs.',
  },
  {
    id: 'forward-propagation-mc-2',
    question:
      'Why is vectorization (using matrix operations) significantly faster than using loops in Python for neural network forward propagation?',
    options: [
      'Vectorization uses less memory than loops',
      'Vectorization leverages optimized C/Fortran code, SIMD instructions, and better cache utilization',
      'Vectorization produces more accurate results',
      'Vectorization automatically uses GPU acceleration',
    ],
    correctAnswer: 1,
    explanation:
      'Vectorization achieves 100-1000x speedups through: (1) compiled C/Fortran vs interpreted Python, (2) SIMD instructions that process multiple elements simultaneously, (3) optimized BLAS/LAPACK libraries, (4) better CPU cache utilization, and (5) automatic multi-threading. Memory usage is actually the same, and accuracy/GPU usage depend on other factors.',
  },
  {
    id: 'forward-propagation-mc-3',
    question:
      'During forward propagation, why is it important to cache intermediate values (pre-activations and post-activations)?',
    options: [
      'To save memory during training',
      'To speed up the forward pass',
      'To compute gradients during backpropagation',
      'To prevent numerical overflow',
    ],
    correctAnswer: 2,
    explanation:
      "Cached values are essential for backpropagation because many gradient computations require the forward pass outputs. For example, sigmoid derivative σ'(z) = σ(z)(1 - σ(z)) needs the forward activation σ(z). Without caching, these would need to be recomputed, significantly slowing training. This actually uses MORE memory, not less.",
  },
  {
    id: 'forward-propagation-mc-4',
    question:
      'For a network with layers [100, 50, 25, 10], how many matrix multiplications occur during one forward pass?',
    options: ['2', '3', '4', '10'],
    correctAnswer: 1,
    explanation:
      'There are 3 matrix multiplications, one for each layer transition: (1) 100→50, (2) 50→25, (3) 25→10. The number of multiplications equals the number of weight matrices, which is len(layer_sizes) - 1 = 4 - 1 = 3.',
  },
  {
    id: 'forward-propagation-mc-5',
    question:
      'How does memory usage for activations scale with batch size in forward propagation?',
    options: [
      'Memory is independent of batch size',
      'Memory scales logarithmically with batch size',
      'Memory scales linearly with batch size',
      'Memory scales quadratically with batch size',
    ],
    correctAnswer: 2,
    explanation:
      'Activation memory scales linearly with batch size: Memory = O(batch_size × Σ layer_sizes). If you double the batch size, you double the memory needed to store activations. Parameter memory (weights and biases) remains constant regardless of batch size.',
  },
];
