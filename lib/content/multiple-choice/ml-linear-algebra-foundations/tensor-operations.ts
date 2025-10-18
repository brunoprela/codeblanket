/**
 * Multiple choice questions for Tensor Operations in Deep Learning section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const tensoroperationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tensor-q1',
    question:
      'What is the shape of a batch of 16 RGB images, each 64×64 pixels, in the standard (batch, channels, height, width) format?',
    options: [
      '(16, 64, 64, 3)',
      '(16, 3, 64, 64)',
      '(64, 64, 3, 16)',
      '(3, 16, 64, 64)',
    ],
    correctAnswer: 1,
    explanation:
      'In PyTorch/TensorFlow, images are typically stored as (batch, channels, height, width). For 16 RGB images (3 channels) of size 64×64: (16, 3, 64, 64). Some frameworks like TensorFlow can use (batch, height, width, channels).',
  },
  {
    id: 'tensor-q2',
    question:
      'Broadcasting allows operations on tensors of different shapes. What is the result shape of A (3, 1) + B (1, 4)?',
    options: ['(3, 4)', '(3, 1)', '(1, 4)', 'Error: incompatible shapes'],
    correctAnswer: 0,
    explanation:
      'Broadcasting expands dimensions of size 1. A (3, 1) broadcasts column-wise to (3, 4), B (1, 4) broadcasts row-wise to (3, 4). Result shape: (3, 4).',
  },
  {
    id: 'tensor-q3',
    question:
      'In a dense neural network layer with input shape (batch_size, input_dim) and weight matrix (input_dim, output_dim), what is the output shape?',
    options: [
      '(input_dim, output_dim)',
      '(batch_size, input_dim)',
      '(batch_size, output_dim)',
      '(output_dim, batch_size)',
    ],
    correctAnswer: 2,
    explanation:
      'Matrix multiplication (batch_size, input_dim) @ (input_dim, output_dim) = (batch_size, output_dim). Each sample (row) is transformed from input_dim to output_dim features.',
  },
  {
    id: 'tensor-q4',
    question: 'What does np.einsum("ij,jk->ik", A, B) compute?',
    options: [
      'Element-wise multiplication',
      'Matrix multiplication A @ B',
      'Outer product',
      'Transpose',
    ],
    correctAnswer: 1,
    explanation:
      'The einsum notation "ij,jk->ik" means: sum over repeated index j, resulting in i×k matrix. This is exactly matrix multiplication: C[i,k] = Σⱼ A[i,j] * B[j,k].',
  },
  {
    id: 'tensor-q5',
    question: 'Why is batching important in deep learning?',
    options: [
      'It makes code simpler',
      'It enables parallel processing on GPUs and reduces per-sample overhead',
      'It always improves model accuracy',
      'It reduces memory usage',
    ],
    correctAnswer: 1,
    explanation:
      "Batching processes multiple samples simultaneously, enabling GPU parallelism (thousands of cores) and amortizing overhead across samples. However, it increases memory usage (trade-off) and doesn't directly affect accuracy (training dynamics may change).",
  },
];
