import { MultipleChoiceQuestion } from '../../../types';

export const backpropagationAlgorithmMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'backpropagation-algorithm-mc-1',
      question:
        'In backpropagation, what is the correct formula for computing the weight gradient ∂L/∂W for a layer where z = Wx + b?',
      options: [
        '∂L/∂W = x @ (∂L/∂z)',
        '∂L/∂W = (∂L/∂z) @ x^T',
        '∂L/∂W = x^T @ (∂L/∂z)',
        '∂L/∂W = (∂L/∂z)^T @ x',
      ],
      correctAnswer: 2,
      explanation:
        'The correct formula is ∂L/∂W = x^T @ (∂L/∂z), where x is the input and ∂L/∂z is the gradient flowing from the next layer. This comes from the chain rule and ensures the gradient shape matches the weight shape. For batch processing, this should be averaged: (x^T @ ∂L/∂z) / batch_size.',
    },
    {
      id: 'backpropagation-algorithm-mc-2',
      question:
        'Why must forward pass values (like activations) be cached during backpropagation?',
      options: [
        'To save computation time in future forward passes',
        'To compute activation derivatives which require forward pass outputs',
        'To reduce memory usage during training',
        'To prevent overfitting',
      ],
      correctAnswer: 1,
      explanation:
        "Caching is essential because activation derivatives need forward pass values. For example, sigmoid derivative σ'(z) = σ(z)(1 - σ(z)) requires σ(z) from the forward pass. Without caching, we would need to recompute the forward pass, making training much slower. This is a memory-speed tradeoff.",
    },
    {
      id: 'backpropagation-algorithm-mc-3',
      question:
        'How many forward passes are required to compute gradients for all parameters using backpropagation versus numerical gradients for a network with 100,000 parameters?',
      options: [
        'Backprop: 1, Numerical: 100,000',
        'Backprop: 100, Numerical: 100,000',
        'Backprop: 3 (equivalent), Numerical: 200,000',
        'Both require 100,000 forward passes',
      ],
      correctAnswer: 2,
      explanation:
        'Backpropagation requires 1 forward + 1 backward pass (≈3 forward equivalents total) to compute ALL gradients simultaneously. Numerical gradients require 2 forward passes per parameter (θ+ε and θ-ε), so 200,000 forward passes for 100,000 parameters. This makes backprop ~66,666x faster, which is why deep learning is practical.',
    },
    {
      id: 'backpropagation-algorithm-mc-4',
      question:
        'What is the purpose of gradient checking in neural network training?',
      options: [
        'To speed up training by avoiding unnecessary gradient computations',
        'To verify backpropagation implementation correctness by comparing with numerical gradients',
        'To reduce memory usage during backpropagation',
        'To automatically fix bugs in the gradient computation',
      ],
      correctAnswer: 1,
      explanation:
        'Gradient checking verifies backpropagation implementation by comparing analytical gradients (from backprop) with numerical gradients (finite differences). Relative error < 10^-7 indicates correct implementation. It is only used for debugging/verification, not regular training, because numerical gradients are O(n) forward passes and very slow.',
    },
    {
      id: 'backpropagation-algorithm-mc-5',
      question:
        'In the backward pass, how is the gradient propagated from layer l to layer l-1?',
      options: [
        '∂L/∂a_{l-1} = W_l @ (∂L/∂z_l)',
        '∂L/∂a_{l-1} = (∂L/∂z_l) @ W_l',
        '∂L/∂a_{l-1} = (∂L/∂z_l) @ W_l^T',
        '∂L/∂a_{l-1} = W_l^T @ (∂L/∂z_l)',
      ],
      correctAnswer: 2,
      explanation:
        'The gradient flows backward as ∂L/∂a_{l-1} = (∂L/∂z_l) @ W_l^T. This comes from the chain rule: since z_l = W_l a_{l-1} + b_l, we have ∂z_l/∂a_{l-1} = W_l, so ∂L/∂a_{l-1} = (∂L/∂z_l) @ W_l^T. The transpose ensures correct shapes for matrix multiplication.',
    },
  ];
