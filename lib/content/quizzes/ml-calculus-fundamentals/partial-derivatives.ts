/**
 * Quiz questions for Partial Derivatives section
 */

export const partialderivativesQuiz = [
  {
    id: 'partial-disc-1',
    question:
      'Explain how partial derivatives enable gradient-based optimization in high-dimensional spaces.',
    hint: 'Consider parameter spaces with millions of dimensions.',
    sampleAnswer: `Partial derivatives decompose high-dimensional optimization into manageable components. For a neural network with millions of parameters θ = [θ₁,...,θₙ], we need ∇L = [∂L/∂θ₁,...,∂L/∂θₙ]. Each partial ∂L/∂θᵢ tells us how to adjust that specific parameter. Gradient descent then updates: θᵢ ← θᵢ - α·∂L/∂θᵢ for each i. This parallelizes naturally, making optimization tractable even for billions of parameters. Without partial derivatives, we couldn't isolate individual parameter effects.`,
    keyPoints: [
      'Partials decompose high-dim gradient into components',
      'Each ∂L/∂θᵢ guides individual parameter update',
      'Enables parallel computation',
      'Makes billion-parameter optimization tractable',
    ],
  },
  {
    id: 'partial-disc-2',
    question:
      'How does the chain rule extend to partial derivatives in backpropagation?',
    hint: 'Consider computing ∂L/∂W₁ through intermediate layers.',
    sampleAnswer: `Backprop applies the multivariable chain rule. For L(f(g(x))), we have: ∂L/∂xᵢ = Σⱼ(∂L/∂fⱼ · ∂fⱼ/∂gₖ · ∂gₖ/∂xᵢ) summing over all paths. In neural networks: ∂L/∂W₁ = ∂L/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁. Each term is a partial derivative or Jacobian matrix. The chain rule says: multiply along paths and sum over branches. This is exactly what backprop does layer-by-layer.`,
    keyPoints: [
      'Multivariable chain rule sums over all paths',
      'Each layer contributes local Jacobian',
      'Matrix multiplication implements chain rule',
      'Backprop efficiently computes through layer composition',
    ],
  },
  {
    id: 'partial-disc-3',
    question: 'Why is the gradient the direction of steepest ascent?',
    hint: 'Consider directional derivatives in all directions.',
    sampleAnswer: `The gradient ∇f points toward maximum rate of increase. Mathematical proof: the directional derivative in direction û is ∇f·û. This is maximized when û is parallel to ∇f (dot product = |∇f|). Any other direction gives slower increase. Geometrically, ∇f is perpendicular to level curves/surfaces. Moving along ∇f crosses the most level curves per unit distance. In optimization, we use -∇f (steepest descent) to minimize functions, which is why gradient descent works.`,
    keyPoints: [
      'Directional derivative is ∇f·û',
      'Maximized when û parallel to ∇f',
      'Gradient perpendicular to level sets',
      'Negative gradient gives steepest descent for minimization',
    ],
  },
];
