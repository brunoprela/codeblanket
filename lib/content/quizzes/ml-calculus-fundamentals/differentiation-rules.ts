/**
 * Quiz questions for Differentiation Rules section
 */

export const differentiationrulesQuiz = [
  {
    id: 'diff-rules-disc-1',
    question: 'Explain how the chain rule enables backpropagation.',
    hint: 'Consider function composition through layers.',
    sampleAnswer: `Backpropagation is the chain rule applied repeatedly. In a neural network, output = f_n(f_{n-1}(...f_1(x))). To find gradients, we apply the chain rule layer by layer going backward, reusing intermediate results. This makes gradient computation O(n) instead of O(n²).`,
    keyPoints: [
      'Neural networks are function compositions',
      'Chain rule computes gradients through compositions',
      'Backprop applies chain rule backward through layers',
      'Intermediate gradient reuse provides efficiency',
    ],
  },
  {
    id: 'diff-rules-disc-2',
    question: 'When and why would you use logarithmic differentiation?',
    hint: 'Consider products, quotients, and complex exponents.',
    sampleAnswer: `Logarithmic differentiation is useful for: (1) Variable exponents like x^x, (2) Products of many functions (converts to sums), (3) Complicated quotients (converts to differences), (4) Numerical stability in log-likelihood computations. It simplifies algebra and prevents numerical underflow in probabilistic models.`,
    keyPoints: [
      'Converts products to sums, quotients to differences',
      'Essential for variable exponents',
      'Used in log-likelihood optimization',
      'Provides numerical stability',
    ],
  },
  {
    id: 'diff-rules-disc-3',
    question: 'Explain implicit differentiation and its ML applications.',
    hint: 'Think about constrained optimization.',
    sampleAnswer: `Implicit differentiation finds dy/dx when y is defined implicitly by F(x,y)=0. In ML, it's used for: (1) Constrained optimization (Lagrange multipliers), (2) Implicit deep learning models (DEQ), (3) Manifold optimization, (4) Neural ODEs. It allows gradient computation without explicit solutions, essential for modern architectures with constraints.`,
    keyPoints: [
      'Works with implicit relationships',
      'Used in constrained optimization',
      'Essential for implicit deep learning',
      'Enables manifold and bilevel optimization',
    ],
  },
];
