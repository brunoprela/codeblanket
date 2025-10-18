/**
 * Multiple choice questions for Algebraic Expressions & Equations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const algebraicexpressionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-algebra-terms',
    question: 'In the expression 5x² - 3x + 7, which statement is correct?',
    options: [
      '5 is a variable and 7 is a coefficient',
      'x² and x are like terms',
      '5 is a coefficient of x² and 7 is a constant',
      'The degree of the expression is 3',
    ],
    correctAnswer: 2,
    explanation:
      "5 is the coefficient of x² (it multiplies x²), and 7 is a constant (it doesn't multiply any variable). x² and x are NOT like terms because they have different powers. The degree is 2 (highest power of x).",
  },
  {
    id: 'mc2-quadratic-discriminant',
    question:
      'For the quadratic equation 2x² + 3x + 5 = 0, what does the discriminant tell us about the solutions?',
    options: [
      'Two distinct real solutions because Δ > 0',
      'One repeated real solution because Δ = 0',
      'Two complex solutions because Δ < 0',
      'No solutions exist',
    ],
    correctAnswer: 2,
    explanation:
      'The discriminant Δ = b² - 4ac = 3² - 4(2)(5) = 9 - 40 = -31. Since Δ < 0, the equation has two complex conjugate solutions, not real solutions. This is common in certain ML optimization scenarios where complex eigenvalues appear.',
  },
  {
    id: 'mc3-system-solutions',
    question:
      'When solving a system of two linear equations with two unknowns, which outcome is NOT possible?',
    options: [
      'Exactly one solution (lines intersect at a point)',
      'No solutions (parallel lines)',
      'Infinitely many solutions (same line)',
      'Exactly three solutions',
    ],
    correctAnswer: 3,
    explanation:
      'A system of two linear equations in two variables can have: (1) exactly one solution (lines intersect), (2) no solutions (parallel lines), or (3) infinitely many solutions (same line). It CANNOT have exactly two, three, or any finite number other than one.',
  },
  {
    id: 'mc4-factoring',
    question: 'Which factoring pattern does x² - 16 follow?',
    options: [
      'Perfect square trinomial',
      'Difference of squares',
      'Sum of squares',
      'Common factor',
    ],
    correctAnswer: 1,
    explanation:
      "x² - 16 = x² - 4² is a difference of squares pattern: a² - b² = (a + b)(a - b). So x² - 16 = (x + 4)(x - 4). Note that sum of squares (a² + b²) doesn't factor over real numbers.",
  },
  {
    id: 'mc5-ml-equation',
    question:
      'In the gradient descent update rule θ_new = θ_old - α∇L(θ), what does solving for α when ∇L(θ) = 10 and we want θ_new = θ_old - 5 give us?',
    options: ['α = 0.5', 'α = 2', 'α = 5', 'α = 50'],
    correctAnswer: 0,
    explanation:
      'From θ_new = θ_old - α∇L(θ), we have: θ_old - 5 = θ_old - α(10). This gives us: -5 = -10α, so α = 5/10 = 0.5. This represents the learning rate needed to achieve the desired parameter update.',
  },
];
