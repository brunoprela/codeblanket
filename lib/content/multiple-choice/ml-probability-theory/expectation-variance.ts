/**
 * Multiple choice questions for Expectation & Variance section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const expectationvarianceMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'If E[X] = 5 and E[Y] = 3, what is E[2X + 3Y]?',
    options: ['11', '16', '19', '34'],
    correctAnswer: 2,
    explanation:
      'By linearity of expectation: E[2X + 3Y] = 2E[X] + 3E[Y] = 2(5) + 3(3) = 10 + 9 = 19. This holds even if X and Y are dependent!',
  },
  {
    id: 'mc2',
    question: 'If Var(X) = 9, what is Var(2X + 5)?',
    options: ['9', '18', '23', '36'],
    correctAnswer: 3,
    explanation:
      "Var (aX + b) = a²Var(X). With a=2, b=5: Var(2X + 5) = 2²(9) = 4(9) = 36. The constant 5 doesn't affect variance - it just shifts the distribution.",
  },
  {
    id: 'mc3',
    question:
      'For independent X and Y with Var(X)=4 and Var(Y)=9, what is Var(X+Y)?',
    options: ['5', '13', '16', '36'],
    correctAnswer: 1,
    explanation:
      "For independent variables, Var(X+Y) = Var(X) + Var(Y) = 4 + 9 = 13. Note: standard deviations don't add (σ_X + σ_Y = 2 + 3 = 5 ≠ √13 ≈ 3.6).",
  },
  {
    id: 'mc4',
    question: 'Which is always true, even for dependent variables?',
    options: [
      'Var(X+Y) = Var(X) + Var(Y)',
      'E[XY] = E[X]E[Y]',
      'E[X+Y] = E[X] + E[Y]',
      'Cov(X,Y) = 0',
    ],
    correctAnswer: 2,
    explanation:
      'Linearity of expectation E[X+Y] = E[X] + E[Y] always holds, even for dependent variables. The other properties require independence.',
  },
  {
    id: 'mc5',
    question:
      'In machine learning, what does Empirical Risk Minimization mean?',
    options: [
      'Minimizing risk on future unseen data',
      'Minimizing average loss on training data',
      'Minimizing maximum loss on any sample',
      'Minimizing computational risk of overfitting',
    ],
    correctAnswer: 1,
    explanation:
      'Empirical Risk Minimization (ERM) means minimizing the average loss on the training data: (1/n)Σ L(yᵢ,ŷᵢ). This is an empirical estimate of the true expected loss over the data distribution.',
  },
];
