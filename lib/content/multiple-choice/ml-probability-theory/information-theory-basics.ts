/**
 * Multiple choice questions for Information Theory Basics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const informationtheorybasicsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'A fair coin flip has entropy of approximately:',
    options: ['0 bits', '0.5 bits', '1 bit', '2 bits'],
    correctAnswer: 2,
    explanation:
      'A fair coin has P(H) = P(T) = 0.5. Entropy H = -Σ P log₂ P = -(0.5 log₂ 0.5 + 0.5 log₂ 0.5) = -(-0.5 - 0.5) = 1 bit. This is the maximum entropy for a binary random variable.',
  },
  {
    id: 'mc2',
    question:
      'Which distribution has the highest entropy for a given number of outcomes?',
    options: [
      'Normal distribution',
      'Uniform distribution',
      'Exponential distribution',
      'Binomial distribution',
    ],
    correctAnswer: 1,
    explanation:
      'The uniform distribution (all outcomes equally likely) has maximum entropy among all distributions with the same number of outcomes. It represents maximum uncertainty.',
  },
  {
    id: 'mc3',
    question:
      'In multi-class classification, the cross-entropy loss is minimized when:',
    options: [
      'All class probabilities are equal',
      'The predicted probability for the true class is 1',
      'The entropy of predictions is maximized',
      'The model predicts randomly',
    ],
    correctAnswer: 1,
    explanation:
      'Cross-entropy loss H(P,Q) = -Σ P(x) log Q(x) is minimized when Q matches P perfectly. For classification, this means predicted probability for the true class should be 1 (and 0 for all other classes).',
  },
  {
    id: 'mc4',
    question: 'KL divergence D_KL(P||Q) is:',
    options: [
      'Always symmetric: D_KL(P||Q) = D_KL(Q||P)',
      'Equal to cross-entropy',
      'Equal to H(P,Q) - H(P)',
      'Always negative',
    ],
    correctAnswer: 2,
    explanation:
      'KL divergence D_KL(P||Q) = H(P,Q) - H(P), the difference between cross-entropy and entropy. It is NOT symmetric and is always ≥ 0.',
  },
  {
    id: 'mc5',
    question:
      'If two variables X and Y are independent, their mutual information I(X;Y) is:',
    options: ['0', '1', 'H(X)', 'H(Y)'],
    correctAnswer: 0,
    explanation:
      'If X and Y are independent, knowing Y gives no information about X, so I(X;Y) = 0. Conversely, if I(X;Y) = 0, then X and Y are independent.',
  },
];
