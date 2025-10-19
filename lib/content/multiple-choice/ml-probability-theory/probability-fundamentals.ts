/**
 * Multiple choice questions for Probability Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const probabilityfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which of the following is NOT one of the Kolmogorov axioms of probability?',
    options: [
      'P(E) ≥ 0 for any event E (non-negativity)',
      'P(Ω) = 1 (normalization)',
      'P(A ∪ B) = P(A) + P(B) for disjoint events (additivity)',
      'P(A ∩ B) = P(A) × P(B) for all events (multiplication)',
    ],
    correctAnswer: 3,
    explanation:
      'The multiplication rule P(A ∩ B) = P(A) × P(B) only holds for independent events, not all events. The three Kolmogorov axioms are non-negativity, normalization, and additivity for disjoint events.',
  },
  {
    id: 'mc2',
    question:
      'If P(A) = 0.7, what is P(not A), the probability of the complement of A?',
    options: ['0.3', '0.7', '1.4', 'Cannot be determined'],
    correctAnswer: 0,
    explanation:
      'By the complement rule, P(not A) = 1 - P(A) = 1 - 0.7 = 0.3. The probabilities of an event and its complement must sum to 1.',
  },
  {
    id: 'mc3',
    question:
      'In a machine learning classification model, what does the output probability P(class=1|features) represent?',
    options: [
      'The theoretical probability from the data generation process',
      'The conditional probability of class 1 given the observed features',
      'The probability that the features are correct',
      'The accuracy of the model',
    ],
    correctAnswer: 1,
    explanation:
      'ML classifiers output conditional probabilities: P(class|features). This is the probability that the true class is 1, given the observed feature values. This is an estimate, not the true theoretical probability.',
  },
  {
    id: 'mc4',
    question:
      'Two events A and B are mutually exclusive (disjoint). If P(A) = 0.4 and P(B) = 0.3, what is P(A ∪ B)?',
    options: ['0.12', '0.58', '0.70', '1.00'],
    correctAnswer: 2,
    explanation:
      'For mutually exclusive events, P(A ∪ B) = P(A) + P(B) = 0.4 + 0.3 = 0.7. Since they cannot occur simultaneously, P(A ∩ B) = 0, so we simply add the probabilities.',
  },
  {
    id: 'mc5',
    question:
      'What happens to empirical probability estimates as we collect more data?',
    options: [
      'They become less accurate due to overfitting',
      'They converge to the theoretical probability (Law of Large Numbers)',
      'They remain constant regardless of sample size',
      'They diverge from the theoretical probability',
    ],
    correctAnswer: 1,
    explanation:
      'The Law of Large Numbers states that as the sample size increases, the empirical probability (observed frequency) converges to the theoretical (true) probability. This is why more data generally leads to better ML models.',
  },
];
