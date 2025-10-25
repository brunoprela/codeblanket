/**
 * Multiple choice questions for Joint & Marginal Distributions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const jointmarginaldistributionsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'How do you obtain the marginal distribution P(X) from the joint distribution P(X,Y)?',
      options: [
        'Multiply P(X,Y) by P(Y)',
        'Divide P(X,Y) by P(Y)',
        'Sum (or integrate) P(X,Y) over all values of Y',
        'Take the derivative of P(X,Y) with respect to Y',
      ],
      correctAnswer: 2,
      explanation:
        'The marginal distribution is obtained by summing (discrete) or integrating (continuous) the joint distribution over all values of the other variable: P(X) = Σ_y P(X,Y) or P(X) = ∫ P(X,Y)dy.',
    },
    {
      id: 'mc2',
      question: 'If X and Y are independent, which is TRUE?',
      options: [
        'Cov(X,Y) = 0 only',
        'P(X,Y) = P(X) + P(Y)',
        'P(X,Y) = P(X) × P(Y)',
        'P(X|Y) = P(Y|X)',
      ],
      correctAnswer: 2,
      explanation:
        "Independence is defined as P(X,Y) = P(X) × P(Y). This implies Cov(X,Y) = 0, but the converse is not always true (zero covariance doesn't guarantee independence).",
    },
    {
      id: 'mc3',
      question: 'What is the range of the correlation coefficient ρ?',
      options: ['0 to 1', '-∞ to +∞', '-1 to 1', '0 to ∞'],
      correctAnswer: 2,
      explanation:
        "The correlation coefficient (Pearson\'s r) is always between -1 and 1: -1 ≤ ρ ≤ 1. ρ = -1 means perfect negative linear relationship, ρ = 0 means no linear relationship, ρ = 1 means perfect positive linear relationship.",
    },
    {
      id: 'mc4',
      question: 'If Cov(X,Y) = 0, what can we conclude?',
      options: [
        'X and Y are independent',
        'X and Y have no linear relationship',
        'X and Y have no relationship at all',
        'P(X,Y) = 0',
      ],
      correctAnswer: 1,
      explanation:
        'Zero covariance means no LINEAR relationship, but X and Y could still be related nonlinearly (e.g., Y = X²). Independence implies zero covariance, but zero covariance does not imply independence.',
    },
    {
      id: 'mc5',
      question:
        'In supervised learning, we typically model which distribution?',
      options: [
        'P(X) - the feature distribution',
        'P(Y) - the label distribution',
        'P(X,Y) - the joint distribution',
        'P(Y|X) - the conditional distribution',
      ],
      correctAnswer: 3,
      explanation:
        'Most supervised learning models P(Y|X) - the probability of the label given features. This is called discriminative modeling. Generative models learn P(X,Y), but discriminative models are more common for pure prediction tasks.',
    },
  ];
