/**
 * Multiple choice questions for Conditional Probability & Independence section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const conditionalprobabilityindependenceMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'If P(A) = 0.6, P(B) = 0.4, and P(A ∩ B) = 0.24, are A and B independent?',
      options: [
        'Yes, because P(A ∩ B) = P(A) × P(B)',
        'No, because P(A) ≠ P(B)',
        'Cannot determine without P(A|B)',
        'Yes, because P(A) + P(B) > 1',
      ],
      correctAnswer: 0,
      explanation:
        'Check if P(A ∩ B) = P(A) × P(B): 0.24 = 0.6 × 0.4 = 0.24 ✓. Yes, they are independent! The test for independence is whether the joint probability equals the product of marginals.',
    },
    {
      id: 'mc2',
      question:
        'In a medical test, P(disease) = 0.01 and P(positive test | disease) = 0.95. If P(positive test) = 0.10, what is P(disease AND positive test)?',
      options: ['0.0095', '0.10', '0.95', '0.9500'],
      correctAnswer: 0,
      explanation:
        'Use the multiplication rule: P(disease ∩ positive) = P(positive | disease) × P(disease) = 0.95 × 0.01 = 0.0095. This is about 0.95% of the population.',
    },
    {
      id: 'mc3',
      question: 'What does the Naive Bayes classifier assume about features?',
      options: [
        'Features are completely uncorrelated',
        'Features are conditionally independent given the class label',
        'Features follow a normal distribution',
        'Features have equal importance',
      ],
      correctAnswer: 1,
      explanation:
        'Naive Bayes assumes features are conditionally independent given the class: P(x₁,x₂,...|y) = P(x₁|y)×P(x₂|y)×... Features can be correlated overall, but are assumed independent within each class.',
    },
    {
      id: 'mc4',
      question: 'If P(A|B) = P(A), what can we conclude?',
      options: [
        'A and B are mutually exclusive',
        'A and B are independent',
        'A causes B',
        'A and B always occur together',
      ],
      correctAnswer: 1,
      explanation:
        "P(A|B) = P(A) is the definition of independence - knowing B occurred doesn't change the probability of A. This is equivalent to P(A ∩ B) = P(A) × P(B).",
    },
    {
      id: 'mc5',
      question:
        'Two features are correlated in your dataset. Can they still be conditionally independent?',
      options: [
        'No, correlation means they are always dependent',
        'Yes, they can be conditionally independent given the class label',
        'Only if they are normally distributed',
        'Only if the correlation is weak',
      ],
      correctAnswer: 1,
      explanation:
        'Features can be correlated overall but conditionally independent given the class. Example: height and weight are correlated, but given the class "adult male", they might be nearly independent. This is why Naive Bayes can work despite violated assumptions.',
    },
  ];
