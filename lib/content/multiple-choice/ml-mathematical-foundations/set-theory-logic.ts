/**
 * Multiple choice questions for Set Theory & Logic section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const settheorylogicMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-set-operations',
    question:
      'If A = {1, 2, 3, 4} and B = {3, 4, 5, 6}, what is A △ B (symmetric difference)?',
    options: ['{3, 4}', '{1, 2, 5, 6}', '{1, 2, 3, 4, 5, 6}', '{1, 2}'],
    correctAnswer: 1,
    explanation:
      'Symmetric difference A △ B contains elements in A or B but not both. A △ B = (A ∪ B) \\ (A ∩ B) = {1, 2, 3, 4, 5, 6} \\ {3, 4} = {1, 2, 5, 6}.',
  },
  {
    id: 'mc2-cardinality',
    question: 'If |A| = 5, |B| = 7, and |A ∩ B| = 2, what is |A ∪ B|?',
    options: ['12', '10', '9', '14'],
    correctAnswer: 1,
    explanation:
      'Using inclusion-exclusion: |A ∪ B| = |A| + |B| - |A ∩ B| = 5 + 7 - 2 = 10.',
  },
  {
    id: 'mc3-logic',
    question: 'What is the truth value of (TRUE AND FALSE) OR TRUE?',
    options: ['TRUE', 'FALSE', 'Cannot determine', 'Undefined'],
    correctAnswer: 0,
    explanation:
      'Evaluate step by step: (TRUE AND FALSE) = FALSE. Then FALSE OR TRUE = TRUE. Remember: OR returns TRUE if at least one operand is TRUE.',
  },
  {
    id: 'mc4-demorgan',
    question: "According to De Morgan\'s Law, ¬(p ∨ q) is equivalent to:",
    options: ['¬p ∨ ¬q', '¬p ∧ ¬q', 'p ∧ q', '¬p → ¬q'],
    correctAnswer: 1,
    explanation:
      "De Morgan\'s Law: ¬(p ∨ q) = ¬p ∧ ¬q. The negation of OR becomes AND of negations.",
  },
  {
    id: 'mc5-implication',
    question: 'The logical implication p → q is FALSE only when:',
    options: [
      'p is TRUE and q is TRUE',
      'p is FALSE and q is FALSE',
      'p is TRUE and q is FALSE',
      'p is FALSE and q is TRUE',
    ],
    correctAnswer: 2,
    explanation:
      'Implication p → q is only FALSE when the premise (p) is TRUE but the conclusion (q) is FALSE. In all other cases, it\'s TRUE. Think: "If p then q" is only violated when p happens but q doesn\'t.',
  },
];
