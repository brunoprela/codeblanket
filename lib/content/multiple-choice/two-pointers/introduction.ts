/**
 * Multiple choice questions for What is the Two Pointers Technique? section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary benefit of the two pointers technique?',
    options: [
      'It reduces space complexity to O(1)',
      'It reduces time complexity from O(n²) to O(n)',
      'It only works on sorted arrays',
      'It uses recursion',
    ],
    correctAnswer: 1,
    explanation:
      'The two pointers technique typically reduces time complexity from O(n²) (nested loops checking all pairs) to O(n) by making smart decisions about which pointer to move based on the data properties.',
  },
  {
    id: 'mc2',
    question:
      'Which type of data structure property makes two pointers most effective?',
    options: [
      'Unsorted arrays',
      'Sorted or sortable data',
      'Hash tables',
      'Binary trees',
    ],
    correctAnswer: 1,
    explanation:
      'Two pointers works best with sorted or sortable data because the sorted property allows us to make predictable decisions about which pointer to move to progress toward the solution.',
  },
  {
    id: 'mc3',
    question: 'What are the three main movement patterns for two pointers?',
    options: [
      'Left, right, and center',
      'Fast, slow, and medium',
      'Opposite direction, same direction, and sliding window',
      'Forward, backward, and circular',
    ],
    correctAnswer: 2,
    explanation:
      'The three main patterns are: opposite direction (converging from both ends), same direction (fast & slow pointers), and sliding window (defining an expanding/shrinking window).',
  },
  {
    id: 'mc4',
    question:
      'When should you consider using two pointers instead of nested loops?',
    options: [
      'Only for very small arrays',
      'When you need to find pairs/triplets or work with sorted data',
      'When working with trees',
      'Only for string manipulation',
    ],
    correctAnswer: 1,
    explanation:
      'Two pointers is ideal for finding pairs/triplets with certain properties, working with sorted data, palindromes, in-place operations, or anytime you catch yourself thinking about nested loops.',
  },
  {
    id: 'mc5',
    question: 'What is the space complexity of most two-pointer solutions?',
    options: ['O(n)', 'O(log n)', 'O(1)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'Most two-pointer solutions use O(1) constant space since they only need a few pointer variables and often modify the array in-place without creating new data structures.',
  },
];
