/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of most interval problems?',
    options: ['O(N)', 'O(N log N) - dominated by sorting', 'O(N²)', 'O(log N)'],
    correctAnswer: 1,
    explanation:
      'Most interval problems require O(N log N) sorting followed by O(N) processing, giving O(N log N) total complexity.',
  },
  {
    id: 'mc2',
    question: 'What is the space complexity of in-place interval merging?',
    options: [
      'O(N)',
      'O(1) - modify input array directly',
      'O(log N)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'If allowed to modify the input, interval merging can be done in-place with O(1) extra space by using write/read pointers.',
  },
  {
    id: 'mc3',
    question:
      'What is the complexity of interval intersection with two pointers?',
    options: [
      'O(N log N)',
      'O(N + M) - linear in sum of list lengths',
      'O(N × M)',
      'O(log N)',
    ],
    correctAnswer: 1,
    explanation:
      'When both interval lists are already sorted, two pointers process each list once, giving O(N + M) time complexity.',
  },
  {
    id: 'mc4',
    question: 'How does using a min heap help in meeting rooms II?',
    options: [
      'Sorts the meetings',
      'Tracks earliest ending meeting in O(log N), avoiding O(N) scan',
      'Counts meetings',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Min heap maintains earliest ending meeting at root. Checking if room is free takes O(log N) instead of O(N) linear scan, making algorithm more efficient.',
  },
  {
    id: 'mc5',
    question: 'What is the benefit of the sweep line technique?',
    options: [
      'Easier to code',
      'Converts overlap counting to event processing - O(N log N) instead of O(N²)',
      'Uses less memory',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Sweep line processes start/end events chronologically instead of comparing all interval pairs. This reduces complexity from O(N²) to O(N log N) for overlap counting.',
  },
];
