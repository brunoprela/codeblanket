/**
 * Multiple choice questions for Sliding Window Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In a fixed-size sliding window of size k, how do you slide the window?',
    options: [
      'Recalculate the entire window sum',
      'Add the new element on the right and subtract the old element on the left',
      'Only move the right pointer',
      'Sort the window elements',
    ],
    correctAnswer: 1,
    explanation:
      'To slide a fixed-size window, add the new element entering on the right and subtract the element leaving on the left. This maintains the window size and updates in O(1) time.',
  },
  {
    id: 'mc2',
    question:
      'For variable-size windows, when do you expand vs shrink the window?',
    options: [
      'Always expand first, then shrink',
      'Expand when condition not met, shrink when condition violated',
      'Random based on problem',
      'Expand and shrink simultaneously',
    ],
    correctAnswer: 1,
    explanation:
      'Expand the window (move right pointer) to include more elements when searching for a valid window. Shrink (move left pointer) when the window violates constraints, trying to minimize while maintaining validity.',
  },
  {
    id: 'mc3',
    question:
      'What is the longest substring without repeating characters problem pattern?',
    options: [
      'Fixed-size window',
      'Variable-size window maximizing length',
      'Two pointers opposite direction',
      'Binary search',
    ],
    correctAnswer: 1,
    explanation:
      'This is a variable-size window pattern where you maximize the window length. Expand to add characters, shrink when you encounter a duplicate, tracking the maximum valid window size.',
  },
  {
    id: 'mc4',
    question:
      'In minimum window substring, what auxiliary data structure is typically used?',
    options: [
      'Stack',
      'Queue',
      'Hash map to count character frequencies',
      'Binary tree',
    ],
    correctAnswer: 2,
    explanation:
      'Minimum window substring uses hash maps: one for target character counts and one for current window counts. This enables O(1) updates and validity checks.',
  },
  {
    id: 'mc5',
    question:
      'What is the key difference between longest and shortest window problems?',
    options: [
      'Longest expands more, shortest shrinks more',
      'Longest maximizes valid windows, shortest minimizes valid windows',
      'They are the same',
      'Longest is easier',
    ],
    correctAnswer: 1,
    explanation:
      'Longest window problems maximize the size of valid windows by expanding when possible and recording maximum. Shortest window problems minimize by shrinking valid windows as much as possible while maintaining validity.',
  },
];
