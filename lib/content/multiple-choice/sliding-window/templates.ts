/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In a fixed-size window template, when do you start sliding the window?',
    options: [
      'From index 0',
      'From index k (after building initial window)',
      'From index k-1',
      'After checking all elements',
    ],
    correctAnswer: 1,
    explanation:
      'You start sliding from index k because indices 0 to k-1 were used to build the initial window. From index k onward, each new element entering triggers a slide.',
  },
  {
    id: 'mc2',
    question:
      'What is the difference between the shrinkable and non-shrinkable window templates?',
    options: [
      'One uses while to shrink, the other uses if to move left once',
      'One is faster than the other',
      'One works on strings, the other on arrays',
      'They are the same',
    ],
    correctAnswer: 0,
    explanation:
      'Shrinkable uses "while" to fully restore validity by shrinking multiple times if needed. Non-shrinkable uses "if" to move left pointer once, maintaining maximum window size and just sliding forward.',
  },
  {
    id: 'mc3',
    question:
      'In the variable-size shrinkable template, where do you update the answer?',
    options: [
      'Before adding the right element',
      'Inside the while loop',
      'After the while loop, using the current valid window',
      'At the end of the algorithm',
    ],
    correctAnswer: 2,
    explanation:
      "You update the answer after the while loop because that's when you have a valid window. The while loop restores validity, then you check if this valid window is better than your current answer.",
  },
  {
    id: 'mc4',
    question:
      'When using a hash set to track unique elements in a window, what operation is performed when shrinking?',
    options: [
      'Add elements',
      'Sort the set',
      'Remove arr[left] from set, then increment left',
      'Clear the entire set',
    ],
    correctAnswer: 2,
    explanation:
      'When shrinking, you remove the element at the left pointer from the set (to remove it from the window), then increment the left pointer to move the window boundary.',
  },
  {
    id: 'mc5',
    question:
      'What is a common pattern for minimum window problems vs maximum window problems?',
    options: [
      'They use the same template',
      'Minimum expands until valid then shrinks to minimize; maximum expands while valid and tracks maximum',
      'Minimum is always harder',
      "Maximum problems don't use sliding window",
    ],
    correctAnswer: 1,
    explanation:
      'Minimum window problems expand until finding a valid window, then shrink as much as possible while maintaining validity to minimize. Maximum window problems expand while remaining valid and track the maximum size achieved.',
  },
];
