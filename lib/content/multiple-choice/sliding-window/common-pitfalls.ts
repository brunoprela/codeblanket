/**
 * Multiple choice questions for Common Pitfalls section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonpitfallsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Why is window size calculated as right - left + 1 instead of right - left?',
    options: [
      'It is a convention',
      'Because indices are inclusive on both ends',
      'To make it more complicated',
      "It doesn't matter",
    ],
    correctAnswer: 1,
    explanation:
      'When both left and right indices are inclusive, the number of elements is right - left + 1. For example, from index 2 to 5 inclusive contains elements at 2, 3, 4, 5 which is 4 elements, not 3.',
  },
  {
    id: 'mc2',
    question:
      'What is a common mistake when cleaning up hash maps in sliding windows?',
    options: [
      'Adding too many elements',
      'Not deleting entries with 0 count, affecting size checks',
      'Using the wrong data structure',
      'Clearing the entire map',
    ],
    correctAnswer: 1,
    explanation:
      'After decrementing a count to 0, you must delete the entry from the map. Otherwise, len(freq) includes keys with 0 count, giving incorrect distinct element counts.',
  },
  {
    id: 'mc3',
    question:
      'For a fixed-size window, what is a common mistake in the implementation?',
    options: [
      'Starting from index 0 and recalculating each window',
      'Using too much memory',
      'Not checking the input',
      'Using the wrong loop',
    ],
    correctAnswer: 0,
    explanation:
      'A common mistake is starting from index 0 and recalculating the entire window sum each time (O(n*k)). The correct approach is to calculate the first window once, then slide from index k, updating in O(1).',
  },
  {
    id: 'mc4',
    question:
      'What is the difference in where you update the answer for maximum vs minimum window problems?',
    options: [
      'No difference',
      'Maximum updates outside while loop, minimum updates inside while loop',
      'Maximum uses if, minimum uses while',
      'They both update at the end',
    ],
    correctAnswer: 1,
    explanation:
      'Maximum window problems update answer outside the while loop (after restoring validity). Minimum window problems update inside the while loop (while shrinking a valid window to find the smallest).',
  },
  {
    id: 'mc5',
    question:
      'Why should you not manually increment the right pointer in a for loop?',
    options: [
      'It causes syntax errors',
      'The for loop already increments it, manual increment causes skipping elements',
      'It is too slow',
      'It uses too much memory',
    ],
    correctAnswer: 1,
    explanation:
      'The for loop automatically increments the right pointer. If you manually increment it again (right += 1), you skip elements and break the algorithm logic.',
  },
];
