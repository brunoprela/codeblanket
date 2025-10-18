/**
 * Multiple choice questions for Advanced Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'After detecting a cycle, how do you find where the cycle starts?',
    options: [
      'Use a hash map',
      'Reset one pointer to head, move both one step at a time until they meet',
      'Count all nodes',
      'Reverse the list',
    ],
    correctAnswer: 1,
    explanation:
      'After fast and slow meet in the cycle, reset one pointer to head and keep the other at the meeting point. Move both one step at a time - they will meet at the cycle start due to the mathematical relationship x = z.',
  },
  {
    id: 'mc2',
    question: 'Why is merge sort preferred over quick sort for linked lists?',
    options: [
      'Quick sort is always slower',
      "Merge sort doesn't require random access which linked lists lack",
      'Merge sort uses less memory',
      "Quick sort doesn't work on lists",
    ],
    correctAnswer: 1,
    explanation:
      "Merge sort works well with linked lists because it doesn't require random access. It uses find-middle, recursively sort, and merge operations that work naturally with linked list structure. Quick sort needs efficient random access for partitioning.",
  },
  {
    id: 'mc3',
    question: 'When reversing nodes in k-groups, what is the main challenge?',
    options: [
      'Finding k nodes',
      'Managing boundary connections between reversed groups',
      'Counting nodes',
      'It is actually easy',
    ],
    correctAnswer: 1,
    explanation:
      'The main challenge is correctly managing connections: save node before group, reverse k nodes, connect previous group tail to reversed head, connect reversed tail to next group. Many pointers to track.',
  },
  {
    id: 'mc4',
    question:
      'How can you copy a linked list with random pointers in O(N) time and O(1) space?',
    options: [
      'Use a hash map',
      'Interweave copied nodes with original nodes, then separate',
      'Use recursion',
      'It cannot be done in O(1) space',
    ],
    correctAnswer: 1,
    explanation:
      "Create new nodes and interweave them with originals (A->A'->B->B'), copy random pointers using the interweaved structure, then separate the two lists. This avoids the hash map.",
  },
  {
    id: 'mc5',
    question:
      'What is the time complexity of finding the middle of a linked list using fast/slow pointers?',
    options: ['O(1)', 'O(N)', 'O(N²)', 'O(log N)'],
    correctAnswer: 1,
    explanation:
      'The fast/slow pointer technique requires traversing the list once. Fast moves 2 steps while slow moves 1 step, so they traverse N nodes total, giving O(N) time complexity.',
  },
];
