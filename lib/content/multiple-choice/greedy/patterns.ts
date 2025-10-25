/**
 * Multiple choice questions for Common Greedy Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In activity selection, why sort by end time instead of start time?',
    options: [
      "It\'s faster",
      'Activities finishing earliest leave maximum time for remaining activities',
      'Random choice',
      'Easier to implement',
    ],
    correctAnswer: 1,
    explanation:
      'Sorting by end time and selecting earliest-finishing activities maximizes remaining time for future selections, giving optimal non-overlapping count. This is the greedy choice.',
  },
  {
    id: 'mc2',
    question: 'What is the greedy choice for fractional knapsack?',
    options: [
      'Pick heaviest items first',
      'Pick items with best value/weight ratio first',
      'Pick lightest items first',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'Fractional knapsack: sort by value/weight ratio, take items with best ratio first (can take fractions). This maximizes value per unit weight, giving optimal solution.',
  },
  {
    id: 'mc3',
    question: 'In two-pointer greedy problems, what is the typical strategy?',
    options: [
      'Move both pointers together',
      'Process extremes first (largest/smallest), move pointers based on greedy criterion',
      'Random pointer movement',
      'Only move one pointer',
    ],
    correctAnswer: 1,
    explanation:
      'Two-pointer greedy: sort array, use pointers at extremes. Make greedy choice based on which extreme to take (e.g., container with most water: move pointer with smaller height).',
  },
  {
    id: 'mc4',
    question:
      'What data structure is often used for greedy resource allocation?',
    options: [
      'Array',
      'Priority queue (heap) to efficiently get best choice',
      'Stack',
      'Linked list',
    ],
    correctAnswer: 1,
    explanation:
      'Priority queue (heap) maintains best available choice at top, enabling O(log N) greedy selections. Used for scheduling, task allocation, and other resource problems.',
  },
  {
    id: 'mc5',
    question: 'Why does greedy work for Huffman coding?',
    options: [
      "It\'s simple",
      'Merging two lowest-frequency nodes minimizes total encoding length (greedy choice property)',
      'Random',
      'Always works',
    ],
    correctAnswer: 1,
    explanation:
      'Huffman coding: repeatedly merge two lowest-frequency nodes. This greedy choice minimizes weighted path length (encoding cost), proven optimal by exchange argument.',
  },
];
