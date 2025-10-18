/**
 * Multiple choice questions for Heap Operations in Detail section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const operationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What operation is bubble up in a heap?',
    options: [
      'Removing the root',
      'Moving a newly inserted element up until heap property is restored',
      'Sorting the heap',
      'Deleting an element',
    ],
    correctAnswer: 1,
    explanation:
      'Bubble up moves a newly inserted element (added at end) up the tree by swapping with its parent until the heap property is satisfied.',
  },
  {
    id: 'mc2',
    question:
      'In extract-min/max, why do we move the last element to the root?',
    options: [
      'For speed',
      'To maintain complete tree structure while removing root',
      'Random choice',
      'For sorting',
    ],
    correctAnswer: 1,
    explanation:
      'Moving the last element to root maintains the complete tree structure (no gaps). Then bubble down restores the heap property.',
  },
  {
    id: 'mc3',
    question: 'During bubble down, which child do you swap with in a min heap?',
    options: [
      'Always left child',
      'The smaller of the two children',
      'The larger of the two children',
      'Random child',
    ],
    correctAnswer: 1,
    explanation:
      'Swap with the smaller child to maintain min heap property. This ensures the parent is smaller than both children after the swap.',
  },
  {
    id: 'mc4',
    question:
      'What is the time complexity of peek (viewing root without removal)?',
    options: ['O(log N)', 'O(1)', 'O(N)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Peek simply returns the root element (array[0]) without modification, taking O(1) constant time.',
  },
  {
    id: 'mc5',
    question: 'Why is heapify O(N) instead of O(N log N)?',
    options: [
      'It uses a special algorithm',
      'Most nodes are near leaves doing little work, amortized analysis shows O(N)',
      'It is actually O(N log N)',
      'Magic',
    ],
    correctAnswer: 1,
    explanation:
      'Heapify works bottom-up. Nodes near leaves (which are most numerous) do little work. Careful amortized analysis shows total work is O(N), not O(N log N).',
  },
];
