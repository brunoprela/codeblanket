/**
 * Multiple choice questions for Interval Operations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const operationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the condition for two intervals to overlap?',
    options: [
      'start1 < start2',
      'start1 <= end2 AND start2 <= end1',
      'end1 == start2',
      'start1 == start2',
    ],
    correctAnswer: 1,
    explanation:
      'Two intervals overlap when: start1 <= end2 AND start2 <= end1. This checks if they share any common point.',
  },
  {
    id: 'mc2',
    question:
      'In the merge intervals pattern, when do you extend the current interval?',
    options: [
      'Always',
      "When new interval's start <= current interval's end (overlap detected)",
      'When new interval is larger',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "After sorting by start time, if new interval's start <= current end, they overlap. Extend current to max(current.end, new.end).",
  },
  {
    id: 'mc3',
    question: 'Why sort intervals before merging?',
    options: [
      'Required by problem',
      'Ensures intervals in order, only need to check consecutive pairs',
      'Makes output sorted',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Sorting by start time ensures we process intervals in order. Then we only need to compare each interval with the last merged one, not all pairs - O(N) instead of O(N²).',
  },
  {
    id: 'mc4',
    question: 'What is the intersection of intervals [1,4] and [2,5]?',
    options: ['[1,5]', '[2,4]', '[1,2]', 'No intersection'],
    correctAnswer: 1,
    explanation:
      'Intersection is [max(start1, start2), min(end1, end2)] = [max(1,2), min(4,5)] = [2,4].',
  },
  {
    id: 'mc5',
    question: 'What operation does "remove interval" perform?',
    options: [
      'Deletes an interval',
      'Subtracts one interval from another, potentially splitting it',
      'Merges intervals',
      'Sorts intervals',
    ],
    correctAnswer: 1,
    explanation:
      'Removing [2,4] from [1,5] splits it into [1,2] and [4,5]. The operation removes the overlapping part and keeps non-overlapping parts.',
  },
];
