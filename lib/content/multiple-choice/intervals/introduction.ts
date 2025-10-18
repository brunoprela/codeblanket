/**
 * Multiple choice questions for Introduction to Intervals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'When do two intervals overlap?',
    options: [
      'When they are equal',
      'When they share at least one point (start1 <= end2 and start2 <= end1)',
      'When they touch',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Two intervals [a,b] and [c,d] overlap when they share at least one point. This happens when: start1 <= end2 AND start2 <= end1.',
  },
  {
    id: 'mc2',
    question: 'Why is sorting usually the first step in interval problems?',
    options: [
      'It makes the output sorted',
      'Establishes order, enabling O(N) overlap detection instead of O(N²)',
      'It is required by the problem',
      'Random choice',
    ],
    correctAnswer: 1,
    explanation:
      'Sorting by start time enables linear-time overlap detection. Without sorting, comparing all pairs takes O(N²). With sorting, one pass detects overlaps in O(N), making the O(N log N) sort worthwhile.',
  },
  {
    id: 'mc3',
    question: 'What is the typical time complexity of interval problems?',
    options: ['O(N)', 'O(N log N) due to initial sorting', 'O(N²)', 'O(log N)'],
    correctAnswer: 1,
    explanation:
      'Most interval problems require O(N log N) time for sorting, followed by O(N) processing, giving O(N log N) total complexity.',
  },
  {
    id: 'mc4',
    question: 'What are intervals commonly used for?',
    options: [
      'Sorting numbers',
      'Scheduling (meetings, tasks), time management, range queries',
      'Graph traversal',
      'String manipulation',
    ],
    correctAnswer: 1,
    explanation:
      'Intervals model time ranges, making them perfect for scheduling (meeting rooms, task scheduling), calendar applications, resource booking, and range-based queries.',
  },
  {
    id: 'mc5',
    question: 'What does it mean for intervals to be "disjoint"?',
    options: [
      'They overlap',
      'No overlap and not touching (completely separate)',
      'They are equal',
      'One contains the other',
    ],
    correctAnswer: 1,
    explanation:
      'Disjoint intervals are completely separate with no overlap or touching. For example, [1,2] and [4,5] are disjoint because end1 < start2.',
  },
];
