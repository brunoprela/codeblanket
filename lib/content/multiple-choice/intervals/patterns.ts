/**
 * Multiple choice questions for Common Interval Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the most common pattern for interval problems?',
    options: [
      'Dynamic programming',
      'Sort by start time + merge/process in one pass',
      'Binary search',
      'Backtracking',
    ],
    correctAnswer: 1,
    explanation:
      'The Sort + Merge pattern is most common: sort intervals by start time, then process in one pass comparing current with last merged. This enables O(N) processing after O(N log N) sort.',
  },
  {
    id: 'mc2',
    question:
      'For counting maximum overlapping intervals, what technique is used?',
    options: [
      'Sort only',
      'Sweep line or min heap to track active intervals',
      'Hash map',
      'DFS',
    ],
    correctAnswer: 1,
    explanation:
      'Use sweep line (sort start/end events) or min heap (track end times) to count active intervals at any time. Max active = maximum overlapping intervals.',
  },
  {
    id: 'mc3',
    question:
      'For maximum non-overlapping intervals (activity selection), what should you sort by?',
    options: ['Start time', 'End time', 'Duration', 'Random'],
    correctAnswer: 1,
    explanation:
      'Sort by end time and greedily select intervals ending earliest. This maximizes remaining time for future intervals, giving optimal non-overlapping selection.',
  },
  {
    id: 'mc4',
    question: 'What is the sweep line technique?',
    options: [
      'Sorting intervals',
      'Processing start/end events in chronological order to track active intervals',
      'Merging intervals',
      'Binary search',
    ],
    correctAnswer: 1,
    explanation:
      'Sweep line treats starts as +1 events and ends as -1 events. Sort all events, process chronologically, track active count. This finds maximum overlapping intervals efficiently.',
  },
  {
    id: 'mc5',
    question:
      'When inserting a new interval into sorted intervals, what are the three phases?',
    options: [
      'Start, middle, end',
      'Before (no overlap), overlap (merge), after (no overlap)',
      'Left, right, center',
      'First, second, third',
    ],
    correctAnswer: 1,
    explanation:
      'Insert interval has 3 phases: 1) Add all intervals ending before new starts (before), 2) Merge all overlapping intervals (overlap), 3) Add remaining intervals (after).',
  },
];
