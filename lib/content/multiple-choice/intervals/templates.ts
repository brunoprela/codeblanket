/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the first step in the merge intervals template?',
    options: [
      'Create result list',
      'Sort intervals by start time',
      'Check for overlaps',
      'Return result',
    ],
    correctAnswer: 1,
    explanation:
      'Always sort intervals by start time first. This enables linear-time merging by comparing each interval only with the last merged one.',
  },
  {
    id: 'mc2',
    question: 'In the insert interval template, what are the three phases?',
    options: [
      'Sort, merge, return',
      'Add intervals before new, merge overlapping, add intervals after',
      'Start, middle, end',
      'Left, center, right',
    ],
    correctAnswer: 1,
    explanation:
      'Insert interval template: 1) Add all intervals ending before new starts (no overlap), 2) Merge all overlapping intervals, 3) Add remaining intervals after overlap region.',
  },
  {
    id: 'mc3',
    question: 'What does the meeting rooms I template check?',
    options: [
      'Number of rooms needed',
      'Whether any two consecutive intervals overlap (one room sufficient)',
      'Maximum overlap',
      'Minimum overlap',
    ],
    correctAnswer: 1,
    explanation:
      'Meeting Rooms I checks if you can attend all meetings. Sort by start time, then check if any consecutive intervals overlap. If none overlap, one room (person) is sufficient.',
  },
  {
    id: 'mc4',
    question:
      'What data structure does meeting rooms II template typically use?',
    options: [
      'Array',
      'Min heap to track earliest ending meetings',
      'Stack',
      'Hash map',
    ],
    correctAnswer: 1,
    explanation:
      'Meeting Rooms II uses min heap to track end times of ongoing meetings. Heap size at any time = rooms needed. Pop when meeting ends, push new end times.',
  },
  {
    id: 'mc5',
    question:
      'In interval intersection template, how do you find the intersection of two intervals?',
    options: [
      'min (start1, start2), max (end1, end2)',
      '[max (start1, start2), min (end1, end2)]',
      'Average of all values',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'Intersection is [max (start1, start2), min (end1, end2)]. The intersection starts at the later start and ends at the earlier end. Valid only if start <= end.',
  },
];
