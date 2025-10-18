/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal an interval problem?',
    options: [
      'Array, list, tree',
      'Merge, overlap, schedule, meeting rooms, conflict',
      'Sort, search, find',
      'Hash, map, set',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "merge", "overlap", "schedule", "calendar", "meeting rooms", "conflict", "range", and "time periods" strongly indicate interval-based solutions.',
  },
  {
    id: 'mc2',
    question:
      'What is the first thing to clarify in an interval interview problem?',
    options: [
      'Complexity requirements',
      'Are intervals inclusive/exclusive? Can I sort? Can I modify input?',
      'Language preference',
      'Test cases',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: Are endpoints inclusive or exclusive? Are intervals already sorted? Can you modify the input? These affect your solution approach significantly.',
  },
  {
    id: 'mc3',
    question:
      'What is the most common first step when solving interval problems?',
    options: [
      'Create result list',
      'Sort intervals by start time (unless already sorted)',
      'Count intervals',
      'Find maximum',
    ],
    correctAnswer: 1,
    explanation:
      'Almost all interval problems start by sorting intervals by start time. This enables efficient linear-time processing and overlap detection.',
  },
  {
    id: 'mc4',
    question: 'What is a common mistake in interval problems?',
    options: [
      'Sorting correctly',
      'Wrong overlap check (using < instead of <=, missing touching intervals)',
      'Using correct data structures',
      'Good variable names',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistake: using start < end instead of start <= end for overlap check, which misses touching intervals. Always clarify if touching counts as overlap.',
  },
  {
    id: 'mc5',
    question: 'What is a good practice progression for interval problems?',
    options: [
      'Start with hardest',
      'Week 1: Merge/Insert, Week 2: Meeting Rooms, Week 3: Advanced patterns',
      'Random order',
      'Skip basics',
    ],
    correctAnswer: 1,
    explanation:
      'Progress: Week 1 basics (merge, insert), Week 2 variations (meeting rooms, overlaps), Week 3 advanced (scheduling, optimization). Build understanding incrementally.',
  },
];
