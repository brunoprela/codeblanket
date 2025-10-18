/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What signals in a problem description indicate you should use linked list techniques?',
    options: [
      'Array sorting required',
      '"Linked list", cycles, middle/nth from end, frequent insertions/deletions',
      'Matrix operations',
      'String manipulation',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "linked list", "cycle", "middle", "nth from end", or scenarios with frequent insertions/deletions without random access needs indicate linked list techniques.',
  },
  {
    id: 'mc2',
    question:
      'What should you clarify first in a linked list interview problem?',
    options: [
      'The programming language',
      'Whether it is singly or doubly linked and if in-place modification is allowed',
      "The interviewer's name",
      'The company size',
    ],
    correctAnswer: 1,
    explanation:
      'Always clarify whether the list is singly or doubly linked and whether in-place modification is allowed, as this affects your approach and space complexity.',
  },
  {
    id: 'mc3',
    question: 'Why should you draw diagrams when solving linked list problems?',
    options: [
      'To waste time',
      'To visualize pointer movements and avoid mistakes',
      'It is not necessary',
      'To impress the interviewer',
    ],
    correctAnswer: 1,
    explanation:
      'Drawing diagrams helps visualize pointer movements step by step, making it easier to track state and avoid common mistakes like null pointer errors or losing references.',
  },
  {
    id: 'mc4',
    question:
      'When asked "Can you do it in O(1) space?", what does this typically mean for linked lists?',
    options: [
      'Use arrays instead',
      'Use pointer manipulation instead of hash sets/maps',
      'Use more memory',
      'It cannot be done',
    ],
    correctAnswer: 1,
    explanation:
      'O(1) space requirement means avoiding auxiliary data structures like hash sets/maps and instead using pointer manipulation techniques like fast/slow pointers or dummy nodes.',
  },
  {
    id: 'mc5',
    question:
      'What is the recommended practice progression for linked list mastery?',
    options: [
      'Start with the hardest problems',
      'Start with basics (reverse, middle), then two pointers, then merging, then advanced',
      'Skip basics and do advanced',
      'Only practice one type',
    ],
    correctAnswer: 1,
    explanation:
      'Progress from basic operations (reverse, find middle) to two-pointer techniques (cycles, nth from end), then merging/rearranging, and finally advanced problems. This builds intuition incrementally.',
  },
];
