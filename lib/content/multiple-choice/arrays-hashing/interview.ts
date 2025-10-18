/**
 * Multiple choice questions for Interview Strategy & Tips section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What keyword in a problem statement strongly suggests using a hash table?',
    options: ['Sorted', 'Binary', 'Count, frequency, or group', 'Recursive'],
    correctAnswer: 2,
    explanation:
      'Keywords like "count", "frequency", "group", "unique", "duplicate", or "exists" strongly indicate that hash tables will be useful for O(1) lookups and tracking.',
  },
  {
    id: 'mc2',
    question:
      'In an interview, when should you mention the brute force approach?',
    options: [
      'Never mention it, go straight to optimal',
      'Only if asked',
      'Always state it first, then explain why you can optimize',
      'Only for easy problems',
    ],
    correctAnswer: 2,
    explanation:
      "Always state the brute force approach first (even if just briefly) to show you understand the problem, then explain why it's not optimal and how you can improve it.",
  },
  {
    id: 'mc3',
    question:
      'What is a common mistake when implementing two sum with a hash table?',
    options: [
      'Using O(n) space',
      'Using the same element twice by not checking indices',
      'Iterating through the array',
      'Using a dictionary',
    ],
    correctAnswer: 1,
    explanation:
      'A common mistake is using the same array element twice. You must ensure the complement you find is at a different index than the current element.',
  },
  {
    id: 'mc4',
    question: 'How should you respond if asked "What about hash collisions?"',
    options: [
      'Say hash collisions never happen',
      "Explain that Python's hash function is robust, average case is O(1) but worst case is O(n)",
      'Say you would use a different data structure',
      'Say you would sort the data instead',
    ],
    correctAnswer: 1,
    explanation:
      "Acknowledge that collisions exist and affect worst-case complexity (O(n)), but Python's hash function is well-designed for average-case O(1) performance.",
  },
  {
    id: 'mc5',
    question:
      'What is the typical time range for solving a medium hash table problem in an interview?',
    options: [
      '5-10 minutes',
      '15-20 minutes',
      '25-30 minutes',
      '45-60 minutes',
    ],
    correctAnswer: 1,
    explanation:
      'Medium hash table problems typically take 15-20 minutes to solve in an interview, including clarification, explanation, coding, and testing phases.',
  },
];
