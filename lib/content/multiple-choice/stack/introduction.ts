/**
 * Multiple choice questions for Introduction to Stacks section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does LIFO stand for in the context of stacks?',
    options: [
      'Last-In-First-Out',
      'Last-In-Forever-Out',
      'Linear-In-First-Out',
      'List-In-First-Out',
    ],
    correctAnswer: 0,
    explanation:
      'LIFO stands for Last-In-First-Out, meaning the most recently added element is the first one to be removed, like a stack of plates where you only take from the top.',
  },
  {
    id: 'mc2',
    question:
      'What is the time complexity of push and pop operations on a stack?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 0,
    explanation:
      'Both push and pop are O(1) constant time operations because they only interact with the top of the stack, requiring no searching or traversal.',
  },
  {
    id: 'mc3',
    question:
      'Which Python data structure is commonly used to implement a stack?',
    options: ['Dictionary', 'Set', 'List (using append and pop)', 'Tuple'],
    correctAnswer: 2,
    explanation:
      'Python lists work perfectly as stacks using append() for push and pop() for pop operations, both of which are O(1) at the end of the list.',
  },
  {
    id: 'mc4',
    question: 'What real-world example best demonstrates the LIFO principle?',
    options: [
      'A queue at a store',
      'Browser back button history',
      'A sorted list',
      'A hash table',
    ],
    correctAnswer: 1,
    explanation:
      'The browser back button works like a stack - you navigate back to the most recently visited page first, which is exactly the LIFO (Last-In-First-Out) principle.',
  },
  {
    id: 'mc5',
    question: 'What type of problems are stacks uniquely good at solving?',
    options: [
      'Sorting problems',
      'Parsing and matching pairs problems',
      'Finding minimum elements',
      'Binary search',
    ],
    correctAnswer: 1,
    explanation:
      'Stacks excel at parsing and matching pairs (like parentheses validation) because they naturally track the most recent opening symbol to match with closing symbols.',
  },
];
