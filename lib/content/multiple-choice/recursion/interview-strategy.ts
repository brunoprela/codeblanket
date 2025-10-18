/**
 * Multiple choice questions for Recursion in Interviews section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-interview1',
    question:
      'In an interview, what should you do first when presented with a recursive problem?',
    options: [
      'Start coding immediately',
      'Define the base case',
      'Clarify inputs/outputs and discuss approach',
      'Analyze time complexity',
    ],
    correctAnswer: 2,
    explanation:
      'Always clarify the problem (inputs, outputs, edge cases) and discuss your approach before coding. This shows communication skills and prevents wasted time.',
  },
  {
    id: 'mc-interview2',
    question:
      'What is the most common mistake in recursive interview solutions?',
    options: [
      'Forgetting to define the function',
      'Missing or incorrect base case',
      'Using too many variables',
      'Not using helper functions',
    ],
    correctAnswer: 1,
    explanation:
      'Missing or incorrect base cases lead to infinite recursion or wrong results. Always identify and test base cases first.',
  },
  {
    id: 'mc-interview3',
    question: 'When should you mention memoization in an interview?',
    options: [
      "Never, it's too advanced",
      'Only if explicitly asked',
      'When you identify overlapping subproblems',
      'Always, regardless of the problem',
    ],
    correctAnswer: 2,
    explanation:
      'Mention memoization when you identify repeated calculations (overlapping subproblems). This shows optimization thinking and understanding of DP.',
  },
  {
    id: 'mc-interview4',
    question: 'How should you test your recursive solution in an interview?',
    options: [
      'Only test the final version',
      'Test base cases first, then build up to larger inputs',
      'Start with the largest input',
      'Skip testing if confident',
    ],
    correctAnswer: 1,
    explanation:
      'Test base cases first to verify they work, then progressively test larger inputs (n=0,1,2,...). This systematic approach catches bugs early.',
  },
  {
    id: 'mc-interview5',
    question: 'What demonstrates strong recursion skills in an interview?',
    options: [
      'Writing code without explanation',
      'Only using recursion for everything',
      'Clearly explaining the subproblem, base case, and complexity',
      'Memorizing solutions',
    ],
    correctAnswer: 2,
    explanation:
      'Strong candidates clearly articulate: what subproblem the function solves, why the base case works, and analyze time/space complexity. Communication is key.',
  },
];
