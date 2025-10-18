/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords in a problem statement suggest using a stack?',
    options: [
      'Sorted, binary, search',
      'Valid, matching, balanced, next greater',
      'Tree, graph, connected',
      'Minimum, maximum, optimize',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "valid", "matching", "balanced" (for parentheses), or "next greater" (for monotonic stack) are strong signals that a stack-based solution is appropriate.',
  },
  {
    id: 'mc2',
    question: 'How long should a medium stack problem take in an interview?',
    options: ['5-10 minutes', '15-20 minutes', '30-45 minutes', '60+ minutes'],
    correctAnswer: 1,
    explanation:
      'Medium stack problems typically take 15-20 minutes including explanation, coding, and testing. Stack problems often have clear patterns once recognized.',
  },
  {
    id: 'mc3',
    question:
      'What should you explain first when solving a stack problem in an interview?',
    options: [
      'The code implementation',
      'Why you chose a stack and which pattern you are using',
      'The test cases',
      'The complexity analysis',
    ],
    correctAnswer: 1,
    explanation:
      'Start by explaining why a stack is appropriate and which pattern (matching pairs, monotonic stack, etc.) you are using. This shows your problem recognition skills.',
  },
  {
    id: 'mc4',
    question:
      'What is a good response when asked "Can you solve it without a stack?"',
    options: [
      'Say it is impossible',
      'Discuss alternatives like counters for simple cases, but explain stack trade-offs',
      'Refuse to answer',
      'Say stacks are always required',
    ],
    correctAnswer: 1,
    explanation:
      'Some problems (like simple parentheses counting) have non-stack solutions, but monotonic stack problems typically need the stack. Discuss the trade-offs and explain why the stack solution is cleaner.',
  },
  {
    id: 'mc5',
    question: 'What is the recommended practice approach for mastering stacks?',
    options: [
      'Only solve hard problems',
      'Start with basics (valid parentheses), then monotonic stack, then advanced',
      'Practice randomly',
      'Memorize all solutions',
    ],
    correctAnswer: 1,
    explanation:
      'Progress from basic problems (valid parentheses, implement stack) to monotonic stack problems, then advanced applications. This builds understanding incrementally.',
  },
];
