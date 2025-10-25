/**
 * Multiple choice questions for The 5-Step DP Framework section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const stepsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the first step in the DP framework?',
    options: [
      'Write code',
      'Define DP state - what does dp[i] represent?',
      'Find base case',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Step 1: Define state clearly. What does dp[i] or dp[i][j] mean? Example: dp[i] = min cost to reach stair i. Clear state definition guides rest of solution.',
  },
  {
    id: 'mc2',
    question: 'What is the recurrence relation in DP?',
    options: [
      'Random formula',
      'Equation expressing current state in terms of previous states: dp[i] = f (dp[i-1], dp[i-2], ...)',
      'Base case',
      'Loop',
    ],
    correctAnswer: 1,
    explanation:
      'Recurrence relation: formula expressing dp[i] using previous states. Example: dp[i] = dp[i-1] + dp[i-2] for Fibonacci. Shows how to build solution from subproblems.',
  },
  {
    id: 'mc3',
    question: 'Why are base cases critical in DP?',
    options: [
      'Optional',
      'Bootstrap the recursion/iteration - smallest subproblems with known answers',
      'Random',
      'For speed',
    ],
    correctAnswer: 1,
    explanation:
      'Base cases: smallest subproblems with direct answers. Example: fib(0)=0, fib(1)=1. Without base cases, recursion never terminates or table has no starting point.',
  },
  {
    id: 'mc4',
    question: 'What order should you compute DP table?',
    options: [
      'Random order',
      'Ensure dependencies computed before current state - topological order',
      'Reverse order',
      'Any order',
    ],
    correctAnswer: 1,
    explanation:
      'Compute in order where dependencies available. For dp[i] = dp[i-1] + dp[i-2], compute i=0,1,2,... sequentially. For 2D, often row-by-row or column-by-column. Ensure required values already computed.',
  },
  {
    id: 'mc5',
    question: 'What is the final step in DP?',
    options: [
      'Print everything',
      'Return/extract answer from DP table (often dp[n] or max/min of certain cells)',
      'Optimize space',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Final step: extract answer from DP table. Often dp[n], dp[n][m], or max/min of specific cells depending on problem. Example: longest subsequence = dp[n-1], max profit = dp[n-1][k].',
  },
];
