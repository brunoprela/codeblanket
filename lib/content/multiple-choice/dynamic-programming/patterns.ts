/**
 * Multiple choice questions for Common DP Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the 1D DP pattern?',
    options: [
      'Random',
      'dp[i] depends on previous indices like dp[i-1], dp[i-2] - linear sequence problems',
      'Always 2D',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      '1D DP: dp[i] represents state at position i. Depends on earlier indices. Examples: Fibonacci, climbing stairs, house robber. Recurrence: dp[i] = f(dp[i-1], dp[i-2], ...).',
  },
  {
    id: 'mc2',
    question: 'What is the 2D DP pattern?',
    options: [
      'Matrix multiplication',
      'dp[i][j] represents state with two dimensions - grid paths, LCS, edit distance',
      'Random',
      'Never used',
    ],
    correctAnswer: 1,
    explanation:
      '2D DP: dp[i][j] for problems with two sequences/dimensions. Examples: longest common subsequence, edit distance, grid paths. Often dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]).',
  },
  {
    id: 'mc3',
    question: 'What is the knapsack pattern?',
    options: [
      'Packing algorithm',
      'Choose/skip items with capacity constraint - dp[i][w] = max value with i items, capacity w',
      'Random',
      'Sorting',
    ],
    correctAnswer: 1,
    explanation:
      'Knapsack: dp[i][w] = maximum value using first i items with weight limit w. For each item: take it (dp[i-1][w-weight]+value) or skip (dp[i-1][w]). Choose max.',
  },
  {
    id: 'mc4',
    question: 'What is the subsequence pattern?',
    options: [
      'Sequential processing',
      'Find optimal subsequence - LIS, LCS - often compare/match elements at i and j',
      'Random',
      'Substring',
    ],
    correctAnswer: 1,
    explanation:
      'Subsequence: elements in order but not necessarily contiguous. LIS: dp[i] = longest increasing ending at i. LCS: dp[i][j] = longest common of s1[0..i] and s2[0..j]. Match or skip patterns.',
  },
  {
    id: 'mc5',
    question: 'What is the partition pattern?',
    options: [
      'Divide array',
      'Split into groups optimally - partition equal subset sum, palindrome partitioning',
      'Random',
      'Sorting',
    ],
    correctAnswer: 1,
    explanation:
      'Partition: split into optimal groups. Example: partition equal subset sum - dp[i][s] = can partition first i elements into sum s. Palindrome partition: dp[i] = min cuts for s[0..i].',
  },
];
