/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the standard 1D DP template?',
    options: [
      'Random',
      'Initialize dp array, set base cases, loop i from start to end, compute dp[i] from previous, return dp[n]',
      'No template',
      'Always recursive',
    ],
    correctAnswer: 1,
    explanation:
      '1D template: 1) Create dp[n+1], 2) Base case dp[0], 3) for i in 1..n: dp[i] = f(dp[i-1], dp[i-2],...), 4) return dp[n]. Works for most 1D problems.',
  },
  {
    id: 'mc2',
    question: 'What is the standard 2D DP template?',
    options: [
      'Random',
      'Create dp[m][n], base cases first row/column, nested loops, dp[i][j] from neighbors, return dp[m-1][n-1]',
      'No template',
      'Only 1D',
    ],
    correctAnswer: 1,
    explanation:
      '2D template: 1) Create dp[m][n], 2) Base: dp[0][j] and dp[i][0], 3) for i,j: dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]), 4) return dp[m-1][n-1] or max.',
  },
  {
    id: 'mc3',
    question: 'What is the top-down memoization template?',
    options: [
      'Random',
      'Recursive function with memo dict, check memo first, compute and store if not cached, return',
      'Only iterative',
      'No template',
    ],
    correctAnswer: 1,
    explanation:
      'Memoization template: def solve(params, memo={}): if base_case: return value; if params in memo: return memo[params]; result = recurse; memo[params] = result; return result.',
  },
  {
    id: 'mc4',
    question: 'What is the knapsack template?',
    options: [
      'Random',
      'dp[i][w] = max value with i items, capacity w. For each item: max(take, skip)',
      'Sorting only',
      'No template',
    ],
    correctAnswer: 1,
    explanation:
      'Knapsack: dp[i][w] for i items, capacity w. Recurrence: dp[i][w] = max(dp[i-1][w] skip, dp[i-1][w-weight[i]]+value[i] take). Return dp[n][W].',
  },
  {
    id: 'mc5',
    question: 'When should you use DP templates?',
    options: [
      'Never',
      'Starting point to understand structure, adapt to specific problem, not rigid formula',
      'Always exactly',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "Templates provide structure and common patterns. Use as starting point, adapt to problem specifics. Understand principles, don't memorize blindly. Each problem may need variations.",
  },
];
