/**
 * Multiple choice questions for DP Pattern Recognition & Decision Guide section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternrecognitionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which problem pattern does "Find the minimum number of coins to make a given amount" match?',
    options: [
      'Fibonacci pattern',
      'Knapsack pattern',
      'Coin Change (Unbounded Knapsack) pattern',
      'LCS pattern',
    ],
    correctAnswer: 2,
    explanation:
      'This is the classic Coin Change pattern, which is a variant of Unbounded Knapsack (each coin can be used unlimited times). State: dp[i] = minimum coins to make amount i.',
  },
  {
    id: 'mc2',
    question:
      'If a DP problem asks "count the number of ways to achieve X", what should your recurrence likely use?',
    options: [
      'max() to find the best way',
      'min() to find the shortest way',
      'sum() to add up ways from different paths',
      'multiply() to combine paths',
    ],
    correctAnswer: 2,
    explanation:
      'Counting problems typically sum the number of ways from different paths. For example, climbing stairs: dp[i] = dp[i-1] + dp[i-2] (sum of ways to reach from previous steps).',
  },
  {
    id: 'mc3',
    question: 'What is the main difference between top-down and bottom-up DP?',
    options: [
      'Top-down is always faster',
      'Top-down uses recursion with memoization, bottom-up uses iteration',
      'Bottom-up always uses less space',
      'They solve different types of problems',
    ],
    correctAnswer: 1,
    explanation:
      'Top-down uses recursion with memoization (start from original problem, break down). Bottom-up uses iteration with tabulation (start from smallest subproblems, build up).',
  },
  {
    id: 'mc4',
    question:
      'Which pattern should you use for "Can you partition array into two equal sum subsets"?',
    options: [
      'Fibonacci pattern',
      'Grid path pattern',
      'Knapsack 0/1 pattern',
      'LCS pattern',
    ],
    correctAnswer: 2,
    explanation:
      'This is a 0/1 Knapsack variant. Each element can be included or excluded (0/1 choice), trying to reach target sum (half of total). State: dp[i][sum] = can make sum with first i numbers.',
  },
  {
    id: 'mc5',
    question: 'When should you NOT use DP?',
    options: [
      'When greedy algorithm gives optimal solution',
      'When there are overlapping subproblems',
      'When you need to find optimal value',
      'When the problem asks to count ways',
    ],
    correctAnswer: 0,
    explanation:
      "Don't use DP when a greedy algorithm gives the optimal solution (DP would be overkill). Also skip DP when there are no overlapping subproblems or when a simple formula exists.",
  },
];
