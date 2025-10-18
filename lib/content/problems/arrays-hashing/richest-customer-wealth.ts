/**
 * Richest Customer Wealth
 * Problem ID: richest-customer-wealth
 * Order: 26
 */

import { Problem } from '../../../types';

export const richest_customer_wealthProblem: Problem = {
  id: 'richest-customer-wealth',
  title: 'Richest Customer Wealth',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  description: `You are given an \`m x n\` integer grid \`accounts\` where \`accounts[i][j]\` is the amount of money the \`i-th\` customer has in the \`j-th\` bank. Return the wealth that the richest customer has.

A customer's wealth is the amount of money they have in all their bank accounts. The richest customer is the customer that has the maximum wealth.`,
  examples: [
    {
      input: 'accounts = [[1,2,3],[3,2,1]]',
      output: '6',
      explanation:
        'Customer 0 has wealth = 1+2+3 = 6. Customer 1 has wealth = 3+2+1 = 6. Both customers have wealth 6.',
    },
    {
      input: 'accounts = [[1,5],[7,3],[3,5]]',
      output: '10',
      explanation:
        'Customer 0 has wealth = 1+5 = 6. Customer 1 has wealth = 7+3 = 10. Customer 2 has wealth = 3+5 = 8. The richest customer has wealth 10.',
    },
  ],
  constraints: [
    'm == accounts.length',
    'n == accounts[i].length',
    '1 <= m, n <= 50',
    '1 <= accounts[i][j] <= 100',
  ],
  hints: [
    'Calculate the sum for each customer',
    'Keep track of the maximum sum seen',
  ],
  starterCode: `from typing import List

def maximum_wealth(accounts: List[List[int]]) -> int:
    """
    Find the maximum wealth among all customers.
    
    Args:
        accounts: 2D array where accounts[i][j] is money in bank j for customer i
        
    Returns:
        Maximum wealth
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 2, 3],
          [3, 2, 1],
        ],
      ],
      expected: 6,
    },
    {
      input: [
        [
          [1, 5],
          [7, 3],
          [3, 5],
        ],
      ],
      expected: 10,
    },
    {
      input: [
        [
          [2, 8, 7],
          [7, 1, 3],
          [1, 9, 5],
        ],
      ],
      expected: 17,
    },
  ],
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/richest-customer-wealth/',
  youtubeUrl: 'https://www.youtube.com/watch?v=fMkOQYMx1p0',
};
