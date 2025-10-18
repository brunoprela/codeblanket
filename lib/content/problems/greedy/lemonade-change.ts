/**
 * Lemonade Change
 * Problem ID: lemonade-change
 * Order: 5
 */

import { Problem } from '../../../types';

export const lemonade_changeProblem: Problem = {
  id: 'lemonade-change',
  title: 'Lemonade Change',
  difficulty: 'Easy',
  topic: 'Greedy',
  description: `At a lemonade stand, each lemonade costs \`$5\`. Customers are standing in a queue to buy from you and order one at a time (in the order specified by bills). Each customer will only buy one lemonade and pay with either a \`$5\`, \`$10\`, or \`$20\` bill. You must provide the correct change to each customer so that the net transaction is that the customer pays \`$5\`.

Note that you do not have any change in hand at first.

Given an integer array \`bills\` where \`bills[i]\` is the bill the \`i-th\` customer pays, return \`true\` if you can provide every customer with the correct change, or \`false\` otherwise.`,
  examples: [
    {
      input: 'bills = [5,5,5,10,20]',
      output: 'true',
    },
    {
      input: 'bills = [5,5,10,10,20]',
      output: 'false',
    },
  ],
  constraints: ['1 <= bills.length <= 10^5', 'bills[i] is either 5, 10, or 20'],
  hints: [
    'Track count of $5 and $10 bills',
    'For $10: need one $5',
    'For $20: prefer three $5s or one $10 + one $5',
  ],
  starterCode: `from typing import List

def lemonade_change(bills: List[int]) -> bool:
    """
    Check if can provide correct change.
    
    Args:
        bills: Customer bills in order
        
    Returns:
        True if can provide change to all
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[5, 5, 5, 10, 20]],
      expected: true,
    },
    {
      input: [[5, 5, 10, 10, 20]],
      expected: false,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/lemonade-change/',
  youtubeUrl: 'https://www.youtube.com/watch?v=6rF0xNFOLbk',
};
