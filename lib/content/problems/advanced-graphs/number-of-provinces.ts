/**
 * Number of Provinces
 * Problem ID: number-of-provinces
 * Order: 6
 */

import { Problem } from '../../../types';

export const number_of_provincesProblem: Problem = {
  id: 'number-of-provinces',
  title: 'Number of Provinces',
  difficulty: 'Easy',
  topic: 'Advanced Graphs',
  description: `There are \`n\` cities. Some of them are connected, while some are not. If city \`a\` is connected directly with city \`b\`, and city \`b\` is connected directly with city \`c\`, then city \`a\` is connected indirectly with city \`c\`.

A **province** is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an \`n x n\` matrix \`isConnected\` where \`isConnected[i][j] = 1\` if the \`i-th\` city and the \`j-th\` city are directly connected, and \`isConnected[i][j] = 0\` otherwise.

Return the total number of **provinces**.`,
  examples: [
    {
      input: 'isConnected = [[1,1,0],[1,1,0],[0,0,1]]',
      output: '2',
    },
    {
      input: 'isConnected = [[1,0,0],[0,1,0],[0,0,1]]',
      output: '3',
    },
  ],
  constraints: [
    '1 <= n <= 200',
    'n == isConnected.length',
    'n == isConnected[i].length',
    'isConnected[i][j] is 1 or 0',
    'isConnected[i][i] == 1',
    'isConnected[i][j] == isConnected[j][i]',
  ],
  hints: ['Use Union-Find or DFS', 'Count number of connected components'],
  starterCode: `from typing import List

def find_circle_num(is_connected: List[List[int]]) -> int:
    """
    Count number of provinces (connected components).
    
    Args:
        is_connected: Adjacency matrix
        
    Returns:
        Number of provinces
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 1, 0],
          [1, 1, 0],
          [0, 0, 1],
        ],
      ],
      expected: 2,
    },
    {
      input: [
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      ],
      expected: 3,
    },
  ],
  timeComplexity: 'O(n^2)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/number-of-provinces/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ZGr5nX-Gi6Y',
};
