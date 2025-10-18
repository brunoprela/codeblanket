/**
 * Critical Connections in a Network
 * Problem ID: critical-connections
 * Order: 9
 */

import { Problem } from '../../../types';

export const critical_connectionsProblem: Problem = {
  id: 'critical-connections',
  title: 'Critical Connections in a Network',
  difficulty: 'Medium',
  topic: 'Advanced Graphs',
  description: `There are \`n\` servers numbered from \`0\` to \`n - 1\` connected by undirected server-to-server \`connections\` forming a network where \`connections[i] = [ai, bi]\` represents a connection between servers \`ai\` and \`bi\`. Any server can reach other servers directly or indirectly through the network.

A **critical connection** is a connection that, if removed, will make some servers unable to reach some other server.

Return all critical connections in the network in any order.`,
  examples: [
    {
      input: 'n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]',
      output: '[[1,3]]',
    },
    {
      input: 'n = 2, connections = [[0,1]]',
      output: '[[0,1]]',
    },
  ],
  constraints: [
    '2 <= n <= 10^5',
    'n - 1 <= connections.length <= 10^5',
    '0 <= ai, bi <= n - 1',
    'ai != bi',
    'There are no repeated connections',
  ],
  hints: [
    'Use Tarjan algorithm for finding bridges',
    'Track discovery time and low-link values',
    'Bridge exists if low[child] > disc[parent]',
  ],
  starterCode: `from typing import List

def critical_connections(n: int, connections: List[List[int]]) -> List[List[int]]:
    """
    Find all bridges (critical edges) using Tarjan.
    
    Args:
        n: Number of nodes
        connections: List of edges
        
    Returns:
        List of critical connections
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        4,
        [
          [0, 1],
          [1, 2],
          [2, 0],
          [1, 3],
        ],
      ],
      expected: [[1, 3]],
    },
    {
      input: [2, [[0, 1]]],
      expected: [[0, 1]],
    },
  ],
  timeComplexity: 'O(V + E)',
  spaceComplexity: 'O(V + E)',
  leetcodeUrl:
    'https://leetcode.com/problems/critical-connections-in-a-network/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Rhxs4k6DyMM',
};
