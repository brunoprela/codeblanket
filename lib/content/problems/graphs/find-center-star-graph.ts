/**
 * Find Center of Star Graph
 * Problem ID: find-center-star-graph
 * Order: 5
 */

import { Problem } from '../../../types';

export const find_center_star_graphProblem: Problem = {
  id: 'find-center-star-graph',
  title: 'Find Center of Star Graph',
  difficulty: 'Easy',
  topic: 'Graphs',
  description: `There is an undirected **star** graph consisting of \`n\` nodes labeled from \`1\` to \`n\`. A star graph is a graph where there is one **center** node and **exactly** \`n - 1\` edges that connect the center node with every other node.

You are given a 2D integer array \`edges\` where each \`edges[i] = [ui, vi]\` indicates that there is an edge between the nodes \`ui\` and \`vi\`. Return the center of the given star graph.`,
  examples: [
    {
      input: 'edges = [[1,2],[2,3],[4,2]]',
      output: '2',
      explanation:
        'Node 2 is connected to every other node, so 2 is the center.',
    },
    {
      input: 'edges = [[1,2],[5,1],[1,3],[1,4]]',
      output: '1',
    },
  ],
  constraints: [
    '3 <= n <= 10^5',
    'edges.length == n - 1',
    'edges[i].length == 2',
    '1 <= ui, vi <= n',
    'ui != vi',
    'The given edges represent a valid star graph',
  ],
  hints: ['The center appears in every edge', 'Check first two edges'],
  starterCode: `from typing import List

def find_center(edges: List[List[int]]) -> int:
    """
    Find center node of star graph.
    
    Args:
        edges: List of edges in star graph
        
    Returns:
        Center node number
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 2],
          [2, 3],
          [4, 2],
        ],
      ],
      expected: 2,
    },
    {
      input: [
        [
          [1, 2],
          [5, 1],
          [1, 3],
          [1, 4],
        ],
      ],
      expected: 1,
    },
  ],
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/find-center-of-star-graph/',
  youtubeUrl: 'https://www.youtube.com/watch?v=KFVweOZyY0I',
};
