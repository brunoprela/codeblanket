/**
 * Find if Path Exists in Graph
 * Problem ID: find-path-exists-graph
 * Order: 4
 */

import { Problem } from '../../../types';

export const find_path_exists_graphProblem: Problem = {
  id: 'find-path-exists-graph',
  title: 'Find if Path Exists in Graph',
  difficulty: 'Easy',
  topic: 'Graphs',
  description: `There is a **bi-directional** graph with \`n\` vertices, where each vertex is labeled from \`0\` to \`n - 1\` (**inclusive**). The edges in the graph are represented as a 2D integer array \`edges\`, where each \`edges[i] = [ui, vi]\` denotes a bi-directional edge between vertex \`ui\` and vertex \`vi\`. Every vertex pair is connected by **at most one** edge, and no vertex has an edge to itself.

You want to determine if there is a **valid path** that exists from vertex \`source\` to vertex \`destination\`.

Given \`edges\` and the integers \`n\`, \`source\`, and \`destination\`, return \`true\` if there is a **valid path** from \`source\` to \`destination\`, or \`false\` otherwise.`,
  examples: [
    {
      input: 'n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2',
      output: 'true',
      explanation:
        'There are two paths from vertex 0 to vertex 2: 0 → 1 → 2 and 0 → 2',
    },
    {
      input:
        'n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], source = 0, destination = 5',
      output: 'false',
      explanation: 'There is no path from vertex 0 to vertex 5.',
    },
  ],
  constraints: [
    '1 <= n <= 2 * 10^5',
    '0 <= edges.length <= 2 * 10^5',
    'edges[i].length == 2',
    '0 <= ui, vi <= n - 1',
    'ui != vi',
    '0 <= source, destination <= n - 1',
    'There are no duplicate edges',
    'There are no self edges',
  ],
  hints: ['Build adjacency list from edges', 'Use BFS or DFS to find path'],
  starterCode: `from typing import List
from collections import deque, defaultdict

def valid_path(n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    """
    Check if path exists from source to destination.
    
    Args:
        n: Number of vertices
        edges: List of edges
        source: Starting vertex
        destination: Target vertex
        
    Returns:
        True if path exists
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        3,
        [
          [0, 1],
          [1, 2],
          [2, 0],
        ],
        0,
        2,
      ],
      expected: true,
    },
    {
      input: [
        6,
        [
          [0, 1],
          [0, 2],
          [3, 5],
          [5, 4],
          [4, 3],
        ],
        0,
        5,
      ],
      expected: false,
    },
    {
      input: [1, [], 0, 0],
      expected: true,
    },
  ],
  timeComplexity: 'O(V + E)',
  spaceComplexity: 'O(V + E)',
  leetcodeUrl: 'https://leetcode.com/problems/find-if-path-exists-in-graph/',
  youtubeUrl: 'https://www.youtube.com/watch?v=muncqlKJrH0',
};
