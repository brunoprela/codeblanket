/**
 * All Paths From Source to Target
 * Problem ID: all-paths-source-target
 * Order: 7
 */

import { Problem } from '../../../types';

export const all_paths_source_targetProblem: Problem = {
  id: 'all-paths-source-target',
  title: 'All Paths From Source to Target',
  difficulty: 'Medium',
  topic: 'Graphs',
  description: `Given a directed acyclic graph (**DAG**) of \`n\` nodes labeled from \`0\` to \`n - 1\`, find all possible paths from node \`0\` to node \`n - 1\` and return them in **any order**.

The graph is given as follows: \`graph[i]\` is a list of all nodes you can visit from node \`i\` (i.e., there is a directed edge from node \`i\` to node \`graph[i][j]\`).`,
  examples: [
    {
      input: 'graph = [[1,2],[3],[3],[]]',
      output: '[[0,1,3],[0,2,3]]',
      explanation: 'There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.',
    },
    {
      input: 'graph = [[4,3,1],[3,2,4],[3],[4],[]]',
      output: '[[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]',
    },
  ],
  constraints: [
    'n == graph.length',
    '2 <= n <= 15',
    '0 <= graph[i][j] < n',
    'graph[i][j] != i (no self-loops)',
    'All the elements of graph[i] are unique',
    'The input graph is guaranteed to be a DAG',
  ],
  hints: [
    'Use DFS backtracking',
    'Track current path',
    'Add path to result when reaching target',
  ],
  starterCode: `from typing import List

def all_paths_source_target(graph: List[List[int]]) -> List[List[int]]:
    """
    Find all paths from source (0) to target (n-1).
    
    Args:
        graph: Adjacency list representation
        
    Returns:
        List of all paths
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[1, 2], [3], [3], []]],
      expected: [
        [0, 1, 3],
        [0, 2, 3],
      ],
    },
    {
      input: [[[4, 3, 1], [3, 2, 4], [3], [4], []]],
      expected: [
        [0, 4],
        [0, 3, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 4],
      ],
    },
  ],
  timeComplexity: 'O(2^N * N)',
  spaceComplexity: 'O(N)',
  leetcodeUrl: 'https://leetcode.com/problems/all-paths-from-source-to-target/',
  youtubeUrl: 'https://www.youtube.com/watch?v=bSfxLRBXQPU',
};
