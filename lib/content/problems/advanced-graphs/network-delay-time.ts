/**
 * Network Delay Time
 * Problem ID: network-delay-time
 * Order: 1
 */

import { Problem } from '../../../types';

export const network_delay_timeProblem: Problem = {
  id: 'network-delay-time',
  title: 'Network Delay Time',
  difficulty: 'Easy',
  description: `You are given a network of \`n\` nodes, labeled from \`1\` to \`n\`. You are also given \`times\`, a list of travel times as directed edges \`times[i] = (ui, vi, wi)\`, where \`ui\` is the source node, \`vi\` is the target node, and \`wi\` is the time it takes for a signal to travel from source to target.

We will send a signal from a given node \`k\`. Return **the minimum time** it takes for all the \`n\` nodes to receive the signal. If it is impossible for all nodes to receive the signal, return \`-1\`.


**Approach:**
This is a **single-source shortest path** problem with non-negative weights. Use **Dijkstra's algorithm** to find shortest paths from \`k\` to all nodes. The answer is the maximum of all shortest paths (time for signal to reach furthest node).

**Key Insight:**
Dijkstra with min-heap finds shortest paths efficiently. The time for all nodes to receive signal = time for slowest (furthest) node.`,
  examples: [
    {
      input: 'times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2',
      output: '2',
      explanation:
        'Signal sent from node 2. Node 1 receives at time 1, node 3 at time 1, node 4 at time 2. All nodes receive by time 2.',
    },
    {
      input: 'times = [[1,2,1]], n = 2, k = 1',
      output: '1',
      explanation: 'Signal reaches node 2 in 1 unit of time.',
    },
    {
      input: 'times = [[1,2,1]], n = 2, k = 2',
      output: '-1',
      explanation: 'Node 1 is not reachable from node 2.',
    },
  ],
  constraints: [
    '1 <= k <= n <= 100',
    '1 <= times.length <= 6000',
    'times[i].length == 3',
    '1 <= ui, vi <= n',
    'ui != vi',
    '0 <= wi <= 100',
    'All pairs (ui, vi) are unique',
  ],
  hints: [
    "Use Dijkstra's algorithm for shortest path from k to all nodes",
    'Build adjacency list from times array',
    'Use min-heap with (distance, node) tuples',
    'Track visited nodes to avoid reprocessing',
    'The answer is max(all shortest distances)',
    'If any node unreachable, return -1',
  ],
  starterCode: `from typing import List

def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """
    Find minimum time for signal to reach all nodes.
    
    Args:
        times: Directed edges [from, to, time]
        n: Number of nodes (1 to n)
        k: Starting node
        
    Returns:
        Minimum time for all nodes to receive signal, or -1
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [2, 1, 1],
          [2, 3, 1],
          [3, 4, 1],
        ],
        4,
        2,
      ],
      expected: 2,
    },
    {
      input: [[[1, 2, 1]], 2, 1],
      expected: 1,
    },
    {
      input: [[[1, 2, 1]], 2, 2],
      expected: -1,
    },
  ],
  solution: `from typing import List
import heapq
from collections import defaultdict


def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """
    Dijkstra's algorithm.
    Time: O((E + V) log V), Space: O(E + V)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Dijkstra's
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0
    
    pq = [(0, k)]  # (distance, node)
    visited = set()
    
    while pq:
        curr_dist, node = heapq.heappop(pq)
        
        if node in visited:
            continue
        visited.add(node)
        
        # Relax edges
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    # Maximum distance (last node to receive signal)
    max_dist = max(distances.values())
    return max_dist if max_dist != float('inf') else -1


# Alternative: Bellman-Ford (works but slower)
def network_delay_time_bf(times: List[List[int]], n: int, k: int) -> int:
    """
    Bellman-Ford approach.
    Time: O(V * E), Space: O(V)
    """
    distances = [float('inf')] * (n + 1)
    distances[k] = 0
    
    # Relax n-1 times
    for _ in range(n - 1):
        for u, v, w in times:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
    
    max_dist = max(distances[1:])
    return max_dist if max_dist != float('inf') else -1`,
  timeComplexity: 'O((E + V) log V) with Dijkstra',
  spaceComplexity: 'O(E + V)',

  leetcodeUrl: 'https://leetcode.com/problems/network-delay-time/',
  youtubeUrl: 'https://www.youtube.com/watch?v=EaphyqKU4PQ',
  order: 1,
  topic: 'Advanced Graphs',
};
