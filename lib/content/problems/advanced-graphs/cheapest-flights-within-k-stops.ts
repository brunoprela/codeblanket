/**
 * Cheapest Flights Within K Stops
 * Problem ID: cheapest-flights-within-k-stops
 * Order: 2
 */

import { Problem } from '../../../types';

export const cheapest_flights_within_k_stopsProblem: Problem = {
  id: 'cheapest-flights-within-k-stops',
  title: 'Cheapest Flights Within K Stops',
  difficulty: 'Medium',
  description: `There are \`n\` cities connected by flights. You are given an array \`flights\` where \`flights[i] = [fromi, toi, pricei]\` indicates a flight from city \`fromi\` to city \`toi\` with cost \`pricei\`.

You are also given three integers \`src\`, \`dst\`, and \`k\`, return **the cheapest price** from \`src\` to \`dst\` with at most \`k\` stops. If there is no such route, return \`-1\`.


**Approach:**
Modified **Bellman-Ford** or **BFS with price tracking**. The constraint is number of stops (edges), not just shortest path. Cannot use standard Dijkstra because it might use more stops.

**Key Insight:**
Relax edges exactly k+1 times (k stops = k+1 flights). Use DP: \`dp[i][v]\` = min cost to reach v using at most i edges.`,
  examples: [
    {
      input:
        'n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1',
      output: '700',
      explanation:
        'Optimal path with at most 1 stop: 0 → 1 → 3, cost = 100 + 600 = 700.',
    },
    {
      input:
        'n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1',
      output: '200',
      explanation: 'Path with 1 stop: 0 → 1 → 2, cost = 200.',
    },
  ],
  constraints: [
    '1 <= n <= 100',
    '0 <= flights.length <= (n * (n - 1) / 2)',
    'flights[i].length == 3',
    '0 <= fromi, toi < n',
    'fromi != toi',
    '1 <= pricei <= 10^4',
    '0 <= src, dst, k < n',
    'src != dst',
  ],
  hints: [
    'Modified Bellman-Ford: relax edges k+1 times',
    'Use DP: dp[stops][city] = min cost to reach city with stops edges',
    'At each iteration, try all flights',
    'Keep previous iteration values to avoid using updated values in same iteration',
    'BFS alternative: track (node, cost, stops) in queue',
  ],
  starterCode: `from typing import List

def find_cheapest_price(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    Find cheapest flight with at most k stops.
    
    Args:
        n: Number of cities
        flights: [from, to, price] edges
        src: Source city
        dst: Destination city
        k: Maximum number of stops
        
    Returns:
        Cheapest price, or -1 if impossible
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        4,
        [
          [0, 1, 100],
          [1, 2, 100],
          [2, 0, 100],
          [1, 3, 600],
          [2, 3, 200],
        ],
        0,
        3,
        1,
      ],
      expected: 700,
    },
    {
      input: [
        3,
        [
          [0, 1, 100],
          [1, 2, 100],
          [0, 2, 500],
        ],
        0,
        2,
        1,
      ],
      expected: 200,
    },
    {
      input: [
        3,
        [
          [0, 1, 100],
          [1, 2, 100],
          [0, 2, 500],
        ],
        0,
        2,
        0,
      ],
      expected: 500,
    },
  ],
  solution: `from typing import List
from collections import deque, defaultdict


def find_cheapest_price(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    Bellman-Ford with k+1 relaxations.
    Time: O(k * E), Space: O(n)
    """
    # Initialize prices
    prices = [float('inf')] * n
    prices[src] = 0
    
    # Relax edges k+1 times (k stops = k+1 flights)
    for _ in range(k + 1):
        temp_prices = prices.copy()
        
        for u, v, price in flights:
            if prices[u] != float('inf'):
                temp_prices[v] = min(temp_prices[v], prices[u] + price)
        
        prices = temp_prices
    
    return prices[dst] if prices[dst] != float('inf') else -1


# Alternative: BFS with stops tracking
def find_cheapest_price_bfs(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    BFS tracking stops.
    Time: O(k * E), Space: O(n + E)
    """
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))
    
    # (node, cost, stops)
    queue = deque([(src, 0, 0)])
    min_cost = {src: 0}
    
    while queue:
        node, cost, stops = queue.popleft()
        
        if stops > k:
            continue
        
        for neighbor, price in graph[node]:
            new_cost = cost + price
            
            # Only add if better cost or fewer stops
            if neighbor not in min_cost or new_cost < min_cost[neighbor]:
                min_cost[neighbor] = new_cost
                queue.append((neighbor, new_cost, stops + 1))
    
    return min_cost.get(dst, -1)


# Alternative: Dijkstra with stops
def find_cheapest_price_dijkstra(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    """
    Modified Dijkstra tracking stops.
    """
    import heapq
    
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))
    
    # (cost, node, stops)
    pq = [(0, src, 0)]
    visited = {}
    
    while pq:
        cost, node, stops = heapq.heappop(pq)
        
        if node == dst:
            return cost
        
        if stops > k:
            continue
        
        if node in visited and visited[node] <= stops:
            continue
        visited[node] = stops
        
        for neighbor, price in graph[node]:
            heapq.heappush(pq, (cost + price, neighbor, stops + 1))
    
    return -1`,
  timeComplexity: 'O(k * E) with Bellman-Ford',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/cheapest-flights-within-k-stops/',
  youtubeUrl: 'https://www.youtube.com/watch?v=5eIK3zUdYmE',
  order: 2,
  topic: 'Advanced Graphs',
};
