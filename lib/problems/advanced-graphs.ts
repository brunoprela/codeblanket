import { Problem } from '../types';

export const advancedGraphsProblems: Problem[] = [
  {
    id: 'network-delay-time',
    title: 'Network Delay Time',
    difficulty: 'Easy',
    description: `You are given a network of \`n\` nodes, labeled from \`1\` to \`n\`. You are also given \`times\`, a list of travel times as directed edges \`times[i] = (ui, vi, wi)\`, where \`ui\` is the source node, \`vi\` is the target node, and \`wi\` is the time it takes for a signal to travel from source to target.

We will send a signal from a given node \`k\`. Return **the minimum time** it takes for all the \`n\` nodes to receive the signal. If it is impossible for all nodes to receive the signal, return \`-1\`.

**LeetCode:** [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)
**YouTube:** [NeetCode - Network Delay Time](https://www.youtube.com/watch?v=EaphyqKU4PQ)

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
    order: 1,
    topic: 'Advanced Graphs',
    leetcodeUrl: 'https://leetcode.com/problems/network-delay-time/',
    youtubeUrl: 'https://www.youtube.com/watch?v=EaphyqKU4PQ',
  },
  {
    id: 'cheapest-flights-within-k-stops',
    title: 'Cheapest Flights Within K Stops',
    difficulty: 'Medium',
    description: `There are \`n\` cities connected by flights. You are given an array \`flights\` where \`flights[i] = [fromi, toi, pricei]\` indicates a flight from city \`fromi\` to city \`toi\` with cost \`pricei\`.

You are also given three integers \`src\`, \`dst\`, and \`k\`, return **the cheapest price** from \`src\` to \`dst\` with at most \`k\` stops. If there is no such route, return \`-1\`.

**LeetCode:** [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)
**YouTube:** [NeetCode - Cheapest Flights](https://www.youtube.com/watch?v=5eIK3zUdYmE)

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
    order: 2,
    topic: 'Advanced Graphs',
    leetcodeUrl:
      'https://leetcode.com/problems/cheapest-flights-within-k-stops/',
    youtubeUrl: 'https://www.youtube.com/watch?v=5eIK3zUdYmE',
  },
  {
    id: 'path-with-minimum-effort',
    title: 'Path With Minimum Effort',
    difficulty: 'Hard',
    description: `You are a hiker preparing for an upcoming hike. You are given a 2D \`heights\` array where \`heights[row][col]\` represents the height of cell \`(row, col)\`. You start at \`(0, 0)\` and want to reach \`(rows-1, cols-1)\`.

A route's **effort** is the maximum absolute difference in heights between two consecutive cells.

Return **the minimum effort** required to travel from the top-left to the bottom-right cell.

**LeetCode:** [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)
**YouTube:** [NeetCode - Path With Minimum Effort](https://www.youtube.com/watch?v=XQlxCCx2vI4)

**Approach:**
Modified **Dijkstra's** where instead of summing distances, we track the maximum difference encountered on the path. Use min-heap prioritized by maximum effort so far.

**Key Insight:**
This is a "minimax" path problem - minimize the maximum edge weight on the path. Dijkstra works by always exploring the path with smallest maximum effort first.`,
    examples: [
      {
        input: 'heights = [[1,2,2],[3,8,2],[5,3,5]]',
        output: '2',
        explanation:
          'Path [1,3,5,3,5] has maximum difference 2 (from 3 to 5 or 5 to 3).',
      },
      {
        input: 'heights = [[1,2,3],[3,8,4],[5,3,5]]',
        output: '1',
        explanation: 'Path [1,2,3,4,5] has maximum difference 1.',
      },
    ],
    constraints: [
      'rows == heights.length',
      'columns == heights[i].length',
      '1 <= rows, columns <= 100',
      '1 <= heights[i][j] <= 10^6',
    ],
    hints: [
      "Use Dijkstra's but track maximum difference, not sum",
      'Priority queue: (max_effort_so_far, row, col)',
      'For each neighbor, effort = max(current_effort, abs(heights[r][c] - heights[nr][nc]))',
      'Update neighbor if we found a path with smaller maximum effort',
      'Can also use binary search + BFS on effort threshold',
    ],
    starterCode: `from typing import List

def minimum_effort_path(heights: List[List[int]]) -> int:
    """
    Find path with minimum maximum effort.
    
    Args:
        heights: 2D grid of heights
        
    Returns:
        Minimum effort required
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 2, 2],
            [3, 8, 2],
            [5, 3, 5],
          ],
        ],
        expected: 2,
      },
      {
        input: [
          [
            [1, 2, 3],
            [3, 8, 4],
            [5, 3, 5],
          ],
        ],
        expected: 1,
      },
      {
        input: [
          [
            [1, 2],
            [3, 8],
          ],
        ],
        expected: 2,
      },
    ],
    solution: `from typing import List
import heapq


def minimum_effort_path(heights: List[List[int]]) -> int:
    """
    Modified Dijkstra for minimax path.
    Time: O(m*n*log(m*n)), Space: O(m*n)
    """
    rows, cols = len(heights), len(heights[0])
    
    # Track minimum effort to reach each cell
    efforts = [[float('inf')] * cols for _ in range(rows)]
    efforts[0][0] = 0
    
    # Min-heap: (max_effort, row, col)
    pq = [(0, 0, 0)]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while pq:
        effort, r, c = heapq.heappop(pq)
        
        # Reached destination
        if r == rows - 1 and c == cols - 1:
            return effort
        
        # Skip if we've found better path
        if effort > efforts[r][c]:
            continue
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                # Effort is max difference on path
                new_effort = max(effort, abs(heights[r][c] - heights[nr][nc]))
                
                if new_effort < efforts[nr][nc]:
                    efforts[nr][nc] = new_effort
                    heapq.heappush(pq, (new_effort, nr, nc))
    
    return efforts[rows-1][cols-1]


# Alternative: Binary search + BFS
def minimum_effort_path_binary_search(heights: List[List[int]]) -> int:
    """
    Binary search on effort threshold.
    Time: O(m*n*log(max_height)), Space: O(m*n)
    """
    from collections import deque
    
    rows, cols = len(heights), len(heights[0])
    
    def can_reach(max_effort):
        """Check if destination reachable with effort <= max_effort."""
        visited = set([(0, 0)])
        queue = deque([(0, 0)])
        
        while queue:
            r, c = queue.popleft()
            
            if r == rows - 1 and c == cols - 1:
                return True
            
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    (nr, nc) not in visited):
                    diff = abs(heights[r][c] - heights[nr][nc])
                    if diff <= max_effort:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        return False
    
    # Binary search on effort
    left, right = 0, max(max(row) for row in heights)
    
    while left < right:
        mid = (left + right) // 2
        if can_reach(mid):
            right = mid
        else:
            left = mid + 1
    
    return left`,
    timeComplexity: 'O(m*n*log(m*n)) with Dijkstra',
    spaceComplexity: 'O(m*n)',
    order: 3,
    topic: 'Advanced Graphs',
    leetcodeUrl: 'https://leetcode.com/problems/path-with-minimum-effort/',
    youtubeUrl: 'https://www.youtube.com/watch?v=XQlxCCx2vI4',
  },
];
