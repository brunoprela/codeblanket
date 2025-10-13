import { Problem } from '../types';

export const advancedGraphsProblems: Problem[] = [
  {
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

  // EASY - Find if Path Exists (Union-Find)
  {
    id: 'redundant-connection',
    title: 'Redundant Connection',
    difficulty: 'Easy',
    topic: 'Advanced Graphs',
    description: `In this problem, a tree is an **undirected graph** that is connected and has no cycles.

You are given a graph that started as a tree with \`n\` nodes labeled from \`1\` to \`n\`, with one additional edge added. The added edge has two **different** vertices chosen from \`1\` to \`n\`, and was not an edge that already existed. The graph is represented as an array \`edges\` of length \`n\` where \`edges[i] = [ai, bi]\` indicates that there is an edge between nodes \`ai\` and \`bi\` in the graph.

Return an edge that can be removed so that the resulting graph is a tree of \`n\` nodes. If there are multiple answers, return the answer that occurs last in the input.`,
    examples: [
      {
        input: 'edges = [[1,2],[1,3],[2,3]]',
        output: '[2,3]',
      },
      {
        input: 'edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]',
        output: '[1,4]',
      },
    ],
    constraints: [
      'n == edges.length',
      '3 <= n <= 1000',
      'edges[i].length == 2',
      '1 <= ai < bi <= edges.length',
      'ai != bi',
      'There are no repeated edges',
      'The given graph is connected',
    ],
    hints: [
      'Use Union-Find (DSU)',
      'If both nodes already in same component, edge creates cycle',
    ],
    starterCode: `from typing import List

def find_redundant_connection(edges: List[List[int]]) -> List[int]:
    """
    Find redundant edge using Union-Find.
    
    Args:
        edges: List of edges
        
    Returns:
        Edge that creates cycle
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 2],
            [1, 3],
            [2, 3],
          ],
        ],
        expected: [2, 3],
      },
      {
        input: [
          [
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 4],
            [1, 5],
          ],
        ],
        expected: [1, 4],
      },
    ],
    timeComplexity: 'O(n * α(n)) where α is inverse Ackermann',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/redundant-connection/',
    youtubeUrl: 'https://www.youtube.com/watch?v=FXWRE67PLL0',
  },

  // EASY - Accounts Merge
  {
    id: 'accounts-merge',
    title: 'Accounts Merge',
    difficulty: 'Easy',
    topic: 'Advanced Graphs',
    description: `Given a list of \`accounts\` where each element \`accounts[i]\` is a list of strings, where the first element \`accounts[i][0]\` is a name, and the rest of the elements are **emails** representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails **in sorted order**. The accounts themselves can be returned in **any order**.`,
    examples: [
      {
        input:
          'accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]',
        output:
          '[["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]',
      },
    ],
    constraints: [
      '1 <= accounts.length <= 1000',
      '2 <= accounts[i].length <= 10',
      '1 <= accounts[i][j].length <= 30',
      'accounts[i][0] consists of English letters',
      'accounts[i][j] (for j > 0) is a valid email',
    ],
    hints: [
      'Use Union-Find on emails',
      'Build mapping from email to name',
      'Group emails by root parent',
    ],
    starterCode: `from typing import List

def accounts_merge(accounts: List[List[str]]) -> List[List[str]]:
    """
    Merge accounts with common emails.
    
    Args:
        accounts: List of [name, email1, email2, ...]
        
    Returns:
        Merged accounts
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            ['John', 'johnsmith@mail.com', 'john_newyork@mail.com'],
            ['John', 'johnsmith@mail.com', 'john00@mail.com'],
            ['Mary', 'mary@mail.com'],
            ['John', 'johnnybravo@mail.com'],
          ],
        ],
        expected: [
          [
            'John',
            'john00@mail.com',
            'john_newyork@mail.com',
            'johnsmith@mail.com',
          ],
          ['Mary', 'mary@mail.com'],
          ['John', 'johnnybravo@mail.com'],
        ],
      },
    ],
    timeComplexity: 'O(n * k * α(n)) where k is avg emails per account',
    spaceComplexity: 'O(n * k)',
    leetcodeUrl: 'https://leetcode.com/problems/accounts-merge/',
    youtubeUrl: 'https://www.youtube.com/watch?v=wU6udHRIkcc',
  },

  // EASY - Number of Provinces
  {
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
  },

  // MEDIUM - Minimum Spanning Tree (Kruskal)
  {
    id: 'min-cost-to-connect-all-points',
    title: 'Min Cost to Connect All Points',
    difficulty: 'Medium',
    topic: 'Advanced Graphs',
    description: `You are given an array \`points\` representing integer coordinates of some points on a 2D-plane, where \`points[i] = [xi, yi]\`.

The cost of connecting two points \`[xi, yi]\` and \`[xj, yj]\` is the **manhattan distance** between them: \`|xi - xj| + |yi - yj|\`, where \`|val|\` denotes the absolute value of \`val\`.

Return the minimum cost to make all points connected. All points are connected if there is **exactly one** simple path between any two points.`,
    examples: [
      {
        input: 'points = [[0,0],[2,2],[3,10],[5,2],[7,0]]',
        output: '20',
      },
      {
        input: 'points = [[3,12],[-2,5],[-4,1]]',
        output: '18',
      },
    ],
    constraints: [
      '1 <= points.length <= 1000',
      '-10^6 <= xi, yi <= 10^6',
      'All pairs (xi, yi) are distinct',
    ],
    hints: [
      'Build MST using Kruskal algorithm',
      'Sort all edges by weight',
      'Use Union-Find to detect cycles',
    ],
    starterCode: `from typing import List

def min_cost_connect_points(points: List[List[int]]) -> int:
    """
    Find MST cost using Kruskal algorithm.
    
    Args:
        points: Array of [x, y] coordinates
        
    Returns:
        Minimum spanning tree cost
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [0, 0],
            [2, 2],
            [3, 10],
            [5, 2],
            [7, 0],
          ],
        ],
        expected: 20,
      },
      {
        input: [
          [
            [3, 12],
            [-2, 5],
            [-4, 1],
          ],
        ],
        expected: 18,
      },
    ],
    timeComplexity: 'O(n^2 log n)',
    spaceComplexity: 'O(n^2)',
    leetcodeUrl:
      'https://leetcode.com/problems/min-cost-to-connect-all-points/',
    youtubeUrl: 'https://www.youtube.com/watch?v=f7JOBJIC-NA',
  },

  // MEDIUM - Swim in Rising Water
  {
    id: 'swim-in-rising-water',
    title: 'Swim in Rising Water',
    difficulty: 'Medium',
    topic: 'Advanced Graphs',
    description: `You are given an \`n x n\` integer matrix \`grid\` where each value \`grid[i][j]\` represents the elevation at that point \`(i, j)\`.

The rain starts to fall. At time \`t\`, the depth of the water everywhere is \`t\`. You can swim from a square to another 4-directionally adjacent square if and only if the elevation of both squares individually are at most \`t\`. You can swim infinite distances in zero time. Of course, you must stay within the boundaries of the grid during your swim.

Return the least time until you can reach the bottom right square \`(n - 1, n - 1)\` if you start at the top left square \`(0, 0)\`.`,
    examples: [
      {
        input: 'grid = [[0,2],[1,3]]',
        output: '3',
      },
      {
        input:
          'grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]',
        output: '16',
      },
    ],
    constraints: [
      'n == grid.length',
      'n == grid[i].length',
      '1 <= n <= 50',
      '0 <= grid[i][j] < n^2',
      'Each value grid[i][j] is unique',
    ],
    hints: [
      'Use Dijkstra or binary search + BFS',
      'Track minimum maximum elevation on path',
    ],
    starterCode: `from typing import List
import heapq

def swim_in_water(grid: List[List[int]]) -> int:
    """
    Find minimum time to reach bottom-right.
    
    Args:
        grid: Elevation matrix
        
    Returns:
        Minimum time needed
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [0, 2],
            [1, 3],
          ],
        ],
        expected: 3,
      },
      {
        input: [
          [
            [0, 1, 2, 3, 4],
            [24, 23, 22, 21, 5],
            [12, 13, 14, 15, 16],
            [11, 17, 18, 19, 20],
            [10, 9, 8, 7, 6],
          ],
        ],
        expected: 16,
      },
    ],
    timeComplexity: 'O(n^2 log n)',
    spaceComplexity: 'O(n^2)',
    leetcodeUrl: 'https://leetcode.com/problems/swim-in-rising-water/',
    youtubeUrl: 'https://www.youtube.com/watch?v=amvrKlMLuGY',
  },

  // MEDIUM - Critical Connections
  {
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
  },
];
