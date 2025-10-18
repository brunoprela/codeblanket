/**
 * Clone Graph
 * Problem ID: clone-graph
 * Order: 3
 */

import { Problem } from '../../../types';

export const clone_graphProblem: Problem = {
  id: 'clone-graph',
  title: 'Clone Graph',
  difficulty: 'Hard',
  description: `Given a reference of a node in a **connected** undirected graph, return a **deep copy** (clone) of the graph.

Each node in the graph contains a value (\`int\`) and a list (\`List[Node]\`) of its neighbors.

\`\`\`
class Node {
    public int val;
    public List<Node> neighbors;
}
\`\`\`


**Approach:**
Use DFS or BFS to traverse the graph. Maintain a hash map to track original → clone mappings to avoid creating duplicate clones and to handle cycles.

**Key Challenge:**
Handling the graph's connections correctly - must clone both nodes AND their relationships.`,
  examples: [
    {
      input: 'adjList = [[2,4],[1,3],[2,4],[1,3]]',
      output: '[[2,4],[1,3],[2,4],[1,3]]',
      explanation:
        'There are 4 nodes in the graph. Node 1 connects to nodes 2 and 4. Node 2 connects to nodes 1 and 3. Node 3 connects to nodes 2 and 4. Node 4 connects to nodes 1 and 3.',
    },
    {
      input: 'adjList = [[]]',
      output: '[[]]',
      explanation: 'The input contains a single node with no neighbors.',
    },
    {
      input: 'adjList = []',
      output: '[]',
      explanation: 'Empty graph.',
    },
  ],
  constraints: [
    'The number of nodes in the graph is in the range [0, 100]',
    '1 <= Node.val <= 100',
    'Node.val is unique for each node',
    'There are no repeated edges and no self-loops',
    'The Graph is connected and all nodes can be visited starting from the given node',
  ],
  hints: [
    'Use a hash map to store original node → cloned node mappings',
    'This prevents creating duplicate clones and handles cycles',
    'DFS/BFS through the graph, cloning each node and its neighbors',
    'When you encounter a node already in the map, return the clone',
    'Time: O(N + E) to visit all nodes and edges',
  ],
  starterCode: `from typing import Optional

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: Optional[Node]) -> Optional[Node]:
    """
    Clone a graph using deep copy.
    
    Args:
        node: Reference node in the original graph
        
    Returns:
        Reference to the cloned graph
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [
            [2, 4],
            [1, 3],
            [2, 4],
            [1, 3],
          ],
        ],
      ],
      expected: [
        [2, 4],
        [1, 3],
        [2, 4],
        [1, 3],
      ],
    },
    {
      input: [[[[]]]],
      expected: [[]],
    },
    {
      input: [[[]]],
      expected: [],
    },
  ],
  solution: `from typing import Optional

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node: Optional[Node]) -> Optional[Node]:
    """
    DFS approach with hash map.
    Time: O(N + E), Space: O(N)
    """
    if not node:
        return None
    
    # Map original node -> cloned node
    clones = {}
    
    def dfs(node):
        # If already cloned, return the clone
        if node in clones:
            return clones[node]
        
        # Create clone
        clone = Node(node.val)
        clones[node] = clone
        
        # Clone all neighbors
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)


# Alternative: BFS approach
def clone_graph_bfs(node: Optional[Node]) -> Optional[Node]:
    """
    BFS approach with hash map.
    Time: O(N + E), Space: O(N)
    """
    if not node:
        return None
    
    from collections import deque
    
    # Map original -> clone
    clones = {node: Node(node.val)}
    queue = deque([node])
    
    while queue:
        curr = queue.popleft()
        
        # Process all neighbors
        for neighbor in curr.neighbors:
            if neighbor not in clones:
                # Create clone
                clones[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            
            # Add neighbor to current node's clone
            clones[curr].neighbors.append(clones[neighbor])
    
    return clones[node]


# Alternative: One-liner using dictionary
def clone_graph_compact(node: Optional[Node]) -> Optional[Node]:
    """
    Compact recursive solution.
    """
    if not node:
        return None
    
    clones = {}
    
    def dfs(n):
        if n not in clones:
            clones[n] = Node(n.val)
            clones[n].neighbors = [dfs(nb) for nb in n.neighbors]
        return clones[n]
    
    return dfs(node)`,
  timeComplexity: 'O(N + E) where N = nodes, E = edges',
  spaceComplexity: 'O(N) for the hash map',

  leetcodeUrl: 'https://leetcode.com/problems/clone-graph/',
  youtubeUrl: 'https://www.youtube.com/watch?v=mQeF6bN8hMk',
  order: 3,
  topic: 'Graphs',
};
