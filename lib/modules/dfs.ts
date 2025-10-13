import { Module } from '@/lib/types';

export const dfsModule: Module = {
  id: 'dfs',
  title: 'Depth-First Search (DFS)',
  description:
    'Master depth-first search for exploring trees and graphs by going as deep as possible before backtracking.',
  icon: '🌊',
  timeComplexity: 'O(V + E) for graphs, O(N) for trees',
  spaceComplexity: 'O(H) for recursion stack, where H is height',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to DFS',
      content: `**Depth-First Search (DFS)** is a fundamental algorithm for exploring trees and graphs. It explores as far as possible along each branch before backtracking.

**Core Concept:**
- Start at a node
- Explore one neighbor completely (go deep)
- Backtrack when you hit a dead end
- Continue until all nodes are visited

**Key Characteristics:**
- Uses **recursion** or an explicit **stack**
- Goes **deep** before going wide
- Natural fit for trees and recursive problems
- Can find paths but NOT shortest paths

**DFS vs BFS:**
- **DFS**: Stack/Recursion → Goes deep → O(H) space
- **BFS**: Queue → Goes wide → O(W) space (W = width)

**When to Use DFS:**
- Tree traversals (preorder, inorder, postorder)
- Finding paths in a graph/maze
- Topological sorting
- Detecting cycles
- Finding connected components
- Generating permutations/combinations
- Backtracking problems`,
    },
    {
      id: 'tree-dfs',
      title: 'DFS on Trees',
      content: `**Tree DFS** is the simplest form of DFS. Trees have no cycles, making recursion straightforward.

**Three Main Orders:**

**1. Preorder (Root → Left → Right)**
- Process root first
- Used for: copying trees, prefix expressions
\`\`\`python
def preorder(root):
    if not root:
        return
    print(root.val)        # Process root
    preorder(root.left)    # Left subtree
    preorder(root.right)   # Right subtree
\`\`\`

**2. Inorder (Left → Root → Right)**
- Process root in the middle
- Used for: BST sorted output, validation
\`\`\`python
def inorder(root):
    if not root:
        return
    inorder(root.left)     # Left subtree
    print(root.val)        # Process root
    inorder(root.right)    # Right subtree
\`\`\`

**3. Postorder (Left → Right → Root)**
- Process root last
- Used for: deleting trees, postfix expressions
\`\`\`python
def postorder(root):
    if not root:
        return
    postorder(root.left)   # Left subtree
    postorder(root.right)  # Right subtree
    print(root.val)        # Process root
\`\`\`

**Common Tree DFS Patterns:**

**Pattern 1: Calculate property for each node**
\`\`\`python
def max_depth(root):
    if not root:
        return 0
    left = max_depth(root.left)
    right = max_depth(root.right)
    return 1 + max(left, right)
\`\`\`

**Pattern 2: Path-based problems**
\`\`\`python
def has_path_sum(root, target):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target
    target -= root.val
    return (has_path_sum(root.left, target) or 
            has_path_sum(root.right, target))
\`\`\``,
      codeExample: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder_traversal(root: Optional[TreeNode]) -> list:
    """Preorder: Root -> Left -> Right"""
    result = []
    
    def dfs(node):
        if not node:
            return
        result.append(node.val)  # Process root
        dfs(node.left)           # Left subtree
        dfs(node.right)          # Right subtree
    
    dfs(root)
    return result


def inorder_traversal(root: Optional[TreeNode]) -> list:
    """Inorder: Left -> Root -> Right"""
    result = []
    
    def dfs(node):
        if not node:
            return
        dfs(node.left)           # Left subtree
        result.append(node.val)  # Process root
        dfs(node.right)          # Right subtree
    
    dfs(root)
    return result


def postorder_traversal(root: Optional[TreeNode]) -> list:
    """Postorder: Left -> Right -> Root"""
    result = []
    
    def dfs(node):
        if not node:
            return
        dfs(node.left)           # Left subtree
        dfs(node.right)          # Right subtree
        result.append(node.val)  # Process root
    
    dfs(root)
    return result


def max_depth(root: Optional[TreeNode]) -> int:
    """Calculate maximum depth using DFS"""
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)


def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """Check if two trees are identical"""
    if not p and not q:
        return True
    if not p or not q:
        return False
    
    return (p.val == q.val and 
            is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))`,
    },
    {
      id: 'graph-dfs',
      title: 'DFS on Graphs',
      content: `**Graph DFS** is more complex than tree DFS because graphs can have cycles. We need to track visited nodes.

**Key Differences from Tree DFS:**
- Must track **visited nodes** (avoid cycles)
- Can have multiple connected components
- May need to explore all nodes as starting points

**Adjacency List Representation:**
\`\`\`python
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}
\`\`\`

**Basic Graph DFS Template:**
\`\`\`python
def dfs(graph, start):
    visited = set()
    
    def explore(node):
        if node in visited:
            return
        
        visited.add(node)
        print(node)  # Process node
        
        for neighbor in graph[node]:
            explore(neighbor)
    
    explore(start)
\`\`\`

**Common Graph DFS Problems:**

**1. Connected Components**
\`\`\`python
def count_components(n, edges):
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    count = 0
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    for node in range(n):
        if node not in visited:
            dfs(node)
            count += 1
    
    return count
\`\`\`

**2. Cycle Detection**
\`\`\`python
def has_cycle(graph):
    visited = set()
    rec_stack = set()  # Nodes in current path
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True  # Back edge = cycle
        
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False
\`\`\`

**3. Path Finding**
\`\`\`python
def find_path(graph, start, end):
    visited = set()
    path = []
    
    def dfs(node):
        if node in visited:
            return False
        
        visited.add(node)
        path.append(node)
        
        if node == end:
            return True
        
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        
        path.pop()  # Backtrack
        return False
    
    dfs(start)
    return path if path and path[-1] == end else []
\`\`\``,
      codeExample: `from typing import List, Set
from collections import defaultdict


def dfs_iterative(graph: dict, start: int) -> List[int]:
    """Iterative DFS using explicit stack"""
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        result.append(node)
        
        # Add neighbors in reverse order for correct traversal
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return result


def dfs_recursive(graph: dict, start: int) -> List[int]:
    """Recursive DFS"""
    visited = set()
    result = []
    
    def explore(node):
        visited.add(node)
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                explore(neighbor)
    
    explore(start)
    return result


def find_all_paths(graph: dict, start: int, end: int) -> List[List[int]]:
    """Find all paths from start to end"""
    paths = []
    
    def dfs(node, path, visited):
        if node == end:
            paths.append(path[:])
            return
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    dfs(start, [start], {start})
    return paths


def is_connected(graph: dict, n: int) -> bool:
    """Check if graph is fully connected"""
    if not graph or n == 0:
        return True
    
    visited = set()
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(0)
    return len(visited) == n`,
    },
    {
      id: 'iterative-dfs',
      title: 'Iterative DFS with Stack',
      content: `While recursion is natural for DFS, you can implement it iteratively using an explicit stack. This is useful to avoid stack overflow for deep graphs.

**Why Use Iterative DFS:**
- Avoid recursion depth limits
- More control over the stack
- Can be more efficient in some cases

**Recursive vs Iterative:**

**Recursive:**
\`\`\`python
def dfs_recursive(node):
    if not node:
        return
    process(node)
    dfs_recursive(node.left)
    dfs_recursive(node.right)
\`\`\`

**Iterative:**
\`\`\`python
def dfs_iterative(root):
    if not root:
        return
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        process(node)
        
        # Push right first (so left is processed first)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
\`\`\`

**Iterative Preorder:**
\`\`\`python
def preorder_iterative(root):
    result = []
    stack = [root] if root else []
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
\`\`\`

**Iterative Inorder (trickier):**
\`\`\`python
def inorder_iterative(root):
    result = []
    stack = []
    curr = root
    
    while stack or curr:
        # Go to leftmost node
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Process node
        curr = stack.pop()
        result.append(curr.val)
        
        # Move to right subtree
        curr = curr.right
    
    return result
\`\`\`

**Iterative Postorder (most complex):**
\`\`\`python
def postorder_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push left first (reverse of preorder)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    return result[::-1]  # Reverse the result
\`\`\``,
    },
    {
      id: 'complexity',
      title: 'Time and Space Complexity',
      content: `**Time Complexity:**

**Tree DFS:**
- **O(N)** where N = number of nodes
- Visit each node exactly once
- Each edge traversed twice (down and back up)

**Graph DFS:**
- **O(V + E)** where V = vertices, E = edges
- Visit each vertex once: O(V)
- Explore each edge once (or twice for undirected): O(E)

**Space Complexity:**

**Recursive DFS:**
- **O(H)** where H = height/depth
- Call stack space
- Best case (balanced tree): O(log N)
- Worst case (skewed tree/graph): O(N)

**Iterative DFS:**
- **O(H)** for explicit stack
- Plus O(V) for visited set in graphs
- Total: O(V) for graphs

**Additional Space:**
- Visited set for graphs: O(V)
- Result array: O(N) or O(V)
- Path tracking: O(H) to O(V)

**Optimization Tips:**
- Use iterative for very deep structures
- Use visited set to avoid revisiting
- Clear visited set if running multiple DFS
- Consider BFS if you need shortest path`,
    },
    {
      id: 'patterns',
      title: 'Common DFS Patterns',
      content: `**Pattern 1: Top-Down (Pass information down)**
\`\`\`python
def path_sum(root, target):
    def dfs(node, remaining):
        if not node:
            return False
        if not node.left and not node.right:
            return remaining == node.val
        return (dfs(node.left, remaining - node.val) or
                dfs(node.right, remaining - node.val))
    return dfs(root, target)
\`\`\`

**Pattern 2: Bottom-Up (Return information up)**
\`\`\`python
def max_depth(root):
    if not root:
        return 0
    left = max_depth(root.left)
    right = max_depth(root.right)
    return 1 + max(left, right)
\`\`\`

**Pattern 3: Path Tracking**
\`\`\`python
def all_paths(root):
    paths = []
    def dfs(node, path):
        if not node:
            return
        path.append(node.val)
        if not node.left and not node.right:
            paths.append(path[:])
        dfs(node.left, path)
        dfs(node.right, path)
        path.pop()  # Backtrack
    dfs(root, [])
    return paths
\`\`\`

**Pattern 4: Validating Trees**
\`\`\`python
def is_valid_bst(root):
    def dfs(node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (dfs(node.left, min_val, node.val) and
                dfs(node.right, node.val, max_val))
    return dfs(root, float('-inf'), float('inf'))
\`\`\`

**Pattern 5: Connected Components (Graph)**
\`\`\`python
def count_components(graph):
    visited = set()
    count = 0
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    for node in graph:
        if node not in visited:
            dfs(node)
            count += 1
    return count
\`\`\``,
    },
    {
      id: 'interview-strategy',
      title: 'Interview Strategy',
      content: `**Recognizing DFS Problems:**

**Keywords to watch for:**
- "Explore all paths"
- "Find all combinations/permutations"
- "Validate tree"
- "Find connected components"
- "Detect cycle"
- "Path sum"
- "Tree traversal"

**DFS vs BFS Decision:**

**Use DFS when:**
- Need to explore ALL paths
- Tree traversal problems
- Backtracking problems
- Path finding (not shortest)
- Cycle detection
- Topological sort
- Memory is limited (O(H) vs O(W))

**Use BFS when:**
- Need SHORTEST path (unweighted)
- Level-order traversal
- Nearest neighbor problems
- Width could be less than height

**Problem-Solving Framework:**

**1. Identify the Structure:**
- Tree or graph?
- Directed or undirected?
- Cycles possible?

**2. Choose DFS Variant:**
- Preorder? Inorder? Postorder?
- Recursive or iterative?
- Need to track visited?

**3. Define the State:**
- What information to pass down?
- What to return up?
- What to track globally?

**4. Handle Base Cases:**
- Null node?
- Leaf node?
- Target found?

**5. Backtrack if Needed:**
- Path problems
- Combination problems
- Restore state after recursion

**Common Pitfalls:**
- Forgetting visited set in graphs
- Not handling null nodes
- Forgetting to backtrack
- Incorrect base cases
- Stack overflow (use iterative)
- Modifying shared state incorrectly

**Interview Tips:**
- Start with recursive (simpler)
- Mention iterative alternative
- Discuss space complexity
- Draw small example
- Trace through recursion
- Handle edge cases (empty, single node)`,
    },
  ],
  keyTakeaways: [
    'DFS explores as deep as possible before backtracking using recursion or a stack',
    'Tree DFS: Preorder (Root→L→R), Inorder (L→Root→R), Postorder (L→R→Root)',
    'Graph DFS requires visited set to avoid cycles, time is O(V+E)',
    'Recursive DFS uses O(H) space for call stack, where H is height',
    'Use DFS for: tree traversals, finding all paths, backtracking, cycle detection',
    'Top-down DFS passes info down, bottom-up DFS returns info up',
    'Iterative DFS with explicit stack avoids recursion depth limits',
    'DFS is natural for problems requiring exhaustive exploration or path tracking',
  ],
  relatedProblems: ['max-depth-binary-tree', 'path-sum', 'number-of-islands'],
};
