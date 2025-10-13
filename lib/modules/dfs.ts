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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain DFS (Depth-First Search). How does it work and what makes it "depth-first"?',
          sampleAnswer:
            'DFS explores as far as possible along each branch before backtracking. Start at root/source, explore one child completely before siblings. "Depth-first" means go deep before wide. Implementation: recursive (natural fit, uses call stack) or iterative (explicit stack). For example, tree [1, [2, [4, 5]], [3]]: DFS visits 1→2→4(backtrack)→5(backtrack to 2, then 1)→3. Compare BFS: 1→2→3→4→5 (level by level). DFS uses stack (LIFO), BFS uses queue (FIFO). Time O(V+E) for graphs, O(n) for trees. Space O(h) for recursive (call stack height), O(V) for visited set. Natural for: tree traversal, cycle detection, topological sort, path finding.',
          keyPoints: [
            'Explore deep before wide, uses stack',
            'Recursive (call stack) or iterative (explicit stack)',
            'LIFO: last child visited first',
            'O(V+E) time, O(h) space for trees',
            'Uses: traversal, cycles, topo sort, paths',
          ],
        },
        {
          id: 'q2',
          question: 'Compare DFS vs BFS. When would you choose each?',
          sampleAnswer:
            'DFS: stack (LIFO), goes deep, O(h) space for trees. BFS: queue (FIFO), goes wide, O(w) space where w is max width. Choose DFS when: exploring all paths, backtracking, tree problems, topological sort, cycle detection, space matters (trees are deep not wide). Choose BFS when: shortest path needed, level-order traversal, closest nodes matter, trees are wide not deep. For example, shortest path in unweighted graph → BFS guarantees first found is shortest. All paths in tree → DFS natural recursion. Finding if graph has cycle → DFS detects back edges. Level order → BFS processes by levels. Space: DFS O(h) good for deep narrow, BFS O(w) good for shallow wide. DFS simpler to code recursively.',
          keyPoints: [
            'DFS: stack, deep, O(h) space',
            'BFS: queue, wide, O(w) space',
            'DFS when: all paths, backtracking, topology',
            'BFS when: shortest path, level order',
            'Choice depends on: problem, tree shape',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the three standard tree traversals (inorder, preorder, postorder)? When do you use each?',
          sampleAnswer:
            'Preorder: root, left, right. Visit node before children. Use: copy tree, prefix expression, serialize tree. Inorder: left, root, right. For BST, gives sorted order. Use: BST sorted output, expression trees (infix). Postorder: left, right, root. Visit children before node. Use: delete tree (children first), postfix expression, tree height. For tree [1, [2, [4, 5]], [3]]: preorder 1,2,4,5,3. Inorder 4,2,5,1,3. Postorder 4,5,2,3,1. Recursive implementation natural: preorder process before recursion, inorder process between left and right recursion, postorder process after recursion. Memory: BST inorder is sorted, useful for validation.',
          keyPoints: [
            'Preorder: root, left, right (copy, serialize)',
            'Inorder: left, root, right (BST sorted)',
            'Postorder: left, right, root (delete, height)',
            'BST inorder gives sorted sequence',
            'Order determines when node is processed',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain recursive DFS on trees. Why is recursion a natural fit for tree traversal?',
          sampleAnswer:
            'Recursive DFS: base case (node is None), recursive case (process node, recurse on children). Natural fit because: trees are recursive structures (subtrees are also trees), call stack handles backtracking automatically, code mirrors problem structure. For example, max depth: if None return 0, else 1 + max(depth(left), depth(right)). Three lines capture entire algorithm. Compare iterative: need explicit stack, manual tracking. Recursion elegance: base case simple, recursive step processes current and delegates to recursion. The call stack is implicit DFS stack. For same tree problem: check val, recursively check left and right subtrees. Each recursive call handles one subtree - clean separation. Drawback: stack overflow for very deep trees (10K+ nodes in linear chain).',
          keyPoints: [
            'Base case (None) + recursive case',
            'Natural: trees are recursive structures',
            'Call stack handles backtracking automatically',
            'Code mirrors problem structure elegantly',
            'Drawback: stack overflow on very deep trees',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through implementing max depth and same tree using recursive DFS. What is the pattern?',
          sampleAnswer:
            'Pattern: 1) Check base case (None node). 2) Process current node. 3) Recurse on children. 4) Combine results. Max depth: base None → 0, recursive 1 + max(left, right). Each node adds 1, max of children gives deepest. Same tree: base both None → True, one None → False, values differ → False, recursive check left and right, combine with AND. For example, max depth [1, [2, [4, 5]], 3]: depth(1) = 1 + max(depth(left=2), depth(right=3)). depth(2) = 1 + max(depth(4)=1, depth(5)=1) = 2. depth(1) = 1 + max(2, 1) = 3. The pattern: recursive calls on subtrees, combine results at current node. This works for: sum, max, min, count, validation.',
          keyPoints: [
            'Pattern: base case, process, recurse, combine',
            'Max depth: 1 + max(left, right)',
            'Same tree: check val, AND of left and right',
            'Combine: max, min, sum, AND, OR',
            'Each node processes and delegates to children',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe path sum problem. How does DFS naturally track the path?',
          sampleAnswer:
            'Path sum: given target, find root-to-leaf path with sum equal to target. DFS natural because: 1) Explores paths one at a time. 2) Backtracking automatically removes nodes. 3) Track running sum as parameter. Algorithm: if leaf and sum == target → True, else recurse left and right with reduced target. For tree [5, [4, [11, [7, 2]]], [8, [13, 4]]], target=22: path 5→4→11→2 sums to 22. DFS explores 5(target=17)→4(target=13)→11(target=2)→7(no match), backtrack, try 2(match!). The call stack naturally tracks the path. For all paths: accumulate path list, when leaf matches add copy to result, backtrack by popping. The key: DFS + recursion gives path tracking for free via call stack.',
          keyPoints: [
            'Track running sum as parameter',
            'Leaf check: sum == target',
            'Recurse with reduced target',
            'Call stack tracks path automatically',
            'Backtracking: return from recursion removes node',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question: 'Explain DFS on graphs. What is different from tree DFS?',
          sampleAnswer:
            'Graph DFS similar to tree but: 1) Need visited set (graphs have cycles). 2) Multiple edges to same node possible. 3) No clear parent-child (undirected or cycles). 4) Start node may not reach all (disconnected). Tree DFS: no visited needed (no cycles), parent known. Graph DFS: visited set prevents infinite loops, track visited in set or array. For example, graph 0→1, 1→2, 2→0 (cycle): without visited, 0→1→2→0→1... infinite. With visited: mark 0, visit 1, mark 1, visit 2, mark 2, try 0 (already visited, skip). Time O(V+E) visit each vertex once, each edge once. Space O(V) for visited. For disconnected graphs: run DFS from each unvisited node (connected components).',
          keyPoints: [
            'Graph needs visited set (cycles exist)',
            'Tree: no cycles, visited not needed',
            'Mark visited before recursing',
            'O(V+E) time, O(V) space',
            'Disconnected: run DFS from each unvisited',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe cycle detection using DFS. How do you differentiate back edge from tree edge?',
          sampleAnswer:
            'Cycle detection: use three states: unvisited, visiting (in current path), visited (finished). Cycle exists if we reach a "visiting" node (back edge to ancestor). Algorithm: mark node as visiting, recurse on neighbors, mark as visited when done. If neighbor is visiting → cycle. For example, 0→1→2→0: visit 0 (visiting), visit 1 (visiting), visit 2 (visiting), try 0 (visiting! cycle found). For DAG: never encounter visiting node. Directed graph: visiting state tracks current path. Undirected: simpler, just check if neighbor is visited and not parent. The key: visiting state means node is in current DFS path (ancestor), detecting back edge proves cycle.',
          keyPoints: [
            'Three states: unvisited, visiting, visited',
            'Cycle: reach node in "visiting" state',
            'Visiting = in current DFS path',
            'Back edge to ancestor = cycle',
            'Undirected: simpler, check visited except parent',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain connected components using DFS. How do you count them?',
          sampleAnswer:
            'Connected components: groups of nodes reachable from each other. Count: for each unvisited node, run DFS (marks entire component), increment counter. For example, graph with nodes 0,1,2,3,4,5 and edges 0-1, 2-3-4: two components. Algorithm: visited = set(), count = 0; for each node: if not visited, DFS(node), count++. First DFS from 0 marks 0,1. Second DFS from 2 marks 2,3,4. Result: 2 components. Each DFS explores one component completely. Time O(V+E) total (each node/edge visited once across all DFS calls). This works for undirected graphs. For directed: strongly connected components need different algorithm (Tarjan, Kosaraju).',
          keyPoints: [
            'Count: DFS from each unvisited node',
            'Each DFS marks one component',
            'Increment counter per DFS call',
            'O(V+E) total across all DFS',
            'Directed: use Tarjan/Kosaraju for SCC',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain iterative DFS using explicit stack. Why use iterative over recursive?',
          sampleAnswer:
            'Iterative DFS: use explicit stack instead of call stack. Push root, loop: pop node, process, push children (right then left for same order as recursive). For example, tree [1, [2, [4, 5]], 3]: stack=[1], pop 1 push 3,2, stack=[3,2], pop 2 push 5,4, etc. Result: 1,2,4,5,3 (same as recursive preorder). Use iterative when: deep trees (avoid stack overflow), need control over stack, language lacks good recursion support, debugging easier with explicit state. Tradeoff: more code (explicit stack management) vs recursive elegance. For graphs: same pattern, add visited set. Iterative gives you: no recursion depth limit, explicit stack inspection, easier to convert to BFS (replace stack with queue).',
          keyPoints: [
            'Explicit stack replaces call stack',
            'Push root, loop: pop, process, push children',
            'Push right before left for left-first order',
            'Use when: deep trees, stack overflow concerns',
            'Tradeoff: more code vs recursive elegance',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through converting recursive DFS to iterative. What changes?',
          sampleAnswer:
            'Conversion steps: 1) Replace recursion with explicit stack. 2) Push root to stack. 3) Loop while stack not empty. 4) Pop node, process. 5) Push children (order matters). For example, recursive max depth: depth(node) = 1 + max(depth(left), depth(right)). Iterative: stack = [(node, depth)], loop: pop (node, d), if leaf update max, push children with d+1. The state (current depth) that was in recursive parameters goes into stack tuples. Recursive return values become: accumulated in variables (max, sum) or checked in conditions. The key: what was in call stack (parameters, local vars) goes into explicit stack as tuples. Backtracking: happens automatically when popping.',
          keyPoints: [
            'Call stack → explicit stack',
            'Parameters → stack tuples (node, state)',
            'Return values → accumulate in variables',
            'Push children to continue exploration',
            'Order: push right before left',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare recursive vs iterative DFS complexity. Which is more space-efficient?',
          sampleAnswer:
            'Time: both O(V+E) for graphs, O(n) for trees - same, visit each node once. Space: depends on tree shape. Recursive: O(h) where h is height (call stack). Iterative: O(h) for balanced, O(n) worst case (stack holds level). For balanced tree: both O(log n). For skewed tree: both O(n). The difference: recursive uses call stack (limited, typically few MB), iterative uses heap (more memory available). For very deep trees (100K nodes in chain): recursive stackoverflow, iterative works. Most cases: recursive simpler and same space. Use iterative only if: proven stack overflow, need explicit control, or converting to BFS.',
          keyPoints: [
            'Time: both O(V+E) or O(n)',
            'Space: both O(h) typically',
            'Recursive: call stack (limited)',
            'Iterative: heap stack (more memory)',
            'Very deep: iterative safer, else prefer recursive',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Analyze DFS time and space complexity for different structures.',
          sampleAnswer:
            'Trees: Time O(n) visit each node once. Space O(h) for call stack where h is height. Balanced tree: O(log n) space. Skewed tree (linear): O(n) space. Graphs: Time O(V+E) visit each vertex and edge once. Space O(V) for visited set plus O(h) for call stack, worst case O(V). Matrix (grid): Time O(rows × cols) visit each cell. Space O(rows × cols) for visited (can optimize with in-place marking). For example, complete binary tree with 1M nodes: height = log(1M) ≈ 20, space O(20). Linked list tree with 1M nodes: height = 1M, space O(1M). Graph with 1000 vertices, 5000 edges: time O(6000), space O(1000).',
          keyPoints: [
            'Trees: O(n) time, O(h) space',
            'Balanced: O(log n) space, skewed: O(n)',
            'Graphs: O(V+E) time, O(V) space',
            'Matrix: O(rows×cols) time and space',
            'Space depends on recursion depth',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare DFS space for recursive vs iterative across different tree shapes.',
          sampleAnswer:
            'Balanced tree: recursive O(log n), iterative O(log n) - same. Skewed tree: recursive O(n), iterative O(n) - same. Complete tree: recursive O(log n), iterative O(w) where w is max width - iterative can be worse. For example, complete binary tree level k has 2^k nodes. At bottom, width is n/2, so iterative O(n) space. Recursive only O(log n) for depth. General: recursive space = height, iterative space = maximum stack size (can be all children of a level). The key: recursive follows one path at a time (height), iterative may have multiple branches in stack. For DFS, iterative is usually O(h) too. For BFS (queue), iterative is O(width).',
          keyPoints: [
            'Balanced: both O(log n)',
            'Skewed: both O(n)',
            'Complete: recursive O(log n) better',
            'Recursive: O(height), iterative: O(max stack)',
            'DFS iterative usually also O(h)',
          ],
        },
        {
          id: 'q3',
          question:
            'How does visited set affect space complexity? When can you optimize it?',
          sampleAnswer:
            'Visited set: O(V) space for V vertices. Cannot avoid for graphs with cycles (infinite loop without it). Trees: do not need visited (no cycles). Optimization 1: for grids, mark visited in-place (change cell value to avoid separate set). Saves O(rows×cols). Optimization 2: for trees, no visited needed. Optimization 3: for graphs where you can modify input, mark nodes in-place. For example, grid "1"→"0" after visiting, saves visited set. Cannot optimize when: cannot modify input, need to preserve state, need to run multiple DFS (must clear visited each time). Tradeoff: in-place marking destroys input.',
          keyPoints: [
            'Visited set: O(V) space',
            'Trees: no visited needed',
            'Grid: mark in-place saves O(rows×cols)',
            'Cannot optimize: cannot modify, multiple runs',
            'Tradeoff: space vs preserving input',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'What are common DFS patterns? How do you recognize when to use each?',
          sampleAnswer:
            'Pattern 1: Tree traversal (inorder, preorder, postorder) - recognize by tree structure, need all nodes. Pattern 2: Path sum/path finding - recognize by root-to-leaf path, target sum. Pattern 3: Cycle detection - recognize by graph, "has cycle" question. Pattern 4: Connected components - recognize by "count islands", "num components". Pattern 5: Topological sort - recognize by DAG, "course schedule", dependencies. Pattern 6: Validate BST - recognize by tree properties, need check ordering. For example, "count number of islands in grid" → component pattern, DFS from each unvisited land cell. "Detect cycle in graph" → cycle detection pattern, three-state DFS. "Binary tree maximum path sum" → tree DFS with global variable.',
          keyPoints: [
            'Traversal: visit all nodes in order',
            'Path finding: root-to-leaf with target',
            'Cycle detection: graph validation',
            'Components: count disconnected parts',
            'Topological sort: DAG ordering',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe flood fill algorithm. How is it a DFS application?',
          sampleAnswer:
            'Flood fill: change color of connected region (4-directional or 8-directional). Example: paint bucket in image editor. Algorithm: DFS from start pixel, if pixel has old color, change to new color, recurse on neighbors. For grid [[1,1,1],[1,1,0],[1,0,1]], start (1,1), old=1, new=2: DFS marks (1,1)→(0,0)→(0,1)→(0,2)→(1,0)→(2,0). All connected 1s become 2s. This is DFS because: explores region depth-first, backtracking automatic, visits each cell once, marks visited by changing color. Time O(rows×cols), space O(rows×cols) for recursion. Application: image processing, game AI, region detection. Optimization: iterative with stack to avoid recursion limit.',
          keyPoints: [
            'Change color of connected region',
            'DFS from start, mark visited by recoloring',
            'Recurse 4 or 8 directions',
            'O(rows×cols) time and space',
            'Applications: paint, games, region detection',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through solving "number of islands" problem with DFS. What is the pattern?',
          sampleAnswer:
            'Number of islands: count connected components of 1s in grid. Pattern: iterate each cell, if land (1) and unvisited, increment counter and DFS to mark entire island. Algorithm: for each cell (i,j): if grid[i][j]==1, islands++, dfs(i,j). DFS marks (i,j) as visited (set to 0 or use visited set), recurses on 4 neighbors. For example, grid [[1,1,0],[0,1,0],[0,0,1]]: cell (0,0)=1, islands=1, DFS marks (0,0),(0,1),(1,1) as one island. Cell (2,2)=1, islands=2, DFS marks only (2,2). Result: 2 islands. Time O(rows×cols) visit each cell once. Space O(rows×cols) recursion worst case (entire grid is one island). This pattern works for all "count components" problems.',
          keyPoints: [
            'Count connected components in grid',
            'Iterate: if unvisited land, DFS + count++',
            'DFS marks entire component',
            'O(rows×cols) time and space',
            'Pattern for all "count regions" problems',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize a DFS problem? What keywords and patterns signal DFS?',
          sampleAnswer:
            'Keywords: "traverse", "visit all", "path", "connected", "cycle", "component", "island", "region", "validate". Patterns: 1) Tree traversal (inorder, preorder, postorder). 2) Path finding (root-to-leaf). 3) Graph exploration (all nodes, components). 4) Backtracking (permutations, combinations). 5) Cycle detection. 6) Topological sort. Signals: tree/graph structure, need to explore deeply, no "shortest" requirement (else BFS). For example, "validate binary search tree" → tree DFS. "Find all root-to-leaf paths" → DFS with path tracking. "Count connected components" → DFS from each unvisited. "Course schedule" → cycle detection DFS. If problem needs shortest path or level-order → BFS. If exploring all possibilities → DFS.',
          keyPoints: [
            'Keywords: traverse, path, connected, cycle, component',
            'Patterns: tree traversal, paths, exploration',
            'Tree/graph + explore deeply → DFS',
            'Need shortest → BFS',
            'Explore all possibilities → DFS',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your DFS interview approach from recognition to implementation.',
          sampleAnswer:
            'First, recognize DFS from keywords (tree, path, component, cycle). Second, identify pattern (traversal? path sum? cycle?). Third, choose recursive vs iterative (recursive simpler unless deep). Fourth, define base case (None/null, leaf, visited). Fifth, define recursive case (process node, recurse on children/neighbors). Sixth, handle visited set for graphs. Seventh, test with examples and edges. Finally, analyze complexity. For example, "validate BST": recognize tree DFS, pattern is tree property validation, recursive approach, base case is None (True), recursive case check val in range and recurse with updated ranges, test with valid/invalid BSTs, O(n) time O(h) space. Show: recognition, pattern, implementation, testing.',
          keyPoints: [
            'Recognize: keywords, tree/graph',
            'Pattern: traversal, path, cycle, component',
            'Choose: recursive (simpler) or iterative',
            'Define: base case, recursive case',
            'Graphs: add visited set',
            'Test and analyze complexity',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes in DFS problems? How do you avoid them?',
          sampleAnswer:
            'First: forgetting visited set in graphs (infinite loop on cycles). Second: wrong base case (None not handled → crash). Third: modifying global state without reset (multiple test cases fail). Fourth: marking visited at wrong time (before vs after recursion matters). Fifth: not handling disconnected graphs (only explore from one start). Sixth: stack overflow on deep trees (use iterative). My strategy: 1) Always check None first. 2) Use visited set for graphs. 3) Mark visited before recursing (avoid re-visiting). 4) For problems needing all paths, backtrack by unmarking. 5) Test: empty, single node, cycle, disconnected. 6) For very deep, consider iterative. Most mistakes from missing visited set or wrong base case.',
          keyPoints: [
            'Mistakes: no visited, wrong base, global state',
            'Visited timing: mark before recursing',
            'Handle: None, cycles, disconnected',
            'Very deep: use iterative',
            'Test: empty, single, cycle, disconnected',
            'Most common: missing visited set',
          ],
        },
      ],
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
