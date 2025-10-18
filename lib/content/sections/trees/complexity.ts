/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Tree Operation Complexities:**

**Binary Search Tree (Balanced):**
| Operation | Average | Worst (Skewed) |
|-----------|---------|----------------|
| Search | O(log N) | O(N) |
| Insert | O(log N) | O(N) |
| Delete | O(log N) | O(N) |
| Min/Max | O(log N) | O(N) |

**Traversal Complexities:**
| Traversal | Time | Space (Recursive) | Space (Iterative) |
|-----------|------|-------------------|-------------------|
| Preorder | O(N) | O(H) | O(H) |
| Inorder | O(N) | O(H) | O(H) |
| Postorder | O(N) | O(H) | O(H) |
| Level Order | O(N) | O(W) | O(W) |

Where:
- N = number of nodes
- H = height (log N for balanced, N for skewed)
- W = maximum width (N/2 for complete trees)

**Common Problem Complexities:**

**Height/Depth:**
- Time: O(N) - visit all nodes
- Space: O(H) - recursion depth

**Diameter:**
- Time: O(N) - visit all nodes
- Space: O(H) - recursion stack

**Lowest Common Ancestor:**
- Time: O(N) - might visit all nodes
- Space: O(H) - recursion depth

**Path Sum:**
- Time: O(N) - might check all paths
- Space: O(H) - recursion + path storage

**Serialize/Deserialize:**
- Time: O(N) - process each node once
- Space: O(N) - store all nodes

**Balanced Tree Check:**
- Time: O(N) - visit each node once
- Space: O(H) - recursion depth

**Key Insights:**

**Space Complexity:**
- **Recursive DFS**: O(H) for call stack
- **Iterative DFS**: O(H) for explicit stack
- **BFS**: O(W) for queue (worst case N/2 for complete tree)

**Height Matters:**
- Balanced tree: H = O(log N)
- Skewed tree: H = O(N)
- Operations are H-dependent, so balance is crucial

**Complete Binary Tree:**
- Height: O(log N)
- Width: O(N/2) at deepest level
- Efficient for heaps and priority queues`,
};
