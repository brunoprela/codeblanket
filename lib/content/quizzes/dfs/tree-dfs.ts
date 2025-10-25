/**
 * Quiz questions for DFS on Trees section
 */

export const treedfsQuiz = [
  {
    id: 'q1',
    question:
      'Explain recursive DFS on trees. Why is recursion a natural fit for tree traversal?',
    sampleAnswer:
      'Recursive DFS: base case (node is None), recursive case (process node, recurse on children). Natural fit because: trees are recursive structures (subtrees are also trees), call stack handles backtracking automatically, code mirrors problem structure. For example, max depth: if None return 0, else 1 + max (depth (left), depth (right)). Three lines capture entire algorithm. Compare iterative: need explicit stack, manual tracking. Recursion elegance: base case simple, recursive step processes current and delegates to recursion. The call stack is implicit DFS stack. For same tree problem: check val, recursively check left and right subtrees. Each recursive call handles one subtree - clean separation. Drawback: stack overflow for very deep trees (10K+ nodes in linear chain).',
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
      'Pattern: 1) Check base case (None node). 2) Process current node. 3) Recurse on children. 4) Combine results. Max depth: base None → 0, recursive 1 + max (left, right). Each node adds 1, max of children gives deepest. Same tree: base both None → True, one None → False, values differ → False, recursive check left and right, combine with AND. For example, max depth [1, [2, [4, 5]], 3]: depth(1) = 1 + max (depth (left=2), depth (right=3)). depth(2) = 1 + max (depth(4)=1, depth(5)=1) = 2. depth(1) = 1 + max(2, 1) = 3. The pattern: recursive calls on subtrees, combine results at current node. This works for: sum, max, min, count, validation.',
    keyPoints: [
      'Pattern: base case, process, recurse, combine',
      'Max depth: 1 + max (left, right)',
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
];
