/**
 * Quiz questions for Introduction to DFS section
 */

export const introductionQuiz = [
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
];
