/**
 * Quiz questions for Tree Traversals section
 */

export const traversalsQuiz = [
  {
    id: 'q1',
    question:
      'Compare inorder, preorder, and postorder traversals. When would you use each one and why?',
    sampleAnswer:
      'Inorder (left, node, right) gives sorted order for BSTs - crucial for retrieving data in order. Use it when you need elements in ascending order. Preorder (node, left, right) processes parent before children, useful for copying trees or creating prefix expressions. Use it when parent context is needed before processing children. Postorder (left, right, node) processes children before parent, useful for deleting trees or evaluating postfix expressions. Use it when you need child results before processing parent, like calculating subtree sizes. The key is understanding what information you need when: inorder for sorted data, preorder for top-down processing, postorder for bottom-up processing.',
    keyPoints: [
      'Inorder: sorted order for BSTs',
      'Preorder: parent before children (copy tree)',
      'Postorder: children before parent (delete tree)',
      'Inorder: ascending order',
      'Preorder: top-down, Postorder: bottom-up',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between DFS and BFS. When would you choose one over the other?',
    sampleAnswer:
      'DFS explores as deep as possible before backtracking, using stack (or recursion). BFS explores level by level, using queue. DFS is better when solutions are likely deep in tree, when you need to explore all paths (like counting paths), or when space is limited (O(h) vs O(w) where h is height, w is width). BFS is better for finding shortest path (level-order guarantees minimum depth), when solutions are likely near root, or for level-specific operations. For example, finding minimum depth: BFS stops at first leaf, efficient. Finding maximum depth: DFS natural recursion. The choice depends on tree shape and what you are searching for.',
    keyPoints: [
      'DFS: deep first, stack/recursion, O(h) space',
      'BFS: level by level, queue, O(w) space',
      'DFS: solutions deep, all paths, save space',
      'BFS: shortest path, solutions near root, level operations',
      'Choice depends on tree shape and goal',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through how iterative DFS works with an explicit stack. Why does it match recursive behavior?',
    sampleAnswer:
      'Iterative DFS uses explicit stack to simulate recursion. Push root onto stack. While stack not empty: pop node, process it, push right child then left child (right first so left is processed first). This matches recursion because recursion uses call stack implicitly - when we call on left subtree, the right subtree call is waiting on the call stack. Our explicit stack does the same: by pushing right first, it waits at bottom of stack while we process left. The stack stores pending work just like recursion stores pending function calls. Order of pushing (right then left) ensures we process nodes in same order as recursive preorder. Iterative gives us control over stack and avoids stack overflow for deep trees.',
    keyPoints: [
      'Explicit stack simulates call stack',
      'Push right then left (left processed first)',
      'Stack stores pending work',
      'Matches recursive call order',
      'Avoids stack overflow, more control',
    ],
  },
];
