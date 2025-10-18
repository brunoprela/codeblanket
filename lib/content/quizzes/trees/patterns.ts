/**
 * Quiz questions for Common Tree Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the recursive DFS pattern for tree problems. What are the three steps and why is this pattern so powerful?',
    sampleAnswer:
      'The recursive DFS pattern has three steps: base case (null check, return default), recurse on children (get results from left and right subtrees), compute current result (use children results and current node). This is powerful because it naturally divides the problem - solve for subtrees recursively, then combine their results. For example, tree height: base case returns 0 for null, recurse gets left and right heights, compute returns 1 + max(left, right). The pattern works for most tree problems: max depth, balanced check, path sum, etc. It leverages trees recursive structure - a tree is root plus two subtrees. Once you master this pattern, many tree problems become straightforward.',
    keyPoints: [
      'Three steps: base case, recurse, compute',
      'Base: null check, return default',
      'Recurse: get children results',
      'Compute: combine results with current node',
      'Works for most tree problems',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the subtree pattern. How do you validate that a tree is a valid BST using this approach?',
    sampleAnswer:
      'The subtree pattern processes each node with context from ancestors. For BST validation, each node must be within a valid range: left subtree must be less than node, right subtree must be greater. I pass down min and max bounds: for left child, max becomes current node value (everything must be less). For right child, min becomes current node value (everything must be greater). At each node, I check if value is in valid range (min < value < max), then recurse with updated bounds. This ensures BST property for entire tree. Without bounds, just checking node > left and node < right is insufficient - you miss violations deeper in tree. The pattern: pass down ancestor context, check current node against context.',
    keyPoints: [
      'Pass context from ancestors',
      'BST: each node has valid range',
      'Left: update max bound',
      'Right: update min bound',
      'Ensures global BST property',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the path pattern for finding all root-to-leaf paths. How do you track and return paths?',
    sampleAnswer:
      'Path pattern maintains current path as we traverse. Use list to track nodes from root to current. At each node, add it to path, check if leaf (no children), if leaf add copy of path to result. Then recurse on children with updated path. After recursion, remove current node from path (backtrack). The backtracking is crucial - when returning from a subtree, we remove that subtree root so path is correct for the other subtree. For example, after exploring left subtree, we backtrack then explore right subtree with correct path. This DFS with backtracking pattern works for all path problems: path sum, max path, specific target path. Key is maintaining and backtracking the path.',
    keyPoints: [
      'Maintain current path as list',
      'Add node, check if leaf',
      'Recurse with updated path',
      'Backtrack: remove node after recursion',
      'Works for all path problems',
    ],
  },
];
