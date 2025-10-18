/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize that a problem requires tree algorithms? What keywords signal this?',
    sampleAnswer:
      'Several signals indicate tree problems. First, explicit mention: "binary tree", "BST", "tree node". Second, hierarchical relationship keywords: "parent-child", "ancestor-descendant", "root-to-leaf". Third, traversal language: "inorder", "preorder", "level-order". Fourth, BST-specific: "sorted order", "search in log time". Fifth, structural properties: "balanced", "symmetric", "path sum", "depth", "diameter". The key question: does the problem involve hierarchical data or recursive substructure? Even if not explicitly tree, thinking in tree terms can help - like decision trees for game states. If you see "node", "children", or "recursive structure", consider tree algorithms.',
    keyPoints: [
      'Explicit: binary tree, BST, tree node',
      'Hierarchical: parent-child, ancestor-descendant',
      'Traversal: inorder, preorder, level-order',
      'BST: sorted, search log time',
      'Structural: balanced, symmetric, paths, depth',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your approach to a tree problem in an interview, from problem statement to explaining complexity.',
    sampleAnswer:
      'First, I clarify: binary tree or BST? Are there null nodes? What should I return? Then I identify the pattern: is it traversal (DFS/BFS), recursive divide-and-conquer, path problem, or BST property? I explain my approach: "I will use recursive DFS, base case for null returns 0, recurse on children, combine results with 1 + max". I draw a small tree example and trace through the recursion step by step, showing how we build up from leaves. While coding, I handle null checks carefully and explain the three recursive steps. After coding, I state complexity: "O(n) time to visit all nodes, O(h) space for recursion where h is height". I mention both approaches if applicable: "Could also do iteratively with stack to avoid recursion overhead".',
    keyPoints: [
      'Clarify: binary/BST, nulls, return value',
      'Identify pattern: traversal, recursive, path, BST',
      'Draw example, trace recursion',
      'Code with null checks and explanation',
      'State time and space complexity',
      'Mention alternative approaches',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes in tree problems and how do you avoid them?',
    sampleAnswer:
      'First: forgetting null check, causing null pointer errors. I always start recursive functions with "if not root: return default". Second: confusing preorder/inorder/postorder - mixing up when to process node vs recurse. I remember: pre = before children, in = between children, post = after children. Third: using wrong base case - returning wrong default value. I think through: what should empty tree return? Fourth: forgetting to pass updated parameters in recursion - like not updating remaining sum in path problems. Fifth: modifying tree structure incorrectly - losing references. I draw pointers before coding. Strategy: always draw example, trace carefully, test null and single node cases.',
    keyPoints: [
      'Null checks: start with if not root',
      'Traversal order: pre/in/post timing',
      'Correct base case return value',
      'Pass updated parameters in recursion',
      'Draw pointers for structure modifications',
      'Test: null, single node cases',
    ],
  },
];
