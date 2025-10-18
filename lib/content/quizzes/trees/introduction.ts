/**
 * Quiz questions for Introduction to Trees section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain why trees are hierarchical and what makes them different from linear structures like arrays or linked lists.',
    sampleAnswer:
      'Trees are hierarchical because each node can have multiple children, creating parent-child relationships in levels. A node at level k has children at level k+1, forming a tree-like branching structure. This is fundamentally different from linear structures like arrays or linked lists where elements have at most one predecessor and one successor, forming a single chain. Trees enable representing hierarchical relationships like file systems (directories contain files and subdirectories), organization charts (managers have employees), or DOM structure (elements contain child elements). The hierarchy enables divide-and-conquer algorithms - we can recursively solve for subtrees independently.',
    keyPoints: [
      'Hierarchical: nodes have multiple children',
      'Parent-child relationships in levels',
      'vs Linear: single chain, one predecessor/successor',
      'Represents: file systems, org charts, DOM',
      'Enables divide-and-conquer on subtrees',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe what makes a binary tree special compared to general trees. When would you use binary trees over general trees?',
    sampleAnswer:
      'Binary trees limit each node to at most two children - left and right. This constraint gives structure and enables specific algorithms. Binary Search Trees use this for efficient O(log n) search by maintaining left < parent < right ordering. General trees allow unlimited children per node, more flexible but harder to balance and search efficiently. Binary trees are preferred when you need fast search (BST), when data naturally has binary decisions (decision trees), or when implementing heaps and priority queues. The two-child limit makes many operations simpler to implement and analyze. For data with many children per node like file systems, general trees are better.',
    keyPoints: [
      'Binary: at most two children per node',
      'Enables BST with O(log n) search',
      'General trees: unlimited children',
      'Binary: fast search, binary decisions, heaps',
      'General: when naturally many children',
    ],
  },
  {
    id: 'q3',
    question:
      'What is a Binary Search Tree and what property makes it useful? Walk me through why search is O(log n).',
    sampleAnswer:
      'A Binary Search Tree maintains the property: for every node, all left subtree values are less than node value, and all right subtree values are greater. This ordering enables binary search. To search for a value, I compare with root: if target is less, search left; if greater, search right; if equal, found. Each comparison eliminates half the remaining tree. In a balanced BST with n nodes, height is log n. Since we make one comparison per level, search is O(log n). For example, searching in a balanced tree of 1000 nodes takes at most 10 comparisons (2^10 = 1024). This is why BSTs are powerful - same logarithmic efficiency as binary search on arrays, but with fast insertion and deletion.',
    keyPoints: [
      'Property: left < node < right for all nodes',
      'Enables binary search on tree',
      'Each comparison eliminates half',
      'Balanced tree height: log n',
      'O(log n) search, insert, delete',
    ],
  },
];
