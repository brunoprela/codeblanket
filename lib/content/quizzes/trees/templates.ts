/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the basic recursive DFS template. What makes this pattern applicable to so many tree problems?',
    sampleAnswer:
      'The basic recursive DFS template has three parts: null check (base case), recursive calls on children (divide), combine results (conquer). This works for many problems because most tree problems have recursive substructure - answer for tree depends on answers for subtrees. For max depth: base returns 0, recursive gets left and right depths, combine with 1 + max. For sum of nodes: base returns 0, recursive gets left and right sums, combine with node.val + left + right. The template is universal because it matches how trees are defined recursively. Once you recognize a problem fits this pattern, implementation is straightforward - just fill in the three parts.',
    keyPoints: [
      'Three parts: base, recurse, combine',
      'Matches recursive tree structure',
      'Works because: answer depends on subtree answers',
      'Just fill in three parts',
      'Universal pattern for tree problems',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the BFS template with queue. Why do we need to track level size separately for level-order problems?',
    sampleAnswer:
      'BFS template: initialize queue with root, while queue not empty, process nodes. For level-order, we need level size because queue mixes nodes from current level with children added during processing. Without tracking size, we cannot tell when one level ends and next begins. The pattern: before processing level, save queue size (number of nodes at this level), process exactly that many nodes, adding their children. This ensures we process one complete level before starting next. For example, with queue [1,2,3], size is 3, we process exactly 3 nodes, adding their children to queue for next level. Level size separates levels within the queue.',
    keyPoints: [
      'Queue processes level by level',
      'Level size separates levels',
      'Save size before processing',
      'Process exactly size nodes',
      'Children added for next level',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the top-down vs bottom-up recursive patterns. When would you use each?',
    sampleAnswer:
      'Top-down passes information from ancestors down to descendants as parameters. Use when current node needs context from parents - like passing bounds for BST validation or accumulated path for path problems. Bottom-up returns information from children up to parents as return values. Use when parent needs results computed by children - like tree height, node counts, balanced checks. Top-down: parameters carry down, process before recursing. Bottom-up: return values bubble up, process after recursing. Many problems can use either, but one feels more natural. For example, path sum is natural top-down (pass remaining sum down), tree height is natural bottom-up (return heights up).',
    keyPoints: [
      'Top-down: pass context down as parameters',
      'Bottom-up: return results up as return values',
      'Top-down: parent context needed',
      'Bottom-up: children results needed',
      'Many problems work with either',
    ],
  },
];
