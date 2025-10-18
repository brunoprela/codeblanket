/**
 * Quiz questions for Common DFS Patterns section
 */

export const patternsQuiz = [
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
    question: 'Describe flood fill algorithm. How is it a DFS application?',
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
];
