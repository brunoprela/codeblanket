import { Problem } from '../types';

export const bfsProblems: Problem[] = [
  {
    id: 'binary-tree-level-order',
    title: 'Binary Tree Level Order Traversal',
    difficulty: 'Easy',
    description: `Given the \`root\` of a binary tree, return the **level order traversal** of its nodes' values. (i.e., from left to right, level by level).


**Approach:**
Use **BFS (Breadth-First Search)** with a queue. Process nodes level by level:
1. Start with root in queue
2. For each level, capture queue size
3. Process all nodes in current level
4. Add their children to queue for next level

**Key Insight:**
The trick is to capture \`len(queue)\` **before** the inner loop to know exactly how many nodes are in the current level.`,
    examples: [
      {
        input: 'root = [3,9,20,null,null,15,7]',
        output: '[[3],[9,20],[15,7]]',
        explanation: 'Level 1: [3], Level 2: [9, 20], Level 3: [15, 7]',
      },
      {
        input: 'root = [1]',
        output: '[[1]]',
        explanation: 'Single node tree',
      },
      {
        input: 'root = []',
        output: '[]',
        explanation: 'Empty tree',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [0, 2000]',
      '-1000 <= Node.val <= 1000',
    ],
    hints: [
      'Use a queue for BFS traversal',
      'Capture queue size before processing level',
      'Process exactly that many nodes for current level',
      'Add children to queue for next level',
    ],
    starterCode: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Return level order traversal as list of levels.
    
    Args:
        root: Root node of the binary tree
        
    Returns:
        List of lists, each inner list represents one level
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 9, 20, null, null, 15, 7]],
        expected: [[3], [9, 20], [15, 7]],
      },
      {
        input: [[1]],
        expected: [[1]],
      },
      {
        input: [[]],
        expected: [],
      },
    ],
    solution: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS solution - process level by level.
    Time: O(N), Space: O(W) where W is max width
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        level_size = len(queue)  # Capture size before loop!
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result


# Alternative: Without capturing level size (flatten instead of levels)
def level_order_flat(root: Optional[TreeNode]) -> List[int]:
    """Return flattened level-order traversal"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result`,
    timeComplexity: 'O(N)',
    spaceComplexity: 'O(W) where W is maximum width',

    leetcodeUrl:
      'https://leetcode.com/problems/binary-tree-level-order-traversal/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6ZnyEApgFYg',
    order: 1,
    topic: 'Breadth-First Search (BFS)',
    leetcodeUrl:
      'https://leetcode.com/problems/binary-tree-level-order-traversal/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6ZnyEApgFYg',
  },
  {
    id: 'shortest-path-binary-matrix',
    title: 'Shortest Path in Binary Matrix',
    difficulty: 'Medium',
    description: `Given an \`n x n\` binary matrix \`grid\`, return the length of the **shortest clear path** in the matrix. If there is no clear path, return \`-1\`.

A **clear path** in a binary matrix is a path from the **top-left** cell (0, 0) to the **bottom-right** cell (n-1, n-1) such that:
- All visited cells are \`0\`
- All adjacent cells are **8-directionally** connected

The **length of a clear path** is the number of visited cells.


**Approach:**
Use **BFS** to find the shortest path in an unweighted grid:
1. Start from (0, 0) if it is 0
2. Explore all 8 directions
3. Track visited cells
4. Return distance when reaching (n-1, n-1)

**Key Insight:**
BFS guarantees the shortest path in an unweighted graph. First time you reach the destination is the shortest path.`,
    examples: [
      {
        input: 'grid = [[0,1],[1,0]]',
        output: '2',
        explanation: 'Path: (0,0) → (1,1)',
      },
      {
        input: 'grid = [[0,0,0],[1,1,0],[1,1,0]]',
        output: '4',
        explanation: 'Path: (0,0) → (0,1) → (0,2) → (1,2) → (2,2)',
      },
      {
        input: 'grid = [[1,0,0],[1,1,0],[1,1,0]]',
        output: '-1',
        explanation: 'Start cell is blocked',
      },
    ],
    constraints: [
      'n == grid.length',
      'n == grid[i].length',
      '1 <= n <= 100',
      'grid[i][j] is 0 or 1',
    ],
    hints: [
      'Use BFS for shortest path in unweighted grid',
      'Check all 8 directions (including diagonals)',
      'Mark cells as visited to avoid revisiting',
      'Track distance/path length as you go',
      'First arrival at destination is shortest',
    ],
    starterCode: `from typing import List
from collections import deque

def shortest_path_binary_matrix(grid: List[List[int]]) -> int:
    """
    Find shortest clear path in binary matrix.
    
    Args:
        grid: n x n binary matrix (0 = clear, 1 = blocked)
        
    Returns:
        Length of shortest path, or -1 if no path exists
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [0, 1],
            [1, 0],
          ],
        ],
        expected: 2,
      },
      {
        input: [
          [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
          ],
        ],
        expected: 4,
      },
      {
        input: [
          [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
          ],
        ],
        expected: -1,
      },
    ],
    solution: `from typing import List
from collections import deque


def shortest_path_binary_matrix(grid: List[List[int]]) -> int:
    """
    BFS solution for shortest path in 8-directional grid.
    Time: O(N²), Space: O(N²)
    """
    n = len(grid)
    
    # Check if start or end is blocked
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    # Single cell case
    if n == 1:
        return 1
    
    # 8 directions: up, down, left, right, and 4 diagonals
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    visited = {(0, 0)}
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    
    while queue:
        r, c, dist = queue.popleft()
        
        # Check all 8 directions
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check if reached destination
            if nr == n - 1 and nc == n - 1:
                return dist + 1
            
            # Check bounds and if cell is valid
            if (0 <= nr < n and 0 <= nc < n and 
                grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    
    return -1  # No path found


# Alternative: Modify grid in-place to mark visited
def shortest_path_inplace(grid: List[List[int]]) -> int:
    """
    Mark visited cells in grid itself (more space efficient).
    Time: O(N²), Space: O(N²) for queue only
    """
    n = len(grid)
    
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    if n == 1:
        return 1
    
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    queue = deque([(0, 0, 1)])
    grid[0][0] = 1  # Mark as visited
    
    while queue:
        r, c, dist = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if nr == n - 1 and nc == n - 1:
                return dist + 1
            
            if (0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0):
                grid[nr][nc] = 1  # Mark visited
                queue.append((nr, nc, dist + 1))
    
    return -1`,
    timeComplexity: 'O(N²)',
    spaceComplexity: 'O(N²) for visited set and queue',

    leetcodeUrl:
      'https://leetcode.com/problems/shortest-path-in-binary-matrix/',
    youtubeUrl: 'https://www.youtube.com/watch?v=caXJJOMLyHk',
    order: 2,
    topic: 'Breadth-First Search (BFS)',
    leetcodeUrl:
      'https://leetcode.com/problems/shortest-path-in-binary-matrix/',
    youtubeUrl: 'https://www.youtube.com/watch?v=caXJJOMLyHk',
  },
  {
    id: 'rotting-oranges',
    title: 'Rotting Oranges',
    difficulty: 'Hard',
    description: `You are given an \`m x n\` grid where each cell can have one of three values:
- \`0\` representing an empty cell
- \`1\` representing a fresh orange
- \`2\` representing a rotten orange

Every minute, any fresh orange that is **4-directionally adjacent** to a rotten orange becomes rotten.

Return the **minimum number of minutes** that must elapse until no cell has a fresh orange. If this is impossible, return \`-1\`.


**Approach:**
Use **Multi-Source BFS** - start from ALL rotten oranges simultaneously:
1. Add all initial rotten oranges to queue
2. Count total fresh oranges
3. BFS: spread rot to adjacent fresh oranges
4. Track minutes elapsed
5. Check if any fresh oranges remain

**Key Insight:**
This is multi-source BFS where rot spreads simultaneously from all sources. Process one "round" of rot per minute.`,
    examples: [
      {
        input: 'grid = [[2,1,1],[1,1,0],[0,1,1]]',
        output: '4',
        explanation:
          'Minute 0: [[2,1,1],[1,1,0],[0,1,1]]\n' +
          'Minute 1: [[2,2,1],[2,1,0],[0,1,1]]\n' +
          'Minute 2: [[2,2,2],[2,2,0],[0,1,1]]\n' +
          'Minute 3: [[2,2,2],[2,2,0],[0,2,1]]\n' +
          'Minute 4: [[2,2,2],[2,2,0],[0,2,2]]',
      },
      {
        input: 'grid = [[2,1,1],[0,1,1],[1,0,1]]',
        output: '-1',
        explanation: 'Orange at bottom left cannot be reached',
      },
      {
        input: 'grid = [[0,2]]',
        output: '0',
        explanation: 'No fresh oranges to rot',
      },
    ],
    constraints: [
      'm == grid.length',
      'n == grid[i].length',
      '1 <= m, n <= 10',
      'grid[i][j] is 0, 1, or 2',
    ],
    hints: [
      'Use multi-source BFS starting from all rotten oranges',
      'Count fresh oranges at the start',
      'Process one level (minute) at a time',
      'Decrease fresh count as oranges rot',
      'Check if any fresh oranges remain at the end',
    ],
    starterCode: `from typing import List
from collections import deque

def oranges_rotting(grid: List[List[int]]) -> int:
    """
    Find minimum minutes for all oranges to rot.
    
    Args:
        grid: m x n grid (0=empty, 1=fresh, 2=rotten)
        
    Returns:
        Minimum minutes, or -1 if impossible
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [2, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
          ],
        ],
        expected: 4,
      },
      {
        input: [
          [
            [2, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
          ],
        ],
        expected: -1,
      },
      {
        input: [[[0, 2]]],
        expected: 0,
      },
    ],
    solution: `from typing import List
from collections import deque


def oranges_rotting(grid: List[List[int]]) -> int:
    """
    Multi-source BFS solution.
    Time: O(M × N), Space: O(M × N)
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh_count += 1
    
    # If no fresh oranges, return 0
    if fresh_count == 0:
        return 0
    
    minutes = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # BFS - process level by level (minute by minute)
    while queue:
        level_size = len(queue)
        
        # Process all oranges that rot in this minute
        for _ in range(level_size):
            r, c = queue.popleft()
            
            # Try to rot adjacent fresh oranges
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == 1):
                    grid[nr][nc] = 2  # Rot it
                    fresh_count -= 1
                    queue.append((nr, nc))
        
        # Increment minutes only if we processed some oranges
        if queue:
            minutes += 1
    
    # Check if any fresh oranges remain
    return minutes if fresh_count == 0 else -1


# Alternative: Track minutes per cell
def oranges_rotting_alt(grid: List[List[int]]) -> int:
    """
    Alternative: track minutes with each cell.
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, minute)
            elif grid[r][c] == 1:
                fresh_count += 1
    
    if fresh_count == 0:
        return 0
    
    max_minutes = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        r, c, minute = queue.popleft()
        max_minutes = max(max_minutes, minute)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                grid[nr][nc] == 1):
                grid[nr][nc] = 2
                fresh_count -= 1
                queue.append((nr, nc, minute + 1))
    
    return max_minutes if fresh_count == 0 else -1`,
    timeComplexity: 'O(M × N)',
    spaceComplexity: 'O(M × N)',

    leetcodeUrl: 'https://leetcode.com/problems/rotting-oranges/',
    youtubeUrl: 'https://www.youtube.com/watch?v=y704fEOx0s0',
    order: 3,
    topic: 'Breadth-First Search (BFS)',
    leetcodeUrl: 'https://leetcode.com/problems/rotting-oranges/',
    youtubeUrl: 'https://www.youtube.com/watch?v=y704fEOx0s0',
  },

  // EASY - N-ary Tree Level Order Traversal
  {
    id: 'n-ary-tree-level-order',
    title: 'N-ary Tree Level Order Traversal',
    difficulty: 'Easy',
    topic: 'Breadth-First Search (BFS)',
    description: `Given an n-ary tree, return the level order traversal of its nodes' values.

*Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value.*`,
    examples: [
      {
        input: 'root = [1,null,3,2,4,null,5,6]',
        output: '[[1],[3,2,4],[5,6]]',
      },
      {
        input:
          'root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]',
        output: '[[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]',
      },
    ],
    constraints: [
      'The height of the n-ary tree is less than or equal to 1000',
      'The total number of nodes is between [0, 10^4]',
    ],
    hints: [
      'Similar to binary tree level order',
      'Each node has list of children',
    ],
    starterCode: `from typing import List
from collections import deque

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

def level_order_n_ary(root: Node) -> List[List[int]]:
    """
    Level order traversal of n-ary tree.
    
    Args:
        root: Root of n-ary tree
        
    Returns:
        List of levels
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, null, 3, 2, 4, null, 5, 6]],
        expected: [[1], [3, 2, 4], [5, 6]],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl:
      'https://leetcode.com/problems/n-ary-tree-level-order-traversal/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ZfgOYcJcFwg',
  },

  // EASY - Average of Levels in Binary Tree
  {
    id: 'average-of-levels',
    title: 'Average of Levels in Binary Tree',
    difficulty: 'Easy',
    topic: 'Breadth-First Search (BFS)',
    description: `Given the \`root\` of a binary tree, return the average value of the nodes on each level in the form of an array. Answers within \`10^-5\` of the actual answer will be accepted.`,
    examples: [
      {
        input: 'root = [3,9,20,null,null,15,7]',
        output: '[3.00000,14.50000,11.00000]',
        explanation: 'Level 0: 3, Level 1: (9+20)/2=14.5, Level 2: (15+7)/2=11',
      },
      {
        input: 'root = [3,9,20,15,7]',
        output: '[3.00000,14.50000,11.00000]',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [1, 10^4]',
      '-2^31 <= Node.val <= 2^31 - 1',
    ],
    hints: ['Use BFS level by level', 'Calculate average for each level'],
    starterCode: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def average_of_levels(root: Optional[TreeNode]) -> List[float]:
    """
    Calculate average value at each level.
    
    Args:
        root: Root of tree
        
    Returns:
        List of averages per level
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 9, 20, null, null, 15, 7]],
        expected: [3.0, 14.5, 11.0],
      },
      {
        input: [[3, 9, 20, 15, 7]],
        expected: [3.0, 14.5, 11.0],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl:
      'https://leetcode.com/problems/average-of-levels-in-binary-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=3yxZMxG4Vdk',
  },

  // MEDIUM - Word Ladder
  {
    id: 'word-ladder',
    title: 'Word Ladder',
    difficulty: 'Medium',
    topic: 'Breadth-First Search (BFS)',
    description: `A **transformation sequence** from word \`beginWord\` to word \`endWord\` using a dictionary \`wordList\` is a sequence of words \`beginWord -> s1 -> s2 -> ... -> sk\` such that:

- Every adjacent pair of words differs by a single letter.
- Every \`si\` for \`1 <= i <= k\` is in \`wordList\`. Note that \`beginWord\` does not need to be in \`wordList\`.
- \`sk == endWord\`

Given two words, \`beginWord\` and \`endWord\`, and a dictionary \`wordList\`, return the **number of words** in the **shortest transformation sequence** from \`beginWord\` to \`endWord\`, or \`0\` if no such sequence exists.`,
    examples: [
      {
        input:
          'beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]',
        output: '5',
        explanation:
          'One shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog"',
      },
      {
        input:
          'beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]',
        output: '0',
        explanation: 'endWord "cog" is not in wordList',
      },
    ],
    constraints: [
      '1 <= beginWord.length <= 10',
      'endWord.length == beginWord.length',
      '1 <= wordList.length <= 5000',
      'wordList[i].length == beginWord.length',
      'beginWord, endWord, and wordList[i] consist of lowercase English letters',
      'beginWord != endWord',
      'All strings in wordList are unique',
    ],
    hints: [
      'Use BFS for shortest path',
      'Try changing each letter',
      'Use set for O(1) word lookup',
    ],
    starterCode: `from typing import List
from collections import deque

def ladder_length(begin_word: str, end_word: str, word_list: List[str]) -> int:
    """
    Find shortest transformation sequence length.
    
    Args:
        begin_word: Starting word
        end_word: Target word
        word_list: Dictionary of valid words
        
    Returns:
        Length of shortest sequence
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']],
        expected: 5,
      },
      {
        input: ['hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log']],
        expected: 0,
      },
    ],
    timeComplexity: 'O(M^2 * N) where M is word length, N is word list size',
    spaceComplexity: 'O(M * N)',
    leetcodeUrl: 'https://leetcode.com/problems/word-ladder/',
    youtubeUrl: 'https://www.youtube.com/watch?v=h9iTnkgv05E',
  },

  // MEDIUM - Open the Lock
  {
    id: 'open-the-lock',
    title: 'Open the Lock',
    difficulty: 'Medium',
    topic: 'Breadth-First Search (BFS)',
    description: `You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: \`'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'\`. The wheels can rotate freely and wrap around: for example we can turn \`'9'\` to be \`'0'\`, or \`'0'\` to be \`'9'\`. Each move consists of turning one wheel one slot.

The lock initially starts at \`'0000'\`, a string representing the state of the 4 wheels.

You are given a list of \`deadends\` dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a \`target\` representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.`,
    examples: [
      {
        input:
          'deadends = ["0201","0101","0102","1212","2002"], target = "0202"',
        output: '6',
        explanation:
          'A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".',
      },
      {
        input: 'deadends = ["8888"], target = "0009"',
        output: '1',
      },
    ],
    constraints: [
      '1 <= deadends.length <= 500',
      'deadends[i].length == 4',
      'target.length == 4',
      'target will not be in the list deadends',
      'target and deadends[i] consist of digits only',
    ],
    hints: [
      'BFS from "0000"',
      'For each position, try +1 and -1',
      'Skip deadends',
    ],
    starterCode: `from typing import List
from collections import deque

def open_lock(deadends: List[str], target: str) -> int:
    """
    Find minimum turns to open lock.
    
    Args:
        deadends: Forbidden combinations
        target: Target combination
        
    Returns:
        Minimum turns or -1
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['0201', '0101', '0102', '1212', '2002'], '0202'],
        expected: 6,
      },
      {
        input: [['8888'], '0009'],
        expected: 1,
      },
    ],
    timeComplexity: 'O(10^4)',
    spaceComplexity: 'O(10^4)',
    leetcodeUrl: 'https://leetcode.com/problems/open-the-lock/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Pzg3bCDY87w',
  },

  // MEDIUM - Minimum Genetic Mutation
  {
    id: 'minimum-genetic-mutation',
    title: 'Minimum Genetic Mutation',
    difficulty: 'Medium',
    topic: 'Breadth-First Search (BFS)',
    description: `A gene string can be represented by an 8-character long string, with choices from \`'A'\`, \`'C'\`, \`'G'\`, and \`'T'\`.

Suppose we need to investigate a mutation from a gene string \`startGene\` to a gene string \`endGene\` where one mutation is defined as one single character changed in the gene string.

- For example, \`"AACCGGTT" --> "AACCGGTA"\` is one mutation.

There is also a gene bank \`bank\` that records all the valid gene mutations. A gene must be in \`bank\` to make it a valid gene string.

Given the two gene strings \`startGene\` and \`endGene\` and the gene bank \`bank\`, return the minimum number of mutations needed to mutate from \`startGene\` to \`endGene\`. If there is no such a mutation, return \`-1\`.

Note that the starting point is assumed to be valid, so it might not be included in the bank.`,
    examples: [
      {
        input:
          'startGene = "AACCGGTT", endGene = "AACCGGTA", bank = ["AACCGGTA"]',
        output: '1',
      },
      {
        input:
          'startGene = "AACCGGTT", endGene = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]',
        output: '2',
      },
    ],
    constraints: [
      '0 <= bank.length <= 10',
      'startGene.length == endGene.length == bank[i].length == 8',
      "startGene, endGene, and bank[i] consist of only the characters ['A', 'C', 'G', 'T']",
    ],
    hints: [
      'BFS from startGene',
      'Try changing each position to A, C, G, T',
      'Only proceed if mutation in bank',
    ],
    starterCode: `from typing import List
from collections import deque

def min_mutation(start_gene: str, end_gene: str, bank: List[str]) -> int:
    """
    Find minimum mutations to reach end gene.
    
    Args:
        start_gene: Starting gene sequence
        end_gene: Target gene sequence
        bank: Valid mutations
        
    Returns:
        Minimum mutations or -1
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['AACCGGTT', 'AACCGGTA', ['AACCGGTA']],
        expected: 1,
      },
      {
        input: ['AACCGGTT', 'AAACGGTA', ['AACCGGTA', 'AACCGCTA', 'AAACGGTA']],
        expected: 2,
      },
    ],
    timeComplexity: 'O(B * 8 * 4) where B is bank size',
    spaceComplexity: 'O(B)',
    leetcodeUrl: 'https://leetcode.com/problems/minimum-genetic-mutation/',
    youtubeUrl: 'https://www.youtube.com/watch?v=rZUv6R7Q8Mo',
  },
];
