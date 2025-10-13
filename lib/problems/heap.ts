import { Problem } from '../types';

export const heapProblems: Problem[] = [
  // EASY - Last Stone Weight
  {
    id: 'last-stone-weight',
    title: 'Last Stone Weight',
    difficulty: 'Easy',
    topic: 'Heap / Priority Queue',
    description: `You are given an array of integers \`stones\` where \`stones[i]\` is the weight of the \`i-th\` stone.

We are playing a game with the stones. On each turn, we choose the **heaviest two stones** and smash them together. Suppose the heaviest two stones have weights \`x\` and \`y\` with \`x <= y\`. The result of this smash is:

- If \`x == y\`, both stones are destroyed, and
- If \`x != y\`, the stone of weight \`x\` is destroyed, and the stone of weight \`y\` has new weight \`y - x\`.

At the end of the game, there is **at most one** stone left.

Return the weight of the last remaining stone. If there are no stones left, return \`0\`.`,
    examples: [
      {
        input: 'stones = [2,7,4,1,8,1]',
        output: '1',
        explanation:
          'Combine 7 and 8 to get 1, then 4 and 1 to get 3, then 2 and 3 to get 1, then 1 and 1 to get 0, return 1.',
      },
      {
        input: 'stones = [1]',
        output: '1',
      },
    ],
    constraints: ['1 <= stones.length <= 30', '1 <= stones[i] <= 1000'],
    hints: [
      'Use a max heap',
      'Pop two largest stones, push difference if not equal',
    ],
    starterCode: `from typing import List
import heapq

def last_stone_weight(stones: List[int]) -> int:
    """
    Find weight of last remaining stone.
    
    Args:
        stones: Array of stone weights
        
    Returns:
        Weight of last stone or 0
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 7, 4, 1, 8, 1]],
        expected: 1,
      },
      {
        input: [[1]],
        expected: 1,
      },
      {
        input: [[2, 2]],
        expected: 0,
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    
    leetcodeUrl: 'https://leetcode.com/problems/last-stone-weight/',
    youtubeUrl: 'https://www.youtube.com/watch?v=B-QCq79-Vfw',
    leetcodeUrl: 'https://leetcode.com/problems/last-stone-weight/',
    youtubeUrl: 'https://www.youtube.com/watch?v=B-QCq79-Vfw',
  },

  // EASY - Kth Largest Element in a Stream
  {
    id: 'kth-largest-element-stream',
    title: 'Kth Largest Element in a Stream',
    difficulty: 'Easy',
    topic: 'Heap / Priority Queue',
    description: `Design a class to find the \`k-th\` largest element in a stream. Note that it is the \`k-th\` largest element in the sorted order, not the \`k-th\` distinct element.

Implement \`KthLargest\` class:
- \`KthLargest(int k, int[] nums)\` Initializes the object with the integer \`k\` and the stream of integers \`nums\`.
- \`int add(int val)\` Appends the integer \`val\` to the stream and returns the element representing the \`k-th\` largest element in the stream.`,
    examples: [
      {
        input:
          '["KthLargest", "add", "add", "add", "add", "add"], [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]',
        output: '[null, 4, 5, 5, 8, 8]',
      },
    ],
    constraints: [
      '1 <= k <= 10^4',
      '0 <= nums.length <= 10^4',
      '-10^4 <= nums[i] <= 10^4',
      '-10^4 <= val <= 10^4',
      'At most 10^4 calls will be made to add',
    ],
    hints: [
      'Use a min heap of size k',
      'Always maintain k largest elements',
      'Top of heap is kth largest',
    ],
    starterCode: `from typing import List
import heapq

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        """
        Initialize with k and initial stream.
        
        Args:
            k: Position of element to track
            nums: Initial stream
        """
        # Write your code here
        pass
    
    def add(self, val: int) -> int:
        """
        Add value to stream and return kth largest.
        
        Args:
            val: Value to add
            
        Returns:
            Kth largest element
        """
        # Write your code here
        pass
`,
    testCases: [
      {
        input: [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]],
        expected: [null, 4, 5, 5, 8, 8],
      },
    ],
    timeComplexity: 'O(log k) for add',
    spaceComplexity: 'O(k)',
    leetcodeUrl:
      'https://leetcode.com/problems/kth-largest-element-in-a-stream/',
    youtubeUrl: 'https://www.youtube.com/watch?v=hOjcdrqMoQ8',
  },

  // EASY - Relative Ranks
  {
    id: 'relative-ranks',
    title: 'Relative Ranks',
    difficulty: 'Easy',
    topic: 'Heap / Priority Queue',
    description: `You are given an integer array \`score\` of size \`n\`, where \`score[i]\` is the score of the \`i-th\` athlete in a competition. All the scores are guaranteed to be **unique**.

The athletes are **placed** based on their scores, where the \`1st\` place athlete has the highest score, the \`2nd\` place athlete has the \`2nd\` highest score, and so on. The placement of each athlete determines their rank:

- The \`1st\` place athlete's rank is \`"Gold Medal"\`.
- The \`2nd\` place athlete's rank is \`"Silver Medal"\`.
- The \`3rd\` place athlete's rank is \`"Bronze Medal"\`.
- For the \`4th\` place to the \`n-th\` place athlete, their rank is their placement number (i.e., the \`x-th\` place athlete's rank is \`"x"\`).

Return an array \`answer\` of size \`n\` where \`answer[i]\` is the **rank** of the \`i-th\` athlete.`,
    examples: [
      {
        input: 'score = [5,4,3,2,1]',
        output: '["Gold Medal","Silver Medal","Bronze Medal","4","5"]',
      },
      {
        input: 'score = [10,3,8,9,4]',
        output: '["Gold Medal","5","Bronze Medal","Silver Medal","4"]',
      },
    ],
    constraints: [
      'n == score.length',
      '1 <= n <= 10^4',
      '0 <= score[i] <= 10^6',
      'All the values in score are unique',
    ],
    hints: [
      'Create pairs of (score, index)',
      'Sort by score descending',
      'Assign ranks based on sorted order',
    ],
    starterCode: `from typing import List

def find_relative_ranks(score: List[int]) -> List[str]:
    """
    Assign ranks to athletes based on scores.
    
    Args:
        score: Array of athlete scores
        
    Returns:
        Array of rank strings
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[5, 4, 3, 2, 1]],
        expected: ['Gold Medal', 'Silver Medal', 'Bronze Medal', '4', '5'],
      },
      {
        input: [[10, 3, 8, 9, 4]],
        expected: ['Gold Medal', '5', 'Bronze Medal', 'Silver Medal', '4'],
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/relative-ranks/',
    youtubeUrl: 'https://www.youtube.com/watch?v=qFKI9TKXRIs',
  },

  // MEDIUM - Kth Largest Element in an Array
  {
    id: 'kth-largest-element-array',
    title: 'Kth Largest Element in an Array',
    difficulty: 'Medium',
    topic: 'Heap / Priority Queue',
    description: `Given an integer array \`nums\` and an integer \`k\`, return the \`k-th\` largest element in the array.

Note that it is the \`k-th\` largest element in the sorted order, not the \`k-th\` distinct element.

Can you solve it without sorting?`,
    examples: [
      {
        input: 'nums = [3,2,1,5,6,4], k = 2',
        output: '5',
      },
      {
        input: 'nums = [3,2,3,1,2,4,5,5,6], k = 4',
        output: '4',
      },
    ],
    constraints: ['1 <= k <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
    hints: [
      'Use a min heap of size k',
      'Maintain k largest elements',
      'Top of heap is kth largest',
    ],
    starterCode: `from typing import List
import heapq

def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Find kth largest element in array.
    
    Args:
        nums: Input array
        k: Position from largest
        
    Returns:
        Kth largest element
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 2, 1, 5, 6, 4], 2],
        expected: 5,
      },
      {
        input: [[3, 2, 3, 1, 2, 4, 5, 5, 6], 4],
        expected: 4,
      },
    ],
    timeComplexity: 'O(n log k)',
    spaceComplexity: 'O(k)',
    leetcodeUrl:
      'https://leetcode.com/problems/kth-largest-element-in-an-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=XEmy13g1Qxc',
  },

  // MEDIUM - K Closest Points to Origin
  {
    id: 'k-closest-points',
    title: 'K Closest Points to Origin',
    difficulty: 'Medium',
    topic: 'Heap / Priority Queue',
    description: `Given an array of \`points\` where \`points[i] = [xi, yi]\` represents a point on the X-Y plane and an integer \`k\`, return the \`k\` closest points to the origin \`(0, 0)\`.

The distance between two points on the X-Y plane is the Euclidean distance (i.e., \`√(x1 - x2)² + (y1 - y2)²\`).

You may return the answer in **any order**. The answer is **guaranteed** to be **unique** (except for the order that it is in).`,
    examples: [
      {
        input: 'points = [[1,3],[-2,2]], k = 1',
        output: '[[-2,2]]',
        explanation:
          'Distance to origin: (1,3) = sqrt(10), (-2,2) = sqrt(8). Closest is (-2,2).',
      },
      {
        input: 'points = [[3,3],[5,-1],[-2,4]], k = 2',
        output: '[[3,3],[-2,4]]',
      },
    ],
    constraints: ['1 <= k <= points.length <= 10^4', '-10^4 <= xi, yi <= 10^4'],
    hints: [
      'Calculate distances from origin',
      'Use max heap of size k',
      'Maintain k smallest distances',
    ],
    starterCode: `from typing import List
import heapq

def k_closest(points: List[List[int]], k: int) -> List[List[int]]:
    """
    Find k closest points to origin.
    
    Args:
        points: Array of [x, y] coordinates
        k: Number of closest points
        
    Returns:
        K closest points
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 3],
            [-2, 2],
          ],
          1,
        ],
        expected: [[-2, 2]],
      },
      {
        input: [
          [
            [3, 3],
            [5, -1],
            [-2, 4],
          ],
          2,
        ],
        expected: [
          [3, 3],
          [-2, 4],
        ],
      },
    ],
    timeComplexity: 'O(n log k)',
    spaceComplexity: 'O(k)',
    leetcodeUrl: 'https://leetcode.com/problems/k-closest-points-to-origin/',
    youtubeUrl: 'https://www.youtube.com/watch?v=rI2EBUEMfTk',
  },

  // MEDIUM - Task Scheduler
  {
    id: 'task-scheduler',
    title: 'Task Scheduler',
    difficulty: 'Medium',
    topic: 'Heap / Priority Queue',
    description: `You are given an array of CPU \`tasks\`, each represented by letters A to Z, and a cooling time, \`n\`. Each cycle or interval allows the completion of one task. Tasks can be completed in any order, but there is a constraint: **identical** tasks must be separated by at least \`n\` intervals due to cooling time.

Return the **minimum number of intervals** required to complete all tasks.`,
    examples: [
      {
        input: 'tasks = ["A","A","A","B","B","B"], n = 2',
        output: '8',
        explanation:
          'A -> B -> idle -> A -> B -> idle -> A -> B. Only 8 units of time are needed.',
      },
      {
        input: 'tasks = ["A","C","A","B","D","B"], n = 1',
        output: '6',
        explanation: 'A -> B -> C -> D -> A -> B. Only 6 units needed.',
      },
    ],
    constraints: [
      '1 <= tasks.length <= 10^4',
      'tasks[i] is an uppercase English letter',
      '0 <= n <= 100',
    ],
    hints: [
      'Count task frequencies',
      'Use max heap to always pick most frequent task',
      'Track cooling time for each task',
    ],
    starterCode: `from typing import List
import heapq
from collections import Counter

def least_interval(tasks: List[str], n: int) -> int:
    """
    Find minimum intervals to complete all tasks.
    
    Args:
        tasks: Array of task letters
        n: Cooling time between same tasks
        
    Returns:
        Minimum number of intervals
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['A', 'A', 'A', 'B', 'B', 'B'], 2],
        expected: 8,
      },
      {
        input: [['A', 'C', 'A', 'B', 'D', 'B'], 1],
        expected: 6,
      },
      {
        input: [['A', 'A', 'A', 'B', 'B', 'B'], 0],
        expected: 6,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/task-scheduler/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s8p8ukTyA2I',
  },
];
