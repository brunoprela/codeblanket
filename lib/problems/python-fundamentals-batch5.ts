/**
 * Python Fundamentals - Batch 5 (Problems 71-80)
 * New unique problems for Python fundamentals
 */

import { Problem } from '../types';

export const pythonFundamentalsBatch5: Problem[] = [
  {
    id: 'fundamentals-find-peak',
    title: 'Find Peak Element',
    difficulty: 'Medium',
    description: `Find a peak element in an array.

A peak element is greater than its neighbors.
- arr[i] > arr[i-1] and arr[i] > arr[i+1]
- Edge elements only compare with one neighbor

**Example:** [1,2,3,1] → peak at index 2 (value 3)

This tests:
- Array traversal
- Binary search (advanced)
- Neighbor comparison`,
    examples: [
      {
        input: 'nums = [1,2,3,1]',
        output: '2',
        explanation: 'Peak at index 2',
      },
      {
        input: 'nums = [1,2,1,3,5,6,4]',
        output: '5',
        explanation: 'Peak at index 5',
      },
    ],
    constraints: ['1 <= len(nums) <= 1000'],
    hints: [
      'Linear scan is O(n)',
      'Binary search is O(log n)',
      'Compare with next element',
    ],
    starterCode: `def find_peak_element(nums):
    """
    Find a peak element index.
    
    Args:
        nums: Array of integers
        
    Returns:
        Index of any peak element
        
    Examples:
        >>> find_peak_element([1,2,3,1])
        2
    """
    pass


# Test
print(find_peak_element([1,2,3,1]))
`,
    testCases: [
      {
        input: [[1, 2, 3, 1]],
        expected: 2,
      },
      {
        input: [[1, 2, 1, 3, 5, 6, 4]],
        expected: 5,
      },
    ],
    solution: `def find_peak_element(nums):
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            return i
    return len(nums) - 1


# Binary search O(log n) solution
def find_peak_element_binary(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left`,
    timeComplexity: 'O(n) or O(log n) with binary search',
    spaceComplexity: 'O(1)',
    order: 71,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-split-array-largest-sum',
    title: 'Minimize Maximum Sum',
    difficulty: 'Hard',
    description: `Split array into k subarrays to minimize the maximum sum.

**Example:** nums = [7,2,5,10,8], k = 2
Split [7,2,5] and [10,8] → max sum is 18

This tests:
- Binary search on answer
- Array splitting
- Greedy validation`,
    examples: [
      {
        input: 'nums = [7,2,5,10,8], k = 2',
        output: '18',
      },
    ],
    constraints: ['1 <= k <= len(nums) <= 1000', '0 <= nums[i] <= 10^6'],
    hints: [
      'Binary search on the answer',
      'Check if split with max_sum is valid',
      'Count required subarrays',
    ],
    starterCode: `def split_array(nums, k):
    """
    Minimize maximum subarray sum when split into k parts.
    
    Args:
        nums: Array of integers
        k: Number of subarrays
        
    Returns:
        Minimized maximum sum
        
    Examples:
        >>> split_array([7,2,5,10,8], 2)
        18
    """
    pass


# Test
print(split_array([7,2,5,10,8], 2))
`,
    testCases: [
      {
        input: [[7, 2, 5, 10, 8], 2],
        expected: 18,
      },
      {
        input: [[1, 2, 3, 4, 5], 2],
        expected: 9,
      },
    ],
    solution: `def split_array(nums, k):
    def can_split(max_sum):
        count = 1
        current_sum = 0
        
        for num in nums:
            if current_sum + num > max_sum:
                count += 1
                current_sum = num
            else:
                current_sum += num
        
        return count <= k
    
    left, right = max(nums), sum(nums)
    
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    
    return left`,
    timeComplexity: 'O(n log S) where S is sum',
    spaceComplexity: 'O(1)',
    order: 72,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-next-permutation',
    title: 'Next Permutation',
    difficulty: 'Medium',
    description: `Find the next lexicographic permutation of an array.

If no next permutation exists, rearrange to lowest order (sorted).

**Example:** [1,2,3] → [1,3,2], [3,2,1] → [1,2,3]

Algorithm:
1. Find rightmost ascending pair
2. Swap with next larger element
3. Reverse suffix

This tests:
- Array manipulation
- Permutation logic
- In-place modification`,
    examples: [
      {
        input: 'nums = [1,2,3]',
        output: '[1,3,2]',
      },
      {
        input: 'nums = [3,2,1]',
        output: '[1,2,3]',
      },
    ],
    constraints: ['1 <= len(nums) <= 100', '0 <= nums[i] <= 100'],
    hints: [
      'Find rightmost i where nums[i] < nums[i+1]',
      'Find rightmost j > i where nums[j] > nums[i]',
      'Swap and reverse suffix',
    ],
    starterCode: `def next_permutation(nums):
    """
    Modify array to next permutation in-place.
    
    Args:
        nums: Array to permute (modified in-place)
        
    Returns:
        None (modifies nums)
        
    Examples:
        >>> nums = [1,2,3]
        >>> next_permutation(nums)
        >>> nums
        [1, 3, 2]
    """
    pass


# Test
nums = [1,2,3]
next_permutation(nums)
print(nums)
`,
    testCases: [
      {
        input: [[1, 2, 3]],
        expected: [1, 3, 2],
      },
      {
        input: [[3, 2, 1]],
        expected: [1, 2, 3],
      },
    ],
    solution: `def next_permutation(nums):
    # Find rightmost ascending pair
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        # Find rightmost element > nums[i]
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    
    # Reverse suffix
    nums[i + 1:] = reversed(nums[i + 1:])`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 73,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-arrange-coins',
    title: 'Arranging Coins',
    difficulty: 'Easy',
    description: `You have n coins to form a staircase.

Each row k must have exactly k coins.
Find complete rows you can form.

**Example:** n = 5 coins
Row 1: 1 coin, Row 2: 2 coins, Row 3: needs 3 but only 2 left
→ 2 complete rows

**Formula:** k(k+1)/2 ≤ n, find max k

This tests:
- Mathematical formula
- Binary search or math
- Integer arithmetic`,
    examples: [
      {
        input: 'n = 5',
        output: '2',
        explanation: 'Rows 1 and 2',
      },
      {
        input: 'n = 8',
        output: '3',
        explanation: 'Rows 1, 2, and 3',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1'],
    hints: [
      'Sum of 1 to k is k(k+1)/2',
      'Use binary search for O(log n)',
      'Or use quadratic formula',
    ],
    starterCode: `def arrange_coins(n):
    """
    Find number of complete staircase rows.
    
    Args:
        n: Number of coins
        
    Returns:
        Number of complete rows
        
    Examples:
        >>> arrange_coins(5)
        2
    """
    pass


# Test
print(arrange_coins(5))
`,
    testCases: [
      {
        input: [5],
        expected: 2,
      },
      {
        input: [8],
        expected: 3,
      },
      {
        input: [1],
        expected: 1,
      },
    ],
    solution: `def arrange_coins(n):
    left, right = 0, n
    
    while left <= right:
        mid = (left + right) // 2
        curr = mid * (mid + 1) // 2
        
        if curr == n:
            return mid
        elif curr < n:
            left = mid + 1
        else:
            right = mid - 1
    
    return right


# Using quadratic formula
import math

def arrange_coins_math(n):
    return int((-1 + math.sqrt(1 + 8 * n)) / 2)`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    order: 74,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-find-difference',
    title: 'Find the Difference',
    difficulty: 'Easy',
    description: `Two strings s and t, where t is s with one added letter.

Find the added letter.

**Example:** s = "abcd", t = "abcde" → "e"

This tests:
- Character frequency
- XOR trick
- Character operations`,
    examples: [
      {
        input: 's = "abcd", t = "abcde"',
        output: '"e"',
      },
      {
        input: 's = "", t = "y"',
        output: '"y"',
      },
    ],
    constraints: ['0 <= len(s) <= 1000', 'Only lowercase letters'],
    hints: [
      'Count characters in both',
      'Or use XOR (a ^ a = 0)',
      'Sum of ASCII values',
    ],
    starterCode: `def find_the_difference(s, t):
    """
    Find the added letter.
    
    Args:
        s: Original string
        t: String with one extra letter
        
    Returns:
        The added character
        
    Examples:
        >>> find_the_difference("abcd", "abcde")
        "e"
    """
    pass


# Test
print(find_the_difference("abcd", "abcde"))
`,
    testCases: [
      {
        input: ['abcd', 'abcde'],
        expected: 'e',
      },
      {
        input: ['', 'y'],
        expected: 'y',
      },
    ],
    solution: `def find_the_difference(s, t):
    from collections import Counter
    
    s_count = Counter(s)
    t_count = Counter(t)
    
    for char in t_count:
        if t_count[char] > s_count[char]:
            return char
    
    return ''


# XOR solution
def find_the_difference_xor(s, t):
    result = 0
    for char in s + t:
        result ^= ord(char)
    return chr(result)


# ASCII sum solution
def find_the_difference_sum(s, t):
    return chr(sum(ord(c) for c in t) - sum(ord(c) for c in s))`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 75,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-third-maximum',
    title: 'Third Maximum Number',
    difficulty: 'Easy',
    description: `Find the third distinct maximum number in array.

If third max doesn't exist, return the maximum.

**Example:** [3,2,1] → 1 (third max)
[1,2] → 2 (only 2 distinct, return max)
[2,2,3,1] → 1 (third distinct max)

This tests:
- Set operations
- Sorting
- Edge case handling`,
    examples: [
      {
        input: 'nums = [3,2,1]',
        output: '1',
      },
      {
        input: 'nums = [1,2]',
        output: '2',
      },
      {
        input: 'nums = [2,2,3,1]',
        output: '1',
      },
    ],
    constraints: ['1 <= len(nums) <= 10^4'],
    hints: [
      'Remove duplicates with set',
      'Sort descending',
      'Return 3rd if exists, else max',
    ],
    starterCode: `def third_max(nums):
    """
    Find third distinct maximum.
    
    Args:
        nums: Array of integers
        
    Returns:
        Third max or max if < 3 distinct
        
    Examples:
        >>> third_max([3,2,1])
        1
    """
    pass


# Test
print(third_max([2,2,3,1]))
`,
    testCases: [
      {
        input: [[3, 2, 1]],
        expected: 1,
      },
      {
        input: [[1, 2]],
        expected: 2,
      },
      {
        input: [[2, 2, 3, 1]],
        expected: 1,
      },
    ],
    solution: `def third_max(nums):
    distinct = list(set(nums))
    distinct.sort(reverse=True)
    
    if len(distinct) >= 3:
        return distinct[2]
    return distinct[0]


# Alternative tracking top 3
def third_max_tracking(nums):
    top3 = [float('-inf')] * 3
    
    for num in nums:
        if num in top3:
            continue
        if num > top3[0]:
            top3 = [num, top3[0], top3[1]]
        elif num > top3[1]:
            top3 = [top3[0], num, top3[1]]
        elif num > top3[2]:
            top3 = [top3[0], top3[1], num]
    
    return top3[2] if top3[2] != float('-inf') else top3[0]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 76,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-sum-of-two-integers',
    title: 'Sum Without + Operator',
    difficulty: 'Medium',
    description: `Calculate sum of two integers without using + or - operators.

Use bit manipulation instead.

**Key insight:**
- XOR gives sum without carry
- AND gives carry positions
- Shift carry left and repeat

This tests:
- Bit manipulation
- Carry calculation
- Iterative bit operations`,
    examples: [
      {
        input: 'a = 1, b = 2',
        output: '3',
      },
      {
        input: 'a = 2, b = 3',
        output: '5',
      },
    ],
    constraints: ['-1000 <= a, b <= 1000'],
    hints: [
      'XOR for sum without carry',
      'AND << 1 for carry',
      'Repeat until no carry',
    ],
    starterCode: `def get_sum(a, b):
    """
    Add two integers without + operator.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Sum of a and b
        
    Examples:
        >>> get_sum(1, 2)
        3
    """
    pass


# Test
print(get_sum(1, 2))
`,
    testCases: [
      {
        input: [1, 2],
        expected: 3,
      },
      {
        input: [2, 3],
        expected: 5,
      },
    ],
    solution: `def get_sum(a, b):
    # 32-bit integer limit
    mask = 0xFFFFFFFF
    
    while b != 0:
        # XOR: sum without carry
        # AND << 1: carry
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask
    
    # Handle negative numbers
    return a if a <= 0x7FFFFFFF else ~(a ^ mask)


# Simpler for Python (no 32-bit limit)
def get_sum_simple(a, b):
    while b != 0:
        carry = a & b
        a = a ^ b
        b = carry << 1
    return a`,
    timeComplexity: 'O(1) - at most 32 iterations',
    spaceComplexity: 'O(1)',
    order: 77,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-convert-sorted-array-to-bst',
    title: 'Sorted Array to BST',
    difficulty: 'Easy',
    description: `Convert a sorted array to a height-balanced Binary Search Tree.

Height-balanced: depth of two subtrees differ by at most 1.

**Strategy:** Use middle element as root, recursively build left/right.

**Note:** This is a simplified version that returns array representation.

This tests:
- Recursion
- Array slicing
- Tree construction logic`,
    examples: [
      {
        input: 'nums = [-10,-3,0,5,9]',
        output: '[0,-3,9,-10,null,5]',
        explanation: 'Middle element as root',
      },
    ],
    constraints: ['1 <= len(nums) <= 10^4', 'Sorted in ascending order'],
    hints: [
      'Use middle element as root',
      'Recursively build left/right subtrees',
      'Base case: empty array',
    ],
    starterCode: `def sorted_array_to_bst(nums):
    """
    Convert sorted array to balanced BST.
    Returns array representation [root, left, right, ...]
    
    Args:
        nums: Sorted array
        
    Returns:
        Root value of balanced BST
        
    Examples:
        >>> sorted_array_to_bst([-10,-3,0,5,9])
        0
    """
    pass


# Test
print(sorted_array_to_bst([-10,-3,0,5,9]))
`,
    testCases: [
      {
        input: [[-10, -3, 0, 5, 9]],
        expected: 0,
      },
      {
        input: [[1, 3]],
        expected: 3,
      },
    ],
    solution: `def sorted_array_to_bst(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    
    # For this simplified version, return root value
    # In full implementation, would create TreeNode
    return nums[mid]


# Full recursive structure
def sorted_array_to_bst_recursive(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    
    root = {
        'val': nums[mid],
        'left': sorted_array_to_bst_recursive(nums[:mid]),
        'right': sorted_array_to_bst_recursive(nums[mid + 1:])
    }
    
    return root`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(log n) recursion stack',
    order: 78,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-minimum-depth-tree',
    title: 'Minimum Depth of Binary Tree',
    difficulty: 'Easy',
    description: `Find minimum depth of a binary tree.

Minimum depth = shortest path from root to a leaf node.

**Input Format:** Array representation [root, left, right, ...]
null represents missing node.

This tests:
- Tree traversal
- BFS or DFS
- Base case handling`,
    examples: [
      {
        input: 'root = [3,9,20,null,null,15,7]',
        output: '2',
        explanation: 'Path: 3 → 9',
      },
    ],
    constraints: ['0 <= number of nodes <= 10^5'],
    hints: [
      'Use BFS for level-order traversal',
      'Return depth when leaf found',
      'Leaf = both children are null',
    ],
    starterCode: `def min_depth(tree_array):
    """
    Find minimum depth of binary tree.
    
    Args:
        tree_array: Array representation of tree
        
    Returns:
        Minimum depth
        
    Examples:
        >>> min_depth([3,9,20,None,None,15,7])
        2
    """
    pass


# Test
print(min_depth([3,9,20,None,None,15,7]))
`,
    testCases: [
      {
        input: [[3, 9, 20, null, null, 15, 7]],
        expected: 2,
      },
      {
        input: [[2, null, 3, null, 4, null, 5, null, 6]],
        expected: 5,
      },
    ],
    solution: `def min_depth(tree_array):
    if not tree_array or tree_array[0] is None:
        return 0
    
    # BFS approach
    from collections import deque
    queue = deque([(0, 1)])  # (index, depth)
    
    while queue:
        idx, depth = queue.popleft()
        
        if idx >= len(tree_array) or tree_array[idx] is None:
            continue
        
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        
        # Check if leaf
        left_null = left_idx >= len(tree_array) or tree_array[left_idx] is None
        right_null = right_idx >= len(tree_array) or tree_array[right_idx] is None
        
        if left_null and right_null:
            return depth
        
        if not left_null:
            queue.append((left_idx, depth + 1))
        if not right_null:
            queue.append((right_idx, depth + 1))
    
    return 0`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 79,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-path-sum',
    title: 'Path Sum',
    difficulty: 'Easy',
    description: `Check if tree has root-to-leaf path with given sum.

Path sum = sum of all node values along the path.

**Input:** Array representation and target sum

This tests:
- Tree traversal
- Path tracking
- Sum calculation`,
    examples: [
      {
        input: 'root = [5,4,8,11,null,13,4,7,2], targetSum = 22',
        output: 'True',
        explanation: 'Path: 5→4→11→2 = 22',
      },
    ],
    constraints: ['0 <= number of nodes <= 5000', '-1000 <= Node.val <= 1000'],
    hints: [
      'Use DFS with running sum',
      'Check sum at leaf nodes',
      'Subtract current value from target',
    ],
    starterCode: `def has_path_sum(tree_array, target_sum):
    """
    Check if path exists with given sum.
    
    Args:
        tree_array: Array representation
        target_sum: Target sum value
        
    Returns:
        True if path exists
        
    Examples:
        >>> has_path_sum([5,4,8,11,None,13,4,7,2], 22)
        True
    """
    pass


# Test
print(has_path_sum([5,4,8,11,None,13,4,7,2,None,None,None,1], 22))
`,
    testCases: [
      {
        input: [[5, 4, 8, 11, null, 13, 4, 7, 2, null, null, null, 1], 22],
        expected: true,
      },
      {
        input: [[1, 2, 3], 5],
        expected: false,
      },
    ],
    solution: `def has_path_sum(tree_array, target_sum):
    if not tree_array or tree_array[0] is None:
        return False
    
    def dfs(idx, current_sum):
        if idx >= len(tree_array) or tree_array[idx] is None:
            return False
        
        current_sum += tree_array[idx]
        
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        
        # Check if leaf
        left_null = left_idx >= len(tree_array) or tree_array[left_idx] is None
        right_null = right_idx >= len(tree_array) or tree_array[right_idx] is None
        
        if left_null and right_null:
            return current_sum == target_sum
        
        return dfs(left_idx, current_sum) or dfs(right_idx, current_sum)
    
    return dfs(0, 0)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h) where h is height',
    order: 80,
    topic: 'Python Fundamentals',
  },
];
