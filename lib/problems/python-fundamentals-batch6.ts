/**
 * Python Fundamentals - Batch 6 (Problems 81-90)
 * New unique problems for Python fundamentals
 */

import { Problem } from '../types';

export const pythonFundamentalsBatch6: Problem[] = [
  {
    id: 'fundamentals-symmetric-tree',
    title: 'Symmetric Tree',
    difficulty: 'Easy',
    description: `Check if a binary tree is symmetric (mirror image of itself).

A tree is symmetric if left subtree is mirror of right subtree.

**Example:** [1,2,2,3,4,4,3] is symmetric

This tests:
- Tree traversal
- Mirror comparison
- Recursion`,
    examples: [
      {
        input: 'root = [1,2,2,3,4,4,3]',
        output: 'True',
      },
      {
        input: 'root = [1,2,2,null,3,null,3]',
        output: 'False',
      },
    ],
    constraints: ['0 <= number of nodes <= 1000'],
    hints: [
      'Compare left and right subtrees',
      'Check if mirror images',
      'Recursively verify symmetry',
    ],
    starterCode: `def is_symmetric(tree_array):
    """
    Check if tree is symmetric.
    
    Args:
        tree_array: Array representation of tree
        
    Returns:
        True if symmetric
        
    Examples:
        >>> is_symmetric([1,2,2,3,4,4,3])
        True
    """
    pass


# Test
print(is_symmetric([1,2,2,3,4,4,3]))
`,
    testCases: [
      {
        input: [[1, 2, 2, 3, 4, 4, 3]],
        expected: true,
      },
      {
        input: [[1, 2, 2, null, 3, null, 3]],
        expected: false,
      },
    ],
    solution: `def is_symmetric(tree_array):
    if not tree_array or tree_array[0] is None:
        return True
    
    def is_mirror(left_idx, right_idx):
        if left_idx >= len(tree_array) and right_idx >= len(tree_array):
            return True
        
        if left_idx >= len(tree_array) or right_idx >= len(tree_array):
            return False
        
        left_val = tree_array[left_idx] if left_idx < len(tree_array) else None
        right_val = tree_array[right_idx] if right_idx < len(tree_array) else None
        
        if left_val != right_val:
            return False
        
        if left_val is None:
            return True
        
        # Compare outer and inner children
        return (is_mirror(2 * left_idx + 1, 2 * right_idx + 2) and
                is_mirror(2 * left_idx + 2, 2 * right_idx + 1))
    
    return is_mirror(1, 2)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(h)',
    order: 81,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-invert-binary-tree',
    title: 'Invert Binary Tree',
    difficulty: 'Easy',
    description: `Invert a binary tree (swap left and right children).

**Example:** [4,2,7,1,3,6,9] → [4,7,2,9,6,3,1]

This tests:
- Tree traversal
- Node swapping
- Recursion or iteration`,
    examples: [
      {
        input: 'root = [4,2,7,1,3,6,9]',
        output: '[4,7,2,9,6,3,1]',
      },
    ],
    constraints: ['0 <= number of nodes <= 100'],
    hints: [
      'Swap left and right children',
      'Recursively invert subtrees',
      'Or use level-order traversal',
    ],
    starterCode: `def invert_tree(tree_array):
    """
    Invert binary tree.
    
    Args:
        tree_array: Array representation
        
    Returns:
        Inverted tree array
        
    Examples:
        >>> invert_tree([4,2,7,1,3,6,9])
        [4, 7, 2, 9, 6, 3, 1]
    """
    pass


# Test
print(invert_tree([4,2,7,1,3,6,9]))
`,
    testCases: [
      {
        input: [[4, 2, 7, 1, 3, 6, 9]],
        expected: [4, 7, 2, 9, 6, 3, 1],
      },
      {
        input: [[2, 1, 3]],
        expected: [2, 3, 1],
      },
    ],
    solution: `def invert_tree(tree_array):
    if not tree_array:
        return []
    
    result = tree_array.copy()
    
    for i in range(len(result)):
        if result[i] is not None:
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            
            # Swap children
            if left_idx < len(result) and right_idx < len(result):
                result[left_idx], result[right_idx] = result[right_idx], result[left_idx]
    
    return result`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 82,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-same-tree',
    title: 'Same Tree',
    difficulty: 'Easy',
    description: `Check if two binary trees are identical.

Trees are same if:
- Same structure
- Same node values

This tests:
- Tree comparison
- Recursive checking
- Base cases`,
    examples: [
      {
        input: 'p = [1,2,3], q = [1,2,3]',
        output: 'True',
      },
      {
        input: 'p = [1,2], q = [1,null,2]',
        output: 'False',
      },
    ],
    constraints: ['0 <= number of nodes <= 100'],
    hints: [
      'Compare root values',
      'Recursively check left/right',
      'Handle null nodes',
    ],
    starterCode: `def is_same_tree(p, q):
    """
    Check if two trees are identical.
    
    Args:
        p: First tree array
        q: Second tree array
        
    Returns:
        True if identical
        
    Examples:
        >>> is_same_tree([1,2,3], [1,2,3])
        True
    """
    pass


# Test
print(is_same_tree([1,2,3], [1,2,3]))
`,
    testCases: [
      {
        input: [
          [1, 2, 3],
          [1, 2, 3],
        ],
        expected: true,
      },
      {
        input: [
          [1, 2],
          [1, null, 2],
        ],
        expected: false,
      },
    ],
    solution: `def is_same_tree(p, q):
    if len(p) != len(q):
        return False
    
    for i in range(len(p)):
        if p[i] != q[i]:
            return False
    
    return True`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 83,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-max-consecutive-ones',
    title: 'Max Consecutive Ones',
    difficulty: 'Easy',
    description: `Find maximum number of consecutive 1s in a binary array.

**Example:** [1,1,0,1,1,1] → 3

This tests:
- Array traversal
- Counting consecutive elements
- Max tracking`,
    examples: [
      {
        input: 'nums = [1,1,0,1,1,1]',
        output: '3',
      },
      {
        input: 'nums = [1,0,1,1,0,1]',
        output: '2',
      },
    ],
    constraints: ['1 <= len(nums) <= 10^5', 'nums[i] is 0 or 1'],
    hints: [
      'Track current consecutive count',
      'Reset count when 0 found',
      'Update max as you go',
    ],
    starterCode: `def find_max_consecutive_ones(nums):
    """
    Find max consecutive 1s.
    
    Args:
        nums: Binary array
        
    Returns:
        Max consecutive ones count
        
    Examples:
        >>> find_max_consecutive_ones([1,1,0,1,1,1])
        3
    """
    pass


# Test
print(find_max_consecutive_ones([1,1,0,1,1,1]))
`,
    testCases: [
      {
        input: [[1, 1, 0, 1, 1, 1]],
        expected: 3,
      },
      {
        input: [[1, 0, 1, 1, 0, 1]],
        expected: 2,
      },
    ],
    solution: `def find_max_consecutive_ones(nums):
    max_count = 0
    current_count = 0
    
    for num in nums:
        if num == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 84,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-hamming-distance',
    title: 'Hamming Distance',
    difficulty: 'Easy',
    description: `Calculate Hamming distance between two integers.

Hamming distance = number of positions with different bits.

**Example:** x=1 (0001), y=4 (0100) → distance = 2

This tests:
- Bit manipulation
- XOR operation
- Bit counting`,
    examples: [
      {
        input: 'x = 1, y = 4',
        output: '2',
      },
    ],
    constraints: ['0 <= x, y <= 2^31 - 1'],
    hints: [
      'XOR gives positions where bits differ',
      'Count 1s in XOR result',
      'Use bit counting techniques',
    ],
    starterCode: `def hamming_distance(x, y):
    """
    Calculate Hamming distance.
    
    Args:
        x: First integer
        y: Second integer
        
    Returns:
        Number of different bit positions
        
    Examples:
        >>> hamming_distance(1, 4)
        2
    """
    pass


# Test
print(hamming_distance(1, 4))
`,
    testCases: [
      {
        input: [1, 4],
        expected: 2,
      },
      {
        input: [3, 1],
        expected: 1,
      },
    ],
    solution: `def hamming_distance(x, y):
    xor = x ^ y
    count = 0
    
    while xor:
        count += xor & 1
        xor >>= 1
    
    return count


# One-liner
def hamming_distance_oneliner(x, y):
    return bin(x ^ y).count('1')`,
    timeComplexity: 'O(1) - max 32 bits',
    spaceComplexity: 'O(1)',
    order: 85,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-complement-base-10',
    title: 'Complement of Base 10 Integer',
    difficulty: 'Easy',
    description: `Return the complement of a number's binary representation.

Complement: flip all bits (0→1, 1→0).

**Example:** 5 = 101 → complement = 010 = 2

**Note:** No leading zeros in binary representation.

This tests:
- Bit manipulation
- Bit flipping
- Binary representation`,
    examples: [
      {
        input: 'n = 5',
        output: '2',
        explanation: '101 → 010',
      },
      {
        input: 'n = 7',
        output: '0',
        explanation: '111 → 000',
      },
    ],
    constraints: ['0 <= n <= 10^9'],
    hints: [
      'Find bit length of number',
      'Create mask of all 1s for that length',
      'XOR with mask',
    ],
    starterCode: `def bitwise_complement(n):
    """
    Return complement of number.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Bitwise complement
        
    Examples:
        >>> bitwise_complement(5)
        2
    """
    pass


# Test
print(bitwise_complement(5))
`,
    testCases: [
      {
        input: [5],
        expected: 2,
      },
      {
        input: [7],
        expected: 0,
      },
      {
        input: [0],
        expected: 1,
      },
    ],
    solution: `def bitwise_complement(n):
    if n == 0:
        return 1
    
    # Find number of bits
    bit_length = n.bit_length()
    
    # Create mask: 2^bit_length - 1 (all 1s)
    mask = (1 << bit_length) - 1
    
    # XOR with mask to flip bits
    return n ^ mask


# Alternative using string
def bitwise_complement_string(n):
    if n == 0:
        return 1
    
    binary = bin(n)[2:]
    complement = ''.join('1' if bit == '0' else '0' for bit in binary)
    return int(complement, 2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 86,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-assign-cookies',
    title: 'Assign Cookies',
    difficulty: 'Easy',
    description: `You have cookies of different sizes and children with different greed factors.

Assign cookies to maximize content children.
- Child i is content if size[j] >= greed[i]
- Each child gets at most one cookie

**Example:** greed=[1,2,3], size=[1,1] → 1 child content

This tests:
- Greedy algorithm
- Sorting
- Two pointer technique`,
    examples: [
      {
        input: 'greed = [1,2,3], size = [1,1]',
        output: '1',
      },
      {
        input: 'greed = [1,2], size = [1,2,3]',
        output: '2',
      },
    ],
    constraints: ['1 <= len(greed), len(size) <= 3*10^4'],
    hints: [
      'Sort both arrays',
      'Use two pointers',
      'Try to satisfy smallest greed first',
    ],
    starterCode: `def find_content_children(greed, size):
    """
    Find max content children.
    
    Args:
        greed: Array of greed factors
        size: Array of cookie sizes
        
    Returns:
        Number of content children
        
    Examples:
        >>> find_content_children([1,2,3], [1,1])
        1
    """
    pass


# Test
print(find_content_children([1,2], [1,2,3]))
`,
    testCases: [
      {
        input: [
          [1, 2, 3],
          [1, 1],
        ],
        expected: 1,
      },
      {
        input: [
          [1, 2],
          [1, 2, 3],
        ],
        expected: 2,
      },
    ],
    solution: `def find_content_children(greed, size):
    greed.sort()
    size.sort()
    
    child = 0
    cookie = 0
    
    while child < len(greed) and cookie < len(size):
        if size[cookie] >= greed[child]:
            child += 1
        cookie += 1
    
    return child`,
    timeComplexity: 'O(n log n + m log m)',
    spaceComplexity: 'O(1)',
    order: 87,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-repeated-substring',
    title: 'Repeated Substring Pattern',
    difficulty: 'Easy',
    description: `Check if string can be constructed by repeating a substring.

**Example:** "abab" = "ab" * 2 → true
"aba" → false

**Trick:** s in (s+s)[1:-1] checks all rotations

This tests:
- String manipulation
- Pattern recognition
- Substring operations`,
    examples: [
      {
        input: 's = "abab"',
        output: 'True',
      },
      {
        input: 's = "aba"',
        output: 'False',
      },
      {
        input: 's = "abcabcabcabc"',
        output: 'True',
      },
    ],
    constraints: ['1 <= len(s) <= 10^4'],
    hints: [
      'Try substrings of length 1 to n//2',
      'Check if repeating forms original',
      'Or use (s+s)[1:-1] trick',
    ],
    starterCode: `def repeated_substring_pattern(s):
    """
    Check if string is repeated substring.
    
    Args:
        s: Input string
        
    Returns:
        True if repeated pattern exists
        
    Examples:
        >>> repeated_substring_pattern("abab")
        True
    """
    pass


# Test
print(repeated_substring_pattern("abcabcabcabc"))
`,
    testCases: [
      {
        input: ['abab'],
        expected: true,
      },
      {
        input: ['aba'],
        expected: false,
      },
      {
        input: ['abcabcabcabc'],
        expected: true,
      },
    ],
    solution: `def repeated_substring_pattern(s):
    return s in (s + s)[1:-1]


# Alternative checking all possible lengths
def repeated_substring_pattern_explicit(s):
    n = len(s)
    
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            pattern = s[:i]
            if pattern * (n // i) == s:
                return True
    
    return False`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 88,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-island-perimeter',
    title: 'Island Perimeter',
    difficulty: 'Easy',
    description: `Calculate perimeter of island in a grid.

Grid: 1 = land, 0 = water
Island: connected 1s (no diagonal connections)

**Example:** 
\`\`\`
[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]
\`\`\`
Perimeter = 16

This tests:
- 2D array traversal
- Neighbor checking
- Counting`,
    examples: [
      {
        input: 'grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]',
        output: '16',
      },
    ],
    constraints: ['1 <= rows, cols <= 100', 'Only one island'],
    hints: [
      'Each land cell contributes 4 to perimeter',
      'Subtract 2 for each shared edge',
      'Check 4 neighbors',
    ],
    starterCode: `def island_perimeter(grid):
    """
    Calculate island perimeter.
    
    Args:
        grid: 2D array (1=land, 0=water)
        
    Returns:
        Perimeter of island
        
    Examples:
        >>> island_perimeter([[0,1,0,0],[1,1,1,0]])
        12
    """
    pass


# Test
print(island_perimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]))
`,
    testCases: [
      {
        input: [
          [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
          ],
        ],
        expected: 16,
      },
      {
        input: [[[1]]],
        expected: 4,
      },
    ],
    solution: `def island_perimeter(grid):
    perimeter = 0
    rows, cols = len(grid), len(grid[0])
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                # Start with 4 sides
                perimeter += 4
                
                # Subtract shared edges
                if i > 0 and grid[i-1][j] == 1:  # Up
                    perimeter -= 2
                if j > 0 and grid[i][j-1] == 1:  # Left
                    perimeter -= 2
    
    return perimeter`,
    timeComplexity: 'O(m * n)',
    spaceComplexity: 'O(1)',
    order: 89,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-fibonacci-number',
    title: 'Fibonacci Number (Nth)',
    difficulty: 'Easy',
    description: `Calculate the Nth Fibonacci number.

F(0) = 0, F(1) = 1
F(n) = F(n-1) + F(n-2) for n > 1

**Example:** F(4) = 3 (0,1,1,2,3)

This tests:
- Dynamic programming
- Iterative vs recursive
- Space optimization`,
    examples: [
      {
        input: 'n = 4',
        output: '3',
      },
      {
        input: 'n = 2',
        output: '1',
      },
    ],
    constraints: ['0 <= n <= 30'],
    hints: [
      'Use iteration for O(n) time, O(1) space',
      'Track only last two values',
      'Recursion with memoization works',
    ],
    starterCode: `def fib(n):
    """
    Calculate Nth Fibonacci number.
    
    Args:
        n: Position in sequence
        
    Returns:
        Nth Fibonacci number
        
    Examples:
        >>> fib(4)
        3
    """
    pass


# Test
print(fib(4))
`,
    testCases: [
      {
        input: [4],
        expected: 3,
      },
      {
        input: [2],
        expected: 1,
      },
      {
        input: [0],
        expected: 0,
      },
    ],
    solution: `def fib(n):
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


# Recursive with memoization
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 90,
    topic: 'Python Fundamentals',
  },
];
