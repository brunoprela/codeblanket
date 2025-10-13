import { Problem } from '@/lib/types';

export const binarySearchProblems: Problem[] = [
  // EASY - Classic Binary Search
  {
    id: 'binary-search',
    title: 'Binary Search',
    difficulty: 'Easy',
    topic: 'Binary Search',
    
    leetcodeUrl: 'https://leetcode.com/problems/binary-search/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s4DPM8ct1pI',
    order: 1,
    description: `Given an array of integers \`nums\` which is sorted in ascending order, and an integer \`target\`, write a function to search \`target\` in \`nums\`. If \`target\` exists, then return its index. Otherwise, return \`-1\`.

You must write an algorithm with **O(log n)** runtime complexity.`,
    examples: [
      {
        input: 'nums = [-1, 0, 3, 5, 9, 12], target = 9',
        output: '4',
        explanation: '9 exists in nums and its index is 4',
      },
      {
        input: 'nums = [-1, 0, 3, 5, 9, 12], target = 2',
        output: '-1',
        explanation: '2 does not exist in nums so return -1',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^4',
      '-10^4 < nums[i], target < 10^4',
      'All the integers in nums are unique',
      'nums is sorted in ascending order',
    ],
    hints: [
      'Think about dividing the search space in half with each iteration',
      'What should you do when the middle element is greater than the target?',
      'What are your loop termination conditions?',
    ],
    starterCode: `from typing import List

def binary_search(nums: List[int], target: int) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[-1, 0, 3, 5, 9, 12], 9],
        expected: 4,
      },
      {
        input: [[-1, 0, 3, 5, 9, 12], 2],
        expected: -1,
      },
      {
        input: [[5], 5],
        expected: 0,
      },
      {
        input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1],
        expected: 0,
      },
      {
        input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10],
        expected: 9,
      },
      {
        input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11],
        expected: -1,
      },
    ],
    solution: `def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
`,
    timeComplexity: 'O(log n) - we halve the search space with each iteration',
    spaceComplexity: 'O(1) - we only use a constant amount of extra space',
    leetcodeUrl: 'https://leetcode.com/problems/binary-search/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s4DPM8ct1pI',
  },

  // MEDIUM - Search in Rotated Sorted Array
  {
    id: 'search-rotated-array',
    title: 'Search in Rotated Sorted Array',
    difficulty: 'Medium',
    topic: 'Binary Search',
    order: 2,
    description: `There is an integer array \`nums\` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, \`nums\` is **rotated** at an unknown pivot index \`k\` (\`0 <= k < nums.length\`) such that the resulting array is \`[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]\` (**0-indexed**). For example, \`[0,1,2,4,5,6,7]\` might be rotated at pivot index \`3\` and become \`[4,5,6,7,0,1,2]\`.

Given the array \`nums\` **after** the rotation and an integer \`target\`, return the index of \`target\` if it is in \`nums\`, or \`-1\` if it is not in \`nums\`.

You must write an algorithm with **O(log n)** runtime complexity.`,
    examples: [
      {
        input: 'nums = [4, 5, 6, 7, 0, 1, 2], target = 0',
        output: '4',
        explanation: 'The target 0 is at index 4',
      },
      {
        input: 'nums = [4, 5, 6, 7, 0, 1, 2], target = 3',
        output: '-1',
        explanation: '3 does not exist in nums',
      },
      {
        input: 'nums = [1], target = 0',
        output: '-1',
      },
    ],
    constraints: [
      '1 <= nums.length <= 5000',
      '-10^4 <= nums[i] <= 10^4',
      'All values of nums are unique',
      'nums is guaranteed to be rotated at some pivot',
    ],
    hints: [
      'At least one half of the array is always sorted',
      'Which half is sorted? Check if nums[left] <= nums[mid]',
      'If the left half is sorted, check if target is in that range',
      'If target is not in the sorted half, search the other half',
    ],
    starterCode: `from typing import List

def search(nums: List[int], target: int) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[4, 5, 6, 7, 0, 1, 2], 0],
        expected: 4,
      },
      {
        input: [[4, 5, 6, 7, 0, 1, 2], 3],
        expected: -1,
      },
      {
        input: [[1], 0],
        expected: -1,
      },
      {
        input: [[1], 1],
        expected: 0,
      },
      {
        input: [[3, 1], 1],
        expected: 1,
      },
      {
        input: [[5, 1, 3], 5],
        expected: 0,
      },
      {
        input: [[4, 5, 6, 7, 8, 1, 2, 3], 8],
        expected: 4,
      },
    ],
    solution: `def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
`,
    timeComplexity: 'O(log n) - binary search on the rotated array',
    spaceComplexity: 'O(1) - constant space usage',
    leetcodeUrl:
      'https://leetcode.com/problems/search-in-rotated-sorted-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=U8XENwh8Oy8',
  },

  // HARD - Median of Two Sorted Arrays
  {
    id: 'median-two-sorted-arrays',
    title: 'Median of Two Sorted Arrays',
    difficulty: 'Hard',
    topic: 'Binary Search',
    order: 3,
    description: `Given two sorted arrays \`nums1\` and \`nums2\` of size \`m\` and \`n\` respectively, return **the median** of the two sorted arrays.

The overall run time complexity should be **O(log(m+n))**.`,
    examples: [
      {
        input: 'nums1 = [1, 3], nums2 = [2]',
        output: '2.0',
        explanation: 'merged array = [1, 2, 3] and median is 2',
      },
      {
        input: 'nums1 = [1, 2], nums2 = [3, 4]',
        output: '2.5',
        explanation:
          'merged array = [1, 2, 3, 4] and median is (2 + 3) / 2 = 2.5',
      },
    ],
    constraints: [
      'nums1.length == m',
      'nums2.length == n',
      '0 <= m <= 1000',
      '0 <= n <= 1000',
      '1 <= m + n <= 2000',
      '-10^6 <= nums1[i], nums2[i] <= 10^6',
    ],
    hints: [
      'The median divides the array into two equal halves',
      'Use binary search on the smaller array',
      'Find the correct partition where elements on the left <= elements on the right',
      'Handle edge cases: empty arrays, arrays of different sizes',
    ],
    starterCode: `from typing import List

def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 3], [2]],
        expected: 2.0,
      },
      {
        input: [
          [1, 2],
          [3, 4],
        ],
        expected: 2.5,
      },
      {
        input: [
          [0, 0],
          [0, 0],
        ],
        expected: 0.0,
      },
      {
        input: [[], [1]],
        expected: 1.0,
      },
      {
        input: [[2], []],
        expected: 2.0,
      },
      {
        input: [
          [1, 3, 5, 7, 9],
          [2, 4, 6, 8, 10],
        ],
        expected: 5.5,
      },
      {
        input: [
          [1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
        ],
        expected: 5.5,
      },
    ],
    solution: `def findMedianSortedArrays(nums1, nums2):
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        # Handle edge cases
        maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        minRight1 = float('inf') if partition1 == m else nums1[partition1]
        
        maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        minRight2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if we found the correct partition
        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            # If total length is even
            if (m + n) % 2 == 0:
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2.0
            # If total length is odd
            else:
                return float(max(maxLeft1, maxLeft2))
        elif maxLeft1 > minRight2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    return 0.0
`,
    timeComplexity: 'O(log(min(m, n))) - binary search on the smaller array',
    spaceComplexity: 'O(1) - constant space usage',
    leetcodeUrl: 'https://leetcode.com/problems/median-of-two-sorted-arrays/',
    youtubeUrl: 'https://www.youtube.com/watch?v=q6IEA26hvXc',
  },
  // EASY - Search Insert Position
  {
    id: 'search-insert-position',
    title: 'Search Insert Position',
    difficulty: 'Easy',
    topic: 'Binary Search',
    order: 4,
    description: `Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with **O(log n)** runtime complexity.`,
    examples: [
      {
        input: 'nums = [1,3,5,6], target = 5',
        output: '2',
      },
      {
        input: 'nums = [1,3,5,6], target = 2',
        output: '1',
      },
      {
        input: 'nums = [1,3,5,6], target = 7',
        output: '4',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^4',
      '-10^4 <= nums[i] <= 10^4',
      'nums contains distinct values sorted in ascending order',
      '-10^4 <= target <= 10^4',
    ],
    hints: [
      'Use binary search to find the position',
      'If target is found, return its index',
      'If not found, left pointer will be at the insert position',
    ],
    starterCode: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    """
    Find target or return insert position.
    
    Args:
        nums: Sorted array of distinct integers
        target: Target value to find
        
    Returns:
        Index of target or insert position
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 3, 5, 6], 5],
        expected: 2,
      },
      {
        input: [[1, 3, 5, 6], 2],
        expected: 1,
      },
      {
        input: [[1, 3, 5, 6], 7],
        expected: 4,
      },
    ],
    solution: `from typing import List

def search_insert(nums: List[int], target: int) -> int:
    """
    Binary search for target or insert position.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # If not found, left is the insert position
    return left
`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/search-insert-position/',
    youtubeUrl: 'https://www.youtube.com/watch?v=K-RYzDZkzCI',
  },
  // EASY - First Bad Version
  {
    id: 'first-bad-version',
    title: 'First Bad Version',
    difficulty: 'Easy',
    topic: 'Binary Search',
    order: 5,
    description: `You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have \`n\` versions \`[1, 2, ..., n]\` and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API \`def is_bad_version(version):\` which returns whether \`version\` is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.`,
    examples: [
      {
        input: 'n = 5, bad = 4',
        output: '4',
        explanation:
          'is_bad_version(3) -> False, is_bad_version(4) -> True, so 4 is the first bad version.',
      },
      {
        input: 'n = 1, bad = 1',
        output: '1',
      },
    ],
    constraints: ['1 <= bad <= n <= 2^31 - 1'],
    hints: [
      'Use binary search to minimize API calls',
      'If version is bad, search left half (including current)',
      'If version is good, search right half',
    ],
    starterCode: `# The is_bad_version API is already defined for you.
def is_bad_version(version: int) -> bool:
    pass

def first_bad_version(n: int) -> int:
    """
    Find the first bad version.
    
    Args:
        n: Total number of versions
        
    Returns:
        The first bad version number
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [5],
        expected: 4,
      },
      {
        input: [1],
        expected: 1,
      },
      {
        input: [10],
        expected: 7,
      },
    ],
    solution: `def is_bad_version(version: int) -> bool:
    # This is a stub - actual implementation provided by API
    pass

def first_bad_version(n: int) -> int:
    """
    Binary search for first bad version.
    Time: O(log n), Space: O(1)
    """
    left, right = 1, n
    
    while left < right:
        mid = (left + right) // 2
        
        if is_bad_version(mid):
            # First bad is at mid or before
            right = mid
        else:
            # First bad is after mid
            left = mid + 1
    
    return left
`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/first-bad-version/',
    youtubeUrl: 'https://www.youtube.com/watch?v=GJVO2BTdBZw',
  },

  // EASY - Find Smallest Letter Greater Than Target
  {
    id: 'find-smallest-letter-greater-than-target',
    title: 'Find Smallest Letter Greater Than Target',
    difficulty: 'Easy',
    topic: 'Binary Search',
    order: 7,
    description: `You are given an array of characters \`letters\` that is sorted in **non-decreasing order**, and a character \`target\`. There are **at least two different** characters in \`letters\`.

Return the smallest character in \`letters\` that is lexicographically greater than \`target\`. If such a character does not exist, return the first character in \`letters\`.`,
    examples: [
      {
        input: 'letters = ["c","f","j"], target = "a"',
        output: '"c"',
        explanation: 'The smallest character that is greater than "a" is "c".',
      },
      {
        input: 'letters = ["c","f","j"], target = "c"',
        output: '"f"',
        explanation: 'The smallest character that is greater than "c" is "f".',
      },
      {
        input: 'letters = ["x","x","y","y"], target = "z"',
        output: '"x"',
        explanation:
          'No character is greater than "z" so we wrap around to "x".',
      },
    ],
    constraints: [
      '2 <= letters.length <= 10^4',
      'letters[i] is a lowercase English letter',
      'letters is sorted in non-decreasing order',
      'letters contains at least two different characters',
      'target is a lowercase English letter',
    ],
    hints: [
      'Use binary search since array is sorted',
      'If no character is greater, return first character',
      'Keep track of the smallest valid answer',
    ],
    starterCode: `from typing import List

def next_greatest_letter(letters: List[str], target: str) -> str:
    """
    Find smallest letter greater than target.
    
    Args:
        letters: Sorted array of characters
        target: Target character
        
    Returns:
        Smallest character greater than target
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['c', 'f', 'j'], 'a'],
        expected: 'c',
      },
      {
        input: [['c', 'f', 'j'], 'c'],
        expected: 'f',
      },
      {
        input: [['x', 'x', 'y', 'y'], 'z'],
        expected: 'x',
      },
    ],
    solution: `from typing import List

def next_greatest_letter(letters: List[str], target: str) -> str:
    """
    Binary search for next greatest letter.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(letters) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if letters[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    
    # If left is out of bounds, wrap around to first element
    return letters[left % len(letters)]
`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/find-smallest-letter-greater-than-target/',
    youtubeUrl: 'https://www.youtube.com/watch?v=W9QJ8HaRvJQ',
  },
  // EASY - Peak Index in a Mountain Array
  {
    id: 'peak-index-mountain-array',
    title: 'Peak Index in a Mountain Array',
    difficulty: 'Easy',
    topic: 'Binary Search',
    order: 8,
    description: `An array \`arr\` is a **mountain array** if and only if:
- \`arr.length >= 3\`
- There exists some \`i\` with \`0 < i < arr.length - 1\` such that:
  - \`arr[0] < arr[1] < ... < arr[i - 1] < arr[i]\`
  - \`arr[i] > arr[i + 1] > ... > arr[arr.length - 1]\`

Given a mountain array \`arr\`, return the index \`i\` such that \`arr[0] < arr[1] < ... < arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1]\`.

You must solve it in **O(log(arr.length))** time complexity.`,
    examples: [
      {
        input: 'arr = [0,1,0]',
        output: '1',
      },
      {
        input: 'arr = [0,2,1,0]',
        output: '1',
      },
      {
        input: 'arr = [0,10,5,2]',
        output: '1',
      },
    ],
    constraints: [
      '3 <= arr.length <= 10^5',
      '0 <= arr[i] <= 10^6',
      'arr is guaranteed to be a mountain array',
    ],
    hints: [
      'Use binary search to find the peak',
      'Compare arr[mid] with arr[mid + 1]',
      'If increasing, peak is on the right; if decreasing, peak is on the left',
    ],
    starterCode: `from typing import List

def peak_index_in_mountain_array(arr: List[int]) -> int:
    """
    Find the peak index in a mountain array.
    
    Args:
        arr: Mountain array
        
    Returns:
        Index of the peak element
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[0, 1, 0]],
        expected: 1,
      },
      {
        input: [[0, 2, 1, 0]],
        expected: 1,
      },
      {
        input: [[0, 10, 5, 2]],
        expected: 1,
      },
    ],
    solution: `from typing import List

def peak_index_in_mountain_array(arr: List[int]) -> int:
    """
    Binary search for peak in mountain array.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] < arr[mid + 1]:
            # We're on the ascending slope, peak is to the right
            left = mid + 1
        else:
            # We're on the descending slope or at peak, peak is to the left or at mid
            right = mid
    
    return left
`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/peak-index-in-a-mountain-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=HtSuA80QTyo',
  },

  // EASY - Count Negative Numbers in a Sorted Matrix
  {
    id: 'count-negative-numbers-sorted-matrix',
    title: 'Count Negative Numbers in a Sorted Matrix',
    difficulty: 'Easy',
    topic: 'Binary Search',
    description: `Given a \`m x n\` matrix \`grid\` which is sorted in non-increasing order both row-wise and column-wise, return the number of negative numbers in \`grid\`.`,
    examples: [
      {
        input: 'grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]',
        output: '8',
        explanation: 'There are 8 negatives number in the matrix.',
      },
      {
        input: 'grid = [[3,2],[1,0]]',
        output: '0',
      },
    ],
    constraints: [
      'm == grid.length',
      'n == grid[i].length',
      '1 <= m, n <= 100',
      '-100 <= grid[i][j] <= 100',
    ],
    hints: [
      'Use binary search for each row',
      'The matrix is sorted, so once you find a negative number, all numbers to its right are also negative',
    ],
    starterCode: `from typing import List

def count_negatives(grid: List[List[int]]) -> int:
    """
    Count negative numbers in sorted matrix.
    
    Args:
        grid: Matrix sorted in non-increasing order
        
    Returns:
        Count of negative numbers
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [4, 3, 2, -1],
            [3, 2, 1, -1],
            [1, 1, -1, -2],
            [-1, -1, -2, -3],
          ],
        ],
        expected: 8,
      },
      {
        input: [
          [
            [3, 2],
            [1, 0],
          ],
        ],
        expected: 0,
      },
      {
        input: [
          [
            [1, -1],
            [-1, -1],
          ],
        ],
        expected: 3,
      },
    ],
    timeComplexity: 'O(m log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/',
    youtubeUrl: 'https://www.youtube.com/watch?v=5BI4BxoVlLo',
  },

  // EASY - Arranging Coins
  {
    id: 'arranging-coins',
    title: 'Arranging Coins',
    difficulty: 'Easy',
    topic: 'Binary Search',
    description: `You have \`n\` coins and you want to build a staircase with these coins. The staircase consists of \`k\` rows where the \`i-th\` row has exactly \`i\` coins. The last row of the staircase may be incomplete.

Given the integer \`n\`, return the number of complete rows of the staircase you will build.`,
    examples: [
      {
        input: 'n = 5',
        output: '2',
        explanation:
          'The coins can form these rows: ¤, ¤ ¤. The 3rd row is incomplete, so return 2.',
      },
      {
        input: 'n = 8',
        output: '3',
        explanation:
          'The coins can form these rows: ¤, ¤ ¤, ¤ ¤ ¤. The 4th row is incomplete, so return 3.',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1'],
    hints: [
      'Use the formula k * (k + 1) / 2 to calculate the total coins needed for k rows',
      'Binary search for the answer',
    ],
    starterCode: `def arrange_coins(n: int) -> int:
    """
    Find complete rows in coin staircase.
    
    Args:
        n: Total number of coins
        
    Returns:
        Number of complete rows
    """
    # Write your code here
    pass
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
      {
        input: [10],
        expected: 4,
      },
    ],
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/arranging-coins/',
    youtubeUrl: 'https://www.youtube.com/watch?v=C4TkkOuBd44',
  },

  // EASY - Guess Number Higher or Lower
  {
    id: 'guess-number-higher-or-lower',
    title: 'Guess Number Higher or Lower',
    difficulty: 'Easy',
    topic: 'Binary Search',
    description: `We are playing the Guess Game. The game is as follows:

I pick a number from \`1\` to \`n\`. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API \`int guess(int num)\`, which returns three possible results:
- \`-1\`: Your guess is higher than the number I picked (i.e. \`num > pick\`)
- \`1\`: Your guess is lower than the number I picked (i.e. \`num < pick\`)
- \`0\`: your guess is equal to the number I picked (i.e. \`num == pick\`)

Return the number that I picked.`,
    examples: [
      {
        input: 'n = 10, pick = 6',
        output: '6',
      },
      {
        input: 'n = 1, pick = 1',
        output: '1',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1', '1 <= pick <= n'],
    hints: [
      'Use binary search',
      'Call guess() to narrow down the search space',
    ],
    starterCode: `# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num: int) -> int:

def guess_number(n: int) -> int:
    """
    Guess the picked number using binary search.
    
    Args:
        n: Upper bound of range [1, n]
        
    Returns:
        The picked number
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [10],
        expected: 6,
      },
      {
        input: [1],
        expected: 1,
      },
      {
        input: [2],
        expected: 1,
      },
    ],
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/guess-number-higher-or-lower/',
    youtubeUrl: 'https://www.youtube.com/watch?v=xW4QsTtaCa4',
  },

  // MEDIUM - Find Minimum in Rotated Sorted Array
  {
    id: 'find-minimum-rotated-sorted-array',
    title: 'Find Minimum in Rotated Sorted Array',
    difficulty: 'Medium',
    topic: 'Binary Search',
    description: `Suppose an array of length \`n\` sorted in ascending order is rotated between \`1\` and \`n\` times. For example, the array \`nums = [0,1,2,4,5,6,7]\` might become:
- \`[4,5,6,7,0,1,2]\` if it was rotated \`4\` times.
- \`[0,1,2,4,5,6,7]\` if it was rotated \`7\` times.

Notice that rotating an array \`[a[0], a[1], a[2], ..., a[n-1]]\` 1 time results in the array \`[a[n-1], a[0], a[1], a[2], ..., a[n-2]]\`.

Given the sorted rotated array \`nums\` of unique elements, return the minimum element of this array.

You must write an algorithm that runs in **O(log n)** time.`,
    examples: [
      {
        input: 'nums = [3,4,5,1,2]',
        output: '1',
        explanation: 'The original array was [1,2,3,4,5] rotated 3 times.',
      },
      {
        input: 'nums = [4,5,6,7,0,1,2]',
        output: '0',
        explanation:
          'The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.',
      },
    ],
    constraints: [
      'n == nums.length',
      '1 <= n <= 5000',
      '-5000 <= nums[i] <= 5000',
      'All the integers of nums are unique',
      'nums is sorted and rotated between 1 and n times',
    ],
    hints: [
      'If the array was not rotated, nums[0] < nums[n-1]',
      'Use binary search. Compare mid with the right boundary',
      'If nums[mid] > nums[right], the minimum is in the right half',
    ],
    starterCode: `from typing import List

def find_min(nums: List[int]) -> int:
    """
    Find minimum in rotated sorted array.
    
    Args:
        nums: Rotated sorted array
        
    Returns:
        Minimum element
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 4, 5, 1, 2]],
        expected: 1,
      },
      {
        input: [[4, 5, 6, 7, 0, 1, 2]],
        expected: 0,
      },
      {
        input: [[11, 13, 15, 17]],
        expected: 11,
      },
      {
        input: [[2, 1]],
        expected: 1,
      },
    ],
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=nIVW4P8b1VA',
  },

  // MEDIUM - Search in Rotated Sorted Array II
  {
    id: 'search-rotated-sorted-array-ii',
    title: 'Search in Rotated Sorted Array II',
    difficulty: 'Medium',
    topic: 'Binary Search',
    description: `There is an integer array \`nums\` sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, \`nums\` is rotated at an unknown pivot index \`k\` (\`0 <= k < nums.length\`) such that the resulting array is \`[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]\` (0-indexed).

Given the array \`nums\` after the rotation and an integer \`target\`, return \`true\` if \`target\` is in \`nums\`, or \`false\` if it is not in \`nums\`.

You must decrease the overall operation steps as much as possible.`,
    examples: [
      {
        input: 'nums = [2,5,6,0,0,1,2], target = 0',
        output: 'true',
      },
      {
        input: 'nums = [2,5,6,0,0,1,2], target = 3',
        output: 'false',
      },
    ],
    constraints: [
      '1 <= nums.length <= 5000',
      '-10^4 <= nums[i] <= 10^4',
      'nums is guaranteed to be rotated at some pivot',
      '-10^4 <= target <= 10^4',
    ],
    hints: [
      'This is the follow-up problem where nums may contain duplicates',
      'When nums[left] == nums[mid] == nums[right], we cannot determine which side is sorted',
      'In worst case, time complexity degrades to O(n)',
    ],
    starterCode: `from typing import List

def search(nums: List[int], target: int) -> bool:
    """
    Search in rotated sorted array with duplicates.
    
    Args:
        nums: Rotated sorted array (may contain duplicates)
        target: Target value to search
        
    Returns:
        True if target is in nums, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 5, 6, 0, 0, 1, 2], 0],
        expected: true,
      },
      {
        input: [[2, 5, 6, 0, 0, 1, 2], 3],
        expected: false,
      },
      {
        input: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1], 2],
        expected: true,
      },
    ],
    timeComplexity: 'O(log n) average, O(n) worst case',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/search-in-rotated-sorted-array-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=w-Aw00H73ak',
  },

  // MEDIUM - Find Peak Element
  {
    id: 'find-peak-element',
    title: 'Find Peak Element',
    difficulty: 'Medium',
    topic: 'Binary Search',
    description: `A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array \`nums\`, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that \`nums[-1] = nums[n] = -∞\`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in **O(log n)** time.`,
    examples: [
      {
        input: 'nums = [1,2,3,1]',
        output: '2',
        explanation:
          '3 is a peak element and your function should return the index number 2.',
      },
      {
        input: 'nums = [1,2,1,3,5,6,4]',
        output: '5',
        explanation:
          'Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.',
      },
    ],
    constraints: [
      '1 <= nums.length <= 1000',
      '-2^31 <= nums[i] <= 2^31 - 1',
      'nums[i] != nums[i + 1] for all valid i',
    ],
    hints: [
      'Use binary search',
      'If nums[mid] < nums[mid + 1], there must be a peak in the right half',
      'Otherwise, there must be a peak in the left half (including mid)',
    ],
    starterCode: `from typing import List

def find_peak_element(nums: List[int]) -> int:
    """
    Find a peak element and return its index.
    
    Args:
        nums: Integer array
        
    Returns:
        Index of a peak element
    """
    # Write your code here
    pass
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
      {
        input: [[1]],
        expected: 0,
      },
      {
        input: [[1, 2]],
        expected: 1,
      },
    ],
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/find-peak-element/',
    youtubeUrl: 'https://www.youtube.com/watch?v=kMzJy9es7Hc',
  },
];

export function getProblemById(id: string): Problem | undefined {
  return binarySearchProblems.find((p) => p.id === id);
}

export function getProblemsByDifficulty(
  difficulty: 'Easy' | 'Medium' | 'Hard',
): Problem[] {
  return binarySearchProblems.filter((p) => p.difficulty === difficulty);
}
