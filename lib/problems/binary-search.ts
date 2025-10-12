import { Problem } from '@/lib/types';

export const binarySearchProblems: Problem[] = [
  // EASY - Classic Binary Search
  {
    id: 'binary-search',
    title: 'Binary Search',
    difficulty: 'Easy',
    topic: 'Binary Search',
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
