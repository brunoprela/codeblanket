import { Problem } from '@/lib/types';

export const twoPointersProblems: Problem[] = [
  // EASY - Valid Palindrome
  {
    id: 'valid-palindrome',
    title: 'Valid Palindrome',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    order: 1,
    description: `A phrase is a **palindrome** if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string \`s\`, return \`true\` if it is a **palindrome**, or \`false\` otherwise.`,
    examples: [
      {
        input: 's = "A man, a plan, a canal: Panama"',
        output: 'true',
        explanation:
          '"amanaplanacanalpanama" is a palindrome after removing non-alphanumeric characters.',
      },
      {
        input: 's = "race a car"',
        output: 'false',
        explanation: '"raceacar" is not a palindrome.',
      },
      {
        input: 's = " "',
        output: 'true',
        explanation:
          'After removing non-alphanumeric characters, s becomes an empty string "" which is a palindrome.',
      },
    ],
    constraints: [
      '1 <= s.length <= 2 * 10^5',
      's consists only of printable ASCII characters',
    ],
    hints: [
      'Use two pointers, one from the start and one from the end',
      'Skip non-alphanumeric characters',
      'Compare characters after converting to lowercase',
    ],
    starterCode: `def isPalindrome(s: str) -> bool:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['A man, a plan, a canal: Panama'],
        expected: true,
      },
      {
        input: ['race a car'],
        expected: false,
      },
      {
        input: [' '],
        expected: true,
      },
      {
        input: ['ab'],
        expected: false,
      },
      {
        input: ['a'],
        expected: true,
      },
    ],
    timeComplexity: 'O(n) - single pass through the string',
    spaceComplexity: 'O(1) - only using two pointers',
    leetcodeUrl: 'https://leetcode.com/problems/valid-palindrome/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jJXJ16kPFWg',
  },

  // MEDIUM - Container With Most Water
  {
    id: 'container-with-most-water',
    title: 'Container With Most Water',
    difficulty: 'Medium',
    topic: 'Two Pointers',
    order: 2,
    description: `You are given an integer array \`height\` of length \`n\`. There are \`n\` vertical lines drawn such that the two endpoints of the \`i\`th line are \`(i, 0)\` and \`(i, height[i])\`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

**Notice** that you may not slant the container.`,
    examples: [
      {
        input: 'height = [1,8,6,2,5,4,8,3,7]',
        output: '49',
        explanation:
          'The vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. The max area of water is 49.',
      },
      {
        input: 'height = [1,1]',
        output: '1',
      },
    ],
    constraints: [
      'n == height.length',
      '2 <= n <= 10^5',
      '0 <= height[i] <= 10^4',
    ],
    hints: [
      'Start with the widest container (leftmost and rightmost lines)',
      'Move the pointer pointing to the shorter line inward',
      'The area is limited by the shorter line',
      'By moving the shorter line pointer, you might find a taller line',
    ],
    starterCode: `from typing import List

def maxArea(height: List[int]) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 8, 6, 2, 5, 4, 8, 3, 7]],
        expected: 49,
      },
      {
        input: [[1, 1]],
        expected: 1,
      },
      {
        input: [[4, 3, 2, 1, 4]],
        expected: 16,
      },
      {
        input: [[1, 2, 1]],
        expected: 2,
      },
    ],
    timeComplexity: 'O(n) - single pass with two pointers',
    spaceComplexity: 'O(1) - only using two pointers',
    leetcodeUrl: 'https://leetcode.com/problems/container-with-most-water/',
    youtubeUrl: 'https://www.youtube.com/watch?v=UuiTKBwPgAo',
  },

  // HARD - Trapping Rain Water
  {
    id: 'trapping-rain-water',
    title: 'Trapping Rain Water',
    difficulty: 'Hard',
    topic: 'Two Pointers',
    order: 3,
    description: `Given \`n\` non-negative integers representing an elevation map where the width of each bar is \`1\`, compute how much water it can trap after raining.`,
    examples: [
      {
        input: 'height = [0,1,0,2,1,0,1,3,2,1,2,1]',
        output: '6',
        explanation:
          'The elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water are being trapped.',
      },
      {
        input: 'height = [4,2,0,3,2,5]',
        output: '9',
      },
    ],
    constraints: [
      'n == height.length',
      '1 <= n <= 2 * 10^4',
      '0 <= height[i] <= 10^5',
    ],
    hints: [
      'Use two pointers from both ends',
      'Track the maximum height seen so far from left and right',
      'Water trapped at a position depends on the minimum of left_max and right_max',
      'Move the pointer with the smaller max height',
    ],
    starterCode: `from typing import List

def trap(height: List[int]) -> int:
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]],
        expected: 6,
      },
      {
        input: [[4, 2, 0, 3, 2, 5]],
        expected: 9,
      },
      {
        input: [[4, 2, 3]],
        expected: 1,
      },
      {
        input: [[3, 0, 2, 0, 4]],
        expected: 7,
      },
    ],
    timeComplexity: 'O(n) - single pass with two pointers',
    spaceComplexity: 'O(1) - constant space with two pointers',
    leetcodeUrl: 'https://leetcode.com/problems/trapping-rain-water/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ZI2z5pq0TqA',
  },
  // EASY - Remove Duplicates from Sorted Array
  {
    id: 'remove-duplicates-sorted-array',
    title: 'Remove Duplicates from Sorted Array',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    order: 4,
    description: `Given an integer array \`nums\` sorted in **non-decreasing order**, remove the duplicates **in-place** such that each unique element appears only **once**. The **relative order** of the elements should be kept the **same**.

Return \`k\` after placing the final result in the first \`k\` slots of \`nums\`.

Do **not** allocate extra space for another array. You must do this by **modifying the input array in-place** with O(1) extra memory.`,
    examples: [
      {
        input: 'nums = [1,1,2]',
        output: '2, nums = [1,2,_]',
        explanation:
          'Your function should return k = 2, with the first two elements of nums being 1 and 2.',
      },
      {
        input: 'nums = [0,0,1,1,1,2,2,3,3,4]',
        output: '5, nums = [0,1,2,3,4,_,_,_,_,_]',
        explanation:
          'Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4.',
      },
    ],
    constraints: [
      '1 <= nums.length <= 3 * 10^4',
      '-100 <= nums[i] <= 100',
      'nums is sorted in non-decreasing order',
    ],
    hints: [
      'Use two pointers: one for reading, one for writing',
      'Only write when you find a new unique element',
      'Return the write pointer position as the length',
    ],
    starterCode: `from typing import List

def remove_duplicates(nums: List[int]) -> int:
    """
    Remove duplicates in-place and return new length.
    
    Args:
        nums: Sorted array with duplicates
        
    Returns:
        Length of array with unique elements
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 1, 2]],
        expected: 2,
      },
      {
        input: [[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]],
        expected: 5,
      },
      {
        input: [[1]],
        expected: 1,
      },
    ],
    solution: `from typing import List

def remove_duplicates(nums: List[int]) -> int:
    """
    Two pointers: slow for writing, fast for reading.
    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0
    
    # Slow pointer for writing unique elements
    slow = 1
    
    # Fast pointer for reading
    for fast in range(1, len(nums)):
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/remove-duplicates-from-sorted-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=DEJAZBq0FDA',
  },
  // EASY - Move Zeroes
  {
    id: 'move-zeroes',
    title: 'Move Zeroes',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    order: 5,
    description: `Given an integer array \`nums\`, move all \`0\`'s to the end of it while maintaining the relative order of the non-zero elements.

**Note** that you must do this in-place without making a copy of the array.`,
    examples: [
      {
        input: 'nums = [0,1,0,3,12]',
        output: '[1,3,12,0,0]',
      },
      {
        input: 'nums = [0]',
        output: '[0]',
      },
    ],
    constraints: ['1 <= nums.length <= 10^4', '-2^31 <= nums[i] <= 2^31 - 1'],
    hints: [
      'Use two pointers: one for reading, one for placing non-zero elements',
      'After placing all non-zero elements, fill remaining with zeros',
      'Can you do it in one pass?',
    ],
    starterCode: `from typing import List

def move_zeroes(nums: List[int]) -> None:
    """
    Move all zeroes to the end in-place.
    
    Args:
        nums: Array with some zeros
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[0, 1, 0, 3, 12]],
        expected: [1, 3, 12, 0, 0],
      },
      {
        input: [[0]],
        expected: [0],
      },
      {
        input: [[1, 2, 3]],
        expected: [1, 2, 3],
      },
    ],
    solution: `from typing import List

def move_zeroes(nums: List[int]) -> None:
    """
    Two pointers: snowball approach.
    Time: O(n), Space: O(1)
    """
    # Pointer for placing non-zero elements
    left = 0
    
    # Move all non-zero elements to the front
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/move-zeroes/',
    youtubeUrl: 'https://www.youtube.com/watch?v=aayNRwUN3Do',
  },
  // EASY - Reverse String
  {
    id: 'reverse-string',
    title: 'Reverse String',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    order: 6,
    description: `Write a function that reverses a string. The input string is given as an array of characters \`s\`.

You must do this by modifying the input array **in-place** with O(1) extra memory.`,
    examples: [
      {
        input: 's = ["h","e","l","l","o"]',
        output: '["o","l","l","e","h"]',
      },
      {
        input: 's = ["H","a","n","n","a","h"]',
        output: '["h","a","n","n","a","H"]',
      },
    ],
    constraints: [
      '1 <= s.length <= 10^5',
      's[i] is a printable ascii character',
    ],
    hints: [
      'Use two pointers: one at start, one at end',
      'Swap characters and move pointers towards center',
      'Stop when pointers meet',
    ],
    starterCode: `from typing import List

def reverse_string(s: List[str]) -> None:
    """
    Reverse string in-place.
    
    Args:
        s: Array of characters
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['h', 'e', 'l', 'l', 'o']],
        expected: ['o', 'l', 'l', 'e', 'h'],
      },
      {
        input: [['H', 'a', 'n', 'n', 'a', 'h']],
        expected: ['h', 'a', 'n', 'n', 'a', 'H'],
      },
    ],
    solution: `from typing import List

def reverse_string(s: List[str]) -> None:
    """
    Two pointers from both ends.
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/reverse-string/',
    youtubeUrl: 'https://www.youtube.com/watch?v=_d0T_2Lk2qA',
  },
  // EASY - Valid Palindrome II
  {
    id: 'valid-palindrome-ii',
    title: 'Valid Palindrome II',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    order: 7,
    description: `Given a string \`s\`, return \`true\` if the \`s\` can be palindrome after deleting **at most one** character from it.`,
    examples: [
      {
        input: 's = "aba"',
        output: 'true',
      },
      {
        input: 's = "abca"',
        output: 'true',
        explanation:
          'You could delete the character "c" or "b" to make it a palindrome.',
      },
      {
        input: 's = "abc"',
        output: 'false',
      },
    ],
    constraints: ['1 <= s.length <= 10^5', 's consists of lowercase English letters'],
    hints: [
      'Use two pointers from both ends',
      'When characters mismatch, try skipping either left or right character',
      'Check if remaining substring is a palindrome',
    ],
    starterCode: `def valid_palindrome(s: str) -> bool:
    """
    Check if string can be palindrome by deleting at most one character.
    
    Args:
        s: Input string
        
    Returns:
        True if can be palindrome with at most one deletion
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['aba'],
        expected: true,
      },
      {
        input: ['abca'],
        expected: true,
      },
      {
        input: ['abc'],
        expected: false,
      },
    ],
    solution: `def valid_palindrome(s: str) -> bool:
    """
    Two pointers with one deletion allowed.
    Time: O(n), Space: O(1)
    """
    def is_palindrome(left: int, right: int) -> bool:
        """Helper to check if substring is palindrome"""
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            # Try skipping either left or right character
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    
    return True
`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/valid-palindrome-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=JrxRYBwG6EI',
  },
  // EASY - Merge Sorted Array
  {
    id: 'merge-sorted-array',
    title: 'Merge Sorted Array',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    order: 8,
    description: `You are given two integer arrays \`nums1\` and \`nums2\`, sorted in **non-decreasing order**, and two integers \`m\` and \`n\`, representing the number of elements in \`nums1\` and \`nums2\` respectively.

**Merge** \`nums1\` and \`nums2\` into a single array sorted in **non-decreasing order**.

The final sorted array should not be returned by the function, but instead be **stored inside the array \`nums1\`**. To accommodate this, \`nums1\` has a length of \`m + n\`, where the first \`m\` elements denote the elements that should be merged, and the last \`n\` elements are set to \`0\` and should be ignored. \`nums2\` has a length of \`n\`.`,
    examples: [
      {
        input: 'nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3',
        output: '[1,2,2,3,5,6]',
      },
      {
        input: 'nums1 = [1], m = 1, nums2 = [], n = 0',
        output: '[1]',
      },
    ],
    constraints: [
      'nums1.length == m + n',
      'nums2.length == n',
      '0 <= m, n <= 200',
      '1 <= m + n <= 200',
      '-10^9 <= nums1[i], nums2[j] <= 10^9',
    ],
    hints: [
      'Start from the end of both arrays',
      'Compare elements and place the larger one at the end',
      'This avoids overwriting elements in nums1',
    ],
    starterCode: `from typing import List

def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Merge nums2 into nums1 in-place.
    
    Args:
        nums1: First sorted array with extra space
        m: Number of elements in nums1
        nums2: Second sorted array
        n: Number of elements in nums2
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3],
        expected: [1, 2, 2, 3, 5, 6],
      },
      {
        input: [[1], 1, [], 0],
        expected: [1],
      },
      {
        input: [[0], 0, [1], 1],
        expected: [1],
      },
    ],
    solution: `from typing import List

def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Three pointers: merge from end to avoid overwriting.
    Time: O(m + n), Space: O(1)
    """
    # Pointers for nums1, nums2, and write position
    p1, p2 = m - 1, n - 1
    write = m + n - 1
    
    # Merge from end to start
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[write] = nums1[p1]
            p1 -= 1
        else:
            nums1[write] = nums2[p2]
            p2 -= 1
        write -= 1
`,
    timeComplexity: 'O(m + n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/merge-sorted-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=P1Ic85RarKY',
  },
];
