import { Problem } from '@/lib/types';

export const twoPointersProblems: Problem[] = [
  // EASY - Valid Palindrome
  {
    id: 'valid-palindrome',
    title: 'Valid Palindrome',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    
    leetcodeUrl: 'https://leetcode.com/problems/valid-palindrome/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jJXJ16kPFWg',
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
    constraints: [
      '1 <= s.length <= 10^5',
      's consists of lowercase English letters',
    ],
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

  // EASY - Remove Element
  {
    id: 'remove-element',
    title: 'Remove Element',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    description: `Given an integer array \`nums\` and an integer \`val\`, remove all occurrences of \`val\` in \`nums\` in-place. The order of the elements may be changed. Then return the number of elements in \`nums\` which are not equal to \`val\`.

Consider the number of elements in \`nums\` which are not equal to \`val\` be \`k\`, to get accepted, you need to do the following things:
- Change the array \`nums\` such that the first \`k\` elements of \`nums\` contain the elements which are not equal to \`val\`. The remaining elements of \`nums\` are not important as well as the size of \`nums\`.
- Return \`k\`.`,
    examples: [
      {
        input: 'nums = [3,2,2,3], val = 3',
        output: '2, nums = [2,2,_,_]',
        explanation:
          'Your function should return k = 2, with the first two elements of nums being 2.',
      },
      {
        input: 'nums = [0,1,2,2,3,0,4,2], val = 2',
        output: '5, nums = [0,1,3,0,4,_,_,_]',
      },
    ],
    constraints: [
      '0 <= nums.length <= 100',
      '0 <= nums[i] <= 50',
      '0 <= val <= 100',
    ],
    hints: [
      'Use two pointers: one for reading, one for writing',
      'Only write to the write pointer when current element is not equal to val',
    ],
    starterCode: `from typing import List

def remove_element(nums: List[int], val: int) -> int:
    """
    Remove all occurrences of val in-place.
    
    Args:
        nums: Integer array
        val: Value to remove
        
    Returns:
        Number of elements not equal to val
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 2, 2, 3], 3],
        expected: 2,
      },
      {
        input: [[0, 1, 2, 2, 3, 0, 4, 2], 2],
        expected: 5,
      },
      {
        input: [[1], 1],
        expected: 0,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/remove-element/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Pcd1ii9P9ZI',
  },

  // EASY - Squares of a Sorted Array
  {
    id: 'squares-sorted-array',
    title: 'Squares of a Sorted Array',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    description: `Given an integer array \`nums\` sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.`,
    examples: [
      {
        input: 'nums = [-4,-1,0,3,10]',
        output: '[0,1,9,16,100]',
        explanation:
          'After squaring, the array becomes [16,1,0,9,100]. After sorting, it becomes [0,1,9,16,100].',
      },
      {
        input: 'nums = [-7,-3,2,3,11]',
        output: '[4,9,9,49,121]',
      },
    ],
    constraints: [
      '1 <= nums.length <= 10^4',
      '-10^4 <= nums[i] <= 10^4',
      'nums is sorted in non-decreasing order',
    ],
    hints: [
      'Use two pointers from both ends',
      'The largest square will always be at one of the two ends',
      'Fill the result array from right to left',
    ],
    starterCode: `from typing import List

def sorted_squares(nums: List[int]) -> List[int]:
    """
    Return squares of sorted array in sorted order.
    
    Args:
        nums: Sorted integer array
        
    Returns:
        Sorted array of squares
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[-4, -1, 0, 3, 10]],
        expected: [0, 1, 9, 16, 100],
      },
      {
        input: [[-7, -3, 2, 3, 11]],
        expected: [4, 9, 9, 49, 121],
      },
      {
        input: [[-5, -3, -2, -1]],
        expected: [1, 4, 9, 25],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    leetcodeUrl: 'https://leetcode.com/problems/squares-of-a-sorted-array/',
    youtubeUrl: 'https://www.youtube.com/watch?v=FPCZsG_AkUg',
  },

  // EASY - Intersection of Two Arrays II
  {
    id: 'intersection-two-arrays-ii',
    title: 'Intersection of Two Arrays II',
    difficulty: 'Easy',
    topic: 'Two Pointers',
    description: `Given two integer arrays \`nums1\` and \`nums2\`, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.`,
    examples: [
      {
        input: 'nums1 = [1,2,2,1], nums2 = [2,2]',
        output: '[2,2]',
      },
      {
        input: 'nums1 = [4,9,5], nums2 = [9,4,9,8,4]',
        output: '[4,9]',
        explanation: '[9,4] is also accepted.',
      },
    ],
    constraints: [
      '1 <= nums1.length, nums2.length <= 1000',
      '0 <= nums1[i], nums2[i] <= 1000',
    ],
    hints: [
      'Sort both arrays first',
      'Use two pointers to traverse both arrays',
      'When values match, add to result and move both pointers',
    ],
    starterCode: `from typing import List

def intersect(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find intersection of two arrays with duplicates.
    
    Args:
        nums1: First integer array
        nums2: Second integer array
        
    Returns:
        Array of intersection elements
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [1, 2, 2, 1],
          [2, 2],
        ],
        expected: [2, 2],
      },
      {
        input: [
          [4, 9, 5],
          [9, 4, 9, 8, 4],
        ],
        expected: [4, 9],
      },
      {
        input: [
          [1, 2, 3],
          [4, 5, 6],
        ],
        expected: [],
      },
    ],
    timeComplexity: 'O(n log n + m log m)',
    spaceComplexity: 'O(min(n, m))',
    leetcodeUrl: 'https://leetcode.com/problems/intersection-of-two-arrays-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=lKuK69-hMcc',
  },

  // MEDIUM - 3Sum
  {
    id: 'three-sum',
    title: '3Sum',
    difficulty: 'Medium',
    topic: 'Two Pointers',
    description: `Given an integer array \`nums\`, return all the triplets \`[nums[i], nums[j], nums[k]]\` such that \`i != j\`, \`i != k\`, and \`j != k\`, and \`nums[i] + nums[j] + nums[k] == 0\`.

Notice that the solution set must not contain duplicate triplets.`,
    examples: [
      {
        input: 'nums = [-1,0,1,2,-1,-4]',
        output: '[[-1,-1,2],[-1,0,1]]',
        explanation:
          'The distinct triplets are [-1,0,1] and [-1,-1,2]. Notice that the order of the output and the order of the triplets does not matter.',
      },
      {
        input: 'nums = [0,1,1]',
        output: '[]',
        explanation: 'The only possible triplet does not sum up to 0.',
      },
    ],
    constraints: ['3 <= nums.length <= 3000', '-10^5 <= nums[i] <= 10^5'],
    hints: [
      'Sort the array first',
      'For each element, use two pointers to find pairs that sum to -element',
      'Skip duplicates to avoid duplicate triplets',
    ],
    starterCode: `from typing import List

def three_sum(nums: List[int]) -> List[List[int]]:
    """
    Find all unique triplets that sum to zero.
    
    Args:
        nums: Integer array
        
    Returns:
        List of triplets that sum to zero
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[-1, 0, 1, 2, -1, -4]],
        expected: [
          [-1, -1, 2],
          [-1, 0, 1],
        ],
      },
      {
        input: [[0, 1, 1]],
        expected: [],
      },
      {
        input: [[0, 0, 0]],
        expected: [[0, 0, 0]],
      },
    ],
    timeComplexity: 'O(n^2)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/3sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jzZsG8n2R9A',
  },

  // MEDIUM - Sort Colors
  {
    id: 'sort-colors',
    title: 'Sort Colors',
    difficulty: 'Medium',
    topic: 'Two Pointers',
    description: `Given an array \`nums\` with \`n\` objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers \`0\`, \`1\`, and \`2\` to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.`,
    examples: [
      {
        input: 'nums = [2,0,2,1,1,0]',
        output: '[0,0,1,1,2,2]',
      },
      {
        input: 'nums = [2,0,1]',
        output: '[0,1,2]',
      },
    ],
    constraints: [
      'n == nums.length',
      '1 <= n <= 300',
      'nums[i] is either 0, 1, or 2',
    ],
    hints: [
      'Use the Dutch National Flag algorithm',
      'Maintain three pointers: low (for 0s), mid (current), high (for 2s)',
      'Swap elements to their correct regions',
    ],
    starterCode: `from typing import List

def sort_colors(nums: List[int]) -> None:
    """
    Sort array of 0s, 1s, and 2s in-place.
    
    Args:
        nums: Array of integers (0, 1, or 2)
        
    Returns:
        None, modifies nums in-place
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[2, 0, 2, 1, 1, 0]],
        expected: [0, 0, 1, 1, 2, 2],
      },
      {
        input: [[2, 0, 1]],
        expected: [0, 1, 2],
      },
      {
        input: [[0]],
        expected: [0],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/sort-colors/',
    youtubeUrl: 'https://www.youtube.com/watch?v=4xbWSRZHqac',
  },

  // MEDIUM - Boats to Save People
  {
    id: 'boats-to-save-people',
    title: 'Boats to Save People',
    difficulty: 'Medium',
    topic: 'Two Pointers',
    description: `You are given an array \`people\` where \`people[i]\` is the weight of the \`i-th\` person, and an infinite number of boats where each boat can carry a maximum weight of \`limit\`. Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most \`limit\`.

Return the minimum number of boats to carry every given person.`,
    examples: [
      {
        input: 'people = [1,2], limit = 3',
        output: '1',
        explanation: '1 boat (1, 2)',
      },
      {
        input: 'people = [3,2,2,1], limit = 3',
        output: '3',
        explanation: '3 boats (1, 2), (2) and (3)',
      },
      {
        input: 'people = [3,5,3,4], limit = 5',
        output: '4',
        explanation: '4 boats (3), (3), (4), (5)',
      },
    ],
    constraints: [
      '1 <= people.length <= 5 * 10^4',
      '1 <= people[i] <= limit <= 3 * 10^4',
    ],
    hints: [
      'Sort the people by weight',
      'Use two pointers: one for lightest, one for heaviest',
      'Try to pair the heaviest with the lightest',
    ],
    starterCode: `from typing import List

def num_rescue_boats(people: List[int], limit: int) -> int:
    """
    Find minimum boats needed to rescue all people.
    
    Args:
        people: Array of weights
        limit: Maximum weight per boat
        
    Returns:
        Minimum number of boats needed
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2], 3],
        expected: 1,
      },
      {
        input: [[3, 2, 2, 1], 3],
        expected: 3,
      },
      {
        input: [[3, 5, 3, 4], 5],
        expected: 4,
      },
      {
        input: [[5, 1, 4, 2], 6],
        expected: 2,
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/boats-to-save-people/',
    youtubeUrl: 'https://www.youtube.com/watch?v=XbaxWuHIWUs',
  },
];
