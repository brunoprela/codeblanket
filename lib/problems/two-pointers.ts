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
  },
];
