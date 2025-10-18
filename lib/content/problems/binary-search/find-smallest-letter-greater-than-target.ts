/**
 * Find Smallest Letter Greater Than Target
 * Problem ID: find-smallest-letter-greater-than-target
 * Order: 7
 */

import { Problem } from '../../../types';

export const find_smallest_letter_greater_than_targetProblem: Problem = {
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
      explanation: 'No character is greater than "z" so we wrap around to "x".',
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
};
