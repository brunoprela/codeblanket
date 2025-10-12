/**
 * Arrays & Hashing problems
 */

import { Problem } from '../types';

export const arraysHashingProblems: Problem[] = [
  {
    id: 'contains-duplicate',
    title: 'Contains Duplicate',
    difficulty: 'Easy',
    description: `Given an integer array \`nums\`, return \`true\` if any value appears **at least twice** in the array, and return \`false\` if every element is distinct.

**LeetCode:** [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
**Video:** [NeetCode Explanation](https://www.youtube.com/watch?v=3OamzN90kPg)

**Approach:**
Use a hash set to track seen numbers. If you encounter a number already in the set, you've found a duplicate.`,
    examples: [
      {
        input: 'nums = [1,2,3,1]',
        output: 'true',
        explanation: 'The number 1 appears twice.',
      },
      {
        input: 'nums = [1,2,3,4]',
        output: 'false',
        explanation: 'All elements are distinct.',
      },
      {
        input: 'nums = [1,1,1,3,3,4,3,2,4,2]',
        output: 'true',
        explanation: 'Multiple numbers appear more than once.',
      },
    ],
    constraints: ['1 <= nums.length <= 10^5', '-10^9 <= nums[i] <= 10^9'],
    hints: [
      "Use a hash set to track numbers you've seen",
      'If you find a number already in the set, return true immediately',
      'Time complexity should be O(n), space complexity O(n)',
    ],
    starterCode: `from typing import List

def contains_duplicate(nums: List[int]) -> bool:
    """
    Determine if array contains any duplicates.
    
    Args:
        nums: List of integers
        
    Returns:
        True if any value appears at least twice, False otherwise
    """
    # Your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3, 1]],
        expected: true,
      },
      {
        input: [[1, 2, 3, 4]],
        expected: false,
      },
      {
        input: [[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]],
        expected: true,
      },
      {
        input: [[1]],
        expected: false,
      },
    ],
    solution: `def contains_duplicate(nums: List[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# Alternative: Use set length comparison
def contains_duplicate_alt(nums: List[int]) -> bool:
    return len(nums) != len(set(nums))`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 1,
    topic: 'Arrays & Hashing',
  },
  {
    id: 'two-sum',
    title: 'Two Sum',
    difficulty: 'Medium',
    description: `Given an array of integers \`nums\` and an integer \`target\`, return **indices** of the two numbers such that they add up to \`target\`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

You can return the answer in any order.

**LeetCode:** [1. Two Sum](https://leetcode.com/problems/two-sum/)
**Video:** [NeetCode Explanation](https://www.youtube.com/watch?v=KLlXCFG5TnA)

**Approach:**
Use a hash map to store each number and its index. For each number, check if its complement (target - num) exists in the map.`,
    examples: [
      {
        input: 'nums = [2,7,11,15], target = 9',
        output: '[0,1]',
        explanation: 'Because nums[0] + nums[1] == 9, we return [0, 1].',
      },
      {
        input: 'nums = [3,2,4], target = 6',
        output: '[1,2]',
        explanation: 'nums[1] + nums[2] == 6',
      },
      {
        input: 'nums = [3,3], target = 6',
        output: '[0,1]',
        explanation: 'Both elements sum to the target.',
      },
    ],
    constraints: [
      '2 <= nums.length <= 10^4',
      '-10^9 <= nums[i] <= 10^9',
      '-10^9 <= target <= 10^9',
      'Only one valid answer exists',
    ],
    hints: [
      'For each number, calculate what its complement should be (target - num)',
      'Use a hash map to check if the complement exists in O(1) time',
      'Store both the number and its index in the hash map',
      "Don't use the same element twice",
    ],
    starterCode: `from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Find indices of two numbers that add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        List of two indices [i, j] where nums[i] + nums[j] == target
    """
    # Your code here
    pass
`,
    testCases: [
      {
        input: [[2, 7, 11, 15], 9],
        expected: [0, 1],
      },
      {
        input: [[3, 2, 4], 6],
        expected: [1, 2],
      },
      {
        input: [[3, 3], 6],
        expected: [0, 1],
      },
      {
        input: [[1, 2, 3, 4, 5], 9],
        expected: [3, 4],
      },
    ],
    solution: `def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []  # Should never reach if input is valid`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 2,
    topic: 'Arrays & Hashing',
  },
  {
    id: 'group-anagrams',
    title: 'Group Anagrams',
    difficulty: 'Hard',
    description: `Given an array of strings \`strs\`, group **the anagrams** together. You can return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**LeetCode:** [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)
**Video:** [NeetCode Explanation](https://www.youtube.com/watch?v=vzdNOK2oB2E)

**Approach:**
Use a hash map where the key is a signature of the anagram (e.g., sorted string or character count), and the value is a list of strings matching that signature.`,
    examples: [
      {
        input: 'strs = ["eat","tea","tan","ate","nat","bat"]',
        output: '[["bat"],["nat","tan"],["ate","eat","tea"]]',
        explanation: 'All anagrams are grouped together.',
      },
      {
        input: 'strs = [""]',
        output: '[[""]]',
        explanation: 'Single empty string.',
      },
      {
        input: 'strs = ["a"]',
        output: '[["a"]]',
        explanation: 'Single character.',
      },
    ],
    constraints: [
      '1 <= strs.length <= 10^4',
      '0 <= strs[i].length <= 100',
      'strs[i] consists of lowercase English letters',
    ],
    hints: [
      'Anagrams have the same characters, just in different order',
      'Use sorted string as a key: sorted("eat") == sorted("tea") == "aet"',
      'Alternative: Use character count as key (more efficient)',
      'Use defaultdict(list) to group strings by their signature',
    ],
    starterCode: `from typing import List

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    Group strings that are anagrams of each other.
    
    Args:
        strs: List of strings
        
    Returns:
        List of groups, where each group contains anagram strings
    """
    # Your code here
    pass
`,
    testCases: [
      {
        input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
        expected: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']],
      },
      {
        input: [['']],
        expected: [['']],
      },
      {
        input: [['a']],
        expected: [['a']],
      },
      {
        input: [['abc', 'bca', 'cab', 'xyz', 'zyx', 'yxz']],
        expected: [
          ['abc', 'bca', 'cab'],
          ['xyz', 'zyx', 'yxz'],
        ],
      },
    ],
    solution: `def group_anagrams(strs: List[str]) -> List[List[str]]:
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

# Alternative: Use character count (faster for long strings)
def group_anagrams_count(strs: List[str]) -> List[List[str]]:
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        key = tuple(count)
        groups[key].append(s)
    
    return list(groups.values())`,
    timeComplexity:
      'O(n * k log k) where n is number of strings and k is max length',
    spaceComplexity: 'O(n * k)',
    order: 3,
    topic: 'Arrays & Hashing',
  },
];
