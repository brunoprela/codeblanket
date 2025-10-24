/**
 * Group Anagrams
 * Problem ID: group-anagrams
 * Order: 3
 */

import { Problem } from '../../../types';

export const group_anagramsProblem: Problem = {
  id: 'group-anagrams',
  title: 'Group Anagrams',
  difficulty: 'Hard',
  description: `Given an array of strings \`strs\`, group **the anagrams** together. You can return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.


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
        key = '.join(sorted(s))
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

  leetcodeUrl: 'https://leetcode.com/problems/group-anagrams/',
  youtubeUrl: 'https://www.youtube.com/watch?v=vzdNOK2oB2E',
  order: 3,
  topic: 'Arrays & Hashing',
};
