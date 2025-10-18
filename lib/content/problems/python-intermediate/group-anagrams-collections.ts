/**
 * Group Anagrams
 * Problem ID: group-anagrams-collections
 * Order: 23
 */

import { Problem } from '../../../types';

export const group_anagrams_collectionsProblem: Problem = {
  id: 'group-anagrams-collections',
  title: 'Group Anagrams',
  difficulty: 'Medium',
  category: 'python-intermediate',
  description: `Given an array of strings \`strs\`, group the anagrams together using \`defaultdict\`. You can return the answer in any order.

An **anagram** is a word formed by rearranging the letters of another, using all original letters exactly once.

**Example 1:**
\`\`\`
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
\`\`\`

**Example 2:**
\`\`\`
Input: strs = [""]
Output: [[""]]
\`\`\`

**Example 3:**
\`\`\`
Input: strs = ["a"]
Output: [["a"]]
\`\`\``,
  starterCode: `from collections import defaultdict

def group_anagrams(strs):
    """
    Group strings that are anagrams of each other.
    
    Args:
        strs: List of strings
    
    Returns:
        List of lists, each containing anagrams
    """
    pass`,
  testCases: [
    {
      input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
      expected: [['bat'], ['nat', 'tan'], ['ate', 'eat', 'tea']],
    },
    {
      input: [['']],
      expected: [['']],
    },
    {
      input: [['a']],
      expected: [['a']],
    },
  ],
  hints: [
    'Use defaultdict(list) to automatically handle new keys',
    'Sort characters in each word to create a signature',
    'Group words by their sorted signature',
  ],
  solution: `from collections import defaultdict

def group_anagrams(strs):
    """
    Group strings that are anagrams of each other.
    
    Args:
        strs: List of strings
    
    Returns:
        List of lists, each containing anagrams
    """
    # Use defaultdict to avoid KeyError
    anagram_groups = defaultdict(list)
    
    for word in strs:
        # Sort characters to create key
        # Anagrams will have same sorted key
        key = ''.join(sorted(word))
        anagram_groups[key].append(word)
    
    return list(anagram_groups.values())


# Alternative: Using tuple of character counts as key
def group_anagrams_alt(strs):
    from collections import Counter
    anagram_groups = defaultdict(list)
    
    for word in strs:
        # Use tuple of sorted counts as key
        key = tuple(sorted(Counter(word).items()))
        anagram_groups[key].append(word)
    
    return list(anagram_groups.values())`,
  timeComplexity:
    'O(n * k log k) where n is number of strings, k is max length',
  spaceComplexity: 'O(n * k)',
  order: 23,
  topic: 'Python Intermediate',
};
