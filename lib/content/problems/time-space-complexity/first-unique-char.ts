/**
 * First Unique Character in a String
 * Problem ID: first-unique-char
 * Order: 3
 */

import { Problem } from '../../../types';

export const first_unique_charProblem: Problem = {
  id: 'first-unique-char',
  title: 'First Unique Character in a String',
  difficulty: 'Easy',
  topic: 'Time & Space Complexity',
  order: 3,
  description: `Given a string \`s\`, find the first non-repeating character in it and return its index. If it does not exist, return \`-1\`.

**Complexity Analysis:** What's the time and space complexity of using a hash map?`,
  examples: [
    {
      input: 's = "leetcode"',
      output: '0',
    },
    {
      input: 's = "loveleetcode"',
      output: '2',
    },
    {
      input: 's = "aabb"',
      output: '-1',
    },
  ],
  constraints: [
    '1 <= s.length <= 10^5',
    's consists of only lowercase English letters',
  ],
  hints: [
    'First pass: count character frequencies',
    'Second pass: find first character with count = 1',
    'Can you do it in O(n) time with O(1) space? (26 letters is constant!)',
  ],
  starterCode: `def first_uniq_char(s: str) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['leetcode'],
      expected: 0,
    },
    {
      input: ['loveleetcode'],
      expected: 2,
    },
    {
      input: ['aabb'],
      expected: -1,
    },
    {
      input: ['z'],
      expected: 0,
    },
    {
      input: ['aabbccddee'],
      expected: -1,
    },
  ],
  solution: `# Optimal: Hash Map - O(n) time, O(1) space (26 letters max)
def first_uniq_char(s):
    # Count frequencies
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Find first unique
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i
    
    return -1

# Using Counter
from collections import Counter
def first_uniq_char_counter(s):
    count = Counter(s)
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    return -1

# Naive: Check each character - O(n²) time
def first_uniq_char_naive(s):
    for i in range(len(s)):
        is_unique = True
        for j in range(len(s)):
            if i != j and s[i] == s[j]:
                is_unique = False
                break
        if is_unique:
            return i
    return -1
`,
  timeComplexity:
    'O(n) with hash map vs O(n²) naive - two passes vs nested loops',
  spaceComplexity: 'O(1) - at most 26 lowercase English letters in hash map',
  leetcodeUrl:
    'https://leetcode.com/problems/first-unique-character-in-a-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=5co5Gvp_-S0',
};
