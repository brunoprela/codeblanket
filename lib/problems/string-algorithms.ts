import { Problem } from '../types';

export const stringAlgorithmsProblems: Problem[] = [
  {
    id: 'longest-palindromic-substring',
    title: 'Longest Palindromic Substring',
    difficulty: 'Medium',
    topic: 'String Algorithms',
    description: `Given a string \`s\`, return the longest palindromic substring in \`s\`.

**Example 1:**
\`\`\`
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
\`\`\`

**Example 2:**
\`\`\`
Input: s = "cbbd"
Output: "bb"
\`\`\`

**Constraints:**
- 1 ≤ s.length ≤ 1000
- s consist of only digits and English letters`,
    starterCode: `def longest_palindrome(s):
    """
    Find longest palindromic substring.
    
    Args:
        s: Input string
        
    Returns:
        Longest palindromic substring
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['"babad"'],
        expected: '"bab"',
      },
      {
        input: ['"cbbd"'],
        expected: ['"bb"'],
      },
      {
        input: ['"a"'],
        expected: '"a"',
      },
    ],
    timeComplexity: 'O(n²)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/longest-palindromic-substring/',
    youtubeUrl: 'https://www.youtube.com/watch?v=XYQecbcd6_c',
  },
  {
    id: 'group-anagrams-string',
    title: 'Group Anagrams',
    difficulty: 'Medium',
    topic: 'String Algorithms',
    description: `Given an array of strings \`strs\`, group the anagrams together. You can return the answer in any order.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

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
    starterCode: `def group_anagrams(strs):
    """
    Group strings that are anagrams.
    
    Args:
        strs: List of strings
        
    Returns:
        List of lists, grouped anagrams
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
        expected: '[["bat"],["nat","tan"],["ate","eat","tea"]]',
      },
      {
        input: [['']],
        expected: '[[""]]',
      },
      {
        input: [['a']],
        expected: '[["a"]]',
      },
    ],
    timeComplexity:
      'O(n * k log k) where n is strs length, k is max string length',
    spaceComplexity: 'O(n * k)',
    leetcodeUrl: 'https://leetcode.com/problems/group-anagrams/',
    youtubeUrl: 'https://www.youtube.com/watch?v=vzdNOK2oB2E',
  },
  {
    id: 'palindromic-substrings',
    title: 'Palindromic Substrings',
    difficulty: 'Medium',
    topic: 'String Algorithms',
    description: `Given a string \`s\`, return the number of **palindromic substrings** in it.

A string is a **palindrome** when it reads the same backward as forward.

A **substring** is a contiguous sequence of characters within the string.

**Example 1:**
\`\`\`
Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
\`\`\`

**Example 2:**
\`\`\`
Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
\`\`\``,
    starterCode: `def count_substrings(s):
    """
    Count palindromic substrings.
    
    Args:
        s: Input string
        
    Returns:
        Count of palindromic substrings
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['"abc"'],
        expected: 3,
      },
      {
        input: ['"aaa"'],
        expected: 6,
      },
      {
        input: ['"racecar"'],
        expected: 10,
      },
    ],
    timeComplexity: 'O(n²)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/palindromic-substrings/',
    youtubeUrl: 'https://www.youtube.com/watch?v=4RACzI5-du8',
  },
  {
    id: 'string-to-integer-atoi',
    title: 'String to Integer (atoi)',
    difficulty: 'Medium',
    topic: 'String Algorithms',
    description: `Implement the \`myAtoi(string s)\` function, which converts a string to a 32-bit signed integer (similar to C/C++'s \`atoi\` function).

The algorithm for \`myAtoi(string s)\` is as follows:

1. Read in and ignore any leading whitespace.
2. Check if the next character (if not already at the end of the string) is '-' or '+'. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.
3. Read in next the characters until the next non-digit character or the end of the input is reached. The rest of the string is ignored.
4. Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, then the integer is 0. Change the sign as necessary (from step 2).
5. If the integer is out of the 32-bit signed integer range [-2^31, 2^31 - 1], then clamp the integer so that it remains in the range. Specifically, integers less than -2^31 should be clamped to -2^31, and integers greater than 2^31 - 1 should be clamped to 2^31 - 1.
6. Return the integer as the final result.

**Example 1:**
\`\`\`
Input: s = "42"
Output: 42
\`\`\`

**Example 2:**
\`\`\`
Input: s = "   -42"
Output: -42
\`\`\`

**Example 3:**
\`\`\`
Input: s = "4193 with words"
Output: 4193
\`\`\``,
    starterCode: `def my_atoi(s):
    """
    Convert string to integer (atoi implementation).
    
    Args:
        s: Input string
        
    Returns:
        32-bit signed integer
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['"42"'],
        expected: 42,
      },
      {
        input: ['"   -42"'],
        expected: -42,
      },
      {
        input: ['"4193 with words"'],
        expected: 4193,
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/string-to-integer-atoi/',
    youtubeUrl: 'https://www.youtube.com/watch?v=LzQCMtBOdUY',
  },
  {
    id: 'implement-strstr',
    title: 'Implement strStr()',
    difficulty: 'Easy',
    topic: 'String Algorithms',
    description: `Given two strings \`needle\` and \`haystack\`, return the index of the first occurrence of \`needle\` in \`haystack\`, or \`-1\` if \`needle\` is not part of \`haystack\`.

**Clarification:**
What should we return when \`needle\` is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when \`needle\` is an empty string. This is consistent to C's strstr() and Java's indexOf().

**Example 1:**
\`\`\`
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.
\`\`\`

**Example 2:**
\`\`\`
Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
\`\`\``,
    starterCode: `def str_str(haystack, needle):
    """
    Find first occurrence of needle in haystack.
    
    Args:
        haystack: String to search in
        needle: String to find
        
    Returns:
        Index of first occurrence, or -1
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['"sadbutsad"', '"sad"'],
        expected: 0,
      },
      {
        input: ['"leetcode"', '"leeto"'],
        expected: -1,
      },
      {
        input: ['"hello"', '""'],
        expected: 0,
      },
    ],
    timeComplexity: 'O(n * m) naive, O(n + m) with KMP',
    spaceComplexity: 'O(1) naive, O(m) with KMP',
    leetcodeUrl:
      'https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Gjkhm1gYIMw',
  },
];
