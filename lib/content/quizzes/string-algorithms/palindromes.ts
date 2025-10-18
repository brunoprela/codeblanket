/**
 * Quiz questions for Palindrome Patterns section
 */

export const palindromesQuiz = [
  {
    id: 'q-pal1',
    question:
      'Explain the expand-around-center technique for finding palindromic substrings.',
    sampleAnswer:
      'Expand-around-center treats each character (and each pair of characters) as a potential palindrome center, then expands outward while characters match. For each position i, expand from (i,i) for odd-length palindromes and (i,i+1) for even-length. This finds all palindromic substrings in O(n²) time with O(1) space. Example: "aba" - from center \'b\', expand to find the entire palindrome.',
    keyPoints: [
      'Two cases: odd-length (single center) and even-length (two centers)',
      'Expand while left == right',
      'O(n²) time: n centers × O(n) expansion',
      'O(1) space - no extra arrays needed',
      'Better than brute force O(n³)',
    ],
  },
  {
    id: 'q-pal2',
    question:
      'How would you efficiently check if a string is a palindrome ignoring non-alphanumeric characters?',
    sampleAnswer:
      'Use two pointers from both ends, skipping non-alphanumeric characters: left=0, right=len-1. While left < right: skip non-alnum at left, skip at right, compare lower case. If mismatch, return False. This is O(n) time, O(1) space. Better than cleaning the string first (which takes O(n) extra space).',
    keyPoints: [
      'Two pointers: left and right',
      'Skip non-alphanumeric: if not s[left].isalnum(): left += 1',
      'Compare case-insensitive: s[left].lower() == s[right].lower()',
      'O(n) time, O(1) space',
      'More efficient than preprocessing',
    ],
  },
  {
    id: 'q-pal3',
    question:
      'What is the difference between palindromic substring and palindromic subsequence?',
    sampleAnswer:
      'Substring must be contiguous - consecutive characters in order (e.g., "aba" in "xabay"). Subsequence can skip characters but must maintain order (e.g., "aba" from "a__b__a" where underscores are skipped chars). Finding longest palindromic substring is O(n²) with expand-around-center. Finding longest palindromic subsequence needs DP: O(n²) time and space, similar to LCS.',
    keyPoints: [
      'Substring: contiguous characters',
      'Subsequence: can skip characters, maintain order',
      'Substring example: "aba" in "xabay"',
      'Subsequence example: "aba" in "aebfca"',
      'Different algorithms: expand-around vs DP',
    ],
  },
];
