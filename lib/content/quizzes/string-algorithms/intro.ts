/**
 * Quiz questions for Introduction to String Algorithms section
 */

export const introQuiz = [
  {
    id: 'q1',
    question:
      'Why is repeated string concatenation in a loop O(n²) instead of O(n)?',
    sampleAnswer:
      "Strings in Python are immutable, so each concatenation s += char creates a new string by copying all existing characters plus the new one. For n iterations: 1st copy=1, 2nd=2, 3rd=3, ..., nth=n characters. Total: 1+2+3+...+n = n (n+1)/2 = O(n²). Solution: Build a list and use '.join (list) at the end, which is O(n) since it only copies characters once.",
    keyPoints: [
      'Strings are immutable in Python',
      'Each += creates new string and copies all chars',
      'n iterations: 1+2+3+...+n = O(n²)',
      "Solution: list.append() + '.join() = O(n)",
      'join() only copies once',
    ],
  },
  {
    id: 'q2',
    question: 'What is the difference between find() and index() for strings?',
    sampleAnswer:
      "Both search for substring position, but handle not found differently: find() returns -1 when substring not found (safe, no exception), while index() raises ValueError (need try/except). Use find() when substring might not exist and you want to check: if s.find (sub) != -1. Use index() when you're certain substring exists and want exception on error. Both are O(n*m) time complexity.",
    keyPoints: [
      'find(): returns -1 if not found (safe)',
      'index(): raises ValueError if not found',
      'Both O(n*m) time complexity',
      'find() better for uncertain existence',
      'index() better when expecting to find',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common patterns for solving string problems and when should you use each?',
    sampleAnswer:
      'Four main patterns: (1) Two Pointers - use for palindromes, comparing from both ends, O(n) time and O(1) space, (2) Sliding Window - use for substring problems (longest substring without repeating chars, max vowels in k-length substring), maintains window state as it slides, O(n) time, (3) Hash Map / Frequency Count - use for anagrams, character frequency problems, group anagrams, O(n) time and space with Counter, (4) Dynamic Programming - use for longest common subsequence, edit distance, longest palindromic substring, builds table of subproblem solutions, O(n²) time and space. Choose based on problem structure: ends of string → two pointers, contiguous substring → sliding window, character counts → hash map, overlapping subproblems → DP.',
    keyPoints: [
      'Two pointers: palindromes, both ends, O(n) time O(1) space',
      'Sliding window: substring problems, O(n) time',
      'Hash map: anagrams, frequency, O(n) time and space',
      'DP: LCS, edit distance, longest palindrome, O(n²)',
      'Choose by structure: ends/middle/frequency/subproblems',
    ],
  },
];
