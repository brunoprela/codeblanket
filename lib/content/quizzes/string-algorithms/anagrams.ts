/**
 * Quiz questions for Anagram Patterns section
 */

export const anagramsQuiz = [
  {
    id: 'q-ana1',
    question:
      'What is the most efficient way to check if two strings are anagrams?',
    sampleAnswer:
      "Use a frequency counter (hash map or array). Count characters in first string, then decrement for second string. If all counts are zero, they're anagrams. Time: O(n+m), Space: O(1) for fixed alphabet or O(k) for unique chars. This is better than sorting O(n log n). For lowercase only, use 26-element array: counts[ord(c) - ord('a')] for O(1) lookups.",
    keyPoints: [
      'Hash map or array for character frequencies',
      'Count first string, decrement for second',
      'O(n+m) time vs O(n log n) for sorting',
      'O(1) space for fixed alphabet (26 letters)',
      'Check if all counts return to zero',
    ],
  },
  {
    id: 'q-ana2',
    question:
      'How would you find all anagrams of a pattern in a string (sliding window)?',
    sampleAnswer:
      'Use sliding window with character frequency map. Create pattern frequency map. Slide a window of pattern.length through string, maintaining window frequency. When window frequency matches pattern frequency, record start index. Optimization: track "matches" count - when matches == 26 (or unique chars), it\'s an anagram. Time: O(n), Space: O(1) for fixed alphabet.',
    keyPoints: [
      'Sliding window of length = pattern.length',
      'Maintain frequency map for current window',
      'Compare window frequency with pattern frequency',
      'Track matches count for optimization',
      'O(n) time with single pass',
    ],
  },
  {
    id: 'q-ana3',
    question:
      "What's the difference between checking anagrams with sorting vs hash map?",
    sampleAnswer:
      'Sorting: Sort both strings, compare if equal. Time: O(n log n), Space: O(1) or O(n) depending on sort. Hash map: Count character frequencies, compare counts. Time: O(n), Space: O(k) for k unique chars. Hash map is faster for long strings. Sorting is simpler to implement and works for any characters. For interviews, mention both but implement hash map for better complexity.',
    keyPoints: [
      'Sorting: O(n log n) time, simpler code',
      'Hash map: O(n) time, better complexity',
      'Sorting: sort both and compare',
      'Hash map: count and compare frequencies',
      'Trade-off: simplicity vs performance',
    ],
  },
];
