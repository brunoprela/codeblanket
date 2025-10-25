/**
 * Quiz questions for Understanding Big O Notation section
 */

export const bigonotationQuiz = [
  {
    id: 'q1',
    question:
      'Look at this code: for i in range (n): for j in range (n): print(i, j). What is the time complexity and why?',
    hint: 'Count how many times print() gets called.',
    sampleAnswer:
      'This is O(n²) - quadratic time. The outer loop runs n times, and for each iteration of the outer loop, the inner loop also runs n times. So the print statement executes n × n = n² times total. If n is 10, we print 100 times. If n is 100, we print 10,000 times. The number of operations grows with the square of the input. This is the signature pattern of nested loops where both iterate n times. Nested loops are a red flag for quadratic complexity.',
    keyPoints: [
      'Nested loops over same input size → O(n²)',
      'Outer loop: n times, inner loop: n times each',
      'Total operations: n × n = n²',
      'Quadratic growth - gets slow with large inputs',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is binary search O(log n) and not O(n)? Walk me through the mathematical reasoning.',
    sampleAnswer:
      'Binary search is O(log n) because with each comparison, we eliminate half the remaining elements. If we start with 1000 elements, after one check we have 500 left, then 250, then 125, then 63, and so on. The question is: how many times can we divide n by 2 until we get down to 1? That is exactly what log₂(n) tells us. For 1000 elements, log₂(1000) is about 10, so we need at most 10 comparisons. For 1 million elements, we only need about 20 comparisons. This logarithmic growth is why binary search is so incredibly efficient compared to linear search which would need to check every element.',
    keyPoints: [
      'Each step eliminates half the search space',
      'Pattern: n → n/2 → n/4 → ... → 1',
      'Number of halvings = log₂(n)',
      'For 1 million items, only ~20 operations needed',
    ],
  },
  {
    id: 'q3',
    question:
      'What complexity would you use to describe checking if two strings are anagrams? Walk through your reasoning.',
    sampleAnswer:
      'The best approach is O(n) where n is the length of the strings. You can count the frequency of each character in both strings using hash maps, which takes O(n) time for each string. Then compare the frequency maps, which takes O(1) for each of the 26 letters (constant). So overall it is O(n) + O(n) + O(26) = O(2n + 26) which simplifies to O(n). You could also sort both strings and compare them, but that would be O(n log n) due to sorting. The hash map approach is better because it is linear. The key insight is that the work scales linearly with string length.',
    keyPoints: [
      'Count character frequencies in both strings',
      'Each character count pass: O(n)',
      'Comparing frequencies: O(1) for fixed alphabet',
      'Total: O(n) - linear in string length',
    ],
  },
];
