/**
 * Quiz questions for Time & Space Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Compare the time complexity of checking if an element exists: linear search in array vs hash table lookup. Why is there such a difference?',
    sampleAnswer:
      'Linear search in an array is O(n) because in the worst case I might have to check every element until I find it or reach the end. A hash table lookup is O(1) average case because I hash the key to get the index and jump directly there - no scanning needed. The massive difference comes from random access based on a computed index versus sequential checking. This is why hash tables are so powerful for existence checks, frequency counting, and lookups. However, arrays have better cache performance and no hash function overhead, so for very small data sets or when elements are likely near the beginning, arrays can be faster in practice.',
    keyPoints: [
      'Array linear search: O(n) - must scan',
      'Hash table: O(1) - direct index jump',
      'Computed index vs sequential search',
      'Hash tables win for large data',
      'Arrays can be faster for tiny data',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the space-time tradeoff when using hash tables. When is it worth it and when is it not?',
    sampleAnswer:
      'Hash tables trade space for speed. We use O(n) extra memory to store our hash map, but gain O(1) lookups instead of O(n) searches. This turns many O(n²) nested loop solutions into O(n) time. It is absolutely worth it when lookup speed is critical and memory is available - like in coding interviews where optimality matters. It is not worth it when memory is severely constrained, when the dataset is tiny and O(n) is fine, or when you are only doing one lookup (the setup cost is not amortized). Also not worth it if you need to maintain order or need the minimum/maximum element frequently. Always consider: will the speed gain justify the memory cost?',
    keyPoints: [
      'Trade O(n) space for O(1) lookups',
      'Turns O(n²) into O(n) solutions',
      'Worth it: when speed critical, memory available',
      'Not worth it: memory constrained, tiny data, single lookup',
      'Consider if speed gain justifies memory cost',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk through the complexity of removing duplicates: with hash set vs without. What makes the hash set approach faster?',
    sampleAnswer:
      'Without hash set, for each element I would need to check if it has appeared before by scanning previous elements - that is O(n) per element, giving O(n²) total. With a hash set, as I iterate through the array, I check if current element is in the set in O(1), add it to the set if not, and skip it if yes. This is O(n) time total. The hash set remembers what I have seen with instant lookup, eliminating the need to search back through processed elements each time. The cost is O(n) space for the hash set, but the time improvement from O(n²) to O(n) is usually worth it. This is a classic example of trading space for time.',
    keyPoints: [
      'Without hash set: O(n²) - check previous elements each time',
      'With hash set: O(n) - instant membership check',
      'Hash set remembers seen elements',
      'Cost: O(n) space',
      'Classic space-time tradeoff',
    ],
  },
];
