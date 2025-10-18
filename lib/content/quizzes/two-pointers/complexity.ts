/**
 * Quiz questions for Time & Space Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain why two pointers is O(n) time complexity, not O(n²). What is the key insight about how the pointers move?',
    sampleAnswer:
      'Two pointers is O(n) because each pointer traverses the array at most once - they never backtrack. Even though we have two pointers, they move a combined total of at most n steps. Think of it this way: in opposite direction, they start n apart and meet in the middle, covering n positions total. In same direction, fast pointer does at most n moves, and slow pointer does at most n moves, but we process each element once. The key is that we never revisit elements or check the same pairs multiple times. Compare this to nested loops where the inner loop resets for each outer loop iteration, giving n × n checks.',
    keyPoints: [
      'Each pointer moves at most n times',
      'Combined movement is O(n), not O(n) + O(n) = O(2n) = O(n)',
      'Never backtrack or revisit elements',
      'Process each element once',
      'Unlike nested loops that reset inner loop',
    ],
  },
  {
    id: 'q2',
    question:
      'Talk about the space complexity of two pointers. Why is it often better than other approaches?',
    sampleAnswer:
      'Two pointers typically has O(1) space complexity because we only need a few extra variables - the two pointer positions and maybe a couple tracking variables. We are not creating any data structures that grow with input size. This is especially powerful when doing in-place modifications like removing duplicates - we transform the array using just a couple of pointers without needing a separate result array. Compare this to approaches that use hash maps (O(n) space) or store intermediate results. The in-place nature of two pointers makes it very memory efficient, which matters for large datasets or memory-constrained environments.',
    keyPoints: [
      'Usually O(1) space - just pointer variables',
      'No data structures that grow with input',
      'In-place modifications possible',
      'More memory efficient than hash map approaches',
      'Great for large datasets',
    ],
  },
  {
    id: 'q3',
    question:
      'Some problems can be solved with either two pointers or a hash map. How would you decide which approach to use?',
    sampleAnswer:
      'I consider several factors. If the array is already sorted or I can sort it without breaking requirements, two pointers is often cleaner and uses O(1) space versus O(n) for a hash map. Two pointers is also better when I need to modify in-place or when I care about memory. However, hash map wins when sorting would break the problem (like needing to return original indices in unsorted order), or when the problem needs more complex lookups than just comparing two elements. Hash map gives O(1) lookup but uses more space. If space is tight and data is sorted, two pointers. If I need fast arbitrary lookups and space is not an issue, hash map.',
    keyPoints: [
      'Sorted/sortable data → two pointers (O(1) space)',
      'Need original indices/order → hash map',
      'In-place modification needed → two pointers',
      'Complex lookups needed → hash map',
      'Trade-off: space vs convenience',
    ],
  },
];
