/**
 * Quiz questions for Arrays & Hash Tables: The Foundation section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain why arrays and hash tables are considered the foundation of data structures. What makes them so fundamental?',
    sampleAnswer:
      'Arrays are the most basic data structure - just a contiguous block of memory with O(1) index access. They are fundamental because many other data structures like strings, stacks, queues, and even heaps are built on top of arrays. Hash tables are fundamental because they solve a critical problem: fast lookup. With O(1) average-case access, hash tables let us trade space for speed, turning many O(n²) brute force solutions into O(n). Together, they give us the two most common patterns: iterate through data with arrays, and instantly look up information with hash tables. About 30-40% of interview problems use these two structures.',
    keyPoints: [
      'Arrays: basic building block with O(1) index access',
      'Many structures built on arrays',
      'Hash tables: O(1) lookup trades space for speed',
      'Turn O(n²) into O(n) solutions',
      'Appear in 30-40% of interview problems',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through how hash tables enable optimization from O(n²) to O(n). Give me a concrete example.',
    sampleAnswer:
      'Take the two sum problem: given an array, find two numbers that add to a target. Brute force would check every pair with nested loops - O(n²). With a hash table, as I iterate through the array once, for each number I check if target minus that number exists in my hash table. If yes, found it. If no, I add the current number to the hash table and continue. This is one pass through the array with O(1) lookups, so O(n) total. The hash table remembers what I have seen so far, eliminating the need to search back through previous elements. We trade O(n) space for a massive time improvement.',
    keyPoints: [
      'Example: two sum O(n²) brute force',
      'Hash table stores seen elements',
      'Check if complement exists in O(1)',
      'Single pass instead of nested loops',
      'Trade O(n) space for O(n) time',
    ],
  },
  {
    id: 'q3',
    question:
      'Arrays and hash tables each have trade-offs. Talk about when you would choose one over the other.',
    sampleAnswer:
      'I choose arrays when I need ordered data, when I am accessing by position/index, or when memory is extremely tight since arrays have no overhead. Arrays are great for sequential processing and when the size is known. I choose hash tables when I need fast lookups by key rather than by position, when I am checking existence or counting frequencies, or when I need to group or deduplicate data. Hash tables win when lookup speed is critical and I am okay with extra memory overhead. For example, checking if a number exists: array is O(n) scan, hash table is O(1) lookup. The trade-off is always speed versus memory and whether order matters.',
    keyPoints: [
      'Arrays: ordered, index access, low memory overhead',
      'Arrays: when size known, sequential processing',
      'Hash tables: fast key lookup, existence checks',
      'Hash tables: grouping, counting, deduplication',
      'Trade-off: speed vs memory, order vs flexibility',
    ],
  },
];
