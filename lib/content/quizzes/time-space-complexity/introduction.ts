/**
 * Quiz questions for What is Complexity Analysis? section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain in your own words what we mean when we say an algorithm is O(n). What does the "n" represent and why do we care about it?',
    hint: 'Think about what happens when you double the input size.',
    sampleAnswer:
      'When we say an algorithm is O(n), we mean that the number of operations grows linearly with the input size. The "n" represents the size of the input - like the number of elements in an array. If I have 10 elements and it takes 10 operations, then with 100 elements it will take roughly 100 operations, and with 1000 elements it will take roughly 1000 operations. We care about this because it tells us how the algorithm will perform as our data grows. An O(n) algorithm scales much better than O(n²), especially with large datasets. It helps us predict performance and make smart choices about which algorithm to use.',
    keyPoints: [
      'O(n) means operations grow linearly with input size',
      'n represents the input size (e.g., array length)',
      'Doubling input size roughly doubles the work',
      'Helps predict scalability and make design decisions',
    ],
  },
  {
    id: 'q2',
    question:
      'Why do we drop constants in Big O notation? For example, why is O(2n) just written as O(n)?',
    sampleAnswer:
      'We drop constants because Big O is about understanding growth rates, not exact measurements. Whether an algorithm does n operations or 2n operations, both grow linearly - they both scale at the same rate. When n is 1 million, the difference between n and 2n is just a constant factor of 2, but both are vastly better than n² which would be 1 trillion operations. Constants like 2 or 100 can be affected by hardware, programming language, and implementation details. Big O abstracts away these details to focus on the fundamental scaling behavior. So O(2n), O(5n), and O(n + 1000) are all just O(n) because they all grow linearly.',
    keyPoints: [
      'Big O focuses on growth rate, not exact operation counts',
      'Constants become less significant as n grows large',
      'Both O(n) and O(2n) scale linearly',
      'We care about order of magnitude differences',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the difference between time complexity and space complexity? Can an algorithm have different complexities for each?',
    hint: 'Think about what resources an algorithm uses.',
    sampleAnswer:
      'Time complexity measures how many operations an algorithm performs as input size grows, while space complexity measures how much memory it uses. They are independent - you can have an algorithm that is fast but uses a lot of memory, or one that is slow but memory-efficient. For example, recursive algorithms often trade space for elegant code - they might be O(n) time but also O(n) space due to the call stack. Or you might use memoization to speed up an algorithm from O(2^n) to O(n) time, but it costs you O(n) extra space to store the cache. Understanding both helps you make informed tradeoffs based on your constraints.',
    keyPoints: [
      'Time complexity: number of operations',
      'Space complexity: amount of memory used',
      'They are independent - can differ for same algorithm',
      'Often there is a time-space tradeoff',
    ],
  },
];
