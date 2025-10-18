/**
 * Quiz questions for Optimization Strategies & Trade-offs section
 */

export const optimizationQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through how using a hash map can turn an O(n²) algorithm into O(n). Use the two-sum problem as your example.',
    sampleAnswer:
      'In the naive two-sum approach, you use nested loops - for each element, you check every other element to see if they add up to the target. That is n × n = O(n²) comparisons. With a hash map, you can do it in one pass. As you iterate through the array, you check if (target - current_number) exists in the hash map. If it does, you found your pair. If not, you add the current number to the map and continue. Hash map lookups are O(1), so you are doing O(1) work for each of n elements, giving O(n) total. The hash map trades O(n) extra space for a massive speedup from O(n²) to O(n) time.',
    keyPoints: [
      'Naive: nested loops → O(n²)',
      'Optimized: hash map for O(1) lookups',
      'One pass, checking (target - current) in map',
      'Trade O(n) space for O(n²) → O(n) time improvement',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain what memoization is and why it can dramatically improve recursive algorithms. What is the trade-off?',
    sampleAnswer:
      'Memoization is caching the results of function calls so you do not recompute them. In recursive algorithms, you often calculate the same subproblems many times - for example, fibonacci(5) calls fibonacci(3) multiple times. With memoization, the first time you calculate fibonacci(3), you store the result in a cache. Next time you need it, you just look it up in O(1) time instead of recalculating. For Fibonacci, this turns O(2^n) exponential time into O(n) linear time because you only calculate each value once. The trade-off is that you use O(n) extra space for the cache. But going from exponential to linear time is usually worth it - the speed improvement is massive.',
    keyPoints: [
      'Memoization: cache function results to avoid recomputation',
      'Recursive algorithms often recalculate same subproblems',
      'Store result first time, look up next time',
      'Trade-off: O(n) space for exponential → linear time improvement',
    ],
  },
  {
    id: 'q3',
    question:
      'When would you choose an O(n log n) algorithm over an O(n²) algorithm? Is the O(n log n) algorithm always better?',
    sampleAnswer:
      'O(n log n) is asymptotically better - it scales much better as n grows large. For 1000 elements, O(n²) is a million operations while O(n log n) is about 10,000 - a hundred times faster. However, for very small inputs, the O(n²) algorithm might actually be faster because it has lower constant factors. For example, insertion sort (O(n²)) can beat merge sort (O(n log n)) on arrays of size 10-20 because it has less overhead. Also, insertion sort is O(n) on already-sorted data. So you need to consider: 1) How large is n typically? 2) What are the constant factors? 3) What is the distribution of input? For large n, O(n log n) almost always wins.',
    keyPoints: [
      'O(n log n) scales much better for large n',
      'But may have higher constants/overhead',
      'For small n, O(n²) might be faster in practice',
      'Consider input size and characteristics',
    ],
  },
];
