/**
 * Quiz questions for Common Recursion Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question: 'What is the key difference between linear and binary recursion?',
    sampleAnswer:
      'Linear recursion makes exactly one recursive call per function invocation, processing elements one at a time (e.g., sum of array). Binary recursion makes two recursive calls, typically dividing the problem in half (e.g., Fibonacci, binary tree traversal). Linear recursion has O(n) call stack depth for n elements. Binary recursion can have exponential time complexity if not optimized (like naive Fibonacci), but can be efficient with divide-and-conquer algorithms like merge sort.',
    keyPoints: [
      'Linear: one recursive call',
      'Binary: two recursive calls',
      'Linear: process one-by-one',
      'Binary: divide and conquer',
      'Different complexity implications',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the helper function pattern in recursion and when you should use it.',
    sampleAnswer:
      "The helper function pattern involves creating an inner recursive function with extra parameters (like indices, accumulators, or state) while keeping the outer function's signature clean. For example, checking if a string is a palindrome: the outer function is_palindrome(s) has a simple signature, but the inner helper(left, right) tracks left and right pointers. Use this pattern when: (1) You need to track extra state (indices, accumulator, visited nodes), (2) The recursive function needs more parameters than the user should provide, (3) You want a clean public API. Benefits: cleaner interface, separation of concerns, easier to use.",
    keyPoints: [
      'Inner recursive function with extra params',
      'Outer function has clean signature',
      'Use when: need extra state tracking',
      'Examples: palindrome check, range-based problems',
      'Benefits: clean API, encapsulation',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is naive Fibonacci O(2^n) and how does this relate to binary recursion?',
    sampleAnswer:
      'Naive Fibonacci is O(2^n) because each call makes two recursive calls (binary recursion), creating an exponential tree of calls. For fib(n), we call both fib(n-1) and fib(n-2). This creates redundant work: fib(3) is calculated many times in fib(5). The recursion tree grows exponentially: depth is n, and each level can have up to 2^level nodes. Total calls ≈ 2^n. This is why naive Fibonacci is extremely slow for large n. Solution: memoization caches results, making it O(n) by eliminating redundant calculations. This shows that binary recursion can be inefficient without optimization.',
    keyPoints: [
      'Each call makes 2 recursive calls',
      'Creates exponential tree of calls',
      'Massive redundant work (fib(3) computed many times)',
      'Depth n, up to 2^n total calls',
      'Fix: memoization reduces to O(n)',
    ],
  },
];
