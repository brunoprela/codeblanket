/**
 * Quiz questions for Space Complexity Analysis section
 */

export const spacecomplexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between auxiliary space and total space. Which one do we typically measure when analyzing space complexity?',
    sampleAnswer:
      'Total space includes everything - the input data plus any extra memory the algorithm uses. Auxiliary space is just the extra memory beyond the input. When we analyze space complexity, we typically measure auxiliary space because we want to know how much additional memory the algorithm needs. For example, if I have an array of n elements and I create a few variables to track sums and indices, my auxiliary space is O(1) even though the total space including the input is O(n). We care about auxiliary space because it tells us about the memory overhead of our algorithm, not just the memory needed to store the data we were given.',
    keyPoints: [
      'Total space = input + extra memory',
      'Auxiliary space = extra memory only',
      'We typically measure auxiliary space',
      'It shows the overhead of the algorithm',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through why a recursive function has space complexity related to the depth of recursion. Use factorial as an example.',
    hint: 'Think about the call stack.',
    sampleAnswer:
      'When you call factorial(5), it calls factorial(4), which calls factorial(3), and so on down to factorial(1). Each of these function calls gets added to the call stack and stays there until it can return. So at the deepest point, you have 5 stack frames sitting in memory: factorial(5) waiting for factorial(4), factorial(4) waiting for factorial(3), all the way down. This is O(n) space where n is the input number. The depth of the recursion determines how many stack frames accumulate. This is different from iterative solutions which can use the same small amount of memory repeatedly. The call stack is often overlooked but is crucial for space complexity analysis of recursive algorithms.',
    keyPoints: [
      'Each recursive call adds a frame to the call stack',
      'Stack frames accumulate until base case is reached',
      'Maximum stack depth = space complexity',
      'factorial(n) has n frames → O(n) space',
    ],
  },
  {
    id: 'q3',
    question:
      'What is a time-space tradeoff? Give me an example where you would intentionally use more space to save time.',
    sampleAnswer:
      'A time-space tradeoff is when you sacrifice memory to gain speed, or vice versa. A classic example is memoization - storing previously computed results to avoid recalculating them. In Fibonacci, the naive recursive solution is O(2^n) time because it recalculates the same values over and over. By using a hash map to cache results, we can bring it down to O(n) time, but now we use O(n) extra space for the cache. We are trading space for a massive speed improvement. This tradeoff makes sense when speed is more important than memory, which is often the case. Another example is creating an index on a database - it uses more disk space but makes queries much faster.',
    keyPoints: [
      'Trade memory for speed (or vice versa)',
      'Memoization: use O(n) space to improve time dramatically',
      'Fibonacci: O(2^n) → O(n) time by using O(n) space',
      'Common when speed matters more than memory',
    ],
  },
];
