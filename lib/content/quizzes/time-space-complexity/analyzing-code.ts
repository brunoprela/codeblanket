/**
 * Quiz questions for How to Analyze Code Complexity section
 */

export const analyzingcodeQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through analyzing this code: def func (arr): for i in range (len (arr)): for j in range (i): print(arr[i], arr[j]). What is the time complexity?',
    hint: 'The inner loop does not run n times for each outer iteration.',
    sampleAnswer:
      'This is still O(n²), but let me explain why. The outer loop runs n times. The inner loop runs a variable number of times - it runs 0 times when i=0, 1 time when i=1, 2 times when i=2, and so on up to n-1 times. So the total number of prints is 0 + 1 + 2 + ... + (n-1), which is the sum of first n-1 numbers. That equals (n-1) × n / 2, which is roughly n² / 2. When we drop constants, n² / 2 becomes O(n²). Even though this does fewer operations than a standard nested loop, it is still quadratic growth. The key insight is that the sum of 1 to n is proportional to n².',
    keyPoints: [
      'Outer loop: n iterations',
      'Inner loop: 0, 1, 2, ..., n-1 iterations',
      'Total operations: 0 + 1 + 2 + ... + (n-1) = n (n-1)/2',
      'Simplifies to O(n²) - still quadratic',
    ],
  },
  {
    id: 'q2',
    question:
      'How would you analyze the space complexity of merge sort? What data structures contribute to it?',
    sampleAnswer:
      'Merge sort has O(n) space complexity. There are two main contributors. First, in the merge step, we create temporary arrays to hold the left and right halves, and these combined are O(n) space. Second, there is the recursive call stack - merge sort divides the array in half each time, so the maximum depth of recursion is log n, which is O(log n) space. Overall, the dominant factor is the O(n) space for the temporary merge arrays. In some implementations, you can reuse the same temporary array, but you still need O(n) auxiliary space. This is different from in-place sorts like heap sort which are O(1) space.',
    keyPoints: [
      'Temporary merge arrays: O(n) space',
      'Recursive call stack: O(log n) depth',
      'Dominant factor: O(n) auxiliary space',
      'Not an in-place sort',
    ],
  },
  {
    id: 'q3',
    question:
      'If you see sorting inside a loop, how does that affect the overall time complexity? Give me an example.',
    hint: 'Sorting is not O(n).',
    sampleAnswer:
      'If you sort inside a loop, you multiply the complexities. For example, if you have a loop that runs n times, and inside that loop you sort an array of size n, the overall complexity is O(n × n log n) = O(n² log n). This is worse than just O(n²). A concrete example would be: for each element in an array, create a sub-array, sort it, and do something with it. The outer loop is O(n), and sorting each sub-array is O(n log n), so total is O(n² log n). This is a common mistake - people forget that sorting is expensive and should be done carefully, not repeatedly inside loops if it can be avoided.',
    keyPoints: [
      'Multiply complexities: loop × sorting',
      'Loop of n × sorting of n → O(n × n log n) = O(n² log n)',
      'Sorting is O(n log n), not O(n)',
      'Avoid sorting inside loops when possible',
    ],
  },
];
