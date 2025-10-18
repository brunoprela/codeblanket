/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the fixed-size window template. What is the key pattern for initialization and sliding?',
    sampleAnswer:
      'For fixed-size windows of size k, the template is: first, build the initial window by iterating 0 to k-1 and computing the initial sum or state. Second, start sliding from index k: for each new position, add the element entering at right and subtract the element leaving at left. Update the answer if current window is better. The key pattern: one loop to build initial window, one loop starting at k to slide. Each slide is constant time: subtract left, add right, check answer. This avoids recalculating the entire window each time. Common for problems like max sum of k elements, average of k elements, or any fixed-size aggregate.',
    keyPoints: [
      'Build initial window: loop 0 to k-1',
      'Slide from index k onward',
      'Each slide: subtract left, add right',
      'Update answer each iteration',
      'Avoid recalculating entire window',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the variable-size shrinkable window template. What goes in the outer loop vs the inner while loop?',
    sampleAnswer:
      'The shrinkable template has right pointer in outer for loop, left pointer in inner while loop. Outer loop: for right in range(n), add arr[right] to window, update state. Inner while loop: while condition violated, remove arr[left] from window, increment left. After inner while, check and update answer using current window. The key structure: unconditionally expand right, conditionally shrink left while needed. This ensures we explore all possible windows and the inner while maintains validity. Used for maximum window problems where we want largest valid window, like longest substring without repeating characters or max consecutive ones after k flips.',
    keyPoints: [
      'Outer for: right pointer, expand',
      'Inner while: left pointer, shrink',
      'Add right unconditionally',
      'Shrink left while condition breaks',
      'Update answer after shrinking',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the shrinkable vs non-shrinkable templates. How does changing while to if affect the solution?',
    sampleAnswer:
      'Shrinkable uses "while condition violated" to move left - shrinks as much as needed to restore validity. Non-shrinkable uses "if condition violated" to move left once - maintains max window size achieved. For example, longest substring with k distinct: shrinkable shrinks fully when exceeding k distinct, exploring all valid windows. Non-shrinkable moves left just once when exceeding k, keeping window size at maximum found so far and just sliding forward. Shrinkable finds exact maximum and tracks all valid windows. Non-shrinkable is optimization that works when we only need final maximum size and can skip smaller valid windows. While gives thorough exploration, if gives efficient maximum tracking.',
    keyPoints: [
      'While: shrink fully, restore validity',
      'If: move once, maintain max size',
      'Shrinkable: explore all valid windows',
      'Non-shrinkable: track maximum, skip smaller',
      'While for thorough, if for optimization',
    ],
  },
];
