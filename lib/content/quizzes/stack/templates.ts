/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the monotonic stack template. When do you pop and when do you push?',
    sampleAnswer:
      'In the monotonic stack template, I iterate through the array and maintain a stack in either increasing or decreasing order. For next greater element (decreasing stack), while the current element is larger than stack top, I pop the stack - these popped elements have found their next greater element. After popping what needs to be popped, I push the current element onto the stack. The key decision: pop while current violates monotonic property, then push current. For next smaller element, I would use an increasing stack and pop when current is smaller. The template is: for each element, pop while condition met, process popped elements, push current element.',
    keyPoints: [
      'Maintain increasing or decreasing order',
      'Pop while current violates monotonic property',
      'Popped elements found their answer',
      'Push current after popping',
      'Next greater: decreasing stack, Next smaller: increasing stack',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the sentinel technique for simplifying stack code. Why does adding dummy values at boundaries help?',
    sampleAnswer:
      'Sentinels are dummy values added at array boundaries to avoid special case handling. For example, in largest rectangle histogram, adding 0 height bars at start and end ensures all bars get processed without extra code. When we encounter the sentinel 0 at the end, it is smaller than any real bar, forcing all remaining bars to be popped and computed. This eliminates the need for a separate loop after the main iteration to handle leftover stack elements. Sentinels simplify code by making edge cases behave like normal cases. The trade-off is slightly more memory but much cleaner logic. Common in monotonic stack problems.',
    keyPoints: [
      'Dummy values at array boundaries',
      'Avoid special case handling',
      'Force final processing of stack',
      'Edge cases behave like normal cases',
      'Trade: slight memory for cleaner code',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare storing values vs storing indices in the stack. When would you choose each approach?',
    sampleAnswer:
      'I store indices when I need to calculate positions, distances, or widths. For example, largest rectangle histogram stores indices so I can compute width = current index minus popped index. Next greater element stores indices so I can fill the result array at the correct position. I store values when I only need comparisons and do not care about positions. For example, valid parentheses just stores bracket characters since I only need to match types, not track positions. The rule: if you need to reference back to array position or compute distances, store indices. If you only need comparisons or value matching, store values directly.',
    keyPoints: [
      'Indices: when need positions, distances, widths',
      'Example: histogram width = index difference',
      'Values: when only need comparisons',
      'Example: parentheses matching',
      'Rule: need position info → indices, need value comparison → values',
    ],
  },
];
