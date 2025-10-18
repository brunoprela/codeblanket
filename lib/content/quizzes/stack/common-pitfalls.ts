/**
 * Quiz questions for Common Pitfalls section
 */

export const commonpitfallsQuiz = [
  {
    id: 'q1',
    question:
      'What happens if you forget to check if the stack is empty before popping or peeking? How do you prevent this error?',
    sampleAnswer:
      'Forgetting to check for empty stack causes a runtime error - in Python, IndexError when accessing stack[-1] or stack.pop() on empty list. This typically happens when processing closing brackets without matching opening brackets, or when popping in monotonic stack without verifying stack has elements. I prevent this by always checking "if stack:" or "if not stack:" before accessing stack top or popping. In matching problems, an empty stack when I need to pop means invalid input. In monotonic stack, I use "while stack and condition" to ensure stack has elements before comparing. The pattern: always guard stack access with empty check.',
    keyPoints: [
      'Empty pop/peek causes IndexError',
      'Common in: unmatched brackets, monotonic stack',
      'Always check "if stack:" before access',
      'Empty when expecting pop = invalid input',
      'Guard with "while stack and condition"',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the common mistake of forgetting what you store in the stack. Why is this important?',
    sampleAnswer:
      'A common mistake is losing track of whether you stored values, indices, or pairs. For example, in next greater element, if you store indices but then try to compare stack top directly with current value, you will compare index with value - wrong. Or in min stack, forgetting you stored (value, min) pairs and trying to access as plain values. This breaks code silently or causes confusing errors. I prevent this by commenting what the stack contains - "stack stores indices" or "stack of (value, min) pairs". Also, use descriptive variable names like "index_stack" instead of just "stack". Clear documentation of stack contents prevents this entire class of bugs.',
    keyPoints: [
      'Mistake: forget if storing values, indices, or pairs',
      'Comparing wrong types silently breaks code',
      'Document: "stack stores indices"',
      'Use descriptive names: index_stack',
      'Clear stack contents prevents bugs',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the off-by-one error when calculating width or distance using stack indices. How do you get it right?',
    sampleAnswer:
      'Off-by-one errors happen when calculating width or distance from indices. In histogram, if you pop index i and stack top is j, the width is NOT i - j, it is i - j - 1, because bars between j and i are included. Wait, actually it depends on what you mean - if j is the left boundary where height starts to be valid, then width is current - j - 1. But if stack is empty after popping, width is current index itself. I get it right by carefully thinking: what indices are included in the range? Draw it out for a small example. The key is understanding whether boundaries are inclusive or exclusive, and handling empty stack case separately for width.',
    keyPoints: [
      'Width calculation: easy to be off by one',
      'Consider: are boundaries inclusive?',
      'Empty stack case: width = current index',
      'Draw small example to verify',
      'Test edge cases to catch errors',
    ],
  },
];
