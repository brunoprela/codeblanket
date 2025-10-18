/**
 * Quiz questions for Common Stack Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the matching pairs pattern for validating parentheses. How does the stack help solve this?',
    sampleAnswer:
      'For matching parentheses, I use a stack to track opening brackets. As I scan the string, when I see an opening bracket like (, [, or {, I push it onto the stack. When I see a closing bracket like ), ], or }, I check if the stack top has the matching opening bracket. If it matches, I pop it off. If it does not match or the stack is empty, the string is invalid. At the end, the stack must be empty for all brackets to be matched. The stack naturally gives me the most recent unmatched opening bracket, which is exactly what I need to match with the next closing bracket. This is O(n) time and O(n) space worst case.',
    keyPoints: [
      'Push opening brackets onto stack',
      'Pop and match when see closing bracket',
      'Stack top = most recent unmatched opening',
      'Must be empty at end',
      'O(n) time, O(n) space',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the monotonic stack pattern. What makes it "monotonic" and when would you use it?',
    sampleAnswer:
      'A monotonic stack maintains elements in increasing or decreasing order. For "next greater element" problems, I use a decreasing stack - as I iterate, if the current element is larger than stack top, I pop elements until I find one larger or the stack is empty. The popped elements have found their next greater element (current element). Then I push current onto the stack. This is monotonic because the stack maintains a decreasing sequence. It is powerful because it solves next greater element in O(n) time - each element is pushed and popped at most once. Use it when finding next/previous greater/smaller elements, or in problems involving views, temperatures, or histogram areas.',
    keyPoints: [
      'Maintains increasing or decreasing order',
      'Pop elements that violate monotonic property',
      'Next greater: use decreasing stack',
      'Each element pushed/popped once: O(n)',
      'Use for: next greater/smaller elements',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the min/max stack pattern. How do you maintain O(1) access to the minimum while still supporting all stack operations?',
    sampleAnswer:
      'To maintain O(1) min access, I use two stacks: the main stack for regular operations and a min stack that tracks minimums. When I push a value, I push it to main stack, and push the minimum of (current value, current min) to the min stack. When I pop from main, I also pop from min stack. The top of min stack always gives the current minimum in O(1). This works because the min stack maintains what the minimum would be at each stack state. Space is O(n) for both stacks but we gain O(1) min access. Alternative: store pairs (value, min-so-far) in one stack. This is classic for problems requiring min/max queries on dynamic data.',
    keyPoints: [
      'Two stacks: main and min/max stack',
      'Push to both: value to main, min-so-far to min stack',
      'Pop from both together',
      'Min stack top = current minimum',
      'O(1) min/max access, O(n) space',
    ],
  },
];
