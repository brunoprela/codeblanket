/**
 * Quiz questions for Advanced Techniques section
 */

export const advancedQuiz = [
  {
    id: 'q1',
    question:
      'Explain how two stacks are used to evaluate mathematical expressions. Why do we need both an operator stack and a value stack?',
    sampleAnswer:
      'Two-stack expression evaluation uses one stack for values and one for operators. As we scan left to right, numbers go on value stack. For operators, we check precedence: if current operator has lower or equal precedence to stack top operator, we pop the operator stack, pop two values, compute, and push result back to value stack. Then push current operator. The separation is crucial because operators have different precedences and we need to delay evaluation until we see what comes next. Parentheses force immediate evaluation of their contents. At the end, we process remaining operators. This handles infix expressions correctly by respecting operator precedence and parentheses.',
    keyPoints: [
      'Two stacks: values and operators',
      'Numbers → value stack',
      'Operators: check precedence before pushing',
      'Pop and compute when precedence says so',
      'Handles precedence and parentheses correctly',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through the largest rectangle in histogram problem. How does the monotonic stack solve it efficiently?',
    sampleAnswer:
      'For largest rectangle, I use a monotonic increasing stack storing indices. As I iterate through heights, if current height is smaller than stack top, I pop the stack. The popped index represents a bar that can extend to current position but no further. Its height is heights[popped], its width is current index minus the index below popped element in stack (or current index if stack empty). I compute area and track maximum. After popping, I push current index. This works because when a bar is popped, we know exactly how far left and right it can extend. Each bar pushed and popped once gives O(n) time instead of O(n²) brute force checking all rectangles.',
    keyPoints: [
      'Monotonic increasing stack of indices',
      'Pop when current smaller than stack top',
      'Popped bar: we know its full extension',
      'Width = current - remaining stack top',
      'O(n): each bar pushed/popped once',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the stock span problem and how stacks provide an elegant solution.',
    sampleAnswer:
      'Stock span asks: for each day, how many consecutive days before it had prices less than or equal to today. Brute force scans backwards each day: O(n²). With a monotonic decreasing stack of indices, I pop all days with prices less than or equal to current day - these are the days in the span. The span is current day minus the day remaining on stack (or current day if stack empty). Then push current day. The stack maintains potential span boundaries - days with prices higher than subsequent days. This is O(n) because each day is pushed and popped at most once. The key insight is that popped days cannot be span boundaries for future days.',
    keyPoints: [
      'Span: consecutive days before with price ≤ current',
      'Monotonic decreasing stack of day indices',
      'Pop days with price ≤ current',
      'Span = current day - remaining stack top',
      'O(n): each day pushed/popped once',
    ],
  },
];
