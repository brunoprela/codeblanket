/**
 * Quiz questions for Introduction to Stacks section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain the LIFO principle and why it makes stacks useful. Give me a real-world example.',
    sampleAnswer:
      'LIFO means Last-In-First-Out - the most recently added element is the first one removed. Think of a stack of plates: you add new plates on top and take plates from the top. You cannot grab a plate from the middle or bottom without removing everything above it first. This makes stacks perfect for tracking state history like browser back button - your most recent page is the first one you go back to. Or function calls in programming - when a function calls another function, the most recent call needs to finish first before returning to the previous function. The LIFO property naturally matches problems where you need to reverse order or process the most recent thing first.',
    keyPoints: [
      'Last-In-First-Out: most recent element removed first',
      'Real example: stack of plates, only access top',
      'Perfect for: browser history, undo, function calls',
      'Naturally reverses order',
      'Process most recent thing first',
    ],
  },
  {
    id: 'q2',
    question: 'Why are all stack operations O(1)? What makes this possible?',
    sampleAnswer:
      'All stack operations are O(1) because we only ever interact with one end - the top of the stack. When we push, we add to the top in constant time. When we pop, we remove from the top in constant time. We never need to search through the stack or access elements in the middle. This is possible because stacks are typically implemented with arrays where we track the top index, or with linked lists where we track the head pointer. Either way, adding or removing at one end is a simple pointer update or array index change, not dependent on how many elements are in the stack. This is why stacks are so efficient.',
    keyPoints: [
      'Only interact with one end (top)',
      'No searching or middle access needed',
      'Array: track top index, update in O(1)',
      'Linked list: track head, update in O(1)',
      'Independent of stack size',
    ],
  },
  {
    id: 'q3',
    question:
      'Talk about when you would choose a stack over other data structures. What problems are stacks uniquely good at?',
    sampleAnswer:
      'I choose stacks when the problem has a clear "most recent" or "reverse order" aspect. Parsing problems like matching parentheses are perfect - when I see a closing bracket, I need to check if it matches the most recent opening bracket, which is exactly what stack top gives me. Backtracking problems like DFS use stacks because I need to explore the most recent branch and backtrack when I hit a dead end. Monotonic stack problems like "next greater element" use stacks to track decreasing sequences efficiently. Stacks are also great when I need to reverse something or track history. If the problem involves processing things in reverse order or matching pairs, stack is usually the answer.',
    keyPoints: [
      'Most recent / reverse order problems',
      'Parsing: match parentheses, evaluate expressions',
      'Backtracking: DFS, explore recent branch',
      'Monotonic patterns: next greater element',
      'Reversing or tracking history',
    ],
  },
];
