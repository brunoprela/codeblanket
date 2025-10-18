/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain how stacks turn O(n²) problems into O(n). Give a concrete example.',
    sampleAnswer:
      'Stacks enable O(n) by remembering information so we do not need to repeatedly scan backwards. Take "next greater element": brute force would check every element to the right for each element, giving O(n²). With a monotonic stack, each element is pushed and popped exactly once as we scan through, giving O(n). The key is that the stack maintains useful information - elements waiting to find their next greater element. When we find it, we pop them immediately. No element is processed more than twice (one push, one pop). This amortized analysis shows that what looks like nested work is actually linear when using a stack to track state.',
    keyPoints: [
      'Stack remembers info, avoids repeated scans',
      'Example: next greater element O(n²) → O(n)',
      'Each element pushed/popped once',
      'Amortized: 2n operations total',
      'Maintains useful state to avoid nested loops',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare the space complexity of stack solutions. When is O(n) space worth it and when should you optimize?',
    sampleAnswer:
      'Most stack solutions use O(n) space worst case - if all elements are pushed without popping. This is worth it when we gain significant time improvement, like O(n²) to O(n). In interviews, O(n) space is usually acceptable since it matches input size. I would optimize space only if explicitly asked or if memory is severely constrained. Some tricks: in-place algorithms for problems like stock span can reuse input array, or streaming algorithms can use limited buffer. But generally, O(n) stack space for O(n) time is excellent trade-off. The question to ask: does the space usage scale with input, and is the time improvement worth it?',
    keyPoints: [
      'Most stack solutions: O(n) space worst case',
      'Worth it for O(n²) → O(n) time improvement',
      'O(n) space usually acceptable in interviews',
      'Optimize only if asked or memory constrained',
      'Trade-off: space for time',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is amortized analysis important for stack problems? Walk me through amortized O(1) for a stack operation.',
    sampleAnswer:
      'Amortized analysis is crucial for stack problems because individual operations might seem expensive but average out over many operations. For example, in monotonic stack, popping elements looks like it could be O(n) in one iteration. But amortized analysis shows each element is pushed exactly once and popped at most once across the entire algorithm, so total work is 2n, giving O(1) amortized per operation. Another example: dynamic array backing a stack might occasionally resize at O(n) cost, but this happens so rarely that amortized cost per push is still O(1). Amortized analysis lets us claim O(n) total time for stack algorithms even when individual steps vary.',
    keyPoints: [
      'Individual operations vary, but average out',
      'Monotonic stack: each element push/pop once total',
      '2n operations over n elements = O(1) amortized',
      'Rare expensive operations averaged over many cheap ones',
      'Enables O(n) total time claims',
    ],
  },
];
