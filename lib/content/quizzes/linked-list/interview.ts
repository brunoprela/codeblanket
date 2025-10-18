/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize that a problem requires linked list techniques? What signals in the problem description tell you?',
    sampleAnswer:
      'Several signals indicate linked list problems. First, obviously "linked list" in the problem statement. Second, sequential access patterns without random access needs - traversing one by one. Third, frequent insertions or deletions, especially at beginning or middle. Fourth, problems involving cycles, finding middle, or nth from end - these are classic linked list patterns. Fifth, memory-constrained problems where you cannot use arrays. The key question: does the problem require pointer manipulation or sequential traversal with O(1) space? If accessing by index is not needed and modifications are frequent, linked list techniques apply. Even if input is array, thinking in linked list terms can help.',
    keyPoints: [
      'Explicit: "linked list" mentioned',
      'Sequential access without random access',
      'Frequent insertions/deletions',
      'Classic patterns: cycles, middle, nth from end',
      'O(1) space pointer manipulation',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your approach to a linked list problem in an interview, from problem statement to explaining your solution.',
    sampleAnswer:
      'First, I clarify: is it singly or doubly linked? Can I modify in-place? Are there cycles? Then I identify the pattern: is it reversal, cycle detection, two pointers, or merging? I explain my approach: "I will use fast and slow pointers to find the middle in one pass". I discuss complexity: "O(n) time for one traversal, O(1) space with just two pointers". Then I draw a small example on paper or whiteboard, showing pointer movements step by step. While coding, I explain: "I initialize dummy node to handle edge cases, then..." After coding, I trace through edge cases: empty list, single node, two nodes. I mention optimizations or alternative approaches. Clear visualization and edge case handling are crucial.',
    keyPoints: [
      'Clarify: singly/doubly, in-place, cycles?',
      'Identify pattern',
      'Explain approach and complexity',
      'Draw example, show pointer movements',
      'Code with explanations',
      'Trace edge cases',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes in linked list problems and how do you avoid them?',
    sampleAnswer:
      'First: null pointer errors from not checking before access. I always use "if curr and curr.next" before curr.next.next. Second: losing head pointer. I use dummy node or save original head. Third: off-by-one in pointer movement - moving too far or not far enough. I trace small examples to verify. Fourth: forgetting edge cases like empty list or single node. I test these mentally before finishing. Fifth: creating cycles accidentally by incorrect pointer updates. I carefully track which pointers I am changing. Sixth: using wrong loop condition, causing infinite loops. I ensure progress in every iteration. My strategy: use dummy nodes, test edge cases, draw examples, and double-check pointer updates.',
    keyPoints: [
      'Null checks: "if curr and curr.next"',
      'Use dummy node to avoid losing head',
      'Trace examples for off-by-one',
      'Test edge cases: empty, single node',
      'Careful pointer updates to avoid cycles',
      'Ensure loop progress to avoid infinite loops',
    ],
  },
];
