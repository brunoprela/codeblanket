/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the iterative reversal template. Why do we need three pointers (prev, curr, next)?',
    sampleAnswer:
      'For iterative reversal, prev tracks the reversed portion, curr is the node we are currently reversing, and next temporarily stores the rest of the list. The pattern: save curr.next in next (so we do not lose the list), reverse curr by setting curr.next = prev, move prev to curr, move curr to next. We need three because we must simultaneously reverse the current link and advance forward. If we only had two pointers and did curr.next = prev, we would lose access to the rest of the list. The next pointer saves that access. After the loop, prev points to the new head. This is O(n) time and O(1) space, elegant and efficient.',
    keyPoints: [
      'prev: reversed portion so far',
      'curr: node currently reversing',
      'next: temporarily saves rest of list',
      'Pattern: save next, reverse curr, advance both',
      'O(n) time, O(1) space',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the dummy node technique for merging sorted lists. Why does it make the code cleaner?',
    sampleAnswer:
      'Dummy node eliminates special case handling for the first node. Without it, I need to check if result list is empty and set head specially for the first node. With dummy, I start with dummy.next = None and use curr = dummy to track the tail of merged list. I just do curr.next = smaller node and advance curr. At the end, return dummy.next as the new head. This works because dummy acts as a placeholder before the real list, so even the first node is handled uniformly - no if statements for "is this the first node". The code becomes a simple loop without conditionals for initialization. This pattern appears in many linked list problems.',
    keyPoints: [
      'Placeholder before real head',
      'Eliminates first node special case',
      'curr tracks tail of merged list',
      'Uniform logic for all nodes',
      'Return dummy.next at end',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the iterative vs recursive approach to linked list problems. When would you choose each?',
    sampleAnswer:
      'Iterative uses explicit pointers and loops, giving O(1) space. Recursive uses call stack for implicit tracking, using O(n) space but often cleaner code. I choose iterative when space is critical or when the iterative logic is straightforward, like reversal or traversal. I choose recursive when the problem has natural recursive structure, like tree-like operations or when backtracking is needed. Recursive can be more elegant for complex pointer manipulation. In interviews, I often code iterative first as it shows space efficiency, then mention recursive as an alternative if asked. Some problems like reversing in groups feel more natural recursively.',
    keyPoints: [
      'Iterative: O(1) space, explicit pointers',
      'Recursive: O(n) space, cleaner for some problems',
      'Iterative: when space critical',
      'Recursive: natural structure, backtracking',
      'Interview strategy: start iterative, mention recursive',
    ],
  },
];
