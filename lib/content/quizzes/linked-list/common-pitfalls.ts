/**
 * Quiz questions for Common Pitfalls section
 */

export const commonpitfallsQuiz = [
  {
    id: 'q1',
    question:
      'What happens if you forget to check for null before accessing node.next? How do you systematically avoid this?',
    sampleAnswer:
      'Forgetting null checks causes null pointer dereference errors - accessing .next on null crashes. This commonly happens in while loops: if I write "while curr.next" but then access "curr.next.next" without checking if curr.next is null first, I crash when curr.next is the last node. I systematically avoid this by always checking: "if curr and curr.next" before accessing curr.next.next. Another pattern: use "while curr" and only access curr properties, never going ahead without explicit checks. In interviews, I state assumptions: "assuming input is not null" or explicitly handle null at the start. Test mental edge cases: empty list, single node, two nodes.',
    keyPoints: [
      'Null access causes crash',
      'Common in while loops accessing ahead',
      'Check: "if curr and curr.next" before curr.next.next',
      'Use "while curr" and only access curr properties',
      'Test edge cases: empty, single, two nodes',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the mistake of losing the head pointer. How does a dummy node help prevent this?',
    sampleAnswer:
      'Losing head happens when you move head pointer during traversal or modification, then cannot return the original head. For example, moving head forward in a loop: "head = head.next" loses the start. Without dummy, you need a separate variable to save original head. Dummy node prevents this by keeping a fixed reference before the real head. I work with dummy.next and never move dummy itself. At the end, dummy.next is always the current head, whether modified or not. This is foolproof - I cannot lose the head because dummy always points to it. The dummy acts as an anchor before the list.',
    keyPoints: [
      'Losing head by moving head pointer',
      'Need separate variable without dummy',
      'Dummy: fixed reference before head',
      'Work with dummy.next, never move dummy',
      'Return dummy.next: always current head',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the pitfall of modifying in-place without considering the implications. When is it problematic?',
    sampleAnswer:
      'Modifying in-place changes the original list, which is problematic when the caller needs the original data, when the list is shared by multiple references, or when the function should be pure. For example, reversing a list in-place destroys the original order. If another part of code holds a reference to a middle node, that reference becomes invalid after modification. In interviews, I clarify: "should I modify in-place or create a new list?" In-place is usually preferred for space efficiency (O(1) vs O(n)), but I mention the trade-off. For some problems like detecting cycles, in-place is necessary. For others like copying, creating new nodes is required.',
    keyPoints: [
      'In-place modifies original: may not be desired',
      'Problematic if: caller needs original, shared references',
      'Destroys original data',
      'In-place: O(1) space, but loses original',
      'Clarify requirements in interview',
    ],
  },
];
