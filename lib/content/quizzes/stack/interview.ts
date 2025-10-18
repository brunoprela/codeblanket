/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize that a problem might need a stack? What are the key signals in the problem description?',
    sampleAnswer:
      'Several signals tell me to consider a stack. First, keywords like "valid", "matching", "balanced" for parentheses or bracket problems - immediate stack signal. Second, "next greater", "next smaller", or "nearest" something - think monotonic stack. Third, any mention of "most recent", "last", "undo", or "backtrack" - LIFO nature of stack. Fourth, if I am thinking about scanning backwards repeatedly - that is O(n²), probably can optimize with stack. Fifth, evaluation or parsing of expressions. The fundamental question: do I need to process things in reverse order or match pairs? If yes, stack is likely the answer.',
    keyPoints: [
      'Keywords: valid, matching, balanced → parentheses problems',
      'Next greater/smaller/nearest → monotonic stack',
      'Most recent, last, undo, backtrack → LIFO',
      'Repeated backward scanning → stack optimization',
      'Expression evaluation or parsing',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through how you would explain a stack solution in an interview, from identifying the pattern to explaining complexity.',
    sampleAnswer:
      'First, I identify the pattern: "I notice this is a matching problem, so I am thinking stack to track unmatched opening brackets". Then I explain the approach: "I will iterate through the string, push opening brackets onto the stack, and when I see a closing bracket, check if it matches the stack top". I mention the key insight: "The stack naturally gives me the most recent unmatched bracket, which is exactly what I need". Then I code carefully, explaining as I go. After coding, I trace through an example: "For input ([)], when we hit ), stack top is (, they match, we pop...". Finally, I state complexity: "O(n) time for one pass, O(n) space worst case if all opening brackets". Clear communication throughout is crucial.',
    keyPoints: [
      'Identify pattern and explain why stack',
      'Explain approach clearly',
      'State key insight',
      'Code with explanations',
      'Trace through example',
      'State time and space complexity',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes people make with stack problems in interviews? How do you avoid them?',
    sampleAnswer:
      'First mistake: forgetting to check empty stack before pop or peek - leads to runtime errors. I always use "if stack:" guards. Second: not being clear about what is stored in the stack - values, indices, or pairs. I comment or use descriptive names. Third: off-by-one errors in width calculations for monotonic stack problems. I draw examples to verify. Fourth: forgetting to process remaining elements in stack after the main loop. I either use sentinels or explicit final processing. Fifth: not handling edge cases like empty input or all same values. I mention these upfront. The key is defensive coding - check empty, document stack contents, test edge cases, and communicate clearly.',
    keyPoints: [
      'Check empty before pop/peek',
      'Document what stack stores',
      'Verify width calculations with examples',
      'Process remaining stack after loop',
      'Handle edge cases: empty, all same',
      'Defensive coding and clear communication',
    ],
  },
];
