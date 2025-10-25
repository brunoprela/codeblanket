/**
 * Quiz questions for Interview Strategy section
 */

export const interviewstrategyQuiz = [
  {
    id: 'fenwick-interview-1',
    question:
      'What are the key phrases in a problem statement that should make you think of Fenwick Tree?',
    hint: 'Think about what operations Fenwick Tree excels at.',
    sampleAnswer:
      'Key phrases include "prefix sum with updates", "range sum queries", "cumulative frequency", "count inversions", or "dynamic ranking". Any time you see both queries and updates on cumulative/additive data, Fenwick Tree is a strong candidate. Also look for "count elements smaller than X" in a dynamic array.',
    keyPoints: [
      '"Prefix sum" + "updates"',
      '"Range sum queries"',
      '"Cumulative" or "frequency"',
      '"Count inversions" or "count smaller"',
    ],
  },
  {
    id: 'fenwick-interview-2',
    question:
      'How should you explain the 1-indexing requirement to an interviewer?',
    hint: 'Connect it to the bit manipulation.',
    sampleAnswer:
      'Explain that Fenwick Tree uses 1-indexing because the bit operation "i & -i" equals 0 when i is 0, which would cause infinite loops. Starting from index 1 makes the bit manipulation work correctly. Mention that tree[0] is unused and you convert 0-indexed input by adding 1 to all indices.',
    keyPoints: [
      '"i & -i" fails for i=0',
      'tree[0] unused, start at 1',
      'Convert: input_index + 1',
    ],
  },
  {
    id: 'fenwick-interview-3',
    question:
      'What common mistake do candidates make when implementing Fenwick Tree updates?',
    hint: 'Think about what update does versus what people expect it to do.',
    sampleAnswer:
      'A common mistake is treating update as "set value" when it actually "adds delta". To set arr[i] to new_val, you must calculate delta = new_val - old_val, then call update (i, delta). Candidates who forget this will get wrong answers. Always track the original array separately if you need to set values.',
    keyPoints: [
      'Update adds delta, does not set value',
      'To set: compute delta = new - old',
      'Keep original array if needed',
    ],
  },
];
