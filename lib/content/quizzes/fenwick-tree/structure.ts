/**
 * Quiz questions for Fenwick Tree Structure section
 */

export const structureQuiz = [
  {
    id: 'fenwick-structure-1',
    question:
      'Explain what the operation "i & -i" does and why it is critical to Fenwick Tree.',
    hint: "Think about what -i looks like in binary (two's complement).",
    sampleAnswer:
      'The operation "i & -i" extracts the least significant bit (LSB) of i. In two\'s complement, -i flips all bits and adds 1, so when you AND them together, only the rightmost 1-bit survives. This tells us how many elements the current index is responsible for, and where to jump next during update or query.',
    keyPoints: [
      'Extracts the least significant set bit',
      'Determines range size at each index',
      'Used to navigate parent/child relationships',
    ],
  },
  {
    id: 'fenwick-structure-2',
    question: 'Why does Fenwick Tree use 1-indexing instead of 0-indexing?',
    hint: 'Consider what happens when you try "0 & -0".',
    sampleAnswer:
      'Fenwick Tree uses 1-indexing because the bit manipulation "i & -i" does not work correctly for i=0. When i=0, "i & -i" equals 0, creating an infinite loop since adding or subtracting 0 does not change the index. Starting from index 1 avoids this issue.',
    keyPoints: [
      '"0 & -0" = 0, causes infinite loops',
      'tree[0] is unused, indices start at 1',
      'Need to convert: arr[i] → tree[i+1]',
    ],
  },
  {
    id: 'fenwick-structure-3',
    question: 'How do you traverse up the tree during an update operation?',
    hint: 'Think about which indices depend on the current index.',
    sampleAnswer:
      'During update, you move up by adding the LSB: "i += i & -i". This takes you to the parent node that needs updating. You keep going until i exceeds the tree size. For example, updating index 5 (binary 101) affects indices 5→6→8→16...',
    keyPoints: [
      'Move up: i += i & -i',
      'Each parent covers a larger range',
      'Continue until i > n',
    ],
  },
];
