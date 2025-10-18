/**
 * Maximum Frequency Stack
 * Problem ID: max-frequency-stack
 * Order: 22
 */

import { Problem } from '../../../types';

export const max_frequency_stackProblem: Problem = {
  id: 'max-frequency-stack',
  title: 'Maximum Frequency Stack',
  difficulty: 'Hard',
  topic: 'Arrays & Hashing',
  order: 22,
  description: `Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the \`FreqStack\` class:
- \`FreqStack()\` constructs an empty frequency stack.
- \`void push(int val)\` pushes an integer \`val\` onto the top of the stack.
- \`int pop()\` removes and returns the most frequent element in the stack. If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.`,
  examples: [
    {
      input:
        '["FreqStack","push","push","push","push","push","push","pop","pop","pop","pop"]\\n[[],[5],[7],[5],[7],[4],[5],[],[],[],[]]',
      output: '[null,null,null,null,null,null,null,5,7,5,4]',
      explanation:
        'FreqStack has [5,7,5,7,4,5]. Pop returns 5 (highest freq), then 7 (highest freq of remaining), then 5, then 4.',
    },
  ],
  constraints: [
    '0 <= val <= 10^9',
    'At most 2 * 10^4 calls will be made to push and pop',
  ],
  hints: [
    'Use a hash map to track frequency of each value',
    'Use a hash map of stacks grouped by frequency',
    'Track maximum frequency',
  ],
  starterCode: `class FreqStack:
    """
    Stack that always pops the most frequent element.
    """
    
    def __init__(self):
        """Initialize the frequency stack."""
        pass
    
    def push(self, val: int) -> None:
        """Push value to the stack."""
        pass
    
    def pop(self) -> int:
        """Pop the most frequent element."""
        pass
`,
  testCases: [
    {
      input: [
        [
          'push',
          'push',
          'push',
          'push',
          'push',
          'push',
          'pop',
          'pop',
          'pop',
          'pop',
        ],
        [[5], [7], [5], [7], [4], [5], [], [], [], []],
      ],
      expected: [null, null, null, null, null, null, 5, 7, 5, 4],
    },
  ],
  solution: `from collections import defaultdict

class FreqStack:
    """
    Frequency-based stack.
    Time: O(1) for push and pop
    Space: O(n)
    """
    
    def __init__(self):
        self.freq = {}  # value -> frequency
        self.group = defaultdict(list)  # frequency -> stack of values
        self.max_freq = 0
    
    def push(self, val: int) -> None:
        # Update frequency
        self.freq[val] = self.freq.get(val, 0) + 1
        f = self.freq[val]
        
        # Update max frequency
        self.max_freq = max(self.max_freq, f)
        
        # Add to frequency group
        self.group[f].append(val)
    
    def pop(self) -> int:
        # Pop from highest frequency group
        val = self.group[self.max_freq].pop()
        
        # Update frequency
        self.freq[val] -= 1
        
        # Update max frequency if needed
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        
        return val
`,
  timeComplexity: 'O(1) for push and pop',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/maximum-frequency-stack/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Z6idIicFDOE',
};
