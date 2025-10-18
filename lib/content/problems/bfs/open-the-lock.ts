/**
 * Open the Lock
 * Problem ID: open-the-lock
 * Order: 7
 */

import { Problem } from '../../../types';

export const open_the_lockProblem: Problem = {
  id: 'open-the-lock',
  title: 'Open the Lock',
  difficulty: 'Medium',
  topic: 'Breadth-First Search (BFS)',
  description: `You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: \`'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'\`. The wheels can rotate freely and wrap around: for example we can turn \`'9'\` to be \`'0'\`, or \`'0'\` to be \`'9'\`. Each move consists of turning one wheel one slot.

The lock initially starts at \`'0000'\`, a string representing the state of the 4 wheels.

You are given a list of \`deadends\` dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a \`target\` representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.`,
  examples: [
    {
      input: 'deadends = ["0201","0101","0102","1212","2002"], target = "0202"',
      output: '6',
      explanation:
        'A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".',
    },
    {
      input: 'deadends = ["8888"], target = "0009"',
      output: '1',
    },
  ],
  constraints: [
    '1 <= deadends.length <= 500',
    'deadends[i].length == 4',
    'target.length == 4',
    'target will not be in the list deadends',
    'target and deadends[i] consist of digits only',
  ],
  hints: [
    'BFS from "0000"',
    'For each position, try +1 and -1',
    'Skip deadends',
  ],
  starterCode: `from typing import List
from collections import deque

def open_lock(deadends: List[str], target: str) -> int:
    """
    Find minimum turns to open lock.
    
    Args:
        deadends: Forbidden combinations
        target: Target combination
        
    Returns:
        Minimum turns or -1
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['0201', '0101', '0102', '1212', '2002'], '0202'],
      expected: 6,
    },
    {
      input: [['8888'], '0009'],
      expected: 1,
    },
  ],
  timeComplexity: 'O(10^4)',
  spaceComplexity: 'O(10^4)',
  leetcodeUrl: 'https://leetcode.com/problems/open-the-lock/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Pzg3bCDY87w',
};
