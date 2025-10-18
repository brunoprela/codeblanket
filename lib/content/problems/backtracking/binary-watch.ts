/**
 * Binary Watch
 * Problem ID: binary-watch
 * Order: 5
 */

import { Problem } from '../../../types';

export const binary_watchProblem: Problem = {
  id: 'binary-watch',
  title: 'Binary Watch',
  difficulty: 'Easy',
  topic: 'Backtracking',
  description: `A binary watch has 4 LEDs on the top to represent the hours (0-11), and 6 LEDs on the bottom to represent the minutes (0-59). Each LED represents a zero or one, with the least significant bit on the right.

Given an integer \`turnedOn\` which represents the number of LEDs that are currently on (ignoring the PM), return all possible times the watch could represent. You may return the answer in **any order**.

The hour must not contain a leading zero. The minute must consist of two digits and may contain a leading zero.`,
  examples: [
    {
      input: 'turnedOn = 1',
      output:
        '["0:01","0:02","0:04","0:08","0:16","0:32","1:00","2:00","4:00","8:00"]',
    },
    {
      input: 'turnedOn = 9',
      output: '[]',
    },
  ],
  constraints: ['0 <= turnedOn <= 10'],
  hints: [
    'Generate all possible hour and minute combinations',
    'Count bits in each combination',
    'Filter by total bits equal to turnedOn',
  ],
  starterCode: `from typing import List

def read_binary_watch(turnedOn: int) -> List[str]:
    """
    Find all possible times with given LED count.
    
    Args:
        turnedOn: Number of LEDs on
        
    Returns:
        List of possible times
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [1],
      expected: [
        '0:01',
        '0:02',
        '0:04',
        '0:08',
        '0:16',
        '0:32',
        '1:00',
        '2:00',
        '4:00',
        '8:00',
      ],
    },
    {
      input: [9],
      expected: [],
    },
    {
      input: [0],
      expected: ['0:00'],
    },
  ],
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/binary-watch/',
  youtubeUrl: 'https://www.youtube.com/watch?v=CwDj8xGG2lU',
};
