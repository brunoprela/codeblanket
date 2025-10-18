/**
 * Daily Temperatures
 * Problem ID: daily-temperatures
 * Order: 11
 */

import { Problem } from '../../../types';

export const daily_temperaturesProblem: Problem = {
  id: 'daily-temperatures',
  title: 'Daily Temperatures',
  difficulty: 'Medium',
  topic: 'Stack',
  description: `Given an array of integers \`temperatures\` represents the daily temperatures, return an array \`answer\` such that \`answer[i]\` is the number of days you have to wait after the \`i-th\` day to get a warmer temperature. If there is no future day for which this is possible, keep \`answer[i] == 0\` instead.`,
  examples: [
    {
      input: 'temperatures = [73,74,75,71,69,72,76,73]',
      output: '[1,1,4,2,1,1,0,0]',
    },
    {
      input: 'temperatures = [30,40,50,60]',
      output: '[1,1,1,0]',
    },
  ],
  constraints: [
    '1 <= temperatures.length <= 10^5',
    '30 <= temperatures[i] <= 100',
  ],
  hints: [
    'Use a monotonic decreasing stack',
    'Store indices in the stack',
    'Pop when you find a warmer temperature',
  ],
  starterCode: `from typing import List

def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Find days until warmer temperature.
    
    Args:
        temperatures: Daily temperatures
        
    Returns:
        Days to wait for warmer temperature
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[73, 74, 75, 71, 69, 72, 76, 73]],
      expected: [1, 1, 4, 2, 1, 1, 0, 0],
    },
    {
      input: [[30, 40, 50, 60]],
      expected: [1, 1, 1, 0],
    },
    {
      input: [[30, 60, 90]],
      expected: [1, 1, 0],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/daily-temperatures/',
  youtubeUrl: 'https://www.youtube.com/watch?v=cTBiBSnjO3c',
};
