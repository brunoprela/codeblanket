/**
 * Meeting Rooms
 * Problem ID: meeting-rooms
 * Order: 4
 */

import { Problem } from '../../../types';

export const meeting_roomsProblem: Problem = {
  id: 'meeting-rooms',
  title: 'Meeting Rooms',
  difficulty: 'Easy',
  topic: 'Intervals',
  description: `Given an array of meeting time intervals where \`intervals[i] = [starti, endi]\`, determine if a person could attend all meetings.`,
  examples: [
    {
      input: 'intervals = [[0,30],[5,10],[15,20]]',
      output: 'false',
    },
    {
      input: 'intervals = [[7,10],[2,4]]',
      output: 'true',
    },
  ],
  constraints: [
    '0 <= intervals.length <= 10^4',
    'intervals[i].length == 2',
    '0 <= starti < endi <= 10^6',
  ],
  hints: [
    'Sort intervals by start time',
    'Check if any two consecutive intervals overlap',
  ],
  starterCode: `from typing import List

def can_attend_meetings(intervals: List[List[int]]) -> bool:
    """
    Check if person can attend all meetings.
    
    Args:
        intervals: List of [start, end] intervals
        
    Returns:
        True if can attend all meetings
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [0, 30],
          [5, 10],
          [15, 20],
        ],
      ],
      expected: false,
    },
    {
      input: [
        [
          [7, 10],
          [2, 4],
        ],
      ],
      expected: true,
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/meeting-rooms/',
  youtubeUrl: 'https://www.youtube.com/watch?v=PaJxqZVPhbg',
};
