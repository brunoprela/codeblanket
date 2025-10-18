/**
 * Employee Free Time
 * Problem ID: employee-free-time
 * Order: 6
 */

import { Problem } from '../../../types';

export const employee_free_timeProblem: Problem = {
  id: 'employee-free-time',
  title: 'Employee Free Time',
  difficulty: 'Easy',
  topic: 'Intervals',
  description: `We are given a list \`schedule\` of employees, which represents the working time for each employee.

Each employee has a list of non-overlapping \`Intervals\`, and these intervals are in sorted order.

Return the list of finite intervals representing **common, positive-length free time** for all employees, also in sorted order.`,
  examples: [
    {
      input: 'schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]',
      output: '[[3,4]]',
    },
    {
      input: 'schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]',
      output: '[[5,6],[7,9]]',
    },
  ],
  constraints: [
    '1 <= schedule.length, schedule[i].length <= 50',
    '0 <= schedule[i][j].start < schedule[i][j].end <= 10^8',
  ],
  hints: [
    'Flatten all intervals into one list',
    'Sort by start time',
    'Find gaps between merged intervals',
  ],
  starterCode: `from typing import List

def employee_free_time(schedule: List[List[List[int]]]) -> List[List[int]]:
    """
    Find common free time for all employees.
    
    Args:
        schedule: List of employee schedules
        
    Returns:
        List of free time intervals
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [
            [1, 2],
            [5, 6],
          ],
          [[1, 3]],
          [[4, 10]],
        ],
      ],
      expected: [[3, 4]],
    },
    {
      input: [
        [
          [
            [1, 3],
            [6, 7],
          ],
          [[2, 4]],
          [
            [2, 5],
            [9, 12],
          ],
        ],
      ],
      expected: [
        [5, 6],
        [7, 9],
      ],
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/employee-free-time/',
  youtubeUrl: 'https://www.youtube.com/watch?v=qbJz_KPyu2w',
};
