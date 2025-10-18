/**
 * Interval List Intersections
 * Problem ID: interval-list-intersections
 * Order: 8
 */

import { Problem } from '../../../types';

export const interval_list_intersectionsProblem: Problem = {
  id: 'interval-list-intersections',
  title: 'Interval List Intersections',
  difficulty: 'Medium',
  topic: 'Intervals',
  description: `You are given two lists of closed intervals, \`firstList\` and \`secondList\`, where \`firstList[i] = [starti, endi]\` and \`secondList[j] = [startj, endj]\`. Each list of intervals is pairwise **disjoint** and in **sorted order**.

Return the **intersection** of these two interval lists.

A **closed interval** \`[a, b]\` (with \`a <= b\`) denotes the set of real numbers \`x\` with \`a <= x <= b\`.

The **intersection** of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of \`[1, 3]\` and \`[2, 4]\` is \`[2, 3]\`.`,
  examples: [
    {
      input:
        'firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]',
      output: '[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]',
    },
  ],
  constraints: [
    '0 <= firstList.length, secondList.length <= 1000',
    'firstList.length + secondList.length >= 1',
    '0 <= starti < endi <= 10^9',
    'endi < starti+1',
    '0 <= startj < endj <= 10^9',
    'endj < startj+1',
  ],
  hints: [
    'Use two pointers',
    'Find intersection of current intervals',
    'Move pointer of interval that ends first',
  ],
  starterCode: `from typing import List

def interval_intersection(firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    """
    Find intersections of two interval lists.
    
    Args:
        firstList: First list of intervals
        secondList: Second list of intervals
        
    Returns:
        List of intersections
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [0, 2],
          [5, 10],
          [13, 23],
          [24, 25],
        ],
        [
          [1, 5],
          [8, 12],
          [15, 24],
          [25, 26],
        ],
      ],
      expected: [
        [1, 2],
        [5, 5],
        [8, 10],
        [15, 23],
        [24, 24],
        [25, 25],
      ],
    },
  ],
  timeComplexity: 'O(m + n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/interval-list-intersections/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Qh8ZjL1RpLI',
};
