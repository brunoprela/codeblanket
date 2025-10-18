/**
 * Insert Interval
 * Problem ID: insert-interval
 * Order: 2
 */

import { Problem } from '../../../types';

export const insert_intervalProblem: Problem = {
  id: 'insert-interval',
  title: 'Insert Interval',
  difficulty: 'Medium',
  description: `You are given an array of non-overlapping intervals \`intervals\` sorted by their start time, and a new interval \`newInterval = [start, end]\`.

Insert \`newInterval\` into \`intervals\` such that \`intervals\` is still sorted and has no overlapping intervals. Merge if necessary.


**Approach:**
Three phases:
1. Add all intervals that end before new interval starts
2. Merge all overlapping intervals with new interval
3. Add remaining intervals

**Key Insight:**
Since input is already sorted, we can solve in O(n) time without sorting.`,
  examples: [
    {
      input: 'intervals = [[1,3],[6,9]], newInterval = [2,5]',
      output: '[[1,5],[6,9]]',
      explanation: 'New interval [2,5] overlaps with [1,3], merge to [1,5].',
    },
    {
      input:
        'intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]',
      output: '[[1,2],[3,10],[12,16]]',
      explanation: 'New interval [4,8] merges [3,5], [6,7], and [8,10].',
    },
  ],
  constraints: [
    '0 <= intervals.length <= 10^4',
    'intervals[i].length == 2',
    '0 <= starti <= endi <= 10^5',
    'intervals is sorted by starti',
    'newInterval.length == 2',
    '0 <= start <= end <= 10^5',
  ],
  hints: [
    'Input is already sorted - no need to sort!',
    'Three phases: before, merge, after',
    'Before: intervals that end before new starts (end < new.start)',
    'Merge: intervals that overlap with new (start <= new.end)',
    'After: remaining intervals',
    'During merge, expand new interval to cover all overlapping',
  ],
  starterCode: `from typing import List

def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    """
    Insert new interval and merge overlaps.
    
    Args:
        intervals: Sorted non-overlapping intervals
        newInterval: New interval to insert
        
    Returns:
        Merged intervals including newInterval
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 3],
          [6, 9],
        ],
        [2, 5],
      ],
      expected: [
        [1, 5],
        [6, 9],
      ],
    },
    {
      input: [
        [
          [1, 2],
          [3, 5],
          [6, 7],
          [8, 10],
          [12, 16],
        ],
        [4, 8],
      ],
      expected: [
        [1, 2],
        [3, 10],
        [12, 16],
      ],
    },
    {
      input: [[], [5, 7]],
      expected: [[5, 7]],
    },
  ],
  solution: `from typing import List


def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    """
    Three-phase insertion without sorting.
    Time: O(n), Space: O(n)
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Phase 1: Add all intervals before new interval
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    # Phase 2: Merge all overlapping intervals
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)
    
    # Phase 3: Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result


# Alternative: More concise
def insert_concise(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    """
    More concise version.
    """
    result = []
    
    for interval in intervals:
        # New interval comes after current
        if interval[1] < newInterval[0]:
            result.append(interval)
        # New interval comes before current
        elif interval[0] > newInterval[1]:
            result.append(newInterval)
            newInterval = interval
        # Overlapping, merge
        else:
            newInterval = [
                min(newInterval[0], interval[0]),
                max(newInterval[1], interval[1])
            ]
    
    result.append(newInterval)
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/insert-interval/',
  youtubeUrl: 'https://www.youtube.com/watch?v=A8NUOmlwOlM',
  order: 2,
  topic: 'Intervals',
};
