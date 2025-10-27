/**
 * Merge Intervals
 * Problem ID: merge-intervals
 * Order: 1
 */

import { Problem } from '../../../types';

export const merge_intervalsProblem: Problem = {
  id: 'merge-intervals',
  title: 'Merge Intervals',
  difficulty: 'Easy',
  description: `Given an array of \`intervals\` where \`intervals[i] = [starti, endi]\`, merge all overlapping intervals and return an array of the non-overlapping intervals.


**Approach:**1. Sort intervals by start time
2. Iterate through sorted intervals
3. If current overlaps with last merged, extend the last
4. Otherwise, add current as new interval

**Key Insight:**
Two intervals \`[a, b]\` and \`[c, d]\` overlap if \`c <= b\` (after sorting by start).`,
  examples: [
    {
      input: 'intervals = [[1,3],[2,6],[8,10],[15,18]]',
      output: '[[1,6],[8,10],[15,18]]',
      explanation:
        'Intervals [1,3] and [2,6] overlap, so merge them into [1,6].',
    },
    {
      input: 'intervals = [[1,4],[4,5]]',
      output: '[[1,5]]',
      explanation: 'Intervals [1,4] and [4,5] are touching, so merge them.',
    },
  ],
  constraints: [
    '1 <= intervals.length <= 10^4',
    'intervals[i].length == 2',
    '0 <= starti <= endi <= 10^4',
  ],
  hints: [
    'Sort the intervals by start time first',
    'Two intervals overlap if second.start <= first.end',
    'When merging, take min of starts and max of ends',
    'Keep track of the last merged interval',
    'If no overlap, add current interval to result',
  ],
  starterCode: `from typing import List

def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge all overlapping intervals.
    
    Args:
        intervals: List of [start, end] intervals
        
    Returns:
        List of merged intervals
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 3],
          [2, 6],
          [8, 10],
          [15, 18],
        ],
      ],
      expected: [
        [1, 6],
        [8, 10],
        [15, 18],
      ],
    },
    {
      input: [
        [
          [1, 4],
          [4, 5],
        ],
      ],
      expected: [[1, 5]],
    },
    {
      input: [[[1, 4]]],
      expected: [[1, 4]],
    },
  ],
  solution: `from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    Sort + Merge approach.
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # Check if current overlaps with last
        if current[0] <= last[1]:
            # Merge by extending end
            last[1] = max(last[1], current[1])
        else:
            # No overlap, add as new interval
            merged.append(current)
    
    return merged


# Alternative: In-place modification
def merge_inplace(intervals: List[List[int]]) -> List[List[int]]:
    """
    In-place merging to save space.
    Time: O(n log n), Space: O(1) excluding sort
    """
    if not intervals:
        return []
    
    intervals.sort()
    write = 0
    
    for read in range(1, len(intervals)):
        if intervals[read][0] <= intervals[write][1]:
            intervals[write][1] = max(intervals[write][1], intervals[read][1])
        else:
            write += 1
            intervals[write] = intervals[read]
    
    return intervals[:write + 1]`,
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/merge-intervals/',
  youtubeUrl: 'https://www.youtube.com/watch?v=44H3cEC2fFM',
  order: 1,
  topic: 'Intervals',
};
