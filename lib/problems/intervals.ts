import { Problem } from '../types';

export const intervalsProblems: Problem[] = [
  {
    id: 'merge-intervals',
    title: 'Merge Intervals',
    difficulty: 'Easy',
    description: `Given an array of \`intervals\` where \`intervals[i] = [starti, endi]\`, merge all overlapping intervals and return an array of the non-overlapping intervals.

**LeetCode:** [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)
**YouTube:** [NeetCode - Merge Intervals](https://www.youtube.com/watch?v=44H3cEC2fFM)

**Approach:**
1. Sort intervals by start time
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
    order: 1,
    topic: 'Intervals',
    leetcodeUrl: 'https://leetcode.com/problems/merge-intervals/',
    youtubeUrl: 'https://www.youtube.com/watch?v=44H3cEC2fFM',
  },
  {
    id: 'insert-interval',
    title: 'Insert Interval',
    difficulty: 'Medium',
    description: `You are given an array of non-overlapping intervals \`intervals\` sorted by their start time, and a new interval \`newInterval = [start, end]\`.

Insert \`newInterval\` into \`intervals\` such that \`intervals\` is still sorted and has no overlapping intervals. Merge if necessary.

**LeetCode:** [57. Insert Interval](https://leetcode.com/problems/insert-interval/)
**YouTube:** [NeetCode - Insert Interval](https://www.youtube.com/watch?v=A8NUOmlwOlM)

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
    order: 2,
    topic: 'Intervals',
    leetcodeUrl: 'https://leetcode.com/problems/insert-interval/',
    youtubeUrl: 'https://www.youtube.com/watch?v=A8NUOmlwOlM',
  },
  {
    id: 'meeting-rooms-ii',
    title: 'Meeting Rooms II',
    difficulty: 'Hard',
    description: `Given an array of meeting time intervals \`intervals\` where \`intervals[i] = [starti, endi]\`, return **the minimum number of conference rooms** required.

**LeetCode:** [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) (Premium)
**YouTube:** [NeetCode - Meeting Rooms II](https://www.youtube.com/watch?v=FdzJmTCVyJU)

**Approach:**
Use the **Sweep Line** algorithm:
1. Separate start and end times
2. Sort both arrays
3. Use two pointers to track overlapping meetings
4. When a meeting starts, increment room count
5. When a meeting ends, decrement room count
6. Track maximum rooms needed

**Key Insight:**
At any time, the number of rooms needed equals the number of meetings happening simultaneously. Sweep line counts this efficiently.`,
    examples: [
      {
        input: 'intervals = [[0,30],[5,10],[15,20]]',
        output: '2',
        explanation:
          "Meeting 1 [0,30] overlaps with meeting 2 [5,10] and meeting 3 [15,20], but meetings 2 and 3 don't overlap. Need 2 rooms.",
      },
      {
        input: 'intervals = [[7,10],[2,4]]',
        output: '1',
        explanation: 'No overlap, only 1 room needed.',
      },
    ],
    constraints: [
      '1 <= intervals.length <= 10^4',
      '0 <= starti < endi <= 10^6',
    ],
    hints: [
      'Think about what happens at each start and end time',
      'Separate starts and ends into two arrays',
      'Sort both arrays independently',
      'Use two pointers to process events in chronological order',
      'If start < end, a meeting is starting (need room)',
      'If start >= end, a meeting is ending (free room)',
      'Track current rooms and maximum rooms ever needed',
    ],
    starterCode: `from typing import List

def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """
    Find minimum number of meeting rooms required.
    
    Args:
        intervals: List of meeting time intervals
        
    Returns:
        Minimum number of rooms needed
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
        expected: 2,
      },
      {
        input: [
          [
            [7, 10],
            [2, 4],
          ],
        ],
        expected: 1,
      },
      {
        input: [
          [
            [1, 5],
            [2, 6],
            [3, 7],
          ],
        ],
        expected: 3,
      },
    ],
    solution: `from typing import List


def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """
    Sweep line algorithm.
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return 0
    
    # Separate and sort starts and ends
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])
    
    rooms = 0
    max_rooms = 0
    s = e = 0
    
    while s < len(starts):
        if starts[s] < ends[e]:
            # Meeting starts, need a room
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            # Meeting ends, free a room
            rooms -= 1
            e += 1
    
    return max_rooms


# Alternative: Using heap (min-heap of end times)
def min_meeting_rooms_heap(intervals: List[List[int]]) -> int:
    """
    Using heap to track earliest ending meeting.
    Time: O(n log n), Space: O(n)
    """
    import heapq
    
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[0])
    heap = []
    
    for start, end in intervals:
        # If earliest meeting has ended, reuse room
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        
        # Add current meeting's end time
        heapq.heappush(heap, end)
    
    return len(heap)


# Alternative: Using events with delta
def min_meeting_rooms_events(intervals: List[List[int]]) -> int:
    """
    Event-based sweep line.
    Time: O(n log n), Space: O(n)
    """
    events = []
    
    for start, end in intervals:
        events.append((start, 1))   # Meeting start: +1 room
        events.append((end, -1))    # Meeting end: -1 room
    
    # Sort by time, ties: ends before starts
    events.sort(key=lambda x: (x[0], x[1]))
    
    rooms = 0
    max_rooms = 0
    
    for time, delta in events:
        rooms += delta
        max_rooms = max(max_rooms, rooms)
    
    return max_rooms`,
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    order: 3,
    topic: 'Intervals',
    leetcodeUrl: 'https://leetcode.com/problems/meeting-rooms-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=FdzJmTCVyJU',
  },
];
