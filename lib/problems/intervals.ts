import { Problem } from '../types';

export const intervalsProblems: Problem[] = [
  {
    id: 'merge-intervals',
    title: 'Merge Intervals',
    difficulty: 'Easy',
    description: `Given an array of \`intervals\` where \`intervals[i] = [starti, endi]\`, merge all overlapping intervals and return an array of the non-overlapping intervals.


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

    leetcodeUrl: 'https://leetcode.com/problems/merge-intervals/',
    youtubeUrl: 'https://www.youtube.com/watch?v=44H3cEC2fFM',
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
    leetcodeUrl: 'https://leetcode.com/problems/insert-interval/',
    youtubeUrl: 'https://www.youtube.com/watch?v=A8NUOmlwOlM',
  },
  {
    id: 'meeting-rooms-ii',
    title: 'Meeting Rooms II',
    difficulty: 'Hard',
    description: `Given an array of meeting time intervals \`intervals\` where \`intervals[i] = [starti, endi]\`, return **the minimum number of conference rooms** required.


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
          'Meeting 1 [0,30] overlaps with meeting 2 [5,10] and meeting 3 [15,20], but meetings 2 and 3 do not overlap. Need 2 rooms.',
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

    leetcodeUrl: 'https://leetcode.com/problems/meeting-rooms-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=FdzJmTCVyJU',
    order: 3,
    topic: 'Intervals',
    leetcodeUrl: 'https://leetcode.com/problems/meeting-rooms-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=FdzJmTCVyJU',
  },

  // EASY - Meeting Rooms
  {
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
  },

  // EASY - Summary Ranges
  {
    id: 'summary-ranges',
    title: 'Summary Ranges',
    difficulty: 'Easy',
    topic: 'Intervals',
    description: `You are given a **sorted unique** integer array \`nums\`.

A **range** \`[a,b]\` is the set of all integers from \`a\` to \`b\` (inclusive).

Return the **smallest sorted** list of ranges that **cover all the numbers in the array exactly**. That is, each element of \`nums\` is covered by exactly one of the ranges, and there is no integer \`x\` such that \`x\` is in one of the ranges but not in \`nums\`.

Each range \`[a,b]\` in the list should be output as:
- \`"a->b"\` if \`a != b\`
- \`"a"\` if \`a == b\``,
    examples: [
      {
        input: 'nums = [0,1,2,4,5,7]',
        output: '["0->2","4->5","7"]',
      },
      {
        input: 'nums = [0,2,3,4,6,8,9]',
        output: '["0","2->4","6","8->9"]',
      },
    ],
    constraints: [
      '0 <= nums.length <= 20',
      '-2^31 <= nums[i] <= 2^31 - 1',
      'All the values of nums are unique',
      'nums is sorted in ascending order',
    ],
    hints: [
      'Track start of current range',
      'When gap found, close current range',
    ],
    starterCode: `from typing import List

def summary_ranges(nums: List[int]) -> List[str]:
    """
    Convert array to summary ranges.
    
    Args:
        nums: Sorted unique integer array
        
    Returns:
        List of range strings
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[0, 1, 2, 4, 5, 7]],
        expected: ['0->2', '4->5', '7'],
      },
      {
        input: [[0, 2, 3, 4, 6, 8, 9]],
        expected: ['0', '2->4', '6', '8->9'],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/summary-ranges/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Cx8LbsHWxY0',
  },

  // EASY - Employee Free Time
  {
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
  },

  // MEDIUM - Non-overlapping Intervals
  {
    id: 'non-overlapping-intervals',
    title: 'Non-overlapping Intervals',
    difficulty: 'Medium',
    topic: 'Intervals',
    description: `Given an array of intervals \`intervals\` where \`intervals[i] = [starti, endi]\`, return the **minimum number of intervals you need to remove** to make the rest of the intervals non-overlapping.`,
    examples: [
      {
        input: 'intervals = [[1,2],[2,3],[3,4],[1,3]]',
        output: '1',
        explanation:
          'Remove [1,3] and the rest of the intervals are non-overlapping.',
      },
      {
        input: 'intervals = [[1,2],[1,2],[1,2]]',
        output: '2',
      },
    ],
    constraints: [
      '1 <= intervals.length <= 10^5',
      'intervals[i].length == 2',
      '-5 * 10^4 <= starti < endi <= 5 * 10^4',
    ],
    hints: ['Sort by end time', 'Greedy: keep interval that ends earliest'],
    starterCode: `from typing import List

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    """
    Find minimum intervals to remove.
    
    Args:
        intervals: List of intervals
        
    Returns:
        Minimum number to remove
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 3],
          ],
        ],
        expected: 1,
      },
      {
        input: [
          [
            [1, 2],
            [1, 2],
            [1, 2],
          ],
        ],
        expected: 2,
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/non-overlapping-intervals/',
    youtubeUrl: 'https://www.youtube.com/watch?v=nONCGxWoUfM',
  },

  // MEDIUM - Interval List Intersections
  {
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
  },

  // MEDIUM - Minimum Number of Arrows to Burst Balloons
  {
    id: 'minimum-arrows-burst-balloons',
    title: 'Minimum Number of Arrows to Burst Balloons',
    difficulty: 'Medium',
    topic: 'Intervals',
    description: `There are some spherical balloons taped onto a flat wall that represents the XY-plane. The balloons are represented as a 2D integer array \`points\` where \`points[i] = [xstart, xend]\` denotes a balloon whose **horizontal diameter** stretches between \`xstart\` and \`xend\`. You do not know the exact y-coordinates of the balloons.

Arrows can be shot up **directly vertically** (in the positive y-direction) from different points along the x-axis. A balloon with \`xstart\` and \`xend\` is **burst** by an arrow shot at \`x\` if \`xstart <= x <= xend\`. There is **no limit** to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

Given the array \`points\`, return the **minimum number** of arrows that must be shot to burst all balloons.`,
    examples: [
      {
        input: 'points = [[10,16],[2,8],[1,6],[7,12]]',
        output: '2',
      },
      {
        input: 'points = [[1,2],[3,4],[5,6],[7,8]]',
        output: '4',
      },
    ],
    constraints: [
      '1 <= points.length <= 10^5',
      'points[i].length == 2',
      '-2^31 <= xstart < xend <= 2^31 - 1',
    ],
    hints: [
      'Sort by end position',
      'Greedy: shoot arrow at end of first balloon',
      'Count how many times we need new arrow',
    ],
    starterCode: `from typing import List

def find_min_arrow_shots(points: List[List[int]]) -> int:
    """
    Find minimum arrows needed.
    
    Args:
        points: Balloon positions
        
    Returns:
        Minimum number of arrows
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [10, 16],
            [2, 8],
            [1, 6],
            [7, 12],
          ],
        ],
        expected: 2,
      },
      {
        input: [
          [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
          ],
        ],
        expected: 4,
      },
    ],
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl:
      'https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/',
    youtubeUrl: 'https://www.youtube.com/watch?v=lPm gVkqBGA',
  },
];
