/**
 * Meeting Rooms II
 * Problem ID: meeting-rooms-ii
 * Order: 3
 */

import { Problem } from '../../../types';

export const meeting_rooms_iiProblem: Problem = {
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
  constraints: ['1 <= intervals.length <= 10^4', '0 <= starti < endi <= 10^6'],
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
};
