import { Module } from '@/lib/types';

export const intervalsModule: Module = {
    id: 'intervals',
    title: 'Intervals',
    description:
        'Master interval manipulation including merging, overlapping, and intersection problems.',
    icon: '↔️',
    timeComplexity: 'O(n log n) for sorting-based approaches',
    spaceComplexity: 'O(n) for result storage',
    sections: [
        {
            id: 'introduction',
            title: 'Introduction to Intervals',
            content: `An **interval** represents a range with a start and end point: \`[start, end]\`. Interval problems involve operations like merging, detecting overlaps, finding intersections, and scheduling.

**Interval Representation:**

\`\`\`python
# Common representations:
interval = [start, end]  # List
interval = (start, end)  # Tuple
\`\`\`

**Interval Terminology:**

- **Overlapping**: Two intervals share at least one point
  - \`[1,3]\` and \`[2,4]\` overlap at \`[2,3]\`
- **Non-overlapping**: No shared points
  - \`[1,2]\` and \`[3,4]\` don't overlap
- **Touching**: End of one equals start of another
  - \`[1,2]\` and \`[2,3]\` touch at 2
- **Contained**: One interval fully inside another
  - \`[2,3]\` contained in \`[1,4]\`
- **Disjoint**: No overlap or touch
  - \`[1,2]\` and \`[4,5]\` are disjoint

---

**Visual Examples:**

\`\`\`
Overlapping:
[-----)      interval 1: [1,4]
   [-----)   interval 2: [2,5]
Overlap: [2,4]

Non-overlapping:
[----)        interval 1: [1,3]
       [----) interval 2: [5,7]

Touching:
[----)        interval 1: [1,3]
    [-----)   interval 2: [3,6]
\`\`\`

---

**Common Use Cases:**

1. **Calendar/Scheduling**
   - Meeting room allocation
   - Event scheduling
   - Resource booking

2. **Time Management**
   - Task scheduling
   - CPU job scheduling
   - Timeline visualization

3. **Geometry**
   - Line segment intersection
   - Rectangle overlap
   - Range queries

4. **Data Processing**
   - Log merging
   - Time series data
   - Range consolidation

---

**Key Insight:**

**Most interval problems become easier after sorting!**

Sort by start time (or end time) to:
- Process intervals in order
- Detect overlaps efficiently
- Merge adjacent intervals`,
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain what intervals are and why sorting is almost always the first step. What does sorting enable?',
                    sampleAnswer:
                        'Intervals represent ranges with start and end points, like [1,3] means 1 to 3. Sorting by start time is usually first step because it establishes order, making overlaps detectable. After sorting, if current interval starts before previous ends, they overlap. Without sorting, you would need to compare every pair - O(n²). With sorting, one pass detects all overlaps - O(n). Sorting also enables greedy algorithms: process intervals in order, make local decisions. For example, merge intervals: after sorting, only need to check current vs last merged. Meeting rooms: after sorting, compare consecutive intervals for conflicts. The pattern: sort enables linear-time algorithms instead of quadratic brute force. Sorting cost O(n log n) is worth it for O(n) processing.',
                    keyPoints: [
                        'Intervals: ranges with start and end',
                        'Sort by start time establishes order',
                        'Enables O(n) overlap detection vs O(n²)',
                        'Greedy algorithms process in order',
                        'Sort O(n log n) worth it for O(n) processing',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Describe how to detect if two intervals overlap. What is the overlap condition?',
                    sampleAnswer:
                        'Two intervals [a_start, a_end] and [b_start, b_end] overlap if they share any points. The overlap condition: a_start < b_end AND b_start < a_end. Or equivalently, NOT (a_end <= b_start OR b_end <= a_start). Intuition: they do NOT overlap if a completely before b (a_end <= b_start) or b completely before a (b_end <= a_start). Otherwise they overlap. For example, [1,3] and [2,5] overlap because 1 < 5 AND 2 < 3. [1,2] and [3,4] do not overlap because 2 <= 3 (first ends before second starts). This condition works regardless of which interval starts first. After sorting, can simplify: if current.start < previous.end, they overlap. The overlap check is fundamental to all interval problems.',
                    keyPoints: [
                        'Overlap: a_start < b_end AND b_start < a_end',
                        'Or: NOT (a before b OR b before a)',
                        'After sorting: current.start < previous.end',
                        'Works regardless of which starts first',
                        'Fundamental to all interval problems',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Walk me through common interval problems. What patterns do you see?',
                    sampleAnswer:
                        'Common problems: Merge Intervals (combine overlapping), Meeting Rooms I (can attend all?), Meeting Rooms II (min rooms needed), Insert Interval (add and merge), Interval List Intersections (find common parts). Patterns: Most start with sorting by start time. Merge pattern: iterate, track current merged interval, extend or add new. Counting pattern: track active intervals using start/end events or heap. Greedy pattern: sort by end time, select maximum non-overlapping. For example, Merge Intervals sorts then iterates comparing current with last merged. Meeting Rooms II uses min heap to track end times of ongoing meetings. Insert Interval handles three phases: before, overlapping, after. Recognize which pattern by what problem asks: merge, count overlaps, select maximum, find intersections.',
                    keyPoints: [
                        'Common: merge, meeting rooms, insert, intersections',
                        'Pattern 1: merge by extending or adding',
                        'Pattern 2: count active with events or heap',
                        'Pattern 3: greedy with end-time sorting',
                        'Recognize pattern from problem requirement',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'When do two intervals overlap?',
                    options: [
                        'When they are equal',
                        'When they share at least one point (start1 <= end2 and start2 <= end1)',
                        'When they touch',
                        'Never',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Two intervals [a,b] and [c,d] overlap when they share at least one point. This happens when: start1 <= end2 AND start2 <= end1.',
                },
                {
                    id: 'mc2',
                    question: 'Why is sorting usually the first step in interval problems?',
                    options: [
                        'It makes the output sorted',
                        'Establishes order, enabling O(N) overlap detection instead of O(N²)',
                        'It is required by the problem',
                        'Random choice',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Sorting by start time enables linear-time overlap detection. Without sorting, comparing all pairs takes O(N²). With sorting, one pass detects overlaps in O(N), making the O(N log N) sort worthwhile.',
                },
                {
                    id: 'mc3',
                    question: 'What is the typical time complexity of interval problems?',
                    options: [
                        'O(N)',
                        'O(N log N) due to initial sorting',
                        'O(N²)',
                        'O(log N)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Most interval problems require O(N log N) time for sorting, followed by O(N) processing, giving O(N log N) total complexity.',
                },
                {
                    id: 'mc4',
                    question: 'What are intervals commonly used for?',
                    options: [
                        'Sorting numbers',
                        'Scheduling (meetings, tasks), time management, range queries',
                        'Graph traversal',
                        'String manipulation',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Intervals model time ranges, making them perfect for scheduling (meeting rooms, task scheduling), calendar applications, resource booking, and range-based queries.',
                },
                {
                    id: 'mc5',
                    question: 'What does it mean for intervals to be "disjoint"?',
                    options: [
                        'They overlap',
                        'No overlap and not touching (completely separate)',
                        'They are equal',
                        'One contains the other',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Disjoint intervals are completely separate with no overlap or touching. For example, [1,2] and [4,5] are disjoint because end1 < start2.',
                },
            ],
        },
        {
            id: 'operations',
            title: 'Interval Operations',
            content: `**Operation 1: Checking Overlap**

Two intervals \`[a_start, a_end]\` and \`[b_start, b_end]\` overlap if:

\`\`\`python
def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]

# Or equivalently:
def overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])
\`\`\`

**Logic:**
- NOT (a ends before b starts OR b ends before a starts)

---

**Operation 2: Merging Two Intervals**

If intervals overlap, merge them:

\`\`\`python
def merge_two(a, b):
    if not overlaps(a, b):
        return [a, b]  # Cannot merge
    return [[min(a[0], b[0]), max(a[1], b[1])]]
\`\`\`

**Example:**
- \`[1,3]\` + \`[2,5]\` → \`[1,5]\`
- \`[1,4]\` + \`[2,3]\` → \`[1,4]\` (b contained in a)

---

**Operation 3: Finding Intersection**

The overlapping part of two intervals:

\`\`\`python
def intersection(a, b):
    if not overlaps(a, b):
        return None
    return [max(a[0], b[0]), min(a[1], b[1])]
\`\`\`

**Example:**
- \`[1,5]\` ∩ \`[3,7]\` = \`[3,5]\`

---

**Operation 4: Merging Multiple Intervals**

Merge all overlapping intervals in a list:

\`\`\`python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # If overlaps, merge
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            # No overlap, add as new interval
            merged.append(current)
    
    return merged
\`\`\`

**Time**: O(n log n) for sorting
**Space**: O(n) for result

---

**Operation 5: Inserting Interval**

Insert new interval and merge overlaps:

\`\`\`python
def insert_interval(intervals, new):
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals before new
    while i < n and intervals[i][1] < new[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new[1]:
        new[0] = min(new[0], intervals[i][0])
        new[1] = max(new[1], intervals[i][1])
        i += 1
    result.append(new)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result
\`\`\`

---

**Operation 6: Finding Free Time**

Given busy intervals, find free slots:

\`\`\`python
def find_free_time(intervals, start, end):
    intervals.sort()
    free = []
    
    prev_end = start
    for interval in intervals:
        if interval[0] > prev_end:
            free.append([prev_end, interval[0]])
        prev_end = max(prev_end, interval[1])
    
    # Last free slot
    if prev_end < end:
        free.append([prev_end, end])
    
    return free
\`\`\``,
            codeExample: `from typing import List


def overlaps(a: List[int], b: List[int]) -> bool:
    """
    Check if two intervals overlap.
    Time: O(1)
    """
    return a[0] <= b[1] and b[0] <= a[1]


def merge_two(a: List[int], b: List[int]) -> List[List[int]]:
    """
    Merge two intervals if they overlap.
    Returns list of 1 or 2 intervals.
    """
    if not overlaps(a, b):
        return [a, b]
    return [[min(a[0], b[0]), max(a[1], b[1])]]


def intersection(a: List[int], b: List[int]) -> List[int]:
    """
    Find intersection of two intervals.
    Returns None if no overlap.
    """
    if not overlaps(a, b):
        return None
    return [max(a[0], b[0]), min(a[1], b[1])]


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge all overlapping intervals.
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # Check overlap
        if current[0] <= last[1]:
            # Merge by extending end
            last[1] = max(last[1], current[1])
        else:
            # No overlap, add new interval
            merged.append(current)
    
    return merged


def insert_interval(
    intervals: List[List[int]], new_interval: List[int]
) -> List[List[int]]:
    """
    Insert interval and merge overlaps.
    Time: O(n), Space: O(n)
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals that end before new starts
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals with new
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result


def find_min_rooms(intervals: List[List[int]]) -> int:
    """
    Minimum meeting rooms needed (interval overlap count).
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return 0
    
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])
    
    rooms = 0
    max_rooms = 0
    s = e = 0
    
    while s < len(starts):
        if starts[s] < ends[e]:
            # New meeting starts, need room
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            # Meeting ends, free room
            rooms -= 1
            e += 1
    
    return max_rooms`,
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain the merge operation for intervals. How do you determine the new merged interval?',
                    sampleAnswer:
                        'Merge combines overlapping intervals into one. After sorting, iterate through intervals. Track current merged interval. For each interval: if it overlaps with current (start <= current.end), extend current by taking max of end times. If no overlap (start > current.end), current is complete, add to result, start new current. At end, add final current. The key: merged interval start is min of all starts (first interval start after sorting), end is max of all ends (keep extending). For example, [1,3], [2,6], [8,10]: merge [1,3] and [2,6] into [1,6] (max of 3,6 is 6), then [8,10] separate. The operation squashes overlapping intervals into minimal set of non-overlapping intervals.',
                    keyPoints: [
                        'After sort, iterate tracking current merged',
                        'Overlap: extend current.end to max(current.end, interval.end)',
                        'No overlap: add current, start new',
                        'Merged start: first start, end: max of all ends',
                        'Output: minimal non-overlapping set',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Describe the insert operation. How do you handle the three phases: before, overlapping, after?',
                    sampleAnswer:
                        'Insert adds new interval and merges overlaps. Three phases: Phase 1 (before): add all intervals that end before new starts (interval.end < new.start). Phase 2 (overlapping): merge all intervals that overlap with new. Track min start and max end of merged interval. Intervals overlap if interval.start <= new.end (since we are extending new). Phase 3 (after): add all remaining intervals that start after new ends. The beauty: we process intervals in one pass, identifying which phase based on start/end comparisons. For example, insert [4,8] into [[1,2], [3,5], [6,7], [9,10]]: before=[1,2], merge [3,5], [4,8], [6,7] into [3,8], after=[9,10]. Result: [[1,2], [3,8], [9,10]].',
                    keyPoints: [
                        'Phase 1: add intervals ending before new starts',
                        'Phase 2: merge overlapping, track min start, max end',
                        'Phase 3: add intervals starting after new ends',
                        'One pass: identify phase by comparisons',
                        'Overlap check: interval.start <= new.end',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Walk me through the interval scheduling problem using min heap. Why use a heap?',
                    sampleAnswer:
                        'Meeting Rooms II: find minimum rooms needed for overlapping meetings. Sort meetings by start time. Use min heap to track end times of ongoing meetings. For each meeting: if heap not empty and earliest ending meeting (heap top) ends before current starts, that room is free (pop heap). Add current meeting end time to heap (allocate room). Heap size at any point is number of active meetings (rooms needed). Return max heap size seen. Why heap? It efficiently tracks which meeting ends earliest - the one we should check first for freeing up. Without heap, we would need to scan all active meetings O(n) each time. With heap, pop and push are O(log n). Total: O(n log n) for sort + O(n log n) for heap operations = O(n log n).',
                    keyPoints: [
                        'Sort by start, heap tracks end times',
                        'Earliest ending meeting at heap top',
                        'Pop if ends before current starts (room free)',
                        'Push current end (allocate room)',
                        'Heap size = active meetings = rooms needed',
                        'Why heap: O(log n) vs O(n) to find earliest',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the condition for two intervals to overlap?',
                    options: [
                        'start1 < start2',
                        'start1 <= end2 AND start2 <= end1',
                        'end1 == start2',
                        'start1 == start2',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Two intervals overlap when: start1 <= end2 AND start2 <= end1. This checks if they share any common point.',
                },
                {
                    id: 'mc2',
                    question: 'In the merge intervals pattern, when do you extend the current interval?',
                    options: [
                        'Always',
                        'When new interval\'s start <= current interval\'s end (overlap detected)',
                        'When new interval is larger',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'After sorting by start time, if new interval\'s start <= current end, they overlap. Extend current to max(current.end, new.end).',
                },
                {
                    id: 'mc3',
                    question: 'Why sort intervals before merging?',
                    options: [
                        'Required by problem',
                        'Ensures intervals in order, only need to check consecutive pairs',
                        'Makes output sorted',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Sorting by start time ensures we process intervals in order. Then we only need to compare each interval with the last merged one, not all pairs - O(N) instead of O(N²).',
                },
                {
                    id: 'mc4',
                    question: 'What is the intersection of intervals [1,4] and [2,5]?',
                    options: [
                        '[1,5]',
                        '[2,4]',
                        '[1,2]',
                        'No intersection',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Intersection is [max(start1, start2), min(end1, end2)] = [max(1,2), min(4,5)] = [2,4].',
                },
                {
                    id: 'mc5',
                    question: 'What operation does "remove interval" perform?',
                    options: [
                        'Deletes an interval',
                        'Subtracts one interval from another, potentially splitting it',
                        'Merges intervals',
                        'Sorts intervals',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Removing [2,4] from [1,5] splits it into [1,2] and [4,5]. The operation removes the overlapping part and keeps non-overlapping parts.',
                },
            ],
        },
        {
            id: 'patterns',
            title: 'Common Interval Patterns',
            content: `**Pattern 1: Sort + Merge**

Most common pattern for interval problems.

**Steps:**
1. Sort intervals by start time
2. Iterate and compare with last merged
3. Merge if overlap, otherwise add new

**Problems**: Merge Intervals, Insert Interval

---

**Pattern 2: Sweep Line Algorithm**

Process events (starts and ends) in order.

**Technique:**
\`\`\`python
events = []
for start, end in intervals:
    events.append((start, 1))   # Start event: +1
    events.append((end, -1))    # End event: -1

events.sort()

active = 0
for time, delta in events:
    active += delta
    # active = number of overlapping intervals
\`\`\`

**Use Cases:**
- Count max overlaps (meeting rooms)
- Find busiest time
- Skyline problem

---

**Pattern 3: Interval Scheduling**

Select maximum non-overlapping intervals.

**Greedy Approach:**
1. Sort by end time (earliest finish first)
2. Select if doesn't overlap with last selected

\`\`\`python
def max_non_overlapping(intervals):
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    last_end = float('-inf')
    
    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end
    
    return count
\`\`\`

---

**Pattern 4: Interval Partitioning**

Partition intervals into minimum groups where no overlap.

**Same as**: Minimum meeting rooms problem

---

**Pattern 5: Point Coverage**

Find point covered by most intervals.

**Approach**: Sweep line algorithm

\`\`\`python
def most_covered_point(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end + 1, -1))  # +1 for exclusive end
    
    events.sort()
    
    max_coverage = 0
    current = 0
    best_point = None
    
    for point, delta in events:
        current += delta
        if current > max_coverage:
            max_coverage = current
            best_point = point
    
    return best_point, max_coverage
\`\`\`

---

**Pattern 6: Interval Intersection**

Find intersection of two lists of intervals.

**Approach**: Two pointers

\`\`\`python
def interval_intersection(A, B):
    i = j = 0
    result = []
    
    while i < len(A) and j < len(B):
        # Find intersection
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])
        
        if start <= end:
            result.append([start, end])
        
        # Move pointer of interval that ends first
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    
    return result
\`\`\``,
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Explain the greedy interval selection pattern. Why sort by end time instead of start time?',
                    sampleAnswer:
                        'Interval selection maximizes non-overlapping intervals chosen. Classic example: activity selection, non-overlapping intervals. Greedy: sort by end time, iterate, select interval if it starts after last selected ends. Why end time? Finishing early leaves more room for future intervals. If we sort by start time, we might pick long interval that blocks many short ones. Proof: if optimal solution differs, we can replace its first interval with our choice (earliest ending) without making solution worse. For example, intervals [[1,5], [2,3], [4,6]]: sort by end gives [2,3], [4,6] (2 selected). Sort by start gives [1,5], [4,6] but [1,5] blocks [2,3]. Greedy by end time is provably optimal for maximizing count.',
                    keyPoints: [
                        'Goal: maximize non-overlapping intervals',
                        'Greedy: sort by end, select if non-overlapping',
                        'End time: finishing early leaves more room',
                        'Start time: might pick long blocking interval',
                        'Provably optimal for maximizing count',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Describe using event-based counting for interval overlap problems. How does it work?',
                    sampleAnswer:
                        'Event-based treats interval as two events: start (+1) and end (-1). Create list of all events with times and types. Sort events by time. Sweep through events, maintain active count. Start event increments count, end event decrements. Track maximum count seen - this is maximum overlapping intervals. For Meeting Rooms II: max count is minimum rooms needed. For example, meetings [[0,30], [5,10], [15,20]]: events [(0,+1), (5,+1), (10,-1), (15,+1), (20,-1), (30,-1)]. Sweep: 0→1, 5→2 (max), 10→1, 15→2 (max), 20→1, 30→0. Max is 2 rooms. This avoids heap, simpler to code, same O(n log n) complexity.',
                    keyPoints: [
                        'Two events per interval: start (+1), end (-1)',
                        'Sort all events by time',
                        'Sweep: maintain active count',
                        'Track maximum count = max overlap',
                        'Alternative to heap, same complexity',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Walk me through interval intersection. How do you find common parts of two sorted interval lists?',
                    sampleAnswer:
                        'Given two sorted interval lists, find all intersections. Use two pointers, one per list. At each step: check if current intervals from both lists overlap. If overlap, intersection is [max(start1, start2), min(end1, end2)] - add to result. Move pointer of interval that ends first (it cannot intersect with future intervals from other list). If no overlap, move pointer of interval that starts first. Continue until either pointer reaches end. For example, A=[[0,2],[5,10]], B=[[1,5],[8,12]]: compare [0,2] and [1,5], overlap [1,2]. Move A pointer (ends at 2). Compare [5,10] and [1,5], overlap [5,5]. Move B pointer (ends at 5). Compare [5,10] and [8,12], overlap [8,10]. Done. The two-pointer technique efficiently finds all intersections in O(n+m).',
                    keyPoints: [
                        'Two pointers, one per sorted list',
                        'Check overlap: intersection is [max(starts), min(ends)]',
                        'Move pointer of interval ending first',
                        'Continue until either list exhausted',
                        'O(n+m) using two pointers',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the most common pattern for interval problems?',
                    options: [
                        'Dynamic programming',
                        'Sort by start time + merge/process in one pass',
                        'Binary search',
                        'Backtracking',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'The Sort + Merge pattern is most common: sort intervals by start time, then process in one pass comparing current with last merged. This enables O(N) processing after O(N log N) sort.',
                },
                {
                    id: 'mc2',
                    question: 'For counting maximum overlapping intervals, what technique is used?',
                    options: [
                        'Sort only',
                        'Sweep line or min heap to track active intervals',
                        'Hash map',
                        'DFS',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Use sweep line (sort start/end events) or min heap (track end times) to count active intervals at any time. Max active = maximum overlapping intervals.',
                },
                {
                    id: 'mc3',
                    question: 'For maximum non-overlapping intervals (activity selection), what should you sort by?',
                    options: [
                        'Start time',
                        'End time',
                        'Duration',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Sort by end time and greedily select intervals ending earliest. This maximizes remaining time for future intervals, giving optimal non-overlapping selection.',
                },
                {
                    id: 'mc4',
                    question: 'What is the sweep line technique?',
                    options: [
                        'Sorting intervals',
                        'Processing start/end events in chronological order to track active intervals',
                        'Merging intervals',
                        'Binary search',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Sweep line treats starts as +1 events and ends as -1 events. Sort all events, process chronologically, track active count. This finds maximum overlapping intervals efficiently.',
                },
                {
                    id: 'mc5',
                    question: 'When inserting a new interval into sorted intervals, what are the three phases?',
                    options: [
                        'Start, middle, end',
                        'Before (no overlap), overlap (merge), after (no overlap)',
                        'Left, right, center',
                        'First, second, third',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Insert interval has 3 phases: 1) Add all intervals ending before new starts (before), 2) Merge all overlapping intervals (overlap), 3) Add remaining intervals (after).',
                },
            ],
        },
        {
            id: 'complexity',
            title: 'Complexity Analysis',
            content: `**Time Complexity Patterns:**

| Operation | Unsorted | Sorted | Notes |
|-----------|----------|--------|-------|
| Merge Intervals | O(n log n) | O(n) | Sorting dominates |
| Insert Interval | O(n log n) | O(n) | Linear if sorted |
| Overlap Check | O(n²) | O(n log n) | Much better sorted |
| Find Free Time | O(n log n) | O(n) | Needs sorting |
| Interval Intersection | O(n²) | O(n+m) | Two pointers |

**Key Insight:** Sorting intervals costs O(n log n) but enables O(n) operations.

---

**Space Complexity:**

**Result Storage:**
- Merging: O(n) worst case (no merges)
- Insertion: O(n)
- Intersection: O(min(n, m))

**Auxiliary Space:**
- Sorting: O(log n) for quicksort
- Sweep line: O(n) for events array

---

**Common Complexities:**

**Merge Intervals:**
- Time: O(n log n) for sort + O(n) for merge = **O(n log n)**
- Space: O(n) for result

**Insert Interval (sorted input):**
- Time: **O(n)**
- Space: O(n)

**Meeting Rooms II:**
- Time: O(n log n) for sorting
- Space: O(n) for events

**Interval Intersection:**
- Time: **O(n + m)** with two pointers
- Space: O(min(n, m)) for result

---

**Optimization Techniques:**

**1. In-place Modification**
If allowed to modify input, can save space:
\`\`\`python
def merge_inplace(intervals):
    intervals.sort()
    write = 0
    for read in range(1, len(intervals)):
        if intervals[read][0] <= intervals[write][1]:
            intervals[write][1] = max(
                intervals[write][1],
                intervals[read][1]
            )
        else:
            write += 1
            intervals[write] = intervals[read]
    return intervals[:write + 1]
\`\`\`

**2. Early Termination**
For insertion, stop once past new interval.

**3. Custom Sorting**
Sort by end time for scheduling problems.`,
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Compare time complexity of different interval approaches: sorting vs heap vs events. When is each best?',
                    sampleAnswer:
                        'Sorting approach: O(n log n) sort + O(n) iterate = O(n log n) total. Works for merge, basic overlap detection. Heap approach: O(n log n) sort + O(n log n) heap operations = O(n log n) total. Best for tracking dynamic set like meeting rooms (need earliest ending). Event-based: O(n) create events + O(n log n) sort + O(n) sweep = O(n log n) total. Simpler than heap for max overlap. All are O(n log n) asymptotically, but constants differ. Sorting is simplest, use when one pass after sort suffices. Heap when need priority queue (earliest/latest). Events when counting overlaps. Space: sorting O(1) extra, heap O(n), events O(n). Choose based on problem needs, not just complexity.',
                    keyPoints: [
                        'All common approaches: O(n log n) time',
                        'Sorting: simplest, O(1) space',
                        'Heap: O(n) space, dynamic priority',
                        'Events: O(n) space, counting overlaps',
                        'Choose by problem needs, not just complexity',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Explain when interval problems require O(n²) time. What signals this?',
                    sampleAnswer:
                        'Some interval problems cannot avoid O(n²). Signal: need to compare every pair or track interactions between all intervals. Examples: find all pairs of overlapping intervals (must compare all pairs), interval queries without preprocessing, problems where order matters for all pairs. However, many apparent O(n²) problems reduce to O(n log n) with sorting. Rule: if can sort and process in order, likely O(n log n). If need all pairwise comparisons with no structure to exploit, likely O(n²). For example, "which intervals overlap with interval i?" is O(n) per query without preprocessing, O(n²) for all queries. With sorting and binary search, O(n log n) preprocess + O(log n) per query. Always try sorting first - it converts many O(n²) to O(n log n).',
                    keyPoints: [
                        'O(n²) when: all pairs, no exploitable structure',
                        'Examples: all overlapping pairs, no sorting helps',
                        'Many apparent O(n²) reduce to O(n log n)',
                        'Sorting converts many O(n²) to O(n log n)',
                        'Try sorting first as optimization',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Describe space complexity tradeoffs in interval problems. When do you need O(n) vs O(1) extra space?',
                    sampleAnswer:
                        'O(1) extra space: in-place merge after sorting, reusing input array. Only need a few variables to track current interval. Possible when output size is at most input size. O(n) extra space: heap for meeting rooms (stores end times), event list (2n events), result array when output size unbounded. Cannot avoid when: need data structure (heap, set), result is separate from input, cannot modify input. For example, merge intervals can be O(1) if allowed to modify input and output fits in input. Meeting rooms II needs O(n) heap. Choose O(1) when possible (cleaner, less memory), but O(n) is fine for most interview problems. Clarify if in-place required. Often the clarity of O(n) space outweighs O(1) complexity.',
                    keyPoints: [
                        'O(1): in-place merge, reuse input, few variables',
                        'O(n): heap, event list, unbounded output',
                        'O(n) needed: data structures, separate result',
                        'In-place often more complex to implement',
                        'Clarity vs space: O(n) often acceptable',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the time complexity of most interval problems?',
                    options: [
                        'O(N)',
                        'O(N log N) - dominated by sorting',
                        'O(N²)',
                        'O(log N)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Most interval problems require O(N log N) sorting followed by O(N) processing, giving O(N log N) total complexity.',
                },
                {
                    id: 'mc2',
                    question: 'What is the space complexity of in-place interval merging?',
                    options: [
                        'O(N)',
                        'O(1) - modify input array directly',
                        'O(log N)',
                        'O(N²)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'If allowed to modify the input, interval merging can be done in-place with O(1) extra space by using write/read pointers.',
                },
                {
                    id: 'mc3',
                    question: 'What is the complexity of interval intersection with two pointers?',
                    options: [
                        'O(N log N)',
                        'O(N + M) - linear in sum of list lengths',
                        'O(N × M)',
                        'O(log N)',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'When both interval lists are already sorted, two pointers process each list once, giving O(N + M) time complexity.',
                },
                {
                    id: 'mc4',
                    question: 'How does using a min heap help in meeting rooms II?',
                    options: [
                        'Sorts the meetings',
                        'Tracks earliest ending meeting in O(log N), avoiding O(N) scan',
                        'Counts meetings',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Min heap maintains earliest ending meeting at root. Checking if room is free takes O(log N) instead of O(N) linear scan, making algorithm more efficient.',
                },
                {
                    id: 'mc5',
                    question: 'What is the benefit of the sweep line technique?',
                    options: [
                        'Easier to code',
                        'Converts overlap counting to event processing - O(N log N) instead of O(N²)',
                        'Uses less memory',
                        'Random',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Sweep line processes start/end events chronologically instead of comparing all interval pairs. This reduces complexity from O(N²) to O(N log N) for overlap counting.',
                },
            ],
        },
        {
            id: 'templates',
            title: 'Code Templates',
            content: `**Template 1: Merge Intervals**
\`\`\`python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged
\`\`\`

---

**Template 2: Insert Interval**
\`\`\`python
def insert(intervals, new):
    result = []
    i = 0
    
    # Before
    while i < len(intervals) and intervals[i][1] < new[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge
    while i < len(intervals) and intervals[i][0] <= new[1]:
        new = [
            min(new[0], intervals[i][0]),
            max(new[1], intervals[i][1])
        ]
        i += 1
    result.append(new)
    
    # After
    result.extend(intervals[i:])
    return result
\`\`\`

---

**Template 3: Sweep Line (Meeting Rooms)**
\`\`\`python
def min_meeting_rooms(intervals):
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])
    
    rooms = max_rooms = 0
    s = e = 0
    
    while s < len(starts):
        if starts[s] < ends[e]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            rooms -= 1
            e += 1
    
    return max_rooms
\`\`\`

---

**Template 4: Interval Intersection**
\`\`\`python
def interval_intersection(A, B):
    i = j = 0
    result = []
    
    while i < len(A) and j < len(B):
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])
        
        if start <= end:
            result.append([start, end])
        
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    
    return result
\`\`\`

---

**Template 5: Check Overlap**
\`\`\`python
def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]

# Or (easier to remember):
def overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])
\`\`\``,
            quiz: [
                {
                    id: 'q1',
                    question:
                        'Walk me through the merge intervals template. What are the key steps?',
                    sampleAnswer:
                        'Merge intervals template has four steps. First, sort intervals by start time. Second, initialize result with first interval. Third, iterate from second interval: if current overlaps with last in result (current.start <= last.end), merge by updating last.end to max(last.end, current.end). If no overlap, append current to result. Fourth, return result. The key insight: after sorting, only need to check current against last merged interval, not all previous intervals. For example, [[1,3], [2,6], [8,10]]: start with [1,3]. [2,6] overlaps (2 <= 3), merge to [1,6]. [8,10] no overlap (8 > 6), append. Result: [[1,6], [8,10]]. This template is foundation for many interval problems.',
                    keyPoints: [
                        'Step 1: sort by start time',
                        'Step 2: result starts with first interval',
                        'Step 3: iterate, merge or append',
                        'Merge: update last.end to max',
                        'Only check against last merged',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Explain the meeting rooms heap template. How do you track end times efficiently?',
                    sampleAnswer:
                        'Meeting rooms template uses min heap for end times. First, sort meetings by start time. Second, create min heap. Third, for each meeting: if heap not empty and heap top (earliest ending meeting) ends before or at current start, pop heap (room freed). Push current meeting end time to heap (allocate room). Fourth, heap size is rooms needed. Return max heap size or final heap size. The min heap automatically gives us earliest ending meeting at top in O(log n). For example, [[0,30], [5,10], [15,20]]: heap=[30], then [10,30], then [20,30] (10 popped), max size 2. The heap efficiently tracks which meeting ends next without scanning all active meetings.',
                    keyPoints: [
                        'Sort by start, min heap for end times',
                        'For each: pop if top ends before current',
                        'Push current end (allocate room)',
                        'Heap size = active meetings',
                        'Min heap: O(log n) to get earliest',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'Describe the overlap check helper function. Why is it useful to abstract this?',
                    sampleAnswer:
                        'Overlap check helper: overlaps(a, b) returns True if intervals overlap. Implementation: return not (a.end <= b.start or b.end <= a.start). Or alternatively: a.start < b.end and b.start < a.end. Abstracting to helper function: makes code cleaner, reusable across problems, easier to test, centralizes overlap logic. If overlap logic changes (inclusive vs exclusive ends), only update one place. Many interval problems need overlap check repeatedly: merge, intersection, conflict detection. For example, in My Calendar problem, check if new interval overlaps with any existing. Helper makes this simple: any(overlaps(new, existing) for existing in booked). Without helper, repeat complex condition everywhere, risking bugs. Clean code principle: abstract repeated logic.',
                    keyPoints: [
                        'Helper: overlaps(a, b) checks overlap',
                        'Implementation: not (a.end <= b.start or b.end <= a.start)',
                        'Benefits: cleaner, reusable, testable',
                        'Centralize logic: change once, effect everywhere',
                        'Many problems need repeated overlap checks',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the first step in the merge intervals template?',
                    options: [
                        'Create result list',
                        'Sort intervals by start time',
                        'Check for overlaps',
                        'Return result',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Always sort intervals by start time first. This enables linear-time merging by comparing each interval only with the last merged one.',
                },
                {
                    id: 'mc2',
                    question: 'In the insert interval template, what are the three phases?',
                    options: [
                        'Sort, merge, return',
                        'Add intervals before new, merge overlapping, add intervals after',
                        'Start, middle, end',
                        'Left, center, right',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Insert interval template: 1) Add all intervals ending before new starts (no overlap), 2) Merge all overlapping intervals, 3) Add remaining intervals after overlap region.',
                },
                {
                    id: 'mc3',
                    question: 'What does the meeting rooms I template check?',
                    options: [
                        'Number of rooms needed',
                        'Whether any two consecutive intervals overlap (one room sufficient)',
                        'Maximum overlap',
                        'Minimum overlap',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Meeting Rooms I checks if you can attend all meetings. Sort by start time, then check if any consecutive intervals overlap. If none overlap, one room (person) is sufficient.',
                },
                {
                    id: 'mc4',
                    question: 'What data structure does meeting rooms II template typically use?',
                    options: [
                        'Array',
                        'Min heap to track earliest ending meetings',
                        'Stack',
                        'Hash map',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Meeting Rooms II uses min heap to track end times of ongoing meetings. Heap size at any time = rooms needed. Pop when meeting ends, push new end times.',
                },
                {
                    id: 'mc5',
                    question: 'In interval intersection template, how do you find the intersection of two intervals?',
                    options: [
                        'min(start1, start2), max(end1, end2)',
                        '[max(start1, start2), min(end1, end2)]',
                        'Average of all values',
                        'Random selection',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Intersection is [max(start1, start2), min(end1, end2)]. The intersection starts at the later start and ends at the earlier end. Valid only if start <= end.',
                },
            ],
        },
        {
            id: 'interview',
            title: 'Interview Strategy',
            content: `**Recognition Signals:**

**Use Intervals when you see:**
- "Merge", "overlap", "conflict"
- "Schedule", "calendar", "meeting rooms"
- "Ranges", "time slots", "bookings"
- Array of \`[start, end]\` pairs
- "Free time", "available slots"
- "Maximum non-overlapping"

---

**Problem-Solving Steps:**

**Step 1: Clarify (2 min)**
- Are intervals inclusive or exclusive?
- Can start == end? (single point)
- Are intervals already sorted?
- Can intervals be negative?
- What about empty input?

**Step 2: Choose Approach (2 min)**
- **Sort + Merge**: Most common
- **Sweep Line**: Count overlaps
- **Two Pointers**: Intersection
- **Greedy**: Scheduling

**Step 3: Sort Decision (1 min)**
- Sort by start (most common)
- Sort by end (scheduling)
- Sort by both (custom comparator)

**Step 4: Handle Edge Cases (2 min)**
- Empty array
- Single interval
- No overlaps
- All overlaps

**Step 5: Implement (10 min)**

**Step 6: Test (3 min)**
- Basic case
- All merge into one
- No merges
- Touch but don't overlap

---

**Interview Communication:**

**Example: Merge Intervals**

*Interviewer: Given intervals, merge all overlapping ones.*

**You:**

1. **Clarify:**
   - "Are intervals inclusive on both ends?"
   - "Can I modify the input?"
   - "Are they already sorted?"

2. **Approach:**
   - "I'll sort by start time - O(n log n)."
   - "Then iterate once, merging overlaps - O(n)."
   - "Overall O(n log n) time, O(n) space."

3. **Overlap Logic:**
   - "Two intervals overlap if start of second ≤ end of first."
   - "Merge by taking min start, max end."

4. **Walk Through:**
   \`\`\`
   Input: [[1,3],[2,6],[8,10],[15,18]]
   After sort: same (already sorted)
   
   merged = [[1,3]]
   Process [2,6]: 2 ≤ 3, merge → [[1,6]]
   Process [8,10]: 8 > 6, add → [[1,6],[8,10]]
   Process [15,18]: 15 > 10, add → [[1,6],[8,10],[15,18]]
   \`\`\`

---

**Common Mistakes:**

**1. Wrong Overlap Check**
\`\`\`python
# Wrong: misses touching intervals
if current[0] < last[1]:

# Right: includes touching
if current[0] <= last[1]:
\`\`\`

**2. Forgetting to Sort**
Always sort unless guaranteed sorted!

**3. Modifying During Iteration**
Use separate result array.

**4. Off-by-One Errors**
Be clear on inclusive/exclusive ends.

---

**Practice Progression:**

**Week 1: Basics**
- Merge Intervals
- Insert Interval
- Non-overlapping Intervals

**Week 2: Variations**
- Meeting Rooms
- Meeting Rooms II
- Minimum Arrows to Burst Balloons

**Week 3: Advanced**
- Interval List Intersections
- Employee Free Time
- My Calendar II`,
            quiz: [
                {
                    id: 'q1',
                    question:
                        'How do you recognize an interval problem? What keywords or patterns signal this?',
                    sampleAnswer:
                        'Keywords: "intervals", "ranges", "meetings", "events", "schedule", "overlapping", "merge", "insert". Problem types: merge overlapping ranges, schedule meetings, find free time, count conflicts, maximize selected intervals. Data format: array of [start, end] pairs. If you see pairs representing ranges and need to process relationships between them, likely interval problem. For example, "given meeting times, find minimum conference rooms" is intervals. "Merge overlapping time ranges" is intervals. Pattern: data is ranges, need to handle overlaps or sequential processing. Key question: does problem involve ranges with start/end that might overlap or need ordering? If yes, probably interval problem with sorting as first step.',
                    keyPoints: [
                        'Keywords: intervals, ranges, meetings, schedule, overlap',
                        'Data: array of [start, end] pairs',
                        'Types: merge, schedule, conflicts, maximize',
                        'Pattern: ranges with potential overlaps',
                        'First step usually: sort by start time',
                    ],
                },
                {
                    id: 'q2',
                    question:
                        'Walk me through your interval interview approach from recognition to implementation.',
                    sampleAnswer:
                        'First, recognize its interval problem from keywords. Second, clarify: are intervals closed/open, can they be negative, are they sorted, what to return. Third, explain approach: sort by start time (or end time for greedy), then iterate processing overlaps/merges. Fourth, state complexity: O(n log n) for sort dominates, O(n) for processing. Fifth, discuss implementation: use helper function for overlap check, consider heap vs events for counting. Sixth, draw example: show a few intervals, demonstrate sorting, show merging/processing. Seventh, code clearly with sort first, iterate with clear overlap logic. Test with edge cases: empty, single interval, all overlap, none overlap. Finally, discuss optimization: can we avoid sort? Is O(1) space possible? This systematic approach demonstrates depth.',
                    keyPoints: [
                        'Recognize keywords, clarify requirements',
                        'Explain: sort by start, process overlaps',
                        'Complexity: O(n log n) sort dominates',
                        'Draw example, show sorting and processing',
                        'Code: sort, iterate, overlap logic',
                        'Test edges, discuss optimizations',
                    ],
                },
                {
                    id: 'q3',
                    question:
                        'What are common interval mistakes and how do you avoid them?',
                    sampleAnswer:
                        'First: forgetting to sort - leads to missing overlaps. Second: wrong overlap condition - off-by-one with <= vs <. Third: not handling empty input. Fourth: modifying last interval incorrectly during merge (updating wrong end). Fifth: for heap solution, forgetting to pop freed rooms. Sixth: sorting by wrong field (start vs end). Seventh: inclusive vs exclusive interval ends not clarified. My strategy: always sort first, use overlap helper function, test with: empty, single, all overlap, none overlap, adjacent intervals ([1,2], [2,3]). Draw diagram to verify overlap logic. For merge, carefully track which interval is current. Heap problems: clearly understand what heap stores (end times). Most bugs come from overlap logic - nail that first.',
                    keyPoints: [
                        'Forget to sort → missing overlaps',
                        'Wrong overlap condition (off-by-one)',
                        'Modify wrong interval during merge',
                        'Test: empty, single, all/none overlap, adjacent',
                        'Use overlap helper, draw diagram',
                    ],
                },
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What keywords signal an interval problem?',
                    options: [
                        'Array, list, tree',
                        'Merge, overlap, schedule, meeting rooms, conflict',
                        'Sort, search, find',
                        'Hash, map, set',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Keywords like "merge", "overlap", "schedule", "calendar", "meeting rooms", "conflict", "range", and "time periods" strongly indicate interval-based solutions.',
                },
                {
                    id: 'mc2',
                    question: 'What is the first thing to clarify in an interval interview problem?',
                    options: [
                        'Complexity requirements',
                        'Are intervals inclusive/exclusive? Can I sort? Can I modify input?',
                        'Language preference',
                        'Test cases',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Clarify: Are endpoints inclusive or exclusive? Are intervals already sorted? Can you modify the input? These affect your solution approach significantly.',
                },
                {
                    id: 'mc3',
                    question: 'What is the most common first step when solving interval problems?',
                    options: [
                        'Create result list',
                        'Sort intervals by start time (unless already sorted)',
                        'Count intervals',
                        'Find maximum',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Almost all interval problems start by sorting intervals by start time. This enables efficient linear-time processing and overlap detection.',
                },
                {
                    id: 'mc4',
                    question: 'What is a common mistake in interval problems?',
                    options: [
                        'Sorting correctly',
                        'Wrong overlap check (using < instead of <=, missing touching intervals)',
                        'Using correct data structures',
                        'Good variable names',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Common mistake: using start < end instead of start <= end for overlap check, which misses touching intervals. Always clarify if touching counts as overlap.',
                },
                {
                    id: 'mc5',
                    question: 'What is a good practice progression for interval problems?',
                    options: [
                        'Start with hardest',
                        'Week 1: Merge/Insert, Week 2: Meeting Rooms, Week 3: Advanced patterns',
                        'Random order',
                        'Skip basics',
                    ],
                    correctAnswer: 1,
                    explanation:
                        'Progress: Week 1 basics (merge, insert), Week 2 variations (meeting rooms, overlaps), Week 3 advanced (scheduling, optimization). Build understanding incrementally.',
                },
            ],
        },
    ],
    keyTakeaways: [
        'Intervals represent ranges [start, end]; sorting makes problems tractable',
        'Two intervals overlap if a[0] <= b[1] AND b[0] <= a[1]',
        'Sort + Merge pattern: sort by start, iterate and merge overlaps in O(n)',
        'Sweep line: process start/end events separately for counting problems',
        'Interval scheduling: sort by end time, greedily select non-overlapping',
        'Most interval problems are O(n log n) due to sorting requirement',
        'Two pointers for intersection of sorted interval lists in O(n + m)',
        'Always clarify: inclusive/exclusive ends, can intervals touch, already sorted?',
    ],
    relatedProblems: ['merge-intervals', 'insert-interval', 'meeting-rooms-ii'],
};
