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
