/**
 * Interval Operations Section
 */

export const operationsSection = {
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
};
