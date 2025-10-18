/**
 * Code Templates Section
 */

export const templatesSection = {
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
};
