/**
 * Common Interval Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Common Interval Patterns',
  content: `**Pattern 1: Sort + Merge**

Most common pattern for interval problems.

**Steps:**1. Sort intervals by start time
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

**Greedy Approach:**1. Sort by end time (earliest finish first)
2. Select if doesn't overlap with last selected

\`\`\`python
def max_non_overlapping (intervals):
    intervals.sort (key=lambda x: x[1])
    
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
def most_covered_point (intervals):
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
};
