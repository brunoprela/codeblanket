/**
 * Introduction to Intervals Section
 */

export const introductionSection = {
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
};
