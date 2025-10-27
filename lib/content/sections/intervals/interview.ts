/**
 * Interview Strategy Section
 */

export const interviewSection = {
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

**You:**1. **Clarify:**
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
};
