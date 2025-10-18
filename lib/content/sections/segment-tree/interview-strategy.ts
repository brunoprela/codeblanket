/**
 * Interview Strategy Section
 */

export const interviewstrategySection = {
  id: 'interview-strategy',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Segment Tree when:**
- Multiple **range queries** on array
- Need **both queries and updates**
- Queries are sum/min/max/gcd
- Can't use simpler approaches

**Keywords:**
- "Range sum/min/max queries"
- "Efficient updates and queries"
- "Interval queries"
- "Dynamic array with queries"

**Problem-Solving Steps:**

**1. Identify if Segment Tree is Needed:**
- Can you use prefix sums? (query-only)
- Can you use Fenwick tree? (simpler, if applicable)
- Do you really need O(log N) for both operations?

**2. Choose Operation Type:**
- Sum? Min? Max? GCD?
- Point update or range update?
- Need lazy propagation?

**3. Implementation Strategy:**
- Start with basic point update
- Add range updates later if needed
- Test with small examples

**4. Edge Cases:**
- Single element array
- Query entire array
- Update and immediate query
- Multiple updates to same position

**Interview Tips:**
- Mention simpler alternatives first
- Explain why you need segment tree
- Draw tree structure for small example
- Discuss complexity trade-offs
- Code recursively (cleaner)

**Common Mistakes:**
- Forgetting to update parent nodes
- Wrong merge function
- Index calculations (2*i+1, 2*i+2)
- Not handling no-overlap case correctly
- Forgetting lazy propagation push`,
};
