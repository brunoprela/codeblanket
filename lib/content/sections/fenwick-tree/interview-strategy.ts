/**
 * Interview Strategy Section
 */

export const interviewstrategySection = {
  id: 'interview-strategy',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Fenwick Tree when:**
- "Prefix sum with updates"
- "Range sum queries"
- "Cumulative frequencies"
- "Count inversions"
- "Dynamic ranking"

**Keywords:**
- "Prefix sum", "cumulative sum"
- "Range sum with updates"
- "Efficient queries and updates"
- "Counting smaller/larger elements"

**Problem-Solving Steps:**

**1. Identify if Fenwick applies:**
- Is it prefix/range sum?
- Do I need updates?
- Is there an inverse operation?

**2. Handle 1-indexing:**
- Fenwick uses 1-indexed arrays
- Convert input: \`update (i+1, val)\`
- Or use 0-indexed version

**3. Choose variant:**
- Basic: Point update, range query
- 2D: For matrix problems
- Binary search: For kth element

**4. Implementation tips:**
- Remember: \`i & -i\` gets last set bit
- Update: add last set bit (\`i += i & -i\`)
- Query: subtract last set bit (\`i -= i & -i\`)

**Common Mistakes:**
- Forgetting 1-indexing
- Wrong bit operation (\`i & i\` vs \`i & -i\`)
- Not initializing tree size correctly
- Using for min/max (need Segment Tree)

**Interview Tips:**
- Mention it's simpler than Segment Tree
- Explain bit manipulation trick
- Draw small example (n=8)
- Discuss limitations (no min/max)`,
};
