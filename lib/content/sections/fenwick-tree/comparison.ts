/**
 * Fenwick Tree vs Segment Tree Section
 */

export const comparisonSection = {
  id: 'comparison',
  title: 'Fenwick Tree vs Segment Tree',
  content: `**When to Use Fenwick Tree:**
- ✅ Prefix sums, range sums
- ✅ Operations with **inverse** (add/subtract, XOR)
- ✅ Simpler to code
- ✅ Less memory

**When to Use Segment Tree:**
- ✅ Min/max queries (no inverse)
- ✅ GCD queries
- ✅ More complex operations
- ✅ Range updates with lazy propagation
- ✅ Easier to understand/debug

**Fenwick Limitations:**
- ❌ Can't do min/max (no inverse operation)
- ❌ Range updates more complex
- ❌ Less intuitive than Segment Tree
- ❌ 1-indexed (not 0-indexed)

**Code Comparison:**

**Fenwick Tree:** ~20 lines
\`\`\`python
class FenwickTree:
    def __init__(self, n):
        self.tree = [0] * (n + 1)
    
    def update (self, i, delta):
        while i < len (self.tree):
            self.tree[i] += delta
            i += i & -i
    
    def query (self, i):
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total
\`\`\`

**Segment Tree:** ~50 lines (more complex but more powerful)

**Decision Guide:**
- Need sum/prefix queries? → Fenwick Tree
- Need min/max/gcd? → Segment Tree
- Simple problem? → Fenwick Tree
- Complex operations? → Segment Tree`,
};
