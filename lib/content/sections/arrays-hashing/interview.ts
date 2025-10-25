/**
 * Interview Strategy & Tips Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy & Tips',
  content: `**Recognition Signals:**

**Use Hash Table when you hear:**
- "Find duplicates"
- "Count frequency/occurrences"
- "Two sum" or pair problems
- "Group by..."
- "Unique elements"
- "First/last occurrence"

**Use Array techniques when you hear:**
- "Subarray" problems
- "In-place" modification
- "Sorted array"
- "Index" or "position"
- "Range" queries

**Step-by-Step Approach:**

**1. Clarify (30 seconds)**
- "Can I use extra space?" (Hash table uses O(n) space)
- "Is the input sorted?" (Might enable different approaches)
- "Are there duplicates?"
- "What\'s the expected size?" (Hash table overhead matters for small inputs)

**2. Brute Force First (1 minute)**
- State the O(n²) or O(n log n) approach
- Explain why it's not optimal
- Then propose hash table optimization

**3. Explain Optimization (1 minute)**
- "I can use a hash table to store..."
- "This reduces lookup from O(n) to O(1)"
- "Overall complexity improves to O(n)"

**4. Code (5-7 minutes)**
- Choose right structure: dict, set, Counter, defaultdict
- Handle edge cases: empty input, single element
- Consider collision/hash function (usually not asked)

**5. Test (2 minutes)**
- Empty input
- Single element
- All same elements
- All different elements
- Duplicates

**Common Mistakes:**

❌ **Using list for lookups** (O(n) instead of O(1))
✅ **Use set or dict**

❌ **Not handling missing keys**
✅ **Use .get() or defaultdict**

❌ **Modifying dict while iterating**
✅ **Iterate over copy: for key in list (dict.keys())**

❌ **Forgetting unhashable types** (lists, dicts can't be keys)
✅ **Convert to tuple: tuple (lst)**

**Follow-Up Questions:**

Q: "What if we can't use extra space?"
A: "I can use two pointers or sorting, but time complexity increases"

Q: "What about hash collisions?"
A: "Python\'s hash function is robust. In theory O(n) worst case, but O(1) average"

Q: "Can you do it without hash table?"
A: "Yes, by sorting first, but that's O(n log n) vs O(n)"

**Time Management:**
- Arrays: Usually 10-15 minutes for easy/medium
- Hash tables: Usually 15-20 minutes for medium
- Combined: 20-25 minutes for complex problems

**Resources:**
- LeetCode: Top 100 Liked (many array/hash problems)
- NeetCode: Arrays & Hashing playlist
- Practice daily until patterns become automatic`,
};
