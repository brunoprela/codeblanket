/**
 * Optimization Strategies & Trade-offs Section
 */

export const optimizationSection = {
  id: 'optimization',
  title: 'Optimization Strategies & Trade-offs',
  content: `**Common Optimization Techniques:**

**1. Use Better Data Structures**

❌ **Slow:**
\`\`\`python
def has_duplicate (arr):  # O(n²)
    for i in range (len (arr)):
        for j in range (i+1, len (arr)):
            if arr[i] == arr[j]:
                return True
    return False
\`\`\`

✅ **Fast:**
\`\`\`python
def has_duplicate (arr):  # O(n)
    seen = set()
    for num in arr:
        if num in seen:
            return True
        seen.add (num)
    return False
\`\`\`

**2. Avoid Redundant Work**

❌ **Slow:**
\`\`\`python
def fibonacci (n):  # O(2ⁿ)
    if n <= 1:
        return n
    return fibonacci (n-1) + fibonacci (n-2)
\`\`\`

✅ **Fast with Memoization:**
\`\`\`python
def fibonacci (n, cache={}):  # O(n)
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fibonacci (n-1, cache) + fibonacci (n-2, cache)
    return cache[n]
\`\`\`

**3. Precompute and Cache**

❌ **Slow:**
\`\`\`python
def query_sum (arr, queries):  # O(q × n)
    results = []
    for start, end in queries:
        results.append (sum (arr[start:end]))
    return results
\`\`\`

✅ **Fast with Prefix Sums:**
\`\`\`python
def query_sum (arr, queries):  # O(n + q)
    prefix = [0]
    for num in arr:
        prefix.append (prefix[-1] + num)
    results = []
    for start, end in queries:
        results.append (prefix[end] - prefix[start])
    return results
\`\`\`

**4. Two Pointers Instead of Nested Loops**

❌ **Slow:**
\`\`\`python
def two_sum_sorted (arr, target):  # O(n²)
    for i in range (len (arr)):
        for j in range (i+1, len (arr)):
            if arr[i] + arr[j] == target:
                return [i, j]
    return None
\`\`\`

✅ **Fast:**
\`\`\`python
def two_sum_sorted (arr, target):  # O(n)
    left, right = 0, len (arr) - 1
    while left < right:
        curr_sum = arr[left] + arr[right]
        if curr_sum == target:
            return [left, right]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return None
\`\`\`

**Common Trade-offs:**

**Time vs. Space:**
- **More Space for Speed:** Memoization, hash tables, prefix sums
- **More Time for Space:** In-place algorithms, iterative instead of recursive

**Preprocessing vs. Query Time:**
- Build index once (O(n)) → Fast queries (O(1))
- Good when many queries expected

**Code Simplicity vs. Performance:**
- Readable but slower
- Optimized but complex
- **Rule:** Optimize only when needed!

**When to Optimize:**1. **Measure first** - profile your code
2. **Identify bottlenecks** - don't optimize everywhere
3. **Consider trade-offs** - is the improvement worth the complexity?
4. **Keep it readable** - optimize only hot paths

**The Optimization Process:**1. **Analyze current complexity** - identify the bottleneck
2. **Look for patterns** - nested loops? redundant work?
3. **Consider better data structures** - hash map? heap? tree?
4. **Apply techniques** - memoization? two pointers? binary search?
5. **Verify correctness** - optimization should not break code!
6. **Measure improvement** - did it actually help?

**Red Flags in Code Review:**

- Nested loops over same data → Try hash map or two pointers
- Repeated calculations → Memoize or precompute
- Searching in list → Use set or hash map
- Sorting inside loop → Sort once outside
- Recursive without memoization → Add caching

**Interview Strategy:**1. Start with brute force - explain it clearly
2. Analyze its complexity
3. Identify what makes it slow
4. Propose optimization
5. Analyze improved complexity
6. Discuss trade-offs`,
};
