/**
 * Arrays & Hashing module content - Professional & comprehensive guide
 */

import { Module } from '@/lib/types';

export const arraysHashingModule: Module = {
  id: 'arrays-hashing',
  title: 'Arrays & Hashing',
  description:
    'Master the fundamentals of array manipulation and hash table techniques for optimal performance.',
  icon: '🔢',
  sections: [
    {
      id: 'introduction',
      title: 'Arrays & Hash Tables: The Foundation',
      content: `Arrays and hash tables are the **most fundamental data structures** in computer science and appear in virtually every coding interview. Understanding these deeply is crucial for success.

**Why These Topics Matter:**
- **Arrays:** The most basic data structure, serving as building blocks for others
- **Hash Tables:** Provide O(1) average-case lookup, enabling powerful optimizations
- Together, they solve 30-40% of all interview problems

**Real-World Applications:**
- **Arrays:** Image processing (pixels), time series data, buffers
- **Hash Tables:** Caching systems, database indexing, symbol tables in compilers
- **Combined:** Frequency counting, grouping, deduplication

**Key Insight:**
Many O(n²) brute force solutions can be optimized to O(n) using hash tables to store and lookup information instantly.`,
    },
    {
      id: 'arrays',
      title: 'Array Fundamentals & Patterns',
      content: `**What is an Array?**
A contiguous block of memory storing elements of the same type, with O(1) access by index.

**Core Operations & Complexity:**
- **Access by index:** O(1)
- **Insert/Delete at end:** O(1) amortized
- **Insert/Delete at beginning/middle:** O(n) - requires shifting
- **Search unsorted:** O(n)
- **Search sorted:** O(log n) with binary search

**Common Array Patterns:**

**1. Two Pointers** (covered in separate module)
- Opposite direction for sorted arrays
- Same direction for in-place modifications

**2. Sliding Window** (covered in separate module)
- Fixed or variable size subarrays
- Running calculations over ranges

**3. Prefix Sum**
Build cumulative sum array for range queries:
\`\`\`python
prefix[i] = prefix[i-1] + arr[i]
range_sum(l, r) = prefix[r] - prefix[l-1]
\`\`\`

**4. Kadane's Algorithm**
Maximum subarray sum in O(n):
\`\`\`python
max_current = max_global = arr[0]
for i in range(1, len(arr)):
    max_current = max(arr[i], max_current + arr[i])
    max_global = max(max_global, max_current)
\`\`\`

**5. In-Place Reversal**
\`\`\`python
left, right = 0, len(arr) - 1
while left < right:
    arr[left], arr[right] = arr[right], arr[left]
    left += 1
    right -= 1
\`\`\``,
      codeExample: `def prefix_sum_array(nums: List[int]) -> List[int]:
    """Build prefix sum array for O(1) range queries."""
    prefix = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        prefix[i + 1] = prefix[i] + nums[i]
    return prefix

def max_subarray_sum(nums: List[int]) -> int:
    """Kadane's algorithm for maximum subarray sum."""
    max_current = max_global = nums[0]
    
    for i in range(1, len(nums)):
        max_current = max(nums[i], max_current + nums[i])
        max_global = max(max_global, max_current)
    
    return max_global`,
    },
    {
      id: 'hashing',
      title: 'Hash Tables: Fast Lookups & Storage',
      content: `**What is a Hash Table?**
A data structure that maps keys to values using a hash function, providing O(1) average-case operations.

**How It Works:**
1. **Hash Function:** Converts key to array index
2. **Collision Handling:** Chaining or open addressing
3. **Load Factor:** Determines when to resize (typically 0.75)

**Python's Hash Tables:**
- **dict:** Key-value pairs, insertion order preserved (Python 3.7+)
- **set:** Unique elements only, no values
- **Counter:** Specialized dict for counting (from collections)
- **defaultdict:** Automatic default values for missing keys

**Common Operations:**
\`\`\`python
# Dictionary
freq = {}
freq[key] = freq.get(key, 0) + 1  # Count occurrences

# Set
seen = set()
if num in seen:  # O(1) membership test
    return True
seen.add(num)

# Counter (best for frequency)
from collections import Counter
freq = Counter(nums)
most_common = freq.most_common(k)

# defaultdict (avoid KeyError)
from collections import defaultdict
graph = defaultdict(list)
graph[node].append(neighbor)  # No initialization needed
\`\`\`

**Hash Table Patterns:**

**1. Frequency Counting**
Count occurrences of elements

**2. Two Sum Pattern**
Store complement, check if current element's complement exists

**3. Grouping/Partitioning**
Group elements by property (anagrams, sum, pattern)

**4. Deduplication**
Use set to track seen elements

**5. Caching/Memoization**
Store computed results for reuse`,
      codeExample: `def two_sum(nums: List[int], target: int) -> List[int]:
    """Classic two sum using hash table."""
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """Group strings that are anagrams."""
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())`,
    },
    {
      id: 'complexity',
      title: 'Time & Space Complexity Analysis',
      content: `**Array Operations:**

| Operation | Time | Notes |
|-----------|------|-------|
| Access by index | O(1) | Direct memory access |
| Search (unsorted) | O(n) | Must check each element |
| Search (sorted) | O(log n) | Binary search |
| Insert/Delete at end | O(1)* | *Amortized with dynamic arrays |
| Insert/Delete at start | O(n) | Requires shifting all elements |
| Insert/Delete in middle | O(n) | Requires shifting elements |

**Hash Table Operations:**

| Operation | Average | Worst | Notes |
|-----------|---------|-------|-------|
| Insert | O(1) | O(n) | Worst case with many collisions |
| Delete | O(1) | O(n) | Same as insert |
| Search | O(1) | O(n) | Usually much faster than arrays |
| Iteration | O(n) | O(n) | Must visit all elements |

**Space Complexity:**
- **Array:** O(n) for n elements
- **Hash Table:** O(n) plus overhead for hash structure
- **Trade-off:** Hash tables use more memory but provide faster operations

**When to Use Each:**

**Use Arrays when:**
- Need ordered/indexed access
- Memory is limited
- Elements are accessed by position
- Implementing other data structures

**Use Hash Tables when:**
- Need fast lookups by key
- Counting frequencies
- Detecting duplicates
- Grouping elements
- Caching results

**Optimization Example:**
Finding if array has duplicate:
- Brute force (nested loops): O(n²)
- Sorting then scanning: O(n log n)
- Using hash set: O(n) ✅ Best!`,
    },
    {
      id: 'patterns',
      title: 'Problem-Solving Patterns',
      content: `**Pattern 1: Frequency Counting**

**When:** Need to count occurrences of elements

**Template:**
\`\`\`python
from collections import Counter
freq = Counter(arr)
# Or manually:
freq = {}
for item in arr:
    freq[item] = freq.get(item, 0) + 1
\`\`\`

**Applications:**
- Find most frequent element
- Check if two strings are anagrams
- Find elements appearing k times

**Pattern 2: Complement Lookup (Two Sum)**

**When:** Need to find pairs satisfying a condition

**Template:**
\`\`\`python
seen = {}
for i, num in enumerate(arr):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
\`\`\`

**Applications:**
- Two sum, three sum
- Find pair with difference k
- Check if complement exists

**Pattern 3: Grouping by Key**

**When:** Partition elements by shared property

**Template:**
\`\`\`python
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    key = compute_key(item)
    groups[key].append(item)
return groups.values()
\`\`\`

**Applications:**
- Group anagrams
- Group by sum/difference
- Partition by parity

**Pattern 4: Deduplication**

**When:** Need unique elements or detect duplicates

**Template:**
\`\`\`python
seen = set()
for item in items:
    if item in seen:
        return True  # Found duplicate
    seen.add(item)
return False  # No duplicates
\`\`\`

**Applications:**
- Contains duplicate
- Longest substring without repeating
- Unique elements

**Pattern 5: Index Mapping**

**When:** Need to find indices of elements quickly

**Template:**
\`\`\`python
index_map = {val: i for i, val in enumerate(arr)}
# Look up index by value in O(1)
idx = index_map.get(target, -1)
\`\`\`

**Applications:**
- Two sum returning indices
- Find position of element
- Reverse mapping`,
    },
    {
      id: 'advanced',
      title: 'Advanced Techniques',
      content: `**Technique 1: Multiple Hash Tables**

Use multiple hash tables to track different properties:

\`\`\`python
def find_intersection(arr1, arr2):
    """Find common elements in O(n) time."""
    set1 = set(arr1)
    set2 = set(arr2)
    return list(set1 & set2)  # Set intersection
\`\`\`

**Technique 2: Hash Table as Visited Tracker**

Track what you've seen in O(1) space per element:

\`\`\`python
def has_cycle(arr):
    """Detect cycle using hash set."""
    visited = set()
    current = arr[0]
    
    while current not in visited:
        if current == END:
            return False
        visited.add(current)
        current = next_value(current)
    
    return True
\`\`\`

**Technique 3: Rolling Hash (Rabin-Karp)**

Efficient string matching using hash:

\`\`\`python
def rabin_karp(text, pattern):
    """Find pattern in text using rolling hash."""
    BASE = 256
    MOD = 10**9 + 7
    m, n = len(pattern), len(text)
    
    # Compute hash of pattern
    pattern_hash = 0
    for char in pattern:
        pattern_hash = (pattern_hash * BASE + ord(char)) % MOD
    
    # Rolling hash for text
    text_hash = 0
    for i in range(n):
        # Add new character
        text_hash = (text_hash * BASE + ord(text[i])) % MOD
        
        # Remove old character if window full
        if i >= m:
            text_hash = (text_hash - ord(text[i-m]) * pow(BASE, m, MOD)) % MOD
        
        # Check match
        if i >= m - 1 and text_hash == pattern_hash:
            if text[i-m+1:i+1] == pattern:
                return i - m + 1
    
    return -1
\`\`\`

**Technique 4: Coordinate Compression**

Map large values to small indices:

\`\`\`python
def compress_coordinates(arr):
    """Compress large values to 0, 1, 2, ..."""
    sorted_unique = sorted(set(arr))
    compress = {val: i for i, val in enumerate(sorted_unique)}
    return [compress[x] for x in arr]
\`\`\`

**Technique 5: Hashable Custom Keys**

Use tuples or strings as keys:

\`\`\`python
# For 2D points
seen = set()
seen.add((x, y))

# For lists (convert to tuple)
key = tuple(sorted(lst))
\`\`\``,
    },
    {
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
- "What's the expected size?" (Hash table overhead matters for small inputs)

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
✅ **Iterate over copy: for key in list(dict.keys())**

❌ **Forgetting unhashable types** (lists, dicts can't be keys)
✅ **Convert to tuple: tuple(lst)**

**Follow-Up Questions:**

Q: "What if we can't use extra space?"
A: "I can use two pointers or sorting, but time complexity increases"

Q: "What about hash collisions?"
A: "Python's hash function is robust. In theory O(n) worst case, but O(1) average"

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
    },
  ],
  keyTakeaways: [
    'Arrays provide O(1) indexed access but O(n) search; use binary search for O(log n) on sorted arrays',
    'Hash tables provide O(1) average lookup, insert, delete - essential for optimization',
    'Frequency counting: use Counter or dict to count occurrences in O(n)',
    'Two Sum pattern: store complements in hash table for O(n) solution',
    'Grouping pattern: use defaultdict to partition elements by key',
    'Deduplication: use set for O(n) duplicate detection vs O(n²) brute force',
    'Space-time tradeoff: hash tables use O(n) space but often reduce time from O(n²) to O(n)',
    'Choose dict for key-value, set for membership, Counter for frequencies, defaultdict for auto-initialization',
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  relatedProblems: ['contains-duplicate', 'two-sum', 'group-anagrams'],
};
