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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why arrays and hash tables are considered the foundation of data structures. What makes them so fundamental?',
          sampleAnswer:
            'Arrays are the most basic data structure - just a contiguous block of memory with O(1) index access. They are fundamental because many other data structures like strings, stacks, queues, and even heaps are built on top of arrays. Hash tables are fundamental because they solve a critical problem: fast lookup. With O(1) average-case access, hash tables let us trade space for speed, turning many O(n²) brute force solutions into O(n). Together, they give us the two most common patterns: iterate through data with arrays, and instantly look up information with hash tables. About 30-40% of interview problems use these two structures.',
          keyPoints: [
            'Arrays: basic building block with O(1) index access',
            'Many structures built on arrays',
            'Hash tables: O(1) lookup trades space for speed',
            'Turn O(n²) into O(n) solutions',
            'Appear in 30-40% of interview problems',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through how hash tables enable optimization from O(n²) to O(n). Give me a concrete example.',
          sampleAnswer:
            'Take the two sum problem: given an array, find two numbers that add to a target. Brute force would check every pair with nested loops - O(n²). With a hash table, as I iterate through the array once, for each number I check if target minus that number exists in my hash table. If yes, found it. If no, I add the current number to the hash table and continue. This is one pass through the array with O(1) lookups, so O(n) total. The hash table remembers what I have seen so far, eliminating the need to search back through previous elements. We trade O(n) space for a massive time improvement.',
          keyPoints: [
            'Example: two sum O(n²) brute force',
            'Hash table stores seen elements',
            'Check if complement exists in O(1)',
            'Single pass instead of nested loops',
            'Trade O(n) space for O(n) time',
          ],
        },
        {
          id: 'q3',
          question:
            'Arrays and hash tables each have trade-offs. Talk about when you would choose one over the other.',
          sampleAnswer:
            'I choose arrays when I need ordered data, when I am accessing by position/index, or when memory is extremely tight since arrays have no overhead. Arrays are great for sequential processing and when the size is known. I choose hash tables when I need fast lookups by key rather than by position, when I am checking existence or counting frequencies, or when I need to group or deduplicate data. Hash tables win when lookup speed is critical and I am okay with extra memory overhead. For example, checking if a number exists: array is O(n) scan, hash table is O(1) lookup. The trade-off is always speed versus memory and whether order matters.',
          keyPoints: [
            'Arrays: ordered, index access, low memory overhead',
            'Arrays: when size known, sequential processing',
            'Hash tables: fast key lookup, existence checks',
            'Hash tables: grouping, counting, deduplication',
            'Trade-off: speed vs memory, order vs flexibility',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the prefix sum technique and walk through how it helps solve range query problems efficiently.',
          sampleAnswer:
            'Prefix sum is where I precompute cumulative sums at each position - prefix[i] equals the sum of all elements from 0 to i. This takes O(n) time to build. Then, to find the sum of any range from i to j, I can do prefix[j] minus prefix[i-1] in O(1) time. Without prefix sum, each range query would be O(n) to add up elements. With it, I pay O(n) once to build, then answer unlimited range queries in O(1) each. It is perfect when you have multiple range sum queries on the same array. The insight is that sum(i to j) equals sum(0 to j) minus sum(0 to i-1).',
          keyPoints: [
            'Precompute cumulative sums: O(n) build',
            'Range sum = prefix[j] - prefix[i-1]',
            'Each query: O(1) instead of O(n)',
            'Great for multiple range queries',
            'Trade preprocessing time for query speed',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe Kadane algorithm for maximum subarray. What is the key insight that makes it work in O(n)?',
          sampleAnswer:
            'Kadane algorithm finds the maximum sum subarray in O(n) by making a smart observation: at each position, the maximum subarray ending here is either the current element alone, or the current element plus the maximum subarray ending at the previous position. We track max_current which is the best subarray ending at current position, and max_global which is the best we have seen overall. The key insight is that if max_current becomes negative, we are better off starting fresh from the next element rather than dragging along a negative sum. This way we make one pass and track two values, avoiding checking all possible subarrays which would be O(n²).',
          keyPoints: [
            'Max subarray ending here = current or current + previous max',
            'Track max_current (ending here) and max_global (best overall)',
            'If max_current negative, start fresh',
            'One pass: O(n) instead of O(n²)',
            'Dynamic programming approach',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through the sliding window technique for finding maximum sum of subarray of size k.',
          sampleAnswer:
            'For max sum of size k subarray, I first calculate the sum of the first k elements - that is my initial window. Then I slide the window one position at a time: add the new element entering the window and subtract the element leaving the window. Each slide is just two operations, so O(1) per position. I track the maximum sum seen as I slide through. Total is O(n) to slide through the array. The key is avoiding recalculating the entire window sum each time - I incrementally update it by removing the left element and adding the right element. This turns what would be O(n×k) brute force into O(n).',
          keyPoints: [
            'Calculate first window sum',
            'Slide: add right element, subtract left element',
            'Each slide: O(1), total O(n)',
            'Track maximum as we go',
            'Incremental update vs recalculation',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain how hash tables achieve O(1) average-case lookup. What is a hash function and why does it matter?',
          sampleAnswer:
            'A hash table uses a hash function to convert keys into array indices. The hash function takes your key and outputs a number, which we modulo by the table size to get an index. Good hash functions distribute keys evenly across the array. So when I want to look up a key, I hash it to get the index and go directly there - O(1). The "average case" qualifier is because of collisions - when two keys hash to the same index. Most hash tables handle this with chaining or open addressing. With a good hash function and proper load factor, collisions are rare, so we get O(1) on average. A bad hash function causes many collisions and degrades to O(n) worst case.',
          keyPoints: [
            'Hash function: key → number → array index',
            'Direct index access: O(1)',
            'Good hash function distributes keys evenly',
            'Collisions handled by chaining or probing',
            'Average O(1), worst O(n) with bad function',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through how you would use a hash map to find the first non-repeating character in a string.',
          sampleAnswer:
            'I would make two passes. First pass: iterate through the string and use a hash map to count how many times each character appears. Key is the character, value is the count. This is O(n). Second pass: iterate through the string again (same order) and for each character, check its count in the hash map. The first character with count equals one is my answer. Return it immediately. This is also O(n). Total O(n) time with O(k) space where k is the number of unique characters, at most 26 for lowercase letters. The hash map lets me instantly check counts instead of scanning the string repeatedly for each character.',
          keyPoints: [
            'First pass: count frequency in hash map',
            'Second pass: check counts in original order',
            'Return first character with count 1',
            'Time: O(n), Space: O(k) where k ≤ 26',
            'Avoids repeated scanning',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the group anagrams problem. How do you use hashing to solve it efficiently?',
          sampleAnswer:
            'Group anagrams means grouping words that contain the same letters in different orders, like "eat", "tea", "ate" are all anagrams. I use a hash map where the key represents the signature of the anagram group. The signature could be the sorted letters, so "eat", "tea", "ate" all become "aet" when sorted. Or I could use character counts like "e1t1a1". All anagrams have the same signature. I iterate through words once, compute each word signature, and append the word to the hash map under that signature key. At the end, the hash map values are the grouped anagrams. This is O(n×m log m) if sorting, or O(n×m) with counting, where n is word count and m is max word length.',
          keyPoints: [
            'Anagrams have same letters, different order',
            'Key: sorted letters or character count signature',
            'Hash map: signature → list of words',
            'One pass: compute signature, append to group',
            'Time: O(n×m log m) sorting or O(n×m) counting',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare the time complexity of checking if an element exists: linear search in array vs hash table lookup. Why is there such a difference?',
          sampleAnswer:
            'Linear search in an array is O(n) because in the worst case I might have to check every element until I find it or reach the end. A hash table lookup is O(1) average case because I hash the key to get the index and jump directly there - no scanning needed. The massive difference comes from random access based on a computed index versus sequential checking. This is why hash tables are so powerful for existence checks, frequency counting, and lookups. However, arrays have better cache performance and no hash function overhead, so for very small data sets or when elements are likely near the beginning, arrays can be faster in practice.',
          keyPoints: [
            'Array linear search: O(n) - must scan',
            'Hash table: O(1) - direct index jump',
            'Computed index vs sequential search',
            'Hash tables win for large data',
            'Arrays can be faster for tiny data',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the space-time tradeoff when using hash tables. When is it worth it and when is it not?',
          sampleAnswer:
            'Hash tables trade space for speed. We use O(n) extra memory to store our hash map, but gain O(1) lookups instead of O(n) searches. This turns many O(n²) nested loop solutions into O(n) time. It is absolutely worth it when lookup speed is critical and memory is available - like in coding interviews where optimality matters. It is not worth it when memory is severely constrained, when the dataset is tiny and O(n) is fine, or when you are only doing one lookup (the setup cost is not amortized). Also not worth it if you need to maintain order or need the minimum/maximum element frequently. Always consider: will the speed gain justify the memory cost?',
          keyPoints: [
            'Trade O(n) space for O(1) lookups',
            'Turns O(n²) into O(n) solutions',
            'Worth it: when speed critical, memory available',
            'Not worth it: memory constrained, tiny data, single lookup',
            'Consider if speed gain justifies memory cost',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk through the complexity of removing duplicates: with hash set vs without. What makes the hash set approach faster?',
          sampleAnswer:
            'Without hash set, for each element I would need to check if it has appeared before by scanning previous elements - that is O(n) per element, giving O(n²) total. With a hash set, as I iterate through the array, I check if current element is in the set in O(1), add it to the set if not, and skip it if yes. This is O(n) time total. The hash set remembers what I have seen with instant lookup, eliminating the need to search back through processed elements each time. The cost is O(n) space for the hash set, but the time improvement from O(n²) to O(n) is usually worth it. This is a classic example of trading space for time.',
          keyPoints: [
            'Without hash set: O(n²) - check previous elements each time',
            'With hash set: O(n) - instant membership check',
            'Hash set remembers seen elements',
            'Cost: O(n) space',
            'Classic space-time tradeoff',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Describe the frequency counting pattern. Give me an example problem and walk through the solution.',
          sampleAnswer:
            'Frequency counting is where you use a hash map to count how many times each element appears. For example, finding the most common element in an array. I iterate through the array once, and for each element, I increment its count in the hash map. After one pass, I find the key with the maximum value. This is O(n) time and O(k) space where k is unique elements. It appears in problems like "first unique character", "most frequent element", "valid anagram" where you compare counts. The pattern is: iterate once to build frequency map, then use the map to answer the question. Hash maps make counting instant instead of repeatedly scanning the array.',
          keyPoints: [
            'Hash map: element → count',
            'One pass to build frequency map',
            'Use map to answer question',
            'Time: O(n), Space: O(k)',
            'Examples: unique char, most frequent, anagrams',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the complement pattern used in two sum. How does the hash map store information as you go?',
          sampleAnswer:
            'In two sum with the complement pattern, as I iterate through the array, for each number I calculate what its complement would be (target minus current number). I check if that complement exists in my hash map. If yes, I found the pair. If no, I add the current number and its index to the hash map and continue. The key insight is the hash map stores numbers I have already seen, so when I check for a complement, I am asking "have I seen the other half of this pair already?". This turns checking all pairs O(n²) into a single pass O(n). The hash map accumulates information as we go, remembering past elements for future lookups.',
          keyPoints: [
            'For each number, calculate complement (target - current)',
            'Check if complement in hash map',
            'If yes: found pair. If no: add current to map',
            'Hash map remembers seen elements',
            'Single pass: O(n) instead of O(n²)',
          ],
        },
        {
          id: 'q3',
          question:
            'Talk about the grouping pattern. How do you choose a good key for the hash map?',
          sampleAnswer:
            'The grouping pattern uses a hash map where each key represents a category and the value is a list of items in that category. The critical decision is choosing the right key. For group anagrams, the key is sorted letters or character counts - all anagrams map to the same key. For grouping by sum, the key is the sum value. The key should capture the essential property that defines the group. It must be: hashable (can be a dictionary key), consistent (same input always gives same key), and distinctive (different groups have different keys). Good keys are strings, tuples, or numbers. Lists are not hashable so convert to tuples. A well-chosen key makes the grouping automatic.',
          keyPoints: [
            'Key represents the category/group',
            'Value is list of items in that group',
            'Key must be hashable, consistent, distinctive',
            'Examples: sorted letters for anagrams, sum for grouping by sum',
            'Lists not hashable - use tuples',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'When would you use multiple hash tables instead of just one? Give me a scenario where this is helpful.',
          sampleAnswer:
            'You use multiple hash tables when you need to maintain different relationships or track different aspects of your data simultaneously. For example, in LRU cache, you need one hash map for key-to-node mapping (for O(1) lookup) and a doubly linked list for ordering (for O(1) removal). Or in a problem tracking both letter frequencies and word frequencies, you might use separate maps. Another example is bidirectional mapping - mapping names to IDs and IDs back to names requires two hash tables. The key is when a single map cannot capture all the relationships you need to query efficiently. Each hash table serves a specific purpose.',
          keyPoints: [
            'Different relationships or aspects need different maps',
            'Example: LRU cache (lookup map + ordering structure)',
            'Bidirectional mapping needs two hash tables',
            'Each map serves specific purpose',
            'When single map insufficient for all queries',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain how you would make a custom object hashable so it can be used as a dictionary key. What makes an object hashable?',
          sampleAnswer:
            'An object is hashable if it has a hash value that never changes and can be compared to other objects. In Python, I need to implement hash and equality methods. The hash should be computed from immutable fields only. For example, if I have a Point class with x and y coordinates, I would implement hash to return hash of the tuple (x, y), and implement equality to check if both x and y match. The critical rule: if two objects are equal, they must have the same hash. Immutable objects like tuples and strings are hashable by default. Mutable objects like lists are not - if they change, their hash would change, breaking the hash table.',
          keyPoints: [
            'Must have consistent hash value',
            'Must be comparable for equality',
            'Hash from immutable fields only',
            'Equal objects must have same hash',
            'Mutable objects cannot be hashable',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the rolling hash technique. What problem does it solve and how does it work?',
          sampleAnswer:
            'Rolling hash is used for efficiently comparing substrings, like in pattern matching. Instead of recomputing the hash of each substring from scratch, we "roll" the hash by removing the contribution of the leaving character and adding the contribution of the entering character. This turns substring hashing from O(m) per substring to O(1) per substring. For example, Rabin-Karp algorithm uses rolling hash to find pattern in text in O(n) average case. The hash is typically computed as a polynomial, like hash = c0 × base^0 + c1 × base^1 + ... The rolling update is: remove c0 × base^0, shift everything, add new character. This enables O(1) hash updates.',
          keyPoints: [
            'Efficiently update hash for sliding window of characters',
            'Remove leaving char, add entering char: O(1)',
            'Used in pattern matching (Rabin-Karp)',
            'Hash as polynomial with base',
            'Enables O(n) substring search',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize in an interview that a problem can be solved with hash tables? What are the key signals?',
          sampleAnswer:
            'Several signals tell me to consider hash tables. First, if I am thinking about nested loops to check all pairs, that is a red flag - can I use a hash table to remember elements and look them up instead? Second, keywords like "count", "frequency", "group", "unique", "duplicate", "exists" all scream hash table. Third, if I need fast lookup or existence checking and I am okay using extra memory, hash table is perfect. Fourth, if the problem involves finding complements or pairs that satisfy a relationship, like two sum. The fundamental question: am I repeatedly searching for elements? If yes, hash table probably helps by giving O(1) lookups.',
          keyPoints: [
            'Alternative to nested loops',
            'Keywords: count, frequency, group, unique, duplicate',
            'Fast lookup/existence checking needed',
            'Finding pairs/complements',
            'Repeatedly searching for elements',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your approach to an array/hash table problem in an interview, from reading the problem to explaining your solution.',
          sampleAnswer:
            'First, I would clarify: can the array have duplicates? Any constraints on values? Can it be empty? Then I explain my thinking: "I notice I need to find pairs/count frequencies, so I am thinking hash table for O(1) lookups". I state the approach: one pass to build hash map, then use it to answer. I mention complexity: O(n) time, O(k) space where k is unique elements. I write the code carefully, explaining as I go. After coding, I trace through an example: "for [2, 7, 11, 15] with target 9, first iteration checks for 7 in map, not there, adds 2..." Finally, I mention edge cases I am handling: empty array, no solution, all same values. The key is clear communication throughout.',
          keyPoints: [
            'Clarify: duplicates? constraints? empty?',
            'Explain thinking: why hash table?',
            'State approach and complexity upfront',
            'Code carefully with explanations',
            'Trace through example',
            'Mention edge cases',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes people make with hash table problems? How do you avoid them?',
          sampleAnswer:
            'First mistake: not handling collisions or understanding that hash table operations are average case O(1), not guaranteed. I avoid this by saying "O(1) average case" explicitly. Second: using unhashable keys like lists instead of tuples - I always check what I am using as keys. Third: not considering space complexity - I mention that hash tables use O(n) extra space. Fourth: in problems like two sum, accidentally using the same element twice by not checking indices. Fifth: not handling edge cases like empty input or no solution. I avoid these by being deliberate about what I store, how I query it, and testing edge cases. Communication is key - explain what goes in the hash table and why.',
          keyPoints: [
            'Remember O(1) is average case, not guaranteed',
            'Use hashable keys (tuples, not lists)',
            'Consider O(n) space complexity',
            'Check indices to avoid using same element twice',
            'Test edge cases: empty, no solution',
          ],
        },
      ],
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
