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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the primary advantage of hash tables over arrays?',
          options: [
            'Hash tables use less memory',
            'Hash tables maintain sorted order',
            'Hash tables provide O(1) average-case lookup',
            'Hash tables can only store unique values',
          ],
          correctAnswer: 2,
          explanation:
            'Hash tables provide O(1) average-case lookup, insert, and delete operations, compared to O(n) for unsorted arrays. This makes them ideal for problems requiring frequent lookups or existence checks.',
        },
        {
          id: 'mc2',
          question:
            'In the two sum problem, why does using a hash table reduce time complexity from O(n²) to O(n)?',
          options: [
            'Hash tables sort the data automatically',
            'Hash tables eliminate the need for a nested loop by providing O(1) lookups',
            'Hash tables reduce the input size',
            'Hash tables only store unique elements',
          ],
          correctAnswer: 1,
          explanation:
            'The brute force approach uses nested loops to check all pairs (O(n²)). With a hash table, we can check if the complement exists in O(1) time, allowing us to solve the problem in a single pass through the array.',
        },
        {
          id: 'mc3',
          question:
            'Which data structure should you use for counting element frequencies?',
          options: ['List', 'Set', 'Counter or Dictionary', 'Tuple'],
          correctAnswer: 2,
          explanation:
            'Counter (from collections) or a regular dictionary is perfect for counting frequencies. Sets only track existence, lists require O(n) search, and tuples are immutable.',
        },
        {
          id: 'mc4',
          question:
            'What percentage of interview problems use arrays or hash tables?',
          options: ['10-20%', '20-30%', '30-40%', '50-60%'],
          correctAnswer: 2,
          explanation:
            'Arrays and hash tables appear in approximately 30-40% of all coding interview problems, making them the most fundamental data structures to master.',
        },
        {
          id: 'mc5',
          question:
            'What is the space complexity when using a hash table to solve the two sum problem?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'We store up to n elements in the hash table in the worst case (when no pair is found or the pair is at the end), resulting in O(n) space complexity.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of accessing an element by index in an array?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
          correctAnswer: 0,
          explanation:
            'Array access by index is O(1) constant time because the memory address can be calculated directly using: base_address + (index × element_size).',
        },
        {
          id: 'mc2',
          question: "What does Kadane's algorithm solve?",
          options: [
            'Maximum element in array',
            'Maximum subarray sum',
            'Minimum subarray sum',
            'Array sorting',
          ],
          correctAnswer: 1,
          explanation:
            "Kadane's algorithm finds the maximum sum of any contiguous subarray in O(n) time by maintaining the maximum sum ending at each position.",
        },
        {
          id: 'mc3',
          question:
            'What is the time complexity of building a prefix sum array?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'Building a prefix sum array requires iterating through the array once to compute cumulative sums, which takes O(n) time. Once built, range queries can be answered in O(1).',
        },
        {
          id: 'mc4',
          question:
            'Why is inserting an element at the beginning of an array O(n) instead of O(1)?',
          options: [
            'The array must be sorted',
            'All existing elements must be shifted to the right',
            'The array must be resized',
            'The element must be hashed first',
          ],
          correctAnswer: 1,
          explanation:
            'Inserting at the beginning requires shifting all n existing elements one position to the right to make room, resulting in O(n) time complexity.',
        },
        {
          id: 'mc5',
          question:
            'In the sliding window technique, how is the window sum updated when moving to the next position?',
          options: [
            'Recalculate the entire sum',
            'Add the new element and subtract the old element',
            'Use binary search',
            'Use a hash table',
          ],
          correctAnswer: 1,
          explanation:
            'The sliding window technique updates the sum in O(1) by adding the element entering the window and subtracting the element leaving the window, avoiding recalculation.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the average-case time complexity for hash table insert, delete, and search operations?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
          correctAnswer: 0,
          explanation:
            'Hash tables provide O(1) average-case complexity for insert, delete, and search operations through direct index computation using a hash function.',
        },
        {
          id: 'mc2',
          question:
            'What causes hash table operations to degrade to O(n) worst case?',
          options: [
            'The table is too small',
            'Many hash collisions occur',
            'The keys are not sorted',
            'The table is empty',
          ],
          correctAnswer: 1,
          explanation:
            'When many keys hash to the same index (collisions), the hash table degenerates to a linked list in that bucket, causing O(n) worst-case lookup time.',
        },
        {
          id: 'mc3',
          question:
            'In Python, which is the best data structure for grouping elements with automatic list initialization?',
          options: ['dict', 'set', 'defaultdict', 'Counter'],
          correctAnswer: 2,
          explanation:
            'defaultdict from collections automatically initializes missing keys with a default value (like an empty list), preventing KeyError and simplifying grouping logic.',
        },
        {
          id: 'mc4',
          question: 'What is the purpose of the load factor in a hash table?',
          options: [
            'To determine the hash function',
            'To determine when to resize the table',
            'To count the number of elements',
            'To sort the elements',
          ],
          correctAnswer: 1,
          explanation:
            'The load factor (typically 0.75) determines when the hash table should resize. When load factor is exceeded, the table resizes to maintain O(1) operations.',
        },
        {
          id: 'mc5',
          question:
            'For finding the first non-repeating character in a string, what is the optimal approach?',
          options: [
            'Sort the string',
            'Use nested loops',
            'Two passes with hash map: count frequencies, then find first with count 1',
            'Use binary search',
          ],
          correctAnswer: 2,
          explanation:
            'Two passes with a hash map is optimal: first pass counts character frequencies in O(n), second pass checks counts in order to find first character with count 1, total O(n) time.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of checking if an element exists in an unsorted array?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'Checking existence in an unsorted array requires linear search, which is O(n) since we may need to check every element. A hash set would reduce this to O(1).',
        },
        {
          id: 'mc2',
          question:
            'When comparing arrays vs hash tables for removing duplicates, what is the time complexity difference?',
          options: [
            'Both are O(n)',
            'Array: O(n), Hash table: O(n²)',
            'Array: O(n²), Hash table: O(n)',
            'Array: O(n log n), Hash table: O(log n)',
          ],
          correctAnswer: 2,
          explanation:
            'Without a hash table, checking if each element is a duplicate requires scanning previous elements (O(n²)). With a hash set, we get O(1) lookups for O(n) total time.',
        },
        {
          id: 'mc3',
          question:
            'What is the space complexity overhead difference between arrays and hash tables?',
          options: [
            'Hash tables use less space',
            'They use the same space',
            'Hash tables use more space due to hash structure overhead',
            'Arrays use more space',
          ],
          correctAnswer: 2,
          explanation:
            'Hash tables require additional memory for the hash structure, pointers, and maintaining load factor. Arrays store elements contiguously with minimal overhead.',
        },
        {
          id: 'mc4',
          question:
            'For a problem requiring both fast lookups and maintaining order, which combination is best?',
          options: [
            'Just use an array',
            'Just use a hash table',
            'Use both: hash table for lookups, array for order',
            'Use a set',
          ],
          correctAnswer: 2,
          explanation:
            'When you need both fast lookups (O(1)) and ordered access, use a hash table for lookups and maintain a separate array or list for the ordered elements.',
        },
        {
          id: 'mc5',
          question:
            'When should you prefer sorting an array over using a hash table?',
          options: [
            'When you need O(1) lookups',
            'When memory is extremely limited and O(n log n) time is acceptable',
            'When counting frequencies',
            'When finding duplicates quickly',
          ],
          correctAnswer: 1,
          explanation:
            'Sorting (O(n log n) time, O(1) space with in-place sort) is preferable when memory is severely constrained and the slower time complexity is acceptable.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the frequency counting pattern best used for?',
          options: [
            'Sorting elements',
            'Counting occurrences of elements',
            'Binary search',
            'Reversing arrays',
          ],
          correctAnswer: 1,
          explanation:
            'The frequency counting pattern uses a hash map to count how many times each element appears, useful for problems like finding most frequent element or checking anagrams.',
        },
        {
          id: 'mc2',
          question:
            'In the complement lookup pattern (two sum), what do you store in the hash table?',
          options: [
            'The target value',
            'All possible sums',
            'Elements seen so far (with their indices)',
            'The largest element',
          ],
          correctAnswer: 2,
          explanation:
            "The complement pattern stores elements you've seen so far in the hash table. For each new element, you check if its complement (target - current) exists in the table.",
        },
        {
          id: 'mc3',
          question:
            'When grouping anagrams, what should be used as the hash map key?',
          options: [
            'The first word in each group',
            'The sorted characters or character count signature',
            'The length of the word',
            'Random numbers',
          ],
          correctAnswer: 1,
          explanation:
            'All anagrams share the same sorted characters (e.g., "eat", "tea", "ate" → "aet") or character counts, making it a perfect key to group them together.',
        },
        {
          id: 'mc4',
          question:
            'What is the time complexity of the deduplication pattern using a hash set?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'The deduplication pattern iterates through elements once, checking each against the hash set in O(1), resulting in O(n) total time instead of O(n²) with nested loops.',
        },
        {
          id: 'mc5',
          question: 'Why is defaultdict useful for the grouping pattern?',
          options: [
            'It sorts keys automatically',
            'It automatically initializes missing keys with a default value',
            'It uses less memory',
            'It is faster than regular dict',
          ],
          correctAnswer: 1,
          explanation:
            'defaultdict automatically creates a default value (like an empty list) for missing keys, eliminating the need to check if a key exists before appending to it.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'When would you use multiple hash tables instead of one?',
          options: [
            'To save memory',
            'When tracking different relationships or bidirectional mappings',
            'To make the code simpler',
            'Never, one is always enough',
          ],
          correctAnswer: 1,
          explanation:
            'Multiple hash tables are useful when tracking different relationships (like LRU cache needing both key→node and ordering) or bidirectional mappings (name→ID and ID→name).',
        },
        {
          id: 'mc2',
          question: 'What makes an object hashable in Python?',
          options: [
            'It must be a string',
            'It must be immutable and have a consistent hash value',
            'It must be a number',
            'It must be small',
          ],
          correctAnswer: 1,
          explanation:
            'Hashable objects must be immutable and provide a consistent hash value. Mutable objects like lists cannot be hashable because their hash would change when modified.',
        },
        {
          id: 'mc3',
          question: 'What problem does the rolling hash technique solve?',
          options: [
            'Sorting strings',
            'Finding duplicates',
            'Efficiently computing hash values for sliding window of characters',
            'Reversing arrays',
          ],
          correctAnswer: 2,
          explanation:
            'Rolling hash (used in Rabin-Karp) efficiently updates the hash value when sliding a window by removing the leaving character and adding the entering character in O(1).',
        },
        {
          id: 'mc4',
          question: 'What is coordinate compression used for?',
          options: [
            'Reducing file sizes',
            'Mapping large values to small sequential indices',
            'Hashing passwords',
            'Sorting arrays',
          ],
          correctAnswer: 1,
          explanation:
            'Coordinate compression maps a set of potentially large or sparse values to a small dense range (0, 1, 2, ...), useful when values are large but their relative order matters.',
        },
        {
          id: 'mc5',
          question: 'Why are tuples used as hash keys instead of lists?',
          options: [
            'Tuples are faster',
            'Tuples are immutable and therefore hashable',
            'Tuples use less memory',
            'Lists are deprecated',
          ],
          correctAnswer: 1,
          explanation:
            'Tuples are immutable, making them hashable and safe to use as dictionary keys. Lists are mutable and would break hash table invariants if their contents changed.',
        },
      ],
    },
    {
      id: 'two-sum-patterns',
      title: 'Two-Sum Patterns Family',
      content: `**Two-Sum patterns** are the **#1 most asked** pattern family in technical interviews. Mastering these patterns is essential for interview success.

---

## Why Two-Sum Matters

**Interview Frequency:**
- Asked at: Amazon (very frequently), Facebook, Google, Microsoft, Apple
- Appears in ~15-20% of all array/hash table interviews
- Foundation for more complex problems (3Sum, 4Sum, subset sum)
- Tests fundamental hash table optimization thinking

**What Makes It Important:**
- Teaches hash table vs. brute force trade-off
- Demonstrates space-time complexity analysis
- Shows pattern recognition across variations
- Gateway to understanding pair/complement problems

---

## Pattern 1: Two Sum (Hash Table Approach)

**Problem:** Find two numbers in an array that add up to a target.

**Brute Force:** O(n²) - check all pairs  
**Optimized:** O(n) - use hash table to store complements

**Key Insight:** As you iterate, check if \`target - current\` was seen before.

**Implementation:**
\`\`\`python
def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Find indices of two numbers that add up to target.
    
    Time: O(n) - single pass
    Space: O(n) - hash table
    
    Key: Store value → index mapping, look for complement
    """
    seen = {}  # value → index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []  # No solution found
\`\`\`

**Why Hash Table:**
- **Before:** Check if complement exists → O(n) linear search
- **After:** Check if complement exists → O(1) hash lookup
- Trade O(n) space for O(n) time (down from O(n²))

**Example:**
\`\`\`
nums = [2, 7, 11, 15], target = 9

Iteration 1: num=2, complement=7
  - 7 not in seen
  - Add 2 → index 0
  - seen = {2: 0}

Iteration 2: num=7, complement=2
  - 2 IS in seen (index 0)
  - Return [0, 1]
\`\`\`

---

## Pattern 2: Two Sum II - Sorted Array

**Problem:** Same as Two Sum, but array is **sorted**.

**New Approach:** Two pointers (O(1) space!)

**Key Insight:** Sorting enables two-pointer technique without extra space.

**Implementation:**
\`\`\`python
def two_sum_sorted(nums: List[int], target: int) -> List[int]:
    """
    Find indices in sorted array that add up to target.
    
    Time: O(n) - two pointers converge
    Space: O(1) - no extra space needed
    
    Key: Use sorted property to move pointers intelligently
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return []
\`\`\`

**Why Two Pointers:**
- If sum too small → increase left (makes sum bigger)
- If sum too large → decrease right (makes sum smaller)
- Converge toward answer without checking all pairs

**Comparison:**
| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| Hash Table | O(n) | O(n) | Unsorted array |
| Two Pointers | O(n) | O(1) | Sorted array |
| Sort + Two Pointers | O(n log n) | O(1) | Space constrained |

---

## Pattern 3: 3Sum (Extension to Three Numbers)

**Problem:** Find all **unique triplets** that sum to zero.

**Challenges:**
1. **Three elements** instead of two
2. Need **all unique triplets** (no duplicates)
3. Time limit: must be better than O(n³)

**Approach:** Sort + Two Pointers (O(n²))

**Key Insight:** Fix one element, then do Two Sum II on the rest.

**Implementation:**
\`\`\`python
def three_sum(nums: List[int]) -> List[List[int]]:
    """
    Find all unique triplets that sum to zero.
    
    Time: O(n²) - n iterations × n two-pointer search
    Space: O(1) - excluding output
    
    Strategy:
    1. Sort array: O(n log n)
    2. Fix first element
    3. Two-pointer search for remaining two
    4. Skip duplicates to ensure uniqueness
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicate values for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        # Two-pointer search for remaining two elements
        left, right = i + 1, n - 1
        target = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for left pointer
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for right pointer
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result
\`\`\`

**Example:**
\`\`\`
nums = [-1, 0, 1, 2, -1, -4]
After sort: [-4, -1, -1, 0, 1, 2]

i=0: nums[0]=-4, target=4
  Two-pointer search: no pairs sum to 4

i=1: nums[1]=-1, target=1
  left=2, right=5: nums[2]=-1, nums[5]=2, sum=1 ✓
  Result: [[-1, -1, 2]]
  left=3, right=4: nums[3]=0, nums[4]=1, sum=1 ✓
  Result: [[-1, -1, 2], [-1, 0, 1]]

i=2: Skip (nums[2] == nums[1])

Final: [[-1, -1, 2], [-1, 0, 1]]
\`\`\`

**Duplicate Handling:**
- Skip duplicate first elements: \`if i > 0 and nums[i] == nums[i - 1]: continue\`
- Skip duplicate left/right pointers after finding a match
- This ensures uniqueness without using a set

---

## Pattern 4: 4Sum (Further Extension)

**Problem:** Find all unique quadruplets that sum to a target.

**Approach:** Sort + Two nested loops + Two Pointers (O(n³))

**Key Insight:** Fix two elements, then do Two Sum II on the rest.

**Implementation Sketch:**
\`\`\`python
def four_sum(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all unique quadruplets that sum to target.
    
    Time: O(n³) - n² pairs × n two-pointer search
    Space: O(1) - excluding output
    
    Strategy:
    1. Sort array
    2. Fix first two elements (nested loops)
    3. Two-pointer search for remaining two
    4. Skip duplicates
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 3):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        for j in range(i + 1, n - 2):
            # Skip duplicates for second element
            if j > i + 1 and nums[j] == nums[j-1]:
                continue
            
            # Two-pointer for remaining elements
            left, right = j + 1, n - 1
            remaining = target - nums[i] - nums[j]
            
            while left < right:
                current_sum = nums[left] + nums[right]
                
                if current_sum == remaining:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    # Skip duplicates...
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current_sum < remaining:
                    left += 1
                else:
                    right -= 1
    
    return result
\`\`\`

---

## Pattern Recognition Guide

**When you see:**
| Keyword | Pattern | Approach |
|---------|---------|----------|
| "Find two numbers that sum to..." | Two Sum | Hash table O(n) |
| "Sorted array, two sum" | Two Sum II | Two pointers O(1) space |
| "Three numbers sum to zero/target" | 3Sum | Sort + fix one + two pointers |
| "Four numbers sum to target" | 4Sum | Sort + fix two + two pointers |
| "K numbers sum to target" | K-Sum | Recursively reduce to 2Sum |
| "Count pairs with sum" | Two Sum variant | Hash table counting |
| "Find all pairs" | Two Sum all | Hash table with list |

---

## Complexity Analysis

| Problem | Best Time | Space | Approach |
|---------|-----------|-------|----------|
| **Two Sum (unsorted)** | O(n) | O(n) | Hash table |
| **Two Sum (sorted)** | O(n) | O(1) | Two pointers |
| **3Sum** | O(n²) | O(1)* | Sort + fixed first + two pointers |
| **4Sum** | O(n³) | O(1)* | Sort + fixed two + two pointers |
| **K-Sum** | O(n^(k-1)) | O(1)* | Generalize to k elements |

*excluding output space

**Pattern Complexity Growth:**
- 2Sum: O(n) with hash table
- 3Sum: O(n²) with sort + two pointers
- 4Sum: O(n³) with sort + two pointers
- K-Sum: O(n^(k-1)) in general

---

## Interview Strategy

**Recognition:**
- "Sum to target" → Think Two-Sum family
- "Pairs" → Two Sum
- "Triplets" → 3Sum  
- "Unique" → Need to skip duplicates

**Approach Selection:**
1. **Unsorted, two elements** → Hash table
2. **Sorted, two elements** → Two pointers
3. **Three+ elements** → Sort + fix first(s) + two pointers
4. **Space constrained** → Sort first, use two pointers

**Communication:**
\`\`\`
"I recognize this as a Two-Sum variant.

For two elements in unsorted array:
- Brute force: O(n²) checking all pairs
- Optimized: O(n) with hash table for complements

For three elements:
- Sort first: O(n log n)
- Fix first element, two-pointer for rest: O(n²)
- Skip duplicates to ensure uniqueness

Time: O(n²), Space: O(1) excluding output"
\`\`\`

**Common Mistakes:**
- ❌ Forgetting to skip duplicates in 3Sum/4Sum
- ❌ Using same element twice (check indices)
- ❌ Not considering sorted vs unsorted
- ❌ Using hash table when two pointers is more efficient

**Practice Progression:**
1. Master Two Sum (hash table)
2. Practice Two Sum II (two pointers)
3. Tackle 3Sum (combining techniques)
4. Attempt 4Sum (harder but same pattern)
5. Try variations (closest sum, count pairs, etc.)`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why we use a hash table for Two Sum instead of nested loops. Walk through the optimization and why it works.',
          sampleAnswer:
            'The brute force approach checks all pairs with nested loops: for each element, check every other element to see if they sum to target. This is O(n²) because we check n elements against n-1 others. The hash table optimization works by storing complements: as we iterate once through the array, we calculate complement = target - current, then check if that complement was already seen in O(1) time using a hash table. If yes, we found our pair. If no, we store current for future lookups. This reduces time from O(n²) to O(n) by trading O(n) space for O(1) lookups. The key insight: instead of searching for complement each time (O(n)), we remember all previous elements and look them up instantly (O(1)).',
          keyPoints: [
            'Brute force: nested loops check all pairs O(n²)',
            'Hash table stores seen elements for O(1) lookup',
            'Check if complement exists instead of searching',
            'Single pass through array: O(n) time',
            'Trade O(n) space for O(n²) → O(n) time improvement',
          ],
        },
        {
          id: 'q2',
          question:
            'For 3Sum, why do we sort first? Could we use a hash table like in Two Sum? Explain the trade-offs.',
          sampleAnswer:
            'We sort for 3Sum because it enables two key optimizations: 1) We can use two-pointer technique which gives O(n²) total time (n iterations × n two-pointer search), and 2) Sorting makes duplicate handling easy - we can skip consecutive duplicates to ensure unique triplets without needing a set. Could we use hash table? Yes, but it is harder: we would need nested loops to fix two elements, then hash lookup for the third - still O(n²) time but more complex duplicate handling, and O(n) extra space. The sorted approach is cleaner: O(n log n) sort + O(n²) search, versus O(n²) with hash table but messier code. The sort time does not dominate since O(n²) is larger. The two-pointer technique on sorted array is the standard approach because it is elegant, space-efficient, and handles duplicates naturally.',
          keyPoints: [
            'Sorting enables two-pointer technique: O(n²) total',
            'Easy duplicate handling with sorted array',
            'Hash table possible but more complex',
            'Sort time O(n log n) does not dominate O(n²)',
            'Two-pointer on sorted array is standard, cleanest approach',
          ],
        },
        {
          id: 'q3',
          question:
            'In Two Sum II (sorted array), how do you decide whether to move the left or right pointer? Why does this guarantee we find the answer?',
          sampleAnswer:
            'The decision rule is: if current_sum < target, move left pointer right (increase sum); if current_sum > target, move right pointer left (decrease sum). This works because the array is sorted. Moving left pointer right means we pick a larger number (since sorted), increasing the sum. Moving right pointer left means we pick a smaller number, decreasing the sum. This guarantees finding the answer because: 1) If the answer exists and current sum is too small, increasing left is the only way to potentially reach target (right cannot go further right without skipping). 2) If current sum is too large, decreasing right is needed. 3) The pointers converge, checking all viable pairs exactly once. We cannot miss the answer because for any pair (i, j), we either check it directly or eliminate one of its elements by proving it cannot be part of the solution.',
          keyPoints: [
            'Sorted property: left pointer = smaller values, right = larger',
            'Sum too small → move left right (increase)',
            'Sum too large → move right left (decrease)',
            'Pointers converge, checking all viable pairs once',
            'Cannot miss answer: every pair considered or eliminated',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the time complexity of Two Sum using a hash table?',
          options: ['O(n²)', 'O(n log n)', 'O(n)', 'O(1)'],
          correctAnswer: 2,
          explanation:
            'Two Sum with hash table is O(n) time - single pass through array with O(1) hash lookups for each element. We check and insert each element once.',
        },
        {
          id: 'mc2',
          question:
            'For Two Sum II (sorted array), what is the space complexity using two pointers?',
          options: ['O(n)', 'O(log n)', 'O(1)', 'O(n²)'],
          correctAnswer: 2,
          explanation:
            'Two pointers approach uses O(1) space - only two pointer variables. We leverage the sorted property instead of extra data structures.',
        },
        {
          id: 'mc3',
          question:
            'What is the time complexity of 3Sum using sort + two pointers?',
          options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
          correctAnswer: 2,
          explanation:
            '3Sum is O(n²): O(n log n) for sorting + O(n²) for fix-one-element (n times) × two-pointer-search (n). The n² dominates, so overall O(n²).',
        },
        {
          id: 'mc4',
          question: 'Why must we skip duplicates in 3Sum?',
          options: [
            'To improve performance',
            'To ensure unique triplets in the result',
            'To reduce space usage',
            'To handle negative numbers',
          ],
          correctAnswer: 1,
          explanation:
            'We skip duplicates to ensure unique triplets. Without skipping, we would find the same triplet multiple times. For example, [-1,-1,2] might be found twice if we do not skip the second -1.',
        },
        {
          id: 'mc5',
          question: 'For 4Sum, what is the time complexity?',
          options: ['O(n²)', 'O(n³)', 'O(n⁴)', 'O(n log n)'],
          correctAnswer: 1,
          explanation:
            '4Sum using sort + nested loops + two pointers is O(n³): fix two elements (n²) × two-pointer search (n) = O(n³).',
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
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What keyword in a problem statement strongly suggests using a hash table?',
          options: [
            'Sorted',
            'Binary',
            'Count, frequency, or group',
            'Recursive',
          ],
          correctAnswer: 2,
          explanation:
            'Keywords like "count", "frequency", "group", "unique", "duplicate", or "exists" strongly indicate that hash tables will be useful for O(1) lookups and tracking.',
        },
        {
          id: 'mc2',
          question:
            'In an interview, when should you mention the brute force approach?',
          options: [
            'Never mention it, go straight to optimal',
            'Only if asked',
            'Always state it first, then explain why you can optimize',
            'Only for easy problems',
          ],
          correctAnswer: 2,
          explanation:
            "Always state the brute force approach first (even if just briefly) to show you understand the problem, then explain why it's not optimal and how you can improve it.",
        },
        {
          id: 'mc3',
          question:
            'What is a common mistake when implementing two sum with a hash table?',
          options: [
            'Using O(n) space',
            'Using the same element twice by not checking indices',
            'Iterating through the array',
            'Using a dictionary',
          ],
          correctAnswer: 1,
          explanation:
            'A common mistake is using the same array element twice. You must ensure the complement you find is at a different index than the current element.',
        },
        {
          id: 'mc4',
          question:
            'How should you respond if asked "What about hash collisions?"',
          options: [
            'Say hash collisions never happen',
            "Explain that Python's hash function is robust, average case is O(1) but worst case is O(n)",
            'Say you would use a different data structure',
            'Say you would sort the data instead',
          ],
          correctAnswer: 1,
          explanation:
            "Acknowledge that collisions exist and affect worst-case complexity (O(n)), but Python's hash function is well-designed for average-case O(1) performance.",
        },
        {
          id: 'mc5',
          question:
            'What is the typical time range for solving a medium hash table problem in an interview?',
          options: [
            '5-10 minutes',
            '15-20 minutes',
            '25-30 minutes',
            '45-60 minutes',
          ],
          correctAnswer: 1,
          explanation:
            'Medium hash table problems typically take 15-20 minutes to solve in an interview, including clarification, explanation, coding, and testing phases.',
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
