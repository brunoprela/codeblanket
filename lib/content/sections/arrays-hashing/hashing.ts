/**
 * Hash Tables: Fast Lookups & Storage Section
 */

export const hashingSection = {
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
};
