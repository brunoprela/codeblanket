/**
 * Problem-Solving Patterns Section
 */

export const patternsSection = {
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
};
