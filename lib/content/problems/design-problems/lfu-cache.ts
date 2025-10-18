/**
 * LFU Cache
 * Problem ID: lfu-cache
 * Order: 7
 */

import { Problem } from '../../../types';

export const lfu_cacheProblem: Problem = {
  id: 'lfu-cache',
  title: 'LFU Cache',
  difficulty: 'Hard',
  topic: 'Design Problems',
  description: `Design and implement a data structure for a **Least Frequently Used (LFU)** cache.

Implement the \`LFUCache\` class:

- \`LFUCache(int capacity)\` Initializes the object with the \`capacity\` of the data structure.
- \`int get(int key)\` Gets the value of the \`key\` if the \`key\` exists in the cache. Otherwise, returns \`-1\`.
- \`void put(int key, int value)\` Update the value of the \`key\` if present, or inserts the \`key\` if not already present. When the cache reaches its \`capacity\`, it should invalidate and remove the **least frequently used** key before inserting a new item. For this problem, when there is a **tie** (i.e., two or more keys with the same frequency), the **least recently used** \`key\` would be invalidated.

To determine the least frequently used key, a **use counter** is maintained for each key in the cache. The key with the smallest **use counter** is the least frequently used key.

When a key is first inserted into the cache, its **use counter** is set to \`1\` (due to the \`put\` operation). The use counter for a key in the cache is incremented either a \`get\` or \`put\` operation is called on it.

The functions \`get\` and \`put\` must each run in **O(1)** average time complexity.`,
  hints: [
    'Track frequency of each key - HashMap: key -> frequency',
    'Group keys by frequency - HashMap: frequency -> list of keys',
    'Within same frequency, LRU order - use doubly linked list',
    'Track minimum frequency for O(1) eviction',
  ],
  approach: `## Intuition

LFU evicts **least frequently used** key. Ties broken by LRU.

Need to track:
1. **Frequency** of each key
2. **Keys at each frequency** (for eviction)
3. **Within frequency, LRU order** (for tie-breaking)

---

## Approach: HashMap + Frequency Buckets

**Data Structures:**

1. \`key_to_val\`: key -> value
2. \`key_to_freq\`: key -> frequency
3. \`freq_to_keys\`: frequency -> OrderedDict of keys (LRU order)
4. \`min_freq\`: Track minimum frequency for O(1) eviction

### Example:

\`\`\`
put(1, 1): freq=1
  key_to_freq = {1: 1}
  freq_to_keys = {1: OrderedDict([1])}
  min_freq = 1

put(2, 2): freq=1
  key_to_freq = {1: 1, 2: 1}
  freq_to_keys = {1: OrderedDict([1, 2])}
  min_freq = 1

get(1): returns 1, frequency → 2
  key_to_freq = {1: 2, 2: 1}
  freq_to_keys = {1: OrderedDict([2]), 2: OrderedDict([1])}
  min_freq = 1

put(3, 3): Cache full, evict key 2 (freq=1, LRU)
  key_to_freq = {1: 2, 3: 1}
  freq_to_keys = {1: OrderedDict([3]), 2: OrderedDict([1])}
\`\`\`

### Operations:

**get(key):**
1. If not exists → return -1
2. Increment frequency
3. Move key from freq to freq+1
4. Update min_freq if needed
5. Return value

**put(key, val):**
1. If capacity is 0 → do nothing
2. If key exists:
   - Update value
   - Increment frequency (same as get)
3. If key is new:
   - If at capacity: evict LFU key (from freq_to_keys[min_freq], first item)
   - Add key with freq=1
   - Set min_freq=1

---

## Time Complexity: O(1) for both get and put
## Space Complexity: O(capacity)`,
  testCases: [
    {
      input: [
        ['LFUCache', 2],
        ['put', 1, 1],
        ['put', 2, 2],
        ['get', 1],
        ['put', 3, 3],
        ['get', 2],
        ['get', 3],
        ['put', 4, 4],
        ['get', 1],
        ['get', 3],
        ['get', 4],
      ],
      expected: [null, null, null, 1, null, -1, 3, null, -1, 3, 4],
    },
  ],
  solution: `from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.key_to_val = {}  # key -> value
        self.key_to_freq = {}  # key -> frequency
        self.freq_to_keys = defaultdict(OrderedDict)  # freq -> OrderedDict of keys
    
    def _update_freq(self, key: int) -> None:
        """Increment frequency of key"""
        freq = self.key_to_freq[key]
        
        # Remove from current frequency bucket
        del self.freq_to_keys[freq][key]
        
        # Update min_freq if this frequency bucket is now empty
        if not self.freq_to_keys[freq] and freq == self.min_freq:
            self.min_freq += 1
        
        # Increment frequency
        self.key_to_freq[key] = freq + 1
        
        # Add to new frequency bucket
        self.freq_to_keys[freq + 1][key] = None  # OrderedDict maintains insertion order
    
    def get(self, key: int) -> int:
        """Get value and increment frequency"""
        if key not in self.key_to_val:
            return -1
        
        self._update_freq(key)
        return self.key_to_val[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair"""
        if self.capacity == 0:
            return
        
        if key in self.key_to_val:
            # Key exists - update value and frequency
            self.key_to_val[key] = value
            self._update_freq(key)
        else:
            # New key
            if len(self.key_to_val) >= self.capacity:
                # Evict LFU key (first key in min_freq bucket)
                lfu_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
                del self.key_to_val[lfu_key]
                del self.key_to_freq[lfu_key]
            
            # Add new key with frequency 1
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.freq_to_keys[1][key] = None
            self.min_freq = 1

# Example usage:
# cache = LFUCache(2)
# cache.put(1, 1)  # freq: {1:1}
# cache.put(2, 2)  # freq: {1:1, 2:1}
# cache.get(1)     # freq: {1:2, 2:1}, returns 1
# cache.put(3, 3)  # Evicts key 2 (freq=1), freq: {1:2, 3:1}`,
  timeComplexity: 'O(1) for both get() and put() operations',
  spaceComplexity: 'O(capacity) for all data structures',
  patterns: ['HashMap', 'Design', 'OrderedDict', 'LFU'],
  companies: ['Amazon', 'Google', 'Facebook', 'Apple'],
};
