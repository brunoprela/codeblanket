/**
 * LRU Cache Implementation
 * Problem ID: lru-cache-implementation
 * Order: 25
 */

import { Problem } from '../../../types';

export const lru_cache_implementationProblem: Problem = {
  id: 'lru-cache-implementation',
  title: 'LRU Cache Implementation',
  difficulty: 'Medium',
  category: 'python-intermediate',
  description: `Design a data structure that follows the constraints of a **Least Recently Used (LRU) cache**.

Implement the \`LRUCache\` class using \`OrderedDict\`:
- \`LRUCache(int capacity)\`: Initialize with positive capacity
- \`int get(int key)\`: Return value of key if exists, otherwise -1
- \`void put(int key, int value)\`: Update value or add new key-value pair. If cache exceeds capacity, evict the least recently used key.

Both \`get\` and \`put\` must run in O(1) average time.

**Example:**
\`\`\`python
cache = LRUCache(2)
cache.put(1, 1)  # cache: {1=1}
cache.put(2, 2)  # cache: {1=1, 2=2}
cache.get(1)     # returns 1, cache: {2=2, 1=1}
cache.put(3, 3)  # evicts key 2, cache: {1=1, 3=3}
cache.get(2)     # returns -1 (not found)
cache.put(4, 4)  # evicts key 1, cache: {3=3, 4=4}
cache.get(1)     # returns -1 (not found)
cache.get(3)     # returns 3
cache.get(4)     # returns 4
\`\`\``,
  starterCode: `from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        """Initialize LRU cache with given capacity."""
        pass
    
    def get(self, key: int) -> int:
        """Get value for key, return -1 if not found."""
        pass
    
    def put(self, key: int, value: int) -> None:
        """Add or update key-value pair."""
        pass`,
  testCases: [
    {
      input: [
        ['LRUCache', 2],
        ['put', 1, 1],
        ['put', 2, 2],
        ['get', 1],
      ],
      expected: [null, null, null, 1],
    },
    {
      input: [
        ['LRUCache', 2],
        ['put', 1, 1],
        ['put', 2, 2],
        ['put', 3, 3],
        ['get', 2],
      ],
      expected: [null, null, null, null, -1],
    },
    {
      input: [
        ['LRUCache', 2],
        ['put', 1, 1],
        ['put', 2, 2],
        ['get', 1],
        ['put', 3, 3],
        ['get', 2],
      ],
      expected: [null, null, null, 1, null, -1],
    },
  ],
  hints: [
    'OrderedDict maintains insertion order',
    'Use move_to_end() to mark items as recently used',
    'Use popitem(last=False) to remove least recently used',
  ],
  solution: `from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        """Initialize LRU cache with given capacity."""
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: int) -> int:
        """Get value for key, return -1 if not found."""
        if key not in self.cache:
            return -1
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Add or update key-value pair."""
        if key in self.cache:
            # Update existing key and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # If over capacity, remove least recently used (first item)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# Test
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))    # 1
cache.put(3, 3)        # Evicts 2
print(cache.get(2))    # -1
cache.put(4, 4)        # Evicts 1
print(cache.get(1))    # -1
print(cache.get(3))    # 3
print(cache.get(4))    # 4`,
  timeComplexity: 'O(1) for both get and put',
  spaceComplexity: 'O(capacity)',
  order: 25,
  topic: 'Python Intermediate',
};
