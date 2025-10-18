/**
 * LRU Cache
 * Problem ID: lru-cache
 * Order: 1
 */

import { Problem } from '../../../types';

export const lru_cacheProblem: Problem = {
  id: 'lru-cache',
  title: 'LRU Cache',
  difficulty: 'Medium',
  topic: 'Design Problems',
  description: `Design a data structure that follows the constraints of a **Least Recently Used (LRU) cache**.

Implement the \`LRUCache\` class:

- \`LRUCache(int capacity)\` Initialize the LRU cache with **positive** size \`capacity\`.
- \`int get(int key)\` Return the value of the \`key\` if the key exists, otherwise return \`-1\`.
- \`void put(int key, int value)\` Update the value of the \`key\` if the \`key\` exists. Otherwise, add the \`key-value\` pair to the cache. If the number of keys exceeds the \`capacity\` from this operation, **evict** the least recently used key.

The functions \`get\` and \`put\` must each run in **O(1)** average time complexity.`,
  hints: [
    'Need O(1) operations - what data structure gives O(1) access?',
    'HashMap for lookup, but how to track order?',
    'Doubly linked list allows O(1) removal from any position',
    'Combine HashMap (key->node) with Doubly LinkedList (LRU order)',
    'Use dummy head and tail to simplify edge cases',
  ],
  approach: `## Intuition

LRU Cache needs:
1. **O(1) access** to any element by key → **HashMap**
2. **O(1) update** of access order → **Doubly Linked List**

Neither structure alone suffices. **Combined**, they give O(1) for both operations.

---

## Why Doubly Linked List?

When we access a key, we need to move it to front (most recent). This requires:
1. **Remove** node from current position - O(1) with doubly linked list
2. **Insert** at front - O(1)

Singly linked list would need O(N) to find previous node for removal.

---

## Approach

**Data Structures:**
- \`HashMap<key, Node>\`: Fast lookup by key
- \`Doubly LinkedList\`: Maintain LRU order
  - Head: Most recently used
  - Tail: Least recently used

**Operations:**

### get(key):
1. If key not in HashMap → return -1
2. Find node via HashMap (O(1))
3. Move node to front (O(1))
4. Return node.value

### put(key, value):
1. If key exists:
   - Update value
   - Move to front
2. If key is new:
   - Create node
   - Add to HashMap
   - Insert at front
   - If over capacity:
     - Remove tail node (LRU)
     - Delete from HashMap

**Dummy Head/Tail Trick:**
Use dummy nodes to avoid null checks:
\`\`\`
head <-> node1 <-> node2 <-> tail
\`\`\`
Always insert after head, remove before tail.

---

## Time Complexity: O(1) for both get and put
- HashMap operations: O(1)
- LinkedList add/remove with node reference: O(1)

## Space Complexity: O(capacity)
- HashMap: O(capacity)  
- LinkedList: O(capacity)`,
  testCases: [
    {
      input: [
        ['LRUCache', 2],
        ['put', 1, 1],
        ['put', 2, 2],
        ['get', 1],
        ['put', 3, 3],
        ['get', 2],
        ['put', 4, 4],
        ['get', 1],
        ['get', 3],
        ['get', 4],
      ],
      expected: [null, null, null, 1, null, -1, null, -1, 3, 4],
    },
  ],
  solution: `class ListNode:
    """Node in doubly linked list"""
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> ListNode
        
        # Dummy head and tail for easy insertion/deletion
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node: ListNode) -> None:
        """Remove node from linked list - O(1)"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_to_front(self, node: ListNode) -> None:
        """Add node right after head (most recent) - O(1)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key: int) -> int:
        """Get value and mark as recently used"""
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        # Move to front (most recently used)
        self._remove(node)
        self._add_to_front(node)
        return node.val
    
    def put(self, key: int, value: int) -> None:
        """Add/update key-value pair"""
        if key in self.cache:
            # Key exists - update and move to front
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # New key
            new_node = ListNode(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)
            
            # Check capacity
            if len(self.cache) > self.capacity:
                # Evict LRU (node before tail)
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]  # Don't forget HashMap!

# Example usage:
# cache = LRUCache(2)
# cache.put(1, 1)  # cache = {1=1}
# cache.put(2, 2)  # cache = {1=1, 2=2}
# cache.get(1)     # returns 1, cache = {2=2, 1=1}
# cache.put(3, 3)  # evicts key 2, cache = {1=1, 3=3}
# cache.get(2)     # returns -1 (not found)`,
  timeComplexity: 'O(1) for both get() and put()',
  spaceComplexity:
    'O(capacity) - HashMap + LinkedList both store capacity elements',
  patterns: ['HashMap', 'Doubly Linked List', 'Design', 'Cache'],
  companies: [
    'Amazon',
    'Google',
    'Microsoft',
    'Facebook',
    'Apple',
    'Bloomberg',
    'Uber',
    'LinkedIn',
  ],
};
