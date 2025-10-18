/**
 * Design HashMap
 * Problem ID: design-hashmap
 * Order: 29
 */

import { Problem } from '../../../types';

export const design_hashmapProblem: Problem = {
  id: 'design-hashmap',
  title: 'Design HashMap',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  description: `Design a HashMap without using any built-in hash table libraries.

Implement the \`MyHashMap\` class:
- \`MyHashMap()\` initializes the object with an empty map.
- \`void put(int key, int value)\` inserts a (key, value) pair into the HashMap. If the key already exists, update the value.
- \`int get(int key)\` returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
- \`void remove(key)\` removes the key and its corresponding value if the map contains the mapping for the key.`,
  examples: [
    {
      input:
        '["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]',
      output: '[null, null, null, 1, -1, null, 2, null, -1]',
      explanation:
        'MyHashMap myHashMap = new MyHashMap(); myHashMap.put(1, 1); myHashMap.put(2, 2); myHashMap.get(1); returns 1; myHashMap.get(3); returns -1; myHashMap.put(2, 1); myHashMap.get(2); returns 1; myHashMap.remove(2); myHashMap.get(2); returns -1',
    },
  ],
  constraints: [
    '0 <= key, value <= 10^6',
    'At most 10^4 calls will be made to put, get, and remove',
  ],
  hints: [
    'Use an array of buckets',
    'Handle collisions with chaining',
    'Use modulo to map keys to bucket indices',
  ],
  starterCode: `class MyHashMap:
    """
    HashMap implementation using array and chaining.
    """
    
    def __init__(self):
        # Write your code here
        pass
        
    def put(self, key: int, value: int) -> None:
        # Write your code here
        pass
        
    def get(self, key: int) -> int:
        # Write your code here
        pass
        
    def remove(self, key: int) -> None:
        # Write your code here
        pass

# Test code
# hashmap = MyHashMap()
# hashmap.put(1, 1)
# hashmap.put(2, 2)
# print(hashmap.get(1))  # returns 1
# print(hashmap.get(3))  # returns -1
`,
  testCases: [
    {
      input: [
        ['put', 'put', 'get'],
        [[1, 1], [2, 2], [1]],
      ],
      expected: [null, null, 1],
    },
    {
      input: [
        ['put', 'get'],
        [[1, 1], [3]],
      ],
      expected: [null, -1],
    },
  ],
  timeComplexity: 'O(n/k) where k is number of buckets',
  spaceComplexity: 'O(k + m) where m is number of unique keys',
  leetcodeUrl: 'https://leetcode.com/problems/design-hashmap/',
  youtubeUrl: 'https://www.youtube.com/watch?v=cNWsgbKwwoU',
};
