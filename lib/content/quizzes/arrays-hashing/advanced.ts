/**
 * Quiz questions for Advanced Techniques section
 */

export const advancedQuiz = [
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
];
