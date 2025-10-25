/**
 * Multiple choice questions for Caching Systems section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const cachingsystemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of LRU Cache get() operation?',
    options: ['O(N)', 'O(log N)', 'O(1)', 'O(N log N)'],
    correctAnswer: 2,
    explanation:
      'LRU Cache get() is O(1). We use HashMap to find the node in O(1), then use doubly linked list to move it to front in O(1). Both operations are constant time, so overall is O(1).',
  },
  {
    id: 'mc2',
    question: "Why can't we implement LRU Cache with just a HashMap?",
    options: [
      'HashMaps are too slow',
      'HashMaps cannot track the order of access efficiently',
      'HashMaps do not support deletion',
      'HashMaps use too much memory',
    ],
    correctAnswer: 1,
    explanation:
      'HashMap provides O(1) lookup but cannot efficiently track which item was least recently used. We would need to iterate through all items to find the LRU item, making eviction O(N). We need a linked list to maintain access order.',
  },
  {
    id: 'mc3',
    question: 'In LRU Cache, when does an item become "most recently used"?',
    options: [
      'Only when we put() it',
      'Only when we get() it',
      'Both when we get() OR put() it',
      'Only when cache is full',
    ],
    correctAnswer: 2,
    explanation:
      "An item becomes most recently used on BOTH get() and put(). When we get (key), we're accessing it (used recently). When we put (key, val), we're either adding new (definitely recent) or updating existing (also recent). Both operations move the node to front.",
  },
  {
    id: 'mc4',
    question: 'What advantage does LFU have over LRU?',
    options: [
      'LFU is simpler to implement',
      'LFU is always faster',
      'LFU is resistant to cache pollution from sequential scans',
      'LFU uses less memory',
    ],
    correctAnswer: 2,
    explanation:
      'LFU resists cache pollution because it tracks frequency, not recency. A one-time sequential scan of 1000 items won\'t evict frequently-used items in LFU, but would evict everything in LRU (since the scanned items are now "most recent"). LFU is actually more complex and uses more memory.',
  },
  {
    id: 'mc5',
    question:
      'Why do we need to remove a node from HashMap when evicting it from LRU Cache?',
    options: [
      'To save memory',
      'To prevent memory leaks and incorrect lookups',
      'HashMap removal is required for LinkedList removal',
      'To make get() faster',
    ],
    correctAnswer: 1,
    explanation:
      "We must remove from HashMap to prevent: (1) Memory leak - HashMap holds reference to evicted node, preventing garbage collection. (2) Incorrect behavior - future get (key) would find stale node that's not in LinkedList anymore. Always maintain consistency: if node is evicted from list, it must be removed from HashMap.",
  },
];
