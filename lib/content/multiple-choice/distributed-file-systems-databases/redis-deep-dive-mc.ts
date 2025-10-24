/**
 * Multiple choice questions for Redis Deep Dive section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const redisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary reason Redis is so fast?',
    options: [
      'Uses GPU acceleration',
      'In-memory storage (no disk I/O for reads/writes)',
      'Distributed across many servers',
      'Uses compression',
    ],
    correctAnswer: 1,
    explanation:
      'Redis is fast primarily because it stores data in memory (RAM), avoiding slow disk I/O. Reads/writes are nanoseconds vs milliseconds for disk. Additionally: (1) Single-threaded (no lock contention), (2) Simple data structures, (3) Efficient protocol. Typical: 100,000+ ops/sec per instance with sub-millisecond latency.',
  },
  {
    id: 'mc2',
    question: 'Which Redis data structure would you use for a leaderboard?',
    options: ['String', 'List', 'Sorted Set (ZSet)', 'Hash'],
    correctAnswer: 2,
    explanation:
      'Sorted Set (ZSet) is perfect for leaderboards. Elements are ordered by score. Operations: ZADD (add/update score), ZRANGE (get by rank), ZREVRANGE (top N), ZRANK (get player rank), ZSCORE (get player score). All O(log N). Example: ZADD leaderboard 1000 player1, ZREVRANGE leaderboard 0 9 (top 10).',
  },
  {
    id: 'mc3',
    question: 'What happens when Redis runs out of memory?',
    options: [
      'Crashes immediately',
      'Automatically increases memory',
      'Uses eviction policy (LRU, LFU, etc.) to remove keys',
      'Stops accepting new connections',
    ],
    correctAnswer: 2,
    explanation:
      'When Redis reaches maxmemory limit, it uses configured eviction policy: (1) noeviction: Return errors on write (default), (2) allkeys-lru: Evict least recently used keys, (3) volatile-lru: Evict LRU keys with expire set, (4) allkeys-lfu: Evict least frequently used. Choose based on use case. Always set maxmemory!',
  },
  {
    id: 'mc4',
    question: 'What is the purpose of Redis pipelining?',
    options: [
      'Encrypt data',
      'Send multiple commands in one network round trip to reduce latency',
      'Compress commands',
      'Backup data',
    ],
    correctAnswer: 1,
    explanation:
      'Pipelining sends multiple commands together without waiting for individual responses, reducing network round trips. Example: Without pipelining (100 commands, 1ms RTT) = 100ms total. With pipelining = 1ms total. Huge performance improvement for batch operations. Essential for high-throughput Redis usage.',
  },
  {
    id: 'mc5',
    question: 'How many hash slots does Redis Cluster use for sharding?',
    options: ['1,024', '4,096', '16,384', '65,536'],
    correctAnswer: 2,
    explanation:
      'Redis Cluster uses 16,384 hash slots. Each key is mapped to a slot via CRC16(key) % 16384. Slots are distributed across nodes. Example: 3 nodes = ~5,461 slots each. This number balances distribution granularity and overhead. Hash tags ({user123}) force keys to same slot.',
  },
];
