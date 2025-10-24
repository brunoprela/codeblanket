/**
 * System Design: Advanced Algorithms & Data Structures Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { bloomfiltersSection } from '../sections/system-design-advanced-algorithms/bloom-filters';
import { consistenthashingSection } from '../sections/system-design-advanced-algorithms/consistent-hashing';
import { quorumconsensusSection } from '../sections/system-design-advanced-algorithms/quorum-consensus';
import { vectorclocksSection } from '../sections/system-design-advanced-algorithms/vector-clocks-version-vectors';
import { merkletreesSection } from '../sections/system-design-advanced-algorithms/merkle-trees';
import { hyperloglogSection } from '../sections/system-design-advanced-algorithms/hyperloglog';
import { geospatialindexesSection } from '../sections/system-design-advanced-algorithms/geospatial-indexes';
import { ratelimitingalgorithmsSection } from '../sections/system-design-advanced-algorithms/rate-limiting-algorithms';

// Import quizzes
import { bloomfiltersQuiz } from '../quizzes/system-design-advanced-algorithms/bloom-filters';
import { consistenthashingQuiz } from '../quizzes/system-design-advanced-algorithms/consistent-hashing';
import { quorumconsensusQuiz } from '../quizzes/system-design-advanced-algorithms/quorum-consensus';
import { vectorclocksQuiz } from '../quizzes/system-design-advanced-algorithms/vector-clocks-version-vectors';
import { merkletreesQuiz } from '../quizzes/system-design-advanced-algorithms/merkle-trees';
import { hyperloglogQuiz } from '../quizzes/system-design-advanced-algorithms/hyperloglog';
import { geospatialindexesQuiz } from '../quizzes/system-design-advanced-algorithms/geospatial-indexes';
import { ratelimitingalgorithmsQuiz } from '../quizzes/system-design-advanced-algorithms/rate-limiting-algorithms';

// Import multiple choice
import { bloomfiltersMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/bloom-filters';
import { consistenthashingMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/consistent-hashing';
import { quorumconsensusMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/quorum-consensus';
import { vectorclocksMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/vector-clocks-version-vectors';
import { merkletreesMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/merkle-trees';
import { hyperloglogMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/hyperloglog';
import { geospatialindexesMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/geospatial-indexes';
import { ratelimitingalgorithmsMultipleChoice } from '../multiple-choice/system-design-advanced-algorithms/rate-limiting-algorithms';

export const systemDesignAdvancedAlgorithmsModule: Module = {
  id: 'system-design-advanced-algorithms',
  title: 'System Design: Advanced Algorithms & Data Structures',
  description:
    'Master specialized algorithms and data structures essential for distributed systems: Bloom filters, consistent hashing, quorum consensus, vector clocks, Merkle trees, HyperLogLog, geospatial indexes, and rate limiting algorithms',
  category: 'System Design',
  difficulty: 'Advanced',
  estimatedTime: '6-8 hours',
  prerequisites: ['system-design-fundamentals'],
  icon: '🧮',
  keyTakeaways: [
    'Bloom filters: Space-efficient probabilistic data structure for membership testing (1% error, 1000x memory savings)',
    'Consistent hashing: Minimal key remapping (1/N) when adding/removing servers using hash ring and virtual nodes',
    'Quorum consensus: W + R > N guarantees strong consistency in replicated systems (Cassandra, DynamoDB)',
    'Vector clocks: Track causality in distributed systems without physical time, detect concurrent writes',
    'Merkle trees: Efficient data verification and synchronization using cryptographic hash trees (O(log N) comparisons)',
    'HyperLogLog: Count billions of unique elements in 16 KB with ~1% error (Redis PFCOUNT, BigQuery)',
    'Geospatial indexes: Efficient proximity queries using Quadtrees, R-Trees, and Geohash (Uber, MongoDB)',
    'Rate limiting algorithms: Token bucket (bursts), Sliding window (smooth), Leaky bucket (traffic shaping)',
    'Production adoption: These are not academic—every major system uses these algorithms',
    'Trade-offs: All algorithms trade accuracy/consistency/simplicity for massive scale improvements',
    'Bloom filters enable 75%+ disk read reduction in BigTable/Cassandra via pre-filtering',
    'Consistent hashing is foundation for DynamoDB, Cassandra partitioning—industry standard',
    'HyperLogLog enables analytics at scale: Facebook DAU, Google Analytics unique visitors',
    'Understanding when to use each algorithm is key to system design interviews',
  ],
  learningObjectives: [
    'Understand when and why to use probabilistic data structures vs exact solutions',
    'Implement and explain Bloom filters for duplicate detection and cache optimization',
    'Design distributed systems using consistent hashing for scalable data partitioning',
    'Apply quorum consensus to balance consistency and availability in replicated systems',
    'Use vector clocks to detect conflicts and track causality in multi-master replication',
    'Implement Merkle trees for efficient data synchronization across replicas',
    'Apply HyperLogLog for memory-efficient cardinality estimation at massive scale',
    'Choose appropriate geospatial index (Quadtree, R-Tree, Geohash) for proximity queries',
    'Implement production-grade rate limiting using token bucket and sliding window algorithms',
    'Analyze trade-offs: memory vs accuracy, consistency vs availability, simplicity vs optimality',
    'Recognize real-world usage: Google (Merkle trees, HyperLogLog), Amazon (consistent hashing, quorum)',
    'Design systems that scale to billions of users using these foundational algorithms',
  ],
  sections: [
    {
      ...bloomfiltersSection,
      quiz: bloomfiltersQuiz,
      multipleChoice: bloomfiltersMultipleChoice,
    },
    {
      ...consistenthashingSection,
      quiz: consistenthashingQuiz,
      multipleChoice: consistenthashingMultipleChoice,
    },
    {
      ...quorumconsensusSection,
      quiz: quorumconsensusQuiz,
      multipleChoice: quorumconsensusMultipleChoice,
    },
    {
      ...vectorclocksSection,
      quiz: vectorclocksQuiz,
      multipleChoice: vectorclocksMultipleChoice,
    },
    {
      ...merkletreesSection,
      quiz: merkletreesQuiz,
      multipleChoice: merkletreesMultipleChoice,
    },
    {
      ...hyperloglogSection,
      quiz: hyperloglogQuiz,
      multipleChoice: hyperloglogMultipleChoice,
    },
    {
      ...geospatialindexesSection,
      quiz: geospatialindexesQuiz,
      multipleChoice: geospatialindexesMultipleChoice,
    },
    {
      ...ratelimitingalgorithmsSection,
      quiz: ratelimitingalgorithmsQuiz,
      multipleChoice: ratelimitingalgorithmsMultipleChoice,
    },
  ],
};
