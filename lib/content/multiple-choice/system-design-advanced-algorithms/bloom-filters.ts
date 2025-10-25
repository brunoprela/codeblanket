/**
 * Multiple choice questions for Bloom Filters section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bloomfiltersMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the key trade-off that Bloom filters make compared to hash tables?',
    options: [
      'Slower lookups for less memory',
      'False positives for significantly less memory',
      'No insertions for faster lookups',
      'False negatives for faster queries',
    ],
    correctAnswer: 1,
    explanation:
      'Bloom filters trade potential false positives (saying an element might be present when it is not) for massive memory savings. While a hash table might use 100+ bytes per element, a Bloom filter uses only 10-20 bits per element (50-80x less memory). Critically, Bloom filters NEVER produce false negatives.',
  },
  {
    id: 'mc2',
    question:
      'A Bloom filter with 10,000 bits has 7 hash functions. An element "alice" is checked and all 7 corresponding bits are set to 1. What can we conclude?',
    options: [
      '"alice" was definitely inserted into the Bloom filter',
      '"alice" was definitely NOT inserted into the Bloom filter',
      '"alice" might have been inserted (could be false positive)',
      'The Bloom filter is corrupted',
    ],
    correctAnswer: 2,
    explanation:
      'When all k bits are set to 1, we can only say the element MIGHT be present. Those bits could have been set by other elements (false positive). The only definitive answer a Bloom filter gives is when at least one bit is 0 → element was definitely NOT inserted.',
  },
  {
    id: 'mc3',
    question:
      'You want to build a Bloom filter with a 1% false positive rate. Approximately how many bits per element do you need?',
    options: [
      '2-3 bits per element',
      '5-6 bits per element',
      '10 bits per element',
      '50 bits per element',
    ],
    correctAnswer: 2,
    explanation:
      'For a 1% false positive rate (p=0.01), you need approximately 10 bits per element with k≈7 hash functions. Formula: m/n = -ln (p)/(ln(2))² ≈ 9.6 bits/element. Lower FPR requires more bits: 0.1% needs ~15 bits, 0.01% needs ~20 bits.',
  },
  {
    id: 'mc4',
    question:
      'Which of the following is a major limitation of standard Bloom filters?',
    options: [
      'Cannot insert elements',
      'Cannot check for element membership',
      'Cannot delete elements once inserted',
      'Cannot handle more than 1000 elements',
    ],
    correctAnswer: 2,
    explanation:
      'Standard Bloom filters cannot delete elements because multiple elements may set the same bit to 1. Removing an element would require setting bits to 0, which could affect other elements. Solutions: Use counting Bloom filters (counters instead of bits) or cuckoo filters which support deletion.',
  },
  {
    id: 'mc5',
    question: 'When is a Bloom filter the WRONG choice for a system design?',
    options: [
      'Checking if URLs have been crawled (web crawler)',
      'Verifying if a transaction has been processed (payment system)',
      'Filtering duplicate cache entries (CDN)',
      'Checking if IP addresses are blacklisted (firewall, low security)',
    ],
    correctAnswer: 1,
    explanation:
      'Bloom filters are WRONG for payment systems because false positives are unacceptable. A false positive would incorrectly indicate a transaction was already processed, potentially causing financial errors. Use cases requiring 100% accuracy (financial, legal, security-critical) should use exact data structures like hash tables or databases despite the memory cost.',
  },
];
