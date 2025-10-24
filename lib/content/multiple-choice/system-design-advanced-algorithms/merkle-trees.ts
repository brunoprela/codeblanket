/**
 * Multiple choice questions for Merkle Trees section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const merkletreesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity to find differences between two datasets using Merkle trees?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Merkle trees require O(log N) comparisons to find differences. Start at root, recursively check subtrees where hashes differ. Tree height is log₂(N), so worst case is checking all levels. This is dramatically better than O(N) naive comparison. For 1 million items: 20 comparisons vs 1 million.',
  },
  {
    id: 'mc2',
    question:
      'How many hashes are required in a Merkle proof to verify an element is in a tree with 1 million elements?',
    options: ['1 hash', 'About 20 hashes', '1000 hashes', '1 million hashes'],
    correctAnswer: 1,
    explanation:
      'A Merkle proof requires one hash per level of the tree. For 1 million elements, tree height = log₂(1,000,000) ≈ 20 levels. Proof provides sibling hash at each level (~20 hashes × 32 bytes = 640 bytes). This allows verifying inclusion without revealing entire tree—perfect for Bitcoin SPV clients.',
  },
  {
    id: 'mc3',
    question: 'Which production systems use Merkle trees?',
    options: [
      'Only academic research',
      'Git, Bitcoin, and Apache Cassandra',
      'Only blockchain systems',
      'Only version control systems',
    ],
    correctAnswer: 1,
    explanation:
      'Merkle trees are industry standard: Git (commits are Merkle trees), Bitcoin (transaction verification), Cassandra (anti-entropy repair), DynamoDB (replica sync), IPFS (content addressing), Certificate Transparency. Any system needing efficient verification or sync uses Merkle trees. This is proven production technology.',
  },
  {
    id: 'mc4',
    question: 'What happens to the root hash if a single leaf node changes?',
    options: [
      'Only the leaf hash changes',
      'The root hash and entire tree remains unchanged',
      'The root hash changes because changes propagate up the tree',
      'Only sibling hashes change',
    ],
    correctAnswer: 2,
    explanation:
      'Any leaf change propagates up to the root. Leaf hash changes → parent hash changes → grandparent changes → ... → root changes. This is the fundamental property making Merkle trees tamper-evident. In Git, changing any file changes the entire commit hash. In Bitcoin, changing any transaction changes the block hash.',
  },
  {
    id: 'mc5',
    question:
      'How do Bitcoin SPV (lightweight) clients verify transactions without downloading full blocks?',
    options: [
      'They trust other nodes without verification',
      'They download full blockchain',
      'They use Merkle proofs (log N hashes) to verify against block headers',
      'They cannot verify, only hope',
    ],
    correctAnswer: 2,
    explanation:
      'SPV clients download block headers (80 bytes each, ~80 MB total). To verify a transaction, they request a Merkle proof (~400 bytes) showing the transaction is in the block. They recompute the path to the Merkle root and verify it matches the block header. This provides cryptographic proof with minimal bandwidth (400 bytes vs 1.5 MB full block).',
  },
];
