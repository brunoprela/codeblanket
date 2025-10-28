import { MultipleChoiceQuestion } from '@/lib/types';

export const blockchainDataStructuresMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bds-mc-1',
      question: 'What is the main advantage of Merkle trees in blockchain?',
      options: [
        'Faster hashing than other structures',
        'Enables compact proofs for transaction inclusion',
        'Reduces storage requirements',
        'Speeds up mining',
      ],
      correctAnswer: 1,
      explanation:
        'Merkle trees enable O(log n) size proofs. To prove transaction exists in block of 1000 txs: Merkle proof = ~10 hashes (320 bytes), Alternative = all 1000 txs (250KB+). This enables SPV (Simplified Payment Verification) clients—verify transactions without full blockchain. Critical for mobile wallets. Bitcoin block header includes Merkle root; light clients download headers only, request proofs for relevant transactions.',
    },
    {
      id: 'bds-mc-2',
      question:
        'What does "SPV" (Simplified Payment Verification) mean in Bitcoin?',
      options: [
        'Super-fast payment validation',
        'Verifying payments using Merkle proofs without full blockchain',
        'Simplified private keys',
        'Secure peer validation',
      ],
      correctAnswer: 1,
      explanation:
        "SPV (Bitcoin whitepaper section 8): Light client downloads block headers only (~80 bytes each), not full blocks (~1MB+). To verify transaction: (1) Download block header containing Merkle root, (2) Request Merkle proof from full node, (3) Verify proof links transaction to root. Space savings: Headers = 80B × 800K blocks = 64MB vs Full blockchain = 500GB+. Trade-off: Trusts longest chain (can't validate all transactions), vulnerable to eclipse attacks, but practical for mobile/IoT.",
    },
    {
      id: 'bds-mc-3',
      question:
        'Why do Bloom filters have false positives but never false negatives?',
      options: [
        'They use approximate hashing',
        'They set multiple bits per element; missing any bit means definitely not in set',
        'They compress data lossily',
        'They use probabilistic algorithms',
      ],
      correctAnswer: 1,
      explanation:
        'Bloom filter: Bit array with k hash functions. Insert element: Set k bits to 1. Query element: Check k bits—if ANY bit is 0, definitely not present (no false negatives). If ALL bits are 1, probably present (might be false positive from other elements). Example: Element x sets bits 5,17,89 to 1. Element y sets bits 5,20,89 to 1. Query z hashes to bits 5,17,89—all are 1 (looks present) but might be coincidence from x and y. SPV clients use Bloom filters to request relevant transactions without revealing all addresses.',
    },
    {
      id: 'bds-mc-4',
      question: 'What is a Patricia trie and why does Ethereum use it?',
      options: [
        'A special tree for sorting transactions',
        'A compressed trie (prefix tree) combined with Merkle proofs for key-value storage',
        'A faster alternative to hash tables',
        'A data structure for smart contracts',
      ],
      correctAnswer: 1,
      explanation:
        'Patricia (Practical Algorithm To Retrieve Information Coded In Alphanumeric) trie: Compressed prefix tree where each node represents common prefix. Ethereum Modified Merkle Patricia Trie (MPT): Combines Merkle tree (cryptographic proofs) with Patricia trie (efficient key lookups). Used for: (1) State trie (address→account), (2) Storage trie (contract storage), (3) Transaction trie, (4) Receipt trie. Advantages: O(key_length) lookups, Merkle proofs for any key-value pair, Space-efficient (shares prefixes). Disadvantage: Complex implementation, larger proofs than binary Merkle tree.',
    },
    {
      id: 'bds-mc-5',
      question:
        'Why are Verkle trees considered an improvement over Merkle trees?',
      options: [
        'They are faster to compute',
        'They produce much smaller proofs (constant size vs logarithmic)',
        'They are simpler to implement',
        'They work better with proof-of-work',
      ],
      correctAnswer: 1,
      explanation:
        'Verkle trees use vector commitments (polynomial commitments) instead of hash-based commitments. Key advantage: Proof size constant (~150 bytes) vs Merkle O(log n) (~640 bytes for 1M leaves). Example Ethereum state proof: Merkle = 20+ hashes (3KB+), Verkle = 1 proof (150B). Enables stateless clients—store nothing, receive state + proof per transaction. Critical for Ethereum scalability post-merge. Trade-offs: (1) Slower proof generation (more crypto), (2) Newer cryptography (less proven), (3) Complex implementation. Ethereum researching Verkle trees for future upgrade.',
    },
  ];
