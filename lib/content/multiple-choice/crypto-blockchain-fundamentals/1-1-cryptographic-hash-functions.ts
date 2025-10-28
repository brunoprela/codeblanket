import { MultipleChoiceQuestion } from '@/lib/types';

export const cryptographicHashFunctionsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'chf-mc-1',
      question:
        'What is the primary security property that makes cryptographic hash functions suitable for blockchain applications?',
      options: [
        'They are fast to compute on modern hardware',
        'They produce fixed-size output regardless of input size',
        'They are one-way functions that are computationally infeasible to reverse',
        'They can be computed in parallel across multiple cores',
      ],
      correctAnswer: 2,
      explanation:
        'The one-way (pre-image resistance) property is fundamental to blockchain security. While properties like fast computation and fixed-size output are important, the inability to reverse a hash (finding input from output) is what makes hashes secure for: (1) Hiding private keys in addresses, (2) Preventing transaction tampering, (3) Securing proof-of-work mining. Example: Given Bitcoin address "1A1zP1...", you cannot reverse it to find the private key. Pre-image resistance requires ~2^256 operations to reverse SHA-256, making it computationally infeasible even with all computers on Earth. Without this property, attackers could derive private keys from public addresses or find mining solutions trivially.',
    },
    {
      id: 'chf-mc-2',
      question:
        'Bitcoin uses double SHA-256 (SHA-256(SHA-256(data))) instead of single SHA-256 for transaction IDs and block hashes. What attack does this protect against?',
      options: [
        'Birthday attack / collision resistance',
        'Length extension attack',
        'Brute force pre-image attack',
        "Quantum computer attacks using Grover's algorithm",
      ],
      correctAnswer: 1,
      explanation:
        "Double SHA-256 protects against length extension attacks, a vulnerability in SHA-256's Merkle-Damgård construction. Length extension attack: Given hash(message), attacker can compute hash(message + attacker_data) WITHOUT knowing original message. Example attack if Bitcoin used single SHA-256: An attacker seeing hash(transaction) could append malicious data and compute valid hash(transaction + malicious_data). With double hashing: SHA-256(SHA-256(data)), the output of inner hash becomes input to outer hash, preventing extension since attacker cannot control outer hash input. This is why Bitcoin, and many cryptocurrencies, use double hashing for critical operations. SHA-3 (Keccak) doesn't have this vulnerability due to sponge construction, but Bitcoin conservatively chose double SHA-256.",
    },
    {
      id: 'chf-mc-3',
      question:
        'According to the birthday paradox, approximately how many SHA-256 hashes must be computed to have a 50% probability of finding a collision?',
      options: [
        '2^64 hashes (~18 quintillion)',
        '2^128 hashes (~3.4 × 10^38)',
        '2^192 hashes (~6.3 × 10^57)',
        '2^256 hashes (~1.2 × 10^77)',
      ],
      correctAnswer: 1,
      explanation:
        'Birthday paradox states that for n-bit hash, collision probability reaches 50% after ~2^(n/2) hashes. For SHA-256 (256 bits): 2^(256/2) = 2^128 hashes needed. This is NOT 2^256. Birthday paradox example: With 365 possible birthdays, only 23 people needed for 50% collision chance (not 183). For SHA-256: 2^128 = 340 trillion trillion trillion hashes. At 1 exahash/s (entire Bitcoin network): ~10^22 years. Age of universe: ~10^10 years. Ratio: 10^12 times longer than age of universe. This is why SHA-256 collision resistance is considered secure. Compare to SHA-1 (160 bits): 2^80 hashes, which Google achieved in 2017, demonstrating SHA-1 is broken. Understanding: n-bit hash has 2^n possible outputs, but birthday attack only needs √(2^n) = 2^(n/2) attempts.',
    },
    {
      id: 'chf-mc-4',
      question:
        'What does the "avalanche effect" mean in the context of cryptographic hash functions, and why is it important for blockchain security?',
      options: [
        'The hash function becomes faster as more data is processed',
        'A small change in input causes a large, unpredictable change in output',
        'The security of the hash increases exponentially with additional rounds',
        'Multiple hash collisions cascade through the blockchain',
      ],
      correctAnswer: 1,
      explanation:
        'Avalanche effect: Small input change → ~50% of output bits flip. Example: "Bitcoin 2009" vs "Bitcoin 2008" (one character) produces completely different SHA-256 hashes with ~128 of 256 bits different. Why critical for blockchain: (1) Tamper detection: Changing one bit in a transaction makes block hash completely different, cascading through chain. (2) Mining unpredictability: Minor nonce changes produce unpredictable hashes, preventing mining shortcuts. (3) Transaction privacy: Similar transactions have uncorrelated hashes. Real example: Block #100000 vs #100001 in Bitcoin - tiny data difference, but block hashes are completely unrelated. Without avalanche effect: Attacker could predict hash changes and manipulate mining or forge transactions. Good hash: ~50% bits flip per bit input change. Poor hash: Predictable patterns in output changes. SHA-256 exhibits excellent avalanche effect, making it suitable for blockchain security.',
    },
    {
      id: 'chf-mc-5',
      question:
        'In a Merkle tree with 1,000 transactions, how many hash operations are needed to verify that a specific transaction is included in the block?',
      options: [
        'Approximately 10 hash operations (log₂1000)',
        'Approximately 100 hash operations (√1000)',
        'Exactly 500 hash operations (1000/2)',
        'All 1,000 hash operations (complete tree verification)',
      ],
      correctAnswer: 0,
      explanation:
        'Merkle tree verification requires O(log n) hash operations, where n is number of leaves. For 1,000 transactions: log₂(1000) ≈ 10 hash operations. How Merkle proofs work: (1) Start with transaction hash, (2) Receive sibling hashes at each level, (3) Compute parent hash, (4) Repeat until reaching root, (5) Compare to known root. Example with 8 transactions (3 levels): To prove TX #5 is included, need 3 sibling hashes (one per level) = 3 hash computations. For 1,000 txs: Need ~10 sibling hashes, compute ~10 hashes. This is why Merkle trees are crucial for SPV (Simplified Payment Verification) in Bitcoin—light clients can verify transactions without downloading entire blockchain. Without Merkle trees: Must hash all 1,000 transactions. With Merkle trees: Only ~10 hashes. Efficiency: O(n) → O(log n). Real Bitcoin: Blocks with 2,500 transactions need only ~12 hashes for proof instead of 2,500.',
    },
  ];
