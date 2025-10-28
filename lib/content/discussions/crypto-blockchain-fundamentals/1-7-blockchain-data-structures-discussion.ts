export const blockchainDataStructuresDiscussion = [
  {
    id: 1,
    question:
      'Why do Merkle trees use O(log n) space for proofs? Walk through the math.',
    answer:
      'Binary Merkle tree with n leaves has log₂(n) levels. Proof includes one hash per level (sibling hashes). Example: 1024 transactions = log₂(1024) = 10 levels = 10 hashes = 320 bytes proof vs 25KB for all transactions. This enables light clients—verify specific transaction without downloading entire block. Critical for mobile wallets and scalability.',
  },
  {
    id: 2,
    question:
      'Explain why Ethereum uses Patricia tries instead of simple Merkle trees.',
    answer:
      'Patricia trie combines Merkle tree with trie structure for key-value storage. Advantages: (1) Efficient key lookup O(key_length), (2) Proof includes path to value, (3) Natural for account storage (address→balance). Merkle tree alone requires full tree for each lookup. Trade-off: More complex implementation, larger proofs than binary Merkle tree, but better for state storage.',
  },
  {
    id: 3,
    question: 'What are Verkle trees and why is Ethereum moving to them?',
    answer:
      "Verkle trees use vector commitments instead of hash commitments. Advantage: Much smaller proofs—constant size vs O(log n). Example: Proof for 1M accounts: Merkle = 20 hashes (640 bytes), Verkle = 1 proof (~150 bytes). Enables stateless clients—don't need full state, just proofs. Critical for Ethereum scalability. Trade-off: Slower to compute, newer cryptography (less battle-tested).",
  },
];
