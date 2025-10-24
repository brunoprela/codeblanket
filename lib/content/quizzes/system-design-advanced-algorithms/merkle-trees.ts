/**
 * Quiz questions for Merkle Trees section
 */

export const merkletreesQuiz = [
  {
    id: 'q1',
    question:
      'Explain how Merkle trees achieve O(log N) comparisons for finding differences between two replicas, compared to O(N) for naive comparison. Walk through a concrete example with 8 data blocks.',
    sampleAnswer:
      'NAIVE APPROACH: Compare all N blocks one-by-one. For 8 blocks: 8 comparisons. For 1M blocks: 1M comparisons. MERKLE TREE APPROACH: Build binary tree of hashes. For 8 blocks: Level 0 (leaves): 8 hashes. Level 1: 4 hashes (pairs). Level 2: 2 hashes. Level 3 (root): 1 hash. Height = log₂(8) = 3. FINDING DIFFERENCES: Step 1: Compare root hashes (1 comparison). If same → Done! If different → Step 2. Step 2: Compare Level 2 (2 comparisons for left and right subtrees). Identify which subtree differs. Step 3: Compare Level 1 within differing subtree (1-2 comparisons). Step 4: Identify differing leaf blocks. Total: 3-4 comparisons for 8 blocks (O(log N)). CONCRETE EXAMPLE: Replica A blocks: [b1, b2, b3, b4, b5, b6, b7, b8]. Replica B has b5 corrupted. Root hashes differ → right subtree hash differs → rightmost level 1 hash differs → block 5 identified. Only 3 comparisons! For 1M blocks: log₂(1,000,000) ≈ 20 comparisons vs 1,000,000. This is why Cassandra uses Merkle trees for repair: can find divergent data ranges with logarithmic network roundtrips instead of scanning entire replica.',
    keyPoints: [
      'Naive: O(N) comparisons, check every block',
      'Merkle tree: O(log N) comparisons, binary search through tree',
      'Start at root, recursively check differing subtrees',
      'For 1M blocks: 20 comparisons vs 1,000,000',
      'Used by Cassandra nodetool repair for efficient sync',
    ],
  },
  {
    id: 'q2',
    question:
      'Bitcoin uses Merkle trees for transaction verification in blocks. Explain how lightweight (SPV) clients can verify a transaction is in a block without downloading the entire block.',
    sampleAnswer:
      'BITCOIN BLOCK STRUCTURE: Block header (80 bytes) contains: previous block hash, Merkle root of all transactions, nonce. Full block: 1-2 MB (thousands of transactions). PROBLEM: Mobile clients cannot download full blockchain (hundreds of gigabytes). SOLUTION: Simplified Payment Verification (SPV) via Merkle proofs. HOW IT WORKS: (1) SPV client downloads ALL block headers (~80 MB for entire blockchain). (2) Client wants to verify transaction TX exists in Block 500,000. (3) Client requests Merkle proof for TX from full node. (4) Full node returns: TX data + sibling hashes up the tree (log₂(N) hashes ≈ 10-12 hashes = 320-384 bytes). (5) Client computes: hash(TX), combines with sibling hash, iterates up tree to root. (6) Client compares computed root with Merkle root in Block 500,000 header. (7) If match → TX definitely in block! Total bandwidth: 384 bytes proof vs 1.5 MB full block (4000x reduction). SECURITY: Cryptographic guarantee. Cannot forge proof without breaking SHA-256. This is how billions of crypto wallets work without downloading 500 GB blockchain. REAL NUMBERS: Verify transaction with ~400 bytes vs 1.5 MB. This enables mobile Bitcoin wallets.',
    keyPoints: [
      'SPV clients download block headers only (80 bytes each)',
      'Request Merkle proof for specific transaction (log N hashes ≈ 400 bytes)',
      'Recompute path to root, verify against block header',
      'Bandwidth: 400 bytes vs 1.5 MB full block (4000x reduction)',
      'Enables lightweight mobile crypto wallets',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare Merkle trees to a single hash of all data for detecting differences. When would you use each approach? Provide specific use cases.',
    sampleAnswer:
      'SINGLE HASH APPROACH: hash(all data) = single hash. Comparison: O(1) (compare one hash). If match → identical. If differ → must transfer ALL data to find differences. MERKLE TREE APPROACH: Tree of hashes. Comparison: O(log N). If match → identical. If differ → recursively find differences, transfer only differing blocks. TRADE-OFFS: Single hash: Simple, fast comparison, but cannot identify WHAT differs. Merkle tree: More complex, slower comparison, but identifies EXACTLY what differs. WHEN TO USE SINGLE HASH: (1) Just need "same or different" answer (file integrity check). (2) If different, will transfer all data anyway. (3) Small datasets (overhead not worth it). Example: Download file, verify hash. If wrong, re-download entire file. (4) Blockchain: Block hash is sufficient for linking blocks. WHEN TO USE MERKLE TREES: (1) Need to identify WHAT differs. (2) Large datasets where transferring all data is expensive. (3) Incremental synchronization (databases). (4) Proving inclusion (Bitcoin SPV). Example: Cassandra anti-entropy. Replicas have 1M rows. Root hashes differ → Merkle tree identifies 1000 differing rows → transfer 1000 rows (not 1M). Single hash would require full table scan. CONCRETE: Git uses both. Commit hash (single) for identity. Tree structure (Merkle) for efficient diff. This combination is optimal.',
    keyPoints: [
      'Single hash: O(1) comparison, but cannot identify differences',
      'Merkle tree: O(log N) comparison, identifies exact differences',
      'Single hash: small datasets, integrity checks',
      'Merkle tree: large datasets, incremental sync, proof of inclusion',
      'Git uses both: commit hash (identity) + tree (efficient diff)',
    ],
  },
];
