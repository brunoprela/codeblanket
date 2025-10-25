/**
 * Merkle Trees Section
 */

export const merkletreesSection = {
  id: 'merkle-trees',
  title: 'Merkle Trees',
  content: `Merkle trees (also called hash trees) are a fundamental data structure that allows efficient and secure verification of large data sets. They answer a critical question: **"How do we quickly verify that two replicas have the same data without transferring all the data?"**

## The Data Synchronization Problem

Imagine two database replicas that should contain the same 1 million records:

\`\`\`
Replica A: 1,000,000 records
Replica B: 1,000,000 records (supposedly identical)

Question: Are they actually identical?
\`\`\`

### Naive Approaches

**Option 1: Transfer all data**
\`\`\`
Send all 1M records from A to B → Compare
\`\`\`
- ❌ Massive bandwidth (gigabytes)
- ❌ Very slow

**Option 2: Hash all data**
\`\`\`
hash_A = hash (all records on A)
hash_B = hash (all records on B)
if hash_A == hash_B: identical
\`\`\`
- ✅ Fast comparison (compare single hash)
- ❌ If different, must transfer ALL data to find differences
- ❌ Cannot identify WHICH records differ

**Merkle tree solution**: Efficiently identify exactly which data differs with logarithmic comparisons.

---

## What is a Merkle Tree?

A **Merkle tree** is a binary tree of cryptographic hashes where:
- **Leaf nodes**: Hashes of individual data blocks
- **Internal nodes**: Hashes of their children's hashes
- **Root node**: Single hash representing entire dataset

\`\`\`
                    Root Hash
                   /          \\
              Hash(A,B)      Hash(C,D)
              /     \\        /      \\
           Hash(A) Hash(B) Hash(C) Hash(D)
             |       |       |       |
           Data A  Data B  Data C  Data D
\`\`\`

**Key property**: Any change to data propagates up to root hash.

---

## How Merkle Trees Work

### Building a Merkle Tree

\`\`\`python
import hashlib

def hash_data (data):
    return hashlib.sha256(data.encode()).hexdigest()

# Step 1: Hash leaf nodes (data blocks)
data = ["Data A", "Data B", "Data C", "Data D"]
leaf_hashes = [hash_data (d) for d in data]

# Step 2: Build tree bottom-up
def build_merkle_tree (hashes):
    if len (hashes) == 1:
        return hashes[0]  # Root
    
    next_level = []
    for i in range(0, len (hashes), 2):
        if i + 1 < len (hashes):
            # Combine two hashes
            parent_hash = hash_data (hashes[i] + hashes[i+1])
        else:
            # Odd number: promote single hash
            parent_hash = hashes[i]
        next_level.append (parent_hash)
    
    return build_merkle_tree (next_level)

root_hash = build_merkle_tree (leaf_hashes)
\`\`\`

### Complete Example

\`\`\`
Data:
Block 0: "Alice"   → hash: a1b2c3...
Block 1: "Bob"     → hash: d4e5f6...
Block 2: "Charlie" → hash: g7h8i9...
Block 3: "Dave"    → hash: j0k1l2...

Level 1 (leaves):
[a1b2c3, d4e5f6, g7h8i9, j0k1l2]

Level 2 (combine pairs):
hash (a1b2c3 + d4e5f6) = m3n4o5
hash (g7h8i9 + j0k1l2) = p6q7r8

Level 3 (root):
hash (m3n4o5 + p6q7r8) = ROOT_HASH

Tree structure:
               ROOT_HASH
              /          \\
          m3n4o5        p6q7r8
          /    \\        /    \\
      a1b2c3  d4e5f6  g7h8i9  j0k1l2
        |       |       |       |
      Alice   Bob   Charlie  Dave
\`\`\`

---

## Efficient Data Verification

### Scenario: Replica Synchronization

Replica A and B both have 8 data blocks:

\`\`\`
Step 1: Compare root hashes
  A_root: abc123...
  B_root: abc123...
  
  Identical? → Done! (1 comparison)

Step 2: If different, compare children
  A_left:  def456
  B_left:  def456  ✓ (match)
  A_right: ghi789
  B_right: xyz999  ✗ (differ!)
  
  Difference is in right subtree

Step 3: Recursively check right subtree
  Eventually identify exact differing blocks
\`\`\`

**Efficiency**:
- **Without Merkle tree**: Compare all N blocks → O(N) comparisons
- **With Merkle tree**: Compare log₂(N) levels → O(log N) comparisons

For 1 million blocks:
- Naive: 1,000,000 comparisons
- Merkle tree: ~20 comparisons (log₂(1,000,000) ≈ 20)

**Bandwidth**:
- Only transfer differing blocks (not all data)
- Identify differences with logarithmic network roundtrips

---

## Implementation

\`\`\`python
class MerkleTree:
    def __init__(self, data_blocks):
        self.leaves = [self._hash (block) for block in data_blocks]
        self.tree = self._build_tree (self.leaves)
        self.root = self.tree[0] if self.tree else None
    
    def _hash (self, data):
        """SHA-256 hash"""
        return hashlib.sha256(str (data).encode()).hexdigest()
    
    def _build_tree (self, hashes):
        """Build complete Merkle tree"""
        if not hashes:
            return []
        
        tree = [hashes]
        while len (tree[-1]) > 1:
            level = tree[-1]
            next_level = []
            
            for i in range(0, len (level), 2):
                if i + 1 < len (level):
                    # Combine two hashes
                    combined = self._hash (level[i] + level[i+1])
                else:
                    # Odd number: duplicate last hash
                    combined = level[i]
                next_level.append (combined)
            
            tree.append (next_level)
        
        return tree
    
    def get_root (self):
        """Get root hash"""
        return self.root
    
    def get_proof (self, index):
        """Get Merkle proof for data at index"""
        if index < 0 or index >= len (self.leaves):
            return None
        
        proof = []
        for level in self.tree[:-1]:  # Exclude root
            sibling_index = index ^ 1  # Flip last bit (if even→odd, odd→even)
            if sibling_index < len (level):
                proof.append({
                    'hash': level[sibling_index],
                    'position': 'right' if index % 2 == 0 else 'left'
                })
            index //= 2  # Move to parent
        
        return proof
    
    def verify_proof (self, data, index, proof, root_hash):
        """Verify Merkle proof"""
        current_hash = self._hash (data)
        
        for node in proof:
            if node['position'] == 'right':
                current_hash = self._hash (current_hash + node['hash'])
            else:
                current_hash = self._hash (node['hash'] + current_hash)
        
        return current_hash == root_hash

# Usage
data = ["Alice", "Bob", "Charlie", "Dave"]
tree = MerkleTree (data)

print(f"Root hash: {tree.get_root()}")

# Prove "Bob" is in tree
proof = tree.get_proof(1)  # Index 1 = "Bob"
is_valid = tree.verify_proof("Bob", 1, proof, tree.get_root())
print(f"Proof valid: {is_valid}")  # True
\`\`\`

---

## Merkle Proofs

A **Merkle proof** proves a specific element is in the tree **without revealing entire tree**.

### Example: Prove "Bob" is in tree

\`\`\`
Tree:
               ROOT
              /    \\
           H(A,B)  H(C,D)
           /  \\    /   \\
         H(A) H(B) H(C) H(D)
          |    |    |    |
         Alice Bob Charlie Dave

To prove "Bob" is in tree, provide:
1. Hash of sibling: H(A)
2. Hash of uncle: H(C,D)

Verification:
1. Compute H(Bob) = H(B)
2. Combine with H(A) → H(A,B)
3. Combine with H(C,D) → ROOT
4. Check if computed ROOT matches known ROOT
\`\`\`

**Proof size**: O(log N) hashes (tree height)

For 1 million elements:
- Proof size: ~20 hashes × 32 bytes = 640 bytes
- Much smaller than transferring all data!

---

## Real-World Use Cases

### 1. Git (Version Control)

**Commits are Merkle trees**:
\`\`\`
Commit hash = hash (tree hash + parent commit + metadata)
Tree hash = hash (file hashes)

Any file change → tree hash changes → commit hash changes
\`\`\`

**Benefits**:
- Tamper-evident: Cannot change history without changing commit hash
- Efficient diff: Compare tree hashes to find changes
- Fast clone: Verify data integrity with root hash

\`\`\`bash
git log --oneline
abc1234 (HEAD) Latest commit
def5678 Previous commit

# Each commit hash is root of Merkle tree
\`\`\`

### 2. Bitcoin & Blockchain

**Block structure**:
\`\`\`
Block:
  - Previous block hash
  - Merkle root (all transactions)
  - Nonce

Block hash = hash (previous hash + merkle root + nonce)
\`\`\`

**Benefits**:
- Lightweight clients: Download block headers only (80 bytes)
- Verify transaction: Request Merkle proof (~1 KB vs 1 MB full block)
- Tamper-proof: Changing any transaction changes block hash

**Simplified Payment Verification (SPV)**:
\`\`\`
1. Download block headers (megabytes, not gigabytes)
2. Request Merkle proof for your transaction
3. Verify proof against block header
4. Confirmed without full blockchain!
\`\`\`

### 3. Apache Cassandra

**Anti-entropy (data repair)**:
\`\`\`
Problem: Replicas drift out of sync over time

Solution: Merkle trees per table
1. Each replica builds Merkle tree of its data
2. Compare root hashes
3. If different, recursively find differing ranges
4. Repair only differing ranges

Efficiency: Identify discrepancies without scanning all data
\`\`\`

**Nodetool repair**:
\`\`\`bash
nodetool repair keyspace_name

# Uses Merkle trees to sync replicas efficiently
\`\`\`

### 4. Amazon DynamoDB

**Replica synchronization**:
- Merkle trees to identify out-of-sync data
- Efficient repair between replicas
- Scales to billions of items

### 5. IPFS (InterPlanetary File System)

**Content-addressable storage**:
\`\`\`
File split into blocks → Each block hashed → Merkle DAG

File hash = Merkle root
Request file by hash → Get blocks with Merkle proofs
\`\`\`

**Benefits**:
- Deduplication: Identical blocks share same hash
- Integrity: Verify each block with Merkle proof
- Distributed: Download blocks from multiple peers

### 6. Certificate Transparency (Google)

**Public log of SSL certificates**:
\`\`\`
All certificates stored in append-only Merkle tree
Root hash published publicly

Browsers verify: Certificate in log?
Merkle proof confirms inclusion
\`\`\`

**Prevents**: Rogue certificate authorities issuing fake certificates.

---

## Merkle Trees vs Alternatives

| Approach | Comparisons | Bandwidth | Identifies Differences |
|----------|-------------|-----------|------------------------|
| **Full comparison** | O(N) | O(N) | Yes |
| **Single hash** | O(1) | O(N) if differs | No |
| **Merkle tree** | O(log N) | O(k log N) | Yes (k = differences) |

**Merkle trees are optimal for**:
- Large datasets (millions of items)
- Frequent synchronization
- Minimal bandwidth
- Untrusted environments (cryptographic verification)

---

## Advanced Variations

### Merkle DAG (Directed Acyclic Graph)

**Used by**: Git, IPFS

**Difference**: Nodes can have multiple parents (not just binary tree)

\`\`\`
Git:
  Merge commit has TWO parents
  Files can be shared across commits (deduplication)
\`\`\`

### Prolly Trees (Probabilistic Merkle Trees)

**Used by**: Noms, Dolt (version-controlled databases)

**Key innovation**: Boundaries determined by content (rolling hash)
- Insert element → Local changes only (not global rebalancing)
- Better for databases with frequent updates

### Merkle Patricia Tries

**Used by**: Ethereum

**Combines**: Merkle tree + Patricia trie (compact prefix tree)
- Efficient key-value storage
- Cryptographic proofs
- Space-efficient for sparse data

---

## Implementation Considerations

### Handling Odd Number of Nodes

**Option 1: Duplicate last node**
\`\`\`
Data: [A, B, C]
Level 1: [H(A), H(B), H(C)]
Level 2: [H(H(A) + H(B)), H(H(C) + H(C))]
\`\`\`

**Option 2: Promote single node**
\`\`\`
Level 2: [H(H(A) + H(B)), H(C)]
\`\`\`

Bitcoin uses option 1, most systems use option 2.

### Choosing Block Size

\`\`\`
Small blocks (1 KB):
  ✅ Granular sync (transfer less data)
  ❌ Deeper tree (more hash computations)

Large blocks (1 MB):
  ✅ Shallower tree (faster)
  ❌ Transfer large blocks even for small changes

Sweet spot: 4-64 KB blocks
\`\`\`

### Hash Function Choice

**Options**:
- SHA-256: Secure, industry standard (Bitcoin, Git)
- SHA-512: More secure, slower
- Blake2: Faster than SHA-256, equally secure
- MD5: INSECURE, deprecated

**Recommendation**: SHA-256 or Blake2

---

## Interview Tips

### Key Talking Points

1. **Problem**: Efficiently verify and sync large datasets
2. **Structure**: Binary tree of hashes (leaves = data, root = entire dataset)
3. **Efficiency**: O(log N) comparisons vs O(N) naive
4. **Proof**: Verify inclusion with O(log N) hashes
5. **Real-world**: Git, Bitcoin, Cassandra, DynamoDB, IPFS

### Common Interview Questions

**"How do Merkle trees make synchronization efficient?"**
- Compare root hashes first (1 comparison)
- If different, recursively check subtrees
- Only transfer differing blocks
- O(log N) comparisons to find differences

**"Explain a Merkle proof"**
- Prove element in tree without revealing entire tree
- Provide sibling hashes at each level (log N hashes)
- Recompute path to root
- Verify computed root matches known root

**"Why does Git use Merkle trees?"**
- Tamper-evident: Any history change changes commit hash
- Efficient diff: Compare tree hashes
- Content-addressable: Same content = same hash
- Distributed: Verify integrity without trusted server

**"What are limitations?"**
- Initial tree construction: O(N) time
- Updates: Recompute hashes up to root (O(log N))
- Not suitable for frequently-updated data (consider Prolly trees)

### Whiteboard Exercise

Design replica synchronization for distributed database:

\`\`\`
1. Each replica maintains Merkle tree of its data
   - Leaf nodes: Hash of each row (or range of rows)
   - Tree depth: log₂(N) where N = number of ranges

2. Periodic sync:
   - Compare root hashes
   - If different, request subtree hashes
   - Recursively narrow down to differing ranges
   - Transfer only differing data

3. For 1M rows:
   - Divide into 1024 ranges (~1000 rows each)
   - Tree depth: log₂(1024) = 10
   - Worst case: 10 roundtrips to find differences
   - Transfer only out-of-sync ranges

4. Complexity:
   - Comparisons: O(log N)
   - Bandwidth: O(k * row_size) where k = differing ranges
   - Much better than O(N) full sync
\`\`\`

---

## Summary

**Merkle trees** are a fundamental data structure for efficient verification and synchronization of large datasets.

**Key principles**:
- ✅ Binary tree of cryptographic hashes
- ✅ Root hash represents entire dataset
- ✅ O(log N) comparisons to find differences
- ✅ Merkle proofs verify inclusion efficiently
- ✅ Tamper-evident (any change changes root hash)

**Industry adoption**: Git (commits), Bitcoin (transactions), Cassandra (anti-entropy), DynamoDB (sync), IPFS (content addressing), Certificate Transparency

**Perfect for**:
- Large datasets requiring verification
- Distributed systems with replicas
- Untrusted environments (blockchain, P2P)
- Efficient incremental synchronization

Understanding Merkle trees is **essential** for distributed systems, blockchain, and any system requiring data integrity verification.`,
};
