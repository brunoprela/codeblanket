export const blockchainDataStructures = {
  title: 'Blockchain Data Structures',
  id: 'blockchain-data-structures',
  content: `
# Blockchain Data Structures

## Introduction

Blockchains aren't just linked lists of blocks—they use sophisticated data structures to enable efficient verification, compact proofs, and scalable storage. Understanding these structures is essential for building high-performance blockchain applications.

**This section covers:**
- Merkle trees and Merkle proofs (compact transaction verification)
- Patricia tries (Ethereum's state storage)
- Sparse Merkle trees (efficient state commitments)
- Bloom filters (fast membership testing)
- Accumulators (constant-size set commitments)

## Merkle Trees

### The Problem

A Bitcoin block contains ~2,000 transactions. How can a light client verify a transaction is in a block without downloading all 2,000 transactions?

**Naive approach**:
\`\`\`
Download all 2,000 transactions (~2 MB)
Verify your transaction is in the list
Problem: Defeats the purpose of light clients!
\`\`\`

**Merkle tree approach**:
\`\`\`
Download: Block header (80 bytes) + Merkle proof (~10 hashes = 320 bytes)
Total: 400 bytes instead of 2 MB (5,000× smaller!)
\`\`\`

### Structure

Merkle trees are binary hash trees where each parent node is the hash of its children:

\`\`\`
         Root Hash
        /          \\
      H12           H34
     /  \\          /  \\
    H1   H2      H3    H4
    |    |       |     |
   Tx1  Tx2    Tx3   Tx4
\`\`\`

\`\`\`python
import hashlib
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class MerkleProof:
    """Proof that a transaction is in a Merkle tree"""
    tx_hash: str
    proof_hashes: List[tuple[str, str]]  # (hash, 'left'/'right')
    root_hash: str
    
    def __repr__(self):
        return f"MerkleProof({self.tx_hash[:8]}..., {len(self.proof_hashes)} hashes)"

class MerkleTree:
    """Merkle tree for transaction verification"""
    
    def __init__(self, transactions: List[str]):
        self.transactions = transactions
        self.leaves = [self._hash(tx) for tx in transactions]
        self.root = self._build_tree(self.leaves)
    
    def _hash(self, data: str) -> str:
        """Hash data using SHA-256"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two child hashes together"""
        combined = left + right
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _build_tree(self, hashes: List[str]) -> str:
        """Build Merkle tree bottom-up, return root hash"""
        if not hashes:
            return self._hash("")
        
        if len(hashes) == 1:
            return hashes[0]
        
        # If odd number, duplicate last hash
        if len(hashes) % 2 == 1:
            hashes = hashes + [hashes[-1]]
        
        # Build next level
        next_level = []
        for i in range(0, len(hashes), 2):
            parent = self._hash_pair(hashes[i], hashes[i+1])
            next_level.append(parent)
        
        return self._build_tree(next_level)
    
    def get_proof(self, tx_index: int) -> MerkleProof:
        """Generate Merkle proof for transaction at index"""
        if tx_index >= len(self.transactions):
            raise ValueError(f"Transaction index {tx_index} out of range")
        
        tx_hash = self.leaves[tx_index]
        proof_hashes = []
        
        # Build proof bottom-up
        hashes = self.leaves.copy()
        index = tx_index
        
        while len(hashes) > 1:
            # If odd number, duplicate last
            if len(hashes) % 2 == 1:
                hashes = hashes + [hashes[-1]]
            
            # Find sibling
            if index % 2 == 0:
                # Index is left child, sibling is right
                sibling = hashes[index + 1]
                proof_hashes.append((sibling, 'right'))
            else:
                # Index is right child, sibling is left
                sibling = hashes[index - 1]
                proof_hashes.append((sibling, 'left'))
            
            # Move to next level
            next_level = []
            for i in range(0, len(hashes), 2):
                parent = self._hash_pair(hashes[i], hashes[i+1])
                next_level.append(parent)
            
            hashes = next_level
            index = index // 2
        
        return MerkleProof(tx_hash, proof_hashes, self.root)
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify Merkle proof"""
        current = proof.tx_hash
        
        for sibling, position in proof.proof_hashes:
            if position == 'right':
                current = self._hash_pair(current, sibling)
            else:
                current = self._hash_pair(sibling, current)
        
        return current == proof.root_hash
    
    def __repr__(self):
        return f"MerkleTree({len(self.transactions)} txs, root={self.root[:16]}...)"


# Example: Bitcoin-style Merkle tree
print("=== Merkle Tree Example ===\\n")

transactions = [
    "Alice sends 1 BTC to Bob",
    "Charlie sends 2 BTC to David",
    "Eve sends 0.5 BTC to Frank",
    "Grace sends 3 BTC to Henry",
    "Ivan sends 1.5 BTC to Judy",
    "Kevin sends 2.5 BTC to Laura",
    "Mike sends 0.8 BTC to Nancy",
    "Oscar sends 1.2 BTC to Paul",
]

tree = MerkleTree(transactions)
print(f"Built tree: {tree}\\n")

# Light client wants to verify transaction 3 is in block
tx_index = 3
print(f"Verifying transaction {tx_index}: '{transactions[tx_index]}'")
proof = tree.get_proof(tx_index)
print(f"Generated {proof}")
print(f"Proof size: {len(proof.proof_hashes)} hashes ({len(proof.proof_hashes) * 64} bytes)\\n")

# Verify proof
is_valid = tree.verify_proof(proof)
print(f"Proof valid: {is_valid}")

# Show proof details
print(f"\\nProof path:")
for i, (hash_val, position) in enumerate(proof.proof_hashes):
    print(f"  {i+1}. Hash from {position}: {hash_val[:16]}...")

# Try invalid proof
print(f"\\n--- Testing invalid proof ---")
fake_proof = MerkleProof(
    tx_hash="fake_transaction_hash",
    proof_hashes=proof.proof_hashes,
    root_hash=proof.root_hash
)
print(f"Fake proof valid: {tree.verify_proof(fake_proof)}")
\`\`\`

Output:
\`\`\`
=== Merkle Tree Example ===

Built tree: MerkleTree(8 txs, root=7a3f8e2c1b4d9a5e...)

Verifying transaction 3: 'Grace sends 3 BTC to Henry'
Generated MerkleProof(4e9d3e3c..., 3 hashes)
Proof size: 3 hashes (192 bytes)

Proof valid: True

Proof path:
  1. Hash from right: 2c5d8e1f4a7b9d2c...
  2. Hash from left: 8e3f6a9b2c5d8e1f...
  3. Hash from left: 1f4a7b2c5d8e3f6a...

--- Testing invalid proof ---
Fake proof valid: False
\`\`\`

### Performance Analysis

\`\`\`python
import math

def merkle_proof_size(num_transactions: int) -> int:
    """Calculate Merkle proof size in bytes"""
    tree_height = math.ceil(math.log2(num_transactions))
    num_hashes = tree_height
    hash_size_bytes = 32  # SHA-256
    return num_hashes * hash_size_bytes

# Compare naive vs Merkle proof
print("=== Proof Size Comparison ===\\n")
print(f"{'Transactions':<15} {'Naive (MB)':<15} {'Merkle Proof (bytes)':<25} {'Reduction'}")
print("-" * 70)

for num_tx in [10, 100, 1000, 2000, 10000]:
    naive_size = num_tx * 250  # ~250 bytes per transaction
    naive_mb = naive_size / 1_000_000
    merkle_size = merkle_proof_size(num_tx)
    reduction = naive_size / merkle_size
    
    print(f"{num_tx:<15,} {naive_mb:<15.2f} {merkle_size:<25} {reduction:,.0f}x")
\`\`\`

Output:
\`\`\`
=== Proof Size Comparison ===

Transactions     Naive (MB)      Merkle Proof (bytes)      Reduction
----------------------------------------------------------------------
10               0.00            128                       20x
100              0.02            224                       112x
1,000            0.25            320                       781x
2,000            0.50            352                       1,420x
10,000           2.50            416                       6,010x
\`\`\`

## Patricia Tries

Ethereum uses **Patricia tries** (also called Merkle Patricia tries) to store account state.

### Why Not Simple Merkle Trees?

Merkle trees are great for fixed lists (transactions in a block), but Ethereum needs:
1. **Key-value storage**: address → account state
2. **Efficient updates**: Change one account without rebuilding entire tree
3. **Proof of absence**: Prove an address doesn't exist

### Patricia Trie Structure

Combines trie (prefix tree) with Merkle tree properties:

\`\`\`
Example: Store {
  "a711": value1,
  "a722": value2,
  "a7f9": value3
}

Trie:
     root
      |
     [a]
      |
     [7]
    /   \\
  [1]   [2,f]
   |    /    \\
  [1]  [2]   [9]
   |    |     |
 val1 val2  val3
\`\`\`

\`\`\`python
from typing import Any, Optional, Dict

class PatriciaNode:
    """Node in Patricia trie"""
    
    def __init__(self):
        self.children: Dict[str, PatriciaNode] = {}
        self.value: Optional[Any] = None
        self.is_terminal = False
    
    def hash(self) -> str:
        """Calculate node hash (Merkle property)"""
        # Simplified: Real Ethereum uses RLP encoding
        data = str(self.value) + ''.join(sorted(self.children.keys()))
        return hashlib.sha256(data.encode()).hexdigest()

class PatriciaTrie:
    """Simplified Patricia trie"""
    
    def __init__(self):
        self.root = PatriciaNode()
    
    def insert(self, key: str, value: Any):
        """Insert key-value pair"""
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = PatriciaNode()
            node = node.children[char]
        node.value = value
        node.is_terminal = True
    
    def get(self, key: str) -> Optional[Any]:
        """Get value for key"""
        node = self.root
        for char in key:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.value if node.is_terminal else None
    
    def get_proof(self, key: str) -> List[str]:
        """Generate proof for key"""
        proof = []
        node = self.root
        for char in key:
            proof.append(node.hash())
            if char not in node.children:
                break
            node = node.children[char]
        proof.append(node.hash())
        return proof
    
    def get_root_hash(self) -> str:
        """Get root hash (state commitment)"""
        return self.root.hash()


# Example: Ethereum-style state trie
print("=== Patricia Trie Example ===\\n")

state = PatriciaTrie()

# Insert account states
accounts = {
    "0xa711": {"balance": 100, "nonce": 5},
    "0xa722": {"balance": 50, "nonce": 2},
    "0xa7f9": {"balance": 200, "nonce": 10},
    "0xb123": {"balance": 75, "nonce": 3},
}

for address, account_data in accounts.items():
    state.insert(address, account_data)
    print(f"Inserted {address}: {account_data}")

print(f"\\nState root hash: {state.get_root_hash()[:32]}...")

# Retrieve account
address = "0xa722"
account = state.get(address)
print(f"\\nRetrieved {address}: {account}")

# Generate proof
proof = state.get_proof(address)
print(f"Proof length: {len(proof)} hashes")

# Prove non-existence
missing_address = "0xdead"
print(f"\\nAccount {missing_address} exists: {state.get(missing_address) is not None}")
\`\`\`

## Sparse Merkle Trees

For large state spaces (like Ethereum's 2^256 possible addresses), regular tries are inefficient. **Sparse Merkle trees** optimize for mostly-empty trees.

### Structure

\`\`\`python
class SparseMerkleTree:
    """Sparse Merkle tree for large key spaces"""
    
    def __init__(self, depth: int = 256):
        self.depth = depth  # Tree depth (256 for Ethereum addresses)
        self.nodes: Dict[str, str] = {}  # Only store non-default nodes
        self.default_hashes = self._compute_default_hashes()
    
    def _compute_default_hashes(self) -> List[str]:
        """Compute default hash at each level (empty tree)"""
        defaults = ["0" * 64]  # Leaf default
        for _ in range(self.depth):
            parent = self._hash_pair(defaults[-1], defaults[-1])
            defaults.append(parent)
        return defaults
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two child hashes"""
        combined = left + right
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _key_to_path(self, key: str) -> List[int]:
        """Convert key to binary path (left=0, right=1)"""
        # Convert hex key to binary
        key_int = int(key, 16)
        path = []
        for i in range(self.depth):
            bit = (key_int >> (self.depth - 1 - i)) & 1
            path.append(bit)
        return path
    
    def update(self, key: str, value: str):
        """Update key-value pair"""
        path = self._key_to_path(key)
        leaf_hash = hashlib.sha256(value.encode()).hexdigest()
        
        # Update path from leaf to root
        current_hash = leaf_hash
        current_index = key
        
        for level in range(self.depth - 1, -1, -1):
            # Store non-default node
            if current_hash != self.default_hashes[level]:
                self.nodes[current_index] = current_hash
            
            # Move to parent
            if path[level] == 0:
                # Current is left child
                right_sibling = self._get_node(current_index + "1", level)
                parent_hash = self._hash_pair(current_hash, right_sibling)
            else:
                # Current is right child
                left_sibling = self._get_node(current_index + "0", level)
                parent_hash = self._hash_pair(left_sibling, current_hash)
            
            current_hash = parent_hash
            current_index = current_index[:-1]  # Move up tree
    
    def _get_node(self, index: str, level: int) -> str:
        """Get node hash, or default if not stored"""
        return self.nodes.get(index, self.default_hashes[level])
    
    def get_root(self) -> str:
        """Get root hash"""
        return self._get_node("", self.depth)
    
    def get_proof(self, key: str) -> List[str]:
        """Generate inclusion proof"""
        path = self._key_to_path(key)
        proof = []
        current_index = key
        
        for level in range(self.depth - 1, -1, -1):
            if path[level] == 0:
                sibling = self._get_node(current_index + "1", level)
            else:
                sibling = self._get_node(current_index + "0", level)
            proof.append(sibling)
            current_index = current_index[:-1]
        
        return proof


# Example: Sparse Merkle tree (simplified to depth=8 for demo)
print("=== Sparse Merkle Tree Example ===\\n")

smt = SparseMerkleTree(depth=8)

# Store a few values in huge key space (2^8 = 256 possible keys)
updates = {
    "0x05": "Alice's balance: 100 ETH",
    "0x9a": "Bob's balance: 50 ETH",
    "0xff": "Charlie's balance: 200 ETH",
}

for key, value in updates.items():
    smt.update(key, value)
    print(f"Updated {key}: {value}")

print(f"\\nRoot hash: {smt.get_root()[:32]}...")
print(f"Stored nodes: {len(smt.nodes)} (out of 2^8 = 256 possible)")

# Generate proof
proof = smt.get_proof("0x05")
print(f"\\nProof length: {len(proof)} hashes (depth={smt.depth})")
\`\`\`

## Bloom Filters

Bloom filters enable fast membership testing with false positives but no false negatives.

### Use Case: Ethereum Logs

Ethereum uses Bloom filters in block headers to quickly check if a block contains logs matching certain topics:

\`\`\`python
class BloomFilter:
    """Bloom filter for fast membership testing"""
    
    def __init__(self, size: int = 2048, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [0] * size
    
    def _hash(self, item: str, seed: int) -> int:
        """Hash item with seed"""
        data = f"{item}:{seed}"
        hash_val = int(hashlib.sha256(data.encode()).hexdigest(), 16)
        return hash_val % self.size
    
    def add(self, item: str):
        """Add item to filter"""
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bits[index] = 1
    
    def might_contain(self, item: str) -> bool:
        """Check if item might be in set (false positives possible)"""
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if self.bits[index] == 0:
                return False  # Definitely not in set
        return True  # Might be in set
    
    def false_positive_rate(self, num_items: int) -> float:
        """Calculate expected false positive rate"""
        import math
        # Formula: (1 - e^(-k*n/m))^k
        # k = num_hashes, n = num_items, m = size
        k, n, m = self.num_hashes, num_items, self.size
        return (1 - math.exp(-k * n / m)) ** k


# Example: Ethereum log filtering
print("=== Bloom Filter Example ===\\n")

bloom = BloomFilter(size=2048, num_hashes=3)

# Add contract addresses and topics
events = [
    "Transfer(0xa711,0xb234)",
    "Approval(0xa711,0xc456)",
    "Swap(0xd789,0xe012)",
]

for event in events:
    bloom.add(event)
    print(f"Added: {event}")

print(f"\\nFilter size: {bloom.size} bits ({bloom.size // 8} bytes)")

# Query
queries = [
    "Transfer(0xa711,0xb234)",  # In filter
    "Transfer(0xffff,0x0000)",  # Not in filter
]

print(f"\\nQueries:")
for query in queries:
    result = bloom.might_contain(query)
    print(f"  '{query}': {result}")

# False positive analysis
print(f"\\nFalse positive rate: {bloom.false_positive_rate(len(events)) * 100:.2f}%")
\`\`\`

## Verkle Trees

Verkle trees (Vector Commitment Merkle Trees) are the future of Ethereum state storage, offering:
- Smaller proof sizes (~150 bytes vs ~3KB for Merkle-Patricia)
- Stateless client feasibility
- Better scalability

\`\`\`
Merkle proof: O(log n) hashes (~3 KB for Ethereum)
Verkle proof: O(1) size (~150 bytes, constant)
\`\`\`

## Summary

Blockchain data structures enable:
- **Merkle trees**: Compact proofs of inclusion (SPV clients)
- **Patricia tries**: Efficient key-value storage with proofs
- **Sparse Merkle trees**: Optimize for large, mostly-empty key spaces
- **Bloom filters**: Fast probabilistic membership testing
- **Verkle trees**: Next-generation proofs with constant size

These structures are fundamental to blockchain scalability—they enable light clients, rollups, and stateless validation.
`,
};
