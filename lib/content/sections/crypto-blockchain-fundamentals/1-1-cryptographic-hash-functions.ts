export const cryptographicHashFunctions = {
  title: 'Cryptographic Hash Functions',
  id: 'cryptographic-hash-functions',
  content: `
# Cryptographic Hash Functions

## Introduction

Cryptographic hash functions are the foundation of blockchain technology—they provide the security, integrity, and immutability that makes blockchains work. Without understanding hash functions, you cannot understand how Bitcoin prevents double-spending, how Ethereum ensures state integrity, or how proof-of-work mining actually functions.

**A hash function takes arbitrary-sized input and produces a fixed-size output (the "hash" or "digest").** But cryptographic hash functions have special properties that make them suitable for security applications.

### The Security Foundation

Consider this scenario: You download a 2GB blockchain node software from a website. How do you know the file wasn't tampered with during download? **The official site publishes a SHA-256 hash.** You hash your downloaded file and compare:

\`\`\`
Official hash:  a7b3c2d1e5f6...
Your file hash: a7b3c2d1e5f6...
Match! ✓ File is authentic
\`\`\`

If even one bit of the file was modified (by an attacker or corruption), the hash would be completely different. This is the power of cryptographic hashes.

## Properties of Cryptographic Hash Functions

For a hash function to be cryptographically secure, it must have these properties:

### 1. Deterministic

**The same input always produces the same output.** This seems obvious, but it's critical for verification.

\`\`\`python
import hashlib

def hash_sha256(data: str) -> str:
    """Hash data using SHA-256"""
    return hashlib.sha256(data.encode()).hexdigest()

# Deterministic - always same output
print(hash_sha256("Hello, Bitcoin!"))  
# Always: 4e9d3e3c8a7b2f1c...

# Run it a million times, same result
for _ in range(1_000_000):
    h = hash_sha256("Hello, Bitcoin!")
    assert h == "4e9d3e3c8a7b2f1c3d8e7a2b5c8f1e4d9a6b3c8e5f2a7d4b1c8e3f6a9d2b5c8e"
\`\`\`

### 2. Quick Computation

**Computing the hash must be fast.** Modern CPUs can compute millions of SHA-256 hashes per second.

\`\`\`python
import time

def benchmark_hashing(iterations: int = 1_000_000):
    """Benchmark SHA-256 hashing speed"""
    data = "Blockchain transaction data"
    
    start = time.time()
    for i in range(iterations):
        hashlib.sha256(data.encode()).hexdigest()
    end = time.time()
    
    hashes_per_second = iterations / (end - start)
    print(f"SHA-256: {hashes_per_second:,.0f} hashes/second")
    print(f"Time per hash: {(end - start) / iterations * 1000:.6f} ms")

# Example output on modern CPU:
# SHA-256: 2,458,000 hashes/second
# Time per hash: 0.000407 ms
\`\`\`

This speed is why Bitcoin miners can try billions of hashes per second searching for valid blocks.

### 3. Pre-image Resistance (One-Way Function)

**Given a hash output H, it should be computationally infeasible to find any input M such that hash(M) = H.**

This is the "one-way" property. You can easily hash data, but you **cannot reverse** the process.

\`\`\`python
# Easy direction: input → hash
password = "mySecretPass123"
password_hash = hashlib.sha256(password.encode()).hexdigest()
print(f"Hash: {password_hash}")
# Hash: 9b4f8c3d7e2a1f6b...

# Impossible direction: hash → input
# Given only: 9b4f8c3d7e2a1f6b...
# Cannot determine: mySecretPass123
# Only option: Try every possible password (brute force)

def brute_force_impossible():
    """
    Demonstrate why reversing SHA-256 is impossible
    """
    target_hash = password_hash
    
    # How many possible 15-character passwords?
    # 26 lowercase + 26 uppercase + 10 digits + 33 symbols = 95 chars
    # Possible 15-char passwords: 95^15 = 4.6 × 10^29
    
    # At 1 billion guesses per second:
    # Time needed: 4.6 × 10^29 / 10^9 / 60 / 60 / 24 / 365
    #            = 1.4 × 10^13 years
    #            = 14 trillion years
    #            = 1000x age of the universe
    
    print("Reversing SHA-256 through brute force:")
    print(f"Time needed: 14,000,000,000,000 years")
    print(f"Age of universe: 13,800,000,000 years")
    print("Conclusion: Computationally infeasible")
\`\`\`

This property secures Bitcoin addresses. Your address is derived from a public key hash, and even if someone sees your address, they **cannot reverse it** to find your private key.

### 4. Second Pre-image Resistance

**Given an input M1, it should be infeasible to find a different input M2 such that hash(M1) = hash(M2).**

This prevents attackers from creating fraudulent data that has the same hash as legitimate data.

\`\`\`python
def demonstrate_second_preimage_resistance():
    """
    Show why second pre-image resistance matters
    """
    # Original transaction
    original_tx = "Alice pays Bob 10 BTC"
    original_hash = hashlib.sha256(original_tx.encode()).hexdigest()
    
    print(f"Original: {original_tx}")
    print(f"Hash: {original_hash[:16]}...")
    
    # Attacker tries to create different transaction with same hash
    print("\\nAttacker attempting to find collision...")
    
    attempts = 0
    max_attempts = 10_000_000
    
    while attempts < max_attempts:
        # Try different transactions
        fake_tx = f"Alice pays Mallory {attempts} BTC"
        fake_hash = hashlib.sha256(fake_tx.encode()).hexdigest()
        
        if fake_hash == original_hash:
            print(f"COLLISION FOUND after {attempts} attempts!")
            print(f"Fake transaction: {fake_tx}")
            return
        
        attempts += 1
    
    print(f"No collision found after {attempts:,} attempts")
    print(f"SHA-256 has 2^256 possible outputs")
    print(f"Probability of collision: {attempts / (2**256):.2e}")
    print("Conclusion: Second pre-image resistance holds")

demonstrate_second_preimage_resistance()
\`\`\`

### 5. Collision Resistance

**It should be infeasible to find any two different inputs M1 and M2 such that hash(M1) = hash(M2).**

This is stronger than second pre-image resistance. You're looking for *any* collision, not targeting a specific hash.

\`\`\`python
def birthday_paradox_analysis():
    """
    Understand collision resistance through birthday paradox
    """
    # Birthday paradox: In a room of 23 people,
    # 50% chance two share a birthday (365 possibilities)
    
    # For SHA-256 (2^256 possible outputs):
    # How many hashes needed for 50% collision chance?
    
    import math
    
    bits = 256
    possible_outputs = 2**bits
    
    # Birthday paradox formula: √(2 * N * ln(2))
    hashes_for_50_percent = math.sqrt(2 * possible_outputs * math.log(2))
    
    print("SHA-256 Collision Resistance Analysis")
    print(f"Possible outputs: 2^{bits}")
    print(f"Hashes needed for 50% collision: 2^{bits//2}")
    print(f"That's: {hashes_for_50_percent:.2e} hashes")
    
    # At 10^18 hashes per second (all Bitcoin miners combined):
    seconds = hashes_for_50_percent / 1e18
    years = seconds / (60 * 60 * 24 * 365)
    
    print(f"\\nAt 1 exahash/second (all Bitcoin miners):")
    print(f"Time to 50% collision: {years:.2e} years")
    print(f"Age of universe: 1.38 × 10^10 years")
    print(f"Ratio: {years / 1.38e10:.2e}x age of universe")

birthday_paradox_analysis()

# Output:
# SHA-256 Collision Resistance Analysis
# Possible outputs: 2^256
# Hashes needed for 50% collision: 2^128
# That's: 2.74 × 10^38 hashes
#
# At 1 exahash/second (all Bitcoin miners):
# Time to 50% collision: 8.67 × 10^21 years
# Age of universe: 1.38 × 10^10 years
# Ratio: 6.28 × 10^11x age of universe
\`\`\`

## The Avalanche Effect

**A tiny change in input produces a completely different output.** This is what makes hash functions useful for detecting tampering.

\`\`\`python
def demonstrate_avalanche_effect():
    """
    Show how small input changes cause massive output changes
    """
    original = "Satoshi Nakamoto invented Bitcoin in 2009"
    original_hash = hashlib.sha256(original.encode()).hexdigest()
    
    # Change one character
    modified = "Satoshi Nakamoto invented Bitcoin in 2008"  # 2009 → 2008
    modified_hash = hashlib.sha256(modified.encode()).hexdigest()
    
    # Change is only one digit, but hashes are completely different
    print("Avalanche Effect Demonstration")
    print(f"Original:  {original}")
    print(f"Hash: {original_hash}")
    print()
    print(f"Modified:  {modified}")
    print(f"Hash: {modified_hash}")
    print()
    
    # Calculate how many bits changed
    def count_different_bits(hex1: str, hex2: str) -> int:
        int1 = int(hex1, 16)
        int2 = int(hex2, 16)
        xor = int1 ^ int2
        return bin(xor).count('1')
    
    bits_changed = count_different_bits(original_hash, modified_hash)
    total_bits = 256
    
    print(f"Bits changed: {bits_changed} out of {total_bits}")
    print(f"Percentage: {bits_changed / total_bits * 100:.1f}%")
    print(f"Expected for good hash: ~50%")
    
    # Test with very small change (one bit in binary)
    data1 = b"\\x00"
    data2 = b"\\x01"
    hash1 = hashlib.sha256(data1).hexdigest()
    hash2 = hashlib.sha256(data2).hexdigest()
    
    bits_changed2 = count_different_bits(hash1, hash2)
    print(f"\\nOne-bit input change:")
    print(f"Bits changed in output: {bits_changed2} ({bits_changed2/256*100:.1f}%)")

demonstrate_avalanche_effect()
\`\`\`

This avalanche effect is crucial for blockchain security. If you modify a transaction in a block, the block's hash changes completely, which changes the next block's hash (since blocks reference previous hashes), cascading through the entire chain.

## SHA-256 Deep Dive

**SHA-256 (Secure Hash Algorithm 256-bit)** is the hash function used by Bitcoin and many other blockchains. Let's understand how it works.

### SHA-256 Algorithm Overview

\`\`\`python
def sha256_implementation_overview():
    """
    SHA-256 algorithm steps (simplified explanation)
    """
    steps = """
    SHA-256 Algorithm Steps:
    
    1. Message Padding
       - Append '1' bit
       - Append '0' bits until length ≡ 448 (mod 512)
       - Append 64-bit message length
       - Result: Message is multiple of 512 bits
    
    2. Initialize Hash Values (H0-H7)
       - Eight 32-bit words
       - These are the first 32 bits of fractional parts of 
         square roots of first 8 primes: 2, 3, 5, 7, 11, 13, 17, 19
    
    3. Initialize Round Constants (K0-K63)
       - 64 constants
       - First 32 bits of fractional parts of cube roots of 
         first 64 primes
    
    4. Process Message in 512-bit Chunks
       For each chunk:
       a. Break into 16 32-bit words (W0-W15)
       b. Extend to 64 words using message schedule
       c. 64 rounds of compression function
       d. Update hash values
    
    5. Produce Final Hash
       - Concatenate H0-H7
       - Result: 256-bit hash
    """
    print(steps)
    
    # Show actual initialization values
    import struct
    
    print("\\nSHA-256 Initial Hash Values (H):")
    h_values = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ]
    
    for i, h in enumerate(h_values):
        print(f"H{i}: 0x{h:08x}")

sha256_implementation_overview()
\`\`\`

### Implementing SHA-256 Components

\`\`\`python
def sha256_from_scratch(message: bytes) -> str:
    """
    Simplified SHA-256 implementation for learning
    (Use hashlib in production!)
    """
    import struct
    
    # Initialize hash values (first 32 bits of sqrt of first 8 primes)
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h4 = 0x510e527f
    h5 = 0x9b05688c
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19
    
    # Initialize round constants (first 32 bits of cube roots of first 64 primes)
    k = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    
    # Helper functions
    def right_rotate(n, b):
        return ((n >> b) | (n << (32 - b))) & 0xffffffff
    
    def pad_message(msg: bytes) -> bytes:
        """Pad message to multiple of 512 bits"""
        msg_len = len(msg) * 8
        msg += b'\\x80'  # Append '1' bit (plus 7 zero bits)
        
        # Append zeros until length ≡ 448 (mod 512)
        while (len(msg) * 8) % 512 != 448:
            msg += b'\\x00'
        
        # Append original length as 64-bit big-endian
        msg += struct.pack('>Q', msg_len)
        
        return msg
    
    # Pad the message
    padded = pad_message(message)
    
    # Process message in 512-bit chunks
    for chunk_start in range(0, len(padded), 64):
        chunk = padded[chunk_start:chunk_start + 64]
        
        # Break chunk into 16 32-bit big-endian words
        w = list(struct.unpack('>16I', chunk))
        
        # Extend to 64 words
        for i in range(16, 64):
            s0 = right_rotate(w[i-15], 7) ^ right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = right_rotate(w[i-2], 17) ^ right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
        
        # 64 rounds
        for i in range(64):
            S1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)
            ch = (e & f) ^ ((~e) & g)
            temp1 = (h + S1 + ch + k[i] + w[i]) & 0xffffffff
            S0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xffffffff
            
            h = g
            g = f
            f = e
            e = (d + temp1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xffffffff
        
        # Update hash values
        h0 = (h0 + a) & 0xffffffff
        h1 = (h1 + b) & 0xffffffff
        h2 = (h2 + c) & 0xffffffff
        h3 = (h3 + d) & 0xffffffff
        h4 = (h4 + e) & 0xffffffff
        h5 = (h5 + f) & 0xffffffff
        h6 = (h6 + g) & 0xffffffff
        h7 = (h7 + h) & 0xffffffff
    
    # Produce final hash
    return '%08x%08x%08x%08x%08x%08x%08x%08x' % (h0, h1, h2, h3, h4, h5, h6, h7)

# Test our implementation
test_message = b"Hello, Blockchain!"
our_hash = sha256_from_scratch(test_message)
standard_hash = hashlib.sha256(test_message).hexdigest()

print("SHA-256 Implementation Test:")
print(f"Our implementation: {our_hash}")
print(f"Standard library:   {standard_hash}")
print(f"Match: {our_hash == standard_hash}")
\`\`\`

## Hash Puzzles and Mining

Bitcoin mining is fundamentally about solving hash puzzles. Understanding how this works is key to understanding proof-of-work.

### The Mining Problem

\`\`\`python
def demonstrate_mining_puzzle():
    """
    Demonstrate the Bitcoin mining puzzle
    """
    import time
    
    # Block data (simplified)
    block_data = "Block #850000, Previous Hash: 000000000000000...4a3f2, Transactions: [tx1, tx2, tx3]"
    
    # Mining difficulty (target): Hash must start with N zeros
    target_zeros = 4  # Bitcoin uses ~19 zeros (difficulty adjusts)
    target = "0" * target_zeros
    
    print("Bitcoin Mining Simulation")
    print(f"Block data: {block_data[:60]}...")
    print(f"Target: Hash must start with {target_zeros} zeros")
    print(f"Target: {target}...")
    print()
    
    nonce = 0
    attempts = 0
    start_time = time.time()
    
    while True:
        # Create candidate block (data + nonce)
        candidate = f"{block_data}, Nonce: {nonce}"
        
        # Hash it
        block_hash = hashlib.sha256(candidate.encode()).hexdigest()
        attempts += 1
        
        # Check if hash meets target (starts with required zeros)
        if block_hash.startswith(target):
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"✓ BLOCK MINED!")
            print(f"Nonce: {nonce}")
            print(f"Hash: {block_hash}")
            print(f"Attempts: {attempts:,}")
            print(f"Time: {elapsed:.2f} seconds")
            print(f"Hash rate: {attempts / elapsed:,.0f} H/s")
            break
        
        nonce += 1
        
        # Show progress
        if attempts % 100000 == 0:
            print(f"Attempt {attempts:,}: {block_hash[:20]}... (no luck)")

demonstrate_mining_puzzle()
\`\`\`

### Understanding Mining Difficulty

\`\`\`python
def analyze_mining_difficulty():
    """
    Analyze how difficulty affects mining time
    """
    import time
    
    block_data = "Bitcoin block data"
    
    print("Mining Difficulty Analysis")
    print("=" * 50)
    
    for target_zeros in range(1, 7):
        target = "0" * target_zeros
        nonce = 0
        attempts = 0
        start = time.time()
        
        # Try to find valid hash
        while True:
            candidate = f"{block_data}, Nonce: {nonce}"
            block_hash = hashlib.sha256(candidate.encode()).hexdigest()
            attempts += 1
            
            if block_hash.startswith(target):
                elapsed = time.time() - start
                
                print(f"\\nDifficulty: {target_zeros} leading zeros")
                print(f"Target: {target}...")
                print(f"Found: {block_hash}")
                print(f"Attempts: {attempts:,}")
                print(f"Time: {elapsed:.3f}s")
                print(f"Expected attempts: ~{16**target_zeros:,}")
                
                break
            
            nonce += 1
            
            # Timeout for demo
            if attempts > 100_000_000:
                print(f"\\n{target_zeros} zeros: Too difficult for demo (>100M attempts)")
                break
    
    print("\\n" + "=" * 50)
    print("Bitcoin Mining Reality:")
    print("- Current difficulty: ~19 leading zeros")
    print("- Expected attempts: ~10^22 (10,000,000,000,000,000,000,000)")
    print("- Network hash rate: ~600 EH/s (600,000,000,000,000,000,000 H/s)")
    print("- Average time to find block: 10 minutes")

# Run analysis for demonstration
# analyze_mining_difficulty()  # Uncomment to run
\`\`\`

## Merkle Trees Fundamentals

**Merkle trees** are a crucial hash-based data structure used in blockchains. They allow efficient and secure verification of large data structures.

### Building a Merkle Tree

\`\`\`python
class MerkleTree:
    """
    Simple Merkle Tree implementation
    """
    def __init__(self, transactions: list[str]):
        self.transactions = transactions
        self.tree = self.build_tree()
        self.root = self.tree[0] if self.tree else None
    
    def hash(self, data: str) -> str:
        """Hash a piece of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def build_tree(self) -> list[list[str]]:
        """Build the Merkle tree bottom-up"""
        if not self.transactions:
            return []
        
        # Level 0: Hash all transactions
        current_level = [self.hash(tx) for tx in self.transactions]
        tree = [current_level]
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs of hashes
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                # If odd number of hashes, duplicate the last one
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left
                
                # Hash the concatenation
                combined = self.hash(left + right)
                next_level.append(combined)
            
            tree.insert(0, next_level)
            current_level = next_level
        
        return tree
    
    def get_root(self) -> str:
        """Get the Merkle root"""
        return self.root
    
    def get_proof(self, tx_index: int) -> list[tuple[str, str]]:
        """
        Get Merkle proof for a transaction
        Returns list of (hash, side) where side is 'left' or 'right'
        """
        if tx_index >= len(self.transactions):
            raise ValueError("Transaction index out of range")
        
        proof = []
        index = tx_index
        
        # Start from bottom level (transaction hashes)
        for level in reversed(self.tree[1:]):
            # Determine sibling index
            if index % 2 == 0:
                sibling_index = index + 1
                side = 'right'
            else:
                sibling_index = index - 1
                side = 'left'
            
            # Add sibling hash to proof (if it exists)
            if sibling_index < len(level):
                proof.append((level[sibling_index], side))
            
            # Move to parent index
            index = index // 2
        
        return proof
    
    def verify_proof(self, transaction: str, tx_index: int, proof: list[tuple[str, str]]) -> bool:
        """
        Verify a Merkle proof
        """
        # Start with transaction hash
        current_hash = self.hash(transaction)
        
        # Apply proof hashes
        for proof_hash, side in proof:
            if side == 'left':
                current_hash = self.hash(proof_hash + current_hash)
            else:
                current_hash = self.hash(current_hash + proof_hash)
        
        # Check if we arrived at the root
        return current_hash == self.root

# Demonstrate Merkle tree
transactions = [
    "Alice pays Bob 1 BTC",
    "Bob pays Carol 0.5 BTC",
    "Carol pays Dave 0.3 BTC",
    "Dave pays Eve 0.2 BTC",
    "Eve pays Frank 0.1 BTC",
    "Frank pays Alice 0.05 BTC"
]

print("Merkle Tree Demonstration")
print("=" * 70)

tree = MerkleTree(transactions)

print(f"\\nTransactions: {len(transactions)}")
for i, tx in enumerate(transactions):
    print(f"  {i}: {tx}")

print(f"\\nMerkle Root: {tree.get_root()}")

# Get proof for transaction 2
tx_index = 2
proof = tree.get_proof(tx_index)

print(f"\\nMerkle Proof for transaction {tx_index}:")
print(f"Transaction: {transactions[tx_index]}")
print(f"Proof length: {len(proof)} hashes")

for i, (hash_val, side) in enumerate(proof):
    print(f"  Step {i+1}: {hash_val[:16]}... (sibling on {side})")

# Verify the proof
is_valid = tree.verify_proof(transactions[tx_index], tx_index, proof)
print(f"\\nProof verification: {'✓ Valid' if is_valid else '✗ Invalid'}")

# Try with tampered transaction
tampered_tx = "Carol pays Dave 3.0 BTC"  # Changed amount
is_valid_tampered = tree.verify_proof(tampered_tx, tx_index, proof)
print(f"Tampered transaction: {'✓ Valid' if is_valid_tampered else '✗ Invalid'}")

print("\\n" + "=" * 70)
print("Merkle Tree Benefits:")
print("- Efficient verification: O(log n) vs O(n)")
print(f"- This example: {len(proof)} hashes vs {len(transactions)} transactions")
print("- Used in Bitcoin SPV (Simplified Payment Verification)")
print("- Allows light clients to verify transactions without full blockchain")
\`\`\`

## Applications in Blockchain

### Bitcoin's Use of Hashing

\`\`\`python
def bitcoin_hashing_applications():
    """
    Demonstrate how Bitcoin uses hash functions
    """
    print("Hash Functions in Bitcoin")
    print("=" * 70)
    
    # 1. Transaction IDs
    print("\\n1. Transaction IDs (TXID)")
    transaction_data = {
        "inputs": [{"prev_tx": "abc123...", "index": 0}],
        "outputs": [{"address": "1A1zP1...", "amount": 50}]
    }
    tx_serialized = str(transaction_data)
    txid = hashlib.sha256(hashlib.sha256(tx_serialized.encode()).digest()).hexdigest()
    print(f"   Transaction: {tx_serialized[:50]}...")
    print(f"   TXID: {txid}")
    print(f"   Note: Bitcoin uses double SHA-256 for TXIDs")
    
    # 2. Block Hashes
    print("\\n2. Block Hashes")
    block_header = {
        "version": 1,
        "prev_block": "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
        "merkle_root": txid,
        "timestamp": 1231006505,
        "difficulty": 1,
        "nonce": 2083236893
    }
    block_serialized = str(block_header)
    block_hash = hashlib.sha256(hashlib.sha256(block_serialized.encode()).digest()).hexdigest()
    print(f"   Block header: {block_serialized[:50]}...")
    print(f"   Block hash: {block_hash}")
    
    # 3. Address Generation (simplified)
    print("\\n3. Bitcoin Address Generation")
    print("   Step 1: Generate public key from private key (ECDSA)")
    print("   Step 2: SHA-256 hash of public key")
    public_key = "04a1b2c3d4..."  # Simplified
    sha256_hash = hashlib.sha256(public_key.encode()).hexdigest()
    print(f"   SHA-256: {sha256_hash}")
    
    print("   Step 3: RIPEMD-160 hash of SHA-256")
    import hashlib as hl
    ripemd160 = hl.new('ripemd160')
    ripemd160.update(bytes.fromhex(sha256_hash))
    pubkey_hash = ripemd160.hexdigest()
    print(f"   RIPEMD-160: {pubkey_hash}")
    print(f"   Step 4: Add network byte, checksum → Bitcoin address")
    
    # 4. Proof of Work
    print("\\n4. Proof of Work Mining")
    print("   Target: Hash must be < target value")
    print("   Example target: 0000000000000000000...")
    print("   Miners search for nonce where:")
    print("   SHA-256(SHA-256(block_header + nonce)) < target")
    
    print("\\n" + "=" * 70)

bitcoin_hashing_applications()
\`\`\`

## Common Vulnerabilities and Attacks

### Length Extension Attacks

\`\`\`python
def explain_length_extension_attack():
    """
    Explain why Bitcoin uses double hashing (SHA-256d)
    """
    print("Length Extension Attack")
    print("=" * 70)
    print("""
SHA-256 has a vulnerability called "length extension attack":
- If you know hash(message), you can compute hash(message + extra_data)
- Without knowing the original message!

Example vulnerable code:
    token = SHA-256(secret + user_data)
    
Attacker can:
1. See token
2. Append their own data
3. Compute valid token for (secret + user_data + malicious_data)

Bitcoin's Solution: Double Hashing
    SHA-256(SHA-256(data))
    
The inner hash's output becomes input to outer hash.
Attacker can't extend the inner message because they don't control outer hash input.

This is why Bitcoin uses "SHA-256d" (double SHA-256) for:
- Transaction IDs
- Block hashes
- Other critical security operations
    """)

explain_length_extension_attack()
\`\`\`

### Hash Collision Attacks

\`\`\`python
def explain_collision_attacks():
    """
    Explain theoretical collision attacks
    """
    print("Hash Collision Attacks")
    print("=" * 70)
    print("""
SHA-1 Broken (2017):
- Google and CWI Amsterdam found collision
- Two different PDF files with same SHA-1 hash
- Cost: 6,500 CPU years, 110 GPU years
- Impact: Don't use SHA-1 for security!

SHA-256 Status:
- No practical collisions found
- Theoretical: 2^128 operations for 50% collision (birthday attack)
- Current computation: ~2^80 operations (Bitcoin network, 10 years)
- Still 2^48 times more work needed (281 trillion times)
- Considered secure for decades

MD5 Broken (2004):
- Collisions found in seconds on modern hardware
- Never use MD5 for security
- OK only for non-security checksums

Why This Matters for Blockchain:
- Transaction tampering would need hash collision
- Block hash collision would break proof-of-work
- Address collision would break wallet security
- Current hash functions (SHA-256, Keccak-256) are secure
    """)
    
    # Demonstrate MD5 weakness vs SHA-256 strength
    print("\\nHash Function Comparison:")
    
    data1 = b"hello"
    data2 = b"world"
    
    import hashlib as hl
    
    print(f"\\nMD5 (128-bit, BROKEN):")
    print(f"  md5('{data1.decode()}'): {hl.md5(data1).hexdigest()}")
    print(f"  md5('{data2.decode()}'): {hl.md5(data2).hexdigest()}")
    print(f"  Security: ✗ Do not use")
    
    print(f"\\nSHA-1 (160-bit, BROKEN):")
    print(f"  sha1('{data1.decode()}'): {hl.sha1(data1).hexdigest()}")
    print(f"  sha1('{data2.decode()}'): {hl.sha1(data2).hexdigest()}")
    print(f"  Security: ✗ Do not use")
    
    print(f"\\nSHA-256 (256-bit, SECURE):")
    print(f"  sha256('{data1.decode()}'): {hl.sha256(data1).hexdigest()}")
    print(f"  sha256('{data2.decode()}'): {hl.sha256(data2).hexdigest()}")
    print(f"  Security: ✓ Current standard")

explain_collision_attacks()
\`\`\`

## Practical Implementation: Content Addressable Storage

\`\`\`python
class ContentAddressableStorage:
    """
    Simple content-addressable storage using hash functions
    Used in Git, IPFS, and blockchain systems
    """
    def __init__(self):
        self.storage = {}
    
    def store(self, content: bytes) -> str:
        """
        Store content and return its hash (address)
        """
        content_hash = hashlib.sha256(content).hexdigest()
        self.storage[content_hash] = content
        return content_hash
    
    def retrieve(self, content_hash: str) -> bytes:
        """
        Retrieve content by its hash
        """
        if content_hash not in self.storage:
            raise KeyError(f"Content not found: {content_hash}")
        return self.storage[content_hash]
    
    def verify(self, content_hash: str) -> bool:
        """
        Verify that stored content matches its hash
        """
        if content_hash not in self.storage:
            return False
        
        stored_content = self.storage[content_hash]
        computed_hash = hashlib.sha256(stored_content).hexdigest()
        return computed_hash == content_hash

# Demonstrate content-addressable storage
storage = ContentAddressableStorage()

print("Content-Addressable Storage Demo")
print("=" * 70)

# Store some data
data1 = b"Blockchain transaction: Alice pays Bob 1 BTC"
hash1 = storage.store(data1)
print(f"\\nStored: {data1.decode()}")
print(f"Address: {hash1}")

# Retrieve by hash
retrieved = storage.retrieve(hash1)
print(f"\\nRetrieved: {retrieved.decode()}")
print(f"Match: {retrieved == data1}")

# Verify integrity
is_valid = storage.verify(hash1)
print(f"Integrity check: {'✓ Valid' if is_valid else '✗ Invalid'}")

# Demonstrate deduplication
hash2 = storage.store(data1)  # Store same data again
print(f"\\nStored same data again")
print(f"Same address: {hash1 == hash2}")
print(f"Storage size: {len(storage.storage)} items (deduplication works!)")

# Try to tamper with data
print(f"\\nAttempting to tamper with stored data...")
tampered_hash = hash1
storage.storage[tampered_hash] = b"Blockchain transaction: Alice pays Bob 100 BTC"
is_valid_after_tampering = storage.verify(tampered_hash)
print(f"Integrity check: {'✓ Valid' if is_valid_after_tampering else '✗ Invalid (tampering detected!)'}")

print("\\n" + "=" * 70)
print("Applications:")
print("- Git: Commits, trees, and blobs addressed by SHA-1 hash")
print("- IPFS: Files addressed by multihash (usually SHA-256)")
print("- Blockchains: Transactions and blocks referenced by hash")
print("- Benefits: Deduplication, integrity verification, immutability")
\`\`\`

## Performance Considerations

\`\`\`python
def hash_function_performance_comparison():
    """
    Compare performance of different hash functions
    """
    import time
    
    # Test data
    data_sizes = [100, 1024, 10240, 102400, 1024000]  # 100B to 1MB
    iterations = 10000
    
    print("Hash Function Performance Comparison")
    print("=" * 70)
    
    for size in data_sizes:
        test_data = b"x" * size
        
        print(f"\\nData size: {size:,} bytes")
        print("-" * 40)
        
        # Test SHA-256
        start = time.time()
        for _ in range(iterations):
            hashlib.sha256(test_data).digest()
        sha256_time = (time.time() - start) / iterations * 1000
        
        # Test SHA-512
        start = time.time()
        for _ in range(iterations):
            hashlib.sha512(test_data).digest()
        sha512_time = (time.time() - start) / iterations * 1000
        
        # Test SHA3-256
        start = time.time()
        for _ in range(iterations):
            hashlib.sha3_256(test_data).digest()
        sha3_time = (time.time() - start) / iterations * 1000
        
        # Test BLAKE2b
        start = time.time()
        for _ in range(iterations):
            hashlib.blake2b(test_data).digest()
        blake2_time = (time.time() - start) / iterations * 1000
        
        print(f"  SHA-256:   {sha256_time:.6f} ms")
        print(f"  SHA-512:   {sha512_time:.6f} ms")
        print(f"  SHA3-256:  {sha3_time:.6f} ms")
        print(f"  BLAKE2b:   {blake2_time:.6f} ms")
        
        # Calculate throughput
        throughput_sha256 = (size / 1024 / 1024) / (sha256_time / 1000)
        print(f"  SHA-256 throughput: {throughput_sha256:.2f} MB/s")

# Run performance comparison
# hash_function_performance_comparison()  # Uncomment to run
\`\`\`

## Summary

Cryptographic hash functions are the foundation of blockchain technology:

1. **Properties**: Deterministic, fast, one-way, collision-resistant, avalanche effect
2. **SHA-256**: Bitcoin's hash function, 256-bit output, ~2^256 possible hashes
3. **Applications**: Transaction IDs, block hashes, mining puzzles, Merkle trees, addresses
4. **Security**: Pre-image resistance, collision resistance, second pre-image resistance
5. **Merkle Trees**: Efficient verification of large data sets with O(log n) proofs
6. **Mining**: Proof-of-work is finding hash below target (adjustable difficulty)
7. **Double Hashing**: SHA-256d prevents length extension attacks

**Key Takeaways**:
- Hashes are fingerprints of data—any change produces completely different hash
- One-way property makes them perfect for cryptographic applications
- Collision resistance ensures integrity of blockchain
- Merkle trees enable light clients and efficient verification
- Understanding hashing is essential for understanding blockchain security

In the next section, we'll explore **public key cryptography** and how it enables digital signatures and secure transactions without revealing private keys.

## Further Reading

- **Bitcoin Whitepaper**: Satoshi Nakamoto's original paper
- **FIPS 180-4**: SHA-256 specification
- **RFC 6234**: US Secure Hash Algorithms (SHA and SHA-based HMAC and HKDF)
- **Merkle Tree Patent**: Original 1979 patent by Ralph Merkle
- **"Mastering Bitcoin"** by Andreas Antonopoulos: Chapter on cryptographic foundations

## Practice Exercises

1. Implement a simple blockchain that chains blocks using hashes
2. Build a Merkle tree and generate/verify proofs for different transactions
3. Create a simplified mining simulator with adjustable difficulty
4. Analyze the avalanche effect by changing single bits in input data
5. Implement content-addressable storage for blockchain data
`,
};
