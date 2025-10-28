export const cryptographicHashFunctionsDiscussion = [
  {
    id: 1,
    question:
      "You're building a new cryptocurrency and need to choose a hash function. Your technical co-founder suggests using MD5 because 'it's faster than SHA-256, and speed matters for transaction throughput.' A security consultant recommends SHA-3 (Keccak-256) arguing it's 'more modern and resistant to future attacks.' Your CTO wants to use double SHA-256 like Bitcoin for 'battle-tested security.' How would you evaluate these options? What technical, security, and practical factors should influence your decision? What would you choose and why?",
    answer: `## Comprehensive Answer:

This is a critical decision that will affect your cryptocurrency's security for its entire lifetime. Let's analyze each option and the underlying factors.

### Evaluating MD5: Speed vs Security Trade-off

**Technical Analysis**:

\`\`\`python
import hashlib
import time

def benchmark_hash_functions(data_size_mb=100, iterations=1000):
    """
    Benchmark different hash functions
    """
    test_data = b"x" * (data_size_mb * 1024 * 1024)
    
    results = {}
    
    # Test MD5
    start = time.time()
    for _ in range(iterations):
        hashlib.md5(test_data).digest()
    results['MD5'] = {
        'time': (time.time() - start) / iterations,
        'bits': 128,
        'status': 'BROKEN'
    }
    
    # Test SHA-256
    start = time.time()
    for _ in range(iterations):
        hashlib.sha256(test_data).digest()
    results['SHA-256'] = {
        'time': (time.time() - start) / iterations,
        'bits': 256,
        'status': 'SECURE'
    }
    
    # Test SHA-3 (Keccak-256)
    start = time.time()
    for _ in range(iterations):
        hashlib.sha3_256(test_data).digest()
    results['SHA3-256'] = {
        'time': (time.time() - start) / iterations,
        'bits': 256,
        'status': 'SECURE'
    }
    
    return results

# MD5 is indeed faster, but...
\`\`\`

**Why MD5 is COMPLETELY UNACCEPTABLE**:

1. **Cryptographically Broken** (2004):
   - Collisions can be found in seconds on modern hardware
   - Not minutes or hours—literally seconds
   - Researchers can generate two different messages with same hash

2. **Real Attack Scenario**:
\`\`\`python
# Attacker could create two transactions with same MD5 hash:
tx1 = "Pay Alice 0.1 BTC"
tx2 = "Pay Attacker 1000 BTC"

# If both hash to same value, attacker could:
# 1. Get tx1 accepted by network
# 2. Later substitute tx2 (same hash = appears valid)
# 3. Steal funds

# With MD5, this is TRIVIAL to execute
\`\`\`

3. **Speed Argument is Flawed**:
   - Transaction throughput bottleneck is NOT hashing speed
   - Bottlenecks: Network propagation, consensus, validation, storage
   - SHA-256 can process 500+ MB/s on single core
   - Transaction size: ~250 bytes
   - SHA-256 can hash: ~2 million transactions/second per core
   
4. **Industry Precedent**:
   - No serious cryptocurrency uses MD5
   - Even Git moved away from SHA-1 (stronger than MD5) after collision attack
   - Using MD5 would make you a laughingstock in security community

**Verdict**: MD5 is absolutely disqualifying. Anyone suggesting it for security doesn't understand cryptography.

### Evaluating SHA-256 (Double): Bitcoin's Proven Choice

**Why Bitcoin Uses Double SHA-256**:

\`\`\`python
def double_sha256(data: bytes) -> bytes:
    """
    Bitcoin's hash function: SHA-256(SHA-256(data))
    """
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# Why double?
# 1. Protects against length extension attacks
# 2. Adds another layer of security
# 3. Historical paranoia (good for security!)
\`\`\`

**Advantages**:

1. **Battle-Tested** (15+ years):
   - Bitcoin: $600B+ market cap, zero hash-related exploits
   - Litecoin, Bitcoin Cash, Dogecoin: Same hash, no issues
   - Attacked by nation-states, academics, criminals—never broken

2. **Hardware Optimization**:
   - ASICs exist for SHA-256 (mining hardware)
   - Massive ecosystem for hardware acceleration
   - Intel/AMD CPUs have SHA extensions (2x-3x faster)

3. **Network Effects**:
   - Developers know SHA-256 intimately
   - Tools, libraries, documentation abundant
   - Security research constantly ongoing

4. **Conservative Security Margin**:
   - 256 bits = 2^256 possible outputs
   - Even with birthday attack: 2^128 operations for collision
   - Current best attack: 2^128 operations (no improvement)
   - Quantum computers: Grover's algorithm reduces to 2^128 (still secure)

**Disadvantages**:

1. **Slightly Slower** than some alternatives (~10-20% slower than BLAKE2)
2. **Potential Future Vulnerability**: Theoretical (none found in 20+ years)
3. **Not "Post-Quantum"**: But collision resistance still 2^128 even with quantum

**Verdict**: Proven, secure, ecosystem advantage. Conservative choice.

### Evaluating SHA-3 (Keccak-256): The Modern Alternative

**Technical Overview**:

\`\`\`python
import hashlib

def analyze_sha3():
    """
    SHA-3 (Keccak) properties
    """
    data = b"Test transaction data"
    
    # SHA-3 uses completely different construction than SHA-2
    # Sponge construction vs Merkle-Damgård
    sha3_hash = hashlib.sha3_256(data).hexdigest()
    sha2_hash = hashlib.sha256(data).hexdigest()
    
    print("Different internal constructions:")
    print(f"SHA-256 (SHA-2 family): {sha2_hash}")
    print(f"SHA3-256: {sha3_hash}")

# SHA-3 was designed to be different from SHA-2 family
# If SHA-2 is ever broken, SHA-3 likely remains secure
\`\`\`

**Advantages**:

1. **Different Design**:
   - Sponge construction (vs Merkle-Damgård in SHA-2)
   - If SHA-2 family has fundamental flaw, SHA-3 unaffected
   - Provides "algorithmic diversity"

2. **Ethereum Uses It**:
   - Keccak-256 (pre-NIST SHA-3 version)
   - Ethereum: $200B+ market cap, secure for 9+ years
   - Proven in production at scale

3. **Theoretical Security**:
   - Won NIST competition (2012)
   - Extensive cryptanalysis during competition
   - Designed by experienced cryptographers (Keccak team)

4. **Resistance to Known Attacks**:
   - Immune to length extension attacks (by design)
   - Different structure makes certain attacks impossible

**Disadvantages**:

1. **Less Battle-Tested** than SHA-256:
   - ~11 years in production vs 20+ for SHA-256
   - Fewer person-years of cryptanalysis

2. **Performance**:
   - Slightly slower than SHA-256 (without hardware acceleration)
   - Less hardware support (fewer ASICs, no CPU extensions yet)

3. **Network Effects Weaker**:
   - Smaller ecosystem than SHA-256
   - Fewer developers deeply familiar

4. **Compatibility**:
   - Can't leverage Bitcoin mining infrastructure
   - Different mining algorithms needed

**Verdict**: Excellent modern choice, particularly if you want algorithmic diversity from Bitcoin. Slightly riskier due to less history.

### The Decision Framework

**Key Factors to Consider**:

\`\`\`python
decision_framework = {
    'Security': {
        'MD5': 0,      # Broken, unusable
        'SHA-256': 10, # Proven for 20+ years
        'SHA3-256': 9  # Solid but less history
    },
    
    'Performance': {
        'MD5': 10,      # Fast but irrelevant (broken)
        'SHA-256': 9,   # Very fast, hardware acceleration
        'SHA3-256': 8   # Good but less optimization
    },
    
    'Ecosystem': {
        'MD5': 0,       # Dead ecosystem
        'SHA-256': 10,  # Massive ecosystem
        'SHA3-256': 7   # Growing ecosystem
    },
    
    'Future-Proofing': {
        'MD5': 0,       # Already broken
        'SHA-256': 8,   # Likely secure for decades
        'SHA3-256': 9   # Modern design, algorithmic diversity
    },
    
    'Community Trust': {
        'MD5': 0,       # Would be ridiculed
        'SHA-256': 10,  # Bitcoin's choice = trusted
        'SHA3-256': 8   # Ethereum's choice = respected
    }
}
\`\`\`

### My Recommendation: SHA-256 (Double) with Path to SHA-3

**Choice**: **Double SHA-256** (like Bitcoin)

**Reasoning**:

1. **Security First**:
   - No cryptocurrency has ever lost funds due to SHA-256 weakness
   - 20+ years of real-world attack resistance
   - Conservative choice protects users' money

2. **Ecosystem Advantages**:
   - Leverage existing Bitcoin tooling, libraries, documentation
   - Developers familiar with SHA-256 can contribute
   - Hardware optimization available

3. **Trust & Adoption**:
   - "We use Bitcoin's proven hash function" is powerful marketing
   - Users trust what Bitcoin uses
   - Easier to get security audits (auditors know SHA-256)

4. **Performance Reality**:
   - Hashing is NOT your bottleneck
   - Network, consensus, storage are real bottlenecks
   - 10-20% hash speed difference is meaningless in practice

**With Strategic Flexibility**:

\`\`\`python
# Design your protocol with hash function abstraction:

from abc import ABC, abstractmethod

class HashFunction(ABC):
    @abstractmethod
    def hash(self, data: bytes) -> bytes:
        pass

class DoubleSHA256(HashFunction):
    def hash(self, data: bytes) -> bytes:
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

class Keccak256(HashFunction):
    def hash(self, data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()

# Protocol can switch hash functions in future if needed
# (though this requires hard fork)
\`\`\`

**If You Want to Be Different**: Choose **Keccak-256** (SHA-3)

Valid reasons:
- Algorithmic diversity from Bitcoin
- Ethereum compatibility
- Modern cryptographic design
- Potential performance improvements as hardware catches up

But understand the trade-offs:
- Less proven in time
- Smaller ecosystem
- Slightly more risk

**NEVER** choose MD5. That would be professional malpractice.

### Real-World Example: Ethereum's Choice

Ethereum chose **Keccak-256** (variant of SHA-3):

\`\`\`python
# Ethereum's hash function
def ethereum_hash(data: bytes) -> bytes:
    """
    Ethereum uses Keccak-256 (pre-NIST finalization version)
    """
    import sha3  # pysha3 library
    return sha3.keccak_256(data).digest()

# Why Ethereum chose Keccak:
reasons = [
    "Algorithmic diversity from Bitcoin (wanted to be different)",
    "Modern cryptographic design",
    "Sponge construction has theoretical advantages",
    "2013 decision: SHA-3 just won NIST competition",
    "Vitalik Buterin's cryptographic preference"
]

# Result: 9+ years later, no security issues
# Proven choice in production
\`\`\`

### Conclusion

**For a new cryptocurrency**:

1. **Default Choice**: Double SHA-256
   - Proven security
   - Massive ecosystem
   - Maximum trust
   - Conservative and safe

2. **Innovative Choice**: Keccak-256 (SHA-3)
   - Modern design
   - Ethereum compatibility
   - Algorithmic diversity
   - Slightly more risk

3. **Never Ever**: MD5 or SHA-1
   - Cryptographically broken
   - Would destroy credibility
   - Security vulnerability day one

**Final Answer**: I'd choose **Double SHA-256** for a new cryptocurrency. Security and trust matter more than bleeding-edge cryptography. Once you have significant value in the system, changing hash functions is nearly impossible (requires hard fork). Better to choose the most proven option.

The 10-20% performance difference doesn't matter. User trust matters. Security matters. SHA-256 delivers both.

*"In cryptography, boring is good. Novel is dangerous."*
`,
  },
  {
    id: 2,
    question:
      "A blockchain project discovers that due to a bug in their Merkle tree implementation, it's possible to create two different sets of transactions that produce the same Merkle root. This effectively means an attacker could substitute transactions in a block without changing the block hash. What cryptographic property has been violated? Walk through the immediate security implications, how an attacker might exploit this, what emergency measures the project should take, and how to prevent this category of bugs in future blockchain implementations.",
    answer: `## Comprehensive Answer:

This is a **critical security vulnerability** that violates **collision resistance**—one of the fundamental properties of cryptographic hash functions. This situation represents a catastrophic failure that could lead to complete loss of blockchain integrity.

### Property Violated: Collision Resistance

**What Should Be True**:
\`\`\`python
# Collision Resistance Property:
# It should be computationally infeasible to find M1 ≠ M2 where hash(M1) == hash(M2)

def should_be_impossible():
    """
    Finding two different transaction sets with same Merkle root
    should be impossible
    """
    transactions_set_1 = ["tx1", "tx2", "tx3", "tx4"]
    transactions_set_2 = ["evil_tx1", "evil_tx2", "evil_tx3", "evil_tx4"]
    
    merkle_root_1 = build_merkle_tree(transactions_set_1).root
    merkle_root_2 = build_merkle_tree(transactions_set_2).root
    
    # These should NEVER be equal (except with negligible probability)
    assert merkle_root_1 != merkle_root_2
    
    # If they ARE equal, collision resistance is broken
\`\`\`

**What the Bug Allows**:
\`\`\`python
# With the bug, attacker can do:
legit_transactions = ["Alice pays Bob 1 BTC", "Carol pays Dave 0.5 BTC"]
malicious_transactions = ["Alice pays Attacker 1000 BTC", "Attacker pays Attacker 999 BTC"]

# Bug allows:
merkle_root(legit_transactions) == merkle_root(malicious_transactions)

# This is a COLLISION - two different inputs with same output
# This breaks the fundamental security assumption of Merkle trees
\`\`\`

### Understanding Common Merkle Tree Bugs

**Bug Category 1: Second Pre-image Attack via Leaf/Node Confusion**

\`\`\`python
# VULNERABLE implementation:
class VulnerableMerkleTree:
    def hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()
    
    def build_tree(self, leaves: list[str]) -> str:
        if len(leaves) == 1:
            return self.hash(leaves[0])
        
        # BUG: No prefix to distinguish leaf vs internal node
        mid = len(leaves) // 2
        left = self.build_tree(leaves[:mid])
        right = self.build_tree(leaves[mid:])
        
        # This allows attack!
        return self.hash(left + right)

# Attack:
# An attacker can construct transactions where:
# hash(leaf) == hash(node_left + node_right)
# This creates a collision between leaf and internal node

attack_demo = """
Legitimate tree:
    Root
   /    \\
  A      B

Attack tree:
    Root
    |
    C

Where hash(C) == hash(A + B)
Both produce same root but different transaction sets!
"""
\`\`\`

**Bug Category 2: Length Extension Vulnerability**

\`\`\`python
# VULNERABLE: Using concatenation without proper domain separation
def vulnerable_combine(left: str, right: str) -> str:
    # BUG: Simple concatenation allows length extension attacks
    return hashlib.sha256((left + right).encode()).hexdigest()

# Attack:
# Attacker can extend messages to create collisions
# Example:
#   hash("AB" + "CD") might equal hash("ABC" + "D")
#   depending on how boundaries are handled
\`\`\`

**Bug Category 3: Duplicate Handling**

\`\`\`python
# VULNERABLE: Improper handling of duplicates
class BuggyMerkleTree:
    def build_tree(self, leaves: list[str]) -> str:
        current_level = [self.hash(leaf) for leaf in leaves]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # BUG: If odd number, what happens?
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left  # Duplicate last element
                
                next_level.append(self.hash(left + right))
            
            current_level = next_level
        
        return current_level[0]

# Attack:
# Different transaction counts can produce same root
# [A, B, C] might have same root as [A, B, C, C]
# Attacker adds/removes duplicate transactions to create collision
\`\`\`

### Immediate Security Implications

**1. Transaction Substitution Attack**

\`\`\`python
class TransactionSubstitutionAttack:
    """
    Demonstrate the attack vector
    """
    def execute_attack(self):
        # Step 1: Create legitimate block
        legitimate_block = {
            'transactions': [
                {'from': 'Alice', 'to': 'Bob', 'amount': 1},
                {'from': 'Carol', 'to': 'Dave', 'amount': 0.5}
            ],
            'merkle_root': '0x1234...abcd'  # Computed from legit txs
        }
        
        # Step 2: Mine the block (POW or POS)
        # Block is accepted by network
        
        # Step 3: After block is accepted, substitute transactions
        malicious_block = {
            'transactions': [
                {'from': 'Alice', 'to': 'Attacker', 'amount': 1000},
                {'from': 'Carol', 'to': 'Attacker', 'amount': 999}
            ],
            'merkle_root': '0x1234...abcd'  # SAME ROOT due to bug!
        }
        
        # Step 4: Distribute malicious version
        # Nodes verify: "Merkle root matches? ✓ Block valid!"
        # But transactions are completely different!
        
        return {
            'attack_type': 'Transaction Substitution',
            'severity': 'CRITICAL - Total loss of transaction integrity',
            'impact': 'Attacker can rewrite any block\'s transactions',
            'funds_at_risk': 'ENTIRE BLOCKCHAIN VALUE'
        }
\`\`\`

**2. Double Spend Attack**

\`\`\`python
def double_spend_via_merkle_collision():
    """
    Use Merkle collision for double spending
    """
    # Original transaction: Pay merchant
    original_txs = [
        {'from': 'Attacker', 'to': 'Merchant', 'amount': 100},
        # ... other txs
    ]
    
    # Get merchant to ship goods based on this block
    
    # After goods shipped, substitute transactions
    malicious_txs = [
        {'from': 'Attacker', 'to': 'Attacker', 'amount': 100},  # Pay self back!
        # ... other txs
    ]
    
    # Same Merkle root due to bug
    # Merchant loses goods, attacker keeps money
    
    impact = {
        'merchant_loss': 'Shipped goods worth 100 BTC',
        'attacker_gain': 'Kept 100 BTC + goods',
        'blockchain_status': 'Accepts malicious block as valid'
    }
    return impact
\`\`\`

**3. Historical Rewrite**

\`\`\`python
def historical_rewrite_attack():
    """
    Rewrite historical transactions
    """
    # Find old blocks with vulnerable Merkle roots
    vulnerable_blocks = find_blocks_with_collision_vulnerability()
    
    for block in vulnerable_blocks:
        # Create alternative transaction set with same Merkle root
        alternative_txs = craft_collision_transactions(
            original_txs=block.transactions,
            target_merkle_root=block.merkle_root
        )
        
        # Distribute alternative history
        # Nodes can't distinguish which transaction set is correct!
        
    result = {
        'impact': 'Entire blockchain history is now ambiguous',
        'trust_loss': 'Complete - no one knows true transaction history',
        'recovery': 'Potentially impossible'
    }
    return result
\`\`\`

### How Attacker Exploits This

**Exploitation Timeline**:

\`\`\`python
class MerkleCollisionExploit:
    """
    Step-by-step attack execution
    """
    def __init__(self):
        self.blockchain = VulnerableBlockchain()
    
    def phase_1_discovery(self):
        """
        Phase 1: Discover the vulnerability
        """
        print("PHASE 1: VULNERABILITY DISCOVERY")
        
        # Test Merkle implementation
        txs1 = ["tx_a", "tx_b"]
        txs2 = ["tx_c", "tx_d"]
        
        root1 = self.blockchain.compute_merkle_root(txs1)
        root2 = self.blockchain.compute_merkle_root(txs2)
        
        # Try to find collision
        if self.try_find_collision(txs1, txs2):
            print("✓ Collision found! Merkle tree is vulnerable")
            return True
        
        return False
    
    def phase_2_weaponization(self):
        """
        Phase 2: Create collision generator
        """
        print("\\nPHASE 2: WEAPONIZATION")
        
        def generate_collision(original_txs: list) -> list:
            """
            Generate alternative transaction set with same Merkle root
            """
            # Exploit the specific bug in implementation
            if bug_type == "leaf_node_confusion":
                return self.craft_leaf_node_collision(original_txs)
            elif bug_type == "length_extension":
                return self.craft_length_extension_collision(original_txs)
            elif bug_type == "duplicate_handling":
                return self.craft_duplicate_collision(original_txs)
        
        print("✓ Collision generator created")
        return generate_collision
    
    def phase_3_high_value_target(self):
        """
        Phase 3: Target high-value transaction
        """
        print("\\nPHASE 3: TARGET SELECTION")
        
        # Find blocks with large transactions
        target_block = self.find_high_value_block()
        
        print(f"Target block: #{target_block.number}")
        print(f"Total value: {target_block.total_value} BTC")
        
        # Create malicious transaction set
        malicious_txs = self.create_malicious_transactions(
            original=target_block.transactions,
            redirect_to='attacker_address'
        )
        
        # Verify collision
        assert (self.blockchain.compute_merkle_root(target_block.transactions) == 
                self.blockchain.compute_merkle_root(malicious_txs))
        
        print("✓ Malicious block prepared with same Merkle root")
        return malicious_txs
    
    def phase_4_execution(self, malicious_txs):
        """
        Phase 4: Execute the attack
        """
        print("\\nPHASE 4: ATTACK EXECUTION")
        
        # Wait for legitimate block to be confirmed
        # (e.g., merchant ships goods, exchange credits deposit)
        wait_for_confirmations(target_block, confirmations=6)
        
        # Broadcast malicious block to network
        self.broadcast_alternative_block(malicious_txs)
        
        # Network sees:
        # - Same block hash ✓
        # - Same Merkle root ✓
        # - Valid POW/POS ✓
        # - Different transactions ✗ (but can't detect!)
        
        print("✓ Attack executed - transactions substituted")
        
    def phase_5_profit(self):
        """
        Phase 5: Profit and chaos
        """
        print("\\nPHASE 5: CONSEQUENCES")
        
        consequences = {
            'attacker_profit': 'Stolen funds + goods received',
            'victim_loss': 'Funds stolen + goods shipped',
            'blockchain_integrity': 'DESTROYED',
            'trust': 'ZERO',
            'token_value': 'Crashes to near zero',
            'recovery': 'Potentially impossible'
        }
        
        return consequences
\`\`\`

### Emergency Measures (Immediate Response)

**Hour 1: Emergency Response**

\`\`\`python
class EmergencyResponse:
    """
    Immediate actions when vulnerability discovered
    """
    def hour_1_immediate_response(self):
        # 1. HALT THE CHAIN
        print("STEP 1: EMERGENCY CHAIN HALT")
        self.broadcast_emergency_alert()
        self.stop_block_production()
        self.freeze_major_exchanges()
        
        # 2. ASSESS DAMAGE
        print("\\nSTEP 2: DAMAGE ASSESSMENT")
        exploited_blocks = self.scan_for_exploitation()
        
        assessment = {
            'vulnerability_confirmed': True,
            'blocks_potentially_affected': 'ALL',
            'exploitation_detected': len(exploited_blocks) > 0,
            'funds_at_risk': 'ENTIRE CHAIN VALUE'
        }
        
        # 3. EMERGENCY COMMUNICATIONS
        print("\\nSTEP 3: COMMUNICATIONS")
        self.notify_validators()
        self.notify_exchanges()
        self.notify_users()
        self.notify_security_researchers()
        
        # 4. FREEZE ASSETS
        print("\\nSTEP 4: ASSET FREEZE")
        self.request_exchange_trading_halt()
        self.request_withdrawal_freeze()
        
        return assessment
\`\`\`

**Day 1-3: Investigation and Fix**

\`\`\`python
def emergency_fix_protocol():
    """
    Fix the Merkle tree implementation
    """
    # 1. IDENTIFY ROOT CAUSE
    print("INVESTIGATION: Root Cause Analysis")
    
    # Original buggy code
    def buggy_merkle():
        # Leaf/node confusion
        return hashlib.sha256((left + right).encode()).hexdigest()
    
    # 2. IMPLEMENT FIX
    print("\\nFIX: Secure Merkle Implementation")
    
    def fixed_merkle_hash(left: bytes, right: bytes, is_leaf: bool = False) -> bytes:
        """
        Secure Merkle hashing with domain separation
        """
        if is_leaf:
            # Prefix 0x00 for leaves
            data = b'\\x00' + left
        else:
            # Prefix 0x01 for internal nodes
            data = b'\\x01' + left + right
        
        return hashlib.sha256(data).digest()
    
    # This prevents leaf/node confusion attacks
    # Different domains (leaf vs internal) can't collide
    
    # 3. ADDITIONAL PROTECTIONS
    def hardened_merkle():
        """
        Additional security measures
        """
        # a) Use double hashing (like Bitcoin)
        def double_hash(data: bytes) -> bytes:
            return hashlib.sha256(hashlib.sha256(data).digest()).digest()
        
        # b) Include length prefixes
        def length_prefixed_hash(data: bytes) -> bytes:
            length = len(data).to_bytes(8, 'big')
            return hashlib.sha256(length + data).digest()
        
        # c) Use standard implementations (don't roll your own!)
        from merkletools import MerkleTools  # Use audited library
        
        return {
            'double_hashing': double_hash,
            'length_prefixes': length_prefixed_hash,
            'use_library': 'Always use audited implementations'
        }
    
    return fixed_merkle_hash, hardened_merkle()
\`\`\`

**Week 1-2: Recovery Plan**

\`\`\`python
def recovery_options():
    """
    Possible recovery strategies
    """
    options = {
        'Option_1_Hard_Fork_Rollback': {
            'description': 'Roll back to block before any exploitation',
            'pros': [
                'Removes all fraudulent transactions',
                'Restores integrity of historical data'
            ],
            'cons': [
                'Legitimate transactions after rollback point are lost',
                'Community may split (some don\'t accept rollback)',
                'Sets precedent for chain intervention'
            ],
            'example': 'Ethereum DAO hack response (2016)'
        },
        
        'Option_2_Hard_Fork_With_Fix': {
            'description': 'Deploy fix, continue from current state',
            'pros': [
                'No transaction history lost',
                'Faster recovery'
            ],
            'cons': [
                'Fraudulent transactions remain in history',
                'Lost funds cannot be recovered',
                'Trust permanently damaged'
            ],
            'viability': 'Only if no exploitation detected'
        },
        
        'Option_3_New_Chain': {
            'description': 'Start completely new blockchain',
            'pros': [
                'Clean slate',
                'New token (old token becomes worthless)'
            ],
            'cons': [
                'All holders lose value',
                'Complete loss of trust',
                'Basically admitting total failure'
            ],
            'when': 'If exploitation is widespread and unrecoverable'
        }
    }
    
    # Decision matrix
    if exploitation_widespread and funds_stolen:
        return options['Option_1_Hard_Fork_Rollback']
    elif exploitation_detected but contained:
        return options['Option_1_Hard_Fork_Rollback']  # Still safest
    elif no_exploitation_yet:
        return options['Option_2_Hard_Fork_With_Fix']
    else:
        return options['Option_3_New_Chain']  # Nuclear option
\`\`\`

### Prevention: How to Avoid This Category of Bugs

**1. Use Standard, Audited Libraries**

\`\`\`python
# DON'T: Roll your own Merkle tree
class MyMerkleTree:
    # Custom implementation - likely has bugs
    pass

# DO: Use proven libraries
from merkletools import MerkleTools
from eth_utils import keccak

# Or better yet: Use blockchain framework's built-in implementation
from bitcoinlib import MerkleTree  # Bitcoin-compatible
from web3 import Web3  # Ethereum-compatible

# These have been battle-tested on live blockchains worth billions
\`\`\`

**2. Implement Domain Separation**

\`\`\`python
def secure_merkle_implementation():
    """
    Always use domain separation
    """
    # Different prefixes for different data types
    LEAF_PREFIX = b'\\x00'
    INTERNAL_PREFIX = b'\\x01'
    
    def hash_leaf(data: bytes) -> bytes:
        return hashlib.sha256(LEAF_PREFIX + data).digest()
    
    def hash_internal(left: bytes, right: bytes) -> bytes:
        return hashlib.sha256(INTERNAL_PREFIX + left + right).digest()
    
    # Now leaf and internal node hashes are in different "domains"
    # Impossible to confuse them or create collisions between them
\`\`\`

**3. Use Double Hashing**

\`\`\`python
def bitcoin_style_hashing(data: bytes) -> bytes:
    """
    Double SHA-256 like Bitcoin
    Protects against length extension and other attacks
    """
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()
\`\`\`

**4. Include Length Commitments**

\`\`\`python
def length_committed_hash(data: bytes) -> bytes:
    """
    Include data length in hash
    Prevents certain collision attacks
    """
    length = len(data).to_bytes(8, 'big')
    return hashlib.sha256(length + data).digest()
\`\`\`

**5. Comprehensive Testing**

\`\`\`python
class MerkleTreeSecurityTests:
    """
    Security test suite for Merkle trees
    """
    def test_collision_resistance(self):
        """Test that different transactions produce different roots"""
        for i in range(100000):
            txs1 = generate_random_transactions()
            txs2 = generate_different_random_transactions()
            
            root1 = compute_merkle_root(txs1)
            root2 = compute_merkle_root(txs2)
            
            assert root1 != root2, f"COLLISION FOUND at iteration {i}!"
    
    def test_leaf_node_separation(self):
        """Test that leaves and nodes are properly separated"""
        # Try to craft a leaf that matches an internal node hash
        # Should be impossible
        pass
    
    def test_tamper_detection(self):
        """Test that any transaction modification changes root"""
        original_txs = ["tx1", "tx2", "tx3"]
        original_root = compute_merkle_root(original_txs)
        
        # Modify single transaction
        tampered_txs = ["tx1_modified", "tx2", "tx3"]
        tampered_root = compute_merkle_root(tampered_txs)
        
        assert original_root != tampered_root
    
    def test_proof_verification(self):
        """Test that proofs work correctly"""
        txs = generate_test_transactions()
        tree = MerkleTree(txs)
        
        for i, tx in enumerate(txs):
            proof = tree.get_proof(i)
            assert tree.verify_proof(tx, i, proof)
            
            # Tampered transaction should fail
            tampered = tx + "_modified"
            assert not tree.verify_proof(tampered, i, proof)
\`\`\`

**6. Security Audits**

\`\`\`python
security_audit_checklist = {
    'Pre-Launch': [
        'External security audit by reputable firm',
        'Bug bounty program ($100k+ for critical bugs)',
        'Formal verification of hash functions if possible',
        'Peer review by cryptographers',
        'Open source code for community review'
    ],
    
    'Post-Launch': [
        'Continuous monitoring for anomalies',
        'Regular security audits (quarterly)',
        'Ongoing bug bounty program',
        'Incident response plan ready',
        'Emergency halt mechanism'
    ]
}
\`\`\`

### Conclusion

This vulnerability is **catastrophic** because it violates collision resistance, a fundamental property required for blockchain security. It allows attackers to:

1. Substitute transactions in blocks
2. Execute double-spends
3. Rewrite historical transactions
4. Destroy all trust in the blockchain

**Emergency response requires**:
- Immediate chain halt
- Rapid deployment of fix
- Likely hard fork to remove exploitation
- Massive loss of trust and token value

**Prevention requires**:
- Never roll your own crypto
- Use standard, audited libraries
- Implement domain separation
- Comprehensive security testing
- Multiple security audits
- Bug bounty programs

**The real lesson**: Cryptographic primitives are **hard to get right**. Even tiny implementation bugs can have catastrophic consequences. Always use proven, battle-tested implementations from established blockchains.

In blockchain, **boring is good**. Novel cryptographic implementations are a red flag. Copy Bitcoin or Ethereum's implementations—they've been tested with hundreds of billions of dollars at stake.
`,
  },
  {
    id: 3,
    question:
      "You're analyzing a blockchain that claims to have 'solved the blockchain trilemma' by using a novel hash function called 'FastHash-512' that is '100x faster than SHA-256 while providing 512 bits of security.' The whitepaper provides impressive benchmarks but limited cryptographic analysis. As the technical due diligence lead for a $50M investment, what specific questions would you ask about this hash function? What tests would you run? What red flags would make you reject the investment? How do you balance innovation vs. security when evaluating novel cryptographic primitives in blockchain projects?",
    answer: `## Comprehensive Answer:

This scenario is a **massive red flag festival** that I would bet heavily against. Let me explain why "novel hash functions" in blockchain projects are almost always a terrible idea, and how to evaluate cryptographic claims properly.

### Immediate Red Flags 🚩🚩🚩

**Red Flag #1: "Novel Hash Function"**

\`\`\`python
red_flag_analysis = {
    'Claim': 'New hash function FastHash-512',
    'Reality': 'Cryptography should be BORING',
    'Concern': 'Why not use SHA-256/SHA-3/BLAKE2?',
    'Probability_of_disaster': '85%'
}

# Historical precedent:
failed_novel_hashes = {
    'MD5': {
        'year_introduced': 1991,
        'year_broken': 2004,
        'time_to_break': '13 years',
        'impact': 'Massive - used everywhere'
    },
    'SHA-1': {
        'year_introduced': 1995,
        'year_broken': 2017,
        'time_to_break': '22 years',
        'impact': 'Git, SSL certificates, etc.'
    },
    'Custom_hashes_in_cryptos': {
        'examples': ['Groestl (Groestlcoin)', 'X11 (Dash)', 'Scrypt variations'],
        'issues': 'Less cryptanalysis, smaller security margins',
        'value_destroyed': 'Hundreds of millions in exploits'
    }
}

# The pattern: Even hash functions designed by experts take 10-20 years to break
# A novel function has near-zero chance of being secure long-term
\`\`\`

**Red Flag #2: "100x Faster"**

\`\`\`python
performance_reality_check = {
    'Claim': '100x faster than SHA-256',
    
    'Questions': [
        'Faster how? CPU? GPU? ASIC?',
        'SHA-256 already does 500+ MB/s per core',
        'What optimization allows 100x improvement?',
        'If it\'s really that fast, what security trade-off was made?'
    ],
    
    'Cryptographic_principle': 'There is no free lunch',
    
    'Reality': """
    You cannot be 100x faster without sacrificing something:
    - Fewer rounds (weaker security)
    - Simpler operations (easier to attack)
    - Smaller state (reduced collision resistance)
    - Parallelization (doesn't help most blockchain uses)
    
    If it sounds too good to be true, it definitely is.
    """
}

# Real performance comparison:
import time
import hashlib

def benchmark_real_hash_functions():
    data = b"x" * 1024 * 1024  # 1 MB
    iterations = 1000
    
    # SHA-256 benchmark
    start = time.time()
    for _ in range(iterations):
        hashlib.sha256(data).digest()
    sha256_time = time.time() - start
    
    print(f"SHA-256: {iterations / sha256_time:.0f} hashes/second")
    print(f"Throughput: {len(data) * iterations / sha256_time / 1024 / 1024:.0f} MB/s")
    
    # A "100x faster" hash would need to do:
    # 50,000+ MB/s = 50 GB/s
    # This is approaching memory bandwidth limits!
    # Highly suspicious

# If someone claims 100x speedup, they're either:
# 1. Lying
# 2. Comparing apples to oranges (GPU vs CPU)
# 3. Made massive security trade-offs
# 4. Don't understand cryptography
\`\`\`

**Red Flag #3: "512 Bits of Security"**

\`\`\`python
def analyze_512_bit_claim():
    """
    More bits doesn't automatically mean more security
    """
    security_analysis = {
        'Claim': '512 bits of security',
        
        'Reality_check': {
            'SHA-256': {
                'output_bits': 256,
                'collision_resistance': '2^128 operations (birthday attack)',
                'preimage_resistance': '2^256 operations',
                'actual_security': '128-bit security level (plenty!)'
            },
            
            'SHA-512': {
                'output_bits': 512,
                'collision_resistance': '2^256 operations',
                'preimage_resistance': '2^512 operations',
                'actual_security': '256-bit security level (overkill)',
                'note': 'Slower than SHA-256 on 32-bit systems'
            },
            
            'FastHash-512': {
                'output_bits': 512,
                'actual_security': 'UNKNOWN - not analyzed',
                'concern': 'Larger output doesn\'t mean better security'
            }
        },
        
        'Questions': [
            'What does "512 bits of security" mean exactly?',
            '512-bit output? Or 512-bit security level?',
            'Do they understand birthday paradox?',
            'Security level is typically output_bits / 2 for collisions',
            'SHA-256 provides 128-bit security - plenty for any use case'
        ],
        
        'Verdict': 'Marketing buzzword, not meaningful security claim'
    }
    
    # Real security requirements:
    security_requirements = {
        '80-bit security': 'Minimum acceptable (2^80 ops)',
        '128-bit security': 'Current standard (SHA-256 collision resistance)',
        '256-bit security': 'Post-quantum resistant (overkill for now)',
        '512-bit security': 'Meaningless marketing'
    }
    
    return security_analysis
\`\`\`

### Critical Questions to Ask

**1. Cryptographic Design Questions**

\`\`\`python
cryptographic_questions = [
    {
        'question': 'Who designed this hash function?',
        'acceptable_answers': [
            'Published cryptographers with h-index > 50',
            'Team that designed SHA-3 (Keccak)',
            'NIST competition winners'
        ],
        'red_flags': [
            'Our internal team',
            'A blockchain consultant',
            'Anonymous developer',
            'No cryptographic credentials listed'
        ]
    },
    
    {
        'question': 'What is the underlying construction?',
        'acceptable_answers': [
            'Merkle-Damgård construction (like SHA-2)',
            'Sponge construction (like SHA-3)',
            'HAIFA construction',
            'With clear security proof'
        ],
        'red_flags': [
            'Novel construction',
            'Trade secret / proprietary',
            'Based on XOR operations',
            'Can\'t explain clearly'
        ]
    },
    
    {
        'question': 'How many rounds of compression?',
        'context': 'SHA-256 has 64 rounds, SHA-3 has 24 rounds',
        'red_flags': [
            'Fewer than 16 rounds',
            '"Adaptive rounds" (smell test)',
            'Don\'t know / won\'t say'
        ]
    },
    
    {
        'question': 'What is the security margin?',
        'explanation': """
        Best attack should require >> 50% of ideal attack cost
        Example: SHA-256 best attack is 2^128 (ideal) vs practical = still 2^128
        Security margin is the gap between best attack and ideal
        """,
        'acceptable': 'Best known attack requires > 90% of ideal cost',
        'red_flags': 'No analysis of security margin'
    },
    
    {
        'question': 'What peer review has it received?',
        'gold_standard': [
            'Published in CRYPTO / EUROCRYPT / ASIACRYPT conferences',
            'NIST competition process (years of analysis)',
            'Multiple independent cryptanalysis papers'
        ],
        'acceptable': [
            'Academic papers by independent researchers',
            'Public for > 5 years with no attacks found'
        ],
        'unacceptable': [
            'Reviewed by our advisors',
            'Posted on GitHub 6 months ago',
            'Whitepaper only'
        ]
    }
]
\`\`\`

**2. Implementation and Testing Questions**

\`\`\`python
implementation_questions = [
    {
        'question': 'Show me the reference implementation',
        'must_have': [
            'Open source (fully auditable)',
            'Clean, readable code',
            'Comprehensive test vectors',
            'Comparison with known hash functions'
        ],
        'red_flags': [
            'Proprietary / closed source',
            'Obfuscated code',
            'No test vectors provided',
            '"We\'ll open source it later"'
        ]
    },
    
    {
        'question': 'What testing have you done?',
        'minimum_acceptable': [
            'Statistical randomness tests (NIST suite)',
            'Avalanche effect analysis',
            'Collision search (birthday attack simulation)',
            'Differential cryptanalysis resistance',
            'Linear cryptanalysis resistance'
        ],
        'red_flags': [
            '"It passes our internal tests"',
            'Only performance benchmarks',
            'No formal cryptanalysis',
            '"We don\'t have time for extensive testing"'
        ]
    },
    
    {
        'question': 'Where are the benchmarks from?',
        'valid_benchmarks': [
            'Reproducible on standard hardware',
            'Multiple independent benchmarkers',
            'Compared against standard implementations (OpenSSL)',
            'Published methodology'
        ],
        'red_flags': [
            'Custom hardware',
            'Unreproducible results',
            'Compared against worst-case scenarios',
            'Cherry-picked metrics'
        ]
    }
]
\`\`\`

**3. Blockchain Integration Questions**

\`\`\`python
integration_questions = [
    {
        'question': 'Why not use SHA-256, SHA-3, or BLAKE2?',
        'acceptable_reasons': [
            # Honestly, there are almost NO acceptable reasons
            'Post-quantum resistance (but use established PQ hash)',
            'NIST competition winner for specific use case'
        ],
        'unacceptable_reasons': [
            'SHA-256 is too slow',  # It's not
            'We want to be different',  # Terrible reason
            'ASICs centralize mining',  # Different problem
            'Quantum computers',  # SHA-256 is quantum-resistant for collisions
            'Innovation'  # Not a valid security reason
        ]
    },
    
    {
        'question': 'What happens if a vulnerability is found?',
        'must_have': [
            'Hard fork plan documented',
            'Ability to migrate to SHA-256',
            'Emergency response process',
            'Bug bounty program ($500k+ for hash collision)'
        ],
        'red_flags': [
            'We believe it\'s secure (not an answer)',
            'No migration plan',
            'Will address if it happens'
        ]
    },
    
    {
        'question': 'Who else uses this hash function?',
        'best_case': [
            'Multiple blockchains (billions of dollars at stake)',
            'Years of production use',
            'No exploits ever found'
        ],
        'red_flags': [
            'We\'re the first!',  # TERRIBLE
            'It\'s too new for others to adopt',
            'Other projects are watching us',
            'We have patents on it'  # Hash functions shouldn\'t be patented
        ]
    }
]
\`\`\`

### Tests I Would Run

**Test 1: Basic Sanity Checks**

\`\`\`python
import hashlib
import time
import os

class HashFunctionTester:
    """
    Basic security tests for hash functions
    """
    def __init__(self, hash_function):
        self.hash = hash_function
    
    def test_determinism(self):
        """Same input should always produce same output"""
        data = b"test data"
        results = [self.hash(data) for _ in range(1000)]
        assert len(set(results)) == 1, "Hash function is non-deterministic!"
        print("✓ Determinism test passed")
    
    def test_avalanche_effect(self):
        """Single bit change should flip ~50% of output bits"""
        data1 = os.urandom(1024)
        data2 = bytearray(data1)
        data2[0] ^= 1  # Flip single bit
        
        hash1 = int.from_bytes(self.hash(bytes(data1)), 'big')
        hash2 = int.from_bytes(self.hash(bytes(data2)), 'big')
        
        xor = hash1 ^ hash2
        bits_different = bin(xor).count('1')
        total_bits = len(self.hash(data1)) * 8
        percentage = bits_different / total_bits * 100
        
        print(f"Avalanche effect: {percentage:.1f}% bits changed")
        assert 45 <= percentage <= 55, f"Poor avalanche effect: {percentage}%"
        print("✓ Avalanche effect test passed")
    
    def test_collision_resistance_weak(self):
        """Try to find collisions in limited search space"""
        hashes = {}
        attempts = 1_000_000
        
        for i in range(attempts):
            data = os.urandom(32)
            h = self.hash(data)
            
            if h in hashes:
                print(f"✗ COLLISION FOUND after {i} attempts!")
                print(f"Data1: {hashes[h].hex()}")
                print(f"Data2: {data.hex()}")
                print(f"Hash: {h.hex()}")
                return False
            
            hashes[h] = data
        
        print(f"✓ No collisions in {attempts:,} attempts")
        return True
    
    def test_distribution(self):
        """Output should be uniformly distributed"""
        # Test first byte distribution
        counts = [0] * 256
        samples = 100_000
        
        for _ in range(samples):
            data = os.urandom(32)
            h = self.hash(data)
            counts[h[0]] += 1
        
        # Expected: ~390 occurrences per byte value
        expected = samples / 256
        max_deviation = max(abs(c - expected) for c in counts)
        
        print(f"Distribution test: max deviation = {max_deviation:.1f}")
        assert max_deviation < expected * 0.2, "Poor distribution"
        print("✓ Distribution test passed")
    
    def test_performance(self):
        """Verify performance claims"""
        data = b"x" * (1024 * 1024)  # 1 MB
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            self.hash(data)
        elapsed = time.time() - start
        
        throughput_mbps = (len(data) * iterations) / elapsed / (1024 * 1024)
        hashes_per_second = iterations / elapsed
        
        print(f"\\nPerformance:")
        print(f"  Throughput: {throughput_mbps:.0f} MB/s")
        print(f"  Hashes/sec: {hashes_per_second:.0f}")
        
        # Compare to SHA-256
        start = time.time()
        for _ in range(iterations):
            hashlib.sha256(data).digest()
        sha256_time = time.time() - start
        
        speedup = sha256_time / elapsed
        print(f"  Speedup vs SHA-256: {speedup:.1f}x")
        
        if speedup > 10:
            print("  ⚠ Suspiciously fast - likely security trade-offs")
        
        return throughput_mbps, speedup

# Test the "FastHash-512"
# (In reality, this would fail spectacularly)
\`\`\`

**Test 2: Cryptanalysis Tests**

\`\`\`python
class AdvancedCryptanalysis:
    """
    More advanced security tests
    """
    def test_length_extension_vulnerability(self, hash_func):
        """
        Test if vulnerable to length extension attacks
        """
        # SHA-256 is vulnerable, SHA-3 is not
        # Novel function: unknown
        
        original = b"known data"
        original_hash = hash_func(original)
        
        # Try to extend without knowing original data
        # (This is simplified - real attack is more complex)
        extended = original + b"\\x80" + b"malicious data"
        extended_hash = hash_func(extended)
        
        # Check if we can compute extended_hash from just original_hash
        # If yes: VULNERABLE
        
        print("Length extension test: Analysis required")
    
    def test_weak_internal_state(self, hash_func):
        """
        Test for weak internal state
        """
        # Generate many hashes, look for patterns
        hashes = []
        for i in range(100_000):
            data = i.to_bytes(8, 'big')
            h = hash_func(data)
            hashes.append(h)
        
        # Analyze for patterns (requires statistical analysis)
        # Linear relationships, modular arithmetic patterns, etc.
        
        print("Internal state test: Statistical analysis required")
    
    def test_differential_cryptanalysis(self, hash_func):
        """
        Test resistance to differential attacks
        """
        # Create pairs of inputs with specific differences
        # Analyze if output differences are predictable
        
        differences = []
        for _ in range(10_000):
            data1 = os.urandom(64)
            data2 = bytearray(data1)
            data2[0] ^= 0x01  # Specific bit flip
            
            hash1 = hash_func(bytes(data1))
            hash2 = hash_func(bytes(data2))
            
            diff = bytes(a ^ b for a, b in zip(hash1, hash2))
            differences.append(diff)
        
        # Look for non-random patterns in differences
        # (Requires expert cryptanalysis)
        
        print("Differential cryptanalysis: Expert analysis required")
\`\`\`

**Test 3: Real-World Attack Simulation**

\`\`\`python
def simulate_blockchain_attack(hash_func):
    """
    Simulate actual blockchain attack scenarios
    """
    print("\\n=== BLOCKCHAIN ATTACK SIMULATIONS ===")
    
    # Attack 1: Transaction collision
    print("\\n1. Transaction Collision Attack")
    try:
        tx1 = "Alice pays Bob 1 BTC"
        tx2 = "Alice pays Attacker 100 BTC"
        
        # Try to find collision (simplified)
        for nonce in range(10_000_000):
            modified_tx2 = f"{tx2}, nonce: {nonce}"
            if hash_func(tx1.encode()) == hash_func(modified_tx2.encode()):
                print(f"✗ CRITICAL: Collision found at nonce {nonce}!")
                print(f"   Attacker can substitute transactions!")
                return False
        
        print("✓ No collision found (10M attempts)")
    except Exception as e:
        print(f"✗ Error in test: {e}")
        return False
    
    # Attack 2: Block hash manipulation
    print("\\n2. Block Hash Manipulation")
    block_data = "Block #12345, Prev: 0000abc..., Txs: [...]"
    
    # Try to find second preimage
    target_hash = hash_func(block_data.encode())
    
    for nonce in range(1_000_000):
        fake_block = f"Fake block {nonce}"
        if hash_func(fake_block.encode()) == target_hash:
            print(f"✗ CRITICAL: Second preimage found!")
            return False
    
    print("✓ No second preimage found (1M attempts)")
    
    # Attack 3: Mining difficulty bypass
    print("\\n3. Mining Difficulty Bypass")
    target = "0000"  # 4 leading zeros
    
    for nonce in range(100_000):
        candidate = f"{block_data}, nonce: {nonce}"
        h = hash_func(candidate.encode()).hex()
        
        if h.startswith(target):
            print(f"✓ Valid block found after {nonce} attempts")
            # Check if finding rate matches expected
            expected = 16**len(target)
            if nonce < expected * 0.001:  # Found way too easily
                print(f"⚠ WARNING: Mining too easy, hash function may be weak")
            break
    
    return True
\`\`\`

### Investment Decision Framework

\`\`\`python
class InvestmentDecision:
    """
    Decision framework for $50M investment
    """
    def __init__(self):
        self.red_flags = []
        self.concerns = []
        self.score = 100  # Start at 100, subtract for each issue
    
    def evaluate(self, project):
        print("=== INVESTMENT EVALUATION ===\\n")
        
        # Automatic disqualifiers (score = 0)
        disqualifiers = [
            ('Novel hash function with < 5 years public scrutiny', -100),
            ('No peer-reviewed cryptanalysis', -100),
            ('Closed source cryptographic code', -100),
            ('Designer has no cryptographic credentials', -100),
            ('Claims "10x" or greater performance improvement', -50),
            ('No major blockchain uses same hash', -40),
            ('Failed basic collision resistance tests', -100),
            ('No bug bounty program', -30),
            ('Cannot explain why not using SHA-256/SHA-3', -50)
        ]
        
        for issue, penalty in disqualifiers:
            if self.check_issue(project, issue):
                self.red_flags.append(issue)
                self.score += penalty
                print(f"✗ {issue} (penalty: {penalty})")
        
        print(f"\\nFinal Score: {max(0, self.score)}/100")
        
        if self.score <= 0:
            print("\\n🚫 RECOMMENDATION: DO NOT INVEST")
            print("   Critical cryptographic risks detected")
            return False
        elif self.score < 50:
            print("\\n⚠ RECOMMENDATION: HIGH RISK")
            print("   Significant concerns, not worth $50M")
            return False
        elif self.score < 75:
            print("\\n⚠ RECOMMENDATION: PROCEED WITH CAUTION")
            print("   Additional due diligence required")
            return "maybe"
        else:
            print("\\n✓ RECOMMENDATION: ACCEPTABLE RISK")
            return True
    
    def check_issue(self, project, issue):
        # In reality, would check actual project details
        # For FastHash-512, likely fails most tests
        return True  # Assume all issues present

# For "FastHash-512" project:
evaluator = InvestmentDecision()
decision = evaluator.evaluate(fast_hash_project)

if decision == False:
    print("\\nDo not invest $50M in this project.")
    print("Novel cryptographic primitives = massive risk")
    print("Historical precedent: Most novel hashes are eventually broken")
\`\`\`

### Balancing Innovation vs. Security

**The Hard Truth**:

\`\`\`python
innovation_vs_security = {
    'In most domains': {
        'innovation': 'Good',
        'rapid_iteration': 'Good',
        'move_fast': 'Good',
        'break_things': 'OK to fix later'
    },
    
    'In cryptography': {
        'innovation': 'DANGEROUS',
        'rapid_iteration': 'DEADLY',
        'move_fast': 'How to lose billions',
        'break_things': 'Catastrophic and irreversible'
    }
}

cryptography_principle = """
In cryptography, boring is good.
Novel is dangerous.
Proven is essential.
Fast-moving is a red flag.

Reason: Cryptographic breaks are:
- Often undetectable until exploit
- Completely irreversible
- Can destroy entire ecosystems
- Take years to discover

Time-tested cryptography has survived:
- Academic scrutiny
- Nation-state attacks
- Financial incentives for breaking
- Quantum computing concerns
"""
\`\`\`

**When Novel Crypto is Acceptable**:

\`\`\`python
acceptable_novel_crypto = {
    'Scenario_1': {
        'description': 'NIST competition winner',
        'example': 'SHA-3 (Keccak)',
        'reason': 'Years of public cryptanalysis by world experts',
        'timeline': '5-10 years of review before adoption'
    },
    
    'Scenario_2': {
        'description': 'Designed by world-renowned cryptographers',
        'example': 'ChaCha20 (Daniel Bernstein)',
        'reason': 'Designer has decades of proven work',
        'caveat': 'Still requires years of review'
    },
    
    'Scenario_3': {
        'description': 'Proven in other blockchains worth billions',
        'example': 'Keccak-256 (Ethereum)',
        'reason': 'Battle-tested with massive economic incentive',
        'minimum': 'Must have $1B+ at stake for 3+ years'
    },
    
    'Scenario_4': {
        'description': 'Post-quantum requirements',
        'example': 'Using NIST PQC competition winners',
        'reason': 'Legitimate need for quantum resistance',
        'note': 'But use competition winners, not custom designs'
    }
}

# FastHash-512 doesn't meet ANY of these criteria
# Therefore: Hard pass
\`\`\`

### My Final Recommendation

\`\`\`python
def final_investment_decision():
    """
    Final recommendation for $50M investment
    """
    decision = {
        'invest': False,
        'confidence': '99%',
        
        'reasons': [
            '1. Novel hash function is massive red flag',
            '2. No peer-reviewed cryptanalysis',
            '3. "100x faster" claim is unrealistic',
            '4. No other blockchain uses it (you\'d be guinea pig)',
            '5. Hash functions take 10-20 years to prove secure',
            '6. No legitimate reason not to use SHA-256/SHA-3',
            '7. Billions of dollars at risk if hash breaks'
        ],
        
        'alternative_recommendation': """
        Tell the project:
        "Use SHA-256 like Bitcoin, or Keccak-256 like Ethereum.
        
        If you want better performance:
        - Use BLAKE2b (well-studied, faster than SHA-256)
        - Optimize your consensus mechanism instead
        - Improve network layer
        
        Do NOT use custom cryptography.
        
        If you insist on FastHash-512:
        - Submit to NIST for 5+ years of review
        - Get independent cryptanalysis by top researchers
        - Prove it on testnet with bug bounty for 2+ years
        - Come back in 2030
        
        Otherwise, we cannot invest."
        """,
        
        'reality': """
        Any blockchain project using novel cryptographic primitives
        without 5+ years of public scrutiny is not investment-grade.
        
        Period.
        
        I don't care how good the team is.
        I don't care about the performance benchmarks.
        I don't care about the innovative consensus mechanism.
        
        If the foundation is novel cryptography, the investment is trash.
        
        Crypto is hard. Really hard.
        Even experts get it wrong.
        Don't bet $50M on untested crypto.
        """
    }
    
    return decision

# Execute decision
verdict = final_investment_decision()
print(verdict['alternative_recommendation'])
\`\`\`

### Conclusion

**For FastHash-512 investment**: **HARD NO**

**Red flags identified**:
1. ✗ Novel hash function
2. ✗ Unrealistic performance claims (100x)
3. ✗ Likely no serious cryptanalysis
4. ✗ No other major blockchain uses it
5. ✗ No compelling reason vs. SHA-256/SHA-3

**Tests that would likely fail**:
1. Peer review check
2. Designer credentials
3. Independent cryptanalysis
4. Production usage elsewhere
5. Justification vs. standard hashes

**Investment recommendation**: Do not invest $50M

**Alternative**: Advise project to use SHA-256, BLAKE2b, or Keccak-256. If they refuse, walk away. Novel cryptography in blockchain is a disaster waiting to happen.

**Historical lesson**: MD5, SHA-1, and countless custom hashes have failed. Don't be the next victim. Boring cryptography keeps your money safe.

*In blockchain, "innovative cryptography" is a euphemism for "future exploit."*
`,
  },
];
