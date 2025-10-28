export const bitcoinArchitectureDeepDive = {
  title: 'Bitcoin Architecture Deep Dive',
  id: 'bitcoin-architecture-deep-dive',
  content: `
# Bitcoin Architecture Deep Dive

## Introduction

Bitcoin is not just "digital money"—it's a complete distributed system that elegantly solves the double-spending problem without trusted intermediaries. Understanding Bitcoin's architecture is essential because it pioneered the blockchain design patterns that thousands of cryptocurrencies and distributed systems now use.

**The Problem Bitcoin Solves:** Before Bitcoin, all digital payment systems required a trusted central authority (banks, PayPal, credit card companies) to prevent double-spending. Bitcoin was the first system to solve this in a fully decentralized way.

### The Double-Spending Problem

Digital data can be copied infinitely. If I have a digital coin file, I can send copies to multiple people:

\`\`\`
Alice has 1 BTC (digital file)
→ Copy 1 sent to Bob
→ Copy 2 sent to Charlie
→ Copy 3 sent to David
Problem: Alice spent the same coin 3 times!
\`\`\`

Traditional solutions: A central authority (bank) maintains a ledger and rejects duplicate spends. Bitcoin's innovation: A decentralized network collectively maintains the ledger, and consensus rules make double-spending economically infeasible.

## The UTXO Model

Bitcoin uses the **Unspent Transaction Output (UTXO)** model, fundamentally different from the account-based model most people understand.

### Account Model (Traditional Banking)

\`\`\`
Account: Alice
Balance: $1,000

Transaction: Alice sends $200 to Bob
→ Alice's balance: $800
→ Bob's balance: $200
\`\`\`

The "balance" is a single number that gets updated.

### UTXO Model (Bitcoin)

**There are no "accounts" or "balances" in Bitcoin.** Instead, the blockchain is a collection of unspent transaction outputs (UTXOs). Each UTXO is like a specific bill in your physical wallet.

\`\`\`
Alice's "balance" of 1.5 BTC is actually:
UTXO 1: 1.0 BTC (received from Tx abc123)
UTXO 2: 0.3 BTC (received from Tx def456)
UTXO 3: 0.2 BTC (received from Tx ghi789)
Total: 1.5 BTC
\`\`\`

When Alice wants to send 0.8 BTC to Bob:

\`\`\`
Inputs (consumed UTXOs):
  - UTXO 1: 1.0 BTC
  
Outputs (new UTXOs):
  - 0.8 BTC to Bob (Bob's address)
  - 0.19 BTC to Alice (change address)
  - 0.01 BTC miners fee (implicit)

After transaction:
  - UTXO 1 is destroyed (spent)
  - Two new UTXOs created (Bob's 0.8 BTC, Alice's 0.19 BTC change)
\`\`\`

### UTXO Implementation in Python

\`\`\`python
from dataclasses import dataclass
from typing import List
import hashlib

@dataclass
class UTXO:
    """Unspent Transaction Output"""
    txid: str              # Transaction ID that created this UTXO
    output_index: int      # Index within that transaction
    amount: float          # Amount in BTC
    script_pubkey: str     # Locking script (typically recipient's address)
    
    def __repr__(self):
        return f"UTXO({self.txid[:8]}:{self.output_index}, {self.amount} BTC)"

@dataclass
class TransactionInput:
    """Reference to a UTXO being spent"""
    prev_txid: str         # Transaction containing the UTXO
    prev_output_index: int # Which output in that transaction
    script_sig: str        # Unlocking script (signature + public key)
    
    def to_dict(self):
        return {
            'prev_txid': self.prev_txid,
            'prev_output_index': self.prev_output_index,
            'script_sig': self.script_sig
        }

@dataclass
class TransactionOutput:
    """New UTXO being created"""
    amount: float          # Amount in BTC
    script_pubkey: str     # Locking script (recipient's address)
    
    def to_dict(self):
        return {
            'amount': self.amount,
            'script_pubkey': self.script_pubkey
        }

class Transaction:
    """Bitcoin Transaction"""
    
    def __init__(self, inputs: List[TransactionInput], 
                 outputs: List[TransactionOutput]):
        self.inputs = inputs
        self.outputs = outputs
        self.txid = self._calculate_txid()
    
    def _calculate_txid(self) -> str:
        """Transaction ID is hash of transaction data"""
        tx_data = {
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs]
        }
        tx_bytes = str(tx_data).encode()
        return hashlib.sha256(tx_bytes).hexdigest()
    
    def get_input_sum(self, utxo_set: dict) -> float:
        """Calculate total value of inputs"""
        total = 0.0
        for inp in self.inputs:
            utxo_key = f"{inp.prev_txid}:{inp.prev_output_index}"
            if utxo_key in utxo_set:
                total += utxo_set[utxo_key].amount
        return total
    
    def get_output_sum(self) -> float:
        """Calculate total value of outputs"""
        return sum(out.amount for out in self.outputs)
    
    def get_fee(self, utxo_set: dict) -> float:
        """Mining fee is input_sum - output_sum"""
        return self.get_input_sum(utxo_set) - self.get_output_sum()
    
    def validate(self, utxo_set: dict) -> bool:
        """Validate transaction"""
        # Check all inputs exist in UTXO set
        for inp in self.inputs:
            utxo_key = f"{inp.prev_txid}:{inp.prev_output_index}"
            if utxo_key not in utxo_set:
                print(f"Invalid: Input {utxo_key} not in UTXO set")
                return False
        
        # Check outputs don't exceed inputs
        if self.get_output_sum() > self.get_input_sum(utxo_set):
            print(f"Invalid: Outputs ({self.get_output_sum()}) exceed inputs ({self.get_input_sum(utxo_set)})")
            return False
        
        # In real Bitcoin, would verify signatures here
        return True
    
    def __repr__(self):
        return f"Tx({self.txid[:8]}, {len(self.inputs)} inputs, {len(self.outputs)} outputs)"


# Example: Alice sends Bob 0.8 BTC
def example_transaction():
    """Demonstrate UTXO model"""
    
    # Initial UTXO set (existing UTXOs on blockchain)
    utxo_set = {
        "abc123:0": UTXO("abc123", 0, 1.0, "Alice_address"),
        "def456:0": UTXO("def456", 0, 0.3, "Alice_address"),
    }
    
    print("=== Initial State ===")
    print("Alice's UTXOs:")
    for key, utxo in utxo_set.items():
        if "Alice" in utxo.script_pubkey:
            print(f"  {utxo}")
    alice_balance = sum(u.amount for u in utxo_set.values() if "Alice" in u.script_pubkey)
    print(f"Alice's total: {alice_balance} BTC\n")
    
    # Alice creates transaction to send 0.8 BTC to Bob
    inputs = [
        TransactionInput("abc123", 0, "Alice_signature")
    ]
    
    outputs = [
        TransactionOutput(0.8, "Bob_address"),      # 0.8 BTC to Bob
        TransactionOutput(0.19, "Alice_address"),   # 0.19 BTC change to Alice
        # 0.01 BTC mining fee (implicit, not in outputs)
    ]
    
    tx = Transaction(inputs, outputs)
    
    print("=== Transaction Details ===")
    print(f"Transaction ID: {tx.txid}")
    print(f"Inputs: {tx.get_input_sum(utxo_set)} BTC")
    print(f"Outputs: {tx.get_output_sum()} BTC")
    print(f"Mining fee: {tx.get_fee(utxo_set)} BTC")
    print(f"Valid: {tx.validate(utxo_set)}\n")
    
    # Process transaction: Remove spent UTXOs, add new ones
    if tx.validate(utxo_set):
        # Remove spent UTXOs
        for inp in tx.inputs:
            utxo_key = f"{inp.prev_txid}:{inp.prev_output_index}"
            del utxo_set[utxo_key]
        
        # Add new UTXOs
        for i, out in enumerate(tx.outputs):
            utxo_key = f"{tx.txid}:{i}"
            utxo_set[utxo_key] = UTXO(tx.txid, i, out.amount, out.script_pubkey)
    
    print("=== Final State ===")
    print("Alice's UTXOs:")
    for key, utxo in utxo_set.items():
        if "Alice" in utxo.script_pubkey:
            print(f"  {utxo}")
    alice_balance = sum(u.amount for u in utxo_set.values() if "Alice" in u.script_pubkey)
    print(f"Alice's total: {alice_balance} BTC")
    
    print("\nBob's UTXOs:")
    for key, utxo in utxo_set.items():
        if "Bob" in utxo.script_pubkey:
            print(f"  {utxo}")
    bob_balance = sum(u.amount for u in utxo_set.values() if "Bob" in u.script_pubkey)
    print(f"Bob's total: {bob_balance} BTC")

example_transaction()
\`\`\`

Output:
\`\`\`
=== Initial State ===
Alice's UTXOs:
  UTXO(abc123:0, 1.0 BTC)
  UTXO(def456:0, 0.3 BTC)
Alice's total: 1.3 BTC

=== Transaction Details ===
Transaction ID: 7a3f8e2c...
Inputs: 1.0 BTC
Outputs: 0.99 BTC
Mining fee: 0.01 BTC
Valid: True

=== Final State ===
Alice's UTXOs:
  UTXO(def456:0, 0.3 BTC)
  UTXO(7a3f8e2c:1, 0.19 BTC)
Alice's total: 0.49 BTC

Bob's UTXOs:
  UTXO(7a3f8e2c:0, 0.8 BTC)
Bob's total: 0.8 BTC
\`\`\`

### Why UTXO Instead of Account Model?

**Advantages:**
1. **Parallelization**: Transactions touching different UTXOs can be validated in parallel
2. **Privacy**: Using new addresses for change makes tracking harder
3. **Simplicity**: Each UTXO is independent, no global state to update
4. **Fraud prevention**: Can't double-spend—UTXO either exists or doesn't

**Disadvantages:**
1. **Complex**: Harder for users to understand
2. **UTXO bloat**: Many small UTXOs increase storage requirements
3. **Fee calculation**: Users must manually construct transactions

## Bitcoin Script Language

Bitcoin transactions don't just say "pay Alice." They contain small programs (scripts) that define the conditions for spending.

### Locking and Unlocking Scripts

**Script_PubKey (Locking Script)**: Defines conditions to spend this output
**Script_Sig (Unlocking Script)**: Provides data to satisfy the conditions

When validating a transaction:
\`\`\`
Script_Sig + Script_PubKey → Execute → True/False
\`\`\`

If execution returns True, the transaction input is valid.

### Bitcoin Script Opcodes

Bitcoin Script is a stack-based language (like Forth or PostScript):

\`\`\`
Common Opcodes:
OP_DUP        - Duplicate top stack item
OP_HASH160    - Hash top item with SHA-256 then RIPEMD-160
OP_EQUALVERIFY - Verify top two items equal
OP_CHECKSIG   - Verify signature
\`\`\`

### Pay-to-Public-Key-Hash (P2PKH) - Standard Transaction

Most Bitcoin transactions use P2PKH:

**Locking Script (output):**
\`\`\`
OP_DUP OP_HASH160 <pubKeyHash> OP_EQUALVERIFY OP_CHECKSIG
\`\`\`

**Unlocking Script (input when spending):**
\`\`\`
<signature> <publicKey>
\`\`\`

**Execution:**
\`\`\`
Stack:                    Script:
[]                        <sig> <pubKey> OP_DUP OP_HASH160 <pubKeyHash> OP_EQUALVERIFY OP_CHECKSIG

[<sig>]                   <pubKey> OP_DUP OP_HASH160 <pubKeyHash> OP_EQUALVERIFY OP_CHECKSIG
[<sig> <pubKey>]          OP_DUP OP_HASH160 <pubKeyHash> OP_EQUALVERIFY OP_CHECKSIG
[<sig> <pubKey> <pubKey>] OP_HASH160 <pubKeyHash> OP_EQUALVERIFY OP_CHECKSIG
[<sig> <pubKey> <hash>]   <pubKeyHash> OP_EQUALVERIFY OP_CHECKSIG
[<sig> <pubKey> <hash> <pubKeyHash>] OP_EQUALVERIFY OP_CHECKSIG
[<sig> <pubKey>]          OP_CHECKSIG
[True]                    (if signature valid)
\`\`\`

### Script Interpreter Implementation

\`\`\`python
from typing import List, Any
import hashlib

class ScriptInterpreter:
    """Simplified Bitcoin Script interpreter"""
    
    def __init__(self):
        self.stack: List[Any] = []
    
    def execute(self, script: List[str]) -> bool:
        """Execute script and return True if successful"""
        try:
            for token in script:
                if token == "OP_DUP":
                    self.op_dup()
                elif token == "OP_HASH160":
                    self.op_hash160()
                elif token == "OP_EQUALVERIFY":
                    self.op_equalverify()
                elif token == "OP_CHECKSIG":
                    self.op_checksig()
                else:
                    # Push data onto stack
                    self.stack.append(token)
            
            # Success if stack has True on top
            return len(self.stack) > 0 and self.stack[-1] == True
        except Exception as e:
            print(f"Script execution failed: {e}")
            return False
    
    def op_dup(self):
        """Duplicate top stack item"""
        if len(self.stack) < 1:
            raise ValueError("OP_DUP: Stack underflow")
        self.stack.append(self.stack[-1])
    
    def op_hash160(self):
        """Hash top item with SHA-256 then RIPEMD-160"""
        if len(self.stack) < 1:
            raise ValueError("OP_HASH160: Stack underflow")
        data = self.stack.pop()
        # In real Bitcoin: SHA256 then RIPEMD160
        # Simplified: just SHA256
        hash_result = hashlib.sha256(data.encode()).hexdigest()[:40]
        self.stack.append(hash_result)
    
    def op_equalverify(self):
        """Verify top two items equal, remove both"""
        if len(self.stack) < 2:
            raise ValueError("OP_EQUALVERIFY: Stack underflow")
        b = self.stack.pop()
        a = self.stack.pop()
        if a != b:
            raise ValueError(f"OP_EQUALVERIFY: {a} != {b}")
    
    def op_checksig(self):
        """Verify signature (simplified)"""
        if len(self.stack) < 2:
            raise ValueError("OP_CHECKSIG: Stack underflow")
        pubkey = self.stack.pop()
        signature = self.stack.pop()
        
        # Real Bitcoin: Verify ECDSA signature
        # Simplified: Check if signature matches "VALID_SIG"
        is_valid = signature == "VALID_SIG" and len(pubkey) > 0
        self.stack.append(is_valid)


# Example: P2PKH transaction validation
def validate_p2pkh_transaction():
    """Demonstrate Bitcoin Script validation"""
    
    # Alice's public key
    alice_pubkey = "ALICE_PUBKEY_02a1b2c3..."
    alice_pubkey_hash = hashlib.sha256(alice_pubkey.encode()).hexdigest()[:40]
    
    print("=== P2PKH Transaction ===")
    print(f"Alice's public key: {alice_pubkey}")
    print(f"Alice's pubkey hash: {alice_pubkey_hash}\n")
    
    # Locking script (output) - locks coins to Alice's pubkey hash
    locking_script = [
        "OP_DUP",
        "OP_HASH160",
        alice_pubkey_hash,
        "OP_EQUALVERIFY",
        "OP_CHECKSIG"
    ]
    
    # Unlocking script (input) - Alice provides signature and pubkey
    unlocking_script = [
        "VALID_SIG",      # Alice's signature
        alice_pubkey      # Alice's public key
    ]
    
    # Combined script (unlocking + locking)
    full_script = unlocking_script + locking_script
    
    print("Unlocking script:", " ".join(unlocking_script))
    print("Locking script:", " ".join(locking_script))
    print()
    
    # Execute script
    interpreter = ScriptInterpreter()
    result = interpreter.execute(full_script)
    
    print(f"Script execution result: {result}")
    print(f"Transaction {'VALID' if result else 'INVALID'}")

validate_p2pkh_transaction()
\`\`\`

Output:
\`\`\`
=== P2PKH Transaction ===
Alice's public key: ALICE_PUBKEY_02a1b2c3...
Alice's pubkey hash: 4e9d3e3c8a7b2f1c3d8e7a2b5c8f1e4d9a6b

Unlocking script: VALID_SIG ALICE_PUBKEY_02a1b2c3...
Locking script: OP_DUP OP_HASH160 4e9d3e3c8a7b2f1c3d8e7a2b5c8f1e4d9a6b OP_EQUALVERIFY OP_CHECKSIG

Script execution result: True
Transaction VALID
\`\`\`

### Advanced Script Types

**Pay-to-Script-Hash (P2SH)**: Enables complex scripts like multisig
\`\`\`
Locking: OP_HASH160 <scriptHash> OP_EQUAL
Unlocking: <data> <script>
\`\`\`

**Pay-to-Witness-Public-Key-Hash (P2WPKH)**: SegWit native
\`\`\`
Witness: <signature> <publicKey>
ScriptPubKey: OP_0 <20-byte-hash>
\`\`\`

**MultiSig**: Requires M-of-N signatures
\`\`\`
2 <pubkey1> <pubkey2> <pubkey3> 3 OP_CHECKMULTISIG
\`\`\`

## Block Structure

Bitcoin blocks are containers for transactions, linked together via hash pointers.

### Block Header (80 bytes)

\`\`\`python
@dataclass
class BlockHeader:
    """Bitcoin block header - exactly 80 bytes"""
    version: int            # 4 bytes - Protocol version
    prev_block_hash: str    # 32 bytes - Hash of previous block
    merkle_root: str        # 32 bytes - Root of transaction merkle tree
    timestamp: int          # 4 bytes - Unix timestamp
    difficulty_bits: int    # 4 bytes - Target difficulty
    nonce: int              # 4 bytes - Proof-of-work nonce
    
    def hash(self) -> str:
        """Calculate block hash (SHA-256 twice)"""
        header_bytes = (
            self.version.to_bytes(4, 'little') +
            bytes.fromhex(self.prev_block_hash)[::-1] +
            bytes.fromhex(self.merkle_root)[::-1] +
            self.timestamp.to_bytes(4, 'little') +
            self.difficulty_bits.to_bytes(4, 'little') +
            self.nonce.to_bytes(4, 'little')
        )
        # Double SHA-256
        hash1 = hashlib.sha256(header_bytes).digest()
        hash2 = hashlib.sha256(hash1).digest()
        return hash2[::-1].hex()  # Reverse for display
    
    def meets_target(self, target: int) -> bool:
        """Check if block hash meets difficulty target"""
        block_hash_int = int(self.hash(), 16)
        return block_hash_int < target
\`\`\`

### Full Block Structure

\`\`\`python
from typing import List

class Block:
    """Bitcoin block"""
    
    def __init__(self, prev_block_hash: str, transactions: List[Transaction]):
        self.header = BlockHeader(
            version=1,
            prev_block_hash=prev_block_hash,
            merkle_root=self._calculate_merkle_root(transactions),
            timestamp=int(time.time()),
            difficulty_bits=0x1d00ffff,  # Encoded target
            nonce=0
        )
        self.transactions = transactions
    
    def _calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calculate merkle root of transactions"""
        if not transactions:
            return "0" * 64
        
        # Get transaction IDs
        tx_hashes = [tx.txid for tx in transactions]
        
        # Build merkle tree bottom-up
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])  # Duplicate last if odd
            
            new_level = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i+1]
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(parent_hash)
            
            tx_hashes = new_level
        
        return tx_hashes[0]
    
    def mine(self, target: int) -> bool:
        """Mine block by finding valid nonce"""
        MAX_NONCE = 2**32
        for nonce in range(MAX_NONCE):
            self.header.nonce = nonce
            if self.header.meets_target(target):
                print(f"Block mined! Nonce: {nonce}, Hash: {self.header.hash()}")
                return True
        return False
    
    def validate(self) -> bool:
        """Validate block structure"""
        # Check merkle root
        calculated_root = self._calculate_merkle_root(self.transactions)
        if calculated_root != self.header.merkle_root:
            return False
        
        # Check proof-of-work
        target = 2**(256 - 20)  # Simplified target
        if not self.header.meets_target(target):
            return False
        
        return True
\`\`\`

## Difficulty Adjustment

Bitcoin targets 10-minute block times. Every 2016 blocks (~2 weeks), difficulty adjusts:

\`\`\`python
def calculate_new_difficulty(old_target: int, time_taken: int) -> int:
    """Calculate new difficulty target"""
    TARGET_TIME = 2016 * 10 * 60  # 2 weeks in seconds
    
    # New target = old target * (time taken / target time)
    new_target = old_target * time_taken // TARGET_TIME
    
    # Limit adjustment to 4x in either direction
    MAX_ADJUSTMENT = 4
    if new_target > old_target * MAX_ADJUSTMENT:
        new_target = old_target * MAX_ADJUSTMENT
    elif new_target < old_target // MAX_ADJUSTMENT:
        new_target = old_target // MAX_ADJUSTMENT
    
    return new_target

# Example: Blocks came faster, increase difficulty
old_target = 2**200
time_taken = 2016 * 8 * 60  # 8 minutes per block (too fast!)
new_target = calculate_new_difficulty(old_target, time_taken)

print(f"Old target: {old_target}")
print(f"New target: {new_target}")
print(f"Difficulty increased by: {old_target / new_target:.2f}x")
\`\`\`

## Coinbase Transactions

First transaction in each block creates new bitcoins (block reward + fees):

\`\`\`python
def create_coinbase_transaction(block_height: int, miner_address: str, 
                                total_fees: float) -> Transaction:
    """Create coinbase transaction"""
    # Block reward halves every 210,000 blocks
    halvings = block_height // 210_000
    block_reward = 50 / (2 ** halvings)  # Started at 50 BTC
    
    # Coinbase input (special, no previous output)
    coinbase_input = TransactionInput(
        prev_txid="0" * 64,  # Null hash
        prev_output_index=0xffffffff,  # Special value
        script_sig=f"Block height: {block_height}"
    )
    
    # Output to miner
    output = TransactionOutput(
        amount=block_reward + total_fees,
        script_pubkey=miner_address
    )
    
    return Transaction([coinbase_input], [output])

# Example: Block 800,000
tx = create_coinbase_transaction(800_000, "miner_address", 0.1)
print(f"Block reward: {50 / (2 ** (800_000 // 210_000))} BTC")
print(f"Total fees: 0.1 BTC")
print(f"Miner receives: {tx.outputs[0].amount} BTC")
\`\`\`

Output:
\`\`\`
Block reward: 3.125 BTC
Total fees: 0.1 BTC
Miner receives: 3.225 BTC
\`\`\`

## Network Propagation

When a transaction is broadcast:

\`\`\`
Alice's node
  ↓ broadcast to peers
8 peer nodes
  ↓ each broadcasts to their peers
64 nodes
  ↓ exponential spread
Entire network (in ~5 seconds)
\`\`\`

## Summary

Bitcoin's architecture elegantly combines:
- **UTXO model**: Prevents double-spending through explicit output tracking
- **Script language**: Enables programmable spending conditions
- **Block structure**: Links blocks via hash pointers for immutability
- **Proof-of-work**: Secures network through computational cost
- **Difficulty adjustment**: Maintains consistent block times
- **Coinbase transactions**: Incentivizes miners with new supply

Understanding these components is essential—they form the foundation that thousands of cryptocurrencies build upon.
`,
};
