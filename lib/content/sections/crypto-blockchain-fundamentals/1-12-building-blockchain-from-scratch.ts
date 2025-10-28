export const buildingBlockchainFromScratch = {
  title: 'Building a Blockchain from Scratch',
  id: 'building-blockchain-from-scratch',
  content: `
# Building a Blockchain from Scratch

## Introduction

The best way to truly understand blockchain is to build one from scratch. This section walks through implementing a functional blockchain in Python, covering:
- Block and transaction structures
- Proof-of-work mining
- Chain validation
- Transaction processing
- Basic P2P networking
- Wallet functionality

By the end, you'll have a working (simplified) blockchain that demonstrates all core concepts.

## Project Architecture

\`\`\`
SimpleCoin Blockchain
├── Block (data structure)
├── Transaction (transfers)
├── Blockchain (chain management)
├── ProofOfWork (consensus)
├── Wallet (key management)
├── Network (P2P)
└── Node (full implementation)
\`\`\`

## Step 1: Block Structure

\`\`\`python
import hashlib
import time
import json
from typing import List, Optional
from dataclasses import dataclass, asdict

@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: float
    transactions: List[dict]
    proof: int  # Proof-of-work nonce
    previous_hash: str
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def __repr__(self):
        return f"Block(#{self.index}, {len(self.transactions)} txs, hash={self.calculate_hash()[:16]}...)"


# Example: Create genesis block
genesis_block = Block(
    index=0,
    timestamp=time.time(),
    transactions=[],
    proof=100,  # Arbitrary
    previous_hash="0" * 64
)

print("=== Genesis Block ===")
print(genesis_block)
print(f"Hash: {genesis_block.calculate_hash()}")
\`\`\`

## Step 2: Transaction Structure

\`\`\`python
import ecdsa
import hashlib

class Transaction:
    """Blockchain transaction"""
    
    def __init__(self, sender: str, recipient: str, amount: float):
        self.sender = sender  # Public key or address
        self.recipient = recipient
        self.amount = amount
        self.timestamp = time.time()
        self.signature: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount,
            'timestamp': self.timestamp
        }
    
    def calculate_hash(self) -> str:
        """Calculate transaction hash"""
        tx_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def sign_transaction(self, private_key: str):
        """Sign transaction with private key"""
        if self.sender == "MINING_REWARD":
            # Coinbase transactions don't need signatures
            return
        
        # Create signature
        tx_hash = self.calculate_hash()
        signature = hashlib.sha256(f"{private_key}{tx_hash}".encode()).hexdigest()
        self.signature = signature
    
    def is_valid(self, public_key: Optional[str] = None) -> bool:
        """Verify transaction validity"""
        # Mining rewards are always valid
        if self.sender == "MINING_REWARD":
            return True
        
        # Must have signature
        if not self.signature:
            print("Transaction has no signature")
            return False
        
        # Verify signature (simplified)
        if public_key:
            tx_hash = self.calculate_hash()
            expected_sig = hashlib.sha256(f"{public_key}{tx_hash}".encode()).hexdigest()
            # In real implementation: ECDSA signature verification
            # For simplicity: just check signature exists
            return len(self.signature) > 0
        
        return True
    
    def __repr__(self):
        return f"Transaction({self.sender[:8]}...→{self.recipient[:8]}..., {self.amount})"


# Example: Create and sign transaction
print("\\n=== Transaction Example ===")
tx = Transaction(
    sender="alice_public_key_abc123",
    recipient="bob_public_key_def456",
    amount=10.5
)
print(f"Created: {tx}")
print(f"Hash: {tx.calculate_hash()}")

tx.sign_transaction("alice_private_key_xyz789")
print(f"Signed: {tx.signature[:32]}...")
print(f"Valid: {tx.is_valid()}")
\`\`\`

## Step 3: Proof of Work

\`\`\`python
class ProofOfWork:
    """Proof-of-work consensus"""
    
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty  # Number of leading zeros
        self.target = "0" * difficulty
    
    def mine_block(self, block: Block) -> int:
        """Mine block by finding valid proof"""
        print(f"\\nMining block {block.index}...")
        start_time = time.time()
        attempts = 0
        
        while True:
            block.proof = attempts
            hash_result = block.calculate_hash()
            
            attempts += 1
            
            # Check if hash meets difficulty target
            if hash_result[:self.difficulty] == self.target:
                elapsed = time.time() - start_time
                hash_rate = attempts / elapsed if elapsed > 0 else 0
                
                print(f"✓ Block mined!")
                print(f"  Proof: {block.proof:,}")
                print(f"  Attempts: {attempts:,}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Hash rate: {hash_rate:,.0f} H/s")
                print(f"  Hash: {hash_result}")
                return block.proof
            
            # Progress update
            if attempts % 100000 == 0:
                print(f"  Attempt {attempts:,}...")
    
    def validate_proof(self, block: Block) -> bool:
        """Verify proof-of-work is valid"""
        hash_result = block.calculate_hash()
        return hash_result[:self.difficulty] == self.target


# Example: Mine a block
print("\\n=== Proof of Work Mining ===")
pow = ProofOfWork(difficulty=4)

block = Block(
    index=1,
    timestamp=time.time(),
    transactions=[{'sender': 'alice', 'recipient': 'bob', 'amount': 10}],
    proof=0,
    previous_hash=genesis_block.calculate_hash()
)

pow.mine_block(block)
print(f"\\nProof valid: {pow.validate_proof(block)}")
\`\`\`

## Step 4: Blockchain Class

\`\`\`python
class Blockchain:
    """Blockchain implementation"""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.mining_reward = 50.0
        self.pow = ProofOfWork(difficulty)
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block"""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            proof=100,
            previous_hash="0" * 64
        )
        self.chain.append(genesis)
        print(f"Genesis block created: {genesis.calculate_hash()[:16]}...")
    
    def get_latest_block(self) -> Block:
        """Get most recent block"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add transaction to pending pool"""
        # Validate transaction
        if not transaction.is_valid():
            print("Invalid transaction")
            return False
        
        # Check sender has sufficient balance
        balance = self.get_balance(transaction.sender)
        if balance < transaction.amount:
            print(f"Insufficient balance: {balance} < {transaction.amount}")
            return False
        
        self.pending_transactions.append(transaction)
        print(f"Transaction added to pending pool")
        return True
    
    def mine_pending_transactions(self, miner_address: str):
        """Mine block with pending transactions"""
        # Create mining reward transaction
        reward_tx = Transaction("MINING_REWARD", miner_address, self.mining_reward)
        self.pending_transactions.append(reward_tx)
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=[tx.to_dict() for tx in self.pending_transactions],
            proof=0,
            previous_hash=self.get_latest_block().calculate_hash()
        )
        
        # Mine block
        self.pow.mine_block(new_block)
        
        # Add block to chain
        self.chain.append(new_block)
        print(f"\\nBlock {new_block.index} added to chain")
        
        # Clear pending transactions
        self.pending_transactions = []
    
    def get_balance(self, address: str) -> float:
        """Calculate balance for address"""
        balance = 0.0
        
        # Iterate through all blocks and transactions
        for block in self.chain:
            for tx_dict in block.transactions:
                if tx_dict['sender'] == address:
                    balance -= tx_dict['amount']
                if tx_dict['recipient'] == address:
                    balance += tx_dict['amount']
        
        # Include pending transactions
        for tx in self.pending_transactions:
            if tx.sender == address:
                balance -= tx.amount
            if tx.recipient == address:
                balance += tx.amount
        
        return balance
    
    def is_chain_valid(self) -> bool:
        """Validate entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if block hash is valid
            if not self.pow.validate_proof(current_block):
                print(f"Invalid proof-of-work at block {i}")
                return False
            
            # Check if previous hash matches
            if current_block.previous_hash != previous_block.calculate_hash():
                print(f"Invalid previous hash at block {i}")
                return False
        
        return True
    
    def get_chain_info(self) -> dict:
        """Get blockchain statistics"""
        total_transactions = sum(len(block.transactions) for block in self.chain)
        return {
            'length': len(self.chain),
            'total_transactions': total_transactions,
            'difficulty': self.difficulty,
            'latest_block_hash': self.get_latest_block().calculate_hash(),
            'is_valid': self.is_chain_valid()
        }
    
    def __repr__(self):
        return f"Blockchain({len(self.chain)} blocks)"


# Example: Full blockchain usage
print("\\n=== Complete Blockchain Example ===\\n")

# Create blockchain
blockchain = Blockchain(difficulty=3)  # Easier difficulty for demo

print(f"\\nInitial state: {blockchain}")
print(f"Genesis block: {blockchain.get_latest_block()}\\n")

# Create transactions
print("=== Creating Transactions ===")
tx1 = Transaction("alice", "bob", 10)
tx1.sign_transaction("alice_private_key")
blockchain.add_transaction(tx1)

tx2 = Transaction("bob", "charlie", 5)
tx2.sign_transaction("bob_private_key")
blockchain.add_transaction(tx2)

print(f"Pending transactions: {len(blockchain.pending_transactions)}")

# Mine block
print("\\n=== Mining Block 1 ===")
blockchain.mine_pending_transactions("miner_alice")

# Check balances
print("\\n=== Balances ===")
for address in ["alice", "bob", "charlie", "miner_alice"]:
    balance = blockchain.get_balance(address)
    print(f"{address}: {balance}")

# Mine another block
print("\\n=== Mining Block 2 ===")
tx3 = Transaction("miner_alice", "david", 20)
tx3.sign_transaction("alice_private_key")
blockchain.add_transaction(tx3)

blockchain.mine_pending_transactions("miner_bob")

# Final state
print("\\n=== Final Blockchain State ===")
info = blockchain.get_chain_info()
for key, value in info.items():
    print(f"{key}: {value}")

print(f"\\nBlockchain valid: {blockchain.is_chain_valid()}")
\`\`\`

## Step 5: Wallet Implementation

\`\`\`python
import secrets

class Wallet:
    """Simple wallet for key management"""
    
    def __init__(self):
        self.private_key = self.generate_private_key()
        self.public_key = self.derive_public_key(self.private_key)
        self.address = self.derive_address(self.public_key)
    
    def generate_private_key(self) -> str:
        """Generate random private key"""
        return secrets.token_hex(32)
    
    def derive_public_key(self, private_key: str) -> str:
        """Derive public key from private key"""
        # In reality: Elliptic curve point multiplication
        # Simplified: Hash private key
        return hashlib.sha256(private_key.encode()).hexdigest()
    
    def derive_address(self, public_key: str) -> str:
        """Derive address from public key"""
        # In reality: Hash public key, add checksum, Base58 encode
        # Simplified: First 40 chars of public key hash
        return hashlib.sha256(public_key.encode()).hexdigest()[:40]
    
    def sign_transaction(self, transaction: Transaction):
        """Sign transaction"""
        transaction.sign_transaction(self.private_key)
    
    def get_balance(self, blockchain: Blockchain) -> float:
        """Get wallet balance from blockchain"""
        return blockchain.get_balance(self.address)
    
    def send(self, blockchain: Blockchain, recipient: str, amount: float) -> bool:
        """Send coins to recipient"""
        # Check balance
        balance = self.get_balance(blockchain)
        if balance < amount:
            print(f"Insufficient funds: {balance} < {amount}")
            return False
        
        # Create transaction
        tx = Transaction(self.address, recipient, amount)
        self.sign_transaction(tx)
        
        # Add to blockchain
        return blockchain.add_transaction(tx)
    
    def __repr__(self):
        return f"Wallet({self.address[:16]}...)"


# Example: Wallet usage
print("\\n=== Wallet Example ===\\n")

# Create wallets
alice_wallet = Wallet()
bob_wallet = Wallet()

print(f"Alice: {alice_wallet}")
print(f"  Address: {alice_wallet.address}")
print(f"  Public key: {alice_wallet.public_key[:32]}...")
print()
print(f"Bob: {bob_wallet}")
print(f"  Address: {bob_wallet.address}")
\`\`\`

## Step 6: Basic P2P Network

\`\`\`python
import socket
import threading
import pickle

class Node:
    """Blockchain network node"""
    
    def __init__(self, host: str, port: int, blockchain: Blockchain):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.peers = []  # List of (host, port) tuples
        self.socket = None
    
    def start(self):
        """Start node server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        
        print(f"Node started on {self.host}:{self.port}")
        
        # Listen for incoming connections
        threading.Thread(target=self.listen_for_peers, daemon=True).start()
    
    def listen_for_peers(self):
        """Listen for incoming peer connections"""
        while True:
            try:
                client_socket, address = self.socket.accept()
                threading.Thread(
                    target=self.handle_peer,
                    args=(client_socket, address),
                    daemon=True
                ).start()
            except Exception as e:
                print(f"Error accepting connection: {e}")
    
    def handle_peer(self, client_socket, address):
        """Handle messages from peer"""
        try:
            data = client_socket.recv(4096)
            message = pickle.loads(data)
            
            if message['type'] == 'new_transaction':
                tx = message['transaction']
                self.blockchain.add_transaction(tx)
                print(f"Received transaction from {address}")
            
            elif message['type'] == 'new_block':
                block = message['block']
                self.receive_block(block)
                print(f"Received block from {address}")
            
            elif message['type'] == 'get_chain':
                # Send our chain
                response = pickle.dumps({
                    'type': 'chain',
                    'chain': self.blockchain.chain
                })
                client_socket.send(response)
        
        except Exception as e:
            print(f"Error handling peer {address}: {e}")
        finally:
            client_socket.close()
    
    def connect_to_peer(self, host: str, port: int):
        """Connect to peer node"""
        if (host, port) not in self.peers:
            self.peers.append((host, port))
            print(f"Connected to peer {host}:{port}")
    
    def broadcast_transaction(self, transaction: Transaction):
        """Broadcast transaction to all peers"""
        message = {
            'type': 'new_transaction',
            'transaction': transaction
        }
        self.broadcast(message)
    
    def broadcast_block(self, block: Block):
        """Broadcast new block to all peers"""
        message = {
            'type': 'new_block',
            'block': block
        }
        self.broadcast(message)
    
    def broadcast(self, message: dict):
        """Send message to all peers"""
        for host, port in self.peers:
            try:
                peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                peer_socket.connect((host, port))
                peer_socket.send(pickle.dumps(message))
                peer_socket.close()
            except Exception as e:
                print(f"Error broadcasting to {host}:{port}: {e}")
    
    def sync_chain(self):
        """Sync blockchain with peers"""
        for host, port in self.peers:
            try:
                peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                peer_socket.connect((host, port))
                
                # Request chain
                message = {'type': 'get_chain'}
                peer_socket.send(pickle.dumps(message))
                
                # Receive chain
                data = peer_socket.recv(40960)
                response = pickle.loads(data)
                
                if response['type'] == 'chain':
                    peer_chain = response['chain']
                    if len(peer_chain) > len(self.blockchain.chain):
                        # Replace with longer valid chain
                        self.blockchain.chain = peer_chain
                        print(f"Synced with peer {host}:{port}")
                
                peer_socket.close()
            except Exception as e:
                print(f"Error syncing with {host}:{port}: {e}")
    
    def receive_block(self, block: Block):
        """Receive block from peer"""
        # Validate block
        latest_block = self.blockchain.get_latest_block()
        
        if block.previous_hash == latest_block.calculate_hash():
            if self.blockchain.pow.validate_proof(block):
                self.blockchain.chain.append(block)
                print(f"Block {block.index} added to chain")
            else:
                print("Invalid proof-of-work")
        else:
            print("Block does not link to our chain")

# Example: Network node
print("\\n=== P2P Network Node ===")
print("Node listens for:")
print("  - New transactions")
print("  - New blocks")
print("  - Chain sync requests")
print("\\nNode broadcasts:")
print("  - Mined blocks")
print("  - New transactions")
\`\`\`

## Complete Example

\`\`\`python
def run_blockchain_demo():
    """Complete blockchain demonstration"""
    print("\\n" + "="*60)
    print("BLOCKCHAIN FROM SCRATCH - COMPLETE DEMO")
    print("="*60)
    
    # Create blockchain
    blockchain = Blockchain(difficulty=3)
    
    # Create wallets
    alice = Wallet()
    bob = Wallet()
    charlie = Wallet()
    
    print(f"\\n📁 Created blockchain with {len(blockchain.chain)} block(s)")
    print(f"👤 Created 3 wallets\\n")
    
    # Give Alice initial coins (manual for demo)
    print("💰 Initial funding (manual)...")
    initial_tx = Transaction("GENESIS", alice.address, 100)
    blockchain.pending_transactions.append(initial_tx)
    blockchain.mine_pending_transactions(alice.address)
    
    print(f"Alice balance: {blockchain.get_balance(alice.address)}")
    
    # Alice sends to Bob
    print(f"\\n📤 Alice sends 30 coins to Bob")
    tx1 = Transaction(alice.address, bob.address, 30)
    alice.sign_transaction(tx1)
    blockchain.add_transaction(tx1)
    
    # Bob sends to Charlie
    print(f"📤 Bob sends 10 coins to Charlie")
    tx2 = Transaction(bob.address, charlie.address, 10)
    bob.sign_transaction(tx2)
    blockchain.add_transaction(tx2)
    
    # Mine block
    print(f"\\n⛏️  Mining block with {len(blockchain.pending_transactions)} transactions...")
    blockchain.mine_pending_transactions(bob.address)
    
    # Show final balances
    print("\\n💰 Final Balances:")
    print(f"  Alice:   {blockchain.get_balance(alice.address):.2f} coins")
    print(f"  Bob:     {blockchain.get_balance(bob.address):.2f} coins")
    print(f"  Charlie: {blockchain.get_balance(charlie.address):.2f} coins")
    
    # Blockchain info
    print("\\n📊 Blockchain Statistics:")
    info = blockchain.get_chain_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Show chain
    print("\\n⛓️  Blockchain:")
    for block in blockchain.chain:
        print(f"  {block}")
    
    print("\\n✅ Demo complete!")

# Run demo
run_blockchain_demo()
\`\`\`

## Extensions and Improvements

This basic blockchain can be extended with:

1. **Smart Contracts**: Add scripting language like Bitcoin Script
2. **Merkle Trees**: Efficient transaction verification
3. **UTXO Model**: More realistic transaction model
4. **Network Protocol**: Full P2P implementation with NAT traversal
5. **Database**: Persist blockchain to disk (SQLite/LevelDB)
6. **API**: REST API for wallets and explorers
7. **Advanced Consensus**: Proof-of-Stake, BFT consensus
8. **Scalability**: Sharding, sidechains, rollups

## Summary

Building a blockchain from scratch teaches:
- **Data structures**: Blocks, transactions, Merkle trees
- **Cryptography**: Hashing, digital signatures
- **Consensus**: Proof-of-work, chain selection
- **Networking**: P2P protocols, broadcasting
- **Economics**: Incentives, game theory

Real blockchains add:
- Production-grade security
- Optimized performance
- Advanced features (smart contracts, etc.)
- Robust networking
- Economic mechanisms

But the core principles remain the same!
`,
};
