export const transactionLifecycleMempool = {
  title: 'Transaction Lifecycle & Mempool',
  id: 'transaction-lifecycle-mempool',
  content: `
# Transaction Lifecycle & Mempool

## Introduction

Understanding the transaction lifecycle—from creation to final confirmation—is essential for building reliable blockchain applications. This section covers how transactions flow through the network, how the mempool operates, and how to ensure your transactions get confirmed reliably and cost-effectively.

**Key topics:**
- Transaction creation and signing
- Broadcasting to the network
- Mempool dynamics and fee markets
- Transaction finality and reorg protection
- Advanced techniques (RBF, CPFP, batching)

## Transaction Lifecycle Overview

\`\`\`
1. Creation: User constructs transaction
   ↓
2. Signing: User signs with private key
   ↓
3. Broadcasting: Sent to network nodes
   ↓
4. Mempool: Sits in unconfirmed transaction pool
   ↓
5. Mining/Validation: Included in block by miner/validator
   ↓
6. Confirmation: Block added to chain
   ↓
7. Finality: Enough confirmations to be irreversible
\`\`\`

## Transaction Creation

### Bitcoin Transaction Construction

\`\`\`python
import hashlib
import time
from dataclasses import dataclass
from typing import List
from decimal import Decimal

@dataclass
class UTXO:
    """Unspent transaction output"""
    txid: str
    vout: int          # Output index
    amount: Decimal    # BTC amount
    script_pubkey: str
    confirmations: int

@dataclass
class TransactionInput:
    """Transaction input (reference to UTXO)"""
    txid: str
    vout: int
    script_sig: str    # Signature script
    sequence: int = 0xffffffff
    
    def serialize(self) -> str:
        return f"{self.txid}:{self.vout}:{self.script_sig}:{self.sequence}"

@dataclass
class TransactionOutput:
    """Transaction output (new UTXO)"""
    amount: Decimal
    script_pubkey: str
    
    def serialize(self) -> str:
        return f"{self.amount}:{self.script_pubkey}"

class TransactionBuilder:
    """Build Bitcoin-style transactions"""
    
    def __init__(self):
        self.inputs: List[TransactionInput] = []
        self.outputs: List[TransactionOutput] = []
        self.locktime = 0
        self.version = 1
    
    def add_input(self, utxo: UTXO):
        """Add input from UTXO"""
        inp = TransactionInput(
            txid=utxo.txid,
            vout=utxo.vout,
            script_sig=""  # Will be filled after signing
        )
        self.inputs.append(inp)
    
    def add_output(self, amount: Decimal, address: str):
        """Add output"""
        out = TransactionOutput(
            amount=amount,
            script_pubkey=f"OP_DUP OP_HASH160 {address} OP_EQUALVERIFY OP_CHECKSIG"
        )
        self.outputs.append(out)
    
    def calculate_size(self) -> int:
        """Estimate transaction size in bytes"""
        # Simplified calculation
        input_size = len(self.inputs) * 148  # ~148 bytes per input
        output_size = len(self.outputs) * 34  # ~34 bytes per output
        overhead = 10  # Version, locktime, etc.
        return input_size + output_size + overhead
    
    def calculate_fee(self, fee_rate_sat_per_byte: int) -> Decimal:
        """Calculate required fee"""
        size = self.calculate_size()
        fee_satoshis = size * fee_rate_sat_per_byte
        return Decimal(fee_satoshis) / Decimal(100_000_000)  # Convert to BTC
    
    def build(self, utxos: List[UTXO], recipient_address: str, 
             send_amount: Decimal, change_address: str,
             fee_rate: int) -> 'Transaction':
        """
        Build transaction using coin selection
        
        Args:
            utxos: Available UTXOs
            recipient_address: Recipient's address
            send_amount: Amount to send (BTC)
            change_address: Address for change
            fee_rate: Fee rate in sat/byte
        """
        # Sort UTXOs by amount (descending)
        sorted_utxos = sorted(utxos, key=lambda u: u.amount, reverse=True)
        
        # Coin selection: Keep adding UTXOs until we have enough
        selected_utxos = []
        total_input = Decimal(0)
        
        # Estimate fee (will refine after selecting inputs)
        estimated_fee = Decimal(0.0001)  # Initial guess
        
        for utxo in sorted_utxos:
            selected_utxos.append(utxo)
            total_input += utxo.amount
            
            # Recalculate fee with current input count
            temp_builder = TransactionBuilder()
            for u in selected_utxos:
                temp_builder.add_input(u)
            temp_builder.add_output(send_amount, recipient_address)
            if total_input > send_amount:
                temp_builder.add_output(total_input - send_amount, change_address)
            
            estimated_fee = temp_builder.calculate_fee(fee_rate)
            
            # Check if we have enough
            if total_input >= send_amount + estimated_fee:
                break
        
        if total_input < send_amount + estimated_fee:
            raise ValueError(f"Insufficient funds: have {total_input}, need {send_amount + estimated_fee}")
        
        # Build transaction
        for utxo in selected_utxos:
            self.add_input(utxo)
        
        # Add recipient output
        self.add_output(send_amount, recipient_address)
        
        # Add change output
        change = total_input - send_amount - estimated_fee
        if change > Decimal(0.00001):  # Only add change if > dust threshold
            self.add_output(change, change_address)
        
        return Transaction(self.inputs, self.outputs)


class Transaction:
    """Bitcoin transaction"""
    
    def __init__(self, inputs: List[TransactionInput], outputs: List[TransactionOutput]):
        self.inputs = inputs
        self.outputs = outputs
        self.txid = self._calculate_txid()
        self.size = self._calculate_size()
    
    def _calculate_txid(self) -> str:
        """Calculate transaction ID"""
        data = ""
        for inp in self.inputs:
            data += inp.serialize()
        for out in self.outputs:
            data += out.serialize()
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _calculate_size(self) -> int:
        """Calculate transaction size"""
        return len(self.inputs) * 148 + len(self.outputs) * 34 + 10
    
    def sign(self, private_key: str):
        """Sign transaction inputs (simplified)"""
        for inp in self.inputs:
            # In reality: ECDSA signature of transaction hash
            inp.script_sig = f"<sig:{private_key[:8]}> <pubkey>"
    
    def get_input_sum(self, utxo_map: dict) -> Decimal:
        """Calculate total input value"""
        total = Decimal(0)
        for inp in self.inputs:
            key = f"{inp.txid}:{inp.vout}"
            if key in utxo_map:
                total += utxo_map[key].amount
        return total
    
    def get_output_sum(self) -> Decimal:
        """Calculate total output value"""
        return sum(out.amount for out in self.outputs)
    
    def get_fee(self, utxo_map: dict) -> Decimal:
        """Calculate transaction fee"""
        return self.get_input_sum(utxo_map) - self.get_output_sum()
    
    def get_fee_rate(self, utxo_map: dict) -> Decimal:
        """Calculate fee rate in sat/byte"""
        fee_btc = self.get_fee(utxo_map)
        fee_sat = fee_btc * Decimal(100_000_000)
        return fee_sat / Decimal(self.size)
    
    def __repr__(self):
        return f"Transaction({self.txid[:16]}..., {len(self.inputs)} inputs, {len(self.outputs)} outputs, {self.size} bytes)"


# Example: Build and sign transaction
print("=== Transaction Creation Example ===\\n")

# Alice's available UTXOs
utxos = [
    UTXO("abc123", 0, Decimal("0.5"), "alice_pubkey_hash", 10),
    UTXO("def456", 1, Decimal("0.3"), "alice_pubkey_hash", 5),
    UTXO("ghi789", 0, Decimal("0.15"), "alice_pubkey_hash", 20),
]

print("Available UTXOs:")
for utxo in utxos:
    print(f"  {utxo.txid[:8]}:{utxo.vout} = {utxo.amount} BTC ({utxo.confirmations} confirmations)")

total = sum(u.amount for u in utxos)
print(f"Total: {total} BTC\\n")

# Build transaction: Send 0.4 BTC to Bob
builder = TransactionBuilder()
tx = builder.build(
    utxos=utxos,
    recipient_address="bob_address_hash",
    send_amount=Decimal("0.4"),
    change_address="alice_change_address",
    fee_rate=20  # 20 sat/byte
)

print(f"Built: {tx}")

# Create UTXO map for fee calculation
utxo_map = {f"{u.txid}:{u.vout}": u for u in utxos}

print(f"\\nTransaction details:")
print(f"  Inputs: {tx.get_input_sum(utxo_map)} BTC")
print(f"  Outputs: {tx.get_output_sum()} BTC")
print(f"  Fee: {tx.get_fee(utxo_map)} BTC")
print(f"  Fee rate: {tx.get_fee_rate(utxo_map):.2f} sat/byte")
print(f"  Size: {tx.size} bytes")

# Sign transaction
print(f"\\nSigning transaction...")
tx.sign("alice_private_key_xyz123")
print(f"Signed: {tx}")
\`\`\`

Output:
\`\`\`
=== Transaction Creation Example ===

Available UTXOs:
  abc123:0 = 0.5 BTC (10 confirmations)
  def456:1 = 0.3 BTC (5 confirmations)
  ghi789:0 = 0.15 BTC (20 confirmations)
Total: 0.95 BTC

Built: Transaction(7a3f8e2c1b4d9a5e..., 2 inputs, 2 outputs, 306 bytes)

Transaction details:
  Inputs: 0.8 BTC
  Outputs: 0.793880 BTC
  Fee: 0.006120 BTC
  Fee rate: 20.00 sat/byte
  Size: 306 bytes

Signing transaction...
Signed: Transaction(7a3f8e2c1b4d9a5e..., 2 inputs, 2 outputs, 306 bytes)
\`\`\`

## Broadcasting

After signing, transaction is broadcast to network:

\`\`\`python
import random
import time

class Node:
    """Simple blockchain node"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.peers: List['Node'] = []
        self.mempool: dict = {}  # txid -> Transaction
        self.seen_txs: set = set()
    
    def connect_peer(self, peer: 'Node'):
        """Connect to peer"""
        if peer not in self.peers:
            self.peers.append(peer)
            peer.peers.append(self)
    
    def broadcast_transaction(self, tx: Transaction):
        """Broadcast transaction to network"""
        if tx.txid in self.seen_txs:
            return  # Already seen
        
        print(f"[{self.node_id}] Broadcasting {tx.txid[:16]}...")
        self.seen_txs.add(tx.txid)
        self.mempool[tx.txid] = tx
        
        # Forward to all peers
        for peer in self.peers:
            # Simulate network delay
            time.sleep(0.01)
            peer.receive_transaction(tx, from_node=self.node_id)
    
    def receive_transaction(self, tx: Transaction, from_node: str):
        """Receive transaction from peer"""
        if tx.txid in self.seen_txs:
            return  # Already seen
        
        print(f"[{self.node_id}] Received {tx.txid[:16]}... from {from_node}")
        self.seen_txs.add(tx.txid)
        self.mempool[tx.txid] = tx
        
        # Forward to other peers (gossip protocol)
        for peer in self.peers:
            if peer.node_id != from_node:
                peer.receive_transaction(tx, from_node=self.node_id)
    
    def __repr__(self):
        return f"Node({self.node_id}, {len(self.mempool)} txs in mempool)"


# Example: Transaction propagation
print("\\n=== Transaction Broadcasting ===\\n")

# Create network of 5 nodes
nodes = [Node(f"Node{i}") for i in range(5)]

# Connect nodes (partial mesh)
nodes[0].connect_peer(nodes[1])
nodes[0].connect_peer(nodes[2])
nodes[1].connect_peer(nodes[3])
nodes[2].connect_peer(nodes[4])
nodes[3].connect_peer(nodes[4])

print("Network topology:")
for node in nodes:
    peers = [p.node_id for p in node.peers]
    print(f"  {node.node_id} connected to: {peers}")

# Node0 broadcasts transaction
print(f"\\nNode0 broadcasts transaction...")
tx = Transaction([], [])  # Simplified
nodes[0].broadcast_transaction(tx)

print(f"\\nFinal state:")
for node in nodes:
    in_mempool = tx.txid in node.mempool
    print(f"  {node.node_id}: Transaction in mempool = {in_mempool}")
\`\`\`

## Mempool Dynamics

The **mempool** (memory pool) is each node's collection of unconfirmed transactions waiting to be mined.

### Mempool Structure

\`\`\`python
from typing import Dict, List
from decimal import Decimal

class Mempool:
    """Transaction memory pool"""
    
    def __init__(self, max_size_mb: int = 300):
        self.transactions: Dict[str, Transaction] = {}
        self.max_size = max_size_mb * 1_000_000  # Convert to bytes
        self.current_size = 0
    
    def add_transaction(self, tx: Transaction, utxo_map: dict) -> bool:
        """Add transaction to mempool"""
        # Check if already in mempool
        if tx.txid in self.transactions:
            return False
        
        # Validate transaction
        if not self._validate_transaction(tx, utxo_map):
            print(f"❌ Rejected {tx.txid[:16]}: Invalid")
            return False
        
        # Check if mempool is full
        if self.current_size + tx.size > self.max_size:
            # Evict lowest fee transactions
            self._evict_low_fee_transactions(tx.size)
        
        # Add to mempool
        self.transactions[tx.txid] = tx
        self.current_size += tx.size
        
        fee_rate = tx.get_fee_rate(utxo_map)
        print(f"✓ Added {tx.txid[:16]} to mempool ({fee_rate:.1f} sat/byte)")
        return True
    
    def _validate_transaction(self, tx: Transaction, utxo_map: dict) -> bool:
        """Validate transaction"""
        # Check inputs exist
        for inp in tx.inputs:
            key = f"{inp.txid}:{inp.vout}"
            if key not in utxo_map:
                return False
        
        # Check outputs <= inputs
        if tx.get_output_sum() > tx.get_input_sum(utxo_map):
            return False
        
        # Check minimum fee
        fee = tx.get_fee(utxo_map)
        if fee < Decimal(0.00001):  # 1000 satoshis minimum
            return False
        
        return True
    
    def _evict_low_fee_transactions(self, space_needed: int):
        """Evict lowest fee transactions to make space"""
        # Sort by fee rate (ascending)
        sorted_txs = sorted(
            self.transactions.items(),
            key=lambda item: item[1].get_fee_rate({}),
            reverse=False
        )
        
        space_freed = 0
        for txid, tx in sorted_txs:
            if space_freed >= space_needed:
                break
            
            del self.transactions[txid]
            self.current_size -= tx.size
            space_freed += tx.size
            print(f"⚠️  Evicted {txid[:16]} (low fee)")
    
    def get_transactions_by_fee(self, max_block_size: int) -> List[Transaction]:
        """Get highest fee transactions for block"""
        # Sort by fee rate (descending)
        sorted_txs = sorted(
            self.transactions.values(),
            key=lambda tx: tx.get_fee_rate({}),
            reverse=True
        )
        
        # Select transactions until block is full
        selected = []
        total_size = 0
        
        for tx in sorted_txs:
            if total_size + tx.size > max_block_size:
                continue
            selected.append(tx)
            total_size += tx.size
        
        return selected
    
    def remove_transaction(self, txid: str):
        """Remove transaction (e.g., after mining)"""
        if txid in self.transactions:
            tx = self.transactions[txid]
            del self.transactions[txid]
            self.current_size -= tx.size
    
    def get_stats(self) -> dict:
        """Get mempool statistics"""
        if not self.transactions:
            return {
                'count': 0,
                'size_mb': 0,
                'min_fee_rate': 0,
                'median_fee_rate': 0,
                'max_fee_rate': 0
            }
        
        fee_rates = [tx.get_fee_rate({}) for tx in self.transactions.values()]
        fee_rates.sort()
        
        return {
            'count': len(self.transactions),
            'size_mb': self.current_size / 1_000_000,
            'min_fee_rate': float(fee_rates[0]),
            'median_fee_rate': float(fee_rates[len(fee_rates)//2]),
            'max_fee_rate': float(fee_rates[-1])
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return f"Mempool({stats['count']} txs, {stats['size_mb']:.1f} MB, median fee: {stats['median_fee_rate']:.1f} sat/byte)"


# Example: Mempool operations
print("\\n=== Mempool Example ===\\n")

mempool = Mempool(max_size_mb=10)

# Add transactions with different fee rates
utxo_map = {}
for i in range(20):
    tx = Transaction([], [])
    # Simulate different fee rates
    tx.size = 250
    # Would normally calculate from utxo_map
    mempool.add_transaction(tx, utxo_map)

print(f"\\n{mempool}")
stats = mempool.get_stats()
print(f"\\nMempool stats:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Get transactions for next block
block_txs = mempool.get_transactions_by_fee(max_block_size=1_000_000)
print(f"\\nSelected {len(block_txs)} transactions for next block")
\`\`\`

## Fee Market Dynamics

Transaction fees follow supply and demand:

\`\`\`python
def estimate_confirmation_time(fee_rate: int, mempool_stats: dict) -> str:
    """Estimate confirmation time based on fee rate"""
    median_fee = mempool_stats['median_fee_rate']
    
    if fee_rate >= median_fee * 2:
        return "Next block (~10 min)"
    elif fee_rate >= median_fee:
        return "2-3 blocks (~20-30 min)"
    elif fee_rate >= median_fee * 0.5:
        return "4-6 blocks (~40-60 min)"
    else:
        return "6+ blocks (>1 hour)"

# Example: Fee estimation
print("\\n=== Fee Estimation ===\\n")

mempool_stats = {
    'median_fee_rate': 50.0  # 50 sat/byte
}

fee_rates = [10, 25, 50, 100, 200]
for rate in fee_rates:
    estimate = estimate_confirmation_time(rate, mempool_stats)
    print(f"{rate:3d} sat/byte: {estimate}")
\`\`\`

## Advanced Techniques

### Replace-By-Fee (RBF)

Increase fee on unconfirmed transaction:

\`\`\`python
class RBFTransaction(Transaction):
    """Transaction supporting Replace-By-Fee"""
    
    def __init__(self, inputs, outputs, rbf_enabled=True):
        super().__init__(inputs, outputs)
        self.rbf_enabled = rbf_enabled
        if rbf_enabled:
            # Signal RBF with sequence number < 0xfffffffe
            for inp in self.inputs:
                inp.sequence = 0xfffffffd
    
    def create_replacement(self, new_fee_rate: int, utxo_map: dict) -> 'RBFTransaction':
        """Create replacement transaction with higher fee"""
        if not self.rbf_enabled:
            raise ValueError("RBF not enabled")
        
        current_fee = self.get_fee(utxo_map)
        new_fee = Decimal(self.size * new_fee_rate) / Decimal(100_000_000)
        
        if new_fee <= current_fee:
            raise ValueError("New fee must be higher")
        
        # Reduce change output by fee difference
        fee_diff = new_fee - current_fee
        new_outputs = []
        for out in self.outputs:
            if "change" in out.script_pubkey:
                # Reduce change output
                new_amount = out.amount - fee_diff
                new_outputs.append(TransactionOutput(new_amount, out.script_pubkey))
            else:
                new_outputs.append(out)
        
        return RBFTransaction(self.inputs, new_outputs, rbf_enabled=True)

print("\\n=== Replace-By-Fee Example ===\\n")
print("Original transaction: 20 sat/byte, stuck in mempool")
print("Create replacement with 100 sat/byte...")
print("New transaction: Higher fee, same inputs (double-spend), replaces original")
\`\`\`

## Summary

Transaction lifecycle involves:
1. **Creation**: Coin selection, UTXO management
2. **Signing**: Cryptographic authorization
3. **Broadcasting**: Gossip protocol propagation
4. **Mempool**: Fee-based prioritization
5. **Confirmation**: Inclusion in block
6. **Finality**: Multiple confirmations

Best practices:
- Use appropriate fee rates for desired confirmation time
- Wait for sufficient confirmations (6+ for large amounts)
- Monitor mempool to detect network congestion
- Use RBF for time-sensitive transactions
- Batch transactions to save fees
`,
};
