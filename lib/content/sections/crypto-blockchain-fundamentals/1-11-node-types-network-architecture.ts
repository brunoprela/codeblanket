export const nodeTypesNetworkArchitecture = {
  title: 'Node Types & Network Architecture',
  id: 'node-types-network-architecture',
  content: `
# Node Types & Network Architecture

## Introduction

The blockchain network is a peer-to-peer (P2P) network of nodes that collectively maintain the distributed ledger. Understanding node types and network architecture is crucial for:
- Choosing the right node type for your needs
- Understanding security trade-offs
- Building scalable blockchain applications
- Defending against network-level attacks

## Node Types

### Full Nodes

**Full nodes download and validate the entire blockchain.**

\`\`\`python
class FullNode:
    """Bitcoin/Ethereum full node"""
    
    def __init__(self):
        self.blockchain = []
        self.utxo_set = {}  # Current UTXO set
        self.mempool = {}
        self.peers = []
        self.storage_gb = 500  # Current blockchain size
    
    def sync_blockchain(self):
        """Download and validate entire blockchain"""
        print("Downloading blockchain...")
        for height in range(1, self.get_network_height() + 1):
            block = self.download_block(height)
            if self.validate_block(block):
                self.blockchain.append(block)
                self.update_utxo_set(block)
            else:
                raise ValueError(f"Invalid block at height {height}")
    
    def validate_block(self, block) -> bool:
        """Validate all transactions in block"""
        # Check proof-of-work
        if not self.check_proof_of_work(block):
            return False
        
        # Validate every transaction
        for tx in block.transactions:
            if not self.validate_transaction(tx):
                return False
        
        return True
    
    def validate_transaction(self, tx) -> bool:
        """Full validation of transaction"""
        # Verify signatures
        for inp in tx.inputs:
            if not self.verify_signature(inp):
                return False
        
        # Verify inputs exist in UTXO set
        for inp in tx.inputs:
            utxo_key = f"{inp.txid}:{inp.vout}"
            if utxo_key not in self.utxo_set:
                return False
        
        # Verify outputs don't exceed inputs
        if sum(out.amount for out in tx.outputs) > sum(self.utxo_set[f"{inp.txid}:{inp.vout}"].amount for inp in tx.inputs):
            return False
        
        return True
    
    def update_utxo_set(self, block):
        """Update UTXO set with block's transactions"""
        for tx in block.transactions:
            # Remove spent UTXOs
            for inp in tx.inputs:
                key = f"{inp.txid}:{inp.vout}"
                if key in self.utxo_set:
                    del self.utxo_set[key]
            
            # Add new UTXOs
            for i, out in enumerate(tx.outputs):
                key = f"{tx.txid}:{i}"
                self.utxo_set[key] = out
    
    def check_proof_of_work(self, block) -> bool:
        """Verify block meets difficulty target"""
        # Simplified
        return True
    
    def verify_signature(self, inp) -> bool:
        """Verify ECDSA signature"""
        # Simplified
        return True
    
    def download_block(self, height: int):
        """Download block from network"""
        # Simplified
        return None
    
    def get_network_height(self) -> int:
        """Get current blockchain height from peers"""
        return 100  # Simplified

# Example: Full node capabilities
print("=== Full Node ===")
print("Storage: 500GB+ (entire blockchain)")
print("Bandwidth: High (downloads all blocks)")
print("Validation: Complete (validates every transaction)")
print("Trust: Trustless (doesn't trust peers)")
print("Use case: Maximum security, supporting network")
\`\`\`

**Characteristics:**
- **Storage**: 500GB+ for Bitcoin, 1TB+ for Ethereum
- **Bandwidth**: 200GB+/month
- **Validation**: Complete validation of all transactions
- **Security**: Maximum security, fully trustless
- **Cost**: ~$50-100/month for VPS

### Light Nodes (SPV)

**SPV (Simplified Payment Verification) nodes download only block headers.**

\`\`\`python
class SPVNode:
    """Simplified Payment Verification node"""
    
    def __init__(self):
        self.headers = []  # Only block headers (80 bytes each)
        self.relevant_txs = {}  # Only transactions relevant to wallet
        self.peers = []
        self.storage_mb = 80  # Just headers (~80MB for Bitcoin)
    
    def sync_headers(self):
        """Download only block headers"""
        print("Downloading headers...")
        for height in range(1, self.get_network_height() + 1):
            header = self.download_header(height)
            if self.validate_header(header):
                self.headers.append(header)
    
    def validate_header(self, header) -> bool:
        """Validate header (proof-of-work only)"""
        # Check proof-of-work
        if not self.check_pow(header):
            return False
        
        # Check links to previous header
        if len(self.headers) > 0:
            if header.prev_hash != self.headers[-1].hash:
                return False
        
        return True
    
    def verify_transaction(self, tx, merkle_proof) -> bool:
        """Verify transaction using Merkle proof"""
        # Get block header
        header = self.headers[tx.block_height]
        
        # Verify Merkle proof
        current = hash(tx)
        for sibling, position in merkle_proof:
            if position == 'left':
                current = hash(sibling + current)
            else:
                current = hash(current + sibling)
        
        # Check if matches merkle root in header
        return current == header.merkle_root
    
    def request_merkle_proof(self, txid: str) -> list:
        """Request Merkle proof from full node"""
        # Ask peer for Merkle proof
        peer = self.peers[0]
        return peer.get_merkle_proof(txid)
    
    def check_pow(self, header) -> bool:
        """Check proof-of-work"""
        return True  # Simplified
    
    def download_header(self, height: int):
        """Download header from peer"""
        return None  # Simplified
    
    def get_network_height(self) -> int:
        return 100  # Simplified
    
    def hash(self, data) -> str:
        import hashlib
        return hashlib.sha256(str(data).encode()).hexdigest()

# Example: SPV node capabilities
print("\\n=== SPV Light Node ===")
print("Storage: ~80MB (just headers)")
print("Bandwidth: Low (only headers + relevant txs)")
print("Validation: Partial (only PoW, not transactions)")
print("Trust: Trusts miners (assumes longest chain is valid)")
print("Use case: Mobile wallets, low-resource devices")
\`\`\`

**Characteristics:**
- **Storage**: ~80MB for Bitcoin (80 bytes × 800K blocks)
- **Bandwidth**: Minimal (<1GB/month)
- **Validation**: Only proof-of-work, not transactions
- **Security**: Trusts miners, vulnerable to eclipse attacks
- **Cost**: Free (can run on phone)

### Archive Nodes

**Archive nodes store complete historical state.**

\`\`\`python
class ArchiveNode:
    """Archive node with full historical state"""
    
    def __init__(self):
        self.blockchain = []
        self.state_history = {}  # State at every block height
        self.storage_tb = 12  # Ethereum archive node
    
    def store_state_snapshot(self, height: int, state: dict):
        """Store complete state at block height"""
        self.state_history[height] = state.copy()
    
    def get_account_balance_at(self, address: str, height: int) -> float:
        """Get account balance at specific block height"""
        if height in self.state_history:
            return self.state_history[height].get(address, {}).get('balance', 0)
        raise ValueError(f"State not available for height {height}")
    
    def query_historical_data(self, address: str, from_height: int, to_height: int) -> list:
        """Query historical state range"""
        results = []
        for height in range(from_height, to_height + 1):
            if height in self.state_history:
                results.append({
                    'height': height,
                    'balance': self.get_account_balance_at(address, height)
                })
        return results

# Example: Archive node capabilities
print("\\n=== Archive Node ===")
print("Storage: 12TB+ for Ethereum (all historical states)")
print("Bandwidth: High (same as full node)")
print("Validation: Complete (same as full node)")
print("Historical queries: YES (can query any past state)")
print("Use case: Block explorers, data analytics, dApp development")
\`\`\`

**Characteristics:**
- **Storage**: 12TB+ for Ethereum, 500GB+ for Bitcoin
- **Bandwidth**: Same as full node
- **Validation**: Complete
- **Historical data**: Every state transition stored
- **Cost**: $200+/month for storage

### Pruned Nodes

**Pruned nodes validate everything but discard old block data.**

\`\`\`python
class PrunedNode:
    """Pruned full node"""
    
    def __init__(self, prune_target_gb: int = 5):
        self.blockchain = []
        self.utxo_set = {}
        self.prune_target = prune_target_gb * 1_000_000_000  # Convert to bytes
        self.current_size = 0
    
    def add_block(self, block):
        """Add block and prune if necessary"""
        self.blockchain.append(block)
        self.current_size += block.size
        
        # Prune old blocks if exceeded target
        if self.current_size > self.prune_target:
            self.prune_old_blocks()
    
    def prune_old_blocks(self):
        """Delete old block data, keep UTXO set"""
        # Keep last N blocks for reorg protection
        keep_blocks = 288  # ~2 days of Bitcoin blocks
        
        while len(self.blockchain) > keep_blocks and self.current_size > self.prune_target:
            old_block = self.blockchain.pop(0)
            self.current_size -= old_block.size
            print(f"Pruned block {old_block.height}")
    
    def can_serve_historical_blocks(self) -> bool:
        """Can this node serve old blocks to peers?"""
        return False  # Pruned nodes can't serve full historical data

# Example: Pruned node
print("\\n=== Pruned Node ===")
print("Storage: ~5GB (configurable)")
print("Bandwidth: Same as full node during sync")
print("Validation: Complete (validates everything)")
print("Historical data: NO (old blocks deleted)")
print("Use case: Personal validation with limited storage")
\`\`\`

### Comparison Table

\`\`\`python
import pandas as pd

node_comparison = pd.DataFrame({
    'Type': ['Full Node', 'Archive Node', 'Pruned Node', 'SPV Light Node'],
    'Storage': ['500GB', '12TB', '5GB', '80MB'],
    'Validates Txs': ['Yes', 'Yes', 'Yes', 'No'],
    'Historical Data': ['Recent', 'All', 'Minimal', 'None'],
    'Trustless': ['Yes', 'Yes', 'Yes', 'No'],
    'Mobile Friendly': ['No', 'No', 'No', 'Yes'],
    'Monthly Cost': ['$50-100', '$200+', '$20-50', 'Free']
})

print("\\n=== Node Type Comparison ===\\n")
print(node_comparison.to_string(index=False))
\`\`\`

## P2P Network Architecture

### Gossip Protocol

Blockchains use gossip (flooding) protocol for message propagation:

\`\`\`python
import random
import time
from typing import Set, List

class P2PNode:
    """Peer-to-peer network node"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.peers: List['P2PNode'] = []
        self.seen_messages: Set[str] = set()
        self.max_peers = 8  # Bitcoin default
    
    def connect_to_peer(self, peer: 'P2PNode'):
        """Establish connection to peer"""
        if len(self.peers) < self.max_peers and peer not in self.peers:
            self.peers.append(peer)
            peer.peers.append(self)
            print(f"{self.node_id} connected to {peer.node_id}")
    
    def broadcast_message(self, message: str, message_id: str):
        """Broadcast message to all peers (gossip)"""
        if message_id in self.seen_messages:
            return  # Already seen
        
        print(f"[{self.node_id}] Broadcasting: {message}")
        self.seen_messages.add(message_id)
        
        # Forward to all peers
        for peer in self.peers:
            peer.receive_message(message, message_id, from_node=self.node_id)
    
    def receive_message(self, message: str, message_id: str, from_node: str):
        """Receive message from peer"""
        if message_id in self.seen_messages:
            return  # Already seen
        
        print(f"[{self.node_id}] Received from {from_node}: {message}")
        self.seen_messages.add(message_id)
        
        # Forward to other peers (not sender)
        for peer in self.peers:
            if peer.node_id != from_node:
                peer.receive_message(message, message_id, from_node=self.node_id)


# Example: Gossip propagation
print("\\n=== Gossip Protocol Simulation ===\\n")

# Create network of 10 nodes
nodes = [P2PNode(f"Node{i}") for i in range(10)]

# Create partial mesh topology
for i in range(len(nodes)):
    # Connect to 3-4 random peers
    num_connections = random.randint(3, 4)
    possible_peers = [n for n in nodes if n != nodes[i]]
    peers_to_connect = random.sample(possible_peers, min(num_connections, len(possible_peers)))
    
    for peer in peers_to_connect:
        if peer not in nodes[i].peers:
            nodes[i].connect_to_peer(peer)

print("\\nNetwork created. Broadcasting message from Node0...")
nodes[0].broadcast_message("New block found!", "msg_001")

print(f"\\nMessage reached {sum(1 for n in nodes if 'msg_001' in n.seen_messages)}/{len(nodes)} nodes")
\`\`\`

### Node Discovery

**How nodes find peers:**

\`\`\`python
class NodeDiscovery:
    """Node discovery mechanisms"""
    
    def __init__(self):
        self.known_peers = set()
        self.dns_seeds = [
            "seed.bitcoin.sipa.be",
            "dnsseed.bluematt.me",
            "seed.bitcoinstats.com"
        ]
    
    def discover_peers(self) -> list:
        """Discover peers using multiple methods"""
        peers = []
        
        # 1. DNS seeds (hardcoded)
        dns_peers = self.query_dns_seeds()
        peers.extend(dns_peers)
        
        # 2. Peer exchange (ask existing peers for their peers)
        if self.known_peers:
            pex_peers = self.peer_exchange()
            peers.extend(pex_peers)
        
        # 3. Hardcoded nodes (fallback)
        fallback_peers = self.get_hardcoded_nodes()
        peers.extend(fallback_peers)
        
        return peers
    
    def query_dns_seeds(self) -> list:
        """Query DNS seeds for peer IP addresses"""
        # In reality: DNS query returning A records
        print("Querying DNS seeds...")
        return [
            ("1.2.3.4", 8333),
            ("5.6.7.8", 8333),
        ]
    
    def peer_exchange(self) -> list:
        """Ask peers for their peer list"""
        print("Requesting peers from connected nodes...")
        peers = []
        for peer in list(self.known_peers)[:3]:  # Ask first 3 peers
            # Send "getaddr" message
            peer_list = self.request_peers_from(peer)
            peers.extend(peer_list)
        return peers
    
    def request_peers_from(self, peer) -> list:
        """Request peer list from specific peer"""
        # In reality: Send P2P message "getaddr"
        return []
    
    def get_hardcoded_nodes(self) -> list:
        """Fallback to hardcoded seed nodes"""
        return [
            ("seed.bitcoin.sipa.be", 8333)
        ]

# Example: Node discovery
print("\\n=== Node Discovery ===\\n")
discovery = NodeDiscovery()
peers = discovery.discover_peers()
print(f"Discovered {len(peers)} potential peers")
\`\`\`

## Network Attacks

### Eclipse Attack

Attacker isolates target node from honest network:

\`\`\`python
class EclipseAttack:
    """Simulate eclipse attack"""
    
    def __init__(self):
        self.attacker_nodes = []
        self.target_node = None
    
    def execute_attack(self, target: P2PNode, num_attacker_nodes: int = 20):
        """Execute eclipse attack"""
        print(f"\\n=== Eclipse Attack on {target.node_id} ===\\n")
        
        self.target_node = target
        
        # Step 1: Create many attacker-controlled nodes (Sybil attack)
        print(f"Creating {num_attacker_nodes} attacker nodes...")
        for i in range(num_attacker_nodes):
            attacker = P2PNode(f"Attacker{i}")
            self.attacker_nodes.append(attacker)
        
        # Step 2: Flood target's connection slots
        print(f"\\nFlooding {target.node_id}'s connection slots...")
        target.peers = []  # Disconnect existing peers
        
        for attacker in self.attacker_nodes[:target.max_peers]:
            target.connect_to_peer(attacker)
        
        print(f"{target.node_id} now connected to {len(target.peers)} attacker nodes")
        
        # Step 3: Target is now eclipsed
        print(f"\\nTarget eclipsed! All peers are attacker-controlled.")
        print("Attacker can now:")
        print("  - Feed target fake blockchain")
        print("  - Hide transactions from target")
        print("  - Double-spend against target")
        
        return True
    
    def defense_strategies(self):
        """Eclipse attack defenses"""
        print("\\n=== Eclipse Attack Defenses ===")
        print("1. Prefer outbound connections (node initiates, harder to poison)")
        print("2. Diverse peer selection (different IPs, ASNs, geographies)")
        print("3. Long-lived connections (maintain some old peers)")
        print("4. Anchor connections (never drop certain trusted peers)")
        print("5. Test before evicting (verify peer is responsive)")

# Example: Eclipse attack
attack = EclipseAttack()
target = P2PNode("VictimNode")
attack.execute_attack(target, num_attacker_nodes=20)
attack.defense_strategies()
\`\`\`

### Sybil Attack

Attacker creates many fake identities:

\`\`\`python
def demonstrate_sybil_attack():
    """Sybil attack: Create many fake nodes"""
    print("\\n=== Sybil Attack ===\\n")
    
    # Attacker creates 1000 nodes with one computer
    print("Attacker creates 1000 virtual nodes...")
    print("Honest network: 10,000 nodes")
    print("Attacker controls: 1,000 nodes (9% of network)")
    
    print("\\nWithout Sybil resistance:")
    print("  - 9% voting power")
    print("  - Can influence peer discovery")
    print("  - Can attempt eclipse attacks")
    
    print("\\nWith Proof-of-Work:")
    print("  - 1,000 nodes = 1,000 CPUs")
    print("  - Voting power based on hash rate, not node count")
    print("  - Creating more nodes doesn't help without more hardware")
    
    print("\\nWith Proof-of-Stake:")
    print("  - Voting power based on stake, not node count")
    print("  - Must acquire cryptocurrency to gain influence")

demonstrate_sybil_attack()
\`\`\`

## Running a Node

### Hardware Requirements

\`\`\`python
def calculate_node_requirements(node_type: str) -> dict:
    """Calculate hardware requirements"""
    
    requirements = {
        'full': {
            'storage_gb': 500,
            'ram_gb': 8,
            'bandwidth_gb_month': 200,
            'cpu_cores': 4,
            'monthly_cost_usd': 50
        },
        'archive': {
            'storage_gb': 12000,
            'ram_gb': 16,
            'bandwidth_gb_month': 200,
            'cpu_cores': 8,
            'monthly_cost_usd': 200
        },
        'pruned': {
            'storage_gb': 5,
            'ram_gb': 4,
            'bandwidth_gb_month': 200,
            'cpu_cores': 2,
            'monthly_cost_usd': 20
        },
        'spv': {
            'storage_gb': 0.1,
            'ram_gb': 1,
            'bandwidth_gb_month': 1,
            'cpu_cores': 1,
            'monthly_cost_usd': 0
        }
    }
    
    return requirements.get(node_type, {})

# Example: Node requirements
print("\\n=== Node Hardware Requirements ===\\n")
for node_type in ['full', 'archive', 'pruned', 'spv']:
    req = calculate_node_requirements(node_type)
    print(f"{node_type.upper()} NODE:")
    for key, value in req.items():
        print(f"  {key}: {value}")
    print()
\`\`\`

## Summary

Understanding nodes and network architecture:

**Node Types:**
- **Full nodes**: Complete validation, trustless
- **Archive nodes**: Full history, high storage
- **Pruned nodes**: Complete validation, low storage
- **SPV nodes**: Lightweight, reduced security

**Network:**
- **P2P topology**: Decentralized, no single point of failure
- **Gossip protocol**: Efficient message propagation
- **Node discovery**: DNS seeds + peer exchange
- **Security**: Defense against eclipse and Sybil attacks

**Running a node:**
- Supports network decentralization
- Enables trustless verification
- Required for mining/validation
- Resource requirements vary by type
`,
};
