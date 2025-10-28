export const blockchainExplorersDataAnalysis = {
  title: 'Blockchain Explorers & Data Analysis',
  id: 'blockchain-explorers-data-analysis',
  content: `
# Blockchain Explorers & Data Analysis

## Introduction

Blockchain explorers are the "Google" of blockchains—they index blockchain data into searchable databases, making it accessible to users, developers, and analysts. Without explorers, users would need to run full nodes and manually parse blockchain data to check transaction status or address balances.

**Popular explorers:**
- Bitcoin: blockchain.com, blockchair.com
- Ethereum: etherscan.io, ethplorer.io
- Multi-chain: blockchair.com, blockcypher.com

This section covers how explorers work, how to build one, and how to analyze on-chain data for insights.

## Block Explorer Architecture

### System Components

\`\`\`
┌─────────────────────────────────────────────┐
│          Block Explorer Stack                │
├─────────────────────────────────────────────┤
│  Frontend (React/Next.js)                   │
│    ↓                                        │
│  API Layer (REST/GraphQL)                   │
│    ↓                                        │
│  Application Server (Node/Python)            │
│    ↓                                        │
│  Database (PostgreSQL/MongoDB)               │
│    ↓                                        │
│  Indexer Service (Background workers)        │
│    ↓                                        │
│  Blockchain Node (Bitcoin Core/Geth)         │
└─────────────────────────────────────────────┘
\`\`\`

### Indexer Service

The indexer continuously monitors the blockchain and stores data in the database:

\`\`\`python
import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Block:
    """Blockchain block"""
    height: int
    hash: str
    prev_hash: str
    timestamp: int
    miner: str
    tx_count: int
    size: int

@dataclass
class Transaction:
    """Blockchain transaction"""
    txid: str
    block_height: int
    block_hash: str
    timestamp: int
    inputs: List[dict]
    outputs: List[dict]
    fee: float
    size: int

class BlockchainIndexer:
    """Indexes blockchain data into database"""
    
    def __init__(self, node_url: str, db_connection):
        self.node_url = node_url
        self.db = db_connection
        self.last_indexed_height = self._get_last_indexed_height()
        self.is_running = False
    
    def _get_last_indexed_height(self) -> int:
        """Get last indexed block height from database"""
        # Query database for max block height
        result = self.db.query("SELECT MAX(height) FROM blocks")
        return result or 0
    
    def start(self):
        """Start indexing daemon"""
        self.is_running = True
        print(f"Starting indexer from block {self.last_indexed_height}...")
        
        while self.is_running:
            try:
                # Get next block from node
                next_height = self.last_indexed_height + 1
                block = self._fetch_block_from_node(next_height)
                
                if block:
                    self._index_block(block)
                    self.last_indexed_height = next_height
                    print(f"Indexed block {next_height} ({block.tx_count} txs)")
                else:
                    # No new block yet, wait
                    time.sleep(10)
            
            except Exception as e:
                print(f"Error indexing block {next_height}: {e}")
                time.sleep(5)
    
    def _fetch_block_from_node(self, height: int) -> Optional[Block]:
        """Fetch block from blockchain node"""
        # In reality: RPC call to node
        # Example: requests.post(node_url, json={"method": "getblock", "params": [height]})
        
        # Simulated block
        if height <= 100:  # Simulate blockchain with 100 blocks
            return Block(
                height=height,
                hash=hashlib.sha256(f"block{height}".encode()).hexdigest(),
                prev_hash=hashlib.sha256(f"block{height-1}".encode()).hexdigest() if height > 0 else "0"*64,
                timestamp=int(time.time()) - (100 - height) * 600,
                miner=f"miner_{height % 5}",
                tx_count=10 + (height % 50),
                size=1000000
            )
        return None
    
    def _index_block(self, block: Block):
        """Index block and its transactions into database"""
        # Insert block
        self.db.insert("blocks", {
            "height": block.height,
            "hash": block.hash,
            "prev_hash": block.prev_hash,
            "timestamp": block.timestamp,
            "miner": block.miner,
            "tx_count": block.tx_count,
            "size": block.size
        })
        
        # Fetch and index transactions
        transactions = self._fetch_transactions_from_node(block.hash)
        for tx in transactions:
            self._index_transaction(tx)
    
    def _fetch_transactions_from_node(self, block_hash: str) -> List[Transaction]:
        """Fetch transactions in block from node"""
        # In reality: RPC call to get block transactions
        # Simulated transactions
        return []
    
    def _index_transaction(self, tx: Transaction):
        """Index transaction into database"""
        # Insert transaction
        self.db.insert("transactions", {
            "txid": tx.txid,
            "block_height": tx.block_height,
            "block_hash": tx.block_hash,
            "timestamp": tx.timestamp,
            "fee": tx.fee,
            "size": tx.size
        })
        
        # Index inputs (update address balances)
        for inp in tx.inputs:
            self._update_address_balance(inp['address'], -inp['amount'])
        
        # Index outputs
        for out in tx.outputs:
            self.db.insert("outputs", {
                "txid": tx.txid,
                "vout": out['index'],
                "address": out['address'],
                "amount": out['amount'],
                "spent": False
            })
            self._update_address_balance(out['address'], out['amount'])
    
    def _update_address_balance(self, address: str, delta: float):
        """Update address balance"""
        # Upsert address balance
        current = self.db.query(f"SELECT balance FROM addresses WHERE address='{address}'")
        new_balance = (current or 0) + delta
        
        self.db.upsert("addresses", {
            "address": address,
            "balance": new_balance,
            "tx_count": "tx_count + 1"
        })
    
    def handle_reorg(self, fork_point: int):
        """Handle blockchain reorganization"""
        print(f"⚠️  Reorg detected! Rolling back to block {fork_point}")
        
        # Delete blocks after fork point
        self.db.delete(f"DELETE FROM blocks WHERE height > {fork_point}")
        self.db.delete(f"DELETE FROM transactions WHERE block_height > {fork_point}")
        
        # Reindex from fork point
        self.last_indexed_height = fork_point
        print(f"Reindexing from block {fork_point}...")
    
    def stop(self):
        """Stop indexing daemon"""
        self.is_running = False
        print("Indexer stopped")


# Simulated database
class SimpleDB:
    """Simple in-memory database for demo"""
    def __init__(self):
        self.tables = {}
    
    def insert(self, table: str, data: dict):
        if table not in self.tables:
            self.tables[table] = []
        self.tables[table].append(data)
    
    def query(self, sql: str):
        # Simplified query simulation
        return 0
    
    def upsert(self, table: str, data: dict):
        self.insert(table, data)
    
    def delete(self, sql: str):
        pass


# Example: Run indexer
print("=== Blockchain Indexer Example ===\\n")

db = SimpleDB()
indexer = BlockchainIndexer("http://localhost:8332", db)

# Index first 5 blocks (simulation)
for i in range(5):
    block = indexer._fetch_block_from_node(i)
    if block:
        indexer._index_block(block)

print(f"\\nIndexed {len(db.tables.get('blocks', []))} blocks")
\`\`\`

### Database Schema

\`\`\`sql
-- Blocks table
CREATE TABLE blocks (
    height BIGINT PRIMARY KEY,
    hash VARCHAR(64) UNIQUE NOT NULL,
    prev_hash VARCHAR(64) NOT NULL,
    timestamp BIGINT NOT NULL,
    miner VARCHAR(64),
    tx_count INTEGER NOT NULL,
    size BIGINT NOT NULL,
    difficulty DECIMAL(40, 20),
    nonce BIGINT,
    INDEX idx_timestamp (timestamp),
    INDEX idx_miner (miner)
);

-- Transactions table
CREATE TABLE transactions (
    txid VARCHAR(64) PRIMARY KEY,
    block_height BIGINT NOT NULL,
    block_hash VARCHAR(64) NOT NULL,
    timestamp BIGINT NOT NULL,
    input_count INTEGER NOT NULL,
    output_count INTEGER NOT NULL,
    fee DECIMAL(16, 8) NOT NULL,
    size INTEGER NOT NULL,
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp),
    FOREIGN KEY (block_height) REFERENCES blocks(height)
);

-- Transaction outputs (UTXOs)
CREATE TABLE outputs (
    txid VARCHAR(64) NOT NULL,
    vout INTEGER NOT NULL,
    address VARCHAR(64) NOT NULL,
    amount DECIMAL(16, 8) NOT NULL,
    spent BOOLEAN DEFAULT FALSE,
    spent_by_txid VARCHAR(64),
    PRIMARY KEY (txid, vout),
    INDEX idx_address (address),
    INDEX idx_spent (spent),
    FOREIGN KEY (txid) REFERENCES transactions(txid)
);

-- Addresses table (aggregated data)
CREATE TABLE addresses (
    address VARCHAR(64) PRIMARY KEY,
    balance DECIMAL(16, 8) DEFAULT 0,
    tx_count BIGINT DEFAULT 0,
    first_seen_block BIGINT,
    last_active_block BIGINT,
    INDEX idx_balance (balance)
);
\`\`\`

## Building a REST API

\`\`\`python
from flask import Flask, jsonify, request

app = Flask(__name__)

class ExplorerAPI:
    """Block explorer REST API"""
    
    def __init__(self, db):
        self.db = db
    
    def get_block_by_height(self, height: int) -> dict:
        """Get block by height"""
        block = self.db.query(f"SELECT * FROM blocks WHERE height = {height}")
        if not block:
            return {"error": "Block not found"}, 404
        
        # Get transactions in block
        transactions = self.db.query(f"SELECT * FROM transactions WHERE block_height = {height}")
        
        return {
            "height": block['height'],
            "hash": block['hash'],
            "timestamp": block['timestamp'],
            "tx_count": block['tx_count'],
            "size": block['size'],
            "transactions": [tx['txid'] for tx in transactions]
        }
    
    def get_transaction(self, txid: str) -> dict:
        """Get transaction details"""
        tx = self.db.query(f"SELECT * FROM transactions WHERE txid = '{txid}'")
        if not tx:
            return {"error": "Transaction not found"}, 404
        
        # Get inputs and outputs
        outputs = self.db.query(f"SELECT * FROM outputs WHERE txid = '{txid}'")
        
        return {
            "txid": tx['txid'],
            "block_height": tx['block_height'],
            "timestamp": tx['timestamp'],
            "fee": tx['fee'],
            "outputs": outputs
        }
    
    def get_address(self, address: str) -> dict:
        """Get address information"""
        addr = self.db.query(f"SELECT * FROM addresses WHERE address = '{address}'")
        if not addr:
            return {"error": "Address not found"}, 404
        
        # Get recent transactions
        recent_txs = self.db.query(f"""
            SELECT t.txid, t.timestamp, o.amount 
            FROM transactions t
            JOIN outputs o ON t.txid = o.txid
            WHERE o.address = '{address}'
            ORDER BY t.timestamp DESC
            LIMIT 10
        """)
        
        return {
            "address": address,
            "balance": addr['balance'],
            "tx_count": addr['tx_count'],
            "recent_transactions": recent_txs
        }
    
    def search(self, query: str) -> dict:
        """Search for block/transaction/address"""
        # Try as block height
        if query.isdigit():
            return self.get_block_by_height(int(query))
        
        # Try as block hash or transaction ID (64 hex chars)
        if len(query) == 64:
            # Check blocks
            block = self.db.query(f"SELECT * FROM blocks WHERE hash = '{query}'")
            if block:
                return self.get_block_by_height(block['height'])
            
            # Check transactions
            return self.get_transaction(query)
        
        # Try as address
        return self.get_address(query)


# Flask routes
db = SimpleDB()  # Replace with real database
api = ExplorerAPI(db)

@app.route('/api/block/<int:height>')
def block_by_height(height):
    return jsonify(api.get_block_by_height(height))

@app.route('/api/tx/<txid>')
def transaction(txid):
    return jsonify(api.get_transaction(txid))

@app.route('/api/address/<address>')
def address(address):
    return jsonify(api.get_address(address))

@app.route('/api/search')
def search():
    query = request.args.get('q', '')
    return jsonify(api.search(query))

print("\\n=== Block Explorer API Endpoints ===")
print("GET /api/block/<height>")
print("GET /api/tx/<txid>")
print("GET /api/address/<address>")
print("GET /api/search?q=<query>")
\`\`\`

## On-Chain Analytics

### Address Clustering

Identify addresses controlled by same entity:

\`\`\`python
class AddressClustering:
    """Identify related addresses"""
    
    def __init__(self, db):
        self.db = db
        self.clusters = {}  # address -> cluster_id
    
    def cluster_by_common_input(self):
        """
        Heuristic: Addresses used as inputs in same transaction
        likely belong to same wallet
        """
        # Get all transactions with multiple inputs
        txs = self.db.query("""
            SELECT txid FROM transactions WHERE input_count > 1
        """)
        
        for tx in txs:
            # Get input addresses
            inputs = self.db.query(f"""
                SELECT DISTINCT address FROM inputs WHERE txid = '{tx['txid']}'
            """)
            
            addresses = [inp['address'] for inp in inputs]
            
            # Assign same cluster ID to all addresses
            cluster_id = min(self.clusters.get(addr, addr) for addr in addresses)
            for addr in addresses:
                self.clusters[addr] = cluster_id
        
        return self.clusters
    
    def get_cluster_balance(self, cluster_id: str) -> float:
        """Get total balance of all addresses in cluster"""
        addresses = [addr for addr, cid in self.clusters.items() if cid == cluster_id]
        
        total = 0.0
        for address in addresses:
            balance = self.db.query(f"SELECT balance FROM addresses WHERE address = '{address}'")
            total += balance or 0.0
        
        return total


# Example: Address clustering
print("\\n=== Address Clustering Example ===\\n")

# Simulated transaction with multiple inputs
print("Transaction tx123 has inputs from:")
print("  - 0xaaaa1111")
print("  - 0xbbbb2222")
print("  - 0xcccc3333")
print("\\nConclusion: These addresses likely belong to same wallet")
print("Cluster them together for analysis")
\`\`\`

### Transaction Graph Analysis

\`\`\`python
class TransactionGraph:
    """Analyze flow of funds"""
    
    def __init__(self, db):
        self.db = db
    
    def trace_funds(self, start_address: str, max_depth: int = 5) -> dict:
        """Trace where funds from address went"""
        visited = set()
        graph = {}
        
        def dfs(address: str, depth: int):
            if depth > max_depth or address in visited:
                return
            
            visited.add(address)
            
            # Get outgoing transactions
            txs = self.db.query(f"""
                SELECT t.txid, o.address, o.amount
                FROM transactions t
                JOIN outputs o ON t.txid = o.txid
                WHERE t.txid IN (
                    SELECT txid FROM inputs WHERE address = '{address}'
                )
            """)
            
            graph[address] = []
            for tx in txs:
                graph[address].append({
                    'to': tx['address'],
                    'amount': tx['amount'],
                    'txid': tx['txid']
                })
                dfs(tx['address'], depth + 1)
        
        dfs(start_address, 0)
        return graph
    
    def detect_mixer(self, address: str) -> bool:
        """Detect if address is likely a mixing service"""
        # Mixers typically have:
        # 1. Many small inputs from different addresses
        # 2. Many small outputs to different addresses
        # 3. High transaction velocity
        
        stats = self.db.query(f"""
            SELECT 
                COUNT(DISTINCT input_address) as unique_inputs,
                COUNT(DISTINCT output_address) as unique_outputs,
                AVG(amount) as avg_amount
            FROM transactions 
            WHERE address = '{address}'
        """)
        
        # Heuristic thresholds
        if stats['unique_inputs'] > 100 and stats['unique_outputs'] > 100:
            return True
        
        return False


# Example: Flow of funds analysis
print("\\n=== Flow of Funds Tracking ===\\n")
print("Tracing funds from suspicious address...")
print("0xHACKER → 0xMIXER → 0xEXCHANGE → (trail goes cold)")
print("\\nMixing services break transaction graph analysis")
\`\`\`

### Rich List Analysis

\`\`\`python
def get_rich_list(db, limit: int = 100) -> List[dict]:
    """Get addresses with highest balances"""
    return db.query(f"""
        SELECT address, balance, tx_count
        FROM addresses
        ORDER BY balance DESC
        LIMIT {limit}
    """)

def analyze_wealth_distribution(db) -> dict:
    """Analyze wealth concentration"""
    total_supply = db.query("SELECT SUM(balance) FROM addresses")
    
    # Top 1% of addresses
    top_1_percent_count = db.query("SELECT COUNT(*) FROM addresses") // 100
    top_1_percent_wealth = db.query(f"""
        SELECT SUM(balance) FROM (
            SELECT balance FROM addresses ORDER BY balance DESC LIMIT {top_1_percent_count}
        )
    """)
    
    return {
        'total_supply': total_supply,
        'top_1_percent_count': top_1_percent_count,
        'top_1_percent_wealth': top_1_percent_wealth,
        'concentration_ratio': top_1_percent_wealth / total_supply
    }

print("\\n=== Wealth Distribution Analysis ===")
print("Top 1% of addresses control X% of total supply")
print("Useful for understanding network decentralization")
\`\`\`

## Performance Optimization

### Caching Strategy

\`\`\`python
import redis
import json

class ExplorerCache:
    """Redis caching layer"""
    
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)
        self.ttl = 60  # 60 seconds
    
    def get_block(self, height: int) -> Optional[dict]:
        """Get block from cache"""
        cached = self.redis.get(f"block:{height}")
        if cached:
            return json.loads(cached)
        return None
    
    def set_block(self, height: int, block: dict):
        """Cache block"""
        self.redis.setex(
            f"block:{height}",
            self.ttl,
            json.dumps(block)
        )
    
    def invalidate_block(self, height: int):
        """Invalidate cached block (e.g., after reorg)"""
        self.redis.delete(f"block:{height}")

print("\\n=== Caching Strategy ===")
print("- Cache recent blocks (last 100)")
print("- Cache popular addresses")
print("- Invalidate on reorg")
print("- Use CDN for static assets")
\`\`\`

## Summary

Block explorers enable:
- **User interface**: Search transactions, check balances
- **Developer tools**: API for blockchain data
- **Analytics**: On-chain insights and metrics
- **Transparency**: Public audit of blockchain

Building an explorer requires:
1. **Indexer**: Parse blockchain into database
2. **API**: Expose data via REST/GraphQL
3. **Frontend**: User-friendly interface
4. **Caching**: Performance optimization
5. **Reorg handling**: Maintain data consistency

On-chain analytics reveals:
- Transaction flow and patterns
- Address clustering and entity identification
- Wealth distribution and concentration
- Network activity and adoption metrics
`,
};
