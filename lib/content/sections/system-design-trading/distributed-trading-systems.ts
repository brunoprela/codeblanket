export const distributedTradingSystems = {
  title: 'Distributed Trading Systems',
  id: 'distributed-trading-systems',
  content: `
# Distributed Trading Systems

## Introduction

Modern trading firms operate globally across multiple exchanges, asset classes, and regions. A **Distributed Trading System** must handle:

- **Multi-region deployment**: Trading in US, Europe, Asia simultaneously
- **Network partitions**: Handling split-brain scenarios
- **Data consistency**: Ensuring order state synchronized across regions
- **Latency optimization**: Routing orders to nearest exchange
- **Failover**: Seamless transition when primary fails
- **Regulatory compliance**: Different rules per jurisdiction

### Why Distributed?

**Global markets**: NYSE (New York), LSE (London), TSE (Tokyo) all require local presence
**Redundancy**: No single point of failure
**Latency**: Lower latency by being near exchanges
**Scale**: Horizontal scaling for throughput

### The CAP Theorem Challenge

**CAP Theorem** states you can only have 2 of 3:
- **C**onsistency: All nodes see the same data
- **A**vailability**: System responds to requests
- **P**artition tolerance**: Works despite network failures

For trading: Typically choose **AP** (Availability + Partition tolerance) over consistency—better to keep trading with stale data than stop trading.

By the end of this section, you'll understand:
- Multi-region architecture patterns
- Clock synchronization (PTP, NTP)
- Distributed consensus (Raft, Paxos)
- Handling network partitions
- Global order routing
- Cross-region data replication

---

## Multi-Region Architecture

### Active-Active Architecture

\`\`\`
       US Region (Primary)              Europe Region (Active)              Asia Region (Active)
            |                                    |                                  |
    ┌───────┴────────┐                  ┌────────┴────────┐                ┌────────┴────────┐
    |                |                  |                 |                |                 |
  OMS-US        Data-US             OMS-EU          Data-EU            OMS-ASIA        Data-ASIA
    |                |                  |                 |                |                 |
    └────────────────┘                  └─────────────────┘                └─────────────────┘
            |                                    |                                  |
      NYSE/NASDAQ                            LSE/Eurex                          TSE/HKEX
      
  Global State Sync: 
  US ←→ Europe: 80ms
  US ←→ Asia: 150ms
  Europe ←→ Asia: 100ms
\`\`\`

**Advantages**:
- Low latency in each region (local orders fast)
- High availability (multiple active systems)
- Scalability (distribute load)

**Challenges**:
- Data consistency (eventual consistency model)
- Conflict resolution (two regions modify same data)
- Complex deployment

### Active-Passive Architecture

\`\`\`
       US Region (Primary)              Europe Region (Standby)
            |                                    |
    ┌───────┴────────┐                  ┌────────┴────────┐
    |                |                  |                 |
  OMS-US        Data-US             OMS-EU (Idle)    Data-EU (Replica)
    |                |                  |                 |
    └────────────────┘                  └─────────────────┘
            |                                    |
      All Exchanges                          (Failover)
\`\`\`

**Advantages**:
- Simpler consistency (single source of truth)
- Easier to reason about

**Disadvantages**:
- Higher latency (all orders through US)
- Single point of failure (until failover)
- Wasted capacity (passive idle)

---

## Clock Synchronization

### The Problem

Distributed systems need synchronized clocks for:
- **Ordering events**: Which order came first?
- **Timestamps**: When did trade execute?
- **Timeouts**: Has request expired?
- **Causality**: Event A caused event B

**Clock drift**: Hardware clocks drift ~50ppm (parts per million)
- 50ppm = 50μs per second = 4.3 seconds per day

### NTP (Network Time Protocol)

\`\`\`python
"""
NTP Client Implementation
Synchronize clock with time server
"""

import socket
import struct
import time

class NTPClient:
    """Simple NTP client"""
    
    NTP_SERVER = "pool.ntp.org"
    NTP_PORT = 123
    
    def get_ntp_time(self) -> float:
        """
        Query NTP server and calculate offset
        Returns: time offset in seconds
        """
        # NTP packet format (48 bytes)
        packet = bytearray(48)
        packet[0] = 0x1B  # NTP version 3, mode 3 (client)
        
        # Record send time
        t1 = time.time()
        
        # Send request
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        sock.sendto(packet, (self.NTP_SERVER, self.NTP_PORT))
        
        # Receive response
        data, _ = sock.recvfrom(1024)
        
        # Record receive time
        t4 = time.time()
        
        # Parse NTP timestamp (seconds since 1900-01-01)
        t2 = struct.unpack('!12I', data)[10] - 2208988800  # Server receive
        t3 = struct.unpack('!12I', data)[10] - 2208988800  # Server transmit
        
        # Calculate offset
        # offset = ((t2 - t1) + (t3 - t4)) / 2
        offset = ((t2 - t1) + (t3 - t4)) / 2
        
        return offset
    
    def sync_clock(self):
        """Continuously sync clock"""
        while True:
            offset = self.get_ntp_time()
            
            print(f"Clock offset: {offset*1000:.2f}ms")
            
            # Adjust system clock (requires root)
            # os.system(f"date -s '@{time.time() + offset}'")
            
            time.sleep(60)  # Sync every minute

# NTP accuracy: 1-50ms over internet
# Good enough for most trading (not HFT)
\`\`\`

### PTP (Precision Time Protocol)

For HFT, need microsecond accuracy:

\`\`\`
PTP Hierarchy:

Grandmaster Clock (GPS-synchronized)
    ↓ (PTP over Ethernet)
Boundary Clock 1
    ↓
Boundary Clock 2
    ↓
Slave Clocks (trading servers)

Accuracy: <1μs in LAN
\`\`\`

**PTP Hardware Support**:
- Network cards with hardware timestamping
- Reduces software latency from packet timestamping
- Cost: $500-2000 per NIC

\`\`\`python
"""
PTP synchronization monitoring
"""

class PTPMonitor:
    """Monitor PTP synchronization"""
    
    def check_sync_status(self) -> dict:
        """
        Check PTP sync status
        Uses ptp4l daemon
        """
        import subprocess
        
        # Query PTP status
        result = subprocess.run(
            ['pmc', '-u', '-b', 0', 'GET CURRENT_DATA_SET'],
            capture_output=True,
            text=True
        )
        
        # Parse offset
        lines = result.stdout.split('\\n')
        offset_line = [l for l in lines if 'offsetFromMaster' in l][0]
        offset_ns = int(offset_line.split()[-1])
        
        return {
            'offset_ns': offset_ns,
            'offset_us': offset_ns / 1000,
            'synchronized': abs(offset_ns) < 1000  # <1μs is good
        }

# Alert if offset > 10μs (indicates sync issue)
\`\`\`

---

## Distributed Consensus

### The Problem

Multiple nodes need to agree on order state:
- Order submitted in US, need Europe to know
- Position updated in Asia, need US to see
- Risk limit breached in Europe, need all regions to enforce

### Raft Consensus Algorithm

\`\`\`python
"""
Simplified Raft implementation
Distributed consensus for trading systems
"""

from enum import Enum
from typing import List, Optional
import time
import random

class NodeState(Enum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"

class LogEntry:
    """Entry in replicated log"""
    def __init__(self, term: int, command: dict):
        self.term = term
        self.command = command  # e.g., {"type": "order", "symbol": "AAPL", ...}

class RaftNode:
    """
    Raft consensus node
    Ensures all nodes agree on order of events
    """
    
    def __init__(self, node_id: int, peers: List[int]):
        self.node_id = node_id
        self.peers = peers
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.state = NodeState.FOLLOWER
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index = {peer: 1 for peer in peers}
        self.match_index = {peer: 0 for peer in peers}
        
        # Timing
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(150, 300) / 1000  # 150-300ms
    
    def append_entry(self, command: dict) -> bool:
        """
        Append command to log
        Only leader can append
        """
        if self.state != NodeState.LEADER:
            return False
        
        # Create log entry
        entry = LogEntry(self.current_term, command)
        self.log.append(entry)
        
        # Replicate to followers
        self.replicate_to_followers()
        
        return True
    
    def replicate_to_followers(self):
        """
        Replicate log entries to followers
        """
        for peer in self.peers:
            # Send AppendEntries RPC
            next_idx = self.next_index[peer]
            entries = self.log[next_idx:]
            
            # In production: Send over network
            # response = send_append_entries(peer, entries)
            
            # if response.success:
            #     self.next_index[peer] += len(entries)
            #     self.match_index[peer] = self.next_index[peer] - 1
            pass
    
    def start_election(self):
        """
        Start leader election
        Called when leader heartbeat times out
        """
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        
        votes_received = 1  # Vote for self
        
        # Request votes from peers
        for peer in self.peers:
            # Send RequestVote RPC
            # response = send_request_vote(peer)
            # if response.vote_granted:
            #     votes_received += 1
            pass
        
        # If majority votes, become leader
        if votes_received > len(self.peers) / 2:
            self.state = NodeState.LEADER
            print(f"Node {self.node_id} elected leader for term {self.current_term}")
    
    def apply_committed_entries(self):
        """
        Apply committed log entries to state machine
        """
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            
            # Apply to state machine (e.g., execute order)
            self.apply_command(entry.command)
    
    def apply_command(self, command: dict):
        """Apply command to trading system"""
        if command['type'] == 'order':
            print(f"Executing order: {command}")
            # Execute order...
        elif command['type'] == 'cancel':
            print(f"Canceling order: {command}")
            # Cancel order...

# Raft guarantees:
# - All nodes see same sequence of orders
# - Once committed, entry cannot be lost
# - Leader election takes ~300ms (one election timeout)
\`\`\`

---

## Handling Network Partitions

### Split-Brain Scenario

\`\`\`
Normal:
US ←→ Europe ←→ Asia
(All connected)

Partition:
US ←→ Europe   |   Asia (isolated)
(Network partition)

Problem:
- US and Asia both think they're leader
- Both accept orders
- Conflicting state
\`\`\`

### Solution: Quorum

\`\`\`python
"""
Quorum-based decision making
Prevents split-brain
"""

class QuorumSystem:
    """
    Require majority agreement
    Prevents split-brain during partition
    """
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.quorum_size = len(nodes) // 2 + 1
    
    def can_make_progress(self, reachable_nodes: List[str]) -> bool:
        """
        Check if we have quorum
        """
        return len(reachable_nodes) >= self.quorum_size
    
    def submit_order(self, order: dict, reachable_nodes: List[str]) -> bool:
        """
        Submit order only if quorum available
        """
        if not self.can_make_progress(reachable_nodes):
            print("No quorum available, rejecting order")
            return False
        
        # Replicate to quorum
        acks = 0
        for node in reachable_nodes:
            # Send order to node
            # if send_order(node, order):
            #     acks += 1
            acks += 1  # Simulate
            
            if acks >= self.quorum_size:
                print(f"Order committed with {acks} acks")
                return True
        
        return False

# Example: 5 nodes
# Quorum = 3
# If partition: 3 nodes vs 2 nodes
# - Majority partition (3) can make progress
# - Minority partition (2) rejects orders (safe)
\`\`\`

---

## Global Order Routing

\`\`\`python
"""
Smart order routing across regions
Route orders to best execution venue
"""

class GlobalOrderRouter:
    """
    Route orders to optimal exchange
    Consider latency, fees, liquidity
    """
    
    def __init__(self):
        self.exchanges = {
            'NYSE': {'region': 'US', 'latency_ms': 1, 'fee': 0.0003},
            'NASDAQ': {'region': 'US', 'latency_ms': 1, 'fee': 0.0003},
            'LSE': {'region': 'EU', 'latency_ms': 80, 'fee': 0.0005},
            'TSE': {'region': 'ASIA', 'latency_ms': 150, 'fee': 0.0004},
        }
        
        self.current_region = 'US'  # Assume we're in US
    
    def route_order(self, symbol: str, quantity: int) -> str:
        """
        Determine best exchange for order
        """
        # Get exchanges that trade this symbol
        candidates = self.get_exchanges_for_symbol(symbol)
        
        # Score each exchange
        scores = {}
        for exchange in candidates:
            info = self.exchanges[exchange]
            
            # Factors:
            # 1. Latency (lower better)
            # 2. Fees (lower better)
            # 3. Liquidity (higher better)
            
            latency_score = 100 / (1 + info['latency_ms'])  # Closer = higher score
            fee_score = 100 * (1 - info['fee'])
            
            # Weighted combination
            scores[exchange] = 0.6 * latency_score + 0.4 * fee_score
        
        # Select best
        best_exchange = max(scores, key=scores.get)
        
        print(f"Routing {symbol} order to {best_exchange} (score: {scores[best_exchange]:.2f})")
        
        return best_exchange
    
    def get_exchanges_for_symbol(self, symbol: str) -> List[str]:
        """Get exchanges that trade symbol"""
        # In production: Query from database
        if symbol in ['AAPL', 'TSLA']:
            return ['NYSE', 'NASDAQ']
        elif symbol in ['VOD.L', 'BP.L']:
            return ['LSE']
        else:
            return ['NYSE']

# Route intelligently:
# - US stocks → US exchanges (1ms)
# - UK stocks → LSE (80ms from US, but only option)
# - Cross-listed → best execution venue
\`\`\`

---

## Data Replication Strategies

### Synchronous Replication

\`\`\`python
"""
Synchronous replication
Wait for all replicas before committing
"""

class SynchronousReplicator:
    """
    Replicate data synchronously
    Guarantees consistency
    """
    
    def write(self, key: str, value: any, replicas: List[str]) -> bool:
        """
        Write to all replicas before returning
        """
        # Write to local
        self.local_store[key] = value
        
        # Write to all replicas
        acks = 0
        for replica in replicas:
            # Send write request
            success = self.send_write(replica, key, value)
            if success:
                acks += 1
        
        # Wait for all acks
        if acks == len(replicas):
            return True  # All replicas updated
        else:
            # Rollback
            del self.local_store[key]
            return False

# Pros: Strong consistency
# Cons: High latency (wait for slowest replica)
# Use for: Critical data (positions, cash balances)
\`\`\`

### Asynchronous Replication

\`\`\`python
"""
Asynchronous replication
Return immediately, replicate in background
"""

class AsynchronousReplicator:
    """
    Replicate data asynchronously
    Low latency but eventual consistency
    """
    
    def __init__(self):
        self.replication_queue = Queue()
        self.replication_thread = Thread(target=self.replicate_worker)
        self.replication_thread.start()
    
    def write(self, key: str, value: any, replicas: List[str]) -> bool:
        """
        Write locally and queue replication
        """
        # Write to local
        self.local_store[key] = value
        
        # Queue replication (async)
        self.replication_queue.put((key, value, replicas))
        
        # Return immediately
        return True
    
    def replicate_worker(self):
        """Background worker for replication"""
        while True:
            key, value, replicas = self.replication_queue.get()
            
            for replica in replicas:
                self.send_write(replica, key, value)

# Pros: Low latency (no waiting)
# Cons: Eventual consistency (replicas lag)
# Use for: Non-critical data (market data, analytics)
\`\`\`

---

## Multi-Region Deployment Best Practices

### 1. Region Affinity

Route requests to nearest region:

\`\`\`nginx
# NGINX configuration
upstream us_region {
    server us-trade-1:8000;
    server us-trade-2:8000;
}

upstream eu_region {
    server eu-trade-1:8000;
    server eu-trade-2:8000;
}

geo $region {
    default us;
    0.0.0.0/0 us;
    85.0.0.0/8 eu;      # Europe IP ranges
    141.0.0.0/8 asia;
}

server {
    location /api/order {
        if ($region = eu) {
            proxy_pass http://eu_region;
        }
        proxy_pass http://us_region;
    }
}
\`\`\`

### 2. Circuit Breaker

Detect and isolate failing regions:

\`\`\`python
class RegionCircuitBreaker:
    """Isolate failing regions"""
    
    def __init__(self, failure_threshold: int = 5):
        self.failure_threshold = failure_threshold
        self.failures = {}  # region -> failure count
        self.states = {}    # region -> CLOSED/OPEN/HALF_OPEN
    
    def call_region(self, region: str, func):
        """Call region with circuit breaker"""
        if self.states.get(region) == 'OPEN':
            raise Exception(f"Region {region} circuit open")
        
        try:
            result = func()
            self.failures[region] = 0  # Reset on success
            return result
        except Exception as e:
            self.failures[region] = self.failures.get(region, 0) + 1
            
            if self.failures[region] >= self.failure_threshold:
                self.states[region] = 'OPEN'
                print(f"Circuit opened for region {region}")
            
            raise
\`\`\`

### 3. Health Checks

Monitor region health:

\`\`\`python
class RegionHealthCheck:
    """Monitor region health"""
    
    def check_health(self, region: str) -> dict:
        """Check if region is healthy"""
        checks = {
            'api_responsive': self.check_api(region),
            'database_connected': self.check_database(region),
            'latency_acceptable': self.check_latency(region) < 100,  # <100ms
            'error_rate_low': self.check_error_rate(region) < 0.01,  # <1%
        }
        
        healthy = all(checks.values())
        
        return {
            'region': region,
            'healthy': healthy,
            'checks': checks
        }
\`\`\`

---

## Summary

Distributed trading systems require careful design:

1. **Multi-region**: Active-active for low latency, active-passive for simplicity
2. **Clock sync**: NTP (1-50ms) for most trading, PTP (<1μs) for HFT
3. **Consensus**: Raft/Paxos for agreement, quorum prevents split-brain
4. **Partitions**: Handle gracefully with quorum, fail-safe not fail-soft
5. **Routing**: Smart order routing considers latency, fees, liquidity
6. **Replication**: Synchronous for critical data, asynchronous for non-critical

**CAP Theorem**: Choose AP (availability + partition tolerance) for trading—better to keep trading with stale data than stop.

In the next section, we'll design regulatory and compliance systems for audit trails and surveillance.
`,
};
