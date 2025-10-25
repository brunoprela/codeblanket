/**
 * Consistent Hashing Section
 */

export const consistenthashingSection = {
  id: 'consistent-hashing',
  title: 'Consistent Hashing',
  content: `Consistent hashing is a distributed hashing technique that elegantly solves one of the most fundamental problems in distributed systems: **"How do we distribute data across servers in a way that minimizes reorganization when servers are added or removed?"**

## The Problem with Traditional Hashing

Imagine you have a cache cluster with 3 servers. You need to decide which server stores each key.

### Naive Approach: Modulo Hashing

\`\`\`python
server_index = hash (key) % num_servers

Examples with 3 servers:
hash("user:1") = 8473 → 8473 % 3 = 2 → Server 2
hash("user:2") = 3241 → 3241 % 3 = 1 → Server 1  
hash("user:3") = 9847 → 9847 % 3 = 1 → Server 1
\`\`\`

This works fine until you add or remove a server.

### The Disaster: Adding Server 4

Now with 4 servers, the formula changes:
\`\`\`
server_index = hash (key) % 4

hash("user:1") = 8473 → 8473 % 4 = 1 → Server 1 (was Server 2!)
hash("user:2") = 3241 → 3241 % 4 = 1 → Server 1 (was Server 1) ✓
hash("user:3") = 9847 → 9847 % 4 = 3 → Server 3 (was Server 1!)
\`\`\`

**Catastrophe**: When you change from 3 to 4 servers, **most keys map to different servers**!

**Impact**:
- 75% cache miss rate (all keys in wrong place)
- Need to rehash and redistribute ~75% of all data
- Thundering herd: All requests miss cache, hit database
- Database overload → potential outage

**Formula**: When servers change from n to n+1, approximately (n/(n+1)) keys need to be remapped.
- 3→4 servers: 75% remapped
- 10→11 servers: 91% remapped
- 100→101 servers: 99% remapped

This is unacceptable at scale.

---

## Consistent Hashing: The Solution

**Consistent hashing** distributes keys across servers such that when a server is added or removed, only **K/n keys need to be remapped** (where K = total keys, n = servers).

**Key properties**:
- ✅ **Minimal disruption**: Only ~1/n of keys affected when adding/removing servers
- ✅ **Decentralized**: No central coordinator needed
- ✅ **Scalable**: Add servers without rehashing everything
- ✅ **Fault tolerant**: Server failure affects only its keys

Used by: **Cassandra, DynamoDB, Memcached, Redis Cluster, Akamai CDN, Discord**

---

## How Consistent Hashing Works

### The Hash Ring

Visualize hash space as a **ring** (circle) from 0 to 2³²-1 (or 2⁶⁴-1).

\`\`\`
           0/2³²
            |
    2³¹ ----●---- 2³¹
            |
          2³⁰
\`\`\`

### Step 1: Place Servers on Ring

Hash each server's identifier (IP address, hostname) to position on ring:

\`\`\`python
hash("Server A") = 84735827  → Position on ring
hash("Server B") = 23847392  → Position on ring
hash("Server C") = 91746382  → Position on ring
\`\`\`

\`\`\`
           0
           |
    Server C (91746382)
           |
    Server A (84735827)
          / \\
         /   \\
Server B     Data points
(23847392)
\`\`\`

### Step 2: Place Keys on Ring

Hash each key to position on ring:

\`\`\`python
hash("user:1") = 45827392
hash("user:2") = 67382943
hash("user:3") = 12847392
\`\`\`

### Step 3: Mapping Keys to Servers

**Rule**: Each key is stored on the **first server encountered clockwise** from its position.

\`\`\`
Example:
hash("user:1") = 45827392
Walk clockwise → First server = Server A (84735827)
→ user:1 stored on Server A

hash("user:2") = 67382943  
Walk clockwise → First server = Server A (84735827)
→ user:2 stored on Server A

hash("user:3") = 12847392
Walk clockwise → First server = Server B (23847392)  
→ user:3 stored on Server B
\`\`\`

\`\`\`
Ring visualization:

     0/2³²-1
        |
    [Server C] - 91M
        |
     user:4
        |
    [Server A] - 84M
        |
     user:1, user:2
        |
    [Server B] - 23M
        |
     user:3
        |
    back to 0
\`\`\`

---

## Adding a Server: Minimal Disruption

Let\'s add **Server D** to the ring:

\`\`\`python
hash("Server D") = 55000000
\`\`\`

**Before** (3 servers):
- user:1 (45M) → Server A (84M)
- user:2 (67M) → Server A (84M)
- user:3 (12M) → Server B (23M)

**After** adding Server D at position 55M:
- user:1 (45M) → Server D (55M) - **MOVED**
- user:2 (67M) → Server A (84M) - unchanged
- user:3 (12M) → Server B (23M) - unchanged

**Only user:1 moved!** Just 1/4 of keys affected (expected with 4 servers).

**Compare to modulo hashing**: 75% of keys would have moved.

---

## Removing a Server: Graceful Degradation

If **Server B fails** (position 23M):

**Before**:
- user:3 (12M) → Server B (23M)

**After** Server B removal:
- user:3 (12M) → Server C (91M) [next server clockwise]

**Only keys on Server B need reassignment** → Minimal impact.

Data redistributes to immediate neighbors only.

---

## The Load Balancing Problem

Simple consistent hashing has a **load balancing issue**:

\`\`\`
Ring with 3 servers:

Server A - 10M
Server B - 40M  
Server C - 80M

Server A range: 80M → 10M (30M keys)
Server B range: 10M → 40M (30M keys)
Server C range: 40M → 80M (40M keys)
\`\`\`

Server C gets 33% more load than A/B! With random hash distribution, servers can get uneven load.

**Worse case**: If servers cluster together on ring, massive imbalance.

---

## Virtual Nodes (VNodes): The Load Balancing Solution

**Key idea**: Instead of placing each server once on the ring, place it **multiple times** with different hash values.

\`\`\`python
# Place Server A at 100 positions
for i in range(100):
    position = hash (f"Server A-{i}")
    ring.add (position, "Server A")
\`\`\`

\`\`\`
Ring with virtual nodes:

A₁, A₂, A₃, B₁, C₁, A₄, B₂, C₂, A₅, B₃, C₃, ...

Each physical server has many virtual nodes distributed around ring
\`\`\`

### Benefits of Virtual Nodes

**1. Even load distribution**
- With 100-200 vnodes per server, load variance < 5%
- Statistical averaging smooths out random hash clustering

**2. Smoother data migration**
- When adding server, data comes from ALL servers (not just one)
- When removing server, data goes to ALL servers

**3. Heterogeneous hardware**
- Powerful server: 200 vnodes (gets 2x traffic)
- Weak server: 100 vnodes (gets 1x traffic)

**4. Faster rebalancing**
- Small vnode migrations instead of large server migrations

### Virtual Node Example

**Before** adding Server D:
\`\`\`
A₁ A₂ B₁ C₁ A₃ B₂ C₂ A₄ B₃ C₃ ...
\`\`\`

**After** adding Server D (with 100 vnodes):
\`\`\`
A₁ D₁ A₂ B₁ D₂ C₁ A₃ D₃ B₂ C₂ D₄ A₄ B₃ D₅ C₃ ...
\`\`\`

Keys near D₁ move from A₁ to D₁
Keys near D₂ move from B₁ to D₂
...

Data migrates evenly from all servers.

---

## Implementation

### Basic Consistent Hash Ring

\`\`\`python
import hashlib
from bisect import bisect_right

class ConsistentHashRing:
    def __init__(self, num_virtual_nodes=100):
        self.num_virtual_nodes = num_virtual_nodes
        self.ring = []  # Sorted list of (hash_value, server) tuples
        self.hash_to_server = {}
    
    def _hash (self, key):
        """Hash function producing integer"""
        return int (hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_server (self, server):
        """Add server with virtual nodes"""
        for i in range (self.num_virtual_nodes):
            vnode_key = f"{server}:{i}"
            hash_value = self._hash (vnode_key)
            self.ring.append((hash_value, server))
            self.hash_to_server[hash_value] = server
        
        # Keep ring sorted
        self.ring.sort()
    
    def remove_server (self, server):
        """Remove server and its virtual nodes"""
        self.ring = [(h, s) for h, s in self.ring if s != server]
    
    def get_server (self, key):
        """Get server for given key"""
        if not self.ring:
            return None
        
        hash_value = self._hash (key)
        
        # Binary search for first server >= hash_value
        index = bisect_right (self.ring, (hash_value, ''))
        
        # Wrap around to beginning if needed
        if index == len (self.ring):
            index = 0
        
        return self.ring[index][1]

# Usage
ring = ConsistentHashRing (num_virtual_nodes=150)

# Add servers
ring.add_server("Server A")
ring.add_server("Server B")
ring.add_server("Server C")

# Route keys
print(ring.get_server("user:1"))  # Server B
print(ring.get_server("user:2"))  # Server A
print(ring.get_server("user:3"))  # Server C

# Add new server - minimal remapping!
ring.add_server("Server D")

# Some keys now route to Server D
print(ring.get_server("user:1"))  # Maybe Server D now
\`\`\`

### Optimized: Binary Search for Performance

\`\`\`python
def get_server_fast (self, key):
    """O(log n) lookup using binary search"""
    if not self.ring:
        return None
    
    hash_value = self._hash (key)
    
    # Binary search in sorted ring
    left, right = 0, len (self.ring)
    
    while left < right:
        mid = (left + right) // 2
        if self.ring[mid][0] < hash_value:
            left = mid + 1
        else:
            right = mid
    
    # Wrap around
    if left >= len (self.ring):
        left = 0
    
    return self.ring[left][1]
\`\`\`

**Time complexity**:
- Add server: O(V log(NV)) where V = vnodes, N = servers
- Remove server: O(NV)
- Get server: O(log(NV))

---

## Real-World Implementations

### 1. Amazon DynamoDB

**Partitioning**:
- Each node has 100-200 virtual nodes
- Data distributed across ring based on partition key
- Automatic rebalancing when nodes added/removed

**Replication**:
- Replicate data to N successive nodes clockwise
- If N=3, key stored on 3 different servers for fault tolerance

\`\`\`
Key "user:123" at position 45M:
- Primary: Server A (55M)
- Replica 1: Server B (70M)  
- Replica 2: Server C (85M)
\`\`\`

### 2. Apache Cassandra

**Token ring**:
- Each node assigned token range
- Virtual nodes (vnodes) enabled by default (256 per node)
- Consistent hashing for partitioning

**Partition key** → Murmur3 hash → Position on ring → Node assignment

\`\`\`sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY,  -- Partition key
    name TEXT
);

-- user_id hashed to determine which node stores row
\`\`\`

**Benefits**:
- No downtime when adding nodes
- Automatic load balancing
- Gossip protocol for cluster state

### 3. Memcached (libmemcached client)

**Client-side consistent hashing**:
- Client maintains hash ring
- No coordination between Memcached servers
- Servers are stateless

\`\`\`python
# Client code
servers = ["10.0.0.1:11211", "10.0.0.2:11211", "10.0.0.3:11211"]
ring = ConsistentHashRing()

for server in servers:
    ring.add_server (server)

# Route request
key = "user:session:1234"
server = ring.get_server (key)
# Connect to that specific server
\`\`\`

### 4. Discord (Chat Platform)

**Guild (server) partitioning**:
- 4000+ nodes serving millions of guilds
- Consistent hashing to assign guilds to nodes
- Hot guild detection and rebalancing

**Challenge**: Popular guilds (millions of users) can overload single node.

**Solution**: Monitor load, manually reassign hot guilds.

### 5. Akamai CDN

**Edge server selection**:
- Consistent hashing for content distribution
- Content stays on same edge servers (better cache hit rate)
- Adding/removing servers minimally disrupts cache

---

## Consistent Hashing vs Rendezvous Hashing

**Consistent hashing**:
- ✅ Efficient O(log n) lookups with sorted ring
- ✅ Widely adopted
- ❌ Virtual nodes add complexity
- ❌ Load balancing requires tuning vnodes

**Rendezvous hashing** (Highest Random Weight):
- Alternative approach: Compute score for key on each server, pick highest
- ✅ Perfect load distribution
- ✅ No virtual nodes needed
- ❌ O(n) lookup (need to check all servers)
- Use case: Small number of servers

---

## Advanced Topics

### Replication Factor

Store data on **R consecutive nodes** for fault tolerance:

\`\`\`python
def get_servers (self, key, replication_factor=3):
    """Get R servers for replication"""
    if not self.ring:
        return []
    
    hash_value = self._hash (key)
    index = bisect_right (self.ring, (hash_value, ''))
    
    servers = []
    seen = set()
    
    # Get next R unique servers clockwise
    while len (servers) < replication_factor:
        server = self.ring[index % len (self.ring)][1]
        if server not in seen:
            servers.append (server)
            seen.add (server)
        index += 1
    
    return servers

# Example: Store on 3 servers for redundancy
servers = ring.get_servers("user:1", replication_factor=3)
# ["Server A", "Server B", "Server C"]
\`\`\`

### Weighted Consistent Hashing

**Problem**: Heterogeneous hardware (different capacity servers)

**Solution**: Assign more vnodes to powerful servers:

\`\`\`python
# Powerful server: 200 vnodes (2x traffic)
ring.add_server("Server A", num_vnodes=200)

# Regular server: 100 vnodes (1x traffic)  
ring.add_server("Server B", num_vnodes=100)

# Weak server: 50 vnodes (0.5x traffic)
ring.add_server("Server C", num_vnodes=50)
\`\`\`

### Bounded Loads

**Problem**: One server becomes hot (popular keys cluster)

**Google\'s solution**: Bounded-load consistent hashing
- Set max load threshold (e.g., 1.25 × average load)
- If server exceeds threshold, route to next server
- Prevents any server from being more than 25% overloaded

---

## Interview Tips

### Key Talking Points

1. **Problem statement**: Modulo hashing requires rehashing ~100% of data when servers change
2. **Consistent hashing solution**: Only ~1/n keys need remapping
3. **Virtual nodes**: Solve load balancing problem (100-200 vnodes per server)
4. **Real-world use**: DynamoDB, Cassandra, Redis Cluster, CDNs
5. **Trade-offs**: More complex than modulo, but necessary at scale

### Common Interview Questions

**"Why not just use modulo hashing?"**
- Works for fixed server count
- Adding/removing servers requires rehashing ~100% of keys
- Unacceptable cache miss rate and data migration cost

**"How many virtual nodes should you use?"**
- Typical: 100-200 vnodes per physical server
- More vnodes → Better load distribution but more memory/CPU
- Diminishing returns beyond 150-200
- Cassandra default: 256 vnodes

**"What happens when a server fails?"**
- Only keys on that server affected (~1/n of total)
- Keys automatically route to next server clockwise
- With replication (3 replicas), read from replica immediately
- No downtime or cascading failures

**"How does consistent hashing compare to sharding?"**
- Consistent hashing: Dynamic, automatic rebalancing
- Sharding: Static ranges, manual rebalancing
- Consistent hashing better for elastic systems (cloud)
- Sharding simpler for fixed infrastructure

### Whiteboard Exercise

Design a distributed cache with 5 servers that can scale to 10 servers:

\`\`\`
1. Use consistent hashing with virtual nodes
   - 150 vnodes per server
   - MD5 or Murmur hash function

2. When adding Server 6:
   - Add its 150 vnodes to ring
   - ~16% of keys (1/6) remap from existing servers
   - Old servers gradually migrate data to new server
   - No downtime, no thundering herd

3. Replication:
   - Store each key on 3 consecutive servers
   - If one fails, read from replicas
   - Automatic failover

4. Client routing:
   - Client maintains hash ring
   - Direct routing (no central coordinator)
   - Periodic ring updates via gossip protocol
\`\`\`

---

## Common Pitfalls

### ❌ Not Using Virtual Nodes

Without vnodes, load distribution can be very uneven (30-40% variance).

**Solution**: Always use 100-200 virtual nodes per physical server.

### ❌ Poor Hash Function

Using poor hash function leads to clustering on ring.

**Solution**: Use cryptographic hash (MD5, SHA-1) or fast hash with good distribution (Murmur3, xxHash).

### ❌ Ignoring Network Topology

Placing replicas on consecutive ring nodes might put them in same rack/datacenter.

**Solution**: Rack-aware placement (Cassandra's rack awareness).

### ❌ Forgetting to Remove Vnodes

When removing server, must remove ALL its virtual nodes.

**Solution**: Track vnode-to-server mapping carefully.

---

## Summary

**Consistent hashing** is the fundamental technique for distributing data across servers in a scalable, fault-tolerant way.

**Key principles**:
- ✅ Hash ring: Servers and keys placed on circle
- ✅ Clockwise rule: Key stored on first server clockwise  
- ✅ Minimal remapping: Only ~1/n keys affected by server changes
- ✅ Virtual nodes: Solve load balancing (100-200 per server)
- ✅ Replication: Store on R consecutive servers

**When to use**:
- Distributed caching (Memcached, Redis)
- Distributed databases (Cassandra, DynamoDB)
- CDN edge selection
- Load balancing with dynamic server pools

**Industry adoption**: This is not academic theory—it's the proven standard for partitioning at scale used by every major distributed system.

Understanding consistent hashing is **essential** for system design interviews and building scalable distributed systems.`,
};
