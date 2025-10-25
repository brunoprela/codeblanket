/**
 * Distributed Object Storage Section
 */

export const distributedObjectStorageSection = {
  id: 'distributed-object-storage',
  title: 'Distributed Object Storage',
  content: `Distributed object storage systems like Ceph and MinIO enable organizations to build their own S3-compatible storage infrastructure, providing control, cost savings, and flexibility.

## Overview

**Distributed object storage** = Self-hosted alternative to cloud object storage

**Key systems**:
- **Ceph**: Enterprise-grade, unified storage (object, block, file)
- **MinIO**: Lightweight, S3-compatible, cloud-native
- **OpenStack Swift**: OpenStack\'s object storage
- **SeaweedFS**: Simple, fast object storage

**Why self-hosted?**
- ✅ Data sovereignty (keep data on-premise)
- ✅ Cost savings (vs cloud at scale)
- ✅ Customization and control
- ✅ Integration with existing infrastructure
- ❌ Operational complexity
- ❌ Need to manage hardware, networking, failures

---

## Ceph Architecture

### Overview

**Ceph** = Unified storage platform providing:
- **RADOS**: Reliable Autonomic Distributed Object Store (foundation)
- **RGW**: RADOS Gateway (S3/Swift API)
- **RBD**: RADOS Block Device (block storage)
- **CephFS**: Ceph File System (POSIX FS)

\`\`\`
Applications
    │
    ├─── S3 API ───────→ RGW (RADOS Gateway)
    ├─── Block Device ─→ RBD
    └─── File System ──→ CephFS
            │
            ↓
        RADOS (core)
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  OSD 1   OSD 2   OSD 3  (Object Storage Daemons)
\`\`\`

### Core Components

**1. MON (Monitor)**
- Maintains cluster map
- Tracks OSD status
- Provides consensus (Paxos-based)
- Minimum 3 for HA (quorum)

**2. OSD (Object Storage Daemon)**
- One per disk
- Stores data as objects
- Handles replication
- Performs recovery
- Typical cluster: 100s to 1000s of OSDs

**3. MDS (Metadata Server)**
- For CephFS only
- Manages file system metadata
- Not needed for object storage

**4. RGW (RADOS Gateway)**
- S3/Swift compatible API
- Stateless (scale horizontally)
- Handles multipart uploads
- Object versioning, lifecycle

**5. MGR (Manager)**
- Monitoring and management
- Dashboard UI
- Metrics collection

### CRUSH Algorithm

**CRUSH** (Controlled Replication Under Scalable Hashing):
- Deterministic placement algorithm
- **No central metadata** for object location!
- Any node can calculate where object should be

**How CRUSH works**:

\`\`\`
Object: "my-file.jpg"
    ↓
Hash (my-file.jpg) → 1234567890
    ↓
CRUSH algorithm + cluster map
    ↓
Placement groups: PG 4.2a
    ↓
OSDs: [OSD.5, OSD.12, OSD.23] (3 replicas)
\`\`\`

**Benefits**:
- ✅ No metadata bottleneck
- ✅ Clients directly compute location
- ✅ Automatic rebalancing on cluster changes
- ✅ Customizable placement rules

**CRUSH map** defines hierarchy:
\`\`\`
datacenter
  ├─ rack1
  │   ├─ host1 (OSD 1, 2, 3)
  │   └─ host2 (OSD 4, 5, 6)
  └─ rack2
      ├─ host3 (OSD 7, 8, 9)
      └─ host4 (OSD 10, 11, 12)
\`\`\`

**Placement rules**: "Put 3 replicas in different racks"

### Ceph Read Flow

\`\`\`
Client                Monitor              OSDs
  |                      |                    |
  |-- Get cluster map -->|                    |
  |<-- Cluster map ------|                    |
  |                      |                    |
  | Calculate object location (CRUSH)         |
  |                      |                    |
  |-- Read object -------|-----------------> OSD.5 (primary)
  |                      |                    |
  |<-- Object data ------|-------------------|
\`\`\`

**Steps**:
1. Client gets cluster map from monitor (cached)
2. Client calculates object location using CRUSH
3. Client reads directly from primary OSD
4. No monitor involvement in data path!

### Ceph Write Flow

\`\`\`
Client                OSDs
  |                    |
  |-- Write object --->| OSD.5 (primary)
  |                    |     |
  |                    |     ├─→ OSD.12 (replica 2)
  |                    |     └─→ OSD.23 (replica 3)
  |                    |     |
  |                    |     | All replicas written
  |                    |<────┘
  |<-- ACK ------------|
\`\`\`

**Replication**: Primary OSD coordinates replication to secondary OSDs

### Erasure Coding

**Alternative to replication** for storage efficiency:

**Replication (3x)**:
\`\`\`
Original: 1 GB
Stored: 3 GB (3x overhead)
Can lose: 2 replicas
\`\`\`

**Erasure Coding (e.g., 8+4)**:
\`\`\`
Original: 1 GB
Split into: 8 data chunks + 4 parity chunks
Stored: 1.5 GB (1.5x overhead)
Can lose: any 4 chunks
\`\`\`

**Trade-offs**:
- ✅ Lower storage overhead (1.5x vs 3x)
- ❌ Higher CPU (encoding/decoding)
- ❌ Slower reads (need to reconstruct)
- ❌ More network traffic (read multiple chunks)

**When to use**:
- Replication: Hot data, performance-critical
- Erasure coding: Cold data, cost-sensitive

### Ceph Data Durability

**Strategies**:
1. **Replication**: 2x, 3x, or more
2. **Erasure coding**: K+M (K data, M parity)
3. **Hybrid**: Replication for metadata, EC for data

**Scrubbing**:
- Light scrub: Check metadata daily
- Deep scrub: Check data integrity weekly
- Automatically repairs corruption

### Ceph Use Cases

**Object storage** (RGW):
- S3-compatible backup storage
- On-premise data lake
- Media storage

**Block storage** (RBD):
- Virtual machine disks (OpenStack, Kubernetes)
- Database storage
- High-performance applications

**File storage** (CephFS):
- Shared file system
- HPC workloads

---

## MinIO Architecture

### Overview

**MinIO** = Lightweight, S3-compatible object storage

**Characteristics**:
- Written in Go
- Kubernetes-native
- S3 API compatible
- Simpler than Ceph
- Faster deployment

**Use cases**:
- Cloud-native applications
- Kubernetes persistent storage
- AI/ML data lakes
- Hybrid cloud

### Architecture

\`\`\`
       Load Balancer
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  Node1   Node2   Node3   Node4  (MinIO servers)
    │       │       │       │
  Disk1   Disk2   Disk3   Disk4  (Drives)
  Disk5   Disk6   Disk7   Disk8
\`\`\`

**Deployment modes**:

**1. Standalone** (single node):
\`\`\`bash
minio server /data
\`\`\`
Simple, no HA

**2. Distributed** (multiple nodes):
\`\`\`bash
minio server http://node{1...4}/data{1...4}
\`\`\`
HA, erasure coded

### Erasure Coding in MinIO

**Default**: N/2 data shards, N/2 parity shards

**Example (4 nodes, 8 drives)**:
\`\`\`
Erasure set: 4 data shards + 4 parity shards
Can tolerate: 4 drive/node failures
\`\`\`

**Write process**:
1. Object split into data shards
2. Parity shards calculated
3. Shards distributed across drives
4. Write successful if majority written

**Read process**:
1. Read data shards (if available)
2. Reconstruct from parity if data shards missing
3. Return object to client

### MinIO Features

**1. S3 Compatibility**:
- 100% AWS S3 API compatible
- Drop-in replacement for S3
- Use AWS SDKs with MinIO

**2. Kubernetes Integration**:
\`\`\`yaml
apiVersion: v1
kind: Pod
metadata:
  name: minio
spec:
  containers:
  - name: minio
    image: minio/minio
    args:
    - server
    - /data
    ports:
    - containerPort: 9000
\`\`\`

**3. Encryption**:
- Server-side encryption (SSE-S3, SSE-C, SSE-KMS)
- At-rest encryption
- In-transit encryption (TLS)

**4. Versioning**:
- Object versioning
- Same as S3 versioning

**5. Lifecycle Policies**:
- Expiration rules
- Transition rules (to other storage classes)

**6. Replication**:
- Cross-site replication
- Active-active or active-passive

### MinIO vs Ceph

| Feature       | MinIO                   | Ceph                    |
|---------------|-------------------------|-------------------------|
| Simplicity    | ✅ Very simple          | ❌ Complex              |
| S3 API        | ✅ Native               | Via RGW                 |
| Block storage | ❌ No                   | ✅ Yes (RBD)            |
| File storage  | ❌ No                   | ✅ Yes (CephFS)         |
| Maturity      | Newer (2015)            | Mature (2006)           |
| Scale         | PB (smaller clusters)   | EB (larger clusters)    |
| Kubernetes    | ✅ Native               | Can integrate           |
| Ops overhead  | Low                     | High                    |

**Choose MinIO if**:
- Need simple S3-compatible storage
- Kubernetes-native applications
- Smaller scale (< 10 PB)
- Want fast deployment

**Choose Ceph if**:
- Need block or file storage too
- Very large scale (> 10 PB)
- Need battle-tested system
- Have ops team to manage complexity

---

## OpenStack Swift

### Overview

**Swift** = OpenStack's object storage

**Characteristics**:
- Python-based
- Eventual consistency
- REST API (not S3-compatible by default)
- Widely used in OpenStack deployments

### Architecture

\`\`\`
              Proxy Servers (stateless)
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
  Storage Node 1  Storage Node 2  Storage Node 3
      │              │              │
  Container DB    Container DB    Container DB
  Account DB      Account DB      Account DB
  Objects         Objects         Objects
\`\`\`

**Components**:

**1. Proxy Server**:
- API endpoint
- Routes requests
- Stateless (scale horizontally)

**2. Storage Nodes**:
- Store objects
- Run account, container, object servers
- Use consistent hashing for placement

**3. Rings**:
- Account ring
- Container ring  
- Object ring
- Define data placement

### Swift vs S3

| Feature       | Swift                   | S3/MinIO/Ceph           |
|---------------|-------------------------|-------------------------|
| API           | Swift API               | S3 API                  |
| Consistency   | Eventual                | Strong (S3 2020+)       |
| Hierarchy     | Account/Container/Object| Bucket/Object           |
| Adoption      | OpenStack ecosystems    | Wider adoption          |

---

## SeaweedFS

### Overview

**SeaweedFS** = Simple, fast distributed file/object storage

**Characteristics**:
- Written in Go
- Optimized for small files
- S3-compatible
- Simple architecture

### Architecture

\`\`\`
       Master Server (metadata)
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  Volume  Volume  Volume  (storage servers)
  Server1 Server2 Server3
\`\`\`

**Key concepts**:

**1. Volumes**:
- Fixed-size containers (default 30 GB)
- Store many small files
- Replicated across servers

**2. Master server**:
- Assigns file IDs
- Manages volume locations
- Lightweight metadata

**3. File storage**:
- Each file gets unique ID
- ID encodes volume and file location
- Direct access to file via ID

**Advantages**:
- ✅ Very efficient for small files
- ✅ Simple architecture
- ✅ Fast reads/writes
- ✅ Low memory usage

**Use cases**:
- Photo storage (millions of small images)
- Log aggregation
- IoT data

---

## Building Your Own Distributed Object Storage

### Design Considerations

**1. Replication Strategy**:
- How many replicas?
- Cross-datacenter?
- Erasure coding?

**2. Consistency Model**:
- Strong consistency (slower writes)
- Eventual consistency (faster, complex)

**3. Metadata Management**:
- Centralized (simpler, bottleneck)
- Distributed (complex, scalable)

**4. Failure Handling**:
- Detection (heartbeats)
- Recovery (re-replication)
- Consistency during failures

**5. Data Placement**:
- Consistent hashing
- CRUSH-like algorithm
- Manual placement rules

### Simplified Implementation

**Components needed**:

**1. API Server** (stateless):
\`\`\`python
@app.post("/bucket/{key}")
def put_object (bucket, key, data):
    # Calculate replicas using consistent hashing
    nodes = hash_ring.get_nodes (key, count=3)
    
    # Write to all replicas
    for node in nodes:
        node.write (bucket, key, data)
    
    return {"status": "success"}

@app.get("/bucket/{key}")
def get_object (bucket, key):
    # Get node list
    nodes = hash_ring.get_nodes (key, count=3)
    
    # Read from first available
    for node in nodes:
        if node.has (bucket, key):
            return node.read (bucket, key)
    
    return {"error": "not found"}
\`\`\`

**2. Storage Node**:
\`\`\`python
class StorageNode:
    def write (self, bucket, key, data):
        # Write to disk
        path = f"/storage/{bucket}/{key}"
        with open (path, 'wb') as f:
            f.write (data)
        
        # Update metadata
        self.metadata[f"{bucket}/{key}"] = {
            "size": len (data),
            "timestamp": time.time()
        }
    
    def read (self, bucket, key):
        path = f"/storage/{bucket}/{key}"
        with open (path, 'rb') as f:
            return f.read()
\`\`\`

**3. Consistent Hashing**:
\`\`\`python
class HashRing:
    def __init__(self, nodes):
        self.ring = {}
        self.nodes = nodes
        for node in nodes:
            for i in range(100):  # Virtual nodes
                hash_val = hash (f"{node}:{i}")
                self.ring[hash_val] = node
        self.sorted_keys = sorted (self.ring.keys())
    
    def get_nodes (self, key, count=3):
        hash_val = hash (key)
        # Find position in ring
        idx = bisect.bisect_right (self.sorted_keys, hash_val)
        # Return N successive nodes
        nodes = []
        for i in range (count):
            pos = (idx + i) % len (self.sorted_keys)
            node = self.ring[self.sorted_keys[pos]]
            if node not in nodes:
                nodes.append (node)
        return nodes
\`\`\`

---

## Operating Distributed Object Storage

### Monitoring

**Key metrics**:
- Storage capacity used/available
- Request rate (GET, PUT, DELETE)
- Error rate
- Latency (p50, p95, p99)
- Data durability (scrub errors)

**Health checks**:
- Node availability
- Disk health (SMART data)
- Network connectivity
- Replication lag

### Capacity Planning

**Growth estimation**:
\`\`\`
Current: 100 TB
Growth rate: 10 TB/month
Plan: Add capacity when 80% full
\`\`\`

**Proactive expansion**:
- Add nodes before 80% capacity
- Rebalance automatically
- Monitor growth trends

### Failure Scenarios

**1. Disk failure**:
- Detect via SMART or I/O errors
- Mark disk as out
- Trigger re-replication from other replicas

**2. Node failure**:
- Detect via heartbeat timeout
- Objects on node now under-replicated
- Re-replicate to other nodes

**3. Network partition**:
- Quorum-based writes prevent split-brain
- Read from available nodes
- Eventual consistency repair

### Backup and Disaster Recovery

**Backup strategies**:
1. **Replication**: Multi-datacenter replication
2. **Snapshots**: Point-in-time consistency
3. **Archive**: Copy to tape/cloud for long-term

**3-2-1 rule**:
- 3 copies
- 2 different media
- 1 offsite

---

## Interview Tips

**Explain distributed object storage in 2 minutes**:
"Distributed object storage like Ceph and MinIO provides self-hosted S3-compatible storage. Ceph uses CRUSH algorithm for deterministic object placement without central metadata. Objects are replicated or erasure-coded across nodes for durability. MinIO is simpler, Kubernetes-native, perfect for cloud-native apps. Key challenges: metadata management, failure handling, data placement. Benefits: data sovereignty, cost savings, customization. Trade-offs: operational complexity vs cloud simplicity."

**Key concepts to mention**:
- CRUSH algorithm (Ceph)
- Erasure coding vs replication
- Consistent hashing for placement
- S3 API compatibility
- Metadata management challenges

**Common interview questions**:
- How does CRUSH algorithm work?
- Replication vs erasure coding trade-offs?
- How to handle node failures?
- Why choose self-hosted vs cloud?

---

## Key Takeaways

🔑 Ceph = enterprise-grade unified storage (object, block, file)
🔑 CRUSH algorithm enables decentralized object placement
🔑 Erasure coding provides storage efficiency (1.5x vs 3x)
🔑 MinIO = simpler, Kubernetes-native, S3-compatible
🔑 Replication vs erasure coding: performance vs storage cost
🔑 Self-hosted provides data sovereignty and cost savings
🔑 Operational complexity is key trade-off vs cloud
🔑 Consistent hashing enables distributed placement
🔑 Monitor capacity, request rate, latency, durability
🔑 Use cases: on-premise data lakes, backup storage, AI/ML
`,
};
