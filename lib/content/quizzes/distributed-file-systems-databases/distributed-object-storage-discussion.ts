/**
 * Quiz questions for Distributed Object Storage section
 */

export const distributedObjectStorageQuiz = [
  {
    id: 'q1',
    question:
      'Explain the CRUSH algorithm in Ceph. Why is it significant that object placement is calculated rather than looked up?',
    sampleAnswer:
      "CRUSH (Controlled Replication Under Scalable Hashing) is Ceph\'s deterministic placement algorithm that calculates where data should be stored without central lookup. Process: hash (object_id) + cluster_map → placement group → OSDs. Key insight: ANY node can calculate object location using same inputs (object ID + cluster map). No metadata server needed! Significance: (1) No metadata bottleneck - traditional systems have central metadata server (like GFS master) that becomes bottleneck. CRUSH eliminates this. (2) Scalability - adding nodes doesn't increase metadata lookups. (3) Client-side placement - clients directly contact correct OSD, no intermediary. (4) Automatic rebalancing - when cluster topology changes, CRUSH recalculates placement, data migrates automatically. (5) Customizable rules - can specify 'place 3 replicas in different racks'. Trade-off: Cluster map must be distributed to all nodes, but it's small (topology info, not per-object metadata). This is why Ceph scales to thousands of nodes without metadata bottleneck.",
    keyPoints: [
      'CRUSH calculates object placement (no central lookup)',
      'Any node can determine location: hash + cluster map',
      'No metadata server bottleneck',
      'Automatic rebalancing on topology changes',
      'Customizable placement rules (rack awareness)',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare MinIO vs Ceph. When would you choose each for your use case?',
    sampleAnswer:
      'MinIO: (1) Simple, Kubernetes-native, S3-compatible. (2) Object storage only (no block/file). (3) Easier to deploy and operate. (4) Smaller scale (PBs). (5) Modern architecture, written in Go. Choose MinIO if: Cloud-native apps, Kubernetes environment, need S3 API, smaller scale (<10 PB), want simplicity, limited ops team. Ceph: (1) Unified storage (object, block, file). (2) More complex but feature-rich. (3) Battle-tested, mature (2006). (4) Larger scale (EBs). (5) Requires significant operational expertise. Choose Ceph if: Need block storage (RBD) for VMs/containers, need file system (CephFS), very large scale (>10 PB), have experienced ops team, need proven enterprise solution. Example scenarios: Startup with Kubernetes → MinIO. Enterprise with OpenStack + large datasets → Ceph. Cloud-native AI/ML platform → MinIO. Traditional enterprise replacing SAN → Ceph. Both offer erasure coding, replication, and multi-site. MinIO wins on simplicity; Ceph wins on features and scale.',
    keyPoints: [
      'MinIO: Simple, S3-only, Kubernetes-native, easier ops',
      'Ceph: Complex, unified storage (object/block/file), larger scale',
      'MinIO for: Cloud-native, K8s, simplicity, <10 PB',
      'Ceph for: Block storage, file system, >10 PB, enterprise',
      'Both scalable and durable, different trade-offs',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain erasure coding vs replication in distributed object storage. When would you choose each?',
    sampleAnswer:
      'Replication: Store N copies of data. Example: 3x replication = 1 GB becomes 3 GB (3x overhead). Can lose N-1 copies. Simple, fast reads/writes (read from any replica). Erasure Coding: Split data into K data chunks + M parity chunks. Example: 8+4 = 12 chunks total. Can lose any M chunks and reconstruct. 1 GB becomes 1.5 GB (1.5x overhead for 8+4). Choose replication when: (1) Hot data (frequently accessed) - faster reads. (2) Small objects - EC overhead not worth it. (3) Low latency critical - read from single replica. (4) Simple operations. Choose erasure coding when: (1) Cold data (rarely accessed) - storage cost matters more. (2) Large objects - overhead justified. (3) Cost-sensitive - 1.5x vs 3x is huge at PB scale (50% savings). (4) High durability needed - 8+4 tolerates 4 failures vs 3x tolerates 2. Trade-offs: EC uses more CPU (encoding/decoding), slower reads (must reconstruct), more network traffic (read multiple chunks). Hybrid approach: Replicate hot data, erasure code cold data. Example: Ceph can use replication for metadata pools, EC for data pools.',
    keyPoints: [
      'Replication: 3x overhead, simple, fast reads',
      'Erasure coding: 1.5x overhead, complex, slower reads',
      'EC saves 50% storage at PB scale',
      'Replication for hot data, EC for cold data',
      'EC trade-off: Storage savings vs CPU and latency',
    ],
  },
];
