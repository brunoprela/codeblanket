/**
 * Quiz questions for HDFS section
 */

export const hdfsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the HDFS rack-aware replication strategy with RF=3. Why place 2 replicas in one rack and 1 in another instead of distributing evenly across 3 racks?',
    sampleAnswer:
      "HDFS's default replication policy for RF=3: 1st replica on client's node (or random if client outside cluster), 2nd replica on different rack, 3rd replica on same rack as 2nd. This balances fault tolerance and network efficiency. Benefits: (1) Rack fault tolerance - losing one rack doesn't lose all replicas. (2) Reduced network traffic - 2 replicas in same rack means only 1 cross-rack write instead of 2. Cross-rack bandwidth is expensive and limited. (3) Good read performance - client can read from nearby replica. Trade-off vs 3 racks: Losing 2 racks could lose all data, but this is extremely rare. The network cost savings (50% less cross-rack traffic) justify the slightly reduced fault tolerance. For higher durability requirements, can increase RF to 4+ with more cross-rack distribution.",
    keyPoints: [
      '1st replica: local/client node, 2nd: different rack, 3rd: same rack as 2nd',
      'Tolerates single rack failure',
      'Minimizes cross-rack writes (expensive)',
      'Trade-off: Network efficiency vs maximum fault tolerance',
      'Can customize for higher RF or different requirements',
    ],
  },
  {
    id: 'q2',
    question:
      'How does HDFS NameNode High Availability work? What happens during a NameNode failover?',
    sampleAnswer:
      'HDFS HA uses Active/Standby NameNode pair sharing state via Shared EditLog (typically QJM - Quorum Journal Manager). Active NameNode writes edits to QJM (Paxos-like consensus among JournalNodes). Standby continuously reads from QJM, keeping namespace up-to-date and performing checkpoints. ZooKeeper coordinates failover via ZKFC (ZooKeeper Failover Controller). On Active failure: (1) ZKFC detects via heartbeat timeout. (2) ZooKeeper triggers failover. (3) Standby promoted to Active (~30 sec). (4) Previous Active is fenced (prevented from serving requests to avoid split-brain). (5) New Active starts serving. Clients automatically retry and connect to new Active. DataNodes send block reports to new Active. Brief unavailability (30 sec) but no data loss. Standby must be caught up (reading EditLog) to minimize failover time.',
    keyPoints: [
      'Active/Standby NameNodes share state via QJM',
      'Standby reads EditLog continuously, stays current',
      'ZooKeeper coordinates automatic failover',
      'Fencing prevents split-brain',
      'Failover time ~30 seconds, no data loss',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is HDFS unsuitable for small files? What problems do millions of small files cause?',
    sampleAnswer:
      'HDFS is optimized for large files; small files create serious problems: (1) NameNode memory pressure - each file/block requires ~150 bytes of metadata in NameNode RAM. 1 million small files (1 KB each) = 1 million blocks = 150 MB metadata vs 1 GB large file = 8 blocks (128 MB blocks) = 1.2 KB metadata. Metadata grows faster than data! (2) Inefficient MapReduce - each block = 1 map task. 1 million blocks = 1 million tasks = massive overhead. (3) Wasted block space - 1 KB file uses entire 128 MB block (internal fragmentation). (4) Increased NameNode load - more metadata operations. Solutions: (1) Archive small files (HAR, SequenceFile, Parquet). (2) Combine small files before upload. (3) Use different storage for small files (HBase, S3). (4) Increase block size. HDFS designed for streaming reads of large files, not random access to many small files.',
    keyPoints: [
      'Each file/block = metadata in NameNode RAM',
      'Small files = disproportionate metadata overhead',
      'Inefficient MapReduce (too many tasks)',
      'Wasted block space (internal fragmentation)',
      'Solutions: Archive, combine, or use different storage',
    ],
  },
];
