/**
 * Quiz questions for Google File System section
 */

export const gfsQuiz = [
  {
    id: 'q1',
    question:
      'Explain why GFS chose a single master architecture despite it being a potential bottleneck. What design decisions mitigate this bottleneck?',
    sampleAnswer:
      "GFS chose a single master for simplicity - all metadata in memory, no distributed consensus needed, easy consistency. The key insight is that the master is OUT of the data path: clients only contact the master for metadata (which chunk servers have which chunks), then communicate directly with chunkservers for data. This separates control flow (master) from data flow (chunkservers). Mitigations: (1) Master stores minimal metadata per chunk (~64 bytes). (2) Large chunk size (64 MB) = fewer chunks = less metadata. (3) Clients cache metadata. (4) Master only involved in initial request, not data transfer. (5) Shadow masters can serve read-only requests. The master bottleneck is acceptable because metadata operations are rare compared to data operations. At Google's scale (petabytes), master's memory could handle billions of chunks.",
    keyPoints: [
      'Single master simplifies design (no distributed consensus)',
      'Master out of data path - clients contact chunkservers directly',
      'Large chunks (64 MB) reduce metadata',
      'Clients cache metadata',
      'Shadow masters for read-only operations',
    ],
  },
  {
    id: 'q2',
    question:
      'Why does GFS use 64 MB chunks instead of smaller chunks like 4 KB (typical file system block size)? Discuss the trade-offs.',
    sampleAnswer:
      "64 MB chunks provide several benefits for Google's workload: (1) Reduced metadata - fewer chunks means less memory at master. 1 TB file = 16,000 chunks (64 MB) vs 268 million blocks (4 KB). Master can handle more data with same memory. (2) Reduced client-master communication - one metadata request covers more data. (3) Long-lived TCP connections - client can perform many operations on same chunk. (4) Reduced network overhead - fewer chunk locations to track. Trade-offs: (a) Small files waste space - 1 KB file still uses 64 MB chunk (but Google has few small files). (b) Hot spots for popular small files - if many clients access same small file, that chunk becomes hot. Google mitigates by increasing replication factor for hot files. (c) Internal fragmentation. Overall, for Google's workload (large files, sequential access, batch processing), large chunks are net positive.",
    keyPoints: [
      'Reduced metadata at master (fewer chunks)',
      'Less client-master communication',
      'Enables long-lived TCP connections',
      'Trade-off: Wastes space for small files',
      'Trade-off: Hot spots for popular small files',
    ],
  },
  {
    id: 'q3',
    question:
      "How does GFS handle consistency in the face of concurrent writers? Why is weak consistency acceptable for Google's use case?",
    sampleAnswer:
      "GFS provides relaxed consistency: file namespace mutations are atomic, but data mutations have weaker guarantees. For concurrent record appends, GFS guarantees 'at-least-once' semantics - records appear at least once but may be duplicated or have padding/gaps. This is acceptable because: (1) Google's workload is append-only (logs, MapReduce output). (2) Applications are designed to handle inconsistency - records have checksums and unique IDs for deduplication. (3) Readers filter padding and duplicates. (4) The alternative (strong consistency via locking) would severely hurt availability and performance. Example: MapReduce writes intermediate data to GFS - if records are duplicated, the reduce phase handles deduplication. The trade-off is: high availability + high performance + simple design vs complexity in application layer. For Google's batch processing workload, this is acceptable. GFS prioritizes availability and throughput over strict consistency.",
    keyPoints: [
      'Weak consistency for data mutations (at-least-once semantics)',
      'Acceptable for append-only workloads (logs, MapReduce)',
      'Applications handle duplicates and padding',
      'Strong consistency would hurt availability and performance',
      'Trade-off: Simplicity and availability vs application complexity',
    ],
  },
];
