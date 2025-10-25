/**
 * Quiz questions for S3 Architecture section
 */

export const s3ArchitectureQuiz = [
  {
    id: 'q1',
    question:
      'Explain the significance of S3 achieving strong read-after-write consistency in December 2020. What changed and why does it matter?',
    sampleAnswer:
      "Before Dec 2020, S3 had eventual consistency: after writing an object, reads might return stale data or 404 for ~1 second. This required application-level retry logic and careful handling. Post-Dec 2020, S3 provides strong read-after-write consistency: PUT succeeds → GET immediately returns latest version. Overwrite → read sees new version. Delete → immediately 404. List operations immediately reflect changes. This simplifies application logic dramatically - no retry loops, no eventual consistency handling. How achieved: Distributed consensus protocol ensures write not acknowledged until replicated to quorum. Metadata updates are atomic. Impact: (1) Simplified app development. (2) Database-like semantics. (3) Safe to read immediately after write. (4) No more race conditions from stale reads. (5) Slight write latency increase (negligible). This was a major engineering achievement - strong consistency at S3's scale (trillions of objects) is extremely difficult.",
    keyPoints: [
      'Pre-2020: Eventual consistency (reads could be stale)',
      'Post-2020: Strong read-after-write consistency',
      'Writes not acknowledged until replicated to quorum',
      'Simplifies application logic (no retry loops)',
      'Achieved via distributed consensus at massive scale',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare S3 Standard vs S3 Glacier Deep Archive. When would you choose each?',
    sampleAnswer:
      'S3 Standard: $0.023/GB-month, millisecond access, 99.99% availability, frequent access. Use for: active data, website assets, mobile app content, data accessed daily/weekly. S3 Glacier Deep Archive: $0.00099/GB-month (23x cheaper!), 12-48 hour retrieval, long-term archive. Use for: regulatory archives (7-10 years), compliance data, rarely accessed backups. Key differences: (1) Cost: Deep Archive is cheapest storage class. (2) Retrieval time: Standard = immediate, Deep Archive = half-day to 2 days. (3) Retrieval cost: Standard = free, Deep Archive = $0.02/GB + request fees. Decision factors: (1) Access frequency - how often do you need data? (2) Retrieval urgency - can you wait 12-48 hours? (3) Total cost = storage + retrieval. Example: 100 TB accessed monthly = Standard cheaper. 100 TB accessed yearly = Deep Archive 95% cheaper despite retrieval costs. Use lifecycle policies to automatically transition data through tiers.',
    keyPoints: [
      'Standard: Fast access, higher cost ($0.023/GB-mo)',
      'Deep Archive: Slowest access (12-48h), lowest cost ($0.001/GB-mo)',
      'Choose based on access frequency and urgency',
      'Deep Archive for compliance, long-term archives',
      'Use lifecycle policies for automatic transitions',
    ],
  },
  {
    id: 'q3',
    question:
      'Why should you avoid designing S3 keys sequentially (like timestamps)? What is the solution?',
    sampleAnswer:
      'Sequential keys create hot partitions. S3 internally partitions data based on key prefixes. Sequential keys (e.g., 2024-01-15-00:00:01, 2024-01-15-00:00:02) all start with same prefix → all writes go to same partition → hot partition → throttling! Old S3 (pre-2018) was very sensitive to this. Modern S3 (post-2018) automatically handles hotspots better but good key design still helps. Solutions: (1) Add random prefix: hash (key) + key (e.g., a3f9/2024-01-15-00:00:01). (2) Reverse key: 10:00:51-15-10-4202 spreads writes. (3) Use UUID as prefix. Performance impact: Sequential keys can hit 3,500 PUT/sec limit per prefix. With random prefixes, you can achieve 100,000+ PUT/sec by distributing across prefixes. Example: Log files - instead of logs/2024/01/15/00:00:01.log, use logs/a3f9/2024/01/15/00:00:01.log where a3f9 = hash (timestamp). This distributes writes across many partitions.',
    keyPoints: [
      'Sequential keys → same partition → hot partition → throttling',
      'S3 limits: 3,500 PUT/sec per prefix',
      'Solution: Add random prefix (hash, UUID) to distribute writes',
      'Modern S3 better at handling, but good design still helps',
      'Random prefixes enable 100,000+ PUT/sec',
    ],
  },
];
