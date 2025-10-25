/**
 * Quiz questions for Dropbox Architecture section
 */

export const dropboxarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain Dropbox\'s block-level deduplication and chunking strategy. How does it reduce storage and upload bandwidth?",
    sampleAnswer:
      'Dropbox uses variable-size chunking with content-defined boundaries (Rabin fingerprinting). Upload process: (1) Client breaks file into chunks (avg 4MB, 1MB-16MB range). (2) Compute SHA-256 hash for each chunk. (3) Client asks server: "Do you have these hashes?" (4) Server responds which chunks already exist (global deduplication across all users). (5) Client uploads only missing chunks. (6) Server stores chunks in S3, metadata in database (file_id → [chunk_hashes]). Benefits: (1) Deduplication - if 1000 users upload same file, store once. (2) Incremental uploads - editing 1MB in 100MB file uploads only 1 new chunk. (3) Bandwidth savings - 95%+ for common files. Example: 1GB movie uploaded by 10,000 users = 1GB stored (not 10TB). Challenges: Privacy (know which chunks exist globally), sync complexity (reassemble chunks). Trade-offs: Chunk metadata overhead vs storage/bandwidth savings.',
    keyPoints: [
      'Variable-size chunking (avg 4MB) with content-defined boundaries',
      'SHA-256 hashing for global deduplication across all users',
      'Upload only missing chunks (95%+ savings for common files)',
      'Incremental updates: edit 1MB in 100MB file → upload 1 chunk',
    ],
  },
  {
    id: 'q2',
    question:
      'How did Dropbox migrate from AWS to its own datacenters (Magic Pocket), and what were the motivations?',
    sampleAnswer:
      'Dropbox stored exabytes on AWS S3 (cost: $75M+/year). Migration motivations: (1) Cost - building own datacenters amortizes over years, saves 30-40%. (2) Control - optimize for Dropbox workload. (3) Performance - co-locate compute and storage. Migration (2014-2016): (1) Build Magic Pocket datacenters (California, Texas). (2) Custom storage nodes (high-density, erasure coding). (3) Dual writes to AWS and Magic Pocket during migration. (4) Gradually shift reads to Magic Pocket. (5) Validate data consistency. (6) Decommission AWS storage. Challenges: Building datacenter expertise, hardware failures, networking complexity. Result: Cost savings, improved performance (lower latency within datacenter), operational control. Keep AWS for: disaster recovery, global edge presence.',
    keyPoints: [
      'Motivations: Cost savings (30-40%), control, performance',
      'Magic Pocket: Custom datacenters with erasure-coded storage',
      'Migration: Dual writes, gradual read shift, validation, cutover',
      'Trade-offs: Datacenter operational complexity vs cost/control benefits',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe Dropbox\'s conflict resolution strategy for file syncing. How does it handle concurrent edits to the same file from multiple devices?",
    sampleAnswer:
      'Dropbox uses last-write-wins with conflict markers. Scenario: User edits file on laptop and phone simultaneously while offline. (1) Laptop syncs first, uploads version A (rev 5). (2) Phone syncs, uploads version B based on same parent (rev 4). (3) Server detects conflict (two versions from same parent). (4) Keep laptop version as canonical (first to sync). (5) Rename phone version to "file (conflicted copy 2024-01-15).txt". (6) Sync conflict marker to both devices. User resolves manually. Why not operational transform (Google Docs)? Dropbox syncs any file type, not just structured documents. Can\'t merge arbitrary binaries. Alternative: Version history - Dropbox keeps 30 days of versions, user can restore previous versions. Best practice: Avoid conflicts via file locking APIs for critical files.',
    keyPoints: [
      'Last-write-wins: First upload becomes canonical version',
      'Conflict files: Rename conflicting versions with timestamp',
      "Can't auto-merge arbitrary file types (unlike Google Docs with OT)",
      'Version history: 30 days of revisions for recovery',
    ],
  },
];
