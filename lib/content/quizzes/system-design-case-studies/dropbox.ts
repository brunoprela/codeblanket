/**
 * Quiz questions for Design Dropbox section
 */

export const dropboxQuiz = [
  {
    id: 'q1',
    question:
      "Explain Dropbox\'s delta sync and chunking strategy. A user edits 1 MB of a 100 MB file. Walk through how Dropbox detects and uploads only the changed portion.",
    sampleAnswer:
      'DELTA SYNC WITH CHUNKING: (1) INITIAL STATE: File stored as 25 blocks (100 MB / 4 MB per block). Each block has SHA-256 hash stored in metadata. (2) USER EDITS FILE: User modifies 1 MB in middle of file (blocks 10-11 affected). (3) RE-CHUNKING: Dropbox client re-chunks entire file into 4 MB blocks. Calculates SHA-256 for each block. (4) CHANGE DETECTION: Compare new hashes with previous hashes: Blocks 1-9: Same hash (unchanged). Block 10: Different hash (modified). Block 11: Different hash (modified). Blocks 12-25: Same hash (unchanged). (5) DELTA UPLOAD: Upload only blocks 10-11 (8 MB) instead of full 100 MB. Server stores new blocks with new hashes. Update file metadata: file_id → [hash1, hash2, ..., hash10_new, hash11_new, ..., hash25]. (6) OTHER DEVICES: Push notification: "File changed". Download only blocks 10-11. Reconstruct file locally. BANDWIDTH SAVINGS: 8 MB uploaded vs 100 MB (92% savings). TRADE-OFF: Fixed 4 MB blocks vs variable-size blocks (rsync). Fixed simpler but less optimal if edit crosses block boundary. KEY INSIGHT: Content-based hashing enables delta sync without comparing files byte-by-byte.',
    keyPoints: [
      'Split file into fixed 4 MB blocks, calculate SHA-256 per block',
      'Compare new hashes with previous version to detect changes',
      'Upload only changed blocks (8 MB vs 100 MB = 92% savings)',
      'Server stores blocks, updates metadata with new hash list',
      'Other devices download only changed blocks',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain block-level deduplication. Two users upload the same 100 MB video. How much storage is actually used? Walk through the implementation.',
    sampleAnswer:
      "BLOCK-LEVEL DEDUPLICATION: (1) USER A UPLOADS: video.mp4 (100 MB). Client chunks into 25 blocks (4 MB each). Calculate SHA-256: block1 → abc123, block2 → def456, ..., block25 → xyz789. Upload blocks to S3: s3://blocks/abc123, s3://blocks/def456, .... Store in blocks table: INSERT INTO blocks (block_hash, storage_path, ref_count) VALUES ('abc123', 's3://blocks/abc123', 1). Create file entry: INSERT INTO files (file_id=1, user_id=A, file_path='/video.mp4'). Link blocks: INSERT INTO file_blocks (file_id=1, block_index=0, block_hash='abc123'). Storage used: 100 MB. (2) USER B UPLOADS SAME FILE: Client chunks video.mp4, calculates same hashes (identical file → identical hashes). Query server: \"Do these blocks exist?\" Server checks blocks table: All 25 blocks already exist! (3) INSTANT \"UPLOAD\": No actual upload needed (0 bytes transferred). Create new file entry: file_id=2, user_id=B. Link to existing blocks: INSERT INTO file_blocks (file_id=2, block_index=0, block_hash='abc123'). Increment ref_count: UPDATE blocks SET ref_count=2 WHERE block_hash IN ('abc123', ...). Storage used: Only metadata (~1 KB). (4) RESULT: 2 users, 1 copy stored = 50% savings. DELETE HANDLING: When user A deletes file: Decrement ref_count for all blocks. Only delete block from S3 if ref_count=0 (no other files use it). SAVINGS: 20-30% typical, 80%+ for organizations (shared company docs). KEY INSIGHT: Content-addressable storage (blocks identified by hash) enables automatic deduplication.",
    keyPoints: [
      'User A uploads: 25 blocks stored in S3, ref_count=1 each',
      'User B uploads same file: All blocks exist, no upload needed',
      'Create file entry linking to existing blocks, increment ref_count',
      'Storage: 100 MB (blocks) + 2 KB (metadata) vs 200 MB without dedup',
      'Delete: Decrement ref_count, only delete block if ref_count=0',
    ],
  },
  {
    id: 'q3',
    question:
      'Design conflict resolution for Dropbox. User edits file offline on laptop and desktop. Both come online. How do you detect and resolve the conflict?',
    sampleAnswer:
      'CONFLICT DETECTION & RESOLUTION: (1) INITIAL STATE: File document.txt at version 1 synced on laptop and desktop. Both devices go offline. (2) CONCURRENT EDITS: Laptop: Edit document.txt → version 2A (based on v1). Desktop: Edit document.txt → version 2B (also based on v1). (3) LAPTOP COMES ONLINE FIRST: Upload version 2A to server. Server accepts: Latest version now 2A. (4) DESKTOP COMES ONLINE: Desktop attempts to upload version 2B. Server detects conflict: Both 2A and 2B claim parent is v1 (fork detected). (5) CONFLICT RESOLUTION STRATEGIES: Option 1 - Last Write Wins: Keep version with latest timestamp. Problem: Clocks unreliable, data loss. Option 2 - Operational Transformation: Merge edits automatically (like Google Docs). Problem: Complex, only works for text. Option 3 - Keep Both (Dropbox approach): Accept desktop upload as version 2B. Rename on laptop: document.txt → document (laptop\'s conflicted copy).txt. Notify user: "Conflicted copy created, please merge manually". (6) IMPLEMENTATION: Server maintains version tree (DAG): v1 → v2A (from laptop), v1 → v2B (from desktop). Flag file as conflicted. Push notification to both devices. Laptop receives conflicted copy, desktop keeps original. (7) USER MANUAL MERGE: User opens both files, copies content, resolves conflicts. Deletes conflicted copy. New version v3 created. TRADE-OFF: No automatic merge (manual work) but no data loss (safe). Alternative (Git-like): Use CRDTs or three-way merge (complex). KEY INSIGHT: Conflict resolution is hard - Dropbox chooses simplicity (keep both) over automation (risk data loss).',
    keyPoints: [
      'Detect conflict: Two versions with same parent (fork in version tree)',
      'Keep both: Rename one as "conflicted copy", notify user',
      'User manually merges, no data loss',
      'Alternatives: Last-write-wins (data loss), OT (complex)',
      'Dropbox prioritizes safety over automation',
    ],
  },
];
