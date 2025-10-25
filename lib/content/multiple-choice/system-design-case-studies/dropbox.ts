/**
 * Multiple choice questions for Design Dropbox section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const dropboxMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Dropbox splits files into 4 MB blocks. A user has a 100 MB file. How many blocks are created?',
    options: ['10 blocks', '25 blocks', '100 blocks', '1 block'],
    correctAnswer: 1,
    explanation:
      '100 MB / 4 MB per block = 25 blocks. Each block is hashed independently (SHA-256). This enables delta sync: If user edits 1 block, only that 4 MB is uploaded (not full 100 MB). BLOCK SIZE CHOICE: Too small (1 MB) → More blocks, more overhead. Too large (64 MB) → Less granular, larger uploads on small edits. 4 MB is sweet spot: Reasonable metadata overhead, good delta sync granularity. For comparison: rsync uses variable block size (adaptive), Dropbox uses fixed (simpler).',
  },
  {
    id: 'mc2',
    question:
      'Why does Dropbox use content-based hashing (SHA-256 of blocks) instead of timestamp-based change detection?',
    options: [
      'SHA-256 is faster than timestamps',
      'Timestamps unreliable (clock skew); hashing detects actual content changes, enables deduplication',
      'Timestamps use more storage',
      'Hashing is required by law',
    ],
    correctAnswer: 1,
    explanation:
      "TIMESTAMP PROBLEMS: (1) Clock skew: Laptop clock wrong → Timestamp unreliable. (2) False positives: Touch file (update timestamp) without changing content → Unnecessary upload. (3) No deduplication: Cannot detect identical blocks across users. CONTENT HASHING BENEFITS: (1) Accurate: Only actual content changes trigger upload. (2) Deduplication: Same content → Same hash → Reuse existing blocks. (3) Idempotent: Uploading same file twice recognized as duplicate. TRADE-OFF: Must hash entire file (CPU cost) vs checking timestamps (cheap). For file sync where bandwidth/storage more expensive than CPU, hashing wins. This is foundational to Dropbox's efficiency (20-30% storage savings).",
  },
  {
    id: 'mc3',
    question:
      'A user edits a file while offline. When they reconnect, how does Dropbox know what changed?',
    options: [
      'User must manually mark changes',
      'Dropbox client tracks file modifications locally (watcher + metadata), syncs changes when online',
      'Server compares files byte-by-byte',
      'Changes are lost (offline edits not supported)',
    ],
    correctAnswer: 1,
    explanation:
      'OFFLINE EDIT TRACKING: (1) WATCHER: File system watcher (inotify/FSEvents) monitors Dropbox folder. Detects when files modified, even offline. (2) LOCAL METADATA: Client maintains local database (SQLite): file_path, last_modified_time, block_hashes. When file changes, mark as "pending upload". (3) RECONNECT: Client comes online, queries local DB for pending files. Re-chunk pending files, calculate new block hashes. Compare with previous hashes (stored locally). (4) DELTA UPLOAD: Upload only changed blocks. This is how Dropbox supports offline mode seamlessly. Without local tracking, would need to compare entire file tree (slow).',
  },
  {
    id: 'mc4',
    question:
      'Two users both edit the same file simultaneously while online. How does Dropbox handle this race condition?',
    options: [
      'First edit wins, second edit rejected',
      'Last edit wins, first edit overwritten',
      'Server detects concurrent edits, creates conflicted copy',
      'Edits automatically merged',
    ],
    correctAnswer: 2,
    explanation:
      "CONCURRENT EDIT HANDLING: User A saves at time T, starts upload. User B saves at time T+1s, starts upload. Server receives both: (1) A's upload arrives first → Version 2A created. (2) B's upload arrives 2 seconds later, claims parent is version 1 (not 2A). (3) Server detects conflict (B's parent is stale). (4) Accept B's upload as version 2B, create conflicted copy on A's device. RESULT: No edits lost, user manually resolves. REAL-TIME APPS (Google Docs): Use Operational Transformation to auto-merge. Complex, requires collaboration server. Dropbox optimizes for simplicity over real-time collaboration.",
  },
  {
    id: 'mc5',
    question:
      'Dropbox stores file versions for 30 days. A user makes 10 versions of a 100 MB file. How much storage is typically used?',
    options: [
      '1 GB (10 × 100 MB)',
      '~120 MB (original + deltas)',
      '100 MB (only latest version)',
      '10 GB (versions + metadata)',
    ],
    correctAnswer: 1,
    explanation:
      'WITH DELTA SYNC: Version 1: 100 MB (25 blocks). Version 2: User edits 4 MB → Only 1 block changes → +4 MB storage. Version 3-10: Similar small edits → +4 MB each. Total: 100 MB (v1) + 9 × 4 MB (deltas) = 136 MB (~1.4x original). WITHOUT DELTA SYNC: Would be 1 GB (10 full copies). BLOCK REUSE: Old versions reference same blocks as new versions (only changed blocks stored). After 30 days: Metadata deleted, but blocks retained if referenced by current version. This is why delta sync + block storage is efficient for versioning.',
  },
];
