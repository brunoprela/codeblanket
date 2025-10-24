/**
 * Multiple choice questions for Design Pastebin section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pastebinMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'You store paste content in a MySQL MEDIUMTEXT column. What is the maximum size you can store, and what should you do for larger pastes?',
    options: [
      'TEXT: 64 KB max; use LONGTEXT for larger',
      'MEDIUMTEXT: 16 MB max; use object storage (S3) for larger',
      'MEDIUMTEXT: 4 GB max; no need for external storage',
      'MEDIUMTEXT: 1 MB max; use NoSQL for larger',
    ],
    correctAnswer: 1,
    explanation:
      'MySQL TEXT types: TEXT=64KB, MEDIUMTEXT=16MB, LONGTEXT=4GB. MEDIUMTEXT handles 99% of pastes (code snippets, logs, configs). For >16MB pastes, storing in database is inefficient - use S3/blob storage instead. Store metadata + S3 key in database, serve large files via CDN. This optimizes database for transactional queries while offloading large binary data to purpose-built object storage.',
  },
  {
    id: 'mc2',
    question:
      'Your Pastebin creates 1 million pastes/day with 30% expiring in 1 day, 20% in 1 week, and 50% never expiring. You run a daily cleanup job that deletes expired pastes. How many deletions per day should the job handle in steady state?',
    options: [
      '300,000 deletions/day (30% of 1M)',
      '500,000 deletions/day (30% + 20% of 1M)',
      'Variable: 300K on day 1, then 300K + (20%×1M/7) ≈ 329K on day 8+',
      '1 million deletions/day (all pastes)',
    ],
    correctAnswer: 2,
    explanation:
      'Day 1: 300K expire (30% of 1M). Day 2-7: Only 300K daily (1-day expiry). Day 8+: 300K (1-day) + ~29K (1-week from 7 days ago, ~200K/7) = ~329K per day in steady state. The 1-week pastes add 200K/7 ≈ 28.6K per day. This illustrates the importance of calculating STEADY-STATE load, not just initial conditions. Cleanup job must handle 329K deletions/day, typically batched as: DELETE ... LIMIT 10000; executed 33 times with throttling.',
  },
  {
    id: 'mc3',
    question:
      'You implement gzip compression for paste content. A 10 KB JavaScript code snippet compresses to 2 KB. What is the trade-off you are making?',
    options: [
      'Slower writes (compression CPU) and slower reads (decompression CPU) for 80% storage savings',
      'Faster writes and slower reads for storage savings',
      'No trade-off - compression is always free',
      'Slower writes but faster reads (compressed data transfers faster)',
    ],
    correctAnswer: 0,
    explanation:
      'Compression ALWAYS trades CPU for storage/bandwidth. WRITE: Must compress before storing (1-5ms CPU). READ: Must decompress before returning (1-5ms CPU). BENEFITS: 80% storage savings (10 KB → 2 KB), reduced network bandwidth (5x less data transfer), lower storage costs. COST: CPU overhead on every read/write. For Pastebin with low write rate (12/sec) and low read rate (120/sec), the CPU cost is trivial compared to storage savings. The math: $0.01 per GB-month storage × 80% savings vs $0.10 per CPU hour. Storage wins. This is why compression is standard for text-heavy systems.',
  },
  {
    id: 'mc4',
    question:
      'You implement content deduplication by storing SHA-256 hashes. Two users paste identical 10 KB error logs. How much storage do you save?',
    options: [
      '10 KB (one paste stored, one pointer)',
      '~9 KB (store once, two small metadata rows)',
      '~10 KB (store content once, but keep separate paste_ids with metadata)',
      '0 KB (each paste needs unique storage)',
    ],
    correctAnswer: 2,
    explanation:
      'DEDUPLICATION APPROACH: Both pastes get unique paste_ids (different URLs). First paste: Insert full content (10 KB) + metadata (~1 KB) + content_hash. Second paste: Check content_hash exists. Found → Insert only metadata (~1 KB) with reference to same content. Storage: 10 KB (content) + 2 KB (two metadata rows) = 12 KB vs 22 KB without dedup (10 KB + 10 KB + 2 KB metadata) = ~10 KB saved. Implementation: Use content_hash as foreign key or shared content table. This is copy-on-write deduplication. Trade-off: Schema complexity vs storage savings (worthwhile for 20-30% duplicate rate).',
  },
  {
    id: 'mc5',
    question:
      'A power user pastes a 40 MB log file. Which statement is FALSE about handling this in production?',
    options: [
      'Store the file in S3 and keep only metadata + S3 key in database',
      'Serve the file via CloudFront CDN for faster global access',
      'Store the entire 40 MB in database LONGTEXT column for simplicity',
      'Apply gzip compression before uploading to S3 (may reduce to 8 MB)',
    ],
    correctAnswer: 2,
    explanation:
      'Storing 40 MB in database is BAD PRACTICE. Database issues: (1) MySQL max_allowed_packet default is 16 MB (would fail). (2) Large rows kill buffer pool efficiency. (3) Replication lag (transferring 40 MB to replicas). (4) Backup slowness. (5) Memory pressure. CORRECT APPROACH: Upload to S3 (designed for large files), enable CloudFront CDN for global delivery, compress to reduce size (logs compress well, 5:1 ratio typical → 8 MB). Database stores: paste_id, s3_key, compressed_size, metadata (<1 KB). This is separation of concerns: database for structured data, object storage for large files. The threshold is typically 1-10 MB.',
  },
];
