/**
 * Quiz questions for Blob Storage Patterns section
 */

export const blobStorageQuiz = [
  {
    id: 'q1',
    question:
      'Explain the presigned URL pattern for user-generated content uploads. Why upload directly to S3 instead of through your backend server?',
    sampleAnswer:
      "Presigned URL pattern: (1) Client requests upload URL from backend. (2) Backend generates presigned URL (valid 15 min) with specific permissions. (3) Client uploads directly to S3 using presigned URL. (4) S3 triggers Lambda on upload for processing. (5) Lambda updates database with metadata. Benefits: (1) Offload bandwidth - backend doesn't handle file traffic, saves bandwidth and CPU. (2) Scalability - S3 handles any upload volume, backend doesn't bottleneck. (3) Cost - cheaper bandwidth from S3 than EC2. (4) Global performance - S3 edge locations optimize upload. (5) Security - presigned URL is time-limited and scoped to specific object/permissions. Example: Instagram - millions of photo uploads/day. Uploading through backend would require massive infrastructure. Direct S3 upload = backend only generates URLs (lightweight). S3 scales automatically. Lambda processes asynchronously. Backend stays responsive for other operations.",
    keyPoints: [
      'Client uploads directly to S3 (not through backend)',
      'Backend generates time-limited presigned URL',
      'Offloads bandwidth and processing from backend',
      'S3 scales automatically to any volume',
      'Async processing via S3 events + Lambda',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe content deduplication in blob storage. How and why would you implement it?',
    sampleAnswer:
      'Content deduplication: Store each unique file only once, even if uploaded by multiple users. Implementation: (1) Calculate hash (SHA-256) of file before upload. (2) Check database if hash exists. (3) If exists: Store reference to existing blob (user_id → blob_key) without uploading. (4) If new: Upload blob, store hash in database. (5) Reference counting for deletion (delete blob only when no references remain). Benefits: (1) Storage savings - Dropbox saves ~75% storage via dedup. (2) Faster uploads - if duplicate, skip upload entirely. (3) Bandwidth savings - no redundant transfer. Use cases: File sharing (Dropbox - many users have same files), email attachments (same PDF sent to many), document management. Trade-offs: (1) Privacy - users can detect if someone else has file (hash matches). (2) Complexity - reference counting, garbage collection. (3) Hash calculation overhead. Works best for large files where storage savings justify complexity.',
    keyPoints: [
      'Store unique files once, multiple references',
      'Use hash (SHA-256) to identify duplicates',
      'Major storage and bandwidth savings',
      'Reference counting for safe deletion',
      'Use case: File sharing, email attachments',
    ],
  },
  {
    id: 'q3',
    question:
      'Why use multipart upload for large files? Explain the benefits and when it is required.',
    sampleAnswer:
      "Multipart upload splits large files into parts (5 MB to 5 GB each, up to 10,000 parts), uploads parts in parallel, then combines. Benefits: (1) Parallel uploads - 10 parts × 10 concurrent = 10x faster. Example: 5 GB file in 10 parts = 5 minutes instead of 50 minutes. (2) Resilience - if one part fails, only retry that part (not entire file). Network hiccup doesn't lose entire upload. (3) Streaming - can upload while still creating file (don't need entire file in memory). (4) Pause/resume - store part ETags, resume from last uploaded part. Required for: Files > 5 GB (single PUT max). Recommended for: Files > 100 MB. Process: (1) Initiate multipart upload (get upload_id). (2) Upload parts (store ETags). (3) Complete upload (provide all ETags). (4) S3 combines parts. Best practices: 100 MB part size, cleanup incomplete uploads via lifecycle policy (avoid orphaned parts consuming storage).",
    keyPoints: [
      'Split file into parts, upload in parallel',
      'Required for files > 5 GB, recommended > 100 MB',
      'Benefits: Speed (parallel), resilience (retry parts), streaming',
      'Process: Initiate → Upload parts → Complete',
      'Cleanup incomplete uploads to save costs',
    ],
  },
];
